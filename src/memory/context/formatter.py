"""Context formatting: entity enrichment, vigilance extraction, section helpers."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from src.core.config import Config
from src.core.models import GraphData, GraphEntity
from src.core.utils import estimate_tokens as _estimate_tokens, filter_live_facts
from src.memory.store import read_entity, parse_observation
from src.memory.context.utilities import (
    _deduplicate_facts_for_context,
    _sort_facts_by_date,
    _group_facts_by_category,
)

logger = logging.getLogger(__name__)


# ── Fact TTL filtering ─────────────────────────────────────


def _is_fact_expired(obs_dict: dict, config: Config, today: date) -> bool:
    """Check if a fact has exceeded its TTL for context display."""
    category = obs_dict.get("category", "")
    ttl_days = config.get_fact_ttl(category)
    if ttl_days == 0:
        return False  # Never expires
    fact_date = obs_dict.get("date", "")
    if not fact_date:
        return False  # No date = can't determine expiry, keep it
    try:
        if len(fact_date) == 7:  # YYYY-MM format
            fact_date += "-15"  # Approximate to mid-month
        d = date.fromisoformat(fact_date)
        return (today - d).days > ttl_days
    except (ValueError, TypeError):
        return False


def _filter_expired_facts(facts: list[str], config: Config | None, today: date) -> list[str]:
    """Remove facts that have exceeded their TTL for context display."""
    if config is None:
        return facts
    result = []
    for fact in facts:
        parsed = parse_observation(fact)
        if parsed and _is_fact_expired(parsed, config, today):
            continue
        result.append(fact)
    return result


# ── Natural context helpers ──────────────────────────────────

RELATION_NATURAL = {
    "parent_of": "parent of",
    "lives_with": "lives with",
    "works_at": "works at",
    "friend_of": "friend of",
    "affects": "linked to",
    "improves": "improved by",
    "worsens": "worsened by",
    "uses": "uses",
    "part_of": "part of",
    "linked_to": "linked to",
    "requires": "requires",
    "contrasts_with": "contrasts with",
    "precedes": "precedes",
}


def _read_entity_facts(eid: str, entity: GraphEntity, memory_path: Path) -> list[str]:
    """Read facts from an entity's MD file."""
    entity_path = (memory_path / entity.file).resolve()
    if not entity_path.exists() or not entity_path.is_relative_to(memory_path.resolve()):
        return []
    try:
        _, sections = read_entity(entity_path)
        return sections.get("Facts", [])
    except Exception:
        return []


def _select_entities_for_natural(
    all_top: list[tuple[str, GraphEntity]], graph: GraphData,
) -> list[tuple[str, GraphEntity]]:
    """Filter entities for natural context: personal memory over transient topics."""
    selected = []
    for eid, entity in all_top:
        # Always include: identity, health, people, animals, ai_self
        if entity.type in ("person", "health", "animal", "ai_self"):
            selected.append((eid, entity))
            continue
        # Work/org/project: only if mentioned 2+ times or high score
        if entity.type in ("work", "organization", "project"):
            if entity.frequency >= 2 or entity.score >= 0.5:
                selected.append((eid, entity))
            continue
        # Interests/places: only if long_term or high frequency
        if entity.type in ("interest", "place"):
            if entity.retention == "long_term" or entity.frequency >= 3:
                selected.append((eid, entity))
            continue
        # Rest: only if permanent or very high score
        if entity.retention == "permanent" or entity.score >= 0.6:
            selected.append((eid, entity))
    return selected


def _classify_temporal(entity: GraphEntity, today: date) -> str:
    """Classify entity into 'long_term', 'medium_term', or 'short_term'."""
    last = date.fromisoformat(entity.last_mentioned) if entity.last_mentioned else None
    if not last:
        return "long_term"
    days_since = (today - last).days

    # Heuristic: identity entity types are always long_term
    if entity.type in ("person", "animal") and entity.frequency >= 5:
        return "long_term"
    if entity.type in ("health", "ai_self"):
        return "long_term"
    if entity.file.startswith("self/"):
        return "long_term"

    # Heuristic: old + frequently mentioned entity
    if entity.created:
        try:
            created = date.fromisoformat(entity.created)
            if (today - created).days > 30 and entity.frequency >= 3:
                return "long_term"
        except (ValueError, TypeError):
            pass

    # Override by retention (works once retention is upgraded)
    if entity.retention in ("long_term", "permanent"):
        return "long_term"

    # Fallback by recency
    if days_since <= 7:
        return "short_term"
    elif days_since <= 30:
        return "medium_term"
    else:
        return "long_term"


def _build_natural_bullet(
    eid: str, entity: GraphEntity, graph: GraphData, memory_path: Path,
) -> str:
    """Build a natural language bullet point for an entity."""
    # 1. Base: summary or most recent/important fact
    if entity.summary:
        base = entity.summary
    else:
        facts = _read_entity_facts(eid, entity, memory_path)
        facts = filter_live_facts(facts)
        if facts:
            # Take the last fact (most recently added), clean markers
            obs = parse_observation(facts[-1])
            base = obs["content"] if obs else facts[-1].lstrip("- ")
        else:
            base = entity.title

    # 2. Integrate key relations (max 2)
    rel_parts = []
    for rel in graph.relations:
        if rel.strength < 0.3:
            continue
        if rel.from_entity == eid:
            target = graph.entities.get(rel.to_entity)
            if target and rel.type in RELATION_NATURAL:
                rel_parts.append(f"{RELATION_NATURAL[rel.type]} {target.title}")
        elif rel.to_entity == eid:
            source = graph.entities.get(rel.from_entity)
            if source and rel.type in RELATION_NATURAL:
                rel_parts.append(f"{RELATION_NATURAL[rel.type]} {source.title}")

    # 3. Assemble
    bullet = f"- {base}"
    if rel_parts:
        bullet += f" ({', '.join(rel_parts[:2])})"
    return bullet


def _extract_vigilances(
    all_selected: list[tuple[str, GraphEntity]],
    graph: GraphData,
    memory_path: Path,
    config: Config | None = None,
) -> list[str]:
    """Extract vigilance/diagnosis/treatment items as natural bullets."""
    vigilances = []
    for eid, entity in all_selected:
        facts = _read_entity_facts(eid, entity, memory_path)
        for f in facts:
            obs = parse_observation(f)
            if obs and obs["category"] in ("vigilance", "diagnosis", "treatment"):
                if obs.get("valence") == "negative" or obs["category"] == "vigilance":
                    vigilances.append(f"- {entity.title}: {obs['content']}")
    # Dedup by content
    seen: set[str] = set()
    unique = []
    for v in vigilances:
        key = v.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(v)
    max_items = config.ctx.max_vigilance_items if config else 15
    return unique[:max_items]


def _enrich_entity_natural(
    entity_id: str, entity: GraphEntity, graph: GraphData, memory_path: Path,
    config: Config | None = None,
) -> str:
    """Build a compact enriched dossier for LLM natural context generation.

    Unlike _enrich_entity(), this omits scores/retention headers and caps
    relations at 3 (strongest only) for cleaner LLM input.
    """
    entity_path = (memory_path / entity.file).resolve()
    facts: list[str] = []
    if entity_path.is_relative_to(memory_path.resolve()) and entity_path.exists():
        try:
            _, sections = read_entity(entity_path)
            facts = sections.get("Facts", [])
        except Exception:
            pass

    lines = [f"### {entity.title} [{entity.type}]"]
    if entity.summary:
        lines.append(f"Summary: {entity.summary}")
    if facts:
        facts = filter_live_facts(facts)
        facts = _filter_expired_facts(facts, config, date.today())
        facts = _deduplicate_facts_for_context(facts, max_per_category=3)
        for f in facts:
            lines.append(f"  {f}")

    # Max 3 strongest relations
    entity_rels: list[tuple[float, str]] = []
    for rel in graph.relations:
        if rel.strength < 0.3:
            continue
        if rel.from_entity == entity_id:
            target = graph.entities.get(rel.to_entity)
            if target:
                entity_rels.append((rel.strength, f"→ {rel.type} {target.title}"))
        elif rel.to_entity == entity_id:
            source = graph.entities.get(rel.from_entity)
            if source:
                entity_rels.append((rel.strength, f"← {rel.type} {source.title}"))
    entity_rels.sort(key=lambda x: x[0], reverse=True)
    for _, line in entity_rels[:3]:
        lines.append(f"  {line}")

    return "\n".join(lines)


def _build_section_llm(
    section_name: str,
    entities: list[tuple[str, GraphEntity]],
    graph: GraphData,
    memory_path: Path,
    config: Config,
    section_budget: int,
) -> str:
    """Build a section using LLM to generate natural narrative."""
    from src.memory.context.builder import _rag_prefetch

    if not entities:
        return ""

    # Build enriched dossier
    dossier_parts = []
    for eid, entity in entities:
        dossier_parts.append(_enrich_entity_natural(eid, entity, graph, memory_path, config))
    raw_dossier = "\n\n".join(dossier_parts)

    # RAG pre-fetch
    rag_text = _rag_prefetch([eid for eid, _ in entities], graph, config, memory_path)

    # LLM call with fallback
    try:
        from src.core.llm import call_natural_context_section
        result = call_natural_context_section(
            section_name=section_name,
            entities_dossier=raw_dossier,
            rag_context=rag_text,
            budget_tokens=section_budget,
            config=config,
        )
        if result.strip():
            return result
    except Exception as e:
        logger.warning("LLM natural section '%s' failed: %s", section_name, e)

    # Deterministic fallback
    lines = []
    used = 0
    for eid, entity in entities:
        bullet = _build_natural_bullet(eid, entity, graph, memory_path)
        cost = _estimate_tokens(bullet, language=config.user_language)
        if used + cost > section_budget and lines:
            break
        lines.append(bullet)
        used += cost
    return "\n".join(lines)


def _enrich_entity(
    entity_id: str, entity: GraphEntity, graph: GraphData, memory_path: Path,
    config: Config | None = None,
) -> str:
    """Build an enriched dossier string for a single entity.

    Reads facts from the entity's MD file and collects graph relations.
    """
    # Read entity facts (with path traversal guard)
    entity_path = (memory_path / entity.file).resolve()
    facts: list[str] = []
    if entity_path.is_relative_to(memory_path.resolve()) and entity_path.exists():
        try:
            _, sections = read_entity(entity_path)
            facts = sections.get("Facts", [])
        except Exception:
            pass

    # Build section
    section_lines = [
        f"### {entity.title} [{entity.type}] (score: {entity.score:.2f}, retention: {entity.retention})",
    ]
    if entity.tags:
        section_lines.append(f"Tags: {', '.join(entity.tags)}")
    if facts:
        section_lines.append("Facts:")
        # Filter superseded facts, sort by date, deduplicate for context
        is_ai_self = entity.type == "ai_self"
        max_cat = (config.ctx.max_facts_per_category_ai_self if is_ai_self else config.ctx.max_facts_per_category) if config else (3 if is_ai_self else 5)
        facts = filter_live_facts(facts)
        facts = _filter_expired_facts(facts, config, date.today())
        sorted_facts = _sort_facts_by_date(facts)
        threshold = config.ctx.fact_dedup_threshold if config else 0.35
        sorted_facts = _deduplicate_facts_for_context(sorted_facts, threshold=threshold, max_per_category=max_cat)
        # Group by category for cleaner output
        grouped = _group_facts_by_category(sorted_facts)
        for cat, cat_facts in grouped.items():
            section_lines.append(f"  [{cat}]")
            for content in cat_facts:
                section_lines.append(f"    - {content}")

    # Get relations for this entity from graph (filter weak/stale)
    today = date.today()
    min_rel_strength = config.ctx.min_rel_strength if config else 0.3
    max_rel_age_days = config.ctx.max_rel_age_days if config else 365
    entity_relations = []
    related_info = []
    for rel in graph.relations:
        # Skip weak relations
        if rel.strength < min_rel_strength:
            continue
        # Skip stale relations (not reinforced in over a year)
        if rel.last_reinforced:
            try:
                last = date.fromisoformat(str(rel.last_reinforced))
                if (today - last).days > max_rel_age_days:
                    continue
            except (ValueError, TypeError):
                pass
        if rel.from_entity == entity_id:
            target = graph.entities.get(rel.to_entity)
            if target:
                entity_relations.append(f"  → {rel.type} {target.title}")
                related_info.append(f"{target.title} ({target.type})")
        elif rel.to_entity == entity_id:
            source = graph.entities.get(rel.from_entity)
            if source:
                entity_relations.append(f"  ← {rel.type} {source.title}")
                related_info.append(f"{source.title} ({source.type})")
    if related_info:
        section_lines.append(f"Related: {', '.join(related_info)}")
    if entity_relations:
        section_lines.append("Relations:")
        section_lines.extend(entity_relations)

    return "\n".join(section_lines)


def _collect_section(
    entities: list[tuple[str, GraphEntity]],
    graph: GraphData,
    memory_path: Path,
    token_budget: int,
    config: Config | None = None,
) -> str:
    """Collect enriched dossiers for entities, respecting a token budget."""
    parts: list[str] = []
    used = 0
    for eid, entity in entities:
        dossier = _enrich_entity(eid, entity, graph, memory_path, config)
        lang = config.user_language if config else "en"
        cost = _estimate_tokens(dossier, language=lang)
        if used + cost > token_budget and parts:
            break
        parts.append(dossier)
        used += cost
    return "\n\n".join(parts)
