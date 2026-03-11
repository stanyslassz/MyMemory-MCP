"""Context generation: builds enriched dossier and generates _context.md and _index.md."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from src.core.config import Config
from src.core.llm import call_context_generation
from src.core.models import GraphData, GraphEntity
from src.core.utils import estimate_tokens as _estimate_tokens_util
from src.memory.scoring import get_top_entities
from src.memory.store import read_entity, _parse_observation


def _sort_facts_by_date(facts: list[str]) -> list[str]:
    """Sort facts by date (chronological). Undated facts go last, preserving order."""
    dated: list[tuple[str, str]] = []
    undated: list[str] = []
    for fact in facts:
        parsed = _parse_observation(fact)
        if parsed and parsed["date"]:
            dated.append((parsed["date"], fact))
        else:
            undated.append(fact)
    dated.sort(key=lambda x: x[0])
    return [f for _, f in dated] + undated


_STOPWORDS = frozenset(
    # French
    "le la les un une des de du d l et en pour aux avec sans par sur dans "
    "qui que qu est sont a au ce cette ces sa son ses se ne pas ni si "
    "on nous vous il elle ils elles "
    # English
    "the a an is are was were be been being have has had do does did "
    "will would shall should can could may might must "
    "i me my we our you your he him his she her it its they them their "
    "and or but not no nor so if then than that this these those "
    "in on at to for of by from with as into about up out".split()
)


def _trigrams(text: str) -> set[str]:
    """Return the set of character trigrams from text."""
    return {text[i:i + 3] for i in range(len(text) - 2)}


def _content_similarity(text_a: str, text_b: str) -> float:
    """Blended similarity: 50% stopword-filtered word Jaccard + 50% trigram Jaccard."""
    words_a = {w for w in text_a.lower().split() if w not in _STOPWORDS and len(w) > 1}
    words_b = {w for w in text_b.lower().split() if w not in _STOPWORDS and len(w) > 1}
    if not words_a or not words_b:
        return 0.0
    word_jaccard = len(words_a & words_b) / len(words_a | words_b)
    tri_a = _trigrams(" ".join(sorted(words_a)))
    tri_b = _trigrams(" ".join(sorted(words_b)))
    tri_union = tri_a | tri_b
    tri_jaccard = len(tri_a & tri_b) / len(tri_union) if tri_union else 0.0
    return 0.5 * word_jaccard + 0.5 * tri_jaccard


def _deduplicate_facts_for_context(
    facts: list[str], threshold: float = 0.35, max_per_category: int = 5,
) -> list[str]:
    """Drop near-duplicate facts within same category using blended similarity.

    Uses stopword-filtered word Jaccard + character trigram overlap.
    After dedup, caps each category to max_per_category (first occurrence wins).
    Preserves input order — only removes later duplicates.
    """
    kept_by_cat: dict[str, list[dict]] = {}
    cat_counts: dict[str, int] = {}
    result = []
    for line in facts:
        obs = _parse_observation(line)
        if not obs:
            result.append(line)
            continue
        cat = obs["category"]
        content = obs["content"]
        # Check duplicate against kept facts in same category
        is_dup = False
        for kept_obs in kept_by_cat.get(cat, []):
            if _content_similarity(content, kept_obs["content"]) > threshold:
                is_dup = True
                break
        if is_dup:
            continue
        # Check category cap
        count = cat_counts.get(cat, 0)
        if count >= max_per_category:
            continue
        result.append(line)
        kept_by_cat.setdefault(cat, []).append(obs)
        cat_counts[cat] = count + 1
    return result


def _group_facts_by_category(facts: list[str]) -> dict[str, list[str]]:
    """Group fact lines by their [category] prefix, stripping the prefix from content.

    Returns an ordered dict of category -> list of content strings (without category prefix).
    Non-parseable lines go under '_other'.
    """
    from collections import OrderedDict
    grouped: dict[str, list[str]] = OrderedDict()
    for line in facts:
        obs = _parse_observation(line)
        if obs:
            cat = obs["category"]
            # Rebuild display content: (date) content [valence] #tags
            parts = []
            if obs.get("date"):
                parts.append(f"({obs['date']})")
            parts.append(obs["content"])
            if obs.get("valence"):
                markers = {"positive": "[+]", "negative": "[-]", "neutral": "[~]"}
                if obs["valence"] in markers:
                    parts.append(markers[obs["valence"]])
            for tag in obs.get("tags", []):
                parts.append(f"#{tag}")
            grouped.setdefault(cat, []).append(" ".join(parts))
        else:
            grouped.setdefault("_other", []).append(line.lstrip("- "))
    return grouped


def _sort_by_cluster(
    entities: list[tuple[str, GraphEntity]],
    graph: GraphData,
) -> list[tuple[str, GraphEntity]]:
    """Sort entities so that members of the same connected component are adjacent.

    Within each cluster, preserves the original order (score-descending).
    """
    if len(entities) <= 1:
        return entities

    entity_ids = {eid for eid, _ in entities}

    # Build adjacency restricted to these entities
    adj: dict[str, set[str]] = {eid: set() for eid in entity_ids}
    for rel in graph.relations:
        if rel.from_entity in entity_ids and rel.to_entity in entity_ids:
            adj[rel.from_entity].add(rel.to_entity)
            adj[rel.to_entity].add(rel.from_entity)

    # BFS connected components
    visited: set[str] = set()
    cluster_map: dict[str, int] = {}
    cluster_id = 0
    for eid in [e for e, _ in entities]:  # iterate in score order
        if eid in visited:
            continue
        queue = [eid]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            cluster_map[node] = cluster_id
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        cluster_id += 1

    # Sort: primary by first-appearance order of cluster, secondary by original order
    cluster_first_idx: dict[int, int] = {}
    for i, (eid, _) in enumerate(entities):
        cid = cluster_map.get(eid, 0)
        if cid not in cluster_first_idx:
            cluster_first_idx[cid] = i

    original_pos = {eid: i for i, (eid, _) in enumerate(entities)}
    return sorted(entities, key=lambda x: (cluster_first_idx.get(cluster_map.get(x[0], 0), 0), original_pos.get(x[0], 0)))


# ── Natural context helpers ──────────────────────────────────

RELATION_NATURAL = {
    "parent_of": "parent de",
    "lives_with": "vit avec",
    "works_at": "travaille chez",
    "friend_of": "ami(e) de",
    "affects": "lié à",
    "improves": "amélioré par",
    "worsens": "aggravé par",
    "uses": "utilise",
    "part_of": "fait partie de",
    "linked_to": "lié à",
    "requires": "nécessite",
    "contrasts_with": "contraste avec",
    "precedes": "précède",
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
    """Classify entity into 'long_terme', 'moyen_terme', or 'court_terme'."""
    last = date.fromisoformat(entity.last_mentioned) if entity.last_mentioned else None
    if not last:
        return "long_terme"
    days_since = (today - last).days

    # Stable entities (people, animals with long history) → always long term
    is_stable = (
        entity.type in ("person", "animal")
        and entity.retention in ("long_term", "permanent")
        and entity.frequency >= 5
    )
    if is_stable:
        return "long_terme"

    if days_since <= 7:
        return "court_terme"
    elif days_since <= 30:
        return "moyen_terme"
    else:
        return "long_terme"


def _build_natural_bullet(
    eid: str, entity: GraphEntity, graph: GraphData, memory_path: Path,
) -> str:
    """Build a natural language bullet point for an entity."""
    # 1. Base: summary or most recent/important fact
    if entity.summary:
        base = entity.summary
    else:
        facts = _read_entity_facts(eid, entity, memory_path)
        facts = [f for f in facts if "[superseded]" not in f]
        if facts:
            # Take the last fact (most recently added), clean markers
            obs = _parse_observation(facts[-1])
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
) -> list[str]:
    """Extract vigilance/diagnosis/treatment items as natural bullets."""
    vigilances = []
    for eid, entity in all_selected:
        facts = _read_entity_facts(eid, entity, memory_path)
        for f in facts:
            obs = _parse_observation(f)
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
    return unique[:15]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: words * 1.3. Delegates to shared util."""
    return _estimate_tokens_util(text)


def _enrich_entity_natural(
    entity_id: str, entity: GraphEntity, graph: GraphData, memory_path: Path,
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
        lines.append(f"Résumé: {entity.summary}")
    if facts:
        facts = [f for f in facts if "[superseded]" not in f]
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
    import logging as _logging

    if not entities:
        return ""

    # Build enriched dossier
    dossier_parts = []
    for eid, entity in entities:
        dossier_parts.append(_enrich_entity_natural(eid, entity, graph, memory_path))
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
        _logging.getLogger(__name__).warning("LLM natural section '%s' failed: %s", section_name, e)

    # Deterministic fallback
    lines = []
    used = 0
    for eid, entity in entities:
        bullet = _build_natural_bullet(eid, entity, graph, memory_path)
        cost = _estimate_tokens(bullet)
        if used + cost > section_budget and lines:
            break
        lines.append(bullet)
        used += cost
    return "\n".join(lines)


def _enrich_entity(
    entity_id: str, entity: GraphEntity, graph: GraphData, memory_path: Path,
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
        max_cat = 3 if is_ai_self else 5
        facts = [f for f in facts if "[superseded]" not in f]
        sorted_facts = _sort_facts_by_date(facts)
        sorted_facts = _deduplicate_facts_for_context(sorted_facts, max_per_category=max_cat)
        # Group by category for cleaner output
        grouped = _group_facts_by_category(sorted_facts)
        for cat, cat_facts in grouped.items():
            section_lines.append(f"  [{cat}]")
            for content in cat_facts:
                section_lines.append(f"    - {content}")

    # Get relations for this entity from graph (filter weak/stale)
    today = date.today()
    min_rel_strength = 0.3
    max_rel_age_days = 365
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
) -> str:
    """Collect enriched dossiers for entities, respecting a token budget."""
    parts: list[str] = []
    used = 0
    for eid, entity in entities:
        dossier = _enrich_entity(eid, entity, graph, memory_path)
        cost = _estimate_tokens(dossier)
        if used + cost > token_budget and parts:
            break
        parts.append(dossier)
        used += cost
    return "\n\n".join(parts)


def build_natural_context(
    graph: GraphData, memory_path: Path, config: Config,
    *, use_llm: bool = False,
) -> str:
    """Build _context.md in natural Claude Chat-like format.

    When use_llm=True, each section is processed by the LLM for narrative
    quality, with deterministic fallback on failure.
    """
    today = date.today()
    today_str = today.isoformat()

    # Template
    template_path = config.prompts_path / "context_natural.md"
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
    else:
        template = (
            "# Mémoire Personnelle — {date}\n\n{ai_personality}\n\n---\n\n"
            "{sections}\n\n---\n\n{available_entities}\n\n{extended_memory}\n\n{custom_instructions}"
        )

    # Custom instructions
    instructions_path = config.prompts_path / "context_instructions.md"
    custom_instructions = ""
    if instructions_path.exists():
        custom_instructions = instructions_path.read_text(encoding="utf-8")

    # Budget
    reserved = 400
    total_budget = max(config.context_max_tokens - reserved, 1000)
    budget = config.context_budget or {}

    # Entities
    min_score = config.scoring.min_score_for_context
    all_top = get_top_entities(graph, n=50, include_permanent=True, min_score=min_score)
    selected = _select_entities_for_natural(all_top, graph)

    # AI Personality
    ai_parts = []
    for eid, e in selected:
        if e.type == "ai_self":
            ai_parts.append(_build_natural_bullet(eid, e, graph, memory_path))
    ai_personality = "\n".join(ai_parts) if ai_parts else ""

    # Classify temporally (excluding ai_self)
    long_terme: list[tuple[str, GraphEntity]] = []
    moyen_terme: list[tuple[str, GraphEntity]] = []
    court_terme: list[tuple[str, GraphEntity]] = []
    for eid, entity in selected:
        if entity.type == "ai_self":
            continue
        tier = _classify_temporal(entity, today)
        if tier == "long_terme":
            long_terme.append((eid, entity))
        elif tier == "moyen_terme":
            moyen_terme.append((eid, entity))
        else:
            court_terme.append((eid, entity))

    # Build sections with token budget
    def build_section(entities: list[tuple[str, GraphEntity]], budget_key: str, default_pct: int = 25, section_label: str = "") -> str:
        pct = budget.get(budget_key, default_pct)
        sb = int(total_budget * pct / 100)
        if use_llm:
            return _build_section_llm(section_label or budget_key, entities, graph, memory_path, config, sb)
        lines = []
        used = 0
        for eid, entity in entities:
            bullet = _build_natural_bullet(eid, entity, graph, memory_path)
            cost = _estimate_tokens(bullet)
            if used + cost > sb and lines:
                break
            lines.append(bullet)
            used += cost
        return "\n".join(lines)

    long_text = build_section(long_terme, "long_terme", 35, "Identité & long terme")
    moyen_text = build_section(moyen_terme, "moyen_terme", 25, "En ce moment")
    court_text = build_section(court_terme, "court_terme", 20, "Cette semaine")

    # Vigilances
    vigilances = _extract_vigilances(selected, graph, memory_path)
    vigilance_text = "\n".join(vigilances)

    # Assemble sections
    sections_parts = []
    if long_text:
        sections_parts.append(f"## Identité & long terme\n\n{long_text}")
    if moyen_text:
        sections_parts.append(f"## En ce moment\n\n{moyen_text}")
    if court_text:
        sections_parts.append(f"## Cette semaine\n\n{court_text}")
    if vigilance_text:
        sections_parts.append(f"## Points de vigilance\n\n{vigilance_text}")

    # Available entities (not shown in detail)
    shown_ids = {eid for eid, _ in selected}
    remaining = sorted(
        [(eid, graph.entities[eid]) for eid in graph.entities if eid not in shown_ids],
        key=lambda x: x[1].score,
        reverse=True,
    )[:30]
    available_text = (
        "Autres sujets en mémoire : " + ", ".join(e.title for _, e in remaining)
        if remaining
        else ""
    )

    extended = "Pour plus de détails sur un sujet, utilise l'outil `search_rag`."

    # Assemble template
    result = template
    result = result.replace("{date}", today_str)
    result = result.replace("{user_language_name}", config.user_language_name)
    result = result.replace("{ai_personality}", ai_personality)
    result = result.replace("{sections}", "\n\n".join(sections_parts))
    result = result.replace("{available_entities}", available_text)
    result = result.replace("{extended_memory}", extended)
    result = result.replace("{custom_instructions}", custom_instructions)

    return result


def build_context(
    graph: GraphData, memory_path: Path, config: Config,
) -> str:
    """Build _context.md using template + enriched entity dossiers + token budget."""
    today = date.today()
    today_str = today.isoformat()

    # Load template
    template_path = config.prompts_path / "context_template.md"
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
    else:
        template = "# Personal Memory — {date}\n\n{sections}\n\n{available_entities}\n{custom_instructions}"

    # Load custom instructions
    instructions_path = config.prompts_path / "context_instructions.md"
    custom_instructions = ""
    if instructions_path.exists():
        custom_instructions = instructions_path.read_text(encoding="utf-8")

    # Budget calculation
    reserved = 500
    total_budget = max(config.context_max_tokens - reserved, 1000)
    budget = config.context_budget or {}

    def section_budget(key: str) -> int:
        pct = budget.get(key, 10)
        return int(total_budget * pct / 100)

    # Get all scored entities above threshold
    min_score = config.scoring.min_score_for_context
    all_top = get_top_entities(graph, n=50, include_permanent=True, min_score=min_score)
    shown_ids: set[str] = set()

    # AI Personality
    ai_personality_parts = []
    for eid, entity in all_top:
        if entity.type == "ai_self":
            dossier = _enrich_entity(eid, entity, graph, memory_path)
            ai_personality_parts.append(dossier)
            shown_ids.add(eid)
    ai_personality = "\n\n".join(ai_personality_parts) if ai_personality_parts else "No personality data yet."

    # Identity
    identity_entities = [(eid, e) for eid, e in all_top if e.file.startswith("self/") and e.type != "ai_self" and eid not in shown_ids]
    identity_text = _collect_section(identity_entities, graph, memory_path, section_budget("identity"))
    for eid, _ in identity_entities:
        shown_ids.add(eid)

    # Work context
    work_entities = [(eid, e) for eid, e in all_top if e.type in ("work", "organization") and eid not in shown_ids]
    work_text = _collect_section(work_entities, graph, memory_path, section_budget("work"))
    for eid, _ in work_entities:
        shown_ids.add(eid)

    # Personal context
    personal_entities = [(eid, e) for eid, e in all_top if e.type in ("person", "animal", "place") and eid not in shown_ids]
    personal_text = _collect_section(personal_entities, graph, memory_path, section_budget("personal"))
    for eid, _ in personal_entities:
        shown_ids.add(eid)

    # Top of mind (everything remaining scored high, grouped by cluster affinity)
    top_entities = [(eid, e) for eid, e in all_top if eid not in shown_ids]
    top_entities.sort(key=lambda x: x[1].score, reverse=True)
    top_entities = top_entities[:10]
    # Group by connected component so related entities appear together
    top_entities = _sort_by_cluster(top_entities, graph)
    top_text = _collect_section(top_entities, graph, memory_path, section_budget("top_of_mind"))
    for eid, _ in top_entities:
        shown_ids.add(eid)

    # Vigilances (compact quick-reference: entity + key content, max 2 per entity)
    vigilance_parts = []
    for eid, entity in all_top:
        if eid in shown_ids and entity.type != "ai_self":
            entity_path = (memory_path / entity.file).resolve()
            if entity_path.exists() and entity_path.is_relative_to(memory_path.resolve()):
                try:
                    _, sections = read_entity(entity_path)
                    facts = sections.get("Facts", [])
                    vig_facts = [f for f in facts if "[vigilance]" in f.lower() or "[diagnosis]" in f.lower() or "[treatment]" in f.lower()]
                    for vf in vig_facts[:2]:  # Max 2 per entity to reduce duplication
                        obs = _parse_observation(vf)
                        content = obs["content"] if obs else vf
                        vigilance_parts.append(f"- {entity.title}: {content}")
                except Exception:
                    pass
    vigilance_text = "\n".join(vigilance_parts)

    # Brief history — split by timeline
    thirty_days = (today - timedelta(days=30)).isoformat()
    one_year = (today - timedelta(days=365)).isoformat()

    remaining_for_history = [(eid, e) for eid, e in all_top if eid not in shown_ids]
    history_recent = [(eid, e) for eid, e in remaining_for_history if e.last_mentioned >= thirty_days]
    history_earlier = [(eid, e) for eid, e in remaining_for_history if thirty_days > e.last_mentioned >= one_year]
    history_longterm = [(eid, e) for eid, e in remaining_for_history if e.last_mentioned < one_year]

    history_recent_text = _collect_section(history_recent, graph, memory_path, section_budget("history_recent"))
    history_earlier_text = _collect_section(history_earlier, graph, memory_path, section_budget("history_earlier"))
    history_longterm_text = _collect_section(history_longterm, graph, memory_path, section_budget("history_longterm"))

    # Assemble sections (skip empty)
    sections_parts = []
    if identity_text:
        sections_parts.append(f"## Identity\n\n{identity_text}")
    if work_text:
        sections_parts.append(f"## Work context\n\n{work_text}")
    if personal_text:
        sections_parts.append(f"## Personal context\n\n{personal_text}")
    if top_text:
        sections_parts.append(f"## Top of mind\n\n{top_text}")
    if vigilance_text:
        sections_parts.append(f"## Vigilances\n\n{vigilance_text}")
    if history_recent_text:
        sections_parts.append(f"## Brief history — Recent\n\n{history_recent_text}")
    if history_earlier_text:
        sections_parts.append(f"## Brief history — Earlier\n\n{history_earlier_text}")
    if history_longterm_text:
        sections_parts.append(f"## Brief history — Long-term\n\n{history_longterm_text}")

    sections_text = "\n\n".join(sections_parts)

    # Available entities (not shown in detail)
    all_entity_ids = set(graph.entities.keys())
    remaining = all_entity_ids - shown_ids
    remaining_sorted = sorted(
        [(eid, graph.entities[eid]) for eid in remaining],
        key=lambda x: x[1].score, reverse=True,
    )[:30]
    available_text = " | ".join(f"{e.title} ({e.type})" for _, e in remaining_sorted) if remaining_sorted else ""

    # Assemble template
    result = template
    result = result.replace("{date}", today_str)
    result = result.replace("{user_language_name}", config.user_language_name)
    result = result.replace("{ai_personality}", ai_personality)
    result = result.replace("{sections}", sections_text)
    result = result.replace("{available_entities}", available_text)
    result = result.replace("{custom_instructions}", custom_instructions)

    return result


def _rag_prefetch(
    entity_ids: list[str],
    graph: GraphData,
    config: Config,
    memory_path: Path,
    max_results_per_entity: int = 2,
) -> str:
    """Pre-fetch related facts from FAISS for a list of entities.

    Returns a text block with related memories to inject in the LLM prompt.
    Gracefully returns empty string if FAISS is unavailable.
    """
    try:
        from src.pipeline.indexer import search as faiss_search
    except ImportError:
        return ""

    seen_chunks: set[str] = set()
    rag_lines: list[str] = []

    for eid in entity_ids:
        entity = graph.entities.get(eid)
        if not entity:
            continue
        try:
            results = faiss_search(entity.title, config, memory_path, top_k=max_results_per_entity)
            for r in results:
                # Skip self-references and duplicates
                if r.entity_id in entity_ids or r.chunk in seen_chunks:
                    continue
                seen_chunks.add(r.chunk)
                related_entity = graph.entities.get(r.entity_id)
                title = related_entity.title if related_entity else r.entity_id
                rag_lines.append(f"- [{title}] {r.chunk[:200]}")
        except Exception:
            continue

    return "\n".join(rag_lines[:15])  # Cap at 15 RAG results total


def build_context_with_llm(
    graph: GraphData, memory_path: Path, config: Config,
) -> str:
    """Build _context.md using per-section LLM cleanup with RAG pre-fetch.

    Each section is processed independently by the LLM for better quality
    with small models. Falls back to deterministic if LLM calls fail.
    """
    import logging
    from src.core.llm import call_context_section

    logger = logging.getLogger(__name__)
    today = date.today()
    today_str = today.isoformat()

    # Load template
    template_path = config.prompts_path / "context_template.md"
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
    else:
        template = "# Personal Memory — {date}\n\n{sections}\n\n{available_entities}\n{custom_instructions}"

    # Load custom instructions
    instructions_path = config.prompts_path / "context_instructions.md"
    custom_instructions = ""
    if instructions_path.exists():
        custom_instructions = instructions_path.read_text(encoding="utf-8")

    # Budget calculation
    reserved = 500
    total_budget = max(config.context_max_tokens - reserved, 1000)
    budget = config.context_budget or {}

    def section_budget(key: str) -> int:
        pct = budget.get(key, 10)
        return int(total_budget * pct / 100)

    # Get all scored entities above threshold
    min_score = config.scoring.min_score_for_context
    all_top = get_top_entities(graph, n=50, include_permanent=True, min_score=min_score)
    shown_ids: set[str] = set()

    # Group entities by section type
    ai_entities = [(eid, e) for eid, e in all_top if e.type == "ai_self"]
    identity_entities = [(eid, e) for eid, e in all_top if e.file.startswith("self/") and e.type != "ai_self"]
    work_entities = [(eid, e) for eid, e in all_top if e.type in ("work", "organization")]
    personal_entities = [(eid, e) for eid, e in all_top if e.type in ("person", "animal", "place")]
    # Remove already-assigned from personal
    assigned = {eid for eid, _ in ai_entities + identity_entities + work_entities}
    personal_entities = [(eid, e) for eid, e in personal_entities if eid not in assigned]

    # Helper: build dossier + call LLM for a section
    def _llm_section(name: str, entities: list[tuple[str, GraphEntity]], budget_key: str) -> str:
        if not entities:
            return ""
        eids = [eid for eid, _ in entities]
        # Build raw dossier
        dossier_parts = []
        for eid, entity in entities:
            dossier_parts.append(_enrich_entity(eid, entity, graph, memory_path))
            shown_ids.add(eid)
        raw_dossier = "\n\n".join(dossier_parts)

        # RAG pre-fetch
        rag_text = _rag_prefetch(eids, graph, config, memory_path)

        # LLM call
        try:
            result = call_context_section(
                section_name=name,
                entities_dossier=raw_dossier,
                rag_context=rag_text,
                budget_tokens=section_budget(budget_key),
                config=config,
            )
            if result.strip():
                return result
        except Exception as e:
            logger.warning("LLM section '%s' failed, using deterministic: %s", name, e)

        # Fallback: return raw dossier
        return raw_dossier

    # Generate each section with LLM
    ai_personality = _llm_section("AI Personality & Interaction Style", ai_entities, "ai_personality")
    if not ai_personality:
        ai_personality = "No personality data yet."

    identity_text = _llm_section("Identity (health, self)", identity_entities, "identity")
    work_text = _llm_section("Work Context", work_entities, "work")
    personal_text = _llm_section("Personal Context (people, places, animals)", personal_entities, "personal")

    # Top of mind (everything remaining)
    top_entities = [(eid, e) for eid, e in all_top if eid not in shown_ids]
    top_entities.sort(key=lambda x: x[1].score, reverse=True)
    top_entities = top_entities[:10]
    top_entities = _sort_by_cluster(top_entities, graph)
    top_text = _llm_section("Top of Mind (current priorities)", top_entities, "top_of_mind")
    for eid, _ in top_entities:
        shown_ids.add(eid)

    # Vigilances — deterministic (no LLM needed for safety-critical data)
    vigilance_parts = []
    for eid, entity in all_top:
        if eid in shown_ids and entity.type != "ai_self":
            entity_path = (memory_path / entity.file).resolve()
            if entity_path.exists() and entity_path.is_relative_to(memory_path.resolve()):
                try:
                    _, sections = read_entity(entity_path)
                    facts = sections.get("Facts", [])
                    vig_facts = [f for f in facts if "[vigilance]" in f.lower() or "[diagnosis]" in f.lower() or "[treatment]" in f.lower()]
                    for vf in vig_facts[:2]:
                        obs = _parse_observation(vf)
                        content = obs["content"] if obs else vf
                        vigilance_parts.append(f"- {entity.title}: {content}")
                except Exception:
                    pass
    vigilance_text = "\n".join(vigilance_parts)

    # Assemble sections
    sections_parts = []
    if identity_text:
        sections_parts.append(f"## Identity\n\n{identity_text}")
    if work_text:
        sections_parts.append(f"## Work context\n\n{work_text}")
    if personal_text:
        sections_parts.append(f"## Personal context\n\n{personal_text}")
    if top_text:
        sections_parts.append(f"## Top of mind\n\n{top_text}")
    if vigilance_text:
        sections_parts.append(f"## Vigilances\n\n{vigilance_text}")
    sections_text = "\n\n".join(sections_parts)

    # Available entities
    all_entity_ids = set(graph.entities.keys())
    remaining = all_entity_ids - shown_ids
    remaining_sorted = sorted(
        [(eid, graph.entities[eid]) for eid in remaining],
        key=lambda x: x[1].score, reverse=True,
    )[:30]
    available_text = " | ".join(f"{e.title} ({e.type})" for _, e in remaining_sorted) if remaining_sorted else ""

    # Assemble template
    result = template
    result = result.replace("{date}", today_str)
    result = result.replace("{user_language_name}", config.user_language_name)
    result = result.replace("{ai_personality}", ai_personality)
    result = result.replace("{sections}", sections_text)
    result = result.replace("{available_entities}", available_text)
    result = result.replace("{custom_instructions}", custom_instructions)

    return result


def build_context_input(graph: GraphData, memory_path: Path, config: Config) -> str:
    """Build enriched dossier for LLM narrative mode input."""
    top = get_top_entities(
        graph,
        n=15,
        include_permanent=True,
        min_score=config.scoring.min_score_for_context,
    )
    sections = []
    for entity_id, entity in top:
        dossier = _enrich_entity(entity_id, entity, graph, memory_path)
        sections.append(dossier)
    return "\n\n".join(sections)


def generate_context(enriched_input: str, config: Config) -> str:
    """Generate _context.md content using LLM."""
    return call_context_generation(enriched_input, config)


def write_context(memory_path: Path, content: str) -> None:
    """Write _context.md file."""
    (memory_path / "_context.md").write_text(content, encoding="utf-8")


def generate_index(graph: GraphData) -> str:
    """Generate _index.md from graph data (deterministic, no LLM)."""
    lines = ["# Memory Index\n"]
    lines.append(f"Generated: {graph.generated}\n")
    lines.append(f"Total entities: {len(graph.entities)}")
    lines.append(f"Total relations: {len(graph.relations)}\n")

    # Group by type
    by_type: dict[str, list[tuple[str, any]]] = {}
    for entity_id, entity in sorted(graph.entities.items()):
        by_type.setdefault(entity.type, []).append((entity_id, entity))

    for type_name, entities in sorted(by_type.items()):
        lines.append(f"\n## {type_name.capitalize()}\n")
        lines.append("| Entity | Score | Frequency | Last Mentioned |")
        lines.append("|--------|-------|-----------|----------------|")
        for entity_id, entity in sorted(entities, key=lambda x: x[1].score, reverse=True):
            lines.append(f"| {entity.title} | {entity.score:.2f} | {entity.frequency} | {entity.last_mentioned} |")

    # Relations summary
    if graph.relations:
        lines.append("\n## Relations\n")
        for rel in graph.relations:
            from_title = graph.entities.get(rel.from_entity, None)
            to_title = graph.entities.get(rel.to_entity, None)
            from_name = from_title.title if from_title else rel.from_entity
            to_name = to_title.title if to_title else rel.to_entity
            lines.append(f"- {from_name} → {rel.type} → {to_name}")

    return "\n".join(lines)


def write_index(memory_path: Path, graph: GraphData) -> None:
    """Write _index.md file."""
    content = generate_index(graph)
    (memory_path / "_index.md").write_text(content, encoding="utf-8")
