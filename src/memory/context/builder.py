"""Context builder: main build functions, RAG helpers, and write helpers."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from src.core.config import Config
from src.core.models import GraphData, GraphEntity
from src.memory.scoring import get_top_entities
from src.memory.store import read_entity, parse_observation, _atomic_write_text
from src.memory.context.utilities import _sort_by_cluster, _estimate_tokens
from src.memory.context.formatter import (
    _select_entities_for_natural,
    _classify_temporal,
    _build_natural_bullet,
    _extract_vigilances,
    _enrich_entity_natural,
    _build_section_llm,
    _enrich_entity,
    _collect_section,
)


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
            "# Personal Memory — {date}\n\n{ai_personality}\n\n---\n\n"
            "{sections}\n\n---\n\n{available_entities}\n\n{extended_memory}\n\n{custom_instructions}"
        )

    # Custom instructions
    instructions_path = config.prompts_path / "context_instructions.md"
    custom_instructions = ""
    if instructions_path.exists():
        custom_instructions = instructions_path.read_text(encoding="utf-8")

    # Budget
    reserved = config.ctx.reserved_tokens_natural
    total_budget = max(config.context_max_tokens - reserved, config.ctx.min_budget_tokens)
    budget = config.context_budget or {}

    # Entities
    min_score = config.scoring.min_score_for_context
    all_top = get_top_entities(graph, n=config.ctx.top_entities_count, include_permanent=True, min_score=min_score)
    selected = _select_entities_for_natural(all_top, graph)

    # AI Personality
    ai_entities = [(eid, e) for eid, e in selected if e.type == "ai_self"]
    if ai_entities:
        pct = budget.get("ai_personality", 8)
        sb = int(total_budget * pct / 100)
        if use_llm:
            ai_personality = _build_section_llm("AI Personality", ai_entities, graph, memory_path, config, sb)
        else:
            ai_parts = []
            for eid, e in ai_entities:
                ai_parts.append(_build_natural_bullet(eid, e, graph, memory_path))
            ai_personality = "\n".join(ai_parts)
    else:
        ai_personality = ""

    # Classify temporally (excluding ai_self)
    long_term: list[tuple[str, GraphEntity]] = []
    medium_term: list[tuple[str, GraphEntity]] = []
    short_term: list[tuple[str, GraphEntity]] = []
    for eid, entity in selected:
        if entity.type == "ai_self":
            continue
        tier = _classify_temporal(entity, today)
        if tier == "long_term":
            long_term.append((eid, entity))
        elif tier == "medium_term":
            medium_term.append((eid, entity))
        else:
            short_term.append((eid, entity))

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

    long_text = build_section(long_term, "long_term", 35, "Identity & long term")
    moyen_text = build_section(medium_term, "medium_term", 25, "Medium term")
    court_text = build_section(short_term, "short_term", 20, "Short term")

    # Vigilances
    vigilances = _extract_vigilances(selected, graph, memory_path, config)
    vigilance_text = "\n".join(vigilances)

    # Assemble sections
    sections_parts = []
    if long_text:
        sections_parts.append(f"## Identity & long term\n\n{long_text}")
    if moyen_text:
        sections_parts.append(f"## Medium term\n\n{moyen_text}")
    if court_text:
        sections_parts.append(f"## Short term\n\n{court_text}")
    if vigilance_text:
        sections_parts.append(f"## Vigilances\n\n{vigilance_text}")

    # Available entities (not shown in detail)
    shown_ids = {eid for eid, _ in selected}
    remaining = sorted(
        [(eid, graph.entities[eid]) for eid in graph.entities if eid not in shown_ids],
        key=lambda x: x[1].score,
        reverse=True,
    )[:config.ctx.available_entities_limit]
    available_text = (
        "Other topics in memory: " + ", ".join(e.title for _, e in remaining)
        if remaining
        else ""
    )

    extended = "For more details on any topic, use the `search_rag` tool."

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
    reserved = config.ctx.reserved_tokens_structured
    total_budget = max(config.context_max_tokens - reserved, config.ctx.min_budget_tokens)
    budget = config.context_budget or {}

    def section_budget(key: str) -> int:
        pct = budget.get(key, config.ctx.default_budget_pct)
        return int(total_budget * pct / 100)

    # Get all scored entities above threshold
    min_score = config.scoring.min_score_for_context
    all_top = get_top_entities(graph, n=config.ctx.top_entities_count, include_permanent=True, min_score=min_score)
    shown_ids: set[str] = set()

    # AI Personality
    ai_personality_parts = []
    for eid, entity in all_top:
        if entity.type == "ai_self":
            dossier = _enrich_entity(eid, entity, graph, memory_path, config)
            ai_personality_parts.append(dossier)
            shown_ids.add(eid)
    ai_personality = "\n\n".join(ai_personality_parts) if ai_personality_parts else "No personality data yet."

    # Identity
    identity_entities = [(eid, e) for eid, e in all_top if e.file.startswith("self/") and e.type != "ai_self" and eid not in shown_ids]
    identity_text = _collect_section(identity_entities, graph, memory_path, section_budget("identity"), config)
    for eid, _ in identity_entities:
        shown_ids.add(eid)

    # Work context
    work_entities = [(eid, e) for eid, e in all_top if e.type in ("work", "organization") and eid not in shown_ids]
    work_text = _collect_section(work_entities, graph, memory_path, section_budget("work"), config)
    for eid, _ in work_entities:
        shown_ids.add(eid)

    # Personal context
    personal_entities = [(eid, e) for eid, e in all_top if e.type in ("person", "animal", "place") and eid not in shown_ids]
    personal_text = _collect_section(personal_entities, graph, memory_path, section_budget("personal"), config)
    for eid, _ in personal_entities:
        shown_ids.add(eid)

    # Top of mind (everything remaining scored high, grouped by cluster affinity)
    top_entities = [(eid, e) for eid, e in all_top if eid not in shown_ids]
    top_entities.sort(key=lambda x: x[1].score, reverse=True)
    top_entities = top_entities[:config.ctx.top_of_mind_limit]
    # Group by connected component so related entities appear together
    top_entities = _sort_by_cluster(top_entities, graph)
    top_text = _collect_section(top_entities, graph, memory_path, section_budget("top_of_mind"), config)
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
                        obs = parse_observation(vf)
                        content = obs["content"] if obs else vf
                        vigilance_parts.append(f"- {entity.title}: {content}")
                except Exception:
                    pass
    vigilance_text = "\n".join(vigilance_parts)

    # Brief history — split by timeline
    history_recent_days = config.ctx.history_recent_days
    thirty_days = (today - timedelta(days=history_recent_days)).isoformat()
    one_year = (today - timedelta(days=365)).isoformat()

    remaining_for_history = [(eid, e) for eid, e in all_top if eid not in shown_ids]
    history_recent = [(eid, e) for eid, e in remaining_for_history if e.last_mentioned >= thirty_days]
    history_earlier = [(eid, e) for eid, e in remaining_for_history if thirty_days > e.last_mentioned >= one_year]
    history_longterm = [(eid, e) for eid, e in remaining_for_history if e.last_mentioned < one_year]

    history_recent_text = _collect_section(history_recent, graph, memory_path, section_budget("history_recent"), config)
    history_earlier_text = _collect_section(history_earlier, graph, memory_path, section_budget("history_earlier"), config)
    history_longterm_text = _collect_section(history_longterm, graph, memory_path, section_budget("history_longterm"), config)

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
    )[:config.ctx.available_entities_limit]
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
        from src.memory.rag import search as rag_search, SearchOptions
    except ImportError:
        return ""

    seen_chunks: set[str] = set()
    rag_lines: list[str] = []

    for eid in entity_ids:
        entity = graph.entities.get(eid)
        if not entity:
            continue
        try:
            results = rag_search(entity.title, config, memory_path, SearchOptions(
                top_k=max_results_per_entity, bump_mentions=False, use_fts5=False, rerank_actr=False,
            ))
            for r in results:
                # Skip self-references and duplicates
                if r.entity_id in entity_ids or r.chunk in seen_chunks:
                    continue
                seen_chunks.add(r.chunk)
                related_entity = graph.entities.get(r.entity_id)
                title = related_entity.title if related_entity else r.entity_id
                preview_len = config.ctx.rag_chunk_preview_len if config else 200
                rag_lines.append(f"- [{title}] {r.chunk[:preview_len]}")
        except Exception:
            continue

    max_results = config.ctx.max_rag_results if config else 15
    return "\n".join(rag_lines[:max_results])


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
    reserved = config.ctx.reserved_tokens_structured
    total_budget = max(config.context_max_tokens - reserved, config.ctx.min_budget_tokens)
    budget = config.context_budget or {}

    def section_budget(key: str) -> int:
        pct = budget.get(key, config.ctx.default_budget_pct)
        return int(total_budget * pct / 100)

    # Get all scored entities above threshold
    min_score = config.scoring.min_score_for_context
    all_top = get_top_entities(graph, n=config.ctx.top_entities_count, include_permanent=True, min_score=min_score)
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
            dossier_parts.append(_enrich_entity(eid, entity, graph, memory_path, config))
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
    top_entities = top_entities[:config.ctx.top_of_mind_limit]
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
                        obs = parse_observation(vf)
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
    )[:config.ctx.available_entities_limit]
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



def write_context(memory_path: Path, content: str) -> None:
    """Write _context.md file."""
    _atomic_write_text(memory_path / "_context.md", content)
    try:
        from src.memory.event_log import append_event
        append_event(memory_path, "context_rebuilt", "context", {})
    except Exception:
        pass


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
    _atomic_write_text(memory_path / "_index.md", content)
