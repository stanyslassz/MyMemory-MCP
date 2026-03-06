"""Context generation: builds enriched dossier and generates _context.md and _index.md."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from src.core.config import Config
from src.core.llm import call_context_generation
from src.core.models import GraphData, GraphEntity
from src.memory.graph import get_related
from src.memory.scoring import get_top_entities
from src.memory.store import read_entity


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: words * 1.3."""
    return int(len(text.split()) * 1.3)


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

    # Get related entities (BFS depth 1)
    related_ids = get_related(graph, entity_id, depth=1)
    related_info = []
    for rel_id in related_ids:
        if rel_id in graph.entities:
            rel_entity = graph.entities[rel_id]
            related_info.append(f"{rel_entity.title} ({rel_entity.type})")

    # Build section
    section_lines = [
        f"### {entity.title} [{entity.type}] (score: {entity.score:.2f}, retention: {entity.retention})",
    ]
    if entity.tags:
        section_lines.append(f"Tags: {', '.join(entity.tags)}")
    if facts:
        section_lines.append("Facts:")
        for fact in facts:
            section_lines.append(f"  {fact}")
    if related_info:
        section_lines.append(f"Related: {', '.join(related_info)}")

    # Get relations for this entity from graph
    entity_relations = []
    for rel in graph.relations:
        if rel.from_entity == entity_id:
            target = graph.entities.get(rel.to_entity)
            if target:
                entity_relations.append(f"  → {rel.type} {target.title}")
        elif rel.to_entity == entity_id:
            source = graph.entities.get(rel.from_entity)
            if source:
                entity_relations.append(f"  ← {rel.type} {source.title}")
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

    # Top of mind (everything remaining scored high)
    top_entities = [(eid, e) for eid, e in all_top if eid not in shown_ids]
    top_entities.sort(key=lambda x: x[1].score, reverse=True)
    top_text = _collect_section(top_entities[:10], graph, memory_path, section_budget("top_of_mind"))
    for eid, _ in top_entities[:10]:
        shown_ids.add(eid)

    # Vigilances (scan shown entities for vigilance/diagnosis facts)
    vigilance_parts = []
    for eid, entity in all_top:
        if eid in shown_ids and entity.type != "ai_self":
            entity_path = (memory_path / entity.file).resolve()
            if entity_path.exists() and entity_path.is_relative_to(memory_path.resolve()):
                try:
                    _, sections = read_entity(entity_path)
                    facts = sections.get("Facts", [])
                    vig_facts = [f for f in facts if "[vigilance]" in f.lower() or "[diagnosis]" in f.lower() or "[treatment]" in f.lower()]
                    for vf in vig_facts:
                        vigilance_parts.append(f"- {entity.title}: {vf}")
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


def build_deterministic_context(
    graph: GraphData, memory_path: Path, config: Config,
) -> str:
    """Legacy: delegate to build_context."""
    return build_context(graph, memory_path, config)


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
