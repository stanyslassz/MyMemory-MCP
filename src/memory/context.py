"""Context generation: builds enriched dossier and generates _context.md and _index.md."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from src.core.config import Config
from src.core.llm import call_context_generation
from src.core.models import GraphData
from src.memory.graph import get_related
from src.memory.scoring import get_top_entities
from src.memory.store import read_entity


def build_context_input(graph: GraphData, memory_path: Path, config: Config) -> str:
    """Build enriched dossier from top entities + BFS depth 1 neighbors.

    Returns a structured text with entity facts and relations.
    """
    top = get_top_entities(
        graph,
        n=15,
        include_permanent=True,
        min_score=config.scoring.min_score_for_context,
    )

    seen_ids = set()
    sections = []

    for entity_id, entity in top:
        if entity_id in seen_ids:
            continue
        seen_ids.add(entity_id)

        # Read entity facts (with path traversal guard)
        entity_path = (memory_path / entity.file).resolve()
        facts = []
        if not entity_path.is_relative_to(memory_path.resolve()):
            continue
        if entity_path.exists():
            try:
                _, entity_sections = read_entity(entity_path)
                facts = entity_sections.get("Facts", [])
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

        sections.append("\n".join(section_lines))

    return "\n\n".join(sections)


def build_deterministic_context(
    graph: GraphData, memory_path: Path, config: Config
) -> str:
    """Build _context.md using deterministic template + pre-computed summaries.

    No LLM call. Sections filled from entity summaries sorted by score.
    """
    today = date.today().isoformat()
    top = get_top_entities(
        graph, n=15, include_permanent=True,
        min_score=config.scoring.min_score_for_context,
    )
    top_ids = {eid for eid, _ in top}

    # Categorize top entities
    identity_entities: list[dict] = []
    vigilance_entities: list[dict] = []
    work_project_entities: list[dict] = []
    close_ones_entities: list[dict] = []
    other_entities: list[dict] = []

    for eid, entity in top:
        # Read entity to check for vigilance/diagnosis observations
        entity_path = (memory_path / entity.file).resolve()
        facts: list[str] = []
        has_vigilance = False
        if entity_path.exists() and entity_path.is_relative_to(memory_path.resolve()):
            try:
                _, sections = read_entity(entity_path)
                facts = sections.get("Facts", [])
                for f in facts:
                    if "[vigilance]" in f.lower() or "[diagnosis]" in f.lower():
                        has_vigilance = True
            except Exception:
                pass

        # Get summary (or fallback to top 3 facts)
        summary = entity.summary
        if not summary and facts:
            summary = " ".join(facts[:3])

        entry = {"id": eid, "entity": entity, "summary": summary, "facts": facts}

        # Categorize
        if entity.file.startswith("self/"):
            identity_entities.append(entry)
        elif has_vigilance:
            vigilance_entities.append(entry)
        elif entity.type in ("work", "project"):
            work_project_entities.append(entry)
        elif entity.type in ("person", "animal"):
            close_ones_entities.append(entry)
        else:
            other_entities.append(entry)

    lines = [f"# Memory Context — {today}\n"]

    # Identity
    lines.append("## Identity\n")
    for entry in identity_entities:
        if entry["summary"]:
            lines.append(f"{entry['summary']}\n")
    if not identity_entities:
        lines.append("No identity information available.\n")

    # Top of mind (everything in top, sorted by score)
    lines.append("## Top of mind\n")
    all_top = sorted(
        [e for e in (other_entities + work_project_entities + close_ones_entities)],
        key=lambda e: e["entity"].score, reverse=True
    )
    for entry in all_top[:10]:
        e = entry["entity"]
        summary = entry["summary"] or f"({e.type}, score: {e.score:.2f})"
        lines.append(f"**{e.title}** ({e.type}, {e.score:.2f}): {summary}\n")

    # Vigilances
    lines.append("## Vigilances\n")
    if vigilance_entities:
        for entry in vigilance_entities:
            vig_facts = [f for f in entry["facts"] if "[vigilance]" in f.lower() or "[diagnosis]" in f.lower()]
            for f in vig_facts:
                lines.append(f"- {entry['entity'].title}: {f}\n")
    else:
        lines.append("No active vigilances.\n")

    # Work & Projects
    lines.append("## Work & Projects\n")
    for entry in work_project_entities:
        e = entry["entity"]
        summary = entry["summary"] or "(no summary)"
        lines.append(f"**{e.title}** ({e.score:.2f}): {summary}\n")
    if not work_project_entities:
        lines.append("No active work or projects.\n")

    # Close ones
    lines.append("## Close ones\n")
    for entry in close_ones_entities:
        e = entry["entity"]
        summary = entry["summary"] or "(no summary)"
        lines.append(f"**{e.title}** ({e.score:.2f}): {summary}\n")
    if not close_ones_entities:
        lines.append("No close ones in memory.\n")

    # Available in memory (entities NOT in top)
    lines.append("## Available in memory (not detailed above)\n")
    all_entities_sorted = sorted(
        [(eid, e) for eid, e in graph.entities.items() if eid not in top_ids],
        key=lambda x: x[1].score, reverse=True
    )[:30]
    if all_entities_sorted:
        available_parts = [f"{e.title} ({e.type}, {e.score:.2f})" for _, e in all_entities_sorted]
        lines.append(" | ".join(available_parts) + "\n")
    else:
        lines.append("No additional entities in memory.\n")

    # Memory tags
    lines.append("## Memory tags\n")
    tag_scores: dict[str, list[float]] = {}
    for _, entity in graph.entities.items():
        for tag in entity.tags:
            tag_scores.setdefault(tag, []).append(entity.score)
    weighted_tags: list[tuple[str, float]] = []
    for tag, scores in tag_scores.items():
        avg_score = sum(scores) / len(scores) if scores else 0
        weighted_tags.append((tag, round(avg_score, 1)))
    weighted_tags.sort(key=lambda x: x[1], reverse=True)
    if weighted_tags:
        tag_parts = [f"#{tag}({score})" for tag, score in weighted_tags[:20]]
        lines.append(" ".join(tag_parts) + "\n")
    else:
        lines.append("No tags available.\n")

    return "\n".join(lines)


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
