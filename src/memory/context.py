"""Context generation: builds enriched dossier and generates _context.md and _index.md."""

from __future__ import annotations

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
