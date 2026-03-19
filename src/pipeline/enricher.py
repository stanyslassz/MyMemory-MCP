"""Step 4: Apply resolved extractions to memory (MD files + graph)."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from src.core.config import Config
from src.core.models import (
    EnrichmentReport,
    EntityFrontmatter,
    GraphEntity,
    GraphRelation,
    ResolvedExtraction,
)
from src.memory.context import write_index
from src.memory.graph import add_entity, add_relation, find_entity_by_name, load_graph, remove_relation, save_graph
from src.memory.scoring import recalculate_all_scores
from src.memory.store import create_entity, create_stub_entity, update_entity, mark_observation_superseded, read_entity, remove_relation_line, write_entity, consolidate_entity_facts
from src.core.utils import filter_live_facts, slugify

logger = logging.getLogger(__name__)


# Families of mutually exclusive relation types between the same entity pair.
# When a new relation of type X is added, any existing relation of a
# conflicting type Y (in the same family) is automatically removed.
EXCLUSIVE_RELATIONS: list[set[str]] = [
    {"parent_of", "friend_of"},
    {"improves", "worsens"},
]


def _initial_retention(entity_type: str) -> str:
    """Determine initial retention based on entity type."""
    if entity_type == "ai_self":
        return "permanent"
    if entity_type in ("person", "animal", "health"):
        return "long_term"
    return "short_term"


def _check_relation_conflicts(
    graph, from_entity: str, to_entity: str, new_type: str, memory_path: Path,
) -> None:
    """Auto-remove relations that contradict new_type between the same pair."""
    for family in EXCLUSIVE_RELATIONS:
        if new_type in family:
            for conflicting_type in family - {new_type}:
                # Check both directions
                if remove_relation(graph, from_entity, to_entity, conflicting_type):
                    entity_data = graph.entities.get(from_entity)
                    if entity_data:
                        to_data = graph.entities.get(to_entity)
                        if to_data:
                            entity_file = memory_path / entity_data.file
                            if entity_file.exists():
                                remove_relation_line(entity_file, conflicting_type, to_data.title)
                if remove_relation(graph, to_entity, from_entity, conflicting_type):
                    entity_data = graph.entities.get(to_entity)
                    if entity_data:
                        from_data = graph.entities.get(from_entity)
                        if from_data:
                            entity_file = memory_path / entity_data.file
                            if entity_file.exists():
                                remove_relation_line(entity_file, conflicting_type, from_data.title)
            break


def enrich_memory(
    resolved: ResolvedExtraction,
    config: Config,
    today: str | None = None,
) -> EnrichmentReport:
    """Apply resolved extraction to memory files and graph.

    For each entity:
    - resolved → update existing MD + graph
    - new → create new MD + add to graph
    - ambiguous (already arbitrated) → handled as resolved or new

    Also handles forward references and score recalculation.
    """
    if today is None:
        today = date.today().isoformat()

    memory_path = config.memory_path
    graph = load_graph(memory_path)
    report = EnrichmentReport()

    # Process each resolved entity
    for item in resolved.resolved:
        raw_entity = item.raw
        resolution = item.resolution

        try:
            if resolution.status == "resolved" and resolution.entity_id:
                _update_existing_entity(
                    resolution.entity_id, raw_entity, graph, memory_path, today, report, config
                )
            elif resolution.status == "new":
                slug = resolution.suggested_slug or slugify(raw_entity.name)
                _create_new_entity(slug, raw_entity, graph, memory_path, config, today, report)
            # "ambiguous" entities should have been arbitrated already
        except Exception as e:
            report.errors.append(f"Error processing {raw_entity.name}: {e}")

    # Process relations
    for rel in resolved.relations:
        try:
            from_slug = _find_entity_slug(rel.from_name, graph)
            to_slug = _find_entity_slug(rel.to_name, graph)

            # Create stub for forward references
            if not to_slug:
                to_slug = slugify(rel.to_name)
                if to_slug not in graph.entities:
                    folder = config.get_folder_for_type("interest")  # default type for stubs
                    create_stub_entity(memory_path, folder, to_slug, rel.to_name, "interest", today)
                    graph = add_entity(graph, to_slug, GraphEntity(
                        file=f"{folder}/{to_slug}.md",
                        type="interest",
                        title=rel.to_name,
                        importance=0.3,
                        frequency=1,
                        last_mentioned=today,
                        retention=_initial_retention("interest"),
                    ))

            if from_slug and to_slug:
                # Handle relation supersession
                if rel.supersedes:
                    parts = rel.supersedes.split(":")
                    if len(parts) == 3:
                        old_from, old_to, old_type = parts
                        remove_relation(graph, old_from, old_to, old_type)
                        # Clean MD file for the source entity
                        old_from_entity = graph.entities.get(old_from)
                        if old_from_entity:
                            old_from_path = memory_path / old_from_entity.file
                            if old_from_path.exists():
                                old_to_entity = graph.entities.get(old_to)
                                if old_to_entity:
                                    remove_relation_line(old_from_path, old_type, old_to_entity.title)

                graph_rel = GraphRelation(from_entity=from_slug, to_entity=to_slug, type=rel.type, context=rel.context)
                graph = add_relation(graph, graph_rel, strength_growth=config.scoring.relation_strength_growth)
                _check_relation_conflicts(graph, from_slug, to_slug, rel.type, memory_path)
                report.relations_added += 1

                # Also add relation text to the source entity MD
                if from_slug in graph.entities:
                    entity_file = memory_path / graph.entities[from_slug].file
                    if entity_file.exists():
                        rel_line = f"- {rel.type} [[{rel.to_name}]]"
                        update_entity(entity_file, new_relations=[rel_line])
        except Exception as e:
            report.errors.append(f"Error processing relation {rel.from_name} → {rel.to_name}: {e}")

    # Recalculate scores (also upgrades retention)
    graph = recalculate_all_scores(graph, config, date.fromisoformat(today))

    # Persist retention upgrades to MD files
    for eid, entity in graph.entities.items():
        entity_path = memory_path / entity.file
        if entity_path.exists():
            try:
                fm, sections = read_entity(entity_path)
                if fm.retention != entity.retention:
                    fm.retention = entity.retention
                    write_entity(entity_path, fm, sections)
            except Exception:
                pass

    # Save graph and regenerate index
    save_graph(memory_path, graph)
    write_index(memory_path, graph)

    return report


def _update_existing_entity(
    entity_id: str,
    raw_entity,
    graph,
    memory_path: Path,
    today: str,
    report: EnrichmentReport,
    config: Config | None = None,
) -> None:
    """Update an existing entity with new observations."""
    if entity_id not in graph.entities:
        return

    entity_meta = graph.entities[entity_id]
    filepath = memory_path / entity_meta.file

    if not filepath.exists():
        report.errors.append(f"File not found for entity {entity_id}: {entity_meta.file}")
        return

    # Handle supersession: mark old facts before adding new ones
    superseding_obs = [obs for obs in raw_entity.observations if obs.supersedes]
    if superseding_obs:
        frontmatter, sections = read_entity(filepath)
        existing_facts = sections.get("Facts", [])
        for obs in superseding_obs:
            existing_facts = mark_observation_superseded(
                existing_facts, obs.category, obs.supersedes,
            )
        sections["Facts"] = existing_facts
        write_entity(filepath, frontmatter, sections)

    # Pre-consolidation gate: if adding facts would exceed max_facts, consolidate first
    max_facts = None
    if config is not None:
        max_facts = config.get_max_facts(entity_meta.type)
        _, sections = read_entity(filepath)
        live_facts = filter_live_facts(sections.get("Facts", []))
        if len(live_facts) + len(raw_entity.observations) > max_facts:
            logger.info(
                "Pre-consolidating %s (%d + %d > %d facts)",
                entity_meta.title, len(live_facts), len(raw_entity.observations), max_facts,
            )
            try:
                consolidate_entity_facts(filepath, config, max_facts=max_facts)
            except Exception as e:
                logger.warning("Pre-consolidation failed for %s: %s", entity_id, e)

    # Prepare observations
    new_obs = [
        {"category": obs.category, "content": obs.content, "tags": obs.tags,
         "date": obs.date, "valence": obs.valence}
        for obs in raw_entity.observations
    ]

    # Compute temporal fields FIRST (before writing to MD)
    from src.memory.mentions import add_mention
    entity_meta.mention_dates, entity_meta.monthly_buckets = add_mention(
        today, entity_meta.mention_dates, entity_meta.monthly_buckets,
        window_size=50,
    )

    # Update MD file with temporal fields persisted to frontmatter
    updated_fm = update_entity(
        filepath, new_observations=new_obs, last_mentioned=today,
        max_facts=max_facts,
        mention_dates=entity_meta.mention_dates,
        monthly_buckets=entity_meta.monthly_buckets,
    )

    # Update graph metadata
    entity_meta.frequency = updated_fm.frequency
    entity_meta.last_mentioned = today

    # Update importance (running average)
    if raw_entity.observations:
        new_importance = sum(o.importance for o in raw_entity.observations) / len(raw_entity.observations)
        entity_meta.importance = (entity_meta.importance + new_importance) / 2

    # Recompute negative_valence_ratio from updated MD
    from src.core.utils import parse_frontmatter
    from src.memory.graph import compute_negative_valence_ratio
    try:
        _, body = parse_frontmatter(filepath.read_text(encoding="utf-8"))
        entity_meta.negative_valence_ratio = compute_negative_valence_ratio(body)
    except Exception:
        pass

    report.entities_updated.append(entity_id)


def _create_new_entity(
    slug: str,
    raw_entity,
    graph,
    memory_path: Path,
    config: Config,
    today: str,
    report: EnrichmentReport,
) -> None:
    """Create a new entity from extraction results."""
    if slug in graph.entities:
        # Already exists, update instead
        _update_existing_entity(slug, raw_entity, graph, memory_path, today, report, config)
        return

    folder = config.get_folder_for_type(raw_entity.type)
    avg_importance = (
        sum(o.importance for o in raw_entity.observations) / len(raw_entity.observations)
        if raw_entity.observations else 0.3
    )

    retention = _initial_retention(raw_entity.type)

    fm = EntityFrontmatter(
        title=raw_entity.name,
        type=raw_entity.type,
        retention=retention,
        score=0.0,
        importance=avg_importance,
        frequency=1,
        last_mentioned=today,
        created=today,
        aliases=[],
        tags=list({tag for obs in raw_entity.observations for tag in obs.tags}),
        mention_dates=[today],
    )

    observations = [
        {"category": obs.category, "content": obs.content,
         "date": obs.date, "valence": obs.valence, "tags": obs.tags}
        for obs in raw_entity.observations
    ]

    create_entity(memory_path, folder, slug, fm, observations=observations)

    # Add to graph
    graph_entity = GraphEntity(
        file=f"{folder}/{slug}.md",
        type=raw_entity.type,
        title=raw_entity.name,
        score=0.0,
        importance=avg_importance,
        frequency=1,
        last_mentioned=today,
        created=today,
        mention_dates=[today],
        retention=retention,
        aliases=[],
        tags=fm.tags,
    )
    add_entity(graph, slug, graph_entity)

    report.entities_created.append(slug)


def _find_entity_slug(name: str, graph) -> str | None:
    """Find entity slug by name or alias. Delegates to graph.find_entity_by_name."""
    return find_entity_by_name(name, graph)
