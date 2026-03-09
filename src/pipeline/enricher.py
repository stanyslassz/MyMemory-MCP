"""Step 4: Apply resolved extractions to memory (MD files + graph)."""

from __future__ import annotations

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
from src.memory.graph import add_entity, add_relation, load_graph, save_graph
from src.memory.scoring import recalculate_all_scores
from src.memory.store import create_entity, create_stub_entity, update_entity, mark_observation_superseded, read_entity, write_entity, consolidate_entity_facts
from src.pipeline.resolver import slugify


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
                        retention="short_term",
                    ))

            if from_slug and to_slug:
                graph_rel = GraphRelation(from_entity=from_slug, to_entity=to_slug, type=rel.type, context=rel.context)
                graph = add_relation(graph, graph_rel, strength_growth=config.scoring.relation_strength_growth)
                report.relations_added += 1

                # Also add relation text to the source entity MD
                if from_slug in graph.entities:
                    entity_file = memory_path / graph.entities[from_slug].file
                    if entity_file.exists():
                        rel_line = f"- {rel.type} [[{rel.to_name}]]"
                        update_entity(entity_file, new_relations=[rel_line])
        except Exception as e:
            report.errors.append(f"Error processing relation {rel.from_name} → {rel.to_name}: {e}")

    # Recalculate scores
    graph = recalculate_all_scores(graph, config, date.fromisoformat(today))

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
    import logging
    _logger = logging.getLogger(__name__)

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
        live_facts = [f for f in sections.get("Facts", []) if "[superseded]" not in f]
        if len(live_facts) + len(raw_entity.observations) > max_facts:
            _logger.info(
                "Pre-consolidating %s (%d + %d > %d facts)",
                entity_meta.title, len(live_facts), len(raw_entity.observations), max_facts,
            )
            try:
                consolidate_entity_facts(filepath, config, max_facts=max_facts)
            except Exception as e:
                _logger.warning("Pre-consolidation failed for %s: %s", entity_id, e)

    # Prepare observations
    new_obs = [
        {"category": obs.category, "content": obs.content, "tags": obs.tags,
         "date": obs.date, "valence": obs.valence}
        for obs in raw_entity.observations
    ]

    # Update MD file
    update_entity(filepath, new_observations=new_obs, last_mentioned=today, max_facts=max_facts)

    # Update graph metadata
    entity_meta.frequency += 1
    entity_meta.last_mentioned = today

    # Update mention_dates (windowed)
    from src.memory.mentions import add_mention
    entity_meta.mention_dates, entity_meta.monthly_buckets = add_mention(
        today, entity_meta.mention_dates, entity_meta.monthly_buckets,
        window_size=50,
    )

    # Update importance (running average)
    if raw_entity.observations:
        new_importance = sum(o.importance for o in raw_entity.observations) / len(raw_entity.observations)
        entity_meta.importance = (entity_meta.importance + new_importance) / 2

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
        _update_existing_entity(slug, raw_entity, graph, memory_path, today, report)
        return

    folder = config.get_folder_for_type(raw_entity.type)
    avg_importance = (
        sum(o.importance for o in raw_entity.observations) / len(raw_entity.observations)
        if raw_entity.observations else 0.3
    )

    fm = EntityFrontmatter(
        title=raw_entity.name,
        type=raw_entity.type,
        retention="short_term",
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
         "date": obs.date, "valence": obs.valence}
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
        retention="short_term",
        aliases=[],
        tags=fm.tags,
    )
    add_entity(graph, slug, graph_entity)

    report.entities_created.append(slug)


def _find_entity_slug(name: str, graph) -> str | None:
    """Find entity slug by name or alias."""
    slug = slugify(name)
    if slug in graph.entities:
        return slug

    name_lower = name.lower()
    for entity_id, meta in graph.entities.items():
        if meta.title.lower() == name_lower:
            return entity_id
        for alias in meta.aliases:
            if alias.lower() == name_lower:
                return entity_id

    return None
