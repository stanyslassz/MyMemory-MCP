"""Dream mode: brain-like memory reorganization during idle time.

8-step pipeline: load → consolidate facts → merge entities → discover relations
→ prune dead → generate summaries → rescore → rebuild context + FAISS.

No new information enters — only reorganization of existing knowledge.
"""

from __future__ import annotations

import logging
import shutil
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import get_args

from rich.console import Console

from src.core.config import Config
from src.core.models import GraphData, GraphRelation, RelationType

logger = logging.getLogger(__name__)

_VALID_RELATION_TYPES: set[str] = set(get_args(RelationType))


class DreamReport:
    """Collects stats from each dream step."""

    def __init__(self):
        self.facts_consolidated: int = 0
        self.entities_merged: int = 0
        self.relations_discovered: int = 0
        self.entities_pruned: int = 0
        self.summaries_generated: int = 0
        self.errors: list[str] = []


def run_dream(
    config: Config,
    console: Console,
    *,
    dry_run: bool = False,
    step: int | None = None,
) -> DreamReport:
    """Run the full dream pipeline (or a single step).

    Args:
        config: Loaded configuration.
        console: Rich console for output.
        dry_run: If True, show what would change without modifying.
        step: If set, run only this step number (1-8).
    """
    report = DreamReport()
    memory_path = config.memory_path

    # Step 1: Load
    if step is None or step == 1:
        console.print("\n[bold]Step 1: Loading graph and entities...[/bold]")
        graph, entity_paths = _step_load(memory_path)
        console.print(f"  Loaded {len(graph.entities)} entities, {len(graph.relations)} relations")
    else:
        graph, entity_paths = _step_load(memory_path)

    # Step 2: Fact consolidation
    if step is None or step == 2:
        console.print("\n[bold]Step 2: Consolidating facts...[/bold]")
        _step_consolidate_facts(graph, entity_paths, config, console, report, dry_run)

    # Step 3: Entity merging
    if step is None or step == 3:
        console.print("\n[bold]Step 3: Merging duplicate entities...[/bold]")
        _step_merge_entities(graph, memory_path, config, console, report, dry_run)

    # Step 4: Relation discovery
    if step is None or step == 4:
        console.print("\n[bold]Step 4: Discovering new relations...[/bold]")
        _step_discover_relations(graph, memory_path, config, console, report, dry_run)

    # Step 5: Dead entity pruning
    if step is None or step == 5:
        console.print("\n[bold]Step 5: Pruning dead entities...[/bold]")
        _step_prune_dead(graph, memory_path, config, console, report, dry_run)

    # Step 6: Summary generation
    if step is None or step == 6:
        console.print("\n[bold]Step 6: Generating summaries...[/bold]")
        _step_generate_summaries(graph, entity_paths, config, console, report, dry_run)

    # Step 7: Rescore
    if step is None or step == 7:
        console.print("\n[bold]Step 7: Rescoring all entities...[/bold]")
        if not dry_run:
            from src.memory.scoring import recalculate_all_scores
            graph = recalculate_all_scores(graph, config)
            console.print("  Scores recalculated")
        else:
            console.print("  [dim]Would recalculate all scores[/dim]")

    # Step 8: Rebuild context + FAISS
    if step is None or step == 8:
        console.print("\n[bold]Step 8: Rebuilding context and FAISS...[/bold]")
        if not dry_run:
            _step_rebuild(graph, memory_path, config, console)
        else:
            console.print("  [dim]Would rebuild _context.md and FAISS index[/dim]")

    return report


# ── Step implementations ─────────────────────────────────────


def _step_load(memory_path: Path) -> tuple[GraphData, dict[str, Path]]:
    """Step 1: Load graph and map entity IDs to file paths."""
    from src.memory.graph import load_graph

    graph = load_graph(memory_path)
    entity_paths = {}
    for eid, entity in graph.entities.items():
        path = memory_path / entity.file
        if path.exists():
            entity_paths[eid] = path
    return graph, entity_paths


def _step_consolidate_facts(
    graph: GraphData,
    entity_paths: dict[str, Path],
    config: Config,
    console: Console,
    report: DreamReport,
    dry_run: bool,
    min_facts: int = 8,
) -> None:
    """Step 2: Consolidate redundant observations for entities with many facts."""
    from src.memory.store import read_entity, consolidate_entity_facts

    for eid, path in entity_paths.items():
        entity = graph.entities.get(eid)
        if not entity:
            continue
        try:
            _, sections = read_entity(path)
            facts = sections.get("Facts", [])
            live_facts = [f for f in facts if "[superseded]" not in f]
            if len(live_facts) < min_facts:
                continue

            if dry_run:
                console.print(f"  [dim]Would consolidate {entity.title} ({len(live_facts)} facts)[/dim]")
                report.facts_consolidated += 1
                continue

            console.print(f"  [cyan]Consolidating {entity.title} ({len(live_facts)} facts)...[/cyan]")
            result = consolidate_entity_facts(path, config)
            if result["changes"]:
                console.print(f"    [green]{', '.join(result['changes'])}[/green]")
                report.facts_consolidated += 1
        except Exception as e:
            report.errors.append(f"Fact consolidation failed for {eid}: {e}")
            console.print(f"    [yellow]Skipped {eid}: {e}[/yellow]")


def _step_merge_entities(
    graph: GraphData,
    memory_path: Path,
    config: Config,
    console: Console,
    report: DreamReport,
    dry_run: bool,
) -> None:
    """Step 3: Detect and merge duplicate entities (slug similarity + FAISS)."""
    from src.core.utils import slugify

    # Group by slug similarity (prefix match or containment)
    slugs = list(graph.entities.keys())
    merge_candidates: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for i, slug_a in enumerate(slugs):
        entity_a = graph.entities[slug_a]
        # Check aliases overlap
        aliases_a = {a.lower() for a in entity_a.aliases} | {entity_a.title.lower()}

        for slug_b in slugs[i + 1:]:
            entity_b = graph.entities[slug_b]
            if entity_a.type != entity_b.type:
                continue

            aliases_b = {a.lower() for a in entity_b.aliases} | {entity_b.title.lower()}

            # Check overlap: title of one matches alias/title of other
            if aliases_a & aliases_b:
                pair = tuple(sorted([slug_a, slug_b]))
                if pair not in seen_pairs:
                    merge_candidates.append((slug_a, slug_b))
                    seen_pairs.add(pair)

    if not merge_candidates:
        console.print("  No duplicate entities detected")
        return

    for slug_a, slug_b in merge_candidates:
        entity_a = graph.entities.get(slug_a)
        entity_b = graph.entities.get(slug_b)
        if not entity_a or not entity_b:
            continue

        # Keep the one with higher score
        keep, drop = (slug_a, slug_b) if entity_a.score >= entity_b.score else (slug_b, slug_a)
        keep_entity = graph.entities[keep]
        drop_entity = graph.entities[drop]

        if dry_run:
            console.print(f"  [dim]Would merge '{drop_entity.title}' into '{keep_entity.title}'[/dim]")
            report.entities_merged += 1
            continue

        console.print(f"  [cyan]Merging '{drop_entity.title}' → '{keep_entity.title}'[/cyan]")
        try:
            _do_merge(keep, drop, graph, memory_path, config)
            report.entities_merged += 1
        except Exception as e:
            report.errors.append(f"Merge failed {drop} → {keep}: {e}")
            console.print(f"    [yellow]Failed: {e}[/yellow]")


def _do_merge(
    keep_id: str,
    drop_id: str,
    graph: GraphData,
    memory_path: Path,
    config: Config,
) -> None:
    """Merge drop entity into keep entity: combine facts, aliases, relations."""
    from src.memory.store import read_entity, write_entity

    keep_entity = graph.entities[keep_id]
    drop_entity = graph.entities[drop_id]

    keep_path = memory_path / keep_entity.file
    drop_path = memory_path / drop_entity.file

    if not keep_path.exists() or not drop_path.exists():
        return

    keep_fm, keep_sections = read_entity(keep_path)
    drop_fm, drop_sections = read_entity(drop_path)

    # Merge aliases
    all_aliases = set(keep_fm.aliases) | set(drop_fm.aliases) | {drop_entity.title}
    all_aliases.discard(keep_entity.title)
    keep_fm.aliases = sorted(all_aliases)

    # Merge facts (dedup by content)
    keep_facts = set(keep_sections.get("Facts", []))
    for fact in drop_sections.get("Facts", []):
        if fact not in keep_facts:
            keep_sections.setdefault("Facts", []).append(fact)

    # Merge tags
    keep_fm.tags = sorted(set(keep_fm.tags) | set(drop_fm.tags))

    # Keep higher importance and frequency sum
    keep_fm.importance = max(keep_fm.importance, drop_fm.importance)
    keep_fm.frequency += drop_fm.frequency

    # Merge mention dates
    keep_fm.mention_dates = sorted(set(keep_fm.mention_dates) | set(drop_fm.mention_dates))

    # Add history entry
    today = date.today().isoformat()
    keep_sections.setdefault("History", []).append(
        f"- {today}: Merged with '{drop_entity.title}' (dream mode)"
    )

    write_entity(keep_path, keep_fm, keep_sections)

    # Update graph: retarget relations from drop to keep
    for rel in graph.relations:
        if rel.from_entity == drop_id:
            rel.from_entity = keep_id
        if rel.to_entity == drop_id:
            rel.to_entity = keep_id

    # Remove drop entity from graph
    del graph.entities[drop_id]

    # Remove self-referencing relations
    graph.relations = [
        r for r in graph.relations
        if r.from_entity != r.to_entity
    ]

    # Move drop file to archive
    archive_dir = memory_path / "_archive"
    archive_dir.mkdir(exist_ok=True)
    shutil.move(str(drop_path), str(archive_dir / drop_path.name))

    # Update keep entity in graph
    keep_entity.aliases = keep_fm.aliases
    keep_entity.tags = keep_fm.tags
    keep_entity.importance = keep_fm.importance
    keep_entity.frequency = keep_fm.frequency


def _step_discover_relations(
    graph: GraphData,
    memory_path: Path,
    config: Config,
    console: Console,
    report: DreamReport,
    dry_run: bool,
) -> None:
    """Step 4: Use FAISS similarity + LLM to discover new relations."""
    from src.pipeline.indexer import search as faiss_search
    from src.memory.store import read_entity
    from src.memory.graph import add_relation, save_graph

    # Build existing relation set for fast lookup
    existing_rels: set[tuple[str, str]] = set()
    for rel in graph.relations:
        existing_rels.add((rel.from_entity, rel.to_entity))
        existing_rels.add((rel.to_entity, rel.from_entity))

    candidates: list[tuple[str, str]] = []
    entity_ids = list(graph.entities.keys())

    for eid in entity_ids:
        entity = graph.entities[eid]
        try:
            results = faiss_search(entity.title, config, memory_path, top_k=5)
        except Exception:
            continue

        for result in results:
            other_id = result.entity_id
            if other_id == eid:
                continue
            if other_id not in graph.entities:
                continue
            pair = tuple(sorted([eid, other_id]))
            if pair in existing_rels:
                continue
            if pair not in {tuple(sorted(c)) for c in candidates}:
                candidates.append((eid, other_id))

    if not candidates:
        console.print("  No new relation candidates found")
        return

    console.print(f"  Found {len(candidates)} candidate pair(s) to evaluate")

    from src.core.llm import call_relation_discovery

    discovered = 0
    for eid_a, eid_b in candidates:
        entity_a = graph.entities.get(eid_a)
        entity_b = graph.entities.get(eid_b)
        if not entity_a or not entity_b:
            continue

        # Build dossiers
        dossier_a = _build_dossier(eid_a, entity_a, memory_path)
        dossier_b = _build_dossier(eid_b, entity_b, memory_path)

        if dry_run:
            console.print(f"  [dim]Would evaluate: {entity_a.title} ↔ {entity_b.title}[/dim]")
            continue

        try:
            proposal = call_relation_discovery(
                entity_a.title, entity_a.type, dossier_a,
                entity_b.title, entity_b.type, dossier_b,
                config,
            )

            if proposal.action == "relate" and proposal.relation_type:
                rel_type = proposal.relation_type
                if rel_type not in _VALID_RELATION_TYPES:
                    logger.warning("Dream discovered invalid relation type '%s', skipping", rel_type)
                    continue

                new_rel = GraphRelation(
                    from_entity=eid_a,
                    to_entity=eid_b,
                    type=rel_type,
                    context=proposal.context,
                )
                add_relation(graph, new_rel, strength_growth=config.scoring.relation_strength_growth)
                discovered += 1
                console.print(f"    [green]{entity_a.title} → {rel_type} → {entity_b.title}[/green]")
        except Exception as e:
            report.errors.append(f"Relation discovery failed for {eid_a}/{eid_b}: {e}")

    report.relations_discovered = discovered
    if discovered and not dry_run:
        save_graph(memory_path, graph)
        console.print(f"  [green]Discovered {discovered} new relation(s)[/green]")


def _build_dossier(eid: str, entity, memory_path: Path) -> str:
    """Build a compact dossier string for an entity."""
    from src.memory.store import read_entity

    path = memory_path / entity.file
    facts_text = ""
    if path.exists():
        try:
            _, sections = read_entity(path)
            facts = sections.get("Facts", [])
            live_facts = [f for f in facts if "[superseded]" not in f]
            facts_text = "\n".join(live_facts[:10])
        except Exception:
            pass

    lines = [f"Title: {entity.title}", f"Type: {entity.type}"]
    if entity.tags:
        lines.append(f"Tags: {', '.join(entity.tags)}")
    if facts_text:
        lines.append(f"Facts:\n{facts_text}")
    if entity.summary:
        lines.append(f"Summary: {entity.summary}")
    return "\n".join(lines)


def _step_prune_dead(
    graph: GraphData,
    memory_path: Path,
    config: Config,
    console: Console,
    report: DreamReport,
    dry_run: bool,
    score_threshold: float = 0.1,
    max_frequency: int = 1,
    min_age_days: int = 90,
) -> None:
    """Step 5: Archive low-score orphan entities."""
    today = date.today()

    # Build set of entities that have relations
    related_entities: set[str] = set()
    for rel in graph.relations:
        related_entities.add(rel.from_entity)
        related_entities.add(rel.to_entity)

    prune_candidates = []
    for eid, entity in graph.entities.items():
        if entity.score >= score_threshold:
            continue
        if entity.frequency > max_frequency:
            continue
        if entity.retention == "permanent":
            continue
        if eid in related_entities:
            continue

        # Check age
        if entity.created:
            try:
                created_date = date.fromisoformat(entity.created)
                age_days = (today - created_date).days
                if age_days < min_age_days:
                    continue
            except (ValueError, TypeError):
                pass

        prune_candidates.append(eid)

    if not prune_candidates:
        console.print("  No entities to prune")
        return

    archive_dir = memory_path / "_archive"

    for eid in prune_candidates:
        entity = graph.entities[eid]
        entity_path = memory_path / entity.file

        if dry_run:
            console.print(f"  [dim]Would archive: {entity.title} (score={entity.score:.3f}, freq={entity.frequency})[/dim]")
            report.entities_pruned += 1
            continue

        console.print(f"  [yellow]Archiving: {entity.title}[/yellow]")
        try:
            if entity_path.exists():
                archive_dir.mkdir(exist_ok=True)
                shutil.move(str(entity_path), str(archive_dir / entity_path.name))
            del graph.entities[eid]
            report.entities_pruned += 1
        except Exception as e:
            report.errors.append(f"Prune failed for {eid}: {e}")

    if not dry_run:
        # Clean up orphan relations
        from src.memory.graph import remove_orphan_relations, save_graph
        remove_orphan_relations(graph)
        save_graph(memory_path, graph)


def _step_generate_summaries(
    graph: GraphData,
    entity_paths: dict[str, Path],
    config: Config,
    console: Console,
    report: DreamReport,
    dry_run: bool,
) -> None:
    """Step 6: Generate/refresh entity summaries via LLM."""
    from src.core.llm import call_entity_summary
    from src.memory.store import read_entity, write_entity

    for eid, path in entity_paths.items():
        entity = graph.entities.get(eid)
        if not entity:
            continue
        if eid not in graph.entities:
            continue  # May have been pruned

        # Skip if summary already exists and entity hasn't changed much
        if entity.summary:
            continue

        try:
            fm, sections = read_entity(path)
            facts = sections.get("Facts", [])
            live_facts = [f for f in facts if "[superseded]" not in f]
            if not live_facts:
                continue

            # Build relation descriptions
            relations = []
            for rel in graph.relations:
                if rel.from_entity == eid:
                    target = graph.entities.get(rel.to_entity)
                    if target:
                        relations.append(f"{rel.type} {target.title}")
                elif rel.to_entity == eid:
                    source = graph.entities.get(rel.from_entity)
                    if source:
                        relations.append(f"{rel.type} (from {source.title})")

            if dry_run:
                console.print(f"  [dim]Would generate summary for: {entity.title}[/dim]")
                report.summaries_generated += 1
                continue

            summary = call_entity_summary(
                entity.title, entity.type, live_facts, relations, entity.tags, config,
            )
            if summary:
                fm.summary = summary
                entity.summary = summary
                write_entity(path, fm, sections)
                report.summaries_generated += 1
                console.print(f"  [green]{entity.title}: {summary[:60]}...[/green]" if len(summary) > 60 else f"  [green]{entity.title}: {summary}[/green]")
        except Exception as e:
            report.errors.append(f"Summary generation failed for {eid}: {e}")
            console.print(f"    [yellow]Skipped {eid}: {e}[/yellow]")


def _step_rebuild(
    graph: GraphData,
    memory_path: Path,
    config: Config,
    console: Console,
) -> None:
    """Step 8: Rebuild context and FAISS index."""
    from src.memory.context import build_context, write_context, write_index
    from src.memory.graph import save_graph
    from src.pipeline.indexer import build_index

    # Save graph first
    save_graph(memory_path, graph)

    # Rebuild context
    context_text = build_context(graph, memory_path, config)
    if context_text.strip():
        write_context(memory_path, context_text)
        console.print("  [green]_context.md updated[/green]")

    # Rebuild index
    write_index(memory_path, graph)
    console.print("  [green]_index.md updated[/green]")

    # Rebuild FAISS
    try:
        manifest = build_index(memory_path, config)
        n_files = len(manifest.get("indexed_files", {}))
        console.print(f"  [green]FAISS rebuilt: {n_files} files indexed[/green]")
    except Exception as e:
        console.print(f"  [yellow]FAISS rebuild warning: {e}[/yellow]")
