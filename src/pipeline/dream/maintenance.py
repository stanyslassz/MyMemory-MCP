"""Dream steps: load, prune, rescore, and rebuild."""

from __future__ import annotations

import logging
import shutil
from datetime import date
from pathlib import Path

from rich.console import Console

from src.core.config import Config
from src.core.models import GraphData

logger = logging.getLogger(__name__)


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


def _step_prune_dead(
    graph: GraphData,
    memory_path: Path,
    config: Config,
    console: Console,
    report,
    dry_run: bool,
    score_threshold: float | None = None,
    max_frequency: int | None = None,
    min_age_days: int | None = None,
) -> None:
    """Step 7: Archive low-score orphan entities."""
    score_threshold = score_threshold if score_threshold is not None else config.dream.prune_score_threshold
    max_frequency = max_frequency if max_frequency is not None else config.dream.prune_max_frequency
    min_age_days = min_age_days if min_age_days is not None else config.dream.prune_min_age_days
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

        logger.info(
            "Pruning entity '%s' (id=%s, score=%.3f, freq=%d, retention=%s, created=%s)",
            entity.title, eid, entity.score, entity.frequency,
            entity.retention, entity.created,
        )
        console.print(f"  [yellow]Archiving: {entity.title}[/yellow]")
        try:
            if entity_path.exists():
                archive_dir.mkdir(exist_ok=True)
                shutil.move(str(entity_path), str(archive_dir / entity_path.name))
            del graph.entities[eid]
            report.entities_pruned += 1
        except (FileNotFoundError, PermissionError, OSError, KeyError) as e:
            report.errors.append(f"Prune failed for {eid}: {e}")
            logger.warning("Prune failed for %s: %s", eid, e)

    if not dry_run:
        from src.memory.graph import remove_orphan_relations, save_graph
        remove_orphan_relations(graph)
        save_graph(memory_path, graph)
        # Mark FAISS as needing rebuild (handled by step 10 in full pipeline,
        # but needed here for standalone --step 7 runs)
        try:
            from src.pipeline.indexer import build_index
            build_index(memory_path, config)
        except Exception as e:
            console.print(f"  [yellow]FAISS rebuild after pruning: {e}[/yellow]")


def _step_rebuild(
    graph: GraphData,
    memory_path: Path,
    config: Config,
    console: Console,
) -> None:
    """Step 10: Rebuild context and FAISS index."""
    from src.memory.context import build_context_for_config, write_context, write_index
    from src.memory.graph import save_graph
    from src.pipeline.indexer import build_index

    save_graph(memory_path, graph)

    context_text = build_context_for_config(graph, memory_path, config, use_llm=False)
    if context_text.strip():
        write_context(memory_path, context_text)
        console.print("  [green]_context.md updated[/green]")

    write_index(memory_path, graph)
    console.print("  [green]_index.md updated[/green]")

    try:
        manifest = build_index(memory_path, config)
        n_files = len(manifest.get("indexed_files", {}))
        console.print(f"  [green]FAISS rebuilt: {n_files} files indexed[/green]")
    except Exception as e:
        console.print(f"  [yellow]FAISS rebuild warning: {e}[/yellow]")
