#!/usr/bin/env python3
"""Backfill mention_dates and monthly_buckets from _graph.json into MD frontmatter.

Pre-existing entities created before WS1 lack temporal data in their MD files.
After rebuild-graph, these entities get mention_dates=[] which collapses ACT-R
scores to ~0. This script copies the temporal data from _graph.json back into
the MD frontmatter so that rebuild-graph preserves correct scores.

Usage:
    uv run python scripts/migrate_temporal_to_md.py           # Apply changes
    uv run python scripts/migrate_temporal_to_md.py --dry-run # Preview only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.memory.graph import load_graph
from src.memory.store import read_entity, write_entity


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill temporal data from graph into MD files.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    args = parser.parse_args()

    config = load_config(config_path=args.config, project_root=PROJECT_ROOT)
    memory_path = config.memory_path

    graph = load_graph(memory_path)
    if graph is None:
        print("ERROR: Could not load _graph.json (or .bak). Aborting.")
        sys.exit(1)

    updated = 0
    already_ok = 0
    skipped = 0

    for entity_id, g_entity in graph.entities.items():
        entity_path = (memory_path / g_entity.file).resolve()
        if not entity_path.exists():
            skipped += 1
            continue

        try:
            frontmatter, sections = read_entity(entity_path)
        except Exception as e:
            print(f"  SKIP {entity_id}: read error: {e}")
            skipped += 1
            continue

        changed = False

        # Backfill mention_dates
        fm_dates = frontmatter.mention_dates or []
        graph_dates = g_entity.mention_dates or []
        if not fm_dates and graph_dates:
            frontmatter.mention_dates = graph_dates
            changed = True

        # Backfill monthly_buckets
        fm_buckets = frontmatter.monthly_buckets or {}
        graph_buckets = g_entity.monthly_buckets or {}
        if not fm_buckets and graph_buckets:
            frontmatter.monthly_buckets = graph_buckets
            changed = True

        if changed:
            if args.dry_run:
                dates_info = f"mention_dates={len(graph_dates)}" if not fm_dates and graph_dates else ""
                buckets_info = f"monthly_buckets={len(graph_buckets)}" if not fm_buckets and graph_buckets else ""
                info = ", ".join(filter(None, [dates_info, buckets_info]))
                print(f"  WOULD UPDATE {entity_id}: {info}")
            else:
                write_entity(entity_path, frontmatter, sections)
            updated += 1
        else:
            already_ok += 1

    label = "Would update" if args.dry_run else "Updated"
    print(f"\n{label}: {updated} | Already OK: {already_ok} | Skipped (missing): {skipped}")


if __name__ == "__main__":
    main()
