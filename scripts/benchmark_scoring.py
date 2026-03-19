#!/usr/bin/env python3
"""Benchmark ACT-R scoring parameters by measuring Context Hit Rate.

Measures how well the scoring system predicts which entities will be mentioned
in conversations. Computes what percentage of entities referenced in each chat
were in the top-50 scored entities at the time.

Usage:
    uv run python scripts/benchmark_scoring.py              # Current hit rate
    uv run python scripts/benchmark_scoring.py --grid-search  # Test parameter combos
    uv run python scripts/benchmark_scoring.py --top-n 30     # Use top-30 instead of top-50
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import product
from pathlib import Path

# Add project root to sys.path so src.* imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import Config, ScoringConfig, load_config
from src.core.utils import parse_frontmatter
from src.memory.graph import load_graph
from src.memory.scoring import recalculate_all_scores


# ── Data structures ──────────────────────────────────────────


@dataclass
class ChatRecord:
    """A processed chat with its referenced entities and date."""
    path: str
    chat_date: str  # ISO date string
    entities: set[str]  # entity slugs (created + updated)


@dataclass
class BenchmarkResult:
    """Result for a single parameter configuration."""
    decay_factor: float
    importance_weight: float
    spreading_weight: float
    emotional_boost_weight: float
    hit_rate: float  # average across all chats
    hit_count: int  # total hits
    total_referenced: int  # total entity references across chats
    chats_evaluated: int


# ── Chat loading ─────────────────────────────────────────────


def load_processed_chats(memory_path: Path) -> list[ChatRecord]:
    """Load all processed chats and extract referenced entity slugs."""
    chats_dir = memory_path / "chats"
    if not chats_dir.exists():
        return []

    records: list[ChatRecord] = []
    for md_file in sorted(chats_dir.glob("*.md")):
        try:
            text = md_file.read_text(encoding="utf-8")
            fm, _ = parse_frontmatter(text)
        except Exception:
            continue

        if not fm.get("processed", False):
            continue

        # Collect entity slugs from both created and updated
        entities_created = fm.get("entities_created", []) or []
        entities_updated = fm.get("entities_updated", []) or []
        entity_slugs = set()
        for slug in entities_created + entities_updated:
            if isinstance(slug, str) and slug.strip():
                entity_slugs.add(slug.strip())

        if not entity_slugs:
            continue  # skip chats that didn't touch any entities

        chat_date = str(fm.get("date", fm.get("processed_at", "")))

        records.append(ChatRecord(
            path=str(md_file.relative_to(memory_path)),
            chat_date=chat_date,
            entities=entity_slugs,
        ))

    return records


# ── Scoring evaluation ───────────────────────────────────────


def evaluate_hit_rate(
    graph_original,
    config: Config,
    chats: list[ChatRecord],
    top_n: int = 50,
    *,
    scoring_override: ScoringConfig | None = None,
) -> BenchmarkResult:
    """Evaluate hit rate for a scoring configuration.

    For each chat, checks what fraction of its referenced entities
    appear in the top-N scored entities. Uses deepcopy to avoid
    mutating the original graph.
    """
    # Apply scoring override if provided
    cfg = copy.deepcopy(config)
    if scoring_override is not None:
        cfg.scoring = scoring_override

    # Ensure no activation noise during benchmarking (deterministic)
    cfg.scoring.activation_noise = 0.0

    # Score entities using the full graph state
    graph = copy.deepcopy(graph_original)
    recalculate_all_scores(graph, cfg, date.today())

    # Get top-N entity IDs by score
    scored_entities = sorted(
        graph.entities.items(),
        key=lambda x: x[1].score,
        reverse=True,
    )
    top_entity_ids = {eid for eid, _ in scored_entities[:top_n]}

    # Also include permanent entities (they're always in context)
    for eid, entity in graph.entities.items():
        if entity.retention == "permanent":
            top_entity_ids.add(eid)

    # Calculate hit rate per chat
    total_hits = 0
    total_referenced = 0
    per_chat_rates: list[float] = []

    for chat in chats:
        # Only count entities that exist in the graph
        referenced = chat.entities & set(graph.entities.keys())
        if not referenced:
            continue

        hits = referenced & top_entity_ids
        rate = len(hits) / len(referenced) if referenced else 0.0

        total_hits += len(hits)
        total_referenced += len(referenced)
        per_chat_rates.append(rate)

    avg_hit_rate = sum(per_chat_rates) / len(per_chat_rates) if per_chat_rates else 0.0

    sc = cfg.scoring
    return BenchmarkResult(
        decay_factor=sc.decay_factor,
        importance_weight=sc.importance_weight,
        spreading_weight=sc.spreading_weight,
        emotional_boost_weight=sc.emotional_boost_weight,
        hit_rate=round(avg_hit_rate, 4),
        hit_count=total_hits,
        total_referenced=total_referenced,
        chats_evaluated=len(per_chat_rates),
    )


# ── Grid search ──────────────────────────────────────────────


GRID_PARAMS = {
    "decay_factor": [0.3, 0.4, 0.5, 0.6, 0.7],
    "importance_weight": [0.1, 0.2, 0.3, 0.4, 0.5],
    "spreading_weight": [0.1, 0.2, 0.3, 0.4],
    "emotional_boost_weight": [0.0, 0.1, 0.15, 0.2, 0.3],
}


def run_grid_search(
    graph_original,
    config: Config,
    chats: list[ChatRecord],
    top_n: int = 50,
) -> list[BenchmarkResult]:
    """Test all parameter combinations and return sorted results."""
    combos = list(product(
        GRID_PARAMS["decay_factor"],
        GRID_PARAMS["importance_weight"],
        GRID_PARAMS["spreading_weight"],
        GRID_PARAMS["emotional_boost_weight"],
    ))
    total = len(combos)
    print(f"Grid search: {total} combinations to evaluate...")

    results: list[BenchmarkResult] = []
    start = time.time()

    for i, (df, iw, sw, ebw) in enumerate(combos, 1):
        if i % 50 == 0 or i == 1:
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(f"  [{i}/{total}] elapsed={elapsed:.0f}s eta={eta:.0f}s")

        sc = copy.deepcopy(config.scoring)
        sc.decay_factor = df
        sc.importance_weight = iw
        sc.spreading_weight = sw
        sc.emotional_boost_weight = ebw

        result = evaluate_hit_rate(graph_original, config, chats, top_n, scoring_override=sc)
        results.append(result)

    results.sort(key=lambda r: r.hit_rate, reverse=True)
    return results


# ── Output ───────────────────────────────────────────────────


def print_result(result: BenchmarkResult, label: str = "") -> None:
    """Print a single benchmark result."""
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Hit rate: {result.hit_rate:.1%}  "
          f"({result.hit_count}/{result.total_referenced} hits, "
          f"{result.chats_evaluated} chats)")
    print(f"  decay_factor={result.decay_factor}  "
          f"importance_weight={result.importance_weight}  "
          f"spreading_weight={result.spreading_weight}  "
          f"emotional_boost_weight={result.emotional_boost_weight}")


def save_results(results: list[BenchmarkResult], output_path: Path, top_n: int) -> None:
    """Save results to JSON file."""
    data = {
        "generated": date.today().isoformat(),
        "top_n": top_n,
        "total_combinations": len(results),
        "results": [
            {
                "rank": i + 1,
                "hit_rate": r.hit_rate,
                "hit_count": r.hit_count,
                "total_referenced": r.total_referenced,
                "chats_evaluated": r.chats_evaluated,
                "params": {
                    "decay_factor": r.decay_factor,
                    "importance_weight": r.importance_weight,
                    "spreading_weight": r.spreading_weight,
                    "emotional_boost_weight": r.emotional_boost_weight,
                },
            }
            for i, r in enumerate(results)
        ],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ACT-R scoring parameters via Context Hit Rate."
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Test all parameter combinations and rank by hit rate.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top-scored entities to use as the 'context' (default: 50).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: auto-detect).",
    )
    args = parser.parse_args()

    # Load config
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(args.config, project_root=project_root)

    print(f"Memory path: {config.memory_path}")
    print(f"Top-N: {args.top_n}")

    # Load graph
    graph = load_graph(config.memory_path)
    entity_count = len(graph.entities)
    relation_count = len(graph.relations)
    print(f"Graph: {entity_count} entities, {relation_count} relations")

    if entity_count == 0:
        print("No entities in graph. Nothing to benchmark.")
        return

    # Load processed chats
    chats = load_processed_chats(config.memory_path)
    print(f"Processed chats with entity references: {len(chats)}")

    if not chats:
        print("No processed chats with entity references found. Nothing to benchmark.")
        return

    # Count unique referenced entities
    all_referenced = set()
    for c in chats:
        all_referenced.update(c.entities)
    in_graph = all_referenced & set(graph.entities.keys())
    print(f"Unique entities referenced in chats: {len(all_referenced)} ({len(in_graph)} in graph)")
    print()

    if args.grid_search:
        # Grid search mode
        results = run_grid_search(graph, config, chats, args.top_n)

        print(f"\n{'='*70}")
        print(f"TOP 10 CONFIGURATIONS (out of {len(results)})")
        print(f"{'='*70}")
        for i, r in enumerate(results[:10], 1):
            print()
            print_result(r, label=f"#{i}")

        # Also show current config for comparison
        print(f"\n{'='*70}")
        print("CURRENT CONFIGURATION")
        print(f"{'='*70}")
        current = evaluate_hit_rate(graph, config, chats, args.top_n)
        print()
        print_result(current, label="current")

        # Find rank of current config
        for i, r in enumerate(results):
            if (r.decay_factor == current.decay_factor
                    and r.importance_weight == current.importance_weight
                    and r.spreading_weight == current.spreading_weight
                    and r.emotional_boost_weight == current.emotional_boost_weight):
                print(f"  Current config rank: #{i+1} of {len(results)}")
                break

        # Save all results
        output_path = config.memory_path / "_benchmark_results.json"
        save_results(results, output_path, args.top_n)

    else:
        # Single evaluation mode
        result = evaluate_hit_rate(graph, config, chats, args.top_n)
        print_result(result, label="current")

        # Show some detail: which entities were missed most often
        graph_copy = copy.deepcopy(graph)
        recalculate_all_scores(graph_copy, config, date.today())
        scored = sorted(
            graph_copy.entities.items(),
            key=lambda x: x[1].score,
            reverse=True,
        )
        top_ids = {eid for eid, _ in scored[:args.top_n]}
        for eid, entity in graph_copy.entities.items():
            if entity.retention == "permanent":
                top_ids.add(eid)

        # Count misses per entity
        miss_counts: dict[str, int] = {}
        for chat in chats:
            referenced = chat.entities & set(graph_copy.entities.keys())
            missed = referenced - top_ids
            for eid in missed:
                miss_counts[eid] = miss_counts.get(eid, 0) + 1

        if miss_counts:
            print(f"\nMost frequently missed entities (not in top-{args.top_n}):")
            sorted_misses = sorted(miss_counts.items(), key=lambda x: x[1], reverse=True)
            for eid, count in sorted_misses[:10]:
                entity = graph_copy.entities.get(eid)
                score = entity.score if entity else 0.0
                title = entity.title if entity else eid
                print(f"  {title} ({eid}): missed {count}x, score={score:.4f}")

        # Save single result
        output_path = config.memory_path / "_benchmark_results.json"
        save_results([result], output_path, args.top_n)


if __name__ == "__main__":
    main()
