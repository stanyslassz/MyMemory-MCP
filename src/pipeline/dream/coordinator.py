"""Dream coordinator: orchestration, checkpointing, validation, and report generation."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import get_args

from rich.console import Console

from src.core.config import Config
from src.core.utils import filter_live_facts
from src.core.models import GraphData, RelationType
from src.memory.event_log import append_event

logger = logging.getLogger(__name__)

_VALID_RELATION_TYPES: set[str] = set(get_args(RelationType))


def _save_checkpoint(memory_path: Path, dream_id: str, step: int, steps_planned: list[int]) -> None:
    """Write checkpoint after each successful dream step."""
    checkpoint = {
        "dream_id": dream_id,
        "last_completed_step": step,
        "steps_planned": steps_planned,
        "started_at": dream_id,
    }
    (memory_path / "_dream_checkpoint.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_checkpoint(memory_path: Path) -> dict | None:
    """Load existing checkpoint if any."""
    path = memory_path / "_dream_checkpoint.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _clear_checkpoint(memory_path: Path) -> None:
    """Remove checkpoint file after successful completion."""
    path = memory_path / "_dream_checkpoint.json"
    if path.exists():
        path.unlink()


def _generate_dream_report(memory_path: Path, dream_id: str, session_start_ts: str, *, health: dict | None = None) -> Path:
    """Generate a markdown report from dream session events."""
    from src.memory.event_log import read_events

    events = read_events(memory_path, source="dream", after=session_start_ts, limit=10_000)
    events = [e for e in events if e.get("data", {}).get("dream_id") == dream_id]

    session_completed = next(
        (e for e in events if e["type"] == "dream_session_completed"), None
    )
    duration_s = session_completed["data"].get("duration_s", 0) if session_completed else 0
    steps_failed = session_completed["data"].get("steps_failed", 0) if session_completed else 0
    status = "completed" if steps_failed == 0 else f"{steps_failed} failed"

    if duration_s >= 60:
        mins = int(duration_s // 60)
        secs = int(duration_s % 60)
        duration_str = f"{mins}m {secs}s"
    else:
        duration_str = f"{duration_s:.1f}s"

    date_str = dream_id[:10]
    lines = [
        f"# Dream Report — {date_str}",
        "",
        f"**Session**: {dream_id} | **Duration**: {duration_str} | **Status**: {status}",
        "",
        "## Steps",
        "",
        "| # | Step | Status | Duration | Summary |",
        "|---|------|--------|----------|---------|",
    ]

    step_completed = {e["data"]["step"]: e for e in events if e["type"] == "dream_step_completed"}
    step_failed = {e["data"]["step"]: e for e in events if e["type"] == "dream_step_failed"}
    step_skipped = {e["data"]["step"]: e for e in events if e["type"] == "dream_step_skipped"}

    details_sections = []

    for s in range(1, 11):
        if s in step_completed:
            d = step_completed[s]["data"]
            dur = f"{d['duration_s']:.1f}s" if d.get("duration_s") else "—"
            lines.append(f"| {s} | {d.get('step_name', '')} | done | {dur} | {d.get('summary', '')} |")
            if d.get("details"):
                detail_lines = [f"### {s}. {d.get('step_name', '')}"]
                for k, v in d["details"].items():
                    detail_lines.append(f"- {k.replace('_', ' ').title()}: {v}")
                details_sections.append("\n".join(detail_lines))
        elif s in step_failed:
            d = step_failed[s]["data"]
            dur = f"{d['duration_s']:.1f}s" if d.get("duration_s") else "—"
            lines.append(f"| {s} | {d.get('step_name', '')} | FAILED | {dur} | {d.get('error', '')} |")
        elif s in step_skipped:
            d = step_skipped[s]["data"]
            lines.append(f"| {s} | {d.get('step_name', '')} | skipped | — | — |")

    if details_sections:
        lines.append("")
        lines.append("## Details")
        lines.append("")
        lines.extend(details_sections)

    if health:
        lines.append("")
        lines.append("## Memory Health")
        lines.append("")
        lines.append(f"**Summary**: {health.get('summary', 'N/A')}")
        if health.get("hot_topics"):
            lines.append("")
            lines.append("**Hot topics** (3+ mentions in 7 days):")
            for h in health["hot_topics"][:10]:
                lines.append(f"- {h['title']} ({h['mentions_7d']} mentions)")
        if health.get("stale_topics"):
            lines.append("")
            lines.append(f"**Stale topics** ({len(health['stale_topics'])} entities, 60+ days):")
            for h in health["stale_topics"][:10]:
                lines.append(f"- {h['title']} ({h['days_since']}d)")
        if health.get("orphans"):
            lines.append("")
            lines.append(f"**Orphans** ({len(health['orphans'])} entities, no relations):")
            for h in health["orphans"][:10]:
                lines.append(f"- {h['title']}")
        if health.get("overloaded"):
            lines.append("")
            lines.append(f"**Overloaded** ({len(health['overloaded'])} entities):")
            for h in health["overloaded"][:10]:
                lines.append(f"- {h['title']} (freq {h['frequency']}/{h['max_facts']})")

    lines.append("")

    report_path = memory_path / "_dream_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def decide_dream_steps(stats: dict, health: dict | None = None) -> list[int]:
    """Deterministic dream step selection. Replaces LLM coordinator.

    If *health* (output of ``analyze_memory_health``) is provided, its
    findings can promote additional steps that the base stats alone would
    not have triggered.
    """
    steps = [1]  # Load always
    if stats.get("unextracted_docs", 0) > 0:
        steps.append(2)
    if stats.get("consolidation_candidates", 0) >= 3:
        steps.append(3)
    if stats.get("merge_candidates", 0) >= 2:
        steps.append(4)
    if stats.get("relation_candidates", 0) >= 5:
        steps.append(5)
    if stats.get("transitive_candidates", 0) >= 3:
        steps.append(6)
    if stats.get("prune_candidates", 0) >= 1:
        steps.append(7)
    if stats.get("summary_candidates", 0) >= 3:
        steps.append(8)

    # Health-based promotions
    if health:
        if len(health.get("overloaded", [])) > 3 and 3 not in steps:
            steps.append(3)
        if len(health.get("orphans", [])) > 5 and 5 not in steps:
            steps.append(5)
        if len(health.get("stale_topics", [])) > 10 and 7 not in steps:
            steps.append(7)

    if any(s in steps for s in [2, 3, 4, 5, 6, 7, 8]):
        steps.extend([9, 10])
    return sorted(set(steps))


class DreamReport:
    """Collects stats from each dream step."""

    def __init__(self):
        self.docs_extracted: int = 0
        self.facts_consolidated: int = 0
        self.entities_merged: int = 0
        self.relations_discovered: int = 0
        self.transitive_relations: int = 0
        self.entities_pruned: int = 0
        self.summaries_generated: int = 0
        self.errors: list[str] = []


def run_dream(
    config: Config,
    console: Console,
    *,
    dry_run: bool = False,
    step: int | None = None,
    resume: bool = False,
    reset: bool = False,
) -> DreamReport:
    """Run the full dream pipeline (or a single step).

    Uses LLM coordinator to plan steps when running full pipeline.
    Dashboard shows real-time progress via Rich Live.
    """
    from src.pipeline.dream_dashboard import DreamDashboard, DREAM_STEPS
    from src.pipeline.dream.maintenance import _step_load, _step_rebuild
    from src.pipeline.dream.consolidator import (
        _step_extract_documents,
        _step_consolidate_facts,
        _step_generate_summaries,
    )
    from src.pipeline.dream.merger import _step_merge_entities
    from src.pipeline.dream.discovery import _step_discover_relations, _step_transitive_relations

    report = DreamReport()
    memory_path = config.memory_path
    dream_id = datetime.now().isoformat()

    # Handle checkpoint resume/reset
    if reset:
        _clear_checkpoint(memory_path)

    checkpoint = _load_checkpoint(memory_path) if not reset else None

    # Always load graph first
    graph, entity_paths = _step_load(memory_path)

    # Determine which steps to run
    if resume and checkpoint:
        steps_to_run = [s for s in checkpoint["steps_planned"] if s > checkpoint["last_completed_step"]]
        dream_id = checkpoint["dream_id"]
        console.print(f"[dim]Resuming dream from step {checkpoint['last_completed_step'] + 1} (started {dream_id})[/dim]")
    elif step is not None:
        steps_to_run = [step]
    else:
        # Deterministic step selection based on stats + health analysis
        stats_text, counts = _collect_dream_stats(graph, entity_paths, config)
        from src.memory.insights import analyze_memory_health
        health = analyze_memory_health(graph, config)
        steps_to_run = decide_dream_steps(counts, health=health)
        logger.info("Dream plan (deterministic): steps=%s, stats=%s, health=%s", steps_to_run, counts, health["summary"])
        console.print(f"[dim]Coordinator plan (deterministic): steps {steps_to_run}[/dim]")
        console.print(f"[dim]Health: {health['summary']}[/dim]")

    # Always include step 1 (load)
    if 1 not in steps_to_run:
        steps_to_run.insert(0, 1)

    resumed = bool(resume and checkpoint)
    session_start_ts = datetime.now().isoformat()
    t0_session = time.monotonic()
    append_event(memory_path, "dream_session_started", "dream", {
        "dream_id": dream_id,
        "steps_planned": steps_to_run,
        "resumed": resumed,
        "entity_count": len(graph.entities),
        "relation_count": len(graph.relations),
    })

    with DreamDashboard(console) as dashboard:
        for s in range(1, 11):
            if s not in steps_to_run:
                dashboard.skip_step(s)
                step_name = DREAM_STEPS.get(s, {}).get("name", f"step_{s}")
                append_event(memory_path, "dream_step_skipped", "dream", {
                    "dream_id": dream_id,
                    "step": s,
                    "step_name": step_name,
                })
                continue

            dashboard.start_step(s)
            step_name = DREAM_STEPS.get(s, {}).get("name", f"step_{s}")
            t0 = time.monotonic()
            step_details: dict = {}
            append_event(memory_path, "dream_step_started", "dream", {
                "dream_id": dream_id,
                "step": s,
                "step_name": step_name,
            })
            try:
                if s == 1:
                    step_details = {"entities": len(graph.entities), "relations": len(graph.relations)}
                    dashboard.complete_step(s, f"{len(graph.entities)} entities, {len(graph.relations)} relations")

                elif s == 2:
                    try:
                        from src.pipeline.indexer import list_unextracted_docs as _list_docs
                        docs_before = len(_list_docs(config.faiss.manifest_path))
                    except Exception as e:
                        logger.warning("Could not count unextracted docs: %s", e)
                        docs_before = 0
                    n = _step_extract_documents(graph, memory_path, config, console, report, dry_run)
                    step_details = {"docs_found": docs_before, "docs_extracted": n}
                    dashboard.complete_step(s, f"{n} docs extracted" if n else "no docs pending")

                elif s == 3:
                    before = {"total_facts": _count_live_facts(entity_paths)}
                    consolidated_before = report.facts_consolidated
                    errors_before = len(report.errors)
                    snapshot = graph.model_copy(deep=True) if not dry_run else None
                    _step_consolidate_facts(graph, entity_paths, config, console, report, dry_run)
                    consolidated_delta = report.facts_consolidated - consolidated_before
                    error_delta = len(report.errors) - errors_before
                    total_candidates = consolidated_delta + error_delta  # rough: processed + errored
                    step_details = {
                        "processed": consolidated_delta,
                        "skipped": max(0, total_candidates - consolidated_delta - error_delta),
                        "errors": error_delta,
                    }
                    summary = f"{report.facts_consolidated} consolidated"
                    if report.facts_consolidated > 0 and not dry_run:
                        after = {"total_facts": _count_live_facts(entity_paths)}
                        approved, issues = validate_dream_step(s, before, after)
                        if not approved:
                            graph = snapshot
                            from src.memory.graph import save_graph
                            save_graph(memory_path, graph)
                            logger.warning("Dream step %d rolled back: %s", s, issues)
                            dashboard.fail_step(s, f"rolled back: {'; '.join(issues)[:40]}")
                            append_event(memory_path, "dream_step_failed", "dream", {
                                "dream_id": dream_id,
                                "step": s,
                                "step_name": step_name,
                                "duration_s": round(time.monotonic() - t0, 3),
                                "error": f"rolled back: {'; '.join(issues)}",
                            })
                            continue
                        summary = _validate_step(s, summary, before, after, report)
                    dashboard.complete_step(s, summary)

                elif s == 4:
                    before = {"total_entities": len(graph.entities)}
                    merged_before = report.entities_merged
                    snapshot = graph.model_copy(deep=True) if not dry_run else None
                    candidates_count = _step_merge_entities(graph, memory_path, config, console, report, dry_run, entity_paths)
                    merged_delta = report.entities_merged - merged_before
                    step_details = {
                        "candidates": candidates_count,
                        "merged": merged_delta,
                        "rejected": candidates_count - merged_delta,
                    }
                    summary = f"{report.entities_merged} merged"
                    if report.entities_merged > 0 and not dry_run:
                        after = {"total_entities": len(graph.entities)}
                        approved, issues = validate_dream_step(s, before, after)
                        if not approved:
                            graph = snapshot
                            from src.memory.graph import save_graph
                            save_graph(memory_path, graph)
                            logger.warning("Dream step %d rolled back: %s", s, issues)
                            dashboard.fail_step(s, f"rolled back: {'; '.join(issues)[:40]}")
                            append_event(memory_path, "dream_step_failed", "dream", {
                                "dream_id": dream_id,
                                "step": s,
                                "step_name": step_name,
                                "duration_s": round(time.monotonic() - t0, 3),
                                "error": f"rolled back: {'; '.join(issues)}",
                            })
                            continue
                        summary = _validate_step(s, summary, before, after, report)
                    dashboard.complete_step(s, summary)

                elif s == 5:
                    before = {"total_relations": len(graph.relations)}
                    discovered_before = report.relations_discovered
                    snapshot = graph.model_copy(deep=True) if not dry_run else None
                    candidates_count = _step_discover_relations(graph, memory_path, config, console, report, dry_run)
                    discovered_delta = report.relations_discovered - discovered_before
                    step_details = {
                        "candidates": candidates_count,
                        "created": discovered_delta,
                        "rejected": candidates_count - discovered_delta,
                    }
                    summary = f"{report.relations_discovered} discovered"
                    if report.relations_discovered > 0 and not dry_run:
                        after = {"total_relations": len(graph.relations)}
                        approved, issues = validate_dream_step(s, before, after)
                        if not approved:
                            graph = snapshot
                            from src.memory.graph import save_graph
                            save_graph(memory_path, graph)
                            logger.warning("Dream step %d rolled back: %s", s, issues)
                            dashboard.fail_step(s, f"rolled back: {'; '.join(issues)[:40]}")
                            append_event(memory_path, "dream_step_failed", "dream", {
                                "dream_id": dream_id,
                                "step": s,
                                "step_name": step_name,
                                "duration_s": round(time.monotonic() - t0, 3),
                                "error": f"rolled back: {'; '.join(issues)}",
                            })
                            continue
                        summary = _validate_step(s, summary, before, after, report)
                    dashboard.complete_step(s, summary)

                elif s == 6:
                    transitive_before = report.transitive_relations
                    _step_transitive_relations(graph, memory_path, config, console, report, dry_run)
                    step_details = {"created": report.transitive_relations - transitive_before}
                    dashboard.complete_step(s, f"{report.transitive_relations} inferred")

                elif s == 7:
                    pruned_before = report.entities_pruned
                    # Rescore before pruning to avoid using stale scores
                    if not dry_run:
                        from src.memory.scoring import recalculate_all_scores as _rescore
                        graph = _rescore(graph, config)
                    from src.pipeline.dream.maintenance import _step_prune_dead
                    _step_prune_dead(graph, memory_path, config, console, report, dry_run)
                    step_details = {"archived": report.entities_pruned - pruned_before}
                    dashboard.complete_step(s, f"{report.entities_pruned} pruned")

                elif s == 8:
                    summaries_before = report.summaries_generated
                    errors_before = len(report.errors)
                    already_have_summary = sum(1 for e in graph.entities.values() if e.summary)
                    _step_generate_summaries(graph, entity_paths, config, console, report, dry_run)
                    summaries_delta = report.summaries_generated - summaries_before
                    step_details = {
                        "generated": summaries_delta,
                        "skipped": already_have_summary,
                        "errors": len(report.errors) - errors_before,
                    }
                    dashboard.complete_step(s, f"{report.summaries_generated} generated")

                elif s == 9:
                    if not dry_run:
                        from src.memory.scoring import recalculate_all_scores
                        from src.memory.graph import save_graph
                        graph = recalculate_all_scores(graph, config)
                        save_graph(memory_path, graph)
                    step_details = {"entities_scored": len(graph.entities)}
                    dashboard.complete_step(s, "scores updated")

                elif s == 10:
                    if not dry_run:
                        _step_rebuild(graph, memory_path, config, console)
                    step_details = {}
                    dashboard.complete_step(s, "context + FAISS rebuilt")

                append_event(memory_path, "dream_step_completed", "dream", {
                    "dream_id": dream_id,
                    "step": s,
                    "step_name": step_name,
                    "duration_s": round(time.monotonic() - t0, 3),
                    "summary": dashboard._steps[s]["summary"],
                    "details": step_details,
                })

                if not dry_run:
                    _save_checkpoint(memory_path, dream_id, s, steps_to_run)

            except Exception as e:
                dashboard.fail_step(s, str(e)[:50])
                append_event(memory_path, "dream_step_failed", "dream", {
                    "dream_id": dream_id,
                    "step": s,
                    "step_name": step_name,
                    "duration_s": round(time.monotonic() - t0, 3),
                    "error": str(e),
                })
                report.errors.append(f"Step {s} failed: {e}")

    steps_completed = sum(1 for st in dashboard._steps.values() if st["status"] == "done")
    steps_failed = sum(1 for st in dashboard._steps.values() if st["status"] == "failed")
    append_event(memory_path, "dream_session_completed", "dream", {
        "dream_id": dream_id,
        "duration_s": round(time.monotonic() - t0_session, 3),
        "steps_completed": steps_completed,
        "steps_failed": steps_failed,
    })

    if not dry_run:
        _clear_checkpoint(memory_path)

    # Generate report (even in dry_run -- events were logged regardless)
    try:
        # Compute health for inclusion in report
        from src.memory.insights import analyze_memory_health as _health_fn
        health_for_report = _health_fn(graph, config)
        _generate_dream_report(memory_path, dream_id, session_start_ts, health=health_for_report)
    except Exception as e:
        logger.warning("Dream report generation failed: %s", e)

    return report


# -- Coordinator helpers --


def _collect_dream_stats(
    graph: GraphData,
    entity_paths: dict[str, Path],
    config: Config,
) -> tuple[str, dict[str, int]]:
    """Collect memory stats for the LLM coordinator. Returns (formatted_stats, counts)."""
    from src.pipeline.indexer import list_unextracted_docs
    from src.memory.store import read_entity
    from src.pipeline.dream.discovery import _TRANSITIVE_RULES

    today = date.today()
    counts = {
        "total_entities": len(graph.entities),
        "total_relations": len(graph.relations),
        "unextracted_docs": 0,
        "consolidation_candidates": 0,
        "merge_candidates": 0,
        "prune_candidates": 0,
        "summary_candidates": 0,
    }

    # Unextracted docs
    try:
        docs = list_unextracted_docs(config.faiss.manifest_path)
        counts["unextracted_docs"] = len(docs)
    except Exception as e:
        logger.warning("Could not list unextracted docs: %s", e)

    # Consolidation candidates (facts > max_facts for type)
    for eid, path in entity_paths.items():
        try:
            entity = graph.entities.get(eid)
            max_facts = config.get_max_facts(entity.type) if entity else 50
            _, sections = read_entity(path)
            facts = filter_live_facts(sections.get("Facts", []))
            if len(facts) > max_facts:
                counts["consolidation_candidates"] += 1
        except Exception as e:
            logger.debug("Could not check consolidation candidate %s: %s", eid, e)

    # Merge candidates (slug/alias overlap)
    seen_pairs: set[tuple[str, str]] = set()
    slugs = list(graph.entities.keys())
    for i, slug_a in enumerate(slugs):
        ea = graph.entities[slug_a]
        aliases_a = {a.lower() for a in ea.aliases} | {ea.title.lower()}
        for slug_b in slugs[i + 1:]:
            eb = graph.entities[slug_b]
            if ea.type != eb.type:
                continue
            aliases_b = {a.lower() for a in eb.aliases} | {eb.title.lower()}
            if aliases_a & aliases_b:
                pair = tuple(sorted([slug_a, slug_b]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    counts["merge_candidates"] += 1

    # Prune candidates
    related = {r.from_entity for r in graph.relations} | {r.to_entity for r in graph.relations}
    for eid, entity in graph.entities.items():
        if entity.score < 0.1 and entity.frequency <= 1 and entity.retention != "permanent" and eid not in related:
            if entity.created:
                try:
                    age = (today - date.fromisoformat(entity.created)).days
                    if age > 90:
                        counts["prune_candidates"] += 1
                except (ValueError, TypeError):
                    pass

    # Summary candidates
    for eid, entity in graph.entities.items():
        if not entity.summary:
            counts["summary_candidates"] += 1

    # Relation candidates (estimate: entities with FAISS neighbors minus existing relations)
    # Use total entities minus connected entity ratio as rough proxy
    connected_entities = len(related)
    total = len(graph.entities)
    counts["relation_candidates"] = max(0, total - connected_entities)

    # Transitive candidates (count eligible triples from adjacency)
    transitive_count = 0
    existing_pairs: set[tuple[str, str]] = set()
    for rel in graph.relations:
        existing_pairs.add((rel.from_entity, rel.to_entity))
        existing_pairs.add((rel.to_entity, rel.from_entity))
    adj_strong: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rel in graph.relations:
        if rel.strength >= 0.4:
            adj_strong[rel.from_entity].append((rel.to_entity, rel.type))
    for a, neighbors_a in adj_strong.items():
        for b, type_ab in neighbors_a:
            for c, type_bc in adj_strong.get(b, []):
                if c != a and (a, c) not in existing_pairs:
                    if (type_ab, type_bc) in _TRANSITIVE_RULES:
                        transitive_count += 1
    counts["transitive_candidates"] = transitive_count

    # Cluster analysis via BFS connected components
    adj: dict[str, set[str]] = defaultdict(set)
    for rel in graph.relations:
        adj[rel.from_entity].add(rel.to_entity)
        adj[rel.to_entity].add(rel.from_entity)

    visited: set[str] = set()
    clusters: list[set[str]] = []
    for eid in graph.entities:
        if eid in visited:
            continue
        cluster: set[str] = set()
        queue = [eid]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            cluster.add(node)
            for neighbor in adj.get(node, []):
                if neighbor not in visited and neighbor in graph.entities:
                    queue.append(neighbor)
        clusters.append(cluster)

    counts["clusters"] = len(clusters)
    counts["largest_cluster"] = max(len(c) for c in clusters) if clusters else 0

    stats = "\n".join(f"- {k.replace('_', ' ').title()}: {v}" for k, v in counts.items())
    return stats, counts


def validate_dream_step(step: int, before_state: dict, after_state: dict) -> tuple[bool, list[str]]:
    """Deterministic validation of dream step results."""
    issues = []
    if step == 3:  # Consolidation
        if after_state.get("total_facts", 0) > before_state.get("total_facts", 0):
            issues.append("Consolidation increased fact count")
    elif step == 4:  # Merge
        if after_state.get("total_entities", 0) > before_state.get("total_entities", 0):
            issues.append("Merge increased entity count")
    elif step == 5:  # Relation discovery
        new_rels = after_state.get("total_relations", 0) - before_state.get("total_relations", 0)
        if new_rels > 50:
            issues.append(f"Relation discovery added {new_rels} relations (suspiciously high)")
    return len(issues) == 0, issues


def _validate_step(step_num: int, summary: str, before_state: dict, after_state: dict, report: DreamReport) -> str:
    """Validate a critical step result using deterministic checks. Returns updated summary."""
    step_names = {3: "Fact Consolidation", 4: "Entity Merging", 5: "Relation Discovery"}
    step_name = step_names.get(step_num, f"Step {step_num}")
    approved, issues = validate_dream_step(step_num, before_state, after_state)
    if not approved:
        issues_str = "; ".join(issues)
        report.errors.append(f"Validation warning for {step_name}: {issues_str}")
        return f"{summary} [!validated: {issues_str[:30]}]"
    return f"{summary} [validated]"


def _count_live_facts(entity_paths: dict[str, Path]) -> int:
    """Count total live (non-superseded) facts across all entities."""
    from src.memory.store import read_entity

    total = 0
    for eid, path in entity_paths.items():
        try:
            _, sections = read_entity(path)
            facts = sections.get("Facts", [])
            total += len(filter_live_facts(facts))
        except Exception as e:
            logger.debug("Could not count facts for %s: %s", eid, e)
    return total
