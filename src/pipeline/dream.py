"""Dream mode: brain-like memory reorganization during idle time.

10-step pipeline: load → extract docs → consolidate facts → merge entities
→ discover relations → transitive relations → prune dead → generate summaries
→ rescore → rebuild.

No new information enters — only reorganization of existing knowledge.
LLM coordinator plans which steps to run and validates critical results.
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
) -> DreamReport:
    """Run the full dream pipeline (or a single step).

    Uses LLM coordinator to plan steps when running full pipeline.
    Dashboard shows real-time progress via Rich Live.
    """
    from src.pipeline.dream_dashboard import DreamDashboard

    report = DreamReport()
    memory_path = config.memory_path

    # Always load graph first
    graph, entity_paths = _step_load(memory_path)

    # Determine which steps to run
    if step is not None:
        steps_to_run = [step]
    else:
        # Try LLM coordinator for planning
        stats_text, counts = _collect_dream_stats(graph, entity_paths, config)
        try:
            from src.core.llm import call_dream_plan
            plan = call_dream_plan(stats_text, config)
            steps_to_run = plan.steps
            logger.info("Dream plan (LLM): steps=%s, reason=%s", plan.steps, plan.reasoning)
            console.print(f"[dim]Coordinator plan: {plan.reasoning}[/dim]")
        except Exception as e:
            logger.warning("Dream coordinator failed, running all steps: %s", e)
            steps_to_run = list(range(1, 11))

    # Always include step 1 (load)
    if 1 not in steps_to_run:
        steps_to_run.insert(0, 1)

    with DreamDashboard(console) as dashboard:
        for s in range(1, 11):
            if s not in steps_to_run:
                dashboard.skip_step(s)
                continue

            dashboard.start_step(s)
            try:
                if s == 1:
                    dashboard.complete_step(s, f"{len(graph.entities)} entities, {len(graph.relations)} relations")

                elif s == 2:
                    n = _step_extract_documents(graph, memory_path, config, console, report, dry_run)
                    dashboard.complete_step(s, f"{n} docs extracted" if n else "no docs pending")

                elif s == 3:
                    _step_consolidate_facts(graph, entity_paths, config, console, report, dry_run)
                    summary = f"{report.facts_consolidated} consolidated"
                    if report.facts_consolidated > 0 and not dry_run:
                        summary = _validate_step(s, summary, config, report)
                    dashboard.complete_step(s, summary)

                elif s == 4:
                    _step_merge_entities(graph, memory_path, config, console, report, dry_run)
                    summary = f"{report.entities_merged} merged"
                    if report.entities_merged > 0 and not dry_run:
                        summary = _validate_step(s, summary, config, report)
                    dashboard.complete_step(s, summary)

                elif s == 5:
                    _step_discover_relations(graph, memory_path, config, console, report, dry_run)
                    summary = f"{report.relations_discovered} discovered"
                    if report.relations_discovered > 0 and not dry_run:
                        summary = _validate_step(s, summary, config, report)
                    dashboard.complete_step(s, summary)

                elif s == 6:
                    _step_transitive_relations(graph, memory_path, config, console, report, dry_run)
                    dashboard.complete_step(s, f"{report.transitive_relations} inferred")

                elif s == 7:
                    _step_prune_dead(graph, memory_path, config, console, report, dry_run)
                    dashboard.complete_step(s, f"{report.entities_pruned} pruned")

                elif s == 8:
                    _step_generate_summaries(graph, entity_paths, config, console, report, dry_run)
                    dashboard.complete_step(s, f"{report.summaries_generated} generated")

                elif s == 9:
                    if not dry_run:
                        from src.memory.scoring import recalculate_all_scores
                        graph = recalculate_all_scores(graph, config)
                    dashboard.complete_step(s, "scores updated")

                elif s == 10:
                    if not dry_run:
                        _step_rebuild(graph, memory_path, config, console)
                    dashboard.complete_step(s, "context + FAISS rebuilt")

            except Exception as e:
                dashboard.fail_step(s, str(e)[:50])
                report.errors.append(f"Step {s} failed: {e}")

    return report


# ── Coordinator helpers ──────────────────────────────────────


def _collect_dream_stats(
    graph: GraphData,
    entity_paths: dict[str, Path],
    config: Config,
) -> tuple[str, dict[str, int]]:
    """Collect memory stats for the LLM coordinator. Returns (formatted_stats, counts)."""
    from src.pipeline.indexer import list_unextracted_docs
    from src.memory.store import read_entity

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
    except Exception:
        pass

    # Consolidation candidates (facts > max_facts for type)
    for eid, path in entity_paths.items():
        try:
            entity = graph.entities.get(eid)
            max_facts = config.get_max_facts(entity.type) if entity else 50
            _, sections = read_entity(path)
            facts = [f for f in sections.get("Facts", []) if "[superseded]" not in f]
            if len(facts) > max_facts:
                counts["consolidation_candidates"] += 1
        except Exception:
            pass

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


def _validate_step(step_num: int, summary: str, config: Config, report: DreamReport) -> str:
    """Validate a critical step result via LLM. Returns updated summary."""
    step_names = {3: "Fact Consolidation", 4: "Entity Merging", 5: "Relation Discovery"}
    step_name = step_names.get(step_num, f"Step {step_num}")
    try:
        from src.core.llm import call_dream_validate
        validation = call_dream_validate(step_name, summary, config)
        if not validation.approved:
            issues = "; ".join(validation.issues)
            report.errors.append(f"Validation warning for {step_name}: {issues}")
            return f"{summary} [!validated: {issues[:30]}]"
        return f"{summary} [validated]"
    except Exception as e:
        logger.warning("Validation failed for step %d: %s", step_num, e)
        return summary


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


def _step_extract_documents(
    graph: GraphData,
    memory_path: Path,
    config: Config,
    console: Console,
    report: DreamReport,
    dry_run: bool,
) -> int:
    """Step 2: Extract entities from unprocessed RAG documents."""
    from src.pipeline.indexer import list_unextracted_docs, mark_doc_extracted
    from src.pipeline.extractor import extract_from_chat, sanitize_extraction
    from src.pipeline.resolver import resolve_all
    from src.pipeline.enricher import enrich_memory
    from src.pipeline.orchestrator import make_faiss_fn
    import pickle

    docs = list_unextracted_docs(config.faiss.manifest_path)
    if not docs:
        return 0

    console.print(f"  Found {len(docs)} unextracted document(s)")
    extracted = 0

    for doc in docs:
        source_id = doc["source_id"]
        doc_key = doc["key"]

        if dry_run:
            console.print(f"  [dim]Would extract entities from: {source_id}[/dim]")
            extracted += 1
            continue

        # Reconstruct text from FAISS chunks
        mapping_path = Path(config.faiss.mapping_path)
        if not mapping_path.exists():
            continue

        with open(mapping_path, "rb") as f:
            chunk_mapping = pickle.load(f)

        # Gather chunks for this document, sorted by index
        doc_chunks = sorted(
            [c for c in chunk_mapping if c.get("file") == doc_key],
            key=lambda c: c.get("chunk_idx", 0),
        )
        if not doc_chunks:
            continue

        text = "\n".join(c.get("chunk_text", "") for c in doc_chunks)
        if not text.strip():
            continue

        console.print(f"  [cyan]Extracting from: {source_id}[/cyan]")
        try:
            extraction = extract_from_chat(text, config)
            extraction = sanitize_extraction(extraction)

            if extraction.entities:
                resolved = resolve_all(extraction, graph, faiss_search_fn=make_faiss_fn(config, memory_path))
                enrich_memory(resolved, config)
                console.print(f"    [green]{len(extraction.entities)} entities extracted[/green]")

            mark_doc_extracted(config.faiss.manifest_path, doc_key)
            extracted += 1
        except Exception as e:
            report.errors.append(f"Doc extraction failed for {source_id}: {e}")
            console.print(f"    [yellow]Failed: {e}[/yellow]")

    report.docs_extracted = extracted
    return extracted


def _step_consolidate_facts(
    graph: GraphData,
    entity_paths: dict[str, Path],
    config: Config,
    console: Console,
    report: DreamReport,
    dry_run: bool,
) -> None:
    """Step 3: Consolidate redundant observations for entities with many facts."""
    from src.memory.store import read_entity, consolidate_entity_facts

    for eid, path in entity_paths.items():
        entity = graph.entities.get(eid)
        if not entity:
            continue
        try:
            max_facts = config.get_max_facts(entity.type)
            _, sections = read_entity(path)
            facts = sections.get("Facts", [])
            live_facts = [f for f in facts if "[superseded]" not in f]
            if len(live_facts) <= max_facts:
                continue

            if dry_run:
                console.print(f"  [dim]Would consolidate {entity.title} ({len(live_facts)} facts, max {max_facts})[/dim]")
                report.facts_consolidated += 1
                continue

            console.print(f"  [cyan]Consolidating {entity.title} ({len(live_facts)} facts, target {max_facts})...[/cyan]")
            result = consolidate_entity_facts(path, config, max_facts=max_facts)
            if result["changes"]:
                console.print(f"    [green]{', '.join(result['changes'])}[/green]")
                report.facts_consolidated += 1
        except Exception as e:
            report.errors.append(f"Fact consolidation failed for {eid}: {e}")
            console.print(f"    [yellow]Skipped {eid}: {e}[/yellow]")


def _find_faiss_dedup_candidates(
    graph: GraphData,
    memory_path: Path,
    config: Config,
    already_paired: set[tuple[str, str]],
    similarity_threshold: float = 0.80,
    max_candidates: int = 20,
) -> list[tuple[str, str]]:
    """Find potential duplicate entities via FAISS similarity search.

    Returns pairs not already covered by deterministic matching.
    These candidates require LLM confirmation before merging.
    """
    try:
        from src.pipeline.indexer import search as faiss_search
    except Exception:
        return []

    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set(already_paired)

    for eid in list(graph.entities.keys()):
        if len(candidates) >= max_candidates:
            break
        entity = graph.entities.get(eid)
        if not entity:
            continue

        try:
            results = faiss_search(entity.title, config, memory_path, top_k=5)
        except Exception:
            continue

        for result in results:
            if len(candidates) >= max_candidates:
                break
            other_id = result.entity_id
            if other_id == eid or other_id not in graph.entities:
                continue
            other = graph.entities[other_id]
            # Only consider same-type entities
            if entity.type != other.type:
                continue
            # Check similarity threshold
            if result.score < similarity_threshold:
                continue
            pair = tuple(sorted([eid, other_id]))
            if pair in seen:
                continue
            seen.add(pair)
            candidates.append((eid, other_id))

    return candidates


def _step_merge_entities(
    graph: GraphData,
    memory_path: Path,
    config: Config,
    console: Console,
    report: DreamReport,
    dry_run: bool,
) -> None:
    """Step 4: Detect and merge duplicate entities (slug similarity + FAISS)."""
    # Group by slug similarity (prefix match or containment)
    slugs = list(graph.entities.keys())
    merge_candidates: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for i, slug_a in enumerate(slugs):
        entity_a = graph.entities[slug_a]
        aliases_a = {a.lower() for a in entity_a.aliases} | {entity_a.title.lower()}

        for slug_b in slugs[i + 1:]:
            entity_b = graph.entities[slug_b]
            if entity_a.type != entity_b.type:
                continue

            aliases_b = {a.lower() for a in entity_b.aliases} | {entity_b.title.lower()}

            if aliases_a & aliases_b:
                pair = tuple(sorted([slug_a, slug_b]))
                if pair not in seen_pairs:
                    merge_candidates.append((slug_a, slug_b))
                    seen_pairs.add(pair)

    # Phase 2: FAISS-based candidate expansion with LLM confirmation
    faiss_candidates = _find_faiss_dedup_candidates(graph, memory_path, config, seen_pairs)
    merge_candidates.extend(faiss_candidates)

    if not merge_candidates:
        console.print("  No duplicate entities detected")
        return

    # Track which pairs came from FAISS (need LLM confirmation)
    faiss_pair_set = {tuple(sorted(p)) for p in faiss_candidates}

    for slug_a, slug_b in merge_candidates:
        entity_a = graph.entities.get(slug_a)
        entity_b = graph.entities.get(slug_b)
        if not entity_a or not entity_b:
            continue

        # FAISS-sourced candidates require LLM confirmation
        pair_key = tuple(sorted([slug_a, slug_b]))
        if pair_key in faiss_pair_set and not dry_run:
            try:
                from src.core.llm import call_dedup_check
                dossier_a = _build_dossier(slug_a, entity_a, memory_path)
                dossier_b = _build_dossier(slug_b, entity_b, memory_path)
                verdict = call_dedup_check(
                    entity_a.title, entity_a.type, dossier_a,
                    entity_b.title, entity_b.type, dossier_b,
                    config,
                )
                if not verdict.is_duplicate or verdict.confidence < 0.7:
                    console.print(
                        f"  [dim]LLM rejected merge: {entity_a.title} / {entity_b.title} "
                        f"(confidence={verdict.confidence:.2f}, reason={verdict.reason})[/dim]"
                    )
                    continue
                console.print(
                    f"  [green]LLM confirmed duplicate: {entity_a.title} / {entity_b.title} "
                    f"(confidence={verdict.confidence:.2f})[/green]"
                )
            except Exception as e:
                report.errors.append(f"Dedup LLM check failed for {slug_a}/{slug_b}: {e}")
                console.print(f"    [yellow]LLM dedup check failed, skipping: {e}[/yellow]")
                continue

        keep, drop = (slug_a, slug_b) if entity_a.score >= entity_b.score else (slug_b, slug_a)
        keep_entity = graph.entities[keep]
        drop_entity = graph.entities[drop]

        if dry_run:
            source = "FAISS+LLM" if pair_key in faiss_pair_set else "deterministic"
            console.print(f"  [dim]Would merge '{drop_entity.title}' into '{keep_entity.title}' ({source})[/dim]")
            report.entities_merged += 1
            continue

        console.print(f"  [cyan]Merging '{drop_entity.title}' -> '{keep_entity.title}'[/cyan]")
        try:
            _do_merge(keep, drop, graph, memory_path, config)
            report.entities_merged += 1
        except Exception as e:
            report.errors.append(f"Merge failed {drop} -> {keep}: {e}")
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
    """Step 5: Use FAISS similarity + LLM to discover new relations."""
    from src.pipeline.indexer import search as faiss_search
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

        dossier_a = _build_dossier(eid_a, entity_a, memory_path)
        dossier_b = _build_dossier(eid_b, entity_b, memory_path)

        if dry_run:
            console.print(f"  [dim]Would evaluate: {entity_a.title} <-> {entity_b.title}[/dim]")
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
                console.print(f"    [green]{entity_a.title} -> {rel_type} -> {entity_b.title}[/green]")
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


# Transitive inference rules: (rel_type_A, rel_type_B) -> inferred_type
_TRANSITIVE_RULES: dict[tuple[str, str], str] = {
    ("affects", "affects"): "affects",
    ("part_of", "part_of"): "part_of",
    ("requires", "requires"): "requires",
    ("improves", "affects"): "improves",
    ("worsens", "affects"): "worsens",
    ("uses", "part_of"): "uses",
}


def _step_transitive_relations(
    graph: GraphData,
    memory_path: Path,
    config: Config,
    console: Console,
    report: DreamReport,
    dry_run: bool,
    min_strength: float = 0.4,
    max_new: int = 20,
) -> None:
    """Step 6: Infer transitive relations (deterministic, no LLM).

    For each triple (A→rel1→B, B→rel2→C) where A and C have no direct relation,
    apply transitive rules to create inferred relations with reduced strength.
    """
    from src.memory.graph import add_relation, save_graph

    # Build adjacency: entity -> list of (target, relation)
    adjacency: dict[str, list[tuple[str, GraphRelation]]] = defaultdict(list)
    for rel in graph.relations:
        if rel.strength >= min_strength:
            adjacency[rel.from_entity].append((rel.to_entity, rel))

    # Build existing relation set
    existing: set[tuple[str, str]] = set()
    for rel in graph.relations:
        existing.add((rel.from_entity, rel.to_entity))
        existing.add((rel.to_entity, rel.from_entity))

    discovered = 0
    for entity_a, neighbors_a in adjacency.items():
        if discovered >= max_new:
            break
        for entity_b, rel_ab in neighbors_a:
            if discovered >= max_new:
                break
            for entity_c, rel_bc in adjacency.get(entity_b, []):
                if discovered >= max_new:
                    break
                if entity_c == entity_a:
                    continue
                if (entity_a, entity_c) in existing:
                    continue
                # Check transitive rule
                rule_key = (rel_ab.type, rel_bc.type)
                inferred_type = _TRANSITIVE_RULES.get(rule_key)
                if not inferred_type:
                    continue
                # Inferred strength = min of both * 0.5
                inferred_strength = min(rel_ab.strength, rel_bc.strength) * 0.5

                title_a = graph.entities.get(entity_a)
                title_b = graph.entities.get(entity_b)
                title_c = graph.entities.get(entity_c)
                if not title_a or not title_c:
                    continue
                context = f"transitive: {title_a.title} →{rel_ab.type}→ {title_b.title if title_b else entity_b} →{rel_bc.type}→ {title_c.title}"

                if dry_run:
                    console.print(f"  [dim]Would infer: {title_a.title} →{inferred_type}→ {title_c.title}[/dim]")
                    discovered += 1
                    existing.add((entity_a, entity_c))
                    existing.add((entity_c, entity_a))
                    continue

                new_rel = GraphRelation(
                    from_entity=entity_a, to_entity=entity_c,
                    type=inferred_type,
                    strength=inferred_strength,
                    context=context,
                )
                add_relation(graph, new_rel, strength_growth=0.0)
                existing.add((entity_a, entity_c))
                existing.add((entity_c, entity_a))
                discovered += 1
                console.print(f"    [green]{title_a.title} →{inferred_type}→ {title_c.title} (transitive)[/green]")

    report.transitive_relations = discovered
    if discovered and not dry_run:
        save_graph(memory_path, graph)
        console.print(f"  [green]Inferred {discovered} transitive relation(s)[/green]")
    elif not discovered:
        console.print("  No transitive relations to infer")


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
    """Step 7: Archive low-score orphan entities."""
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
    """Step 8: Generate/refresh entity summaries via LLM."""
    from src.core.llm import call_entity_summary
    from src.memory.store import read_entity, write_entity

    for eid, path in entity_paths.items():
        entity = graph.entities.get(eid)
        if not entity:
            continue
        if eid not in graph.entities:
            continue  # May have been pruned

        if entity.summary:
            continue

        try:
            fm, sections = read_entity(path)
            facts = sections.get("Facts", [])
            live_facts = [f for f in facts if "[superseded]" not in f]
            if not live_facts:
                continue

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
                display = f"{summary[:60]}..." if len(summary) > 60 else summary
                console.print(f"  [green]{entity.title}: {display}[/green]")
        except Exception as e:
            report.errors.append(f"Summary generation failed for {eid}: {e}")
            console.print(f"    [yellow]Skipped {eid}: {e}[/yellow]")


def _step_rebuild(
    graph: GraphData,
    memory_path: Path,
    config: Config,
    console: Console,
) -> None:
    """Step 10: Rebuild context and FAISS index."""
    from src.memory.context import build_context, write_context, write_index
    from src.memory.graph import save_graph
    from src.pipeline.indexer import build_index

    save_graph(memory_path, graph)

    context_text = build_context(graph, memory_path, config)
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
