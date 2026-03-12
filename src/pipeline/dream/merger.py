"""Dream step 4: Detect and merge duplicate entities."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

from rich.console import Console

from src.core.config import Config
from src.core.models import GraphData


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
    report,
    dry_run: bool,
    entity_paths: dict[str, Path] | None = None,
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
    faiss_candidates = _find_faiss_dedup_candidates(
        graph, memory_path, config, seen_pairs,
        similarity_threshold=config.dream.faiss_merge_threshold,
        max_candidates=config.dream.faiss_merge_max_candidates,
    )
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
                dossier_a = _build_dossier(slug_a, entity_a, memory_path, config)
                dossier_b = _build_dossier(slug_b, entity_b, memory_path, config)
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
            _do_merge(keep, drop, graph, memory_path, config, entity_paths)
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
    entity_paths: dict[str, Path] | None = None,
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
    for fact in drop_sections.get("Facts", []):
        if fact not in keep_sections.get("Facts", []):
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
    if entity_paths is not None:
        entity_paths.pop(drop_id, None)

    # Update keep entity in graph
    keep_entity.aliases = keep_fm.aliases
    keep_entity.tags = keep_fm.tags
    keep_entity.importance = keep_fm.importance
    keep_entity.frequency = keep_fm.frequency


def _build_dossier(eid: str, entity, memory_path: Path, config=None) -> str:
    """Build a compact dossier string for an entity."""
    from src.memory.store import read_entity

    path = memory_path / entity.file
    facts_text = ""
    if path.exists():
        try:
            _, sections = read_entity(path)
            facts = sections.get("Facts", [])
            live_facts = [f for f in facts if "[superseded]" not in f]
            max_facts = config.dream.dossier_max_facts if config else 3
            facts_text = "\n".join(live_facts[:max_facts])
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
