"""CLI for memory-ai. All commands use Click."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from src.core.config import load_config

console = Console()
logger = logging.getLogger("memory-ai")


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option("--config", "-c", "config_path", default=None, help="Path to config.yaml")
@click.pass_context
def cli(ctx, verbose, config_path):
    """memory-ai — Personal persistent memory system for LLMs."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    project_root = Path.cwd()
    ctx.obj["config"] = load_config(config_path=config_path, project_root=project_root)


@cli.command()
@click.pass_context
def run(ctx):
    """Process all pending chats → extraction → enrichment → context → FAISS."""
    config = ctx.obj["config"]
    memory_path = config.memory_path

    from src.memory.store import (
        list_unprocessed_chats, get_chat_content, mark_chat_processed,
        mark_chat_fallback, increment_extraction_retries,
    )
    from src.memory.graph import load_graph
    from src.pipeline.extractor import extract_from_chat
    from src.pipeline.resolver import resolve_all
    from src.pipeline.arbitrator import arbitrate_entity
    from src.pipeline.enricher import enrich_memory
    from src.pipeline.indexer import incremental_update
    from src.memory.context import write_context
    from src.core.models import Resolution

    # Max extraction retries before falling back to doc_ingest
    EXTRACTION_MAX_RETRIES = 2

    chats = list_unprocessed_chats(memory_path)
    if not chats:
        console.print("[yellow]No pending chats to process.[/yellow]")
        return

    max_chats = config.job_max_chats_per_run
    chats = chats[:max_chats]
    console.print(f"[bold]Processing {len(chats)} pending chat(s)...[/bold]")

    for chat_path in chats:
        console.print(f"\n[cyan]→ {chat_path.name}[/cyan]")

        # Step 1: Extract
        content = get_chat_content(chat_path)
        if not content.strip():
            console.print("  [dim]Empty chat, skipping[/dim]")
            mark_chat_processed(chat_path, [], [])
            continue

        try:
            extraction = extract_from_chat(content, config)
            console.print(f"  Extracted {len(extraction.entities)} entities, {len(extraction.relations)} relations")
        except Exception as e:
            is_timeout = _is_timeout_error(e)
            retries = increment_extraction_retries(chat_path)
            should_fallback = is_timeout or retries >= EXTRACTION_MAX_RETRIES

            if should_fallback:
                reason = f"timeout: {e}" if is_timeout else f"max retries ({retries}): {e}"
                console.print(f"  [yellow]Extraction failed, falling back to doc_ingest: {reason}[/yellow]")
                _fallback_to_doc_ingest(chat_path, content, reason, memory_path, config)
                # Record in retry ledger for potential future replay
                from src.pipeline.ingest_state import record_failure
                record_failure(chat_path, reason, config)
            else:
                console.print(f"  [red]Extraction failed (retry {retries}/{EXTRACTION_MAX_RETRIES}): {e}[/red]")
            continue

        # Step 2: Resolve
        graph = load_graph(memory_path)
        resolved = resolve_all(extraction, graph, faiss_search_fn=_make_faiss_fn(config, memory_path))

        # Step 3: Arbitrate ambiguous
        for item in resolved.resolved:
            if item.resolution.status == "ambiguous":
                try:
                    arb_result = arbitrate_entity(
                        item.raw.name,
                        extraction.summary,
                        item.resolution.candidates,
                        graph,
                        config,
                    )
                    if arb_result.action == "existing" and arb_result.existing_id:
                        item.resolution = Resolution(
                            status="resolved",
                            entity_id=arb_result.existing_id,
                        )
                    else:
                        from src.pipeline.resolver import slugify
                        item.resolution = Resolution(
                            status="new",
                            suggested_slug=slugify(item.raw.name),
                        )
                except Exception as e:
                    console.print(f"  [yellow]Arbitration failed for {item.raw.name}: {e}[/yellow]")
                    from src.pipeline.resolver import slugify
                    item.resolution = Resolution(
                        status="new",
                        suggested_slug=slugify(item.raw.name),
                    )

        # Step 4: Enrich
        try:
            report = enrich_memory(resolved, config)
            console.print(f"  Updated: {report.entities_updated}, Created: {report.entities_created}")
            if report.errors:
                for err in report.errors:
                    console.print(f"  [yellow]Warning: {err}[/yellow]")
        except Exception as e:
            console.print(f"  [red]Enrichment failed: {e}[/red]")
            continue

        # Mark processed
        mark_chat_processed(chat_path, report.entities_updated, report.entities_created)

    # Step 7: Generate context (deterministic template)
    try:
        console.print("\n[bold]Generating context...[/bold]")
        graph = load_graph(memory_path)
        from src.memory.context import build_deterministic_context
        context_text = build_deterministic_context(graph, memory_path, config)
        if context_text.strip():
            write_context(memory_path, context_text)
            console.print("  [green]_context.md updated[/green]")
        else:
            console.print("  [dim]No entities for context[/dim]")
    except Exception as e:
        console.print(f"  [yellow]Context generation warning: {e}[/yellow]")

    # Step 8: FAISS indexing
    try:
        console.print("[bold]Updating FAISS index...[/bold]")
        incremental_update(memory_path, config)
        console.print("  [green]FAISS index updated[/green]")
    except Exception as e:
        console.print(f"  [yellow]FAISS indexing warning: {e}[/yellow]")

    console.print("\n[bold green]✓ Pipeline complete[/bold green]")


@cli.command("rebuild-graph")
@click.pass_context
def rebuild_graph(ctx):
    """Rebuild _graph.json from all MD files."""
    config = ctx.obj["config"]
    from src.memory.graph import rebuild_from_md, save_graph

    console.print("[bold]Rebuilding graph from MD files...[/bold]")
    graph = rebuild_from_md(config.memory_path)
    save_graph(config.memory_path, graph)
    console.print(f"[green]✓ Graph rebuilt: {len(graph.entities)} entities, {len(graph.relations)} relations[/green]")


@cli.command("rebuild-faiss")
@click.pass_context
def rebuild_faiss(ctx):
    """Full FAISS index rebuild."""
    config = ctx.obj["config"]
    from src.pipeline.indexer import build_index

    console.print("[bold]Rebuilding FAISS index...[/bold]")
    manifest = build_index(config.memory_path, config)
    n_files = len(manifest.get("indexed_files", {}))
    console.print(f"[green]✓ FAISS rebuilt: {n_files} files indexed[/green]")


@cli.command("rebuild-all")
@click.pass_context
def rebuild_all(ctx):
    """Rebuild graph + context + FAISS."""
    config = ctx.obj["config"]
    from src.memory.graph import rebuild_from_md, save_graph
    from src.memory.context import build_deterministic_context, write_context, write_index
    from src.memory.scoring import recalculate_all_scores
    from src.pipeline.indexer import build_index

    console.print("[bold]Rebuilding everything...[/bold]")

    # Graph
    graph = rebuild_from_md(config.memory_path)
    graph = recalculate_all_scores(graph, config)

    # Repair entities with empty mention_dates (created before the fix)
    repaired = 0
    for eid, entity in graph.entities.items():
        if not entity.mention_dates and entity.created:
            entity.mention_dates = [entity.created]
            repaired += 1
        elif not entity.mention_dates and entity.last_mentioned:
            entity.mention_dates = [entity.last_mentioned]
            repaired += 1
    if repaired:
        console.print(f"  [yellow]Repaired {repaired} entities with empty mention_dates[/yellow]")

    save_graph(config.memory_path, graph)
    console.print(f"  Graph: {len(graph.entities)} entities, {len(graph.relations)} relations")

    # Index
    write_index(config.memory_path, graph)
    console.print("  _index.md updated")

    # Context (deterministic template)
    context_text = build_deterministic_context(graph, config.memory_path, config)
    if context_text.strip():
        write_context(config.memory_path, context_text)
        console.print("  _context.md updated")

    # FAISS
    manifest = build_index(config.memory_path, config)
    console.print(f"  FAISS: {len(manifest.get('indexed_files', {}))} files indexed")

    console.print("[bold green]✓ Full rebuild complete[/bold green]")


@cli.command()
@click.pass_context
def validate(ctx):
    """Check graph consistency (orphan relations, missing files)."""
    config = ctx.obj["config"]
    from src.memory.graph import load_graph, validate_graph

    graph = load_graph(config.memory_path)
    warnings = validate_graph(graph, config.memory_path)

    if not warnings:
        console.print("[green]✓ Graph is consistent[/green]")
    else:
        console.print(f"[yellow]Found {len(warnings)} warning(s):[/yellow]")
        for w in warnings:
            console.print(f"  ⚠ {w}")


@cli.command()
@click.pass_context
def stats(ctx):
    """Display memory metrics."""
    config = ctx.obj["config"]
    from src.memory.graph import load_graph
    from src.memory.store import list_unprocessed_chats, list_entities

    graph = load_graph(config.memory_path)
    unprocessed = list_unprocessed_chats(config.memory_path)
    entities = list_entities(config.memory_path)

    # Summary table
    table = Table(title="memory-ai Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total entities (graph)", str(len(graph.entities)))
    table.add_row("Total entities (files)", str(len(entities)))
    table.add_row("Total relations", str(len(graph.relations)))
    table.add_row("Pending chats", str(len(unprocessed)))

    # Count by type
    type_counts: dict[str, int] = {}
    for _, entity in graph.entities.items():
        type_counts[entity.type] = type_counts.get(entity.type, 0) + 1

    for t, c in sorted(type_counts.items()):
        table.add_row(f"  └ {t}", str(c))

    # Context/index status
    context_exists = (config.memory_path / "_context.md").exists()
    index_exists = (config.memory_path / "_index.md").exists()
    faiss_exists = Path(config.faiss.index_path).exists()

    table.add_row("_context.md", "✓" if context_exists else "✗")
    table.add_row("_index.md", "✓" if index_exists else "✗")
    table.add_row("FAISS index", "✓" if faiss_exists else "✗")

    console.print(table)


@cli.command()
@click.pass_context
def inbox(ctx):
    """Process files in _inbox/."""
    config = ctx.obj["config"]
    from src.pipeline.inbox import process_inbox

    processed = process_inbox(config.memory_path, config)
    if processed:
        console.print(f"[green]✓ Processed {len(processed)} file(s): {', '.join(processed)}[/green]")
        console.print("[dim]Run 'memory run' to extract and enrich from these chats.[/dim]")
    else:
        console.print("[yellow]No files to process in _inbox/[/yellow]")


@cli.command()
@click.option("--all", "clean_all", is_flag=True, help="Remove all generated artifacts and caches")
@click.option("--artifacts", is_flag=True, help="Remove generated artifacts (_context.md, _index.md, FAISS, graph)")
@click.option("--dry-run", is_flag=True, help="Show what would be removed without deleting")
@click.pass_context
def clean(ctx, clean_all, artifacts, dry_run):
    """Remove generated files and caches. Backs up before destructive operations."""
    import shutil
    import tarfile
    from datetime import datetime

    config = ctx.obj["config"]
    memory_path = config.memory_path

    # Determine what to clean
    if not clean_all and not artifacts:
        console.print("[yellow]Specify --all or --artifacts. Use --dry-run to preview.[/yellow]")
        return

    targets: list[tuple[Path, str]] = []

    # Artifact targets (always included with --artifacts or --all)
    artifact_files = [
        (memory_path / "_context.md", "generated context"),
        (memory_path / "_index.md", "generated index"),
        (memory_path / "_graph.json", "entity graph"),
        (memory_path / "_graph.json.bak", "graph backup"),
        (memory_path / "_graph.lock", "graph lockfile"),
        (Path(config.faiss.index_path), "FAISS index"),
        (Path(config.faiss.mapping_path), "FAISS mapping"),
        (Path(config.faiss.manifest_path), "FAISS manifest"),
        (Path(config.ingest.jobs_path), "ingest jobs"),
    ]
    for path, desc in artifact_files:
        if path.exists():
            targets.append((path, desc))

    # Extended targets for --all
    if clean_all:
        pycache_dirs = list(Path(".").rglob("__pycache__"))
        for d in pycache_dirs:
            targets.append((d, "__pycache__"))
        processed_dir = memory_path / "_inbox" / "_processed"
        if processed_dir.exists():
            targets.append((processed_dir, "processed inbox archive"))

    if not targets:
        console.print("[green]Nothing to clean.[/green]")
        return

    # Preview
    console.print(f"[bold]{'[DRY RUN] ' if dry_run else ''}Files to remove:[/bold]")
    for path, desc in targets:
        console.print(f"  {desc}: {path}")

    if dry_run:
        console.print(f"\n[dim]{len(targets)} item(s) would be removed.[/dim]")
        return

    # Backup before destructive operations
    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"pre-clean-{stamp}.tar.gz"

    existing_targets = [p for p, _ in targets if p.exists() and not str(p).endswith("__pycache__")]
    if existing_targets:
        console.print(f"\n[dim]Backing up to {backup_path}...[/dim]")
        with tarfile.open(backup_path, "w:gz") as tar:
            for path in existing_targets:
                try:
                    tar.add(path)
                except Exception:
                    pass
        console.print(f"  [green]Backup saved ({backup_path})[/green]")

    # Delete
    removed = 0
    for path, desc in targets:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            removed += 1
        except Exception as e:
            console.print(f"  [red]Failed to remove {path}: {e}[/red]")

    console.print(f"\n[bold green]Cleaned {removed} item(s).[/bold green]")


@cli.command()
@click.option(
    "--transport", "-t",
    type=click.Choice(["stdio", "sse"], case_sensitive=False),
    default=None,
    help="Transport mode (overrides config.yaml). Default: from config.",
)
@click.pass_context
def serve(ctx, transport):
    """Start the MCP server."""
    from src.mcp.server import run_server
    config = ctx.obj["config"]
    effective = transport or config.mcp_transport
    console.print(f"[bold]Starting MCP server (transport={effective})...[/bold]")
    run_server(config=config, transport_override=transport)


@cli.command()
@click.option("--list", "list_only", is_flag=True, help="List retriable failures without replaying")
@click.pass_context
def replay(ctx, list_only):
    """Replay failed extractions from the retry ledger.

    Retries chats that previously failed extraction, using the full
    pipeline path (not doc-ingest fallback). Use --list to preview.
    """
    config = ctx.obj["config"]
    from src.pipeline.ingest_state import list_retriable, mark_replayed

    entries = list_retriable(config)
    if not entries:
        console.print("[green]No retriable failures in ledger.[/green]")
        return

    if list_only:
        table = Table(title="Retry Ledger (pending)")
        table.add_column("File", style="cyan")
        table.add_column("Error", style="yellow", max_width=60)
        table.add_column("Attempts", style="red")
        table.add_column("Recorded", style="dim")
        for e in entries:
            table.add_row(
                Path(e["file"]).name,
                e.get("error", "")[:60],
                str(e.get("attempts", 0)),
                e.get("recorded", ""),
            )
        console.print(table)
        return

    console.print(f"[bold]Replaying {len(entries)} failed extraction(s)...[/bold]")

    from src.memory.store import get_chat_content
    from src.pipeline.extractor import extract_from_chat
    from src.pipeline.resolver import resolve_all
    from src.pipeline.enricher import enrich_memory
    from src.memory.graph import load_graph
    from src.memory.store import mark_chat_processed
    from src.core.models import Resolution

    for entry in entries:
        chat_path = Path(entry["file"])
        if not chat_path.exists():
            console.print(f"  [dim]{chat_path.name}: file gone, skipping[/dim]")
            mark_replayed(str(chat_path), success=False, config=config, error="file not found")
            continue

        console.print(f"\n[cyan]→ Replaying {chat_path.name}[/cyan]")
        content = get_chat_content(chat_path)

        try:
            extraction = extract_from_chat(content, config)
            console.print(f"  Extracted {len(extraction.entities)} entities")

            graph = load_graph(config.memory_path)
            resolved = resolve_all(extraction, graph, faiss_search_fn=_make_faiss_fn(config, config.memory_path))

            report = enrich_memory(resolved, config)
            mark_chat_processed(chat_path, report.entities_updated, report.entities_created)
            mark_replayed(str(chat_path), success=True, config=config)
            console.print(f"  [green]Replay succeeded[/green]")

        except Exception as e:
            mark_replayed(str(chat_path), success=False, config=config, error=str(e))
            console.print(f"  [red]Replay failed: {e}[/red]")

    console.print("\n[bold green]Replay complete.[/bold green]")


@cli.command()
@click.option("--dry-run", is_flag=True, help="Preview duplicates without merging")
@click.option("--facts", is_flag=True, help="Consolidate redundant observations within entities via LLM")
@click.option("--min-facts", default=8, help="Minimum facts to trigger consolidation (default: 8)")
@click.pass_context
def consolidate(ctx, dry_run, facts, min_facts):
    """Detect and report duplicate entities, or consolidate facts within entities."""
    config = ctx.obj["config"]

    if facts:
        _consolidate_facts(config, dry_run, min_facts)
        return

    from src.memory.graph import load_graph
    from collections import defaultdict

    graph = load_graph(config.memory_path)

    # Simple name-based duplicate detection
    name_groups = defaultdict(list)
    for eid, entity in graph.entities.items():
        key = entity.title.lower().strip()
        name_groups[key].append((eid, entity))
        for alias in entity.aliases:
            alias_key = alias.lower().strip()
            name_groups[alias_key].append((eid, entity))

    duplicates = {k: v for k, v in name_groups.items() if len(set(eid for eid, _ in v)) > 1}

    if not duplicates:
        console.print("[green]No duplicate entities detected.[/green]")
        return

    console.print(f"[yellow]Found {len(duplicates)} potential duplicate group(s):[/yellow]")
    for name, entities in duplicates.items():
        unique_ids = list(set(eid for eid, _ in entities))
        if len(unique_ids) > 1:
            console.print(f"\n  '{name}':")
            for eid in unique_ids:
                e = graph.entities[eid]
                console.print(f"    - {eid} ({e.type}, score: {e.score:.2f}, freq: {e.frequency})")

    if dry_run:
        console.print(f"\n[dim]Dry run -- no changes made. Run without --dry-run to merge.[/dim]")


def _consolidate_facts(config, dry_run: bool, min_facts: int) -> None:
    """Consolidate redundant observations within entities via LLM."""
    from src.memory.graph import load_graph
    from src.memory.store import read_entity, consolidate_entity_facts

    graph = load_graph(config.memory_path)
    memory_path = config.memory_path

    # Find entities with enough facts to warrant consolidation
    candidates = []
    for eid, entity in graph.entities.items():
        entity_path = memory_path / entity.file
        if not entity_path.exists():
            continue
        try:
            _, sections = read_entity(entity_path)
            facts = sections.get("Facts", [])
            live_facts = [f for f in facts if "[superseded]" not in f]
            if len(live_facts) >= min_facts:
                candidates.append((eid, entity, entity_path, len(live_facts)))
        except Exception:
            continue

    if not candidates:
        console.print(f"[green]No entities with {min_facts}+ facts to consolidate.[/green]")
        return

    console.print(f"[bold]Found {len(candidates)} entity/ies with {min_facts}+ facts:[/bold]")
    for eid, entity, _, fact_count in candidates:
        console.print(f"  {entity.title} ({entity.type}): {fact_count} facts")

    if dry_run:
        console.print(f"\n[dim]Dry run -- no changes made. Run without --dry-run to consolidate.[/dim]")
        return

    for eid, entity, entity_path, _ in candidates:
        console.print(f"\n[cyan]→ Consolidating {entity.title}...[/cyan]")
        try:
            result = consolidate_entity_facts(entity_path, config)
            if result["changes"]:
                console.print(f"  [green]{', '.join(result['changes'])}[/green]")
            else:
                console.print(f"  [dim]No changes needed[/dim]")
        except Exception as e:
            console.print(f"  [red]Failed: {e}[/red]")

    console.print("\n[bold green]Fact consolidation complete.[/bold green]")


def _make_faiss_fn(config, memory_path):
    """Create a FAISS search wrapper compatible with resolver's faiss_search_fn signature.

    The resolver expects: fn(query: str, top_k: int, threshold: float) -> list[dict]
    The indexer.search expects: search(query, config, memory_path, top_k) -> list[SearchResult]
    """
    from src.pipeline.indexer import search as _search

    def fn(query: str, top_k: int = 3, threshold: float = 0.85):
        results = _search(query, config, memory_path, top_k=top_k)
        return [
            {"entity_id": r.entity_id, "score": r.score}
            for r in results
            if r.score >= threshold
        ]

    return fn


def _is_timeout_error(exc: Exception) -> bool:
    """Check if an exception is a timeout-related error.

    Recognizes StallError from stall-aware extraction, plus standard
    timeout exceptions from httpx/litellm.
    """
    from src.core.llm import StallError
    if isinstance(exc, StallError):
        return True
    timeout_indicators = ("timeout", "timed out", "ReadTimeout", "ConnectTimeout", "stall")
    exc_str = str(exc).lower()
    exc_type = type(exc).__name__.lower()
    return any(t.lower() in exc_str or t.lower() in exc_type for t in timeout_indicators)


def _fallback_to_doc_ingest(
    chat_path, content: str, reason: str, memory_path, config,
) -> None:
    """Fall back to doc_ingest for a chat that failed extraction."""
    from src.memory.store import mark_chat_fallback
    from src.pipeline.ingest_state import compute_ingest_key, create_job, transition_job, has_been_ingested
    from src.pipeline.doc_ingest import ingest_document

    source_id = chat_path.name
    key = compute_ingest_key(source_id, content)

    if has_been_ingested(key, config):
        console.print(f"  [dim]Already doc-ingested, marking processed[/dim]")
        mark_chat_fallback(chat_path, fallback="doc_ingest", error=reason)
        return

    try:
        job = create_job(key, config, route="fallback_doc_ingest")
        transition_job(job.job_id, "running", config)
        result = ingest_document(source_id, content, key, memory_path, config)
        transition_job(
            job.job_id, "succeeded", config,
            chunks_indexed=result.get("chunks_indexed", 0),
        )
        console.print(f"  [green]Doc-ingest fallback OK: {result.get('chunks_indexed', 0)} chunks indexed[/green]")
    except Exception as e2:
        console.print(f"  [red]Doc-ingest fallback also failed: {e2}[/red]")

    # Always mark processed so it's not retried for extraction
    mark_chat_fallback(chat_path, fallback="doc_ingest", error=reason)


if __name__ == "__main__":
    cli()
