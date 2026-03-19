"""CLI for memory-ai. All commands use Click."""

from __future__ import annotations

import logging
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from src.core.config import load_config
from src.pipeline.orchestrator import (
    auto_consolidate,
    consolidate_facts,
    fallback_to_doc_ingest,
    is_timeout_error,
    run_pipeline,
)

console = Console()
logger = logging.getLogger(__name__)


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
    config = load_config(config_path=config_path, project_root=project_root)
    ctx.obj["config"] = config

    from src.memory.store import init_memory_structure
    init_memory_structure(config.memory_path)


@cli.command()
@click.pass_context
def run(ctx):
    """Process all pending chats → extraction → enrichment → consolidation → context → FAISS."""
    run_pipeline(ctx.obj["config"], console, consolidate=True)


@cli.command("run-light")
@click.pass_context
def run_light(ctx):
    """Lightweight run: same as 'run' but skips auto-consolidation (no LLM calls for merging)."""
    run_pipeline(ctx.obj["config"], console, consolidate=False)


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
    from src.memory.context import build_context_for_config, write_context, write_index
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

    # Context
    context_text = build_context_for_config(graph, config.memory_path, config)
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
@click.option("--full", is_flag=True, help="Everything in --all + reset chats to unprocessed")
@click.option("--chats", "reset_chats", is_flag=True, help="Only reset chats to unprocessed")
@click.option("--dry-run", is_flag=True, help="Show what would be removed without deleting")
@click.pass_context
def clean(ctx, clean_all, artifacts, full, reset_chats, dry_run):
    """Remove generated files and caches. Backs up before destructive operations."""
    import shutil
    import tarfile
    import yaml
    from datetime import datetime

    config = ctx.obj["config"]
    memory_path = config.memory_path

    # --full implies --all
    if full:
        clean_all = True

    # Determine what to clean
    if not clean_all and not artifacts and not full and not reset_chats:
        console.print("[yellow]Specify --all, --artifacts, --full, or --chats. Use --dry-run to preview.[/yellow]")
        return

    targets: list[tuple[Path, str]] = []

    # Artifact targets (always included with --artifacts or --all)
    if clean_all or artifacts:
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

    if targets:
        # Preview
        console.print(f"[bold]{'[DRY RUN] ' if dry_run else ''}Files to remove:[/bold]")
        for path, desc in targets:
            console.print(f"  {desc}: {path}")

    if targets and not dry_run:
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
    elif targets and dry_run:
        console.print(f"\n[dim]{len(targets)} item(s) would be removed.[/dim]")

    # Chat reset (for --full or --chats)
    if full or reset_chats:
        chats_dir = memory_path / "chats"
        reset_count = 0
        if chats_dir.exists():
            for chat_file in sorted(chats_dir.glob("*.md")):
                try:
                    text = chat_file.read_text(encoding="utf-8")
                    if "processed: true" in text or "processed: True" in text:
                        if dry_run:
                            console.print(f"  Would reset: {chat_file.name}")
                        else:
                            # Re-parse and rewrite with processed: false
                            from src.core.utils import parse_frontmatter
                            fm_data, body = parse_frontmatter(text)
                            fm_data["processed"] = False
                            # Remove processing metadata
                            for key in ["processed_at", "entities_updated", "entities_created", "fallback", "fallback_reason"]:
                                fm_data.pop(key, None)
                            fm_yaml = yaml.safe_dump(fm_data, default_flow_style=False, allow_unicode=True)
                            chat_file.write_text(f"---\n{fm_yaml}---\n{body}", encoding="utf-8")
                        reset_count += 1
                except Exception:
                    continue

        # Move _processed back to _inbox
        processed_dir = memory_path / "_inbox" / "_processed"
        inbox_moved = 0
        if processed_dir.exists():
            for f in processed_dir.iterdir():
                if dry_run:
                    console.print(f"  Would move back to inbox: {f.name}")
                else:
                    shutil.move(str(f), str(memory_path / "_inbox" / f.name))
                inbox_moved += 1
            if not dry_run and not any(processed_dir.iterdir()):
                processed_dir.rmdir()

        # Remove dream checkpoint and event log
        extra_files = ["_dream_checkpoint.json", "_event_log.jsonl"]
        for fname in extra_files:
            p = memory_path / fname
            if p.exists():
                if dry_run:
                    console.print(f"  Would remove: {fname}")
                else:
                    p.unlink()

        if reset_count or inbox_moved:
            action = "Would reset" if dry_run else "Reset"
            console.print(f"  [green]{action} {reset_count} chat(s), moved {inbox_moved} inbox file(s)[/green]")

    if not targets and not (full or reset_chats):
        console.print("[green]Nothing to clean.[/green]")


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
    from src.pipeline.extractor import extract_from_chat, sanitize_extraction
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
            extraction = extract_from_chat(content, config, config.memory_path)
            extraction = sanitize_extraction(extraction)
            console.print(f"  Extracted {len(extraction.entities)} entities")

            graph = load_graph(config.memory_path)
            resolved = resolve_all(extraction, graph, config=config, memory_path=config.memory_path)

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
        consolidate_facts(config, console, dry_run, min_facts)
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


@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would change without modifying")
@click.option("--step", type=int, default=None, help="Run only step N (1-10)")
@click.option("--resume", is_flag=True, help="Resume from last checkpoint")
@click.option("--reset", is_flag=True, help="Clear checkpoint and restart")
@click.option("--report", "show_report", is_flag=True, help="Show last dream report and exit")
@click.pass_context
def dream(ctx, dry_run, step, resume, reset, show_report):
    """Brain-like memory reorganization: consolidate, prune, discover, rebuild.

    10 steps: load → extract docs → consolidate facts → merge entities
    → discover relations → transitive relations → prune dead → generate summaries
    → rescore → rebuild. LLM coordinator plans which steps to run.
    """
    if show_report:
        config = ctx.obj["config"]
        report_path = config.memory_path / "_dream_report.md"
        if report_path.exists():
            click.echo(report_path.read_text(encoding="utf-8"))
        else:
            click.echo("No dream report found. Run `memory dream` first.")
        return

    from src.pipeline.dream import run_dream

    config = ctx.obj["config"]
    mode = "[DRY RUN] " if dry_run else ""
    step_info = f" (step {step} only)" if step else ""
    console.print(f"\n[bold]{mode}Dream mode{step_info}[/bold]")

    report = run_dream(config, console, dry_run=dry_run, step=step, resume=resume, reset=reset)

    # Summary
    console.print("\n[bold]Dream report:[/bold]")
    console.print(f"  Docs extracted: {report.docs_extracted}")
    console.print(f"  Facts consolidated: {report.facts_consolidated}")
    console.print(f"  Entities merged: {report.entities_merged}")
    console.print(f"  Relations discovered: {report.relations_discovered}")
    console.print(f"  Transitive relations: {report.transitive_relations}")
    console.print(f"  Entities pruned: {report.entities_pruned}")
    console.print(f"  Summaries generated: {report.summaries_generated}")
    if report.errors:
        console.print(f"  [yellow]Errors: {len(report.errors)}[/yellow]")
        for err in report.errors:
            console.print(f"    [dim]{err}[/dim]")

    console.print("\n[bold green]Dream complete.[/bold green]")


@cli.command()
@click.pass_context
def context(ctx):
    """Rebuild _context.md from current graph (no extraction, no LLM)."""
    config = ctx.obj["config"]
    from src.memory.graph import load_graph, save_graph
    from src.memory.context import build_context_for_config, write_context, write_index
    from src.memory.scoring import recalculate_all_scores

    console.print("[bold]Rebuilding context...[/bold]")
    graph = load_graph(config.memory_path)
    graph = recalculate_all_scores(graph, config)
    save_graph(config.memory_path, graph)

    # Context
    context_text = build_context_for_config(graph, config.memory_path, config)
    if context_text.strip():
        write_context(config.memory_path, context_text)
    write_index(config.memory_path, graph)
    console.print("[green]✓ _context.md and _index.md updated[/green]")


@cli.command()
@click.option("--entity", default=None, help="Discover relations for a single entity")
@click.option("--dry-run", is_flag=True, help="Preview without writing")
@click.pass_context
def relations(ctx, entity, dry_run):
    """Discover new relations using FAISS similarity + tag overlap + co-occurrence (zero LLM)."""
    from src.pipeline.orchestrator import discover_relations_deterministic
    from src.core.utils import slugify

    config = ctx.obj["config"]
    entity_filter = slugify(entity) if entity else None

    mode = "[DRY RUN] " if dry_run else ""
    scope = f"for {entity}" if entity else "full scan"
    console.print(f"\n[bold]{mode}Relation discovery ({scope})...[/bold]")

    n = discover_relations_deterministic(config, config.memory_path, console, entity_filter=entity_filter, dry_run=dry_run)

    if dry_run:
        console.print(f"\n[dim]{n} relation(s) would be created[/dim]")
    else:
        console.print(f"\n[bold green]✓ {n} new relation(s) discovered[/bold green]")


@cli.command()
@click.pass_context
def graph(ctx):
    """Open interactive dashboard with graph, timeline, dream replay and search."""
    config = ctx.obj["config"]
    from src.pipeline.dashboard_server import start_server

    start_server(config)


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=5, help="Number of results")
@click.option("--expand/--no-expand", default=True, help="Expand relations")
@click.pass_context
def search(ctx, query, top_k, expand):
    """Search memory via RAG."""
    config = ctx.obj["config"]
    memory_path = config.memory_path
    from src.memory.rag import search as rag_search, SearchOptions
    results = rag_search(query, config, memory_path, SearchOptions(
        top_k=top_k, expand_relations=expand, include_chunk_text=True,
    ))
    if not results:
        console.print("[dim]No results found.[/dim]")
        return
    table = Table(title=f"Results for '{query}'")
    table.add_column("Entity", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Chunk", max_width=60)
    for r in results:
        table.add_row(r.entity_id, f"{r.score:.3f}", (r.chunk or "")[:60])
    console.print(table)


@cli.command()
@click.option("--last", default=20, help="Show last N actions")
@click.option("--entity", default=None, help="Filter by entity name")
@click.option("--action", "action_type", default=None, help="Filter by action type")
@click.pass_context
def actions(ctx, last, entity, action_type):
    """Show centralized action history."""
    import json as json_mod
    from src.core.action_log import read_actions

    config = ctx.obj["config"]
    entries = read_actions(config.memory_path, entity_id=entity, action=action_type, last_n=last)
    if not entries:
        click.echo("No actions found.")
        return
    for e in entries:
        ts = e.get("timestamp", "")[:19]
        act = e.get("action", "")
        eid = e.get("entity_id", "")
        details = e.get("details", {})
        click.echo(f"[{ts}] {act:15s} {eid:25s} {json_mod.dumps(details, ensure_ascii=False)}")


@cli.command()
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
@click.pass_context
def health(ctx, fmt):
    """Analyze memory health and show recommendations."""
    import json as json_mod
    from src.memory.insights import analyze_memory_health
    from src.memory.graph import load_graph

    config = ctx.obj["config"]
    graph = load_graph(config.memory_path)
    result = analyze_memory_health(graph, config)

    if fmt == "json":
        click.echo(json_mod.dumps(result, indent=2, ensure_ascii=False))
        return

    console.print(f"\n[bold]Memory Health Report[/bold]\n")
    console.print(f"[dim]{result['summary']}[/dim]\n")

    # Hot topics
    if result["hot_topics"]:
        table = Table(title="Hot Topics (3+ mentions in 7 days)", title_style="bold red")
        table.add_column("Entity", style="cyan")
        table.add_column("Mentions (7d)", justify="right", style="red")
        for item in result["hot_topics"]:
            table.add_row(item["title"], str(item["mentions_7d"]))
        console.print(table)
        console.print()

    # Stale topics
    if result["stale_topics"]:
        table = Table(title="Stale Topics (60+ days without mention)", title_style="bold yellow")
        table.add_column("Entity", style="cyan")
        table.add_column("Days Since", justify="right", style="yellow")
        for item in result["stale_topics"][:20]:
            table.add_row(item["title"], str(item["days_since"]))
        if len(result["stale_topics"]) > 20:
            console.print(f"  [dim]... and {len(result['stale_topics']) - 20} more[/dim]")
        console.print(table)
        console.print()

    # Orphans
    if result["orphans"]:
        table = Table(title="Orphans (no relations)", title_style="bold blue")
        table.add_column("Entity", style="cyan")
        for item in result["orphans"][:20]:
            table.add_row(item["title"])
        if len(result["orphans"]) > 20:
            console.print(f"  [dim]... and {len(result['orphans']) - 20} more[/dim]")
        console.print(table)
        console.print()

    # Overloaded
    if result["overloaded"]:
        table = Table(title="Overloaded Entities (nearing max facts)", title_style="bold magenta")
        table.add_column("Entity", style="cyan")
        table.add_column("Frequency", justify="right")
        table.add_column("Max Facts", justify="right")
        for item in result["overloaded"]:
            table.add_row(item["title"], str(item["frequency"]), str(item["max_facts"]))
        console.print(table)
        console.print()

    # Recommendations
    recs = []
    if len(result["overloaded"]) > 3:
        recs.append("Run [bold]memory dream[/bold] or [bold]memory consolidate --facts[/bold] to consolidate overloaded entities")
    if len(result["orphans"]) > 5:
        recs.append("Run [bold]memory relations[/bold] or [bold]memory dream[/bold] to discover relations for orphans")
    if len(result["stale_topics"]) > 10:
        recs.append("Run [bold]memory dream[/bold] to prune stale entities")
    if recs:
        console.print("[bold]Recommendations:[/bold]")
        for r in recs:
            console.print(f"  - {r}")
        console.print()

    if not any([result["hot_topics"], result["stale_topics"], result["orphans"], result["overloaded"]]):
        console.print("[green]Memory is healthy — no issues detected.[/green]")


@cli.command()
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
@click.pass_context
def insights(ctx, fmt):
    """Show ACT-R cognitive insights about memory state."""
    import json as json_mod
    from src.memory.insights import compute_insights
    from src.memory.graph import load_graph

    config = ctx.obj["config"]
    graph = load_graph(config.memory_path)
    result = compute_insights(graph)

    if fmt == "json":
        click.echo(json_mod.dumps(result, indent=2, ensure_ascii=False))
    else:
        click.echo(f"Entities: {result['total_entities']}  Relations: {result['total_relations']}")
        click.echo(f"\nScore distribution:")
        for bucket, count in result["scoring_distribution"].items():
            bar = "█" * count
            click.echo(f"  {bucket:8s} | {bar} ({count})")
        if result["forgetting_curve"]:
            click.echo(f"\nForgetting curve ({len(result['forgetting_curve'])} entities near threshold):")
            for e in result["forgetting_curve"][:10]:
                click.echo(f"  {e['title']:30s} score={e['score']}")
        if result["emotional_hotspots"]:
            click.echo(f"\nEmotional hotspots:")
            for e in result["emotional_hotspots"][:10]:
                click.echo(f"  {e['title']:30s} valence_ratio={e['ratio']}")
        if result["weak_relations"]:
            click.echo(f"\nWeak relations ({len(result['weak_relations'])}):")
            for r in result["weak_relations"][:10]:
                click.echo(f"  {r['from']} -> {r['type']} -> {r['to']}  strength={r['strength']}")
        if result["network_hubs"]:
            click.echo(f"\nNetwork hubs:")
            for h in result["network_hubs"]:
                click.echo(f"  {h['title']:30s} degree={h['degree']}")


if __name__ == "__main__":
    cli()
