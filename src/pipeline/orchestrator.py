"""Pipeline orchestrator — business logic extracted from cli.py.

All functions accept a `console` parameter (rich.console.Console) for
user-facing output.  No Click dependency here.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("memory-ai")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def is_timeout_error(exc: Exception) -> bool:
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


def make_faiss_fn(config, memory_path):
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


def fallback_to_doc_ingest(
    chat_path, content: str, reason: str, memory_path, config, console,
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


# ------------------------------------------------------------------
# Auto-consolidation
# ------------------------------------------------------------------

def auto_consolidate(memory_path, config, console, min_facts: int = 8) -> None:
    """Auto-consolidate entities with too many facts (called during 'run')."""
    from src.memory.graph import load_graph
    from src.memory.store import read_entity, consolidate_entity_facts

    graph = load_graph(memory_path)
    consolidated_count = 0

    for eid, entity in graph.entities.items():
        entity_path = memory_path / entity.file
        if not entity_path.exists():
            continue
        try:
            _, sections = read_entity(entity_path)
            facts = sections.get("Facts", [])
            live_facts = [f for f in facts if "[superseded]" not in f]
            if len(live_facts) >= min_facts:
                console.print(f"  [cyan]Auto-consolidating {entity.title} ({len(live_facts)} facts)...[/cyan]")
                result = consolidate_entity_facts(entity_path, config)
                if result["changes"]:
                    console.print(f"    [green]{', '.join(result['changes'])}[/green]")
                    consolidated_count += 1
        except Exception as e:
            console.print(f"    [yellow]Consolidation skipped for {entity.title}: {e}[/yellow]")

    if consolidated_count:
        console.print(f"  [green]Consolidated {consolidated_count} entity/ies[/green]")


# ------------------------------------------------------------------
# Fact consolidation (CLI consolidate --facts)
# ------------------------------------------------------------------

def consolidate_facts(config, console, dry_run: bool, min_facts: int) -> None:
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


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run_pipeline(config, console, *, consolidate: bool = True) -> None:
    """Shared pipeline logic for run and run-light."""
    memory_path = config.memory_path

    from src.memory.store import (
        list_unprocessed_chats, get_chat_content, mark_chat_processed,
        mark_chat_fallback, increment_extraction_retries,
    )
    from src.memory.graph import load_graph
    from src.pipeline.extractor import extract_from_chat, sanitize_extraction
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
            extraction = sanitize_extraction(extraction)
            console.print(f"  Extracted {len(extraction.entities)} entities, {len(extraction.relations)} relations")
        except Exception as e:
            is_timeout = is_timeout_error(e)
            retries = increment_extraction_retries(chat_path)
            should_fallback = is_timeout or retries >= EXTRACTION_MAX_RETRIES

            if should_fallback:
                reason = f"timeout: {e}" if is_timeout else f"max retries ({retries}): {e}"
                console.print(f"  [yellow]Extraction failed, falling back to doc_ingest: {reason}[/yellow]")
                fallback_to_doc_ingest(chat_path, content, reason, memory_path, config, console)
                # Record in retry ledger for potential future replay
                from src.pipeline.ingest_state import record_failure
                record_failure(chat_path, reason, config)
            else:
                console.print(f"  [red]Extraction failed (retry {retries}/{EXTRACTION_MAX_RETRIES}): {e}[/red]")
            continue

        # Step 2: Resolve
        graph = load_graph(memory_path)
        resolved = resolve_all(extraction, graph, faiss_search_fn=make_faiss_fn(config, memory_path))

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

    # Step 5b: Auto-consolidate entities with too many facts (skipped in run-light)
    if consolidate:
        try:
            auto_consolidate(memory_path, config, console)
        except Exception as e:
            console.print(f"  [yellow]Auto-consolidation warning: {e}[/yellow]")

    # Step 7: Generate context
    try:
        graph = load_graph(memory_path)
        use_llm = consolidate and getattr(config, "context_llm_sections", False)
        if use_llm:
            console.print("\n[bold]Generating context (LLM per-section)...[/bold]")
            from src.memory.context import build_context_with_llm
            context_text = build_context_with_llm(graph, memory_path, config)
        else:
            console.print("\n[bold]Generating context (deterministic)...[/bold]")
            from src.memory.context import build_context
            context_text = build_context(graph, memory_path, config)
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
