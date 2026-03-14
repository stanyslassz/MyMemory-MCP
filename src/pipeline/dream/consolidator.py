"""Dream steps 2, 3, 8: Document extraction, fact consolidation, summary generation."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from src.core.config import Config
from src.core.models import GraphData


def _step_extract_documents(
    graph: GraphData,
    memory_path: Path,
    config: Config,
    console: Console,
    report,
    dry_run: bool,
) -> int:
    """Step 2: Extract entities from unprocessed RAG documents."""
    from src.pipeline.indexer import list_unextracted_docs, mark_doc_extracted
    from src.pipeline.extractor import extract_from_chat, sanitize_extraction
    from src.pipeline.resolver import resolve_all
    from src.pipeline.enricher import enrich_memory
    # Config import already available via function parameter
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
            extraction = extract_from_chat(text, config, memory_path)
            extraction = sanitize_extraction(extraction)

            if extraction.entities:
                resolved = resolve_all(extraction, graph, config=config, memory_path=memory_path)
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
    report,
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


def _step_generate_summaries(
    graph: GraphData,
    entity_paths: dict[str, Path],
    config: Config,
    console: Console,
    report,
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
