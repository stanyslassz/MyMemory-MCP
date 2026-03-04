"""Process files dropped into _inbox/ folder."""

from __future__ import annotations

import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

from src.core.config import Config
from src.core.metrics import metrics
from src.memory.store import save_chat
from src.pipeline.ingest_state import (
    compute_ingest_key,
    create_job,
    has_been_ingested,
    transition_job,
)
from src.pipeline.router import classify

logger = logging.getLogger(__name__)


def process_inbox(memory_path: Path, config: Config) -> list[str]:
    """Process files in _inbox/ → route → conversation or document path → archive.

    When features.doc_pipeline is enabled:
      - Conversation-classified files → save_chat (unchanged path)
      - Document-classified files → normalize → chunk → embed → index (immediate retrieval)
      - Uncertain → document path (safe default for immediate retrieval)

    When features.doc_pipeline is disabled (legacy):
      - All files → pseudo-chat conversion (existing behavior)

    Returns list of processed file names.
    """
    inbox_path = memory_path / "_inbox"
    processed_path = inbox_path / "_processed"
    processed_path.mkdir(parents=True, exist_ok=True)

    if not inbox_path.exists():
        return []

    processed_files = []

    for filepath in sorted(inbox_path.iterdir()):
        if filepath.is_dir():
            continue
        if filepath.suffix not in (".md", ".txt"):
            continue

        try:
            content = filepath.read_text(encoding="utf-8")
            if not content.strip():
                continue

            if not config.features.doc_pipeline:
                # Legacy path: all files → pseudo-chat
                _legacy_ingest(content, memory_path)
            else:
                # P1.1: Route and split
                _routed_ingest(filepath.name, content, memory_path, config)

            # Move to _processed/
            dest = processed_path / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filepath.name}"
            shutil.move(str(filepath), str(dest))
            processed_files.append(filepath.name)

        except Exception as e:
            logger.error("Error processing inbox file %s: %s", filepath, e)

    return processed_files


def _legacy_ingest(content: str, memory_path: Path) -> None:
    """Legacy path: convert to pseudo-chat format."""
    messages = [{"role": "user", "content": content}]
    save_chat(messages, memory_path)


def _routed_ingest(filename: str, content: str, memory_path: Path, config: Config) -> None:
    """Route content and dispatch to appropriate pipeline."""
    # Compute ingest key for idempotency
    source_id = filename
    key = compute_ingest_key(source_id, content)

    # Upsert guard
    if has_been_ingested(key, config):
        metrics.record_duplicate_skip(source_id)
        logger.info("Skipping duplicate: %s", source_id)
        return

    # Route
    t0 = time.monotonic()
    decision = classify(content, source_filename=filename)
    router_latency = time.monotonic() - t0
    metrics.record_route(decision.route, router_latency)

    logger.info(
        "Routed %s → %s (confidence=%.2f, reasons=%s)",
        filename, decision.route, decision.confidence, decision.reasons,
    )

    if decision.route == "conversation":
        # Conversation path: unchanged
        messages = [{"role": "user", "content": content}]
        save_chat(messages, memory_path)
    else:
        # Document or uncertain → document pipeline (immediate retrieval)
        job = create_job(key, config, route=decision.route)
        metrics.start_timer(f"ingest:{source_id}")
        metrics.start_timer(f"readiness:{source_id}")

        try:
            transition_job(job.job_id, "running", config)

            from src.pipeline.doc_ingest import ingest_document
            result = ingest_document(source_id, content, key, memory_path, config)

            transition_job(
                job.job_id, "succeeded", config,
                chunks_indexed=result.get("chunks_indexed", 0),
            )
            metrics.record_readiness(source_id)

        except Exception as e:
            transition_job(job.job_id, "failed", config, error=str(e))
            metrics.record_ingest_failure(source_id, str(e))
            raise
