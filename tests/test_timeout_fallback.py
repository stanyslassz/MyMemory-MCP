"""Tests for extraction timeout → doc_ingest fallback and non-endless-retry behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.core.config import load_config
from src.memory.store import (
    init_memory_structure,
    save_chat,
    list_unprocessed_chats,
    mark_chat_fallback,
    increment_extraction_retries,
    get_chat_content,
)


def _make_config(tmp_path):
    """Create test config pointing to tmp_path."""
    config = load_config(project_root=Path(__file__).parent.parent)
    config.memory_path = tmp_path
    config.faiss.index_path = str(tmp_path / "_memory.faiss")
    config.faiss.mapping_path = str(tmp_path / "_memory.pkl")
    config.faiss.manifest_path = str(tmp_path / "_faiss_manifest.json")
    config.ingest.jobs_path = str(tmp_path / "_ingest_jobs.json")
    config.features.doc_pipeline = True
    return config


def _run_cli(config, catch_exceptions=True):
    """Invoke `memory run` with patched config."""
    from click.testing import CliRunner
    from src.cli import cli

    runner = CliRunner()
    with patch("src.cli.load_config", return_value=config):
        return runner.invoke(cli, ["run"], catch_exceptions=catch_exceptions)


# ── mark_chat_fallback ────────────────────────────────────────


def test_mark_chat_fallback_sets_processed(tmp_path):
    """mark_chat_fallback marks the chat as processed with fallback metadata."""
    init_memory_structure(tmp_path)
    msgs = [{"role": "user", "content": "Some big doc content"}]
    chat_path = save_chat(msgs, tmp_path)

    assert len(list_unprocessed_chats(tmp_path)) == 1

    mark_chat_fallback(chat_path, fallback="doc_ingest", error="timeout after 60s")

    # Now processed
    assert len(list_unprocessed_chats(tmp_path)) == 0

    # Verify frontmatter metadata
    import yaml, re
    text = chat_path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?\n)---\n", text, re.DOTALL)
    fm = yaml.safe_load(match.group(1))
    assert fm["processed"] is True
    assert fm["fallback"] == "doc_ingest"
    assert fm["fallback_reason"] == "timeout after 60s"


# ── increment_extraction_retries ──────────────────────────────


def test_extraction_retries_increment(tmp_path):
    """extraction_retries increments on each call and persists."""
    init_memory_structure(tmp_path)
    msgs = [{"role": "user", "content": "test"}]
    chat_path = save_chat(msgs, tmp_path)

    assert increment_extraction_retries(chat_path) == 1
    assert increment_extraction_retries(chat_path) == 2
    assert increment_extraction_retries(chat_path) == 3

    # Still unprocessed (retries alone don't mark processed)
    assert len(list_unprocessed_chats(tmp_path)) == 1


# ── _is_timeout_error ─────────────────────────────────────────


def test_is_timeout_error_detection():
    """Various timeout-like exceptions are detected."""
    from src.pipeline.orchestrator import is_timeout_error as _is_timeout_error

    assert _is_timeout_error(TimeoutError("connection timed out"))
    assert _is_timeout_error(Exception("ReadTimeout: server did not respond"))
    assert _is_timeout_error(Exception("Request timeout after 60s"))
    assert not _is_timeout_error(Exception("Invalid JSON response"))
    assert not _is_timeout_error(ValueError("bad input"))


# ── Full fallback flow (integration via CLI) ──────────────────


def test_timeout_triggers_immediate_doc_ingest_fallback(tmp_path):
    """When extraction raises a timeout, chat is immediately doc-ingested and marked processed."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    msgs = [{"role": "user", "content": "A very large document pretending to be a chat."}]
    chat_path = save_chat(msgs, tmp_path)

    assert len(list_unprocessed_chats(tmp_path)) == 1

    with patch("src.pipeline.extractor.extract_from_chat", side_effect=TimeoutError("LLM timeout after 60s")), \
         patch("src.pipeline.doc_ingest.ingest_document", return_value={"chunks_indexed": 5, "source_id": "test"}):
        result = _run_cli(config)

    assert "falling back to doc_ingest" in result.output.lower() or "fallback" in result.output.lower()

    # Chat should now be processed
    assert len(list_unprocessed_chats(tmp_path)) == 0

    # Verify fallback metadata
    import yaml, re
    text = chat_path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?\n)---\n", text, re.DOTALL)
    fm = yaml.safe_load(match.group(1))
    assert fm["processed"] is True
    assert fm["fallback"] == "doc_ingest"
    assert "timeout" in fm.get("fallback_reason", "").lower()


def test_repeated_non_timeout_failures_trigger_fallback(tmp_path):
    """After EXTRACTION_MAX_RETRIES non-timeout failures, fallback kicks in."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    msgs = [{"role": "user", "content": "Content that consistently fails extraction."}]
    chat_path = save_chat(msgs, tmp_path)

    # First run: extraction fails, retry counter = 1 (< 2), no fallback yet
    with patch("src.pipeline.extractor.extract_from_chat", side_effect=Exception("JSON parse error")):
        _run_cli(config)

    # Still unprocessed after first failure
    assert len(list_unprocessed_chats(tmp_path)) == 1

    # Second run: extraction fails again, retry counter = 2 (>= 2), fallback triggers
    with patch("src.pipeline.extractor.extract_from_chat", side_effect=Exception("JSON parse error")), \
         patch("src.pipeline.doc_ingest.ingest_document", return_value={"chunks_indexed": 3, "source_id": "test"}):
        _run_cli(config)

    # Now processed via fallback
    assert len(list_unprocessed_chats(tmp_path)) == 0


def test_fallback_not_triggered_on_first_non_timeout_failure(tmp_path):
    """A single non-timeout failure does NOT trigger fallback — allows retry."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    msgs = [{"role": "user", "content": "Transient failure content."}]
    save_chat(msgs, tmp_path)

    with patch("src.pipeline.extractor.extract_from_chat", side_effect=Exception("temporary LLM glitch")):
        _run_cli(config)

    # Should still be unprocessed — gets another chance next run
    assert len(list_unprocessed_chats(tmp_path)) == 1


def test_fallback_marks_processed_even_when_doc_ingest_fails(tmp_path):
    """Even if doc_ingest itself fails, chat is marked processed to prevent endless retries."""
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    msgs = [{"role": "user", "content": "Problematic content."}]
    save_chat(msgs, tmp_path)

    with patch("src.pipeline.extractor.extract_from_chat", side_effect=TimeoutError("timeout")), \
         patch("src.pipeline.doc_ingest.ingest_document", side_effect=Exception("embedding server down")):
        _run_cli(config)

    # Must be marked processed regardless — no endless retry
    assert len(list_unprocessed_chats(tmp_path)) == 0
