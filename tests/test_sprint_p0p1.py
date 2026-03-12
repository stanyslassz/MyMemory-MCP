"""Tests for Sprint P0+P1 features: route fix, stall detection, clean, ledger, replay."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.config import load_config
from src.core.models import IngestJob, IngestKey, RouteType


# ── Route validation fix ────────────────────────────────────

class TestRouteValidation:
    def test_fallback_doc_ingest_is_valid_route(self):
        """fallback_doc_ingest must be accepted by IngestJob.route."""
        job = IngestJob(
            job_id="test",
            ingest_key=IngestKey(source_id="x", content_hash="y"),
            route="fallback_doc_ingest",
        )
        assert job.route == "fallback_doc_ingest"

    def test_standard_routes_still_valid(self):
        for route in ("conversation", "document", "uncertain"):
            job = IngestJob(
                job_id="t",
                ingest_key=IngestKey(source_id="x", content_hash="y"),
                route=route,
            )
            assert job.route == route

    def test_invalid_route_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            IngestJob(
                job_id="t",
                ingest_key=IngestKey(source_id="x", content_hash="y"),
                route="bogus",
            )


# ── Stall detection ──────────────────────────────────────────

class TestStallDetection:
    def test_stall_error_is_timeout(self):
        from src.core.llm import StallError
        assert issubclass(StallError, TimeoutError)

    def test_is_timeout_recognizes_stall_error(self):
        from src.pipeline.orchestrator import is_timeout_error as _is_timeout_error
        from src.core.llm import StallError
        assert _is_timeout_error(StallError("stalled"))

    def test_is_timeout_recognizes_standard_timeout(self):
        from src.pipeline.orchestrator import is_timeout_error as _is_timeout_error
        assert _is_timeout_error(TimeoutError("connection timed out"))
        assert _is_timeout_error(Exception("ReadTimeout"))

    def test_is_timeout_rejects_unrelated(self):
        from src.pipeline.orchestrator import is_timeout_error as _is_timeout_error
        assert not _is_timeout_error(ValueError("bad value"))


# ── Clean command ────────────────────────────────────────────

class TestCleanCommand:
    def test_clean_dry_run_no_deletions(self, tmp_path):
        from click.testing import CliRunner
        from src.cli import cli

        # Create some artifacts
        (tmp_path / "_context.md").write_text("ctx")
        (tmp_path / "_index.md").write_text("idx")

        result = CliRunner().invoke(cli, [
            "--config", str(Path(__file__).parent.parent / "config.yaml"),
            "clean", "--artifacts", "--dry-run",
        ])
        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "would be removed" in result.output

    def test_clean_no_flags_shows_help(self, tmp_path):
        from click.testing import CliRunner
        from src.cli import cli

        result = CliRunner().invoke(cli, [
            "--config", str(Path(__file__).parent.parent / "config.yaml"),
            "clean",
        ])
        assert result.exit_code == 0
        assert "Specify --all" in result.output


# ── Retry ledger ─────────────────────────────────────────────

class TestRetryLedger:
    def _make_config(self, tmp_path):
        config = load_config(project_root=Path(__file__).parent.parent)
        config.memory_path = tmp_path
        return config

    def test_record_and_list(self, tmp_path):
        from src.pipeline.ingest_state import record_failure, list_retriable
        config = self._make_config(tmp_path)

        record_failure(tmp_path / "chat1.md", "timeout", config)
        record_failure(tmp_path / "chat2.md", "stall", config)

        entries = list_retriable(config)
        assert len(entries) == 2
        assert entries[0]["file"] == str(tmp_path / "chat1.md")

    def test_no_duplicate_recording(self, tmp_path):
        from src.pipeline.ingest_state import record_failure, list_retriable
        config = self._make_config(tmp_path)

        record_failure(tmp_path / "chat1.md", "timeout", config)
        record_failure(tmp_path / "chat1.md", "timeout again", config)

        entries = list_retriable(config)
        assert len(entries) == 1

    def test_mark_replayed_success(self, tmp_path):
        from src.pipeline.ingest_state import record_failure, list_retriable, mark_replayed
        config = self._make_config(tmp_path)

        record_failure(tmp_path / "chat1.md", "err", config)
        mark_replayed(str(tmp_path / "chat1.md"), success=True, config=config)

        pending = list_retriable(config)
        assert len(pending) == 0

    def test_mark_replayed_exhausted_after_3(self, tmp_path):
        from src.pipeline.ingest_state import record_failure, list_retriable, mark_replayed, _load_ledger
        config = self._make_config(tmp_path)

        record_failure(tmp_path / "chat1.md", "err", config)
        for _ in range(3):
            mark_replayed(str(tmp_path / "chat1.md"), success=False, config=config, error="still failing")

        entries = _load_ledger(config)
        assert entries[0]["status"] == "exhausted"


# ── Replay CLI ───────────────────────────────────────────────

class TestReplayCLI:
    def test_replay_list_empty(self):
        from click.testing import CliRunner
        from src.cli import cli

        result = CliRunner().invoke(cli, [
            "--config", str(Path(__file__).parent.parent / "config.yaml"),
            "replay", "--list",
        ])
        assert result.exit_code == 0
        assert "No retriable failures" in result.output


# ── Transport defaults ───────────────────────────────────────

class TestTransportDefaults:
    def test_config_default_is_stdio(self):
        config = load_config(project_root=Path(__file__).parent.parent)
        assert config.mcp_transport == "stdio"
