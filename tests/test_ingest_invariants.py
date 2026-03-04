"""Invariant tests for P1.1: idempotency, dedup, crash recovery, job state machine."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.config import Config, FeaturesConfig, IngestConfig, EmbeddingsConfig, FAISSConfig, ScoringConfig
from src.core.models import IngestKey, IngestJob, CHUNK_POLICY_VERSION
from src.pipeline.ingest_state import (
    compute_ingest_key,
    create_job,
    get_job,
    has_been_ingested,
    list_jobs,
    recover_stale_jobs,
    transition_job,
)


def _make_config(tmp_path) -> Config:
    config = Config.__new__(Config)
    config.memory_path = tmp_path
    config.features = FeaturesConfig(doc_pipeline=True)
    config.ingest = IngestConfig(
        recovery_threshold_seconds=300,
        max_retries=3,
        jobs_path=str(tmp_path / "_ingest_jobs.json"),
    )
    config.embeddings = EmbeddingsConfig()
    config.faiss = FAISSConfig(
        index_path=str(tmp_path / "_memory.faiss"),
        mapping_path=str(tmp_path / "_memory.pkl"),
        manifest_path=str(tmp_path / "_faiss_manifest.json"),
        top_k=5,
    )
    config.scoring = ScoringConfig()
    return config


class TestIngestKey:
    def test_compute_key(self):
        key = compute_ingest_key("file.md", "hello world")
        assert key.source_id == "file.md"
        assert len(key.content_hash) == 64  # SHA256
        assert key.chunk_policy_version == CHUNK_POLICY_VERSION

    def test_same_content_same_hash(self):
        k1 = compute_ingest_key("a.md", "content")
        k2 = compute_ingest_key("a.md", "content")
        assert k1.content_hash == k2.content_hash
        assert k1.canonical == k2.canonical

    def test_different_content_different_hash(self):
        k1 = compute_ingest_key("a.md", "content1")
        k2 = compute_ingest_key("a.md", "content2")
        assert k1.content_hash != k2.content_hash

    def test_different_source_different_canonical(self):
        k1 = compute_ingest_key("a.md", "content")
        k2 = compute_ingest_key("b.md", "content")
        assert k1.canonical != k2.canonical


class TestJobStateMachine:
    def test_create_job(self, tmp_path):
        config = _make_config(tmp_path)
        key = compute_ingest_key("test.md", "content")
        job = create_job(key, config, route="document")

        assert job.status == "pending"
        assert job.retries == 0
        assert job.route == "document"
        assert job.ingest_key.source_id == "test.md"

    def test_valid_transitions(self, tmp_path):
        config = _make_config(tmp_path)
        key = compute_ingest_key("test.md", "content")
        job = create_job(key, config)

        # pending → running
        job = transition_job(job.job_id, "running", config)
        assert job.status == "running"

        # running → succeeded
        job = transition_job(job.job_id, "succeeded", config, chunks_indexed=5)
        assert job.status == "succeeded"
        assert job.chunks_indexed == 5

    def test_invalid_transition_rejected(self, tmp_path):
        config = _make_config(tmp_path)
        key = compute_ingest_key("test.md", "content")
        job = create_job(key, config)

        # pending → succeeded (invalid, must go through running)
        result = transition_job(job.job_id, "succeeded", config)
        assert result is None

    def test_running_to_failed(self, tmp_path):
        config = _make_config(tmp_path)
        key = compute_ingest_key("test.md", "content")
        job = create_job(key, config)
        transition_job(job.job_id, "running", config)
        job = transition_job(job.job_id, "failed", config, error="timeout")
        assert job.status == "failed"
        assert job.error == "timeout"

    def test_running_to_retriable(self, tmp_path):
        config = _make_config(tmp_path)
        key = compute_ingest_key("test.md", "content")
        job = create_job(key, config)
        transition_job(job.job_id, "running", config)
        job = transition_job(job.job_id, "retriable", config, error="transient error")
        assert job.status == "retriable"
        assert job.retries == 1

    def test_max_retries_escalates_to_failed(self, tmp_path):
        config = _make_config(tmp_path)
        key = compute_ingest_key("test.md", "content")
        job = create_job(key, config)

        for i in range(3):
            transition_job(job.job_id, "running", config)
            job = transition_job(job.job_id, "retriable", config, error=f"fail {i}")

        # After 3 retries (max_retries=3), should be failed
        assert job.status == "failed"
        assert "max retries" in job.error

    def test_get_job(self, tmp_path):
        config = _make_config(tmp_path)
        key = compute_ingest_key("test.md", "content")
        job = create_job(key, config)
        retrieved = get_job(job.job_id, config)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    def test_get_nonexistent_job(self, tmp_path):
        config = _make_config(tmp_path)
        assert get_job("nonexistent", config) is None

    def test_list_jobs_by_status(self, tmp_path):
        config = _make_config(tmp_path)
        k1 = compute_ingest_key("a.md", "content1")
        k2 = compute_ingest_key("b.md", "content2")
        j1 = create_job(k1, config)
        j2 = create_job(k2, config)
        transition_job(j1.job_id, "running", config)

        pending = list_jobs(config, status="pending")
        running = list_jobs(config, status="running")
        assert len(pending) == 1
        assert len(running) == 1


class TestDuplicateSubmit:
    """Invariant: duplicate submit → single indexed artifact."""

    def test_duplicate_submit_returns_single_artifact(self, tmp_path):
        config = _make_config(tmp_path)
        content = "# Important Document\n\nThis is test content for dedup."

        # Submit 3 times with same content
        for i in range(3):
            key = compute_ingest_key("same_file.md", content)
            if i == 0:
                assert not has_been_ingested(key, config)
                job = create_job(key, config)
                transition_job(job.job_id, "running", config)
                transition_job(job.job_id, "succeeded", config, chunks_indexed=2)
            else:
                # Subsequent submits should be detected as duplicates
                assert has_been_ingested(key, config)

        # Only one succeeded job should exist for this key
        all_jobs = list_jobs(config)
        succeeded_for_key = [
            j for j in all_jobs
            if j.ingest_key.canonical == key.canonical and j.status == "succeeded"
        ]
        assert len(succeeded_for_key) == 1


class TestRetryReplay:
    """Invariant: retry replay → no duplicate visible results."""

    def test_retry_replay_no_duplicates(self, tmp_path):
        config = _make_config(tmp_path)
        content = "Retry test content"
        key = compute_ingest_key("retry.md", content)

        # First attempt: create, run, fail
        job = create_job(key, config)
        transition_job(job.job_id, "running", config)
        transition_job(job.job_id, "retriable", config, error="transient")

        # Retry: run again, succeed
        transition_job(job.job_id, "running", config)
        transition_job(job.job_id, "succeeded", config, chunks_indexed=1)

        # Should have exactly one job (not duplicated)
        all_jobs = list_jobs(config)
        jobs_for_key = [
            j for j in all_jobs
            if j.ingest_key.canonical == key.canonical
        ]
        assert len(jobs_for_key) == 1
        assert jobs_for_key[0].status == "succeeded"
        assert jobs_for_key[0].retries == 1


class TestCrashRecovery:
    """Invariant: crash recovery → no stuck 'running' beyond threshold."""

    def test_stale_running_jobs_recovered(self, tmp_path):
        config = _make_config(tmp_path)
        config.ingest.recovery_threshold_seconds = 1  # 1 second for test

        key = compute_ingest_key("crash.md", "crash content")
        job = create_job(key, config)
        transition_job(job.job_id, "running", config)

        # Simulate time passing
        jobs_path = Path(config.ingest.jobs_path)
        jobs_data = json.loads(jobs_path.read_text())
        old_time = (datetime.now() - timedelta(seconds=10)).isoformat()
        jobs_data[job.job_id]["updated"] = old_time
        jobs_path.write_text(json.dumps(jobs_data))

        # Recover
        recovered = recover_stale_jobs(config)
        assert job.job_id in recovered

        # Job should no longer be running
        recovered_job = get_job(job.job_id, config)
        assert recovered_job.status != "running"

    def test_no_false_recovery(self, tmp_path):
        config = _make_config(tmp_path)
        config.ingest.recovery_threshold_seconds = 600  # 10 minutes

        key = compute_ingest_key("recent.md", "recent content")
        job = create_job(key, config)
        transition_job(job.job_id, "running", config)

        # Should not recover a job that was just updated
        recovered = recover_stale_jobs(config)
        assert len(recovered) == 0

        current_job = get_job(job.job_id, config)
        assert current_job.status == "running"
