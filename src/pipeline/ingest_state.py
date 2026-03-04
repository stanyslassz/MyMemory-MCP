"""Ingest job state machine: idempotent ingest keys, upsert guard, job lifecycle.

Also includes the retry ledger for tracking extraction failures and enabling replay.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

from src.core.config import Config
from src.core.models import CHUNK_POLICY_VERSION, IngestJob, IngestKey

logger = logging.getLogger(__name__)

# ── Retry ledger ─────────────────────────────────────────────

LEDGER_FILENAME = "_retry_ledger.json"


def _ledger_path(config: Config) -> Path:
    return config.memory_path / LEDGER_FILENAME


def _load_ledger(config: Config) -> list[dict]:
    path = _ledger_path(config)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_ledger(config: Config, entries: list[dict]) -> None:
    path = _ledger_path(config)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def record_failure(chat_path: Path, error: str, config: Config) -> None:
    """Record a failed extraction in the retry ledger."""
    entries = _load_ledger(config)
    # Avoid duplicate entries for the same file
    existing = {e["file"] for e in entries if e.get("status") == "pending"}
    fname = str(chat_path)
    if fname in existing:
        return
    entries.append({
        "file": fname,
        "error": error,
        "status": "pending",
        "attempts": 1,
        "recorded": datetime.now().isoformat(),
        "last_attempt": datetime.now().isoformat(),
    })
    _save_ledger(config, entries)


def list_retriable(config: Config) -> list[dict]:
    """List all pending retry-ledger entries."""
    return [e for e in _load_ledger(config) if e.get("status") == "pending"]


def mark_replayed(chat_path: str, success: bool, config: Config, error: str | None = None) -> None:
    """Update a ledger entry after replay attempt."""
    entries = _load_ledger(config)
    for entry in entries:
        if entry["file"] == chat_path:
            entry["attempts"] = entry.get("attempts", 0) + 1
            entry["last_attempt"] = datetime.now().isoformat()
            if success:
                entry["status"] = "succeeded"
            elif entry["attempts"] >= 3:
                entry["status"] = "exhausted"
                entry["error"] = error or entry.get("error", "")
            else:
                entry["error"] = error or entry.get("error", "")
            break
    _save_ledger(config, entries)


def compute_ingest_key(source_id: str, content: str) -> IngestKey:
    """Compute an idempotent ingest key from source and content."""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return IngestKey(
        source_id=source_id,
        content_hash=content_hash,
        chunk_policy_version=CHUNK_POLICY_VERSION,
    )


def _load_jobs(jobs_path: str) -> dict[str, dict]:
    """Load jobs index from JSON file."""
    path = Path(jobs_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupt jobs file at %s, starting fresh", jobs_path)
        return {}


def _save_jobs(jobs_path: str, jobs: dict[str, dict]) -> None:
    """Save jobs index to JSON file."""
    path = Path(jobs_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jobs, indent=2, ensure_ascii=False), encoding="utf-8")


def has_been_ingested(key: IngestKey, config: Config) -> bool:
    """Upsert guard: check if content has already been successfully ingested."""
    jobs = _load_jobs(config.ingest.jobs_path)
    canonical = key.canonical
    for job_data in jobs.values():
        ik = job_data.get("ingest_key", {})
        job_canonical = f"{ik.get('source_id', '')}::{ik.get('content_hash', '')}::{ik.get('chunk_policy_version', '')}"
        if job_canonical == canonical and job_data.get("status") == "succeeded":
            return True
    return False


def create_job(key: IngestKey, config: Config, route: str | None = None) -> IngestJob:
    """Create a new ingest job. Returns the job if upsert guard passes."""
    now = datetime.now().isoformat()
    job = IngestJob(
        job_id=str(uuid.uuid4()),
        ingest_key=key,
        status="pending",
        retries=0,
        max_retries=config.ingest.max_retries,
        created=now,
        updated=now,
        route=route,
    )

    jobs = _load_jobs(config.ingest.jobs_path)
    jobs[job.job_id] = job.model_dump()
    _save_jobs(config.ingest.jobs_path, jobs)
    return job


def transition_job(job_id: str, new_status: str, config: Config,
                   error: str | None = None, chunks_indexed: int | None = None) -> IngestJob | None:
    """Transition a job to a new status. Returns updated job or None if not found."""
    jobs = _load_jobs(config.ingest.jobs_path)
    if job_id not in jobs:
        return None

    job_data = jobs[job_id]
    old_status = job_data["status"]

    # Valid transitions
    valid = {
        "pending": {"running"},
        "running": {"succeeded", "failed", "retriable"},
        "retriable": {"running"},
        "failed": set(),  # terminal
        "succeeded": set(),  # terminal
    }

    if new_status not in valid.get(old_status, set()):
        logger.warning("Invalid transition %s → %s for job %s", old_status, new_status, job_id)
        return None

    job_data["status"] = new_status
    job_data["updated"] = datetime.now().isoformat()

    if error is not None:
        job_data["error"] = error

    if chunks_indexed is not None:
        job_data["chunks_indexed"] = chunks_indexed

    if new_status == "retriable":
        job_data["retries"] = job_data.get("retries", 0) + 1
        if job_data["retries"] >= job_data.get("max_retries", 3):
            job_data["status"] = "failed"
            job_data["error"] = f"max retries ({job_data['retries']}) exceeded"

    jobs[job_id] = job_data
    _save_jobs(config.ingest.jobs_path, jobs)
    return IngestJob(**job_data)


def get_job(job_id: str, config: Config) -> IngestJob | None:
    """Get a job by ID."""
    jobs = _load_jobs(config.ingest.jobs_path)
    if job_id not in jobs:
        return None
    return IngestJob(**jobs[job_id])


def list_jobs(config: Config, status: str | None = None) -> list[IngestJob]:
    """List all jobs, optionally filtered by status."""
    jobs = _load_jobs(config.ingest.jobs_path)
    result = []
    for job_data in jobs.values():
        if status is None or job_data.get("status") == status:
            result.append(IngestJob(**job_data))
    return result


def recover_stale_jobs(config: Config) -> list[str]:
    """Find jobs stuck in 'running' beyond recovery threshold and mark retriable.

    Returns list of recovered job IDs.
    """
    jobs = _load_jobs(config.ingest.jobs_path)
    threshold = config.ingest.recovery_threshold_seconds
    now = datetime.now()
    recovered = []

    for job_id, job_data in jobs.items():
        if job_data.get("status") != "running":
            continue

        updated = job_data.get("updated", "")
        if not updated:
            continue

        try:
            job_time = datetime.fromisoformat(updated)
            elapsed = (now - job_time).total_seconds()
            if elapsed > threshold:
                job_data["status"] = "retriable"
                job_data["retries"] = job_data.get("retries", 0) + 1
                job_data["updated"] = now.isoformat()
                job_data["error"] = f"stale running state recovered after {elapsed:.0f}s"

                if job_data["retries"] >= job_data.get("max_retries", 3):
                    job_data["status"] = "failed"
                    job_data["error"] = f"max retries after recovery ({job_data['retries']})"

                recovered.append(job_id)
        except (ValueError, TypeError):
            continue

    if recovered:
        _save_jobs(config.ingest.jobs_path, jobs)

    return recovered
