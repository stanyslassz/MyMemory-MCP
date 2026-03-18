"""Append-only event log for memory-ai pipeline operations.

Events are stored in memory/_event_log.jsonl (one JSON object per line).
Thread-safe via threading.Lock.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def append_event(
    memory_path: Path,
    event_type: str,
    source: str,
    data: dict[str, Any] | None = None,
) -> None:
    """Append a single event to the event log.

    Args:
        memory_path: Path to memory directory.
        event_type: Event type (e.g., 'entity_created', 'chat_ingested').
        source: Module that generated the event (e.g., 'pipeline', 'dream', 'mcp').
        data: Arbitrary event payload.
    """
    event = {
        "ts": datetime.now().isoformat(),
        "type": event_type,
        "source": source,
        "data": data or {},
    }
    line = json.dumps(event, ensure_ascii=False) + "\n"
    log_path = memory_path / "_event_log.jsonl"

    with _lock:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            logger.warning("Failed to write event log: %s", e)


def read_events(
    memory_path: Path,
    *,
    event_type: str | None = None,
    source: str | None = None,
    after: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Read events from the log with optional filters.

    Args:
        memory_path: Path to memory directory.
        event_type: Filter by event type.
        source: Filter by source module.
        after: Only return events after this ISO timestamp.
        limit: Maximum number of events to return.

    Returns:
        List of event dicts, oldest first, capped at limit.
    """
    log_path = memory_path / "_event_log.jsonl"
    if not log_path.exists():
        return []

    events: list[dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event_type and event.get("type") != event_type:
                continue
            if source and event.get("source") != source:
                continue
            if after and event.get("ts", "") <= after:
                continue
            events.append(event)

    return events[-limit:]


