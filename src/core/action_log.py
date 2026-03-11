"""Centralized action log for memory operations."""

import json
from datetime import datetime
from pathlib import Path


def log_action(
    memory_path: Path,
    action: str,
    entity_id: str = "",
    details: dict = None,
    source: str = "pipeline",
) -> None:
    """Append action to centralized log (_actions.jsonl)."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "entity_id": entity_id,
        "source": source,
        "details": details or {},
    }
    log_path = memory_path / "_actions.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_actions(
    memory_path: Path,
    entity_id: str = None,
    action: str = None,
    last_n: int = 0,
) -> list[dict]:
    """Read and filter action log entries."""
    log_path = memory_path / "_actions.jsonl"
    if not log_path.exists():
        return []
    entries = []
    for line in log_path.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entity_id and entry.get("entity_id") != entity_id:
            continue
        if action and entry.get("action") != action:
            continue
        entries.append(entry)
    if last_n > 0:
        entries = entries[-last_n:]
    return entries
