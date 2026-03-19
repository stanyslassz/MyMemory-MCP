"""Tests for dream report generation."""

from datetime import datetime
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

from rich.console import Console

from src.pipeline.dream import run_dream, _generate_dream_report
from src.core.models import GraphData
from src.memory.event_log import append_event, read_events


def _make_config(tmp_path: Path):
    config = MagicMock()
    memory_path = tmp_path / "memory"
    memory_path.mkdir()
    (memory_path / "chats").mkdir()
    (memory_path / "_inbox").mkdir()
    config.memory_path = memory_path
    config.dream = MagicMock()
    config.dream.prune_score_threshold = 0.1
    config.dream.prune_max_frequency = 1
    config.dream.prune_min_age_days = 90
    config.scoring = MagicMock()
    config.scoring.relation_strength_growth = 0.05
    return config


def test_dream_generates_report_file(tmp_path):
    """run_dream should generate _dream_report.md after completion."""
    config = _make_config(tmp_path)
    console = Console(file=StringIO())
    empty_graph = GraphData(generated=datetime.now().isoformat(), entities={}, relations=[])

    with patch("src.pipeline.dream.maintenance._step_load", return_value=(empty_graph, {})):
        run_dream(config, console, dry_run=True, step=1)

    report_path = config.memory_path / "_dream_report.md"
    assert report_path.exists()
    content = report_path.read_text()
    assert "# Dream Report" in content
    assert "Load" in content


def test_generate_dream_report_format(tmp_path):
    """Report should have session header, steps table, and details sections."""
    memory_path = tmp_path / "memory"
    memory_path.mkdir()

    dream_id = "2026-03-15T14:00:00"
    ts_before = "2026-03-15T13:59:59"

    append_event(memory_path, "dream_session_started", "dream", {
        "dream_id": dream_id, "steps_planned": [1, 9, 10],
        "resumed": False, "entity_count": 42, "relation_count": 10,
    })
    append_event(memory_path, "dream_step_completed", "dream", {
        "dream_id": dream_id, "step": 1, "step_name": "Load",
        "duration_s": 0.2, "summary": "42 entities, 10 relations",
        "details": {"entities": 42, "relations": 10},
    })
    append_event(memory_path, "dream_step_skipped", "dream", {
        "dream_id": dream_id, "step": 2, "step_name": "Extract docs",
    })
    append_event(memory_path, "dream_session_completed", "dream", {
        "dream_id": dream_id, "duration_s": 1.5,
        "steps_completed": 1, "steps_failed": 0,
    })

    report_path = _generate_dream_report(memory_path, dream_id, ts_before)

    assert report_path.exists()
    content = report_path.read_text()
    assert "# Dream Report" in content
    assert "2026-03-15" in content
    assert "Load" in content
    assert "42" in content
    assert "skipped" in content
