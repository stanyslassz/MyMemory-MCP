"""Tests for dream event logging."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from rich.console import Console
from io import StringIO

from src.pipeline.dream import run_dream
from src.core.config import Config
from src.memory.event_log import read_events


def _make_config(tmp_path: Path):
    """Create a minimal config for testing."""
    memory_path = tmp_path / "memory"
    memory_path.mkdir()
    (memory_path / "chats").mkdir()
    (memory_path / "_inbox").mkdir()
    config = MagicMock(spec=Config)
    config.memory_path = memory_path
    config.dream = MagicMock()
    config.dream.prune_score_threshold = 0.1
    config.dream.prune_max_frequency = 1
    config.dream.prune_min_age_days = 90
    config.scoring = MagicMock()
    config.scoring.relation_strength_growth = 0.05
    return config


def test_dream_emits_session_events(tmp_path):
    """run_dream should emit dream_session_started and dream_session_completed events."""
    config = _make_config(tmp_path)
    console = Console(file=StringIO())

    from src.core.models import GraphData
    empty_graph = GraphData(generated=datetime.now().isoformat(), entities={}, relations=[])

    with patch("src.pipeline.dream.maintenance._step_load", return_value=(empty_graph, {})):
        run_dream(config, console, dry_run=True, step=1)

    events = read_events(config.memory_path, source="dream", limit=10_000)
    types = [e["type"] for e in events]
    assert "dream_session_started" in types
    assert "dream_session_completed" in types

    started = next(e for e in events if e["type"] == "dream_session_started")
    assert "dream_id" in started["data"]
    assert "steps_planned" in started["data"]
    assert started["data"]["entity_count"] == 0


def test_dream_emits_step_events(tmp_path):
    """Each executed step should emit started + completed/skipped events."""
    config = _make_config(tmp_path)
    console = Console(file=StringIO())

    from src.core.models import GraphData
    empty_graph = GraphData(generated=datetime.now().isoformat(), entities={}, relations=[])

    with patch("src.pipeline.dream.maintenance._step_load", return_value=(empty_graph, {})):
        run_dream(config, console, dry_run=True, step=1)

    events = read_events(config.memory_path, source="dream", limit=10_000)
    step_events = [e for e in events if e["type"].startswith("dream_step_")]

    step1_types = [e["type"] for e in step_events if e["data"].get("step") == 1]
    assert "dream_step_started" in step1_types
    assert "dream_step_completed" in step1_types

    skipped = [e for e in step_events if e["type"] == "dream_step_skipped"]
    assert len(skipped) == 9  # steps 2-10


def test_dream_step_completed_has_duration(tmp_path):
    """Completed step events should have duration_s field."""
    config = _make_config(tmp_path)
    console = Console(file=StringIO())

    from src.core.models import GraphData
    empty_graph = GraphData(generated=datetime.now().isoformat(), entities={}, relations=[])

    with patch("src.pipeline.dream.maintenance._step_load", return_value=(empty_graph, {})):
        run_dream(config, console, dry_run=True, step=1)

    events = read_events(config.memory_path, source="dream", limit=10_000)
    completed = [e for e in events if e["type"] == "dream_step_completed"]
    assert len(completed) >= 1
    assert "duration_s" in completed[0]["data"]
    assert isinstance(completed[0]["data"]["duration_s"], (int, float))


def test_dream_step_completed_has_details(tmp_path):
    """Completed step events should have a details dict with step-specific counters."""
    config = _make_config(tmp_path)
    console = Console(file=StringIO())

    from src.core.models import GraphData, GraphEntity
    entity = GraphEntity(
        file="self/test.md", type="person", title="Test",
        score=0.5, importance=0.5, frequency=1,
        last_mentioned="2026-03-15", retention="long_term",
        aliases=[], tags=[], mention_dates=["2026-03-15"],
        monthly_buckets={}, created="2026-03-15", summary="",
        negative_valence_ratio=0.0,
    )
    graph = GraphData(
        generated=datetime.now().isoformat(),
        entities={"test": entity},
        relations=[],
    )

    with patch("src.pipeline.dream.maintenance._step_load", return_value=(graph, {})):
        run_dream(config, console, dry_run=True, step=1)

    events = read_events(config.memory_path, source="dream", limit=10_000)
    completed = [e for e in events if e["type"] == "dream_step_completed" and e["data"].get("step") == 1]
    assert len(completed) == 1
    details = completed[0]["data"]["details"]
    assert "entities" in details
    assert details["entities"] == 1
    assert "relations" in details
    assert details["relations"] == 0
