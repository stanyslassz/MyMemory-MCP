"""Tests for dashboard dream session extraction."""

from src.pipeline.dashboard import _extract_dream_sessions


def test_extract_dream_sessions_groups_by_dream_id():
    """Sessions should be grouped by dream_id, not by calendar day."""
    events = [
        {"type": "dream_session_started", "ts": "2026-03-15T10:00:00", "data": {"dream_id": "2026-03-15T10:00:00", "steps_planned": [1, 9, 10]}},
        {"type": "dream_step_completed", "ts": "2026-03-15T10:00:01", "data": {"dream_id": "2026-03-15T10:00:00", "step": 1, "step_name": "Load", "duration_s": 0.2, "summary": "ok"}},
        {"type": "dream_session_completed", "ts": "2026-03-15T10:00:05", "data": {"dream_id": "2026-03-15T10:00:00", "duration_s": 5.0, "steps_completed": 1}},
        {"type": "dream_session_started", "ts": "2026-03-15T14:00:00", "data": {"dream_id": "2026-03-15T14:00:00", "steps_planned": [1, 3]}},
        {"type": "dream_step_completed", "ts": "2026-03-15T14:00:01", "data": {"dream_id": "2026-03-15T14:00:00", "step": 1, "step_name": "Load", "duration_s": 0.1, "summary": "ok"}},
        {"type": "dream_session_completed", "ts": "2026-03-15T14:00:10", "data": {"dream_id": "2026-03-15T14:00:00", "duration_s": 10.0, "steps_completed": 1}},
    ]
    sessions = _extract_dream_sessions(events)
    assert len(sessions) == 2


def test_extract_dream_sessions_includes_session_metadata():
    """Session should have duration and step counts from session_completed event."""
    events = [
        {"type": "dream_session_started", "ts": "2026-03-15T10:00:00", "data": {"dream_id": "d1", "steps_planned": [1]}},
        {"type": "dream_step_completed", "ts": "2026-03-15T10:00:01", "data": {"dream_id": "d1", "step": 1, "step_name": "Load"}},
        {"type": "dream_session_completed", "ts": "2026-03-15T10:00:05", "data": {"dream_id": "d1", "duration_s": 5.0, "steps_completed": 1, "steps_failed": 0}},
    ]
    sessions = _extract_dream_sessions(events)
    assert len(sessions) == 1
    assert sessions[0]["duration_s"] == 5.0
    assert sessions[0]["steps_completed"] == 1


def test_extract_dream_sessions_has_date_label():
    """Each session should have a date label for display."""
    events = [
        {"type": "dream_step_completed", "ts": "2026-03-15T10:00:01", "data": {"dream_id": "2026-03-15T10:00:00", "step": 1, "step_name": "Load"}},
    ]
    sessions = _extract_dream_sessions(events)
    assert len(sessions) == 1
    assert sessions[0]["date"] == "2026-03-15"
