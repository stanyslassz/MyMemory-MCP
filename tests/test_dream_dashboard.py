"""Tests for dream mode Rich Live dashboard."""

from io import StringIO
from rich.console import Console

from src.pipeline.dream_dashboard import DreamDashboard, DREAM_STEPS


def test_dashboard_step_lifecycle():
    """Dashboard should track step states: pending -> running -> done."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=80)
    dash = DreamDashboard(console)

    assert all(s["status"] == "pending" for s in dash._steps.values())

    dash.start_step(1)
    assert dash._steps[1]["status"] == "running"

    dash.complete_step(1, "42 entities loaded")
    assert dash._steps[1]["status"] == "done"
    assert dash._steps[1]["summary"] == "42 entities loaded"


def test_dashboard_skip_step():
    """Skipped steps should show skip status."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=80)
    dash = DreamDashboard(console)

    dash.skip_step(3)
    assert dash._steps[3]["status"] == "skipped"


def test_dashboard_fail_step():
    """Failed steps should show error."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=80)
    dash = DreamDashboard(console)

    dash.fail_step(2, "LLM timeout")
    assert dash._steps[2]["status"] == "failed"


def test_dashboard_render():
    """Dashboard render should produce table text."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=80)
    dash = DreamDashboard(console)
    dash.start_step(1)

    table = dash.render()
    assert table is not None


def test_dream_steps_list():
    """All 10 steps should be defined (including transitive relations step)."""
    assert len(DREAM_STEPS) == 10
    assert DREAM_STEPS[1]["name"] == "Load"
    assert DREAM_STEPS[6]["name"] == "Transitive"
