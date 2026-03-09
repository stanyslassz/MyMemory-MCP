"""Rich Live dashboard for dream mode — real-time step progress."""

from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text


# Step definitions: number → name
DREAM_STEPS = {
    1: {"name": "Load"},
    2: {"name": "Extract docs"},
    3: {"name": "Consolidate"},
    4: {"name": "Merge"},
    5: {"name": "Relations"},
    6: {"name": "Prune"},
    7: {"name": "Summaries"},
    8: {"name": "Rescore"},
    9: {"name": "Rebuild"},
}

_STATUS_STYLE = {
    "pending": ("○", "dim"),
    "running": ("◉", "cyan bold"),
    "done": ("✓", "green"),
    "skipped": ("⊘", "dim"),
    "failed": ("✗", "red bold"),
}


class DreamDashboard:
    """Tracks and renders dream step progress."""

    def __init__(self, console: Console):
        self._console = console
        self._steps: dict[int, dict] = {
            n: {"status": "pending", "summary": ""}
            for n in DREAM_STEPS
        }
        self._live: Live | None = None

    def start_step(self, step: int, detail: str = "") -> None:
        self._steps[step]["status"] = "running"
        self._steps[step]["summary"] = detail
        self._refresh()

    def complete_step(self, step: int, summary: str = "") -> None:
        self._steps[step]["status"] = "done"
        self._steps[step]["summary"] = summary
        self._refresh()

    def skip_step(self, step: int, reason: str = "skipped by plan") -> None:
        self._steps[step]["status"] = "skipped"
        self._steps[step]["summary"] = reason
        self._refresh()

    def fail_step(self, step: int, error: str = "") -> None:
        self._steps[step]["status"] = "failed"
        self._steps[step]["summary"] = error
        self._refresh()

    def render(self) -> Table:
        """Build the dashboard table."""
        table = Table(
            title="Dream Mode",
            show_header=False,
            box=None,
            padding=(0, 1),
            expand=False,
        )
        table.add_column("status", width=3)
        table.add_column("step", width=20)
        table.add_column("detail", max_width=50)

        for n, step_def in DREAM_STEPS.items():
            state = self._steps[n]
            icon, style = _STATUS_STYLE[state["status"]]
            name = f"{n}. {step_def['name']}"
            summary = state["summary"]
            if state["status"] == "running" and not summary:
                summary = "..."

            table.add_row(
                Text(icon, style=style),
                Text(name, style=style),
                Text(summary[:50], style="dim" if state["status"] != "failed" else "red"),
            )
        return table

    def __enter__(self) -> "DreamDashboard":
        self._live = Live(
            self.render(),
            console=self._console,
            refresh_per_second=2,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._live:
            self._live.__exit__(*args)

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self.render())
