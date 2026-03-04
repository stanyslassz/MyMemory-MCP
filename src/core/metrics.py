"""Metrics emission scaffolding for SLO dashboards.

Collects ingest success/failure, latency, and readiness timing.
Currently stores in-memory with JSON export. Ready for future Prometheus/StatsD backend.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class _Counter:
    name: str
    value: int = 0
    labels: dict[str, str] = field(default_factory=dict)

    def inc(self, n: int = 1) -> None:
        self.value += n


@dataclass
class _Histogram:
    name: str
    values: list[float] = field(default_factory=list)

    def observe(self, v: float) -> None:
        self.values.append(v)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def p50(self) -> float:
        if not self.values:
            return 0.0
        s = sorted(self.values)
        return s[len(s) // 2]

    @property
    def p95(self) -> float:
        if not self.values:
            return 0.0
        s = sorted(self.values)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    @property
    def p99(self) -> float:
        if not self.values:
            return 0.0
        s = sorted(self.values)
        idx = int(len(s) * 0.99)
        return s[min(idx, len(s) - 1)]


class MetricsCollector:
    """In-memory metrics collector for SLO tracking."""

    def __init__(self) -> None:
        self.ingest_success = _Counter("ingest_success_total")
        self.ingest_failure = _Counter("ingest_failure_total")
        self.ingest_duplicate_skip = _Counter("ingest_duplicate_skip_total")
        self.ingest_latency = _Histogram("ingest_latency_seconds")
        self.readiness_latency = _Histogram("readiness_latency_seconds")
        self.router_latency = _Histogram("router_decision_latency_seconds")
        self.route_counts: dict[str, int] = {
            "conversation": 0,
            "document": 0,
            "uncertain": 0,
        }
        self._timers: dict[str, float] = {}

    def start_timer(self, name: str) -> None:
        self._timers[name] = time.monotonic()

    def stop_timer(self, name: str) -> float:
        start = self._timers.pop(name, None)
        if start is None:
            return 0.0
        return time.monotonic() - start

    def record_ingest_success(self, source_id: str, chunks: int) -> None:
        self.ingest_success.inc()
        elapsed = self.stop_timer(f"ingest:{source_id}")
        if elapsed > 0:
            self.ingest_latency.observe(elapsed)

    def record_ingest_failure(self, source_id: str, error: str) -> None:
        self.ingest_failure.inc()
        self.stop_timer(f"ingest:{source_id}")
        logger.warning("Ingest failed for %s: %s", source_id, error)

    def record_duplicate_skip(self, source_id: str) -> None:
        self.ingest_duplicate_skip.inc()

    def record_route(self, route: str, latency: float) -> None:
        self.route_counts[route] = self.route_counts.get(route, 0) + 1
        self.router_latency.observe(latency)

    def record_readiness(self, source_id: str) -> None:
        elapsed = self.stop_timer(f"readiness:{source_id}")
        if elapsed > 0:
            self.readiness_latency.observe(elapsed)

    def snapshot(self) -> dict:
        """Export current metrics as a dict."""
        return {
            "ingest_success_total": self.ingest_success.value,
            "ingest_failure_total": self.ingest_failure.value,
            "ingest_duplicate_skip_total": self.ingest_duplicate_skip.value,
            "ingest_latency": {
                "count": self.ingest_latency.count,
                "p50": round(self.ingest_latency.p50, 3),
                "p95": round(self.ingest_latency.p95, 3),
                "p99": round(self.ingest_latency.p99, 3),
            },
            "readiness_latency": {
                "count": self.readiness_latency.count,
                "p50": round(self.readiness_latency.p50, 3),
                "p95": round(self.readiness_latency.p95, 3),
            },
            "router_latency": {
                "count": self.router_latency.count,
                "p95": round(self.router_latency.p95, 3),
            },
            "route_counts": dict(self.route_counts),
        }

    def export_json(self, path: str | Path) -> None:
        """Write metrics snapshot to JSON file."""
        Path(path).write_text(
            json.dumps(self.snapshot(), indent=2), encoding="utf-8"
        )

    def reset(self) -> None:
        """Reset all counters (for testing)."""
        self.__init__()


# Global singleton
metrics = MetricsCollector()
