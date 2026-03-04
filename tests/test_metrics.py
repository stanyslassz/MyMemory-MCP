"""Tests for core/metrics.py — metrics emission scaffolding."""

import json
from pathlib import Path

from src.core.metrics import MetricsCollector, metrics


class TestMetricsCollector:
    def test_counter_increment(self):
        m = MetricsCollector()
        assert m.ingest_success.value == 0
        m.ingest_success.inc()
        assert m.ingest_success.value == 1
        m.ingest_success.inc(5)
        assert m.ingest_success.value == 6

    def test_histogram_observe(self):
        m = MetricsCollector()
        m.ingest_latency.observe(1.5)
        m.ingest_latency.observe(2.0)
        m.ingest_latency.observe(0.5)
        assert m.ingest_latency.count == 3
        assert m.ingest_latency.p50 == 1.5

    def test_histogram_percentiles(self):
        m = MetricsCollector()
        for i in range(100):
            m.ingest_latency.observe(float(i))
        assert m.ingest_latency.p50 == 50.0
        assert m.ingest_latency.p95 == 95.0
        assert m.ingest_latency.p99 == 99.0

    def test_histogram_empty(self):
        m = MetricsCollector()
        assert m.ingest_latency.p50 == 0.0
        assert m.ingest_latency.p95 == 0.0

    def test_timer_start_stop(self):
        m = MetricsCollector()
        m.start_timer("test")
        import time
        time.sleep(0.01)
        elapsed = m.stop_timer("test")
        assert elapsed > 0

    def test_timer_missing_returns_zero(self):
        m = MetricsCollector()
        assert m.stop_timer("nonexistent") == 0.0

    def test_record_ingest_success(self):
        m = MetricsCollector()
        m.start_timer("ingest:file.md")
        m.record_ingest_success("file.md", 5)
        assert m.ingest_success.value == 1

    def test_record_ingest_failure(self):
        m = MetricsCollector()
        m.record_ingest_failure("file.md", "timeout")
        assert m.ingest_failure.value == 1

    def test_record_duplicate_skip(self):
        m = MetricsCollector()
        m.record_duplicate_skip("file.md")
        assert m.ingest_duplicate_skip.value == 1

    def test_record_route(self):
        m = MetricsCollector()
        m.record_route("document", 0.005)
        m.record_route("conversation", 0.003)
        assert m.route_counts["document"] == 1
        assert m.route_counts["conversation"] == 1
        assert m.router_latency.count == 2

    def test_snapshot(self):
        m = MetricsCollector()
        m.ingest_success.inc(3)
        m.ingest_failure.inc(1)
        snap = m.snapshot()
        assert snap["ingest_success_total"] == 3
        assert snap["ingest_failure_total"] == 1
        assert "ingest_latency" in snap
        assert "route_counts" in snap

    def test_export_json(self, tmp_path):
        m = MetricsCollector()
        m.ingest_success.inc(2)
        path = tmp_path / "metrics.json"
        m.export_json(path)
        data = json.loads(path.read_text())
        assert data["ingest_success_total"] == 2

    def test_reset(self):
        m = MetricsCollector()
        m.ingest_success.inc(10)
        m.reset()
        assert m.ingest_success.value == 0

    def test_global_singleton(self):
        metrics.reset()
        metrics.ingest_success.inc()
        assert metrics.ingest_success.value == 1
        metrics.reset()
