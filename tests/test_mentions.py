"""Tests for mention_dates windowing logic."""

from src.memory.mentions import add_mention, consolidate_window


def test_add_mention_basic():
    dates = []
    buckets = {}
    dates, buckets = add_mention("2026-03-05", dates, buckets, window_size=50)
    assert dates == ["2026-03-05"]
    assert buckets == {}


def test_add_mention_window_overflow():
    dates = [f"2026-01-{d:02d}" for d in range(1, 51)]  # 50 dates in Jan
    buckets = {}
    dates, buckets = add_mention("2026-02-01", dates, buckets, window_size=50)
    assert len(dates) == 50
    assert "2026-02-01" in dates
    assert "2026-01" in buckets
    assert buckets["2026-01"] >= 1


def test_add_mention_preserves_existing_buckets():
    dates = [f"2026-03-{d:02d}" for d in range(1, 51)]
    buckets = {"2025-06": 10}
    dates, buckets = add_mention("2026-04-01", dates, buckets, window_size=50)
    assert buckets["2025-06"] == 10
    assert len(dates) == 50


def test_consolidate_window():
    dates = [f"2026-01-{d:02d}" for d in range(1, 61)]  # 60 dates
    buckets = {}
    dates, buckets = consolidate_window(dates, buckets, window_size=50)
    assert len(dates) == 50
    assert "2026-01" in buckets
