"""Tests for memory/scoring.py."""

import math
from datetime import date

from src.core.config import ScoringConfig, Config
from src.core.models import GraphData, GraphEntity
from src.memory.scoring import calculate_score, get_top_entities, recalculate_all_scores


def _make_config(**overrides):
    scoring = ScoringConfig(
        weight_importance=0.4,
        weight_frequency=0.3,
        weight_recency=0.3,
        frequency_cap=20,
        recency_halflife_days=30,
        min_score_for_context=0.3,
    )
    for k, v in overrides.items():
        setattr(scoring, k, v)
    config = Config.__new__(Config)
    config.scoring = scoring
    return config


def test_score_calculation_today():
    config = _make_config()
    entity = GraphEntity(
        file="moi/test.md", type="health", title="Test",
        importance=0.8, frequency=10, last_mentioned="2026-03-03",
    )
    score = calculate_score(entity, config, today=date(2026, 3, 3))
    # importance: 0.8 * 0.4 = 0.32
    # frequency: min(10/20, 1.0) * 0.3 = 0.5 * 0.3 = 0.15
    # recency: e^(0/30) * 0.3 = 1.0 * 0.3 = 0.3
    expected = 0.32 + 0.15 + 0.30
    assert abs(score - expected) < 0.01


def test_score_calculation_old_entity():
    config = _make_config()
    entity = GraphEntity(
        file="moi/test.md", type="health", title="Test",
        importance=0.5, frequency=5, last_mentioned="2025-12-03",
    )
    score = calculate_score(entity, config, today=date(2026, 3, 3))
    # 90 days old → recency = e^(-90/30) = e^(-3) ≈ 0.05
    days = 90
    recency = math.exp(-days / 30)
    expected = 0.5 * 0.4 + min(5 / 20, 1.0) * 0.3 + recency * 0.3
    assert abs(score - expected) < 0.01


def test_score_frequency_cap():
    config = _make_config()
    entity = GraphEntity(
        file="moi/test.md", type="health", title="Test",
        importance=0.5, frequency=100, last_mentioned="2026-03-03",
    )
    score = calculate_score(entity, config, today=date(2026, 3, 3))
    # frequency capped at 1.0 → 0.3
    assert score <= 1.0


def test_recalculate_all_scores():
    config = _make_config()
    graph = GraphData()
    graph.entities["a"] = GraphEntity(
        file="moi/a.md", type="health", title="A",
        importance=0.9, frequency=15, last_mentioned="2026-03-03",
    )
    graph.entities["b"] = GraphEntity(
        file="moi/b.md", type="interest", title="B",
        importance=0.3, frequency=2, last_mentioned="2025-01-01",
    )
    graph = recalculate_all_scores(graph, config, today=date(2026, 3, 3))
    assert graph.entities["a"].score > graph.entities["b"].score


def test_get_top_entities():
    graph = GraphData()
    graph.entities["high"] = GraphEntity(
        file="moi/h.md", type="health", title="High", score=0.9,
    )
    graph.entities["low"] = GraphEntity(
        file="moi/l.md", type="interest", title="Low", score=0.1,
    )
    graph.entities["perm"] = GraphEntity(
        file="moi/p.md", type="health", title="Permanent",
        score=0.2, retention="permanent",
    )

    top = get_top_entities(graph, n=1, include_permanent=True, min_score=0.0)
    ids = [eid for eid, _ in top]
    assert "perm" in ids  # always included
    assert "high" in ids


def test_get_top_entities_min_score():
    graph = GraphData()
    graph.entities["high"] = GraphEntity(
        file="moi/h.md", type="health", title="High", score=0.9,
    )
    graph.entities["low"] = GraphEntity(
        file="moi/l.md", type="interest", title="Low", score=0.1,
    )
    top = get_top_entities(graph, n=10, include_permanent=True, min_score=0.5)
    ids = [eid for eid, _ in top]
    assert "high" in ids
    assert "low" not in ids
