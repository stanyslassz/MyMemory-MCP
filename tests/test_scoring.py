"""Tests for ACT-R scoring + spreading activation."""

import math
from datetime import date
from src.core.config import ScoringConfig, Config
from src.core.models import GraphData, GraphEntity, GraphRelation
from src.memory.scoring import (
    calculate_score, calculate_actr_base, recalculate_all_scores,
    get_top_entities, spreading_activation,
)


def _make_config(**overrides):
    scoring = ScoringConfig()  # Uses ACT-R defaults
    for k, v in overrides.items():
        setattr(scoring, k, v)
    config = Config.__new__(Config)
    config.scoring = scoring
    return config


def test_actr_burst_beats_spread():
    """5 mentions in 2 days scores higher than 5 over 5 months."""
    config = _make_config()
    today = date(2026, 3, 5)
    burst = GraphEntity(
        file="self/a.md", type="health", title="Burst",
        importance=0.5,
        mention_dates=["2026-03-04", "2026-03-04", "2026-03-05", "2026-03-05", "2026-03-05"],
    )
    spread = GraphEntity(
        file="self/b.md", type="health", title="Spread",
        importance=0.5,
        mention_dates=["2025-10-05", "2025-11-05", "2025-12-05", "2026-01-05", "2026-02-05"],
    )
    assert calculate_score(burst, config, today) > calculate_score(spread, config, today)


def test_actr_normalized_0_to_1():
    config = _make_config()
    today = date(2026, 3, 5)
    entities = [
        GraphEntity(file="self/a.md", type="health", title="A",
                    importance=1.0, mention_dates=[today.isoformat()] * 50),
        GraphEntity(file="self/b.md", type="health", title="B",
                    importance=0.0, mention_dates=[]),
        GraphEntity(file="self/c.md", type="health", title="C",
                    importance=0.5, mention_dates=["2020-01-01"]),
    ]
    for e in entities:
        score = calculate_score(e, config, today)
        assert 0.0 <= score <= 1.0, f"Score {score} out of bounds for {e.title}"


def test_actr_no_mentions_low_score():
    config = _make_config()
    today = date(2026, 3, 5)
    entity = GraphEntity(
        file="self/a.md", type="health", title="Empty",
        importance=0.3, mention_dates=[],
    )
    assert calculate_score(entity, config, today) < 0.5


def test_actr_permanent_min_score():
    config = _make_config()
    today = date(2026, 3, 5)
    entity = GraphEntity(
        file="self/a.md", type="health", title="Perm",
        importance=0.1, retention="permanent",
        mention_dates=["2020-01-01"],
    )
    assert calculate_score(entity, config, today) >= config.scoring.permanent_min_score


def test_actr_short_term_decays_faster():
    config = _make_config()
    today = date(2026, 3, 5)
    kwargs = dict(
        file="self/a.md", type="health", title="X",
        importance=0.5,
        mention_dates=["2025-12-01", "2025-12-15", "2026-01-01"],
    )
    lt = GraphEntity(**kwargs, retention="long_term")
    st = GraphEntity(**kwargs, retention="short_term")
    assert calculate_score(lt, config, today) > calculate_score(st, config, today)


def test_actr_monthly_buckets_contribute():
    config = _make_config()
    today = date(2026, 3, 5)
    with_b = GraphEntity(
        file="self/a.md", type="health", title="With",
        importance=0.5, mention_dates=["2026-03-01"],
        monthly_buckets={"2025-06": 10, "2025-09": 5},
    )
    without_b = GraphEntity(
        file="self/b.md", type="health", title="Without",
        importance=0.5, mention_dates=["2026-03-01"],
    )
    assert calculate_score(with_b, config, today) > calculate_score(without_b, config, today)


def test_spreading_weak_boosted_by_strong():
    config = _make_config()
    today = date(2026, 3, 5)
    graph = GraphData()
    graph.entities["strong"] = GraphEntity(
        file="self/s.md", type="health", title="Strong",
        importance=0.9, mention_dates=["2026-03-05"] * 10,
    )
    graph.entities["weak"] = GraphEntity(
        file="self/w.md", type="health", title="Weak",
        importance=0.1, mention_dates=["2025-01-01"],
    )
    graph.relations = [
        GraphRelation(from_entity="strong", to_entity="weak", type="affects",
                      strength=0.8, last_reinforced="2026-03-05"),
    ]
    spreading = spreading_activation(graph, config, today)
    assert spreading["weak"] > 0


def test_spreading_isolated_zero():
    config = _make_config()
    today = date(2026, 3, 5)
    graph = GraphData()
    graph.entities["isolated"] = GraphEntity(
        file="self/i.md", type="health", title="Isolated",
        importance=0.5, mention_dates=["2026-03-01"],
    )
    spreading = spreading_activation(graph, config, today)
    assert spreading["isolated"] == 0.0


def test_spreading_decayed_relation():
    config = _make_config()
    today = date(2026, 3, 5)
    graph = GraphData()
    graph.entities["a"] = GraphEntity(
        file="self/a.md", type="health", title="A",
        importance=0.9, mention_dates=["2026-03-05"] * 5,
    )
    graph.entities["b_fresh"] = GraphEntity(
        file="self/bf.md", type="health", title="BF",
        importance=0.1, mention_dates=["2025-01-01"],
    )
    graph.entities["b_stale"] = GraphEntity(
        file="self/bs.md", type="health", title="BS",
        importance=0.1, mention_dates=["2025-01-01"],
    )
    graph.relations = [
        GraphRelation(from_entity="a", to_entity="b_fresh", type="affects",
                      strength=0.8, last_reinforced="2026-03-01"),
        GraphRelation(from_entity="a", to_entity="b_stale", type="affects",
                      strength=0.8, last_reinforced="2024-01-01"),
    ]
    spreading = spreading_activation(graph, config, today)
    assert spreading["b_fresh"] > spreading["b_stale"]


def test_recalculate_all_scores_with_spreading():
    config = _make_config()
    today = date(2026, 3, 5)
    graph = GraphData()
    graph.entities["a"] = GraphEntity(
        file="self/a.md", type="health", title="A",
        importance=0.9, mention_dates=["2026-03-05"] * 5,
    )
    graph.entities["b"] = GraphEntity(
        file="self/b.md", type="interest", title="B",
        importance=0.3, mention_dates=["2025-01-01"],
    )
    graph = recalculate_all_scores(graph, config, today)
    assert graph.entities["a"].score > graph.entities["b"].score
    assert 0 <= graph.entities["a"].score <= 1
    assert 0 <= graph.entities["b"].score <= 1


def test_get_top_entities():
    graph = GraphData()
    graph.entities["high"] = GraphEntity(
        file="self/h.md", type="health", title="High", score=0.9,
    )
    graph.entities["low"] = GraphEntity(
        file="self/l.md", type="interest", title="Low", score=0.1,
    )
    graph.entities["perm"] = GraphEntity(
        file="self/p.md", type="health", title="Permanent",
        score=0.2, retention="permanent",
    )
    top = get_top_entities(graph, n=1, include_permanent=True, min_score=0.0)
    ids = [eid for eid, _ in top]
    assert "perm" in ids
    assert "high" in ids


def test_hebbian_strength_boosts_spreading():
    """Relations with higher strength from Hebbian learning should produce stronger spreading."""
    config = _make_config()
    today = date(2026, 3, 5)
    graph = GraphData()
    graph.entities["hub"] = GraphEntity(
        file="self/hub.md", type="health", title="Hub",
        importance=0.9, mention_dates=["2026-03-05"] * 10,
    )
    graph.entities["weak_link"] = GraphEntity(
        file="self/wl.md", type="health", title="WeakLink",
        importance=0.1, mention_dates=["2025-01-01"],
    )
    graph.entities["strong_link"] = GraphEntity(
        file="self/sl.md", type="health", title="StrongLink",
        importance=0.1, mention_dates=["2025-01-01"],
    )
    graph.relations = [
        GraphRelation(from_entity="hub", to_entity="weak_link", type="affects",
                      strength=0.5, last_reinforced="2026-03-05"),
        GraphRelation(from_entity="hub", to_entity="strong_link", type="affects",
                      strength=0.9, last_reinforced="2026-03-05"),  # Hebbian-reinforced
    ]
    spreading = spreading_activation(graph, config, today)
    assert spreading["strong_link"] > spreading["weak_link"]


def test_get_top_entities_min_score():
    graph = GraphData()
    graph.entities["high"] = GraphEntity(
        file="self/h.md", type="health", title="High", score=0.9,
    )
    graph.entities["low"] = GraphEntity(
        file="self/l.md", type="interest", title="Low", score=0.1,
    )
    top = get_top_entities(graph, n=10, include_permanent=True, min_score=0.5)
    ids = [eid for eid, _ in top]
    assert "high" in ids
    assert "low" not in ids
