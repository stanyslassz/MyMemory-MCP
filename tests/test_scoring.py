"""Tests for ACT-R scoring + spreading activation."""

import math
from datetime import date
from src.core.config import ScoringConfig, Config
from src.core.models import GraphData, GraphEntity, GraphRelation
from src.memory.scoring import (
    calculate_score, calculate_actr_base, recalculate_all_scores,
    get_top_entities, spreading_activation, _apply_ltd,
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
    """Fresh relation should contribute more spreading than a stale one.

    With normalization, a single-neighbor entity always gets the neighbor's
    base score as its spreading bonus (eff * base / eff = base). To test
    decay differentiation, we give each entity two neighbors: one shared
    low-score neighbor and one unique high/low-score neighbor, with the
    fresh vs stale relation controlling how much the high-score neighbor
    contributes to the weighted average.
    """
    config = _make_config()
    today = date(2026, 3, 5)
    graph = GraphData()
    graph.entities["high_score"] = GraphEntity(
        file="self/hs.md", type="health", title="HighScore",
        importance=0.9, mention_dates=["2026-03-05"] * 5,
    )
    graph.entities["low_score"] = GraphEntity(
        file="self/ls.md", type="health", title="LowScore",
        importance=0.1, mention_dates=["2020-01-01"],
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
        # Both connect to low_score with same fresh relation
        GraphRelation(from_entity="low_score", to_entity="b_fresh", type="affects",
                      strength=0.8, last_reinforced="2026-03-01"),
        GraphRelation(from_entity="low_score", to_entity="b_stale", type="affects",
                      strength=0.8, last_reinforced="2026-03-01"),
        # b_fresh has a FRESH relation to high_score (high effective strength)
        GraphRelation(from_entity="high_score", to_entity="b_fresh", type="affects",
                      strength=0.8, last_reinforced="2026-03-01"),
        # b_stale has a STALE relation to high_score (low effective strength due to decay)
        GraphRelation(from_entity="high_score", to_entity="b_stale", type="affects",
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
    """Relations with higher strength from Hebbian learning should produce stronger spreading.

    With normalization, a single-neighbor entity always gets the neighbor's
    base score as bonus (eff * base / eff = base). To test Hebbian differentiation,
    each entity has two neighbors: one high-score and one low-score. The Hebbian-
    reinforced entity has a stronger relation to the high-score neighbor, giving
    it a higher weighted average.
    """
    config = _make_config()
    today = date(2026, 3, 5)
    graph = GraphData()
    graph.entities["high_hub"] = GraphEntity(
        file="self/hh.md", type="health", title="HighHub",
        importance=0.9, mention_dates=["2026-03-05"] * 10,
    )
    graph.entities["low_hub"] = GraphEntity(
        file="self/lh.md", type="health", title="LowHub",
        importance=0.1, mention_dates=["2020-01-01"],
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
        # Both connect to low_hub with same strength
        GraphRelation(from_entity="low_hub", to_entity="weak_link", type="affects",
                      strength=0.5, last_reinforced="2026-03-05"),
        GraphRelation(from_entity="low_hub", to_entity="strong_link", type="affects",
                      strength=0.5, last_reinforced="2026-03-05"),
        # weak_link has a weak relation to high_hub
        GraphRelation(from_entity="high_hub", to_entity="weak_link", type="affects",
                      strength=0.5, last_reinforced="2026-03-05"),
        # strong_link has a Hebbian-reinforced strong relation to high_hub
        GraphRelation(from_entity="high_hub", to_entity="strong_link", type="affects",
                      strength=0.9, last_reinforced="2026-03-05"),
    ]
    spreading = spreading_activation(graph, config, today)
    assert spreading["strong_link"] > spreading["weak_link"]


def test_spreading_normalized_by_total_strength():
    """Entity with 10 relations should NOT get ~10x the spreading bonus of one with 1 relation.

    Before the fix, spreading was the raw sum (eff * base_score) across all neighbors,
    so an entity connected to 10 identical neighbors would get ~10x the bonus of one
    connected to 1. After normalization (dividing by total_strength), both should get
    roughly the same bonus since each neighbor is identical.
    """
    config = _make_config()
    today = date(2026, 3, 5)
    graph = GraphData()

    # Create a pool of identical neighbor entities
    for i in range(10):
        graph.entities[f"neighbor_{i}"] = GraphEntity(
            file=f"self/n{i}.md", type="health", title=f"Neighbor{i}",
            importance=0.8, mention_dates=["2026-03-05"] * 5,
        )

    # Entity with 1 relation
    graph.entities["one_link"] = GraphEntity(
        file="self/one.md", type="health", title="OneLink",
        importance=0.3, mention_dates=["2025-06-01"],
    )
    # Entity with 10 relations (to 10 identical neighbors)
    graph.entities["ten_links"] = GraphEntity(
        file="self/ten.md", type="health", title="TenLinks",
        importance=0.3, mention_dates=["2025-06-01"],
    )

    relations = []
    # one_link connects to neighbor_0
    relations.append(GraphRelation(
        from_entity="one_link", to_entity="neighbor_0", type="affects",
        strength=0.8, last_reinforced="2026-03-05",
    ))
    # ten_links connects to all 10 neighbors
    for i in range(10):
        relations.append(GraphRelation(
            from_entity="ten_links", to_entity=f"neighbor_{i}", type="affects",
            strength=0.8, last_reinforced="2026-03-05",
        ))
    graph.relations = relations

    spreading = spreading_activation(graph, config, today)

    # With normalization, one_link and ten_links should get similar spreading bonuses
    # (both are connected to identical neighbors via identical strength relations).
    # Without normalization, ten_links would get ~10x the bonus.
    ratio = spreading["ten_links"] / spreading["one_link"] if spreading["one_link"] > 0 else float("inf")
    assert ratio < 2.0, (
        f"Spreading ratio ten_links/one_link = {ratio:.2f}, expected < 2.0 "
        f"(ten={spreading['ten_links']:.4f}, one={spreading['one_link']:.4f})"
    )


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


# ── GAP 1: LTD (Long-Term Depression) ────────────────────────


def test_ltd_decays_stale_relations():
    """Relations not reinforced for >90 days should have reduced stored strength."""
    config = _make_config()
    today = date(2026, 3, 5)
    graph = GraphData()
    graph.entities["a"] = GraphEntity(
        file="self/a.md", type="health", title="A",
        importance=0.5, mention_dates=["2026-03-05"],
    )
    graph.entities["b"] = GraphEntity(
        file="self/b.md", type="health", title="B",
        importance=0.5, mention_dates=["2026-03-05"],
    )
    graph.relations = [
        GraphRelation(from_entity="a", to_entity="b", type="affects",
                      strength=0.8, last_reinforced="2025-06-01"),  # ~9 months old
    ]
    original_strength = graph.relations[0].strength
    _apply_ltd(graph, config, today)
    assert graph.relations[0].strength < original_strength
    assert graph.relations[0].strength >= 0.1  # minimum floor


def test_ltd_preserves_fresh_relations():
    """Relations reinforced recently (<90 days) should not decay."""
    config = _make_config()
    today = date(2026, 3, 5)
    graph = GraphData()
    graph.entities["a"] = GraphEntity(
        file="self/a.md", type="health", title="A",
        importance=0.5, mention_dates=["2026-03-05"],
    )
    graph.entities["b"] = GraphEntity(
        file="self/b.md", type="health", title="B",
        importance=0.5, mention_dates=["2026-03-05"],
    )
    graph.relations = [
        GraphRelation(from_entity="a", to_entity="b", type="affects",
                      strength=0.8, last_reinforced="2026-02-15"),  # <90 days
    ]
    _apply_ltd(graph, config, today)
    assert graph.relations[0].strength == 0.8  # unchanged


def test_ltd_minimum_floor():
    """LTD should not reduce strength below 0.1."""
    config = _make_config()
    today = date(2026, 3, 5)
    graph = GraphData()
    graph.entities["a"] = GraphEntity(
        file="self/a.md", type="health", title="A",
        importance=0.5, mention_dates=["2026-03-05"],
    )
    graph.entities["b"] = GraphEntity(
        file="self/b.md", type="health", title="B",
        importance=0.5, mention_dates=["2026-03-05"],
    )
    graph.relations = [
        GraphRelation(from_entity="a", to_entity="b", type="affects",
                      strength=0.15, last_reinforced="2020-01-01"),  # very old
    ]
    _apply_ltd(graph, config, today)
    assert graph.relations[0].strength == 0.1


# ── GAP 2: Retrieval threshold ────────────────────────────────


def test_retrieval_threshold_true_forgetting():
    """Entities with very low activation should score exactly 0.0."""
    config = _make_config(retrieval_threshold=0.05)
    today = date(2026, 3, 5)
    entity = GraphEntity(
        file="self/a.md", type="health", title="Forgotten",
        importance=0.0, mention_dates=["2018-01-01"],  # 8 years ago, low importance
    )
    score = calculate_score(entity, config, today)
    assert score == 0.0, f"Expected 0.0 for truly forgotten entity, got {score}"


def test_retrieval_threshold_spares_permanent():
    """Permanent entities should never be zeroed out by retrieval threshold."""
    config = _make_config(retrieval_threshold=0.05)
    today = date(2026, 3, 5)
    entity = GraphEntity(
        file="self/a.md", type="health", title="Perm",
        importance=0.0, retention="permanent",
        mention_dates=["2018-01-01"],
    )
    score = calculate_score(entity, config, today)
    assert score >= config.scoring.permanent_min_score


def test_retrieval_threshold_passes_active_entities():
    """Active entities should not be affected by retrieval threshold."""
    config = _make_config(retrieval_threshold=0.05)
    today = date(2026, 3, 5)
    entity = GraphEntity(
        file="self/a.md", type="health", title="Active",
        importance=0.8, mention_dates=["2026-03-05"] * 5,
    )
    score = calculate_score(entity, config, today)
    assert score > 0.5


# ── GAP 3: Power-law relation decay ──────────────────────────


def test_power_law_relation_decay():
    """Spreading activation should use power-law decay for relation effective strength.

    A 1-day-old relation should have much higher effective strength than a 365-day one,
    following power-law (days+0.5)^(-d) rather than exponential.
    """
    config = _make_config(relation_decay_power=0.3)
    today = date(2026, 3, 5)
    graph = GraphData()
    graph.entities["hub"] = GraphEntity(
        file="self/hub.md", type="health", title="Hub",
        importance=0.9, mention_dates=["2026-03-05"] * 5,
    )
    graph.entities["anchor"] = GraphEntity(
        file="self/anchor.md", type="health", title="Anchor",
        importance=0.1, mention_dates=["2020-01-01"],
    )
    graph.entities["fresh_link"] = GraphEntity(
        file="self/fl.md", type="health", title="FreshLink",
        importance=0.1, mention_dates=["2025-01-01"],
    )
    graph.entities["stale_link"] = GraphEntity(
        file="self/sl.md", type="health", title="StaleLink",
        importance=0.1, mention_dates=["2025-01-01"],
    )
    graph.relations = [
        # Both connect to same anchor with same strength
        GraphRelation(from_entity="anchor", to_entity="fresh_link", type="affects",
                      strength=0.8, last_reinforced="2026-03-01"),
        GraphRelation(from_entity="anchor", to_entity="stale_link", type="affects",
                      strength=0.8, last_reinforced="2026-03-01"),
        # fresh_link has a fresh relation to hub
        GraphRelation(from_entity="hub", to_entity="fresh_link", type="affects",
                      strength=0.8, last_reinforced="2026-03-04"),
        # stale_link has a stale relation to hub
        GraphRelation(from_entity="hub", to_entity="stale_link", type="affects",
                      strength=0.8, last_reinforced="2024-03-05"),
    ]
    spreading = spreading_activation(graph, config, today)
    assert spreading["fresh_link"] > spreading["stale_link"]


# ── GAP 4: Emotional valence boost ───────────────────────────


def test_emotional_boost_negative_valence():
    """Entities with high negative_valence_ratio should score higher (amygdala effect)."""
    config = _make_config(emotional_boost_weight=0.15)
    today = date(2026, 3, 5)
    kwargs = dict(
        file="self/a.md", type="health", title="X",
        importance=0.5,
        mention_dates=["2026-01-01", "2026-02-01"],
    )
    neutral = GraphEntity(**kwargs, negative_valence_ratio=0.0)
    emotional = GraphEntity(**kwargs, negative_valence_ratio=0.8)
    assert calculate_score(emotional, config, today) > calculate_score(neutral, config, today)


def test_emotional_boost_zero_when_no_negative():
    """Entity with no negative facts should get no emotional boost."""
    config = _make_config(emotional_boost_weight=0.15)
    today = date(2026, 3, 5)
    entity = GraphEntity(
        file="self/a.md", type="health", title="Neutral",
        importance=0.5, mention_dates=["2026-03-01"],
        negative_valence_ratio=0.0,
    )
    # Score should be the same as without the feature
    config_no_emotion = _make_config(emotional_boost_weight=0.0)
    assert calculate_score(entity, config, today) == calculate_score(entity, config_no_emotion, today)
