"""Phase 4 integration test -- full pipeline verification."""

from datetime import date
from src.core.config import ScoringConfig, Config, NLPConfig
from src.core.models import (
    GraphData, GraphEntity, GraphRelation,
)
from src.memory.scoring import calculate_score, recalculate_all_scores, spreading_activation
from src.memory.mentions import add_mention
from src.pipeline.nlp_prefilter import _try_parse_french_date


def test_full_scoring_pipeline():
    """End-to-end: create entities, add mentions, score with ACT-R + spreading."""
    config = Config.__new__(Config)
    config.scoring = ScoringConfig()
    today = date(2026, 3, 5)

    graph = GraphData()

    # Create two related entities
    graph.entities["back-pain"] = GraphEntity(
        file="self/back-pain.md", type="health", title="Back Pain",
        importance=0.8, mention_dates=["2026-03-01", "2026-03-03", "2026-03-05"],
        retention="long_term",
    )
    graph.entities["physio"] = GraphEntity(
        file="self/physio.md", type="health", title="Physiotherapy",
        importance=0.5, mention_dates=["2026-02-01"],
        retention="long_term",
    )
    graph.relations = [
        GraphRelation(
            **{"from": "back-pain", "to": "physio"},
            type="improves",
            strength=0.7, last_reinforced="2026-03-05",
        ),
    ]

    # Score with spreading
    graph = recalculate_all_scores(graph, config, today)

    # Back pain (recently mentioned, high importance) should score high
    assert graph.entities["back-pain"].score > 0.6

    # Physio (less recently mentioned) should get a boost from back-pain via spreading
    spreading = spreading_activation(graph, config, today)
    assert spreading["physio"] > 0  # Gets boost from related back-pain

    # All scores normalized
    for _, e in graph.entities.items():
        assert 0 <= e.score <= 1


def test_mention_windowing_in_scoring():
    """Verify mention_dates + monthly_buckets both contribute to score."""
    config = Config.__new__(Config)
    config.scoring = ScoringConfig()
    today = date(2026, 3, 5)

    entity = GraphEntity(
        file="self/test.md", type="health", title="Test",
        importance=0.5,
        mention_dates=["2026-03-05"],
        monthly_buckets={"2025-06": 10},
    )
    score = calculate_score(entity, config, today)
    assert score > 0.5  # Should be boosted by both recent mention + historical buckets


def test_nlp_date_parsing_pipeline():
    """NLP date parsing integrates with mention_dates."""
    # Parse a French date
    parsed = _try_parse_french_date("il y a 3 jours", "2026-03-05")
    assert parsed == "2026-03-02"

    # Add it as a mention
    dates = []
    buckets = {}
    dates, buckets = add_mention(parsed, dates, buckets)
    assert "2026-03-02" in dates


def test_config_has_nlp_section():
    """Config should have NLP section with defaults."""
    config = Config()
    assert hasattr(config, 'nlp')
    assert config.nlp.enabled == True
    assert config.nlp.dedup_threshold == 0.85
