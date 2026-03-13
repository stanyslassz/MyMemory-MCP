"""Tests for externalized config constants (Phase A WS6)."""

from src.core.config import Config, ScoringConfig, SearchConfig, DreamConfig, ContextConfig


def test_scoring_config_new_fields_have_defaults():
    s = ScoringConfig()
    assert s.ltd_onset_days == 90
    assert s.min_relation_strength == 0.1


def test_search_config_new_fields_have_defaults():
    s = SearchConfig()
    assert s.resolver_threshold == 0.75
    assert s.linear_faiss_weight == 0.6
    assert s.linear_actr_weight == 0.4


def test_dream_config_new_fields_have_defaults():
    d = DreamConfig()
    assert d.prune_score_threshold == 0.1
    assert d.prune_min_age_days == 90
    assert d.prune_max_frequency == 1
    assert d.transitive_min_strength == 0.4
    assert d.transitive_max_new == 20


def test_context_config_new_fields_have_defaults():
    c = ContextConfig()
    assert c.history_recent_days == 30


def test_config_loads_new_fields_from_yaml(tmp_path):
    """New fields should be loadable from YAML."""
    import yaml
    from src.core.config import load_config

    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(yaml.dump({
        "scoring": {"ltd_onset_days": 60, "min_relation_strength": 0.2},
        "search": {"resolver_threshold": 0.80, "linear_faiss_weight": 0.7, "linear_actr_weight": 0.3},
        "dream": {
            "prune_score_threshold": 0.15,
            "prune_min_age_days": 120,
            "prune_max_frequency": 2,
            "transitive_min_strength": 0.5,
            "transitive_max_new": 30,
        },
        "context": {"history_recent_days": 45},
    }))

    config = load_config(config_yaml, tmp_path)
    assert config.scoring.ltd_onset_days == 60
    assert config.scoring.min_relation_strength == 0.2
    assert config.search.resolver_threshold == 0.80
    assert config.search.linear_faiss_weight == 0.7
    assert config.dream.prune_score_threshold == 0.15
    assert config.dream.prune_min_age_days == 120
    assert config.dream.transitive_min_strength == 0.5
    assert config.dream.transitive_max_new == 30
    assert config.ctx.history_recent_days == 45
