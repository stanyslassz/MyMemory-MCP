"""Tests for core/config.py."""

from pathlib import Path

from src.core.config import load_config


def test_load_config_from_yaml(tmp_path):
    """Load config from a real config.yaml file."""
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text("""
user_language: "en"
llm:
  extraction:
    model: "openai/gpt-4o-mini"
    temperature: 0
    max_retries: 3
    timeout: 60
  arbitration:
    model: "openai/gpt-4o-mini"
    temperature: 0
  context:
    model: "openai/gpt-4o-mini"
    temperature: 0.3
  consolidation:
    model: "openai/gpt-4o-mini"
embeddings:
  provider: "sentence-transformers"
  model: "all-MiniLM-L6-v2"
memory:
  path: "./memory"
  context_max_tokens: 3000
  context_budget:
    identity: 10
    top_of_mind: 25
scoring:
  decay_factor: 0.5
  importance_weight: 0.3
categories:
  observations: [fact, preference]
  entity_types: [person, health]
  relation_types: [affects, improves]
  folders:
    person: close_ones
    health: self
prompts:
  path: "./prompts"
""")
    config = load_config(config_path=config_yaml, project_root=tmp_path)

    assert config.user_language == "en"
    assert config.llm_extraction.model == "openai/gpt-4o-mini"
    assert config.llm_extraction.temperature == 0
    assert config.scoring.importance_weight == 0.3
    assert config.categories.observations == ["fact", "preference"]
    assert config.categories.folders["person"] == "close_ones"
    assert config.context_budget["identity"] == 10


def test_load_config_defaults(tmp_path):
    """Loading a missing config returns defaults."""
    config = load_config(config_path=tmp_path / "nonexistent.yaml", project_root=tmp_path)
    assert config.user_language == "fr"
    assert config.mcp_transport == "stdio"


def test_get_folder_for_type(tmp_path):
    """Test entity type → folder mapping."""
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text("""
categories:
  folders:
    person: close_ones
    health: self
""")
    config = load_config(config_path=config_yaml, project_root=tmp_path)
    assert config.get_folder_for_type("person") == "close_ones"
    assert config.get_folder_for_type("health") == "self"
    assert config.get_folder_for_type("unknown") == "interests"  # default fallback


def test_load_real_config():
    """Load the actual project config.yaml."""
    project_root = Path(__file__).parent.parent
    config = load_config(project_root=project_root)
    assert config.user_language in ("fr", "en")
    assert len(config.categories.observations) == 17
    assert len(config.categories.entity_types) == 9
    assert len(config.categories.relation_types) == 13


def test_scoring_config_actr_fields():
    """Verify ACT-R scoring parameters load from project config."""
    project_root = Path(__file__).parent.parent
    config = load_config(project_root=project_root)
    s = config.scoring
    assert s.model == "act_r"
    assert s.decay_factor == 0.5
    assert s.decay_factor_short_term == 0.8
    assert s.importance_weight == 0.3
    assert s.spreading_weight == 0.2
    assert s.permanent_min_score == 0.5
    assert s.window_size == 50
