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
    identite: 10
    top_of_mind: 25
scoring:
  weight_importance: 0.4
  weight_frequency: 0.3
  weight_recency: 0.3
  frequency_cap: 20
categories:
  observations: [fait, preference]
  entity_types: [personne, sante]
  relation_types: [affecte, ameliore]
  folders:
    personne: proches
    sante: moi
prompts:
  path: "./prompts"
""")
    config = load_config(config_path=config_yaml, project_root=tmp_path)

    assert config.user_language == "en"
    assert config.llm_extraction.model == "openai/gpt-4o-mini"
    assert config.llm_extraction.temperature == 0
    assert config.scoring.weight_importance == 0.4
    assert config.categories.observations == ["fait", "preference"]
    assert config.categories.folders["personne"] == "proches"
    assert config.context_budget["identite"] == 10


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
    personne: proches
    sante: moi
""")
    config = load_config(config_path=config_yaml, project_root=tmp_path)
    assert config.get_folder_for_type("personne") == "proches"
    assert config.get_folder_for_type("sante") == "moi"
    assert config.get_folder_for_type("unknown") == "interets"  # default fallback


def test_load_real_config():
    """Load the actual project config.yaml."""
    project_root = Path(__file__).parent.parent
    config = load_config(project_root=project_root)
    assert config.user_language == "fr"
    assert len(config.categories.observations) == 14
    assert len(config.categories.entity_types) == 8
    assert len(config.categories.relation_types) == 13
