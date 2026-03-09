"""Tests for max_facts configuration and enforcement."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.config import Config, load_config


def test_get_max_facts_default():
    """Config.get_max_facts returns default for unknown types."""
    config = Config()
    assert config.get_max_facts("person") == 50
    assert config.get_max_facts("health") == 50


def test_get_max_facts_ai_self():
    """Config.get_max_facts returns specific limit for ai_self."""
    config = Config()
    assert config.get_max_facts("ai_self") == 20


def test_get_max_facts_custom():
    """Config.get_max_facts respects custom config."""
    config = Config(max_facts={"default": 30, "health": 10})
    assert config.get_max_facts("health") == 10
    assert config.get_max_facts("person") == 30
    assert config.get_max_facts("ai_self") == 30  # not set, falls back to default


def test_consolidate_entity_facts_passes_max_facts(tmp_path):
    """consolidate_entity_facts passes max_facts to call_fact_consolidation."""
    from src.memory.store import consolidate_entity_facts, create_entity
    from src.core.models import EntityFrontmatter

    fm = EntityFrontmatter(
        title="Test Entity",
        type="ai_self",
        retention="long_term",
        score=0.5,
        importance=0.7,
        frequency=5,
        last_mentioned="2026-03-09",
        created="2026-01-01",
        aliases=[],
        tags=[],
    )
    # Create entity with many facts
    obs = [{"category": "ai_style", "content": f"Fact number {i}", "tags": []} for i in range(25)]
    filepath = create_entity(tmp_path, "self", "test-entity", fm, observations=obs)

    mock_result = MagicMock()
    mock_result.consolidated = [
        MagicMock(content="Consolidated fact", category="ai_style", date="", valence="", tags=[])
    ]

    config = Config(max_facts={"default": 50, "ai_self": 20})

    with patch("src.core.llm.call_fact_consolidation", return_value=mock_result) as mock_call:
        consolidate_entity_facts(filepath, config, max_facts=20)
        # Verify max_facts was passed through
        mock_call.assert_called_once()
        _, kwargs = mock_call.call_args
        assert kwargs["max_facts"] == 20


def test_update_entity_hard_cap(tmp_path):
    """update_entity truncates facts when way over max_facts * 2."""
    from src.memory.store import create_entity, update_entity, read_entity
    from src.core.models import EntityFrontmatter

    fm = EntityFrontmatter(
        title="Big Entity",
        type="ai_self",
        retention="long_term",
        score=0.5,
        importance=0.7,
        frequency=5,
        last_mentioned="2026-03-09",
        created="2026-01-01",
        aliases=[],
        tags=[],
    )
    # Create entity with 50 facts (> 20 * 2 = 40 threshold)
    obs = [{"category": "ai_style", "content": f"Unique fact {i}", "tags": []} for i in range(50)]
    filepath = create_entity(tmp_path, "self", "big-entity", fm, observations=obs)

    # Add more observations with max_facts=20 -> hard cap at 40
    new_obs = [{"category": "ai_style", "content": f"New fact {i}", "tags": []} for i in range(5)]
    update_entity(filepath, new_observations=new_obs, max_facts=20)

    # Read back and check
    _, sections = read_entity(filepath)
    live_facts = [f for f in sections.get("Facts", []) if "[superseded]" not in f]
    assert len(live_facts) <= 40  # max_facts * 2
