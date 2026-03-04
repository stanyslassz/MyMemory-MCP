"""Tests for memory/context.py."""

from pathlib import Path
from unittest.mock import patch

from src.core.config import Config, ScoringConfig
from src.core.models import EntityFrontmatter, GraphData, GraphEntity, GraphRelation
from src.memory.context import build_context_input, generate_index, write_index
from src.memory.store import write_entity


def _make_config(memory_path):
    config = Config.__new__(Config)
    config.scoring = ScoringConfig(min_score_for_context=0.0)
    config.memory_path = memory_path
    config.context_max_tokens = 3000
    config.context_budget = {"identite": 10, "top_of_mind": 25}
    config.user_language = "fr"
    return config


def test_build_context_input(tmp_path):
    config = _make_config(tmp_path)

    # Create entity files
    fm1 = EntityFrontmatter(
        title="Mal de dos", type="sante", score=0.8,
        importance=0.9, frequency=10,
        last_mentioned="2026-03-03", created="2025-01-01",
        tags=["santé", "douleur"],
    )
    write_entity(
        tmp_path / "moi" / "mal-de-dos.md", fm1,
        {"Faits": ["- [diagnostic] Sciatique chronique"], "Relations": [], "Historique": []},
    )

    graph = GraphData()
    graph.entities["mal-de-dos"] = GraphEntity(
        file="moi/mal-de-dos.md", type="sante", title="Mal de dos",
        score=0.8, importance=0.9, frequency=10,
        last_mentioned="2026-03-03", tags=["santé", "douleur"],
    )

    enriched = build_context_input(graph, tmp_path, config)
    assert "Mal de dos" in enriched
    assert "Sciatique chronique" in enriched


def test_generate_index():
    graph = GraphData(generated="2026-03-03")
    graph.entities["test"] = GraphEntity(
        file="moi/test.md", type="sante", title="Test Entity",
        score=0.5, frequency=3, last_mentioned="2026-03-03",
    )
    graph.entities["other"] = GraphEntity(
        file="interets/other.md", type="interet", title="Other",
        score=0.3, frequency=1, last_mentioned="2026-01-01",
    )
    graph.relations.append(GraphRelation(from_entity="test", to_entity="other", type="affecte"))

    index = generate_index(graph)
    assert "Memory Index" in index
    assert "Test Entity" in index
    assert "Other" in index
    assert "affecte" in index
    assert "Total entities: 2" in index


def test_path_traversal_entity_file_blocked(tmp_path):
    """Entity with path-traversal entity.file must not read outside memory_path."""
    config = _make_config(tmp_path)

    # Create a sensitive file outside memory_path
    secret_file = tmp_path.parent / "secret.txt"
    secret_file.write_text("API_KEY=sk-should-not-leak", encoding="utf-8")

    graph = GraphData()
    graph.entities["evil"] = GraphEntity(
        file="../secret.txt",  # path traversal
        type="sante",
        title="Evil",
        score=0.99,
        importance=1.0,
        frequency=100,
        last_mentioned="2026-03-04",
    )

    enriched = build_context_input(graph, tmp_path, config)
    # The traversal entity should be skipped entirely — no secret content leaked
    assert "sk-should-not-leak" not in enriched
    assert "API_KEY" not in enriched


def test_write_index(tmp_path):
    graph = GraphData(generated="2026-03-03")
    graph.entities["test"] = GraphEntity(
        file="moi/test.md", type="sante", title="Test",
        score=0.5, frequency=1, last_mentioned="2026-03-03",
    )
    write_index(tmp_path, graph)
    assert (tmp_path / "_index.md").exists()
    content = (tmp_path / "_index.md").read_text()
    assert "Test" in content
