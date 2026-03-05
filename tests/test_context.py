"""Tests for memory/context.py."""

from pathlib import Path
from unittest.mock import patch

from src.core.config import Config, ScoringConfig
from src.core.models import EntityFrontmatter, GraphData, GraphEntity, GraphRelation
from src.memory.context import build_context_input, build_deterministic_context, generate_index, write_index
from src.memory.store import write_entity


def _make_config(memory_path):
    config = Config.__new__(Config)
    config.scoring = ScoringConfig(min_score_for_context=0.0)
    config.memory_path = memory_path
    config.context_max_tokens = 3000
    config.context_budget = {"identity": 10, "top_of_mind": 25}
    config.user_language = "fr"
    return config


def test_build_context_input(tmp_path):
    config = _make_config(tmp_path)

    # Create entity files
    fm1 = EntityFrontmatter(
        title="Mal de dos", type="health", score=0.8,
        importance=0.9, frequency=10,
        last_mentioned="2026-03-03", created="2025-01-01",
        tags=["santé", "douleur"],
    )
    write_entity(
        tmp_path / "moi" / "mal-de-dos.md", fm1,
        {"Facts": ["- [diagnosis] Sciatique chronique"], "Relations": [], "History": []},
    )

    graph = GraphData()
    graph.entities["mal-de-dos"] = GraphEntity(
        file="moi/mal-de-dos.md", type="health", title="Mal de dos",
        score=0.8, importance=0.9, frequency=10,
        last_mentioned="2026-03-03", tags=["santé", "douleur"],
    )

    enriched = build_context_input(graph, tmp_path, config)
    assert "Mal de dos" in enriched
    assert "Sciatique chronique" in enriched


def test_generate_index():
    graph = GraphData(generated="2026-03-03")
    graph.entities["test"] = GraphEntity(
        file="moi/test.md", type="health", title="Test Entity",
        score=0.5, frequency=3, last_mentioned="2026-03-03",
    )
    graph.entities["other"] = GraphEntity(
        file="interets/other.md", type="interest", title="Other",
        score=0.3, frequency=1, last_mentioned="2026-01-01",
    )
    graph.relations.append(GraphRelation(from_entity="test", to_entity="other", type="affects"))

    index = generate_index(graph)
    assert "Memory Index" in index
    assert "Test Entity" in index
    assert "Other" in index
    assert "affects" in index
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
        type="health",
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


def test_deterministic_context_has_all_sections(tmp_path):
    """Context output should have all required sections."""
    # Create minimal memory structure
    (tmp_path / "self").mkdir()

    graph = GraphData(generated="2026-03-05")
    graph.entities["me"] = GraphEntity(
        file="self/me.md", type="health", title="Me",
        score=0.8, importance=0.9, summary="Identity summary.",
    )

    # Write a minimal MD file
    fm = EntityFrontmatter(
        title="Me", type="health", score=0.8, importance=0.9,
    )
    write_entity(
        tmp_path / "self" / "me.md", fm,
        {"Facts": [], "Relations": [], "History": []},
    )

    config = _make_config(tmp_path)

    result = build_deterministic_context(graph, tmp_path, config)
    assert "# Memory Context" in result
    assert "## Identity" in result
    assert "## Top of mind" in result
    assert "## Vigilances" in result
    assert "## Work & Projects" in result
    assert "## Close ones" in result
    assert "## Available in memory" in result
    assert "## Memory tags" in result


def test_deterministic_context_categorization(tmp_path):
    """Entities should be categorized correctly by type and file path."""
    (tmp_path / "self").mkdir()
    (tmp_path / "close_ones").mkdir()
    (tmp_path / "projects").mkdir()

    graph = GraphData(generated="2026-03-05")

    # Identity entity (self/ path)
    graph.entities["me"] = GraphEntity(
        file="self/me.md", type="health", title="Me",
        score=0.9, importance=0.9, summary="The user.",
    )
    fm_me = EntityFrontmatter(title="Me", type="health", score=0.9, importance=0.9)
    write_entity(tmp_path / "self" / "me.md", fm_me,
                 {"Facts": [], "Relations": [], "History": []})

    # Person entity
    graph.entities["alice"] = GraphEntity(
        file="close_ones/alice.md", type="person", title="Alice",
        score=0.7, importance=0.6, summary="A close friend.",
    )
    fm_alice = EntityFrontmatter(title="Alice", type="person", score=0.7, importance=0.6)
    write_entity(tmp_path / "close_ones" / "alice.md", fm_alice,
                 {"Facts": [], "Relations": [], "History": []})

    # Project entity
    graph.entities["proj"] = GraphEntity(
        file="projects/proj.md", type="project", title="MyProject",
        score=0.6, importance=0.5, summary="A coding project.",
    )
    fm_proj = EntityFrontmatter(title="MyProject", type="project", score=0.6, importance=0.5)
    write_entity(tmp_path / "projects" / "proj.md", fm_proj,
                 {"Facts": [], "Relations": [], "History": []})

    config = _make_config(tmp_path)
    result = build_deterministic_context(graph, tmp_path, config)

    # Identity section should contain "The user."
    assert "The user." in result
    # Close ones section should contain Alice
    assert "Alice" in result
    # Work & Projects should contain MyProject
    assert "MyProject" in result


def test_deterministic_context_vigilance(tmp_path):
    """Entities with vigilance/diagnosis facts should appear in Vigilances section."""
    (tmp_path / "health").mkdir()

    graph = GraphData(generated="2026-03-05")
    graph.entities["back-pain"] = GraphEntity(
        file="health/back-pain.md", type="health", title="Back Pain",
        score=0.8, importance=0.9,
    )

    fm = EntityFrontmatter(title="Back Pain", type="health", score=0.8, importance=0.9)
    write_entity(tmp_path / "health" / "back-pain.md", fm,
                 {"Facts": ["- [vigilance] Chronic sciatica requires monitoring"],
                  "Relations": [], "History": []})

    config = _make_config(tmp_path)
    result = build_deterministic_context(graph, tmp_path, config)
    assert "Chronic sciatica" in result
    assert "## Vigilances" in result


def test_deterministic_context_tags(tmp_path):
    """Tags from entities should appear in Memory tags section."""
    (tmp_path / "interests").mkdir()

    graph = GraphData(generated="2026-03-05")
    graph.entities["python"] = GraphEntity(
        file="interests/python.md", type="interest", title="Python",
        score=0.6, importance=0.5, tags=["coding", "language"],
    )

    fm = EntityFrontmatter(title="Python", type="interest", score=0.6, importance=0.5,
                           tags=["coding", "language"])
    write_entity(tmp_path / "interests" / "python.md", fm,
                 {"Facts": [], "Relations": [], "History": []})

    config = _make_config(tmp_path)
    result = build_deterministic_context(graph, tmp_path, config)
    assert "#coding" in result or "#language" in result


def test_write_index(tmp_path):
    graph = GraphData(generated="2026-03-03")
    graph.entities["test"] = GraphEntity(
        file="moi/test.md", type="health", title="Test",
        score=0.5, frequency=1, last_mentioned="2026-03-03",
    )
    write_index(tmp_path, graph)
    assert (tmp_path / "_index.md").exists()
    content = (tmp_path / "_index.md").read_text()
    assert "Test" in content
