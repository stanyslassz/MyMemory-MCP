"""Tests for memory/context.py."""

from pathlib import Path
from unittest.mock import patch

from src.core.config import Config, ScoringConfig
from src.core.models import EntityFrontmatter, GraphData, GraphEntity, GraphRelation
from src.memory.context import build_context, generate_index, write_index
from src.memory.store import write_entity


def _make_config(tmp_path):
    config = Config.__new__(Config)
    config.scoring = ScoringConfig(min_score_for_context=0.0)
    config.memory_path = tmp_path
    config.context_max_tokens = 3000
    config.context_budget = {"identity": 10, "top_of_mind": 25}
    config.user_language = "fr"
    config.prompts_path = tmp_path / "prompts"
    config.context_narrative = False
    return config



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



def test_deterministic_context_has_all_sections(tmp_path):
    """Context output should have template structure with key sections."""
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

    result = build_context(graph, tmp_path, config)
    # Template-based structure
    assert "Personal Memory" in result
    assert "## Identity" in result


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

    # Project entity (type "project" goes to Top of mind since it's not "work" or "organization")
    graph.entities["proj"] = GraphEntity(
        file="projects/proj.md", type="project", title="MyProject",
        score=0.6, importance=0.5, summary="A coding project.",
    )
    fm_proj = EntityFrontmatter(title="MyProject", type="project", score=0.6, importance=0.5)
    write_entity(tmp_path / "projects" / "proj.md", fm_proj,
                 {"Facts": [], "Relations": [], "History": []})

    config = _make_config(tmp_path)
    result = build_context(graph, tmp_path, config)

    # Identity section should contain "Me" entity
    assert "## Identity" in result
    assert "Me" in result
    # Person entity should appear in Personal context
    assert "Alice" in result
    assert "## Personal context" in result
    # Project entity should appear in Top of mind (type "project" is not "work"/"organization")
    assert "MyProject" in result
    assert "## Top of mind" in result


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
    result = build_context(graph, tmp_path, config)
    assert "Chronic sciatica" in result
    assert "## Vigilances" in result


def test_build_context_uses_template(tmp_path):
    """build_context should use the template from prompts_path."""
    config = _make_config(tmp_path)

    # Create a custom template
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    template = "# My Memory — {date}\n\n{sections}\n\nEntities: {available_entities}\n{ai_personality}\n{user_language_name}\n{custom_instructions}"
    (prompts_dir / "context_template.md").write_text(template, encoding="utf-8")

    graph = GraphData(generated="2026-03-05")
    graph.entities["test-ent"] = GraphEntity(
        file="self/test.md", type="health", title="Test Entity",
        score=0.7, importance=0.5,
    )
    (tmp_path / "self").mkdir()
    fm = EntityFrontmatter(title="Test Entity", type="health", score=0.7, importance=0.5)
    write_entity(tmp_path / "self" / "test.md", fm,
                 {"Facts": [], "Relations": [], "History": []})

    result = build_context(graph, tmp_path, config)
    assert "# My Memory" in result
    assert "Test Entity" in result
    assert "French" in result


def test_facts_sorted_by_date_in_context(tmp_path):
    """Facts should be sorted chronologically within same category in context output."""
    (tmp_path / "self").mkdir()

    graph = GraphData(generated="2026-03-05")
    graph.entities["health-ent"] = GraphEntity(
        file="self/health-ent.md", type="health", title="Health Issue",
        score=0.8, importance=0.9,
    )

    fm = EntityFrontmatter(title="Health Issue", type="health", score=0.8, importance=0.9)
    write_entity(tmp_path / "self" / "health-ent.md", fm,
                 {"Facts": [
                     "- [fact] Undated fact",
                     "- [fact] (2025-11) Later fact [-]",
                     "- [fact] (2024-03) Earlier fact [+]",
                 ],
                  "Relations": [], "History": []})

    config = _make_config(tmp_path)
    result = build_context(graph, tmp_path, config)

    # Within same category, dated facts should appear chronologically, undated last
    idx_earlier = result.find("Earlier fact")
    idx_later = result.find("Later fact")
    idx_undated = result.find("Undated fact")
    assert idx_earlier < idx_later < idx_undated


def test_context_filters_weak_relations():
    """Context builder should exclude relations with strength below threshold."""
    from datetime import date

    relations = [
        GraphRelation(from_entity="alice", to_entity="bob", type="friend_of", strength=0.8, last_reinforced="2026-01-01"),
        GraphRelation(from_entity="alice", to_entity="carol", type="linked_to", strength=0.1, last_reinforced="2026-01-01"),  # Weak
        GraphRelation(from_entity="alice", to_entity="dave", type="affects", strength=0.5, last_reinforced="2024-01-01"),  # Stale (>1 year)
    ]

    min_strength = 0.3
    max_age_days = 365
    today = date(2026, 3, 11)

    filtered = []
    for r in relations:
        if r.strength < min_strength:
            continue
        if r.last_reinforced:
            try:
                last = date.fromisoformat(r.last_reinforced)
                if (today - last).days > max_age_days:
                    continue
            except (ValueError, TypeError):
                pass
        filtered.append(r)

    assert len(filtered) == 1  # Only bob's strong, recent relation
    assert filtered[0].to_entity == "bob"


def test_context_enrich_entity_excludes_weak_relations(tmp_path):
    """_enrich_entity should not include weak or stale relations in output."""
    from unittest.mock import patch
    from datetime import date

    (tmp_path / "self").mkdir()

    # Create entity file
    fm = EntityFrontmatter(title="Alice", type="person", score=0.8, importance=0.7)
    write_entity(tmp_path / "self" / "alice.md", fm,
                 {"Facts": ["- [fact] Some fact"], "Relations": [], "History": []})

    graph = GraphData(generated="2026-03-11")
    graph.entities["alice"] = GraphEntity(
        file="self/alice.md", type="person", title="Alice",
        score=0.8, importance=0.7,
    )
    graph.entities["bob"] = GraphEntity(
        file="self/bob.md", type="person", title="Bob",
        score=0.5, importance=0.5,
    )
    graph.entities["carol"] = GraphEntity(
        file="self/carol.md", type="person", title="Carol",
        score=0.5, importance=0.5,
    )
    graph.entities["dave"] = GraphEntity(
        file="self/dave.md", type="person", title="Dave",
        score=0.5, importance=0.5,
    )

    # Strong recent relation
    graph.relations.append(GraphRelation(
        from_entity="alice", to_entity="bob", type="friend_of",
        strength=0.8, last_reinforced="2026-01-01",
    ))
    # Weak relation (below 0.3 threshold)
    graph.relations.append(GraphRelation(
        from_entity="alice", to_entity="carol", type="linked_to",
        strength=0.1, last_reinforced="2026-01-01",
    ))
    # Stale relation (over 1 year old)
    graph.relations.append(GraphRelation(
        from_entity="alice", to_entity="dave", type="affects",
        strength=0.5, last_reinforced="2024-01-01",
    ))

    from src.memory.context import _enrich_entity
    result = _enrich_entity("alice", graph.entities["alice"], graph, tmp_path)

    # Bob's relation should be present (strong + recent)
    assert "Bob" in result
    # Carol's relation should be filtered (weak)
    assert "Carol" not in result
    # Dave's relation should be filtered (stale)
    assert "Dave" not in result


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
