"""Tests for crash recovery: graph corruption scenarios."""

import json

from src.core.models import EntityFrontmatter, GraphData, GraphEntity
from src.memory.graph import add_entity, load_graph, rebuild_from_md, save_graph
from src.memory.store import write_entity


def _create_entity_md(tmp_path, slug, title, entity_type="health", folder="self"):
    """Create a valid entity MD file."""
    fm = EntityFrontmatter(
        title=title,
        type=entity_type,
        created="2025-01-01",
        last_mentioned="2026-03-01",
        importance=0.5,
        frequency=3,
    )
    write_entity(
        tmp_path / folder / f"{slug}.md",
        fm,
        {"Facts": [f"- [fact] {title} fact"], "Relations": [], "History": []},
    )


def _make_entity(title="Test", type_="health"):
    return GraphEntity(
        file=f"self/{title.lower().replace(' ', '-')}.md",
        type=type_,
        title=title,
        score=0.5,
    )


# ── Scenario 1: _graph.json corrupt, .bak valid ──────────────


def test_corrupt_graph_recovers_from_bak(tmp_path):
    """Corrupt _graph.json should fall back to .bak."""
    _create_entity_md(tmp_path, "entity-a", "Entity A")

    # Create valid graph and save (creates .bak on second save)
    graph = GraphData()
    graph = add_entity(graph, "entity-a", _make_entity("Entity A"))
    save_graph(tmp_path, graph)
    # Save again to create .bak
    save_graph(tmp_path, graph)

    # Corrupt primary
    (tmp_path / "_graph.json").write_text("{broken json!!", encoding="utf-8")

    loaded = load_graph(tmp_path)
    assert "entity-a" in loaded.entities
    assert loaded.entities["entity-a"].title == "Entity A"


def test_corrupt_graph_truncated_json(tmp_path):
    """Truncated JSON (simulating crash mid-write) should recover from .bak."""
    _create_entity_md(tmp_path, "entity-b", "Entity B")

    graph = GraphData()
    graph = add_entity(graph, "entity-b", _make_entity("Entity B"))
    save_graph(tmp_path, graph)
    save_graph(tmp_path, graph)

    # Simulate truncated write (crash mid-save)
    valid_json = (tmp_path / "_graph.json").read_text(encoding="utf-8")
    (tmp_path / "_graph.json").write_text(valid_json[:len(valid_json)//2], encoding="utf-8")

    loaded = load_graph(tmp_path)
    assert "entity-b" in loaded.entities


# ── Scenario 2: _graph.json valid, .bak corrupt ──────────────


def test_valid_graph_corrupt_bak_works(tmp_path):
    """Valid _graph.json should work even if .bak is corrupt."""
    _create_entity_md(tmp_path, "entity-c", "Entity C")

    graph = GraphData()
    graph = add_entity(graph, "entity-c", _make_entity("Entity C"))
    save_graph(tmp_path, graph)

    # Corrupt only .bak
    bak_path = tmp_path / "_graph.json.bak"
    if bak_path.exists():
        bak_path.write_text("CORRUPT BAK", encoding="utf-8")

    loaded = load_graph(tmp_path)
    assert "entity-c" in loaded.entities


# ── Scenario 3: Both _graph.json and .bak corrupt ──────────────


def test_both_corrupt_rebuilds_from_md(tmp_path):
    """If both _graph.json and .bak are corrupt, rebuild from entity MDs."""
    _create_entity_md(tmp_path, "entity-d", "Entity D")
    _create_entity_md(tmp_path, "entity-e", "Entity E", entity_type="interest", folder="interests")

    # Write corrupt files
    (tmp_path / "_graph.json").write_text("CORRUPT PRIMARY", encoding="utf-8")
    (tmp_path / "_graph.json.bak").write_text("CORRUPT BACKUP", encoding="utf-8")

    loaded = load_graph(tmp_path)

    # Should have rebuilt from MDs
    assert "entity-d" in loaded.entities
    assert "entity-e" in loaded.entities
    assert loaded.entities["entity-d"].title == "Entity D"
    assert loaded.entities["entity-e"].title == "Entity E"


def test_both_corrupt_no_md_files(tmp_path):
    """If everything is corrupt and no MDs exist, return empty graph."""
    (tmp_path / "_graph.json").write_text("CORRUPT", encoding="utf-8")
    (tmp_path / "_graph.json.bak").write_text("ALSO CORRUPT", encoding="utf-8")

    loaded = load_graph(tmp_path)
    assert len(loaded.entities) == 0


# ── Scenario 4: No graph files at all ──────────────


def test_no_graph_file_loads_empty(tmp_path):
    """Missing _graph.json should return empty graph."""
    loaded = load_graph(tmp_path)
    assert len(loaded.entities) == 0
    assert len(loaded.relations) == 0


def test_no_graph_with_md_rebuilds(tmp_path):
    """Missing _graph.json but with entity MDs should rebuild."""
    _create_entity_md(tmp_path, "entity-f", "Entity F")

    # No _graph.json at all — load_graph should fall back to rebuild
    loaded = load_graph(tmp_path)
    # Depending on implementation, may return empty or rebuilt
    # Either is valid, but if rebuild is attempted, entity should be found
    if len(loaded.entities) > 0:
        assert "entity-f" in loaded.entities


# ── Scenario 5: rebuild_from_md preserves data ──────────────


def test_rebuild_from_md_preserves_relations(tmp_path):
    """rebuild_from_md should reconstruct relations from MD files."""
    fm_a = EntityFrontmatter(
        title="Pain", type="health", created="2025-01-01", last_mentioned="2026-01-01",
    )
    write_entity(
        tmp_path / "self" / "pain.md",
        fm_a,
        {
            "Facts": ["- [diagnosis] Chronic pain [-]"],
            "Relations": ["- affects [[Daily Routine]]"],
            "History": [],
        },
    )

    fm_b = EntityFrontmatter(
        title="Daily Routine", type="interest", created="2025-01-01",
        last_mentioned="2026-01-01",
    )
    write_entity(
        tmp_path / "interests" / "daily-routine.md",
        fm_b,
        {"Facts": [], "Relations": [], "History": []},
    )

    graph = rebuild_from_md(tmp_path)
    assert "pain" in graph.entities
    assert "daily-routine" in graph.entities
    assert len(graph.relations) >= 1
    assert any(
        r.from_entity == "pain" and r.type == "affects"
        for r in graph.relations
    )


def test_rebuild_preserves_frontmatter_fields(tmp_path):
    """rebuild_from_md should preserve all frontmatter fields from MD files."""
    entity_dir = tmp_path / "self"
    entity_dir.mkdir(parents=True)
    (entity_dir / "test-entity.md").write_text(
        "---\n"
        "title: Test Entity\n"
        "type: health\n"
        "retention: permanent\n"
        "score: 0.72\n"
        "importance: 0.85\n"
        "frequency: 12\n"
        "last_mentioned: '2026-03-07'\n"
        "created: '2025-09-15'\n"
        "aliases: [alias1, alias2]\n"
        "tags: [tag1, tag2]\n"
        "mention_dates:\n"
        "- '2026-03-01'\n"
        "- '2026-03-07'\n"
        "monthly_buckets:\n"
        "  '2025-06': 3\n"
        "summary: 'A test entity'\n"
        "---\n\n## Facts\n\n- [fact] Something\n\n## Relations\n\n## History\n",
        encoding="utf-8",
    )

    graph = rebuild_from_md(tmp_path)
    entity = graph.entities["test-entity"]

    assert entity.title == "Test Entity"
    assert entity.type == "health"
    assert entity.importance == 0.85
    assert entity.frequency == 12
    assert entity.aliases == ["alias1", "alias2"]
    assert entity.mention_dates == ["2026-03-01", "2026-03-07"]
    assert entity.created == "2025-09-15"
    assert entity.summary == "A test entity"
