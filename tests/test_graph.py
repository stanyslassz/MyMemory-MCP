"""Tests for memory/graph.py."""

import json
from pathlib import Path

from src.core.models import EntityFrontmatter, GraphData, GraphEntity, GraphRelation
from src.memory.graph import (
    add_entity,
    add_relation,
    get_aliases_lookup,
    get_related,
    load_graph,
    rebuild_from_md,
    remove_orphan_relations,
    save_graph,
    validate_graph,
)
from src.memory.store import write_entity


def _make_entity(title="Test", type_="health", score=0.5, aliases=None):
    return GraphEntity(
        file=f"moi/{title.lower().replace(' ', '-')}.md",
        type=type_,
        title=title,
        score=score,
        aliases=aliases or [],
    )


def test_load_save_graph(tmp_path):
    graph = GraphData(generated="2026-03-03")
    graph = add_entity(graph, "test", _make_entity())

    save_graph(tmp_path, graph)
    assert (tmp_path / "_graph.json").exists()

    loaded = load_graph(tmp_path)
    assert "test" in loaded.entities
    assert loaded.entities["test"].title == "Test"


def test_load_nonexistent_graph(tmp_path):
    graph = load_graph(tmp_path)
    assert len(graph.entities) == 0


def test_backup_on_save(tmp_path):
    graph = GraphData()
    graph = add_entity(graph, "v1", _make_entity("V1"))
    save_graph(tmp_path, graph)

    graph = add_entity(graph, "v2", _make_entity("V2"))
    save_graph(tmp_path, graph)

    assert (tmp_path / "_graph.json.bak").exists()


def test_add_relation_dedup(tmp_path):
    graph = GraphData()
    graph = add_entity(graph, "a", _make_entity("A"))
    graph = add_entity(graph, "b", _make_entity("B"))

    rel = GraphRelation(from_entity="a", to_entity="b", type="affects")
    graph = add_relation(graph, rel)
    graph = add_relation(graph, rel)  # duplicate
    assert len(graph.relations) == 1


def test_remove_orphan_relations():
    graph = GraphData()
    graph = add_entity(graph, "a", _make_entity("A"))
    rel = GraphRelation(from_entity="a", to_entity="nonexistent", type="affects")
    graph.relations.append(rel)

    graph = remove_orphan_relations(graph)
    assert len(graph.relations) == 0


def test_get_related():
    graph = GraphData()
    graph = add_entity(graph, "a", _make_entity("A"))
    graph = add_entity(graph, "b", _make_entity("B"))
    graph = add_entity(graph, "c", _make_entity("C"))

    graph = add_relation(graph, GraphRelation(from_entity="a", to_entity="b", type="affects"))
    graph = add_relation(graph, GraphRelation(from_entity="b", to_entity="c", type="improves"))

    # Depth 1
    related = get_related(graph, "a", depth=1)
    assert "b" in related
    assert "c" not in related

    # Depth 2
    related_d2 = get_related(graph, "a", depth=2)
    assert "b" in related_d2
    assert "c" in related_d2


def test_get_related_bidirectional():
    graph = GraphData()
    graph = add_entity(graph, "a", _make_entity("A"))
    graph = add_entity(graph, "b", _make_entity("B"))
    graph = add_relation(graph, GraphRelation(from_entity="a", to_entity="b", type="affects"))

    # Should find 'a' when starting from 'b' (reverse traversal)
    related = get_related(graph, "b", depth=1)
    assert "a" in related


def test_get_aliases_lookup():
    graph = GraphData()
    graph = add_entity(graph, "mal-de-dos", _make_entity("Mal de dos", aliases=["dos", "sciatique"]))

    lookup = get_aliases_lookup(graph)
    assert lookup["dos"] == "mal-de-dos"
    assert lookup["sciatique"] == "mal-de-dos"
    assert lookup["mal de dos"] == "mal-de-dos"


def test_validate_graph(tmp_path):
    graph = GraphData()
    graph = add_entity(graph, "test", GraphEntity(
        file="moi/test.md", type="health", title="Test"
    ))
    # File doesn't exist
    warnings = validate_graph(graph, tmp_path)
    assert len(warnings) >= 1
    assert "missing file" in warnings[0]


def test_load_corrupt_graph_restores_from_bak(tmp_path):
    """Corrupt _graph.json should be recovered from .bak."""
    graph = GraphData()
    graph = add_entity(graph, "entity-a", _make_entity("Entity A"))
    save_graph(tmp_path, graph)

    # Corrupt primary file
    (tmp_path / "_graph.json").write_text("{invalid json!!", encoding="utf-8")
    # .bak should still be valid from previous save
    (tmp_path / "_graph.json.bak").write_text(
        json.dumps(graph.model_dump(by_alias=True), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    loaded = load_graph(tmp_path)
    assert "entity-a" in loaded.entities


def test_load_corrupt_graph_and_bak_rebuilds_from_md(tmp_path):
    """If both _graph.json and .bak are corrupt, rebuild from MDs."""
    # Create a valid entity MD file
    fm = EntityFrontmatter(
        title="Rebuilt Entity",
        type="health",
        created="2025-01-01",
        last_mentioned="2026-03-01",
    )
    write_entity(
        tmp_path / "moi" / "rebuilt-entity.md",
        fm,
        {"Facts": ["- [fact] A fact"], "Relations": [], "History": []},
    )

    # Write corrupt primary and backup
    (tmp_path / "_graph.json").write_text("CORRUPT", encoding="utf-8")
    (tmp_path / "_graph.json.bak").write_text("ALSO CORRUPT", encoding="utf-8")

    loaded = load_graph(tmp_path)
    assert "rebuilt-entity" in loaded.entities


def test_save_graph_is_atomic(tmp_path):
    """Verify save produces valid JSON even if tested naively."""
    graph = GraphData()
    graph = add_entity(graph, "x", _make_entity("X"))
    save_graph(tmp_path, graph)

    # Read back raw and verify it's valid JSON
    raw = (tmp_path / "_graph.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    assert "entities" in data
    assert "x" in data["entities"]


def test_add_relation_reinforces_existing():
    graph = GraphData()
    graph.entities["a"] = GraphEntity(file="self/a.md", type="health", title="A")
    graph.entities["b"] = GraphEntity(file="self/b.md", type="health", title="B")

    rel1 = GraphRelation(from_entity="a", to_entity="b", type="affects",
                         strength=0.5, mention_count=1, created="2026-01-01")
    add_relation(graph, rel1)
    assert len(graph.relations) == 1
    assert graph.relations[0].mention_count == 1

    rel2 = GraphRelation(from_entity="a", to_entity="b", type="affects",
                         context="new context")
    add_relation(graph, rel2)
    assert len(graph.relations) == 1  # Still 1, not duplicated
    assert graph.relations[0].mention_count == 2  # Reinforced
    assert graph.relations[0].context == "new context"
    assert graph.relations[0].last_reinforced != ""


def test_rebuild_from_md(tmp_path):
    # Create entity MD files
    fm1 = EntityFrontmatter(
        title="Entity One",
        type="health",
        score=0.5,
        frequency=3,
        last_mentioned="2026-03-03",
        created="2025-01-01",
        aliases=["e1"],
    )
    write_entity(
        tmp_path / "moi" / "entity-one.md",
        fm1,
        {
            "Facts": ["- [fact] A fact"],
            "Relations": ["- affects [[Entity Two]]"],
            "History": [],
        },
    )

    fm2 = EntityFrontmatter(
        title="Entity Two",
        type="interest",
        created="2025-01-01",
        last_mentioned="2026-01-01",
    )
    write_entity(
        tmp_path / "interets" / "entity-two.md",
        fm2,
        {"Facts": [], "Relations": [], "History": []},
    )

    graph = rebuild_from_md(tmp_path)
    assert "entity-one" in graph.entities
    assert "entity-two" in graph.entities
    assert graph.entities["entity-one"].title == "Entity One"
    # Should have parsed the relation
    assert len(graph.relations) >= 1
    assert graph.relations[0].from_entity == "entity-one"


def test_rebuild_from_md_reads_mention_dates(tmp_path):
    """rebuild_from_md must read mention_dates, monthly_buckets, created from frontmatter."""
    entity_dir = tmp_path / "close_ones"
    entity_dir.mkdir(parents=True)
    (entity_dir / "sophie.md").write_text(
        "---\n"
        "title: Sophie\n"
        "type: person\n"
        "retention: short_term\n"
        "score: 0.0\n"
        "importance: 0.5\n"
        "frequency: 2\n"
        "last_mentioned: '2026-03-05'\n"
        "created: '2026-03-03'\n"
        "aliases: []\n"
        "tags: [relation]\n"
        "mention_dates:\n"
        "- '2026-03-03'\n"
        "- '2026-03-05'\n"
        "monthly_buckets: {}\n"
        "summary: ''\n"
        "---\n\n# Sophie\n\n## Facts\n\n- [fact] Infirmière\n\n## Relations\n\n## History\n",
        encoding="utf-8",
    )

    from src.memory.graph import rebuild_from_md
    graph = rebuild_from_md(tmp_path)

    sophie = graph.entities["sophie"]
    assert sophie.mention_dates == ["2026-03-03", "2026-03-05"]
    assert sophie.created == "2026-03-03"
    assert sophie.monthly_buckets == {}
