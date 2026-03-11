"""Tests for exclusive relation conflict resolution in enricher.py."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

from src.core.models import GraphData, GraphEntity, GraphRelation
from src.memory.graph import add_relation
from src.memory.store import write_entity, read_entity
from src.core.models import EntityFrontmatter
from src.pipeline.enricher import _check_relation_conflicts, EXCLUSIVE_RELATIONS


def _make_graph_with_entities(memory_path: Path) -> tuple[GraphData, str, str]:
    """Create a graph with two entities and their MD files."""
    graph = GraphData(generated=datetime.now().isoformat())

    # Entity A
    entity_a = GraphEntity(
        file="close_ones/alice.md",
        type="person",
        title="Alice",
        score=0.5,
        importance=0.5,
        frequency=3,
        last_mentioned="2026-03-01",
        retention="long_term",
    )
    graph.entities["alice"] = entity_a

    # Entity B
    entity_b = GraphEntity(
        file="close_ones/bob.md",
        type="person",
        title="Bob",
        score=0.4,
        importance=0.4,
        frequency=2,
        last_mentioned="2026-03-01",
        retention="short_term",
    )
    graph.entities["bob"] = entity_b

    # Create MD files
    (memory_path / "close_ones").mkdir(parents=True, exist_ok=True)
    for slug, entity in [("alice", entity_a), ("bob", entity_b)]:
        fm = EntityFrontmatter(
            title=entity.title,
            type=entity.type,
            retention=entity.retention,
            score=entity.score,
            importance=entity.importance,
            frequency=entity.frequency,
            last_mentioned=entity.last_mentioned,
            created="2026-01-01",
        )
        write_entity(
            memory_path / entity.file,
            fm,
            {"Facts": [], "Relations": [], "History": []},
        )

    return graph, "alice", "bob"


def test_parent_of_removes_friend_of():
    """Adding parent_of should auto-remove existing friend_of between same pair."""
    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        graph, a, b = _make_graph_with_entities(memory_path)

        # Add friend_of relation
        rel_friend = GraphRelation(from_entity=a, to_entity=b, type="friend_of")
        add_relation(graph, rel_friend)
        assert any(
            r.from_entity == a and r.to_entity == b and r.type == "friend_of"
            for r in graph.relations
        )

        # Also add relation line to MD
        from src.memory.store import update_entity
        update_entity(memory_path / "close_ones/alice.md", new_relations=["- friend_of [[Bob]]"])

        # Now add parent_of
        rel_parent = GraphRelation(from_entity=a, to_entity=b, type="parent_of")
        add_relation(graph, rel_parent)
        _check_relation_conflicts(graph, a, b, "parent_of", memory_path)

        # friend_of should be gone from graph
        assert not any(
            r.from_entity == a and r.to_entity == b and r.type == "friend_of"
            for r in graph.relations
        ), "friend_of should have been removed"

        # parent_of should still exist
        assert any(
            r.from_entity == a and r.to_entity == b and r.type == "parent_of"
            for r in graph.relations
        )

        # MD file should no longer have friend_of
        _, sections = read_entity(memory_path / "close_ones/alice.md")
        rel_lines = sections.get("Relations", [])
        assert not any("friend_of" in line for line in rel_lines), \
            f"friend_of should be removed from MD, got: {rel_lines}"


def test_worsens_removes_improves():
    """Adding worsens should auto-remove existing improves between same pair."""
    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        graph, a, b = _make_graph_with_entities(memory_path)

        # Add improves relation
        rel_improves = GraphRelation(from_entity=a, to_entity=b, type="improves")
        add_relation(graph, rel_improves)

        # Add relation line to MD
        from src.memory.store import update_entity
        update_entity(memory_path / "close_ones/alice.md", new_relations=["- improves [[Bob]]"])

        # Now add worsens
        rel_worsens = GraphRelation(from_entity=a, to_entity=b, type="worsens")
        add_relation(graph, rel_worsens)
        _check_relation_conflicts(graph, a, b, "worsens", memory_path)

        # improves should be gone
        assert not any(
            r.from_entity == a and r.to_entity == b and r.type == "improves"
            for r in graph.relations
        )

        # worsens should still exist
        assert any(
            r.from_entity == a and r.to_entity == b and r.type == "worsens"
            for r in graph.relations
        )


def test_affects_does_not_remove_anything():
    """Adding affects (not in any exclusive family) should leave other relations intact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        graph, a, b = _make_graph_with_entities(memory_path)

        # Add friend_of and improves
        add_relation(graph, GraphRelation(from_entity=a, to_entity=b, type="friend_of"))
        add_relation(graph, GraphRelation(from_entity=a, to_entity=b, type="improves"))
        assert len(graph.relations) == 2

        # Now add affects
        add_relation(graph, GraphRelation(from_entity=a, to_entity=b, type="affects"))
        _check_relation_conflicts(graph, a, b, "affects", memory_path)

        # All 3 relations should still exist
        types = {r.type for r in graph.relations}
        assert "friend_of" in types
        assert "improves" in types
        assert "affects" in types
