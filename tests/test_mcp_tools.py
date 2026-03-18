"""Tests for MCP CRUD tools (Phase 2): delete_fact, delete_relation, modify_fact, correct_entity."""

import json
from datetime import datetime
from pathlib import Path

from src.core.config import Config, CategoriesConfig
from src.core.models import EntityFrontmatter, GraphData, GraphEntity, GraphRelation
from src.memory.graph import load_graph, save_graph, remove_relation
from src.memory.store import (
    create_entity,
    read_entity,
    write_entity,
    remove_relation_line,
)
from src.memory.graph import find_entity_by_name as _resolve_entity_by_name
from src.mcp.server import (
    _delete_fact_impl,
    _delete_relation_impl,
    _modify_fact_impl,
    _correct_entity_impl,
)


# ── Helpers ──────────────────────────────────────────────────


def _make_entity(tmp_path, slug="back-pain", title="Back Pain", entity_type="health",
                 facts=None, relations=None):
    """Create a test entity MD file and return its path."""
    fm = EntityFrontmatter(
        title=title,
        type=entity_type,
        retention="long_term",
        score=0.5,
        importance=0.7,
        frequency=3,
        last_mentioned="2026-03-07",
        created="2025-09-15",
        aliases=["sciatica", "back issue"],
        tags=["health"],
    )
    sections = {
        "Facts": facts or [
            "- [diagnosis] (2024-03) Chronic sciatica [-]",
            "- [treatment] (2025-11) Started physiotherapy [+]",
            "- [fact] Regular monitoring needed",
        ],
        "Relations": relations or [
            "- affects [[Daily Routine]]",
            "- improves [[Swimming]]",
        ],
        "History": ["- 2025-09-15: Created"],
    }
    filepath = tmp_path / "self" / f"{slug}.md"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    write_entity(filepath, fm, sections)
    return filepath


def _make_graph(tmp_path, entities=None, relations=None):
    """Create and save a test graph, return GraphData."""
    graph = GraphData(generated=datetime.now().isoformat())
    if entities:
        for eid, e in entities.items():
            graph.entities[eid] = e
    if relations:
        graph.relations = relations
    save_graph(tmp_path, graph)
    return graph


def _make_config(tmp_path):
    """Create a minimal Config pointing to tmp_path as memory."""
    return Config(
        memory_path=tmp_path,
        categories=CategoriesConfig(
            folders={
                "person": "close_ones",
                "health": "self",
                "work": "work",
                "project": "projects",
                "interest": "interests",
                "ai_self": "self",
                "place": "interests",
                "animal": "close_ones",
                "organization": "work",
            }
        ),
    )


# ── _resolve_entity_by_name ─────────────────────────────────


class TestResolveEntityByName:
    def test_resolve_by_slug(self, tmp_path):
        graph = GraphData(generated="now")
        graph.entities["back-pain"] = GraphEntity(
            file="self/back-pain.md", type="health", title="Back Pain",
            score=0.5, importance=0.7, frequency=3, last_mentioned="2026-03-07",
        )
        assert _resolve_entity_by_name("back-pain", graph) == "back-pain"

    def test_resolve_by_title(self, tmp_path):
        graph = GraphData(generated="now")
        graph.entities["back-pain"] = GraphEntity(
            file="self/back-pain.md", type="health", title="Back Pain",
            score=0.5, importance=0.7, frequency=3, last_mentioned="2026-03-07",
        )
        assert _resolve_entity_by_name("Back Pain", graph) == "back-pain"

    def test_resolve_by_alias(self, tmp_path):
        graph = GraphData(generated="now")
        graph.entities["back-pain"] = GraphEntity(
            file="self/back-pain.md", type="health", title="Back Pain",
            score=0.5, importance=0.7, frequency=3, last_mentioned="2026-03-07",
            aliases=["sciatica"],
        )
        assert _resolve_entity_by_name("sciatica", graph) == "back-pain"

    def test_resolve_not_found(self, tmp_path):
        graph = GraphData(generated="now")
        assert _resolve_entity_by_name("nonexistent", graph) is None

    def test_resolve_case_insensitive(self, tmp_path):
        graph = GraphData(generated="now")
        graph.entities["back-pain"] = GraphEntity(
            file="self/back-pain.md", type="health", title="Back Pain",
            score=0.5, importance=0.7, frequency=3, last_mentioned="2026-03-07",
            aliases=["Sciatica"],
        )
        assert _resolve_entity_by_name("SCIATICA", graph) == "back-pain"


# ── delete_fact ──────────────────────────────────────────────


class TestDeleteFact:
    def test_delete_fact_happy_path(self, tmp_path):
        _make_entity(tmp_path)
        graph = _make_graph(tmp_path, entities={
            "back-pain": GraphEntity(
                file="self/back-pain.md", type="health", title="Back Pain",
                score=0.5, importance=0.7, frequency=3, last_mentioned="2026-03-07",
                aliases=["sciatica"],
            ),
        })
        config = _make_config(tmp_path)

        result = json.loads(_delete_fact_impl("Back Pain", "Chronic sciatica", config))
        assert result["status"] == "deleted"
        assert "Chronic sciatica" in result["deleted_fact"]

        # Verify fact is gone from MD
        fm, sections = read_entity(tmp_path / "self" / "back-pain.md")
        fact_texts = "\n".join(sections.get("Facts", []))
        assert "Chronic sciatica" not in fact_texts

    def test_delete_fact_entity_not_found(self, tmp_path):
        graph = _make_graph(tmp_path)
        config = _make_config(tmp_path)

        result = json.loads(_delete_fact_impl("Nonexistent", "whatever", config))
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    def test_delete_fact_fact_not_found(self, tmp_path):
        _make_entity(tmp_path)
        graph = _make_graph(tmp_path, entities={
            "back-pain": GraphEntity(
                file="self/back-pain.md", type="health", title="Back Pain",
                score=0.5, importance=0.7, frequency=3, last_mentioned="2026-03-07",
            ),
        })
        config = _make_config(tmp_path)

        result = json.loads(_delete_fact_impl("Back Pain", "nonexistent fact content", config))
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    def test_delete_fact_partial_match(self, tmp_path):
        _make_entity(tmp_path)
        graph = _make_graph(tmp_path, entities={
            "back-pain": GraphEntity(
                file="self/back-pain.md", type="health", title="Back Pain",
                score=0.5, importance=0.7, frequency=3, last_mentioned="2026-03-07",
            ),
        })
        config = _make_config(tmp_path)

        # Partial match: "physiotherapy" should match "Started physiotherapy"
        result = json.loads(_delete_fact_impl("Back Pain", "physiotherapy", config))
        assert result["status"] == "deleted"

    def test_delete_fact_adds_history(self, tmp_path):
        _make_entity(tmp_path)
        graph = _make_graph(tmp_path, entities={
            "back-pain": GraphEntity(
                file="self/back-pain.md", type="health", title="Back Pain",
                score=0.5, importance=0.7, frequency=3, last_mentioned="2026-03-07",
            ),
        })
        config = _make_config(tmp_path)

        _delete_fact_impl("Back Pain", "monitoring", config)
        fm, sections = read_entity(tmp_path / "self" / "back-pain.md")
        history = "\n".join(sections.get("History", []))
        assert "Deleted fact" in history


# ── delete_relation ──────────────────────────────────────────


class TestDeleteRelation:
    def test_delete_relation_happy_path(self, tmp_path):
        _make_entity(tmp_path)
        _make_entity(tmp_path, slug="daily-routine", title="Daily Routine",
                     entity_type="interest", facts=["- [fact] Routine info"],
                     relations=[])
        graph = _make_graph(tmp_path, entities={
            "back-pain": GraphEntity(
                file="self/back-pain.md", type="health", title="Back Pain",
                score=0.5, importance=0.7, frequency=3, last_mentioned="2026-03-07",
            ),
            "daily-routine": GraphEntity(
                file="self/daily-routine.md", type="interest", title="Daily Routine",
                score=0.3, importance=0.5, frequency=1, last_mentioned="2026-03-07",
            ),
        }, relations=[
            GraphRelation(from_entity="back-pain", to_entity="daily-routine", type="affects"),
        ])
        config = _make_config(tmp_path)

        result = json.loads(_delete_relation_impl("Back Pain", "Daily Routine", "affects", config))
        assert result["status"] == "deleted"

        # Verify relation removed from graph
        graph = load_graph(tmp_path)
        matching = [r for r in graph.relations
                    if r.from_entity == "back-pain" and r.to_entity == "daily-routine" and r.type == "affects"]
        assert len(matching) == 0

    def test_delete_relation_entity_not_found(self, tmp_path):
        graph = _make_graph(tmp_path)
        config = _make_config(tmp_path)

        result = json.loads(_delete_relation_impl("Nonexistent", "Other", "affects", config))
        assert result["status"] == "error"

    def test_delete_relation_not_found(self, tmp_path):
        _make_entity(tmp_path)
        _make_entity(tmp_path, slug="daily-routine", title="Daily Routine",
                     entity_type="interest", facts=["- [fact] Routine info"],
                     relations=[])
        graph = _make_graph(tmp_path, entities={
            "back-pain": GraphEntity(
                file="self/back-pain.md", type="health", title="Back Pain",
                score=0.5, importance=0.7, frequency=3, last_mentioned="2026-03-07",
            ),
            "daily-routine": GraphEntity(
                file="self/daily-routine.md", type="interest", title="Daily Routine",
                score=0.3, importance=0.5, frequency=1, last_mentioned="2026-03-07",
            ),
        })
        config = _make_config(tmp_path)

        result = json.loads(_delete_relation_impl("Back Pain", "Daily Routine", "worsens", config))
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()


# ── modify_fact ──────────────────────────────────────────────


class TestModifyFact:
    def test_modify_fact_happy_path(self, tmp_path):
        _make_entity(tmp_path)
        graph = _make_graph(tmp_path, entities={
            "back-pain": GraphEntity(
                file="self/back-pain.md", type="health", title="Back Pain",
                score=0.5, importance=0.7, frequency=3, last_mentioned="2026-03-07",
            ),
        })
        config = _make_config(tmp_path)

        result = json.loads(_modify_fact_impl(
            "Back Pain", "Chronic sciatica", "Chronic sciatica with L5-S1 herniation", config
        ))
        assert result["status"] == "modified"

        fm, sections = read_entity(tmp_path / "self" / "back-pain.md")
        fact_texts = "\n".join(sections.get("Facts", []))
        assert "L5-S1 herniation" in fact_texts
        # Metadata preserved
        assert "[diagnosis]" in fact_texts
        assert "(2024-03)" in fact_texts
        assert "[-]" in fact_texts

    def test_modify_fact_entity_not_found(self, tmp_path):
        graph = _make_graph(tmp_path)
        config = _make_config(tmp_path)

        result = json.loads(_modify_fact_impl("Nonexistent", "old", "new", config))
        assert result["status"] == "error"

    def test_modify_fact_fact_not_found(self, tmp_path):
        _make_entity(tmp_path)
        graph = _make_graph(tmp_path, entities={
            "back-pain": GraphEntity(
                file="self/back-pain.md", type="health", title="Back Pain",
                score=0.5, importance=0.7, frequency=3, last_mentioned="2026-03-07",
            ),
        })
        config = _make_config(tmp_path)

        result = json.loads(_modify_fact_impl("Back Pain", "nonexistent", "new", config))
        assert result["status"] == "error"

    def test_modify_fact_preserves_tags(self, tmp_path):
        facts = ["- [fact] Important info [+] #health #chronic"]
        _make_entity(tmp_path, facts=facts)
        graph = _make_graph(tmp_path, entities={
            "back-pain": GraphEntity(
                file="self/back-pain.md", type="health", title="Back Pain",
                score=0.5, importance=0.7, frequency=3, last_mentioned="2026-03-07",
            ),
        })
        config = _make_config(tmp_path)

        result = json.loads(_modify_fact_impl("Back Pain", "Important info", "Updated info", config))
        assert result["status"] == "modified"

        fm, sections = read_entity(tmp_path / "self" / "back-pain.md")
        fact_line = sections["Facts"][0]
        assert "Updated info" in fact_line
        assert "[+]" in fact_line
        assert "#health" in fact_line


# ── correct_entity ───────────────────────────────────────────


class TestCorrectEntity:
    def test_correct_title(self, tmp_path):
        _make_entity(tmp_path)
        graph = _make_graph(tmp_path, entities={
            "back-pain": GraphEntity(
                file="self/back-pain.md", type="health", title="Back Pain",
                score=0.5, importance=0.7, frequency=3, last_mentioned="2026-03-07",
            ),
        })
        config = _make_config(tmp_path)

        result = json.loads(_correct_entity_impl("Back Pain", "title", "Lower Back Pain", config))
        assert result["status"] == "updated"

        graph = load_graph(tmp_path)
        assert graph.entities["back-pain"].title == "Lower Back Pain"

    def test_correct_type_moves_file(self, tmp_path):
        _make_entity(tmp_path, slug="swimming", title="Swimming",
                     entity_type="health", facts=["- [fact] Good exercise"])
        graph = _make_graph(tmp_path, entities={
            "swimming": GraphEntity(
                file="self/swimming.md", type="health", title="Swimming",
                score=0.5, importance=0.7, frequency=3, last_mentioned="2026-03-07",
            ),
        })
        config = _make_config(tmp_path)

        result = json.loads(_correct_entity_impl("Swimming", "type", "interest", config))
        assert result["status"] == "updated"

        # File should have moved
        assert (tmp_path / "interests" / "swimming.md").exists()
        assert not (tmp_path / "self" / "swimming.md").exists()

        # Graph should reflect new file path
        graph = load_graph(tmp_path)
        assert graph.entities["swimming"].file == "interests/swimming.md"
        assert graph.entities["swimming"].type == "interest"

    def test_correct_aliases(self, tmp_path):
        _make_entity(tmp_path)
        graph = _make_graph(tmp_path, entities={
            "back-pain": GraphEntity(
                file="self/back-pain.md", type="health", title="Back Pain",
                score=0.5, importance=0.7, frequency=3, last_mentioned="2026-03-07",
                aliases=["sciatica"],
            ),
        })
        config = _make_config(tmp_path)

        result = json.loads(_correct_entity_impl(
            "Back Pain", "aliases", "sciatica, lumbar pain, herniated disc", config
        ))
        assert result["status"] == "updated"

        graph = load_graph(tmp_path)
        assert "lumbar pain" in graph.entities["back-pain"].aliases
        assert "herniated disc" in graph.entities["back-pain"].aliases

    def test_correct_retention(self, tmp_path):
        _make_entity(tmp_path)
        graph = _make_graph(tmp_path, entities={
            "back-pain": GraphEntity(
                file="self/back-pain.md", type="health", title="Back Pain",
                score=0.5, importance=0.7, frequency=3, last_mentioned="2026-03-07",
            ),
        })
        config = _make_config(tmp_path)

        result = json.loads(_correct_entity_impl("Back Pain", "retention", "permanent", config))
        assert result["status"] == "updated"

        fm, _ = read_entity(tmp_path / "self" / "back-pain.md")
        assert fm.retention == "permanent"

    def test_correct_entity_not_found(self, tmp_path):
        graph = _make_graph(tmp_path)
        config = _make_config(tmp_path)

        result = json.loads(_correct_entity_impl("Nonexistent", "title", "New", config))
        assert result["status"] == "error"

    def test_correct_invalid_field(self, tmp_path):
        _make_entity(tmp_path)
        graph = _make_graph(tmp_path, entities={
            "back-pain": GraphEntity(
                file="self/back-pain.md", type="health", title="Back Pain",
                score=0.5, importance=0.7, frequency=3, last_mentioned="2026-03-07",
            ),
        })
        config = _make_config(tmp_path)

        result = json.loads(_correct_entity_impl("Back Pain", "score", "0.9", config))
        assert result["status"] == "error"
        assert "invalid field" in result["message"].lower()


# ── remove_relation (graph.py) ───────────────────────────────


class TestRemoveRelation:
    def test_remove_existing_relation(self):
        graph = GraphData(generated="now")
        graph.relations = [
            GraphRelation(from_entity="a", to_entity="b", type="affects"),
            GraphRelation(from_entity="a", to_entity="c", type="linked_to"),
        ]
        assert remove_relation(graph, "a", "b", "affects") is True
        assert len(graph.relations) == 1
        assert graph.relations[0].to_entity == "c"

    def test_remove_nonexistent_relation(self):
        graph = GraphData(generated="now")
        graph.relations = [
            GraphRelation(from_entity="a", to_entity="b", type="affects"),
        ]
        assert remove_relation(graph, "a", "b", "linked_to") is False
        assert len(graph.relations) == 1


# ── remove_relation_line (store.py) ──────────────────────────


class TestRemoveRelationLine:
    def test_remove_existing_line(self, tmp_path):
        _make_entity(tmp_path)
        filepath = tmp_path / "self" / "back-pain.md"
        assert remove_relation_line(filepath, "affects", "Daily Routine") is True

        _, sections = read_entity(filepath)
        rels = "\n".join(sections.get("Relations", []))
        assert "Daily Routine" not in rels
        assert "Swimming" in rels

    def test_remove_nonexistent_line(self, tmp_path):
        _make_entity(tmp_path)
        filepath = tmp_path / "self" / "back-pain.md"
        assert remove_relation_line(filepath, "worsens", "Daily Routine") is False
