"""Integration tests for MCP CRUD operations.

Tests the _impl functions from src.mcp.server against real file I/O.
Each test is standalone (no conftest.py, no shared fixtures — project convention).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.core.config import CategoriesConfig, Config
from src.core.models import EntityFrontmatter, GraphData, GraphEntity, GraphRelation
from src.mcp.server import (
    _correct_entity_impl,
    _delete_fact_impl,
    _delete_relation_impl,
    _modify_fact_impl,
)
from src.memory.event_log import append_event, read_events
from src.memory.graph import save_graph
from src.memory.store import read_entity, write_entity


def _make_config(tmp_path: Path) -> Config:
    """Create a minimal Config pointing at tmp_path as memory_path."""
    memory_path = tmp_path / "memory"
    memory_path.mkdir(parents=True, exist_ok=True)
    return Config(
        memory_path=memory_path,
        categories=CategoriesConfig(
            folders={
                "person": "close_ones",
                "health": "self",
                "work": "work",
                "project": "projects",
                "interest": "interests",
                "place": "interests",
                "animal": "close_ones",
                "organization": "work",
                "ai_self": "self",
            },
        ),
    )


def _make_entity_md(
    memory_path: Path,
    folder: str,
    slug: str,
    title: str,
    entity_type: str,
    facts: list[str] | None = None,
    relations: list[str] | None = None,
) -> Path:
    """Create an entity MD file and return its path."""
    filepath = memory_path / folder / f"{slug}.md"
    fm = EntityFrontmatter(
        title=title,
        type=entity_type,
        retention="short_term",
        score=0.5,
        importance=0.5,
        frequency=3,
        last_mentioned="2026-03-19",
        created="2026-01-01",
        aliases=[],
        tags=[],
    )
    sections = {
        "Facts": facts or [],
        "Relations": relations or [],
        "History": [f"- 2026-01-01: Created"],
    }
    write_entity(filepath, fm, sections)
    return filepath


def _make_graph(
    memory_path: Path,
    entities: dict[str, GraphEntity],
    relations: list[GraphRelation] | None = None,
) -> GraphData:
    """Create and save a graph, return the GraphData."""
    graph = GraphData(
        generated=datetime.now().isoformat(),
        entities=entities,
        relations=relations or [],
    )
    save_graph(memory_path, graph)
    return graph


# ── test_delete_fact ────────────────────────────────────────────


def test_delete_fact(tmp_path: Path):
    """Delete one fact from an entity with 3 facts; verify removal and remaining facts."""
    config = _make_config(tmp_path)
    memory_path = config.memory_path

    facts = [
        "- [fact] Started swimming regularly",
        "- [diagnosis] (2025-11) Chronic sciatica [-]",
        "- [treatment] (2026-01) Physiotherapy sessions [+]",
    ]
    _make_entity_md(memory_path, "self", "back-pain", "Back Pain", "health", facts=facts)

    entity = GraphEntity(
        file="self/back-pain.md",
        type="health",
        title="Back Pain",
        score=0.5,
        frequency=3,
    )
    _make_graph(memory_path, {"back-pain": entity})

    result_json = _delete_fact_impl("Back Pain", "Chronic sciatica", config)
    result = json.loads(result_json)

    assert result["status"] == "deleted"
    assert "Back Pain" in result["entity"]
    assert "sciatica" in result["deleted_fact"].lower()

    # Verify file contents
    fm, sections = read_entity(memory_path / "self" / "back-pain.md")
    remaining_facts = sections["Facts"]
    assert len(remaining_facts) == 2
    # The two remaining facts should be swimming and physiotherapy
    contents = " ".join(remaining_facts).lower()
    assert "swimming" in contents
    assert "physiotherapy" in contents
    assert "sciatica" not in contents

    # History entry recorded
    history = sections["History"]
    assert any("Deleted fact" in h for h in history)


# ── test_delete_fact_not_found ──────────────────────────────────


def test_delete_fact_not_found(tmp_path: Path):
    """Deleting a nonexistent fact returns an error."""
    config = _make_config(tmp_path)
    memory_path = config.memory_path

    facts = ["- [fact] Some fact"]
    _make_entity_md(memory_path, "self", "back-pain", "Back Pain", "health", facts=facts)

    entity = GraphEntity(file="self/back-pain.md", type="health", title="Back Pain")
    _make_graph(memory_path, {"back-pain": entity})

    result = json.loads(_delete_fact_impl("Back Pain", "nonexistent content", config))
    assert result["status"] == "error"
    assert "not found" in result["message"].lower()


# ── test_modify_fact ────────────────────────────────────────────


def test_modify_fact(tmp_path: Path):
    """Modify a fact's content and verify metadata (category, date, valence) is preserved."""
    config = _make_config(tmp_path)
    memory_path = config.memory_path

    facts = [
        "- [treatment] (2025-11) Started physiotherapy [+] #health",
    ]
    _make_entity_md(memory_path, "self", "back-pain", "Back Pain", "health", facts=facts)

    entity = GraphEntity(file="self/back-pain.md", type="health", title="Back Pain")
    _make_graph(memory_path, {"back-pain": entity})

    result_json = _modify_fact_impl("Back Pain", "physiotherapy", "Switched to osteopathy", config)
    result = json.loads(result_json)

    assert result["status"] == "modified"
    assert "Back Pain" in result["entity"]

    # Verify file: content changed, metadata preserved
    fm, sections = read_entity(memory_path / "self" / "back-pain.md")
    modified_line = sections["Facts"][0]
    assert "osteopathy" in modified_line.lower()
    assert "[treatment]" in modified_line  # category preserved
    assert "(2025-11)" in modified_line  # date preserved
    assert "[+]" in modified_line  # valence preserved
    assert "#health" in modified_line  # tag preserved

    # History entry recorded
    assert any("Modified fact" in h for h in sections["History"])


# ── test_correct_entity_type ────────────────────────────────────


def test_correct_entity_type(tmp_path: Path):
    """Change entity type from interest to health and verify file move + graph update."""
    config = _make_config(tmp_path)
    memory_path = config.memory_path

    facts = ["- [fact] Some yoga practice"]
    _make_entity_md(memory_path, "interests", "yoga", "Yoga", "interest", facts=facts)

    entity = GraphEntity(
        file="interests/yoga.md",
        type="interest",
        title="Yoga",
        score=0.5,
    )
    _make_graph(memory_path, {"yoga": entity})

    result_json = _correct_entity_impl("Yoga", "type", "health", config)
    result = json.loads(result_json)

    assert result["status"] == "updated"
    assert result["changes"]["field"] == "type"
    assert result["changes"]["old"] == "interest"
    assert result["changes"]["new"] == "health"
    assert "self/yoga.md" in result["changes"]["moved_to"]

    # Old file should be gone, new file should exist
    assert not (memory_path / "interests" / "yoga.md").exists()
    new_path = memory_path / "self" / "yoga.md"
    assert new_path.exists()

    # Verify frontmatter in new file
    fm, sections = read_entity(new_path)
    assert fm.type == "health"

    # Verify graph updated
    from src.memory.graph import load_graph
    graph = load_graph(memory_path)
    assert graph.entities["yoga"].type == "health"
    assert graph.entities["yoga"].file == "self/yoga.md"

    # History entry recorded
    assert any("Corrected type" in h for h in sections["History"])


# ── test_correct_entity_title ───────────────────────────────────


def test_correct_entity_title(tmp_path: Path):
    """Change an entity's title and verify graph + frontmatter updated."""
    config = _make_config(tmp_path)
    memory_path = config.memory_path

    _make_entity_md(memory_path, "close_ones", "alice", "Alice", "person")

    entity = GraphEntity(file="close_ones/alice.md", type="person", title="Alice")
    _make_graph(memory_path, {"alice": entity})

    result = json.loads(_correct_entity_impl("Alice", "title", "Alice Dupont", config))

    assert result["status"] == "updated"
    assert result["entity"] == "Alice Dupont"

    fm, _ = read_entity(memory_path / "close_ones" / "alice.md")
    assert fm.title == "Alice Dupont"

    from src.memory.graph import load_graph
    graph = load_graph(memory_path)
    assert graph.entities["alice"].title == "Alice Dupont"


# ── test_correct_entity_aliases ─────────────────────────────────


def test_correct_entity_aliases(tmp_path: Path):
    """Set aliases via comma-separated string."""
    config = _make_config(tmp_path)
    memory_path = config.memory_path

    _make_entity_md(memory_path, "self", "back-pain", "Back Pain", "health")

    entity = GraphEntity(file="self/back-pain.md", type="health", title="Back Pain")
    _make_graph(memory_path, {"back-pain": entity})

    result = json.loads(_correct_entity_impl("Back Pain", "aliases", "sciatica, lumbar pain", config))

    assert result["status"] == "updated"

    from src.memory.graph import load_graph
    graph = load_graph(memory_path)
    assert "sciatica" in graph.entities["back-pain"].aliases
    assert "lumbar pain" in graph.entities["back-pain"].aliases


# ── test_correct_entity_invalid_field ───────────────────────────


def test_correct_entity_invalid_field(tmp_path: Path):
    """Attempting to correct an invalid field returns an error."""
    config = _make_config(tmp_path)

    result = json.loads(_correct_entity_impl("Whatever", "score", "0.9", config))
    assert result["status"] == "error"
    assert "Invalid field" in result["message"]


# ── test_delete_relation ────────────────────────────────────────


def test_delete_relation(tmp_path: Path):
    """Delete a relation between two entities; verify removal from graph and MD."""
    config = _make_config(tmp_path)
    memory_path = config.memory_path

    # Create two entities with a relation in the MD
    _make_entity_md(
        memory_path, "self", "back-pain", "Back Pain", "health",
        facts=["- [fact] Chronic condition"],
        relations=["- affects [[Daily Routine]]"],
    )
    _make_entity_md(
        memory_path, "interests", "daily-routine", "Daily Routine", "interest",
        facts=["- [fact] Morning exercises"],
    )

    entities = {
        "back-pain": GraphEntity(file="self/back-pain.md", type="health", title="Back Pain"),
        "daily-routine": GraphEntity(file="interests/daily-routine.md", type="interest", title="Daily Routine"),
    }
    relation = GraphRelation(**{
        "from": "back-pain",
        "to": "daily-routine",
        "type": "affects",
        "strength": 0.5,
        "created": "2026-01-01",
        "last_reinforced": "2026-03-01",
        "mention_count": 2,
    })
    _make_graph(memory_path, entities, relations=[relation])

    result_json = _delete_relation_impl("Back Pain", "Daily Routine", "affects", config)
    result = json.loads(result_json)

    assert result["status"] == "deleted"
    assert result["type"] == "affects"

    # Verify relation removed from graph
    from src.memory.graph import load_graph
    graph = load_graph(memory_path)
    matching = [r for r in graph.relations if r.from_entity == "back-pain" and r.to_entity == "daily-routine"]
    assert len(matching) == 0

    # Verify relation removed from entity MD
    _, sections = read_entity(memory_path / "self" / "back-pain.md")
    rel_lines = sections.get("Relations", [])
    assert not any("Daily Routine" in line for line in rel_lines)


# ── test_delete_relation_not_found ──────────────────────────────


def test_delete_relation_not_found(tmp_path: Path):
    """Deleting a nonexistent relation returns an error."""
    config = _make_config(tmp_path)
    memory_path = config.memory_path

    _make_entity_md(memory_path, "self", "back-pain", "Back Pain", "health")
    _make_entity_md(memory_path, "interests", "swimming", "Swimming", "interest")

    entities = {
        "back-pain": GraphEntity(file="self/back-pain.md", type="health", title="Back Pain"),
        "swimming": GraphEntity(file="interests/swimming.md", type="interest", title="Swimming"),
    }
    _make_graph(memory_path, entities, relations=[])

    result = json.loads(_delete_relation_impl("Back Pain", "Swimming", "affects", config))
    assert result["status"] == "error"
    assert "not found" in result["message"].lower()


# ── test_suggest_correction ─────────────────────────────────────


def test_suggest_correction(tmp_path: Path):
    """suggest_correction logs an event and it can be retrieved via read_events."""
    memory_path = tmp_path / "memory"
    memory_path.mkdir(parents=True, exist_ok=True)

    # Simulate what suggest_correction does internally
    append_event(memory_path, "correction_suggested", "mcp", {
        "entity": "Back Pain",
        "issue": "User mentioned sciatica is resolved",
        "suggested_fix": "Update diagnosis to resolved",
    })

    events = read_events(memory_path, event_type="correction_suggested")

    assert len(events) == 1
    event = events[0]
    assert event["type"] == "correction_suggested"
    assert event["source"] == "mcp"
    assert event["data"]["entity"] == "Back Pain"
    assert event["data"]["issue"] == "User mentioned sciatica is resolved"
    assert event["data"]["suggested_fix"] == "Update diagnosis to resolved"


# ── test_suggest_correction_multiple ────────────────────────────


def test_suggest_correction_multiple(tmp_path: Path):
    """Multiple suggestions are all stored and retrievable."""
    memory_path = tmp_path / "memory"
    memory_path.mkdir(parents=True, exist_ok=True)

    for i in range(3):
        append_event(memory_path, "correction_suggested", "mcp", {
            "entity": f"Entity {i}",
            "issue": f"Issue {i}",
            "suggested_fix": f"Fix {i}",
        })

    events = read_events(memory_path, event_type="correction_suggested")
    assert len(events) == 3

    # Other event types should not appear
    append_event(memory_path, "other_event", "pipeline", {"foo": "bar"})
    events = read_events(memory_path, event_type="correction_suggested")
    assert len(events) == 3  # Still 3, the other_event is filtered out


# ── test_entity_not_found ───────────────────────────────────────


def test_entity_not_found(tmp_path: Path):
    """Operations on nonexistent entities return proper errors."""
    config = _make_config(tmp_path)
    memory_path = config.memory_path

    # Create an empty graph
    _make_graph(memory_path, {})

    result = json.loads(_delete_fact_impl("Nonexistent", "some fact", config))
    assert result["status"] == "error"
    assert "not found" in result["message"].lower()

    result = json.loads(_modify_fact_impl("Nonexistent", "old", "new", config))
    assert result["status"] == "error"

    result = json.loads(_correct_entity_impl("Nonexistent", "title", "New Title", config))
    assert result["status"] == "error"
