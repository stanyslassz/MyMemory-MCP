"""Tests for pipeline/keyword_index.py — SQLite FTS5 keyword search."""

from pathlib import Path

import pytest

from src.core.models import EntityFrontmatter
from src.memory.store import init_memory_structure, write_entity
from src.pipeline.keyword_index import build_keyword_index, search_keyword


def _create_entity(tmp_path, folder, slug, title, facts):
    """Helper to create an entity MD file."""
    fm = EntityFrontmatter(
        title=title,
        type="person",
        created="2026-01-01",
        last_mentioned="2026-03-01",
    )
    path = tmp_path / folder / f"{slug}.md"
    write_entity(path, fm, {"Facts": facts, "Relations": [], "History": []})
    return path


def test_build_and_search_keyword_index(tmp_path):
    """Build FTS5 index from test entities, search by exact name."""
    init_memory_structure(tmp_path)

    _create_entity(
        tmp_path, "close_ones", "dr-martin",
        "Dr. Martin",
        ["- [fact] Médecin traitant depuis 2020", "- [fact] Spécialiste en rhumatologie"],
    )
    _create_entity(
        tmp_path, "close_ones", "sophie",
        "Sophie",
        ["- [fact] Collègue de travail"],
    )

    db_path = tmp_path / "_memory_fts.db"
    count = build_keyword_index(tmp_path, db_path)
    assert count >= 2  # At least one chunk per entity
    assert db_path.exists()

    # Search by name
    results = search_keyword("Dr. Martin", db_path, top_k=5)
    assert len(results) >= 1
    assert results[0].entity_id == "dr-martin"
    assert results[0].bm25_score > 0


def test_search_keyword_proper_names(tmp_path):
    """Proper names like 'Dr. Martin' should match via FTS5."""
    init_memory_structure(tmp_path)

    _create_entity(
        tmp_path, "close_ones", "dr-martin",
        "Dr. Martin",
        ["- [fact] Médecin traitant", "- [fact] Cabinet rue de la Paix"],
    )
    _create_entity(
        tmp_path, "interests", "act-r",
        "ACT-R",
        ["- [fact] Cognitive architecture for memory scoring"],
    )

    db_path = tmp_path / "_memory_fts.db"
    build_keyword_index(tmp_path, db_path)

    # Exact proper name
    results = search_keyword("Martin", db_path, top_k=5)
    assert any(r.entity_id == "dr-martin" for r in results)

    # Abbreviation
    results = search_keyword("ACT-R", db_path, top_k=5)
    assert any(r.entity_id == "act-r" for r in results)


def test_search_keyword_empty_db(tmp_path):
    """Searching a non-existent DB returns empty list."""
    db_path = tmp_path / "nonexistent.db"
    results = search_keyword("anything", db_path)
    assert results == []


def test_build_keyword_index_empty_memory(tmp_path):
    """Building index with no entity files produces zero chunks."""
    init_memory_structure(tmp_path)
    db_path = tmp_path / "_memory_fts.db"
    count = build_keyword_index(tmp_path, db_path)
    assert count == 0


def test_build_keyword_index_skips_chats(tmp_path):
    """Chat files should not be indexed."""
    init_memory_structure(tmp_path)
    chats_dir = tmp_path / "chats"
    chats_dir.mkdir(exist_ok=True)
    (chats_dir / "chat-001.md").write_text("---\nprocessed: true\n---\nSome chat content")

    db_path = tmp_path / "_memory_fts.db"
    count = build_keyword_index(tmp_path, db_path)
    assert count == 0


def test_build_keyword_index_skips_underscore_files(tmp_path):
    """Files starting with _ should not be indexed."""
    init_memory_structure(tmp_path)
    (tmp_path / "_context.md").write_text("Context content here")
    (tmp_path / "_index.md").write_text("Index content here")

    db_path = tmp_path / "_memory_fts.db"
    count = build_keyword_index(tmp_path, db_path)
    assert count == 0


def test_search_keyword_individual_terms_fallback(tmp_path):
    """When phrase match fails, individual terms should still match."""
    init_memory_structure(tmp_path)

    _create_entity(
        tmp_path, "close_ones", "jean-dupont",
        "Jean Dupont",
        ["- [fact] Ami proche depuis le lycée"],
    )

    db_path = tmp_path / "_memory_fts.db"
    build_keyword_index(tmp_path, db_path)

    # Search for individual term
    results = search_keyword("Jean", db_path, top_k=5)
    assert any(r.entity_id == "jean-dupont" for r in results)
