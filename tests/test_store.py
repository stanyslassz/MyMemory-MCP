"""Tests for memory/store.py."""

from pathlib import Path

from src.core.models import EntityFrontmatter
from src.memory.store import (
    create_entity,
    create_stub_entity,
    get_chat_content,
    init_memory_structure,
    list_entities,
    list_unprocessed_chats,
    mark_chat_processed,
    read_entity,
    save_chat,
    update_entity,
    write_entity,
)


def test_init_memory_structure(tmp_path):
    init_memory_structure(tmp_path / "mem")
    assert (tmp_path / "mem" / "self").is_dir()
    assert (tmp_path / "mem" / "chats").is_dir()
    assert (tmp_path / "mem" / "_inbox" / "_processed").is_dir()


def test_write_and_read_entity(tmp_path):
    fm = EntityFrontmatter(
        title="Test Entity",
        type="health",
        retention="long_term",
        score=0.5,
        importance=0.7,
        frequency=3,
        last_mentioned="2026-03-03",
        created="2025-09-15",
        aliases=["test", "alias"],
        tags=["tag1"],
    )
    sections = {
        "Facts": ["- [fact] Some fact", "- [diagnosis] A diagnosis"],
        "Relations": ["- affects [[Other]]"],
        "History": ["- 2025-09: Created"],
    }
    filepath = tmp_path / "moi" / "test-entity.md"
    write_entity(filepath, fm, sections)

    assert filepath.exists()

    fm2, sections2 = read_entity(filepath)
    assert fm2.title == "Test Entity"
    assert fm2.type == "health"
    assert fm2.retention == "long_term"
    assert fm2.frequency == 3
    assert len(sections2["Facts"]) == 2
    assert len(sections2["Relations"]) == 1


def test_update_entity(tmp_path):
    fm = EntityFrontmatter(
        title="Update Test",
        type="interest",
        frequency=1,
        last_mentioned="2026-01-01",
        created="2026-01-01",
    )
    filepath = tmp_path / "interets" / "update-test.md"
    write_entity(filepath, fm, {"Facts": ["- [fact] Old fact"], "Relations": [], "History": []})

    updated_fm = update_entity(
        filepath,
        new_observations=[{"category": "fact", "content": "New fact", "tags": []}],
        new_relations=["- improves [[Something]]"],
        last_mentioned="2026-03-03",
    )
    assert updated_fm.frequency == 2
    assert updated_fm.last_mentioned == "2026-03-03"

    fm3, sections3 = read_entity(filepath)
    assert len(sections3["Facts"]) == 2
    assert len(sections3["Relations"]) == 1


def test_create_entity(tmp_path):
    fm = EntityFrontmatter(
        title="New Entity",
        type="person",
        created="2026-03-03",
        last_mentioned="2026-03-03",
    )
    path = create_entity(
        tmp_path, "proches", "new-entity", fm,
        observations=[{"category": "fact", "content": "A person fact"}],
    )
    assert path.exists()
    fm2, sections = read_entity(path)
    assert fm2.title == "New Entity"
    assert len(sections["Facts"]) == 1


def test_create_stub_entity(tmp_path):
    path = create_stub_entity(tmp_path, "interets", "natation", "Natation", "interest", "2026-03-03")
    assert path.exists()
    fm, sections = read_entity(path)
    assert fm.title == "Natation"
    assert fm.importance == 0.3
    assert fm.retention == "short_term"


def test_save_and_list_chats(tmp_path):
    init_memory_structure(tmp_path)
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    filepath = save_chat(messages, tmp_path)
    assert filepath.exists()

    unprocessed = list_unprocessed_chats(tmp_path)
    assert len(unprocessed) == 1


def test_mark_chat_processed(tmp_path):
    init_memory_structure(tmp_path)
    messages = [{"role": "user", "content": "Test"}]
    filepath = save_chat(messages, tmp_path)

    mark_chat_processed(filepath, ["entity1"], ["entity2"])

    unprocessed = list_unprocessed_chats(tmp_path)
    assert len(unprocessed) == 0


def test_get_chat_content(tmp_path):
    init_memory_structure(tmp_path)
    messages = [{"role": "user", "content": "Important message"}]
    filepath = save_chat(messages, tmp_path)
    content = get_chat_content(filepath)
    assert "Important message" in content


def test_list_entities(tmp_path):
    fm = EntityFrontmatter(title="E1", type="health", created="2026-01-01", last_mentioned="2026-01-01")
    write_entity(tmp_path / "moi" / "e1.md", fm, {"Facts": [], "Relations": [], "History": []})

    fm2 = EntityFrontmatter(title="E2", type="person", created="2026-01-01", last_mentioned="2026-01-01")
    write_entity(tmp_path / "proches" / "e2.md", fm2, {"Facts": [], "Relations": [], "History": []})

    entities = list_entities(tmp_path)
    assert len(entities) == 2


def test_observation_with_date_and_valence(tmp_path):
    """Observations with date and valence should be stored in the expected format."""
    fm = EntityFrontmatter(title="DateTest", type="health", created="2026-01-01", last_mentioned="2026-01-01")
    filepath = tmp_path / "moi" / "date-test.md"
    write_entity(filepath, fm, {"Facts": [], "Relations": [], "History": []})

    update_entity(filepath, new_observations=[
        {"category": "diagnosis", "content": "Endometriosis diagnosed", "tags": ["health"],
         "date": "2024-03", "valence": "negative"},
        {"category": "fact", "content": "Started yoga", "tags": [],
         "date": "", "valence": "positive"},
        {"category": "fact", "content": "Regular checkups", "tags": [],
         "date": "2025-11-15", "valence": ""},
    ])

    _, sections = read_entity(filepath)
    facts = sections["Facts"]
    assert len(facts) == 3
    assert "(2024-03)" in facts[0]
    assert "[-]" in facts[0]
    assert "[+]" in facts[1]
    assert "(2025-11-15)" in facts[2]


def test_create_entity_with_date_valence(tmp_path):
    """create_entity should format observations with date and valence."""
    fm = EntityFrontmatter(title="New", type="health", created="2026-01-01", last_mentioned="2026-01-01")
    path = create_entity(
        tmp_path, "moi", "new", fm,
        observations=[
            {"category": "fact", "content": "Some fact", "date": "2025-06", "valence": "positive"},
        ],
    )
    _, sections = read_entity(path)
    assert "(2025-06)" in sections["Facts"][0]
    assert "[+]" in sections["Facts"][0]


def test_duplicate_observation_ignores_date_valence(tmp_path):
    """Duplicate check should match on category+content, ignoring date/valence."""
    fm = EntityFrontmatter(title="Dup2", type="health", created="2026-01-01", last_mentioned="2026-01-01")
    filepath = tmp_path / "moi" / "dup2.md"
    write_entity(filepath, fm, {
        "Facts": ["- [fact] (2024-03) Existing fact [+]"],
        "Relations": [], "History": [],
    })
    update_entity(filepath, new_observations=[
        {"category": "fact", "content": "Existing fact", "tags": [],
         "date": "2025-01", "valence": "negative"},
    ])
    _, sections = read_entity(filepath)
    assert len(sections["Facts"]) == 1  # Not duplicated


def test_duplicate_observation_skipped(tmp_path):
    fm = EntityFrontmatter(title="Dup Test", type="health", created="2026-01-01", last_mentioned="2026-01-01")
    filepath = tmp_path / "moi" / "dup.md"
    write_entity(filepath, fm, {"Facts": ["- [fact] Existing fact"], "Relations": [], "History": []})

    update_entity(filepath, new_observations=[{"category": "fact", "content": "Existing fact", "tags": []}])
    _, sections = read_entity(filepath)
    assert len(sections["Facts"]) == 1  # Not duplicated
