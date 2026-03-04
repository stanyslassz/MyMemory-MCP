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
    assert (tmp_path / "mem" / "moi").is_dir()
    assert (tmp_path / "mem" / "chats").is_dir()
    assert (tmp_path / "mem" / "_inbox" / "_processed").is_dir()


def test_write_and_read_entity(tmp_path):
    fm = EntityFrontmatter(
        title="Test Entity",
        type="sante",
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
        "Faits": ["- [fait] Some fact", "- [diagnostic] A diagnosis"],
        "Relations": ["- affecte [[Other]]"],
        "Historique": ["- 2025-09: Created"],
    }
    filepath = tmp_path / "moi" / "test-entity.md"
    write_entity(filepath, fm, sections)

    assert filepath.exists()

    fm2, sections2 = read_entity(filepath)
    assert fm2.title == "Test Entity"
    assert fm2.type == "sante"
    assert fm2.retention == "long_term"
    assert fm2.frequency == 3
    assert len(sections2["Faits"]) == 2
    assert len(sections2["Relations"]) == 1


def test_update_entity(tmp_path):
    fm = EntityFrontmatter(
        title="Update Test",
        type="interet",
        frequency=1,
        last_mentioned="2026-01-01",
        created="2026-01-01",
    )
    filepath = tmp_path / "interets" / "update-test.md"
    write_entity(filepath, fm, {"Faits": ["- [fait] Old fact"], "Relations": [], "Historique": []})

    updated_fm = update_entity(
        filepath,
        new_observations=[{"category": "fait", "content": "New fact", "tags": []}],
        new_relations=["- ameliore [[Something]]"],
        last_mentioned="2026-03-03",
    )
    assert updated_fm.frequency == 2
    assert updated_fm.last_mentioned == "2026-03-03"

    fm3, sections3 = read_entity(filepath)
    assert len(sections3["Faits"]) == 2
    assert len(sections3["Relations"]) == 1


def test_create_entity(tmp_path):
    fm = EntityFrontmatter(
        title="New Entity",
        type="personne",
        created="2026-03-03",
        last_mentioned="2026-03-03",
    )
    path = create_entity(
        tmp_path, "proches", "new-entity", fm,
        observations=[{"category": "fait", "content": "A person fact"}],
    )
    assert path.exists()
    fm2, sections = read_entity(path)
    assert fm2.title == "New Entity"
    assert len(sections["Faits"]) == 1


def test_create_stub_entity(tmp_path):
    path = create_stub_entity(tmp_path, "interets", "natation", "Natation", "interet", "2026-03-03")
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
    fm = EntityFrontmatter(title="E1", type="sante", created="2026-01-01", last_mentioned="2026-01-01")
    write_entity(tmp_path / "moi" / "e1.md", fm, {"Faits": [], "Relations": [], "Historique": []})

    fm2 = EntityFrontmatter(title="E2", type="personne", created="2026-01-01", last_mentioned="2026-01-01")
    write_entity(tmp_path / "proches" / "e2.md", fm2, {"Faits": [], "Relations": [], "Historique": []})

    entities = list_entities(tmp_path)
    assert len(entities) == 2


def test_duplicate_observation_skipped(tmp_path):
    fm = EntityFrontmatter(title="Dup Test", type="sante", created="2026-01-01", last_mentioned="2026-01-01")
    filepath = tmp_path / "moi" / "dup.md"
    write_entity(filepath, fm, {"Faits": ["- [fait] Existing fact"], "Relations": [], "Historique": []})

    update_entity(filepath, new_observations=[{"category": "fait", "content": "Existing fact", "tags": []}])
    _, sections = read_entity(filepath)
    assert len(sections["Faits"]) == 1  # Not duplicated
