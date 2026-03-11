"""Tests for core/action_log.py."""

import json

from src.core.action_log import log_action, read_actions


def test_log_action_creates_file(tmp_path):
    log_action(tmp_path, "entity_created", entity_id="test-entity")
    log_path = tmp_path / "_actions.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["action"] == "entity_created"
    assert entry["entity_id"] == "test-entity"
    assert entry["source"] == "pipeline"
    assert "timestamp" in entry


def test_log_action_appends(tmp_path):
    log_action(tmp_path, "entity_created", entity_id="a")
    log_action(tmp_path, "entity_updated", entity_id="b")
    log_path = tmp_path / "_actions.jsonl"
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_log_action_with_details(tmp_path):
    log_action(tmp_path, "merge", entity_id="x", details={"merged_from": "y"})
    entries = read_actions(tmp_path)
    assert entries[0]["details"] == {"merged_from": "y"}


def test_log_action_custom_source(tmp_path):
    log_action(tmp_path, "prune", entity_id="z", source="dream")
    entries = read_actions(tmp_path)
    assert entries[0]["source"] == "dream"


def test_read_actions_empty(tmp_path):
    entries = read_actions(tmp_path)
    assert entries == []


def test_read_actions_filter_by_entity(tmp_path):
    log_action(tmp_path, "created", entity_id="alpha")
    log_action(tmp_path, "created", entity_id="beta")
    log_action(tmp_path, "updated", entity_id="alpha")

    entries = read_actions(tmp_path, entity_id="alpha")
    assert len(entries) == 2
    assert all(e["entity_id"] == "alpha" for e in entries)


def test_read_actions_filter_by_action(tmp_path):
    log_action(tmp_path, "created", entity_id="a")
    log_action(tmp_path, "updated", entity_id="a")
    log_action(tmp_path, "created", entity_id="b")

    entries = read_actions(tmp_path, action="created")
    assert len(entries) == 2
    assert all(e["action"] == "created" for e in entries)


def test_read_actions_last_n(tmp_path):
    for i in range(10):
        log_action(tmp_path, "tick", entity_id=f"e{i}")

    entries = read_actions(tmp_path, last_n=3)
    assert len(entries) == 3
    assert entries[0]["entity_id"] == "e7"
    assert entries[2]["entity_id"] == "e9"


def test_read_actions_combined_filters(tmp_path):
    log_action(tmp_path, "created", entity_id="x")
    log_action(tmp_path, "updated", entity_id="x")
    log_action(tmp_path, "created", entity_id="y")

    entries = read_actions(tmp_path, entity_id="x", action="created")
    assert len(entries) == 1
    assert entries[0]["entity_id"] == "x"
    assert entries[0]["action"] == "created"


def test_read_actions_skips_corrupt_lines(tmp_path):
    log_path = tmp_path / "_actions.jsonl"
    log_path.write_text(
        '{"action": "ok", "entity_id": "a", "timestamp": "t", "source": "p", "details": {}}\n'
        "not valid json\n"
        '{"action": "ok2", "entity_id": "b", "timestamp": "t", "source": "p", "details": {}}\n',
        encoding="utf-8",
    )
    entries = read_actions(tmp_path)
    assert len(entries) == 2
