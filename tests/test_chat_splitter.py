"""Tests for JSON export splitter (Claude, ChatGPT, generic)."""

import json
from pathlib import Path

from src.pipeline.chat_splitter import split_export_json


def test_split_claude_export(tmp_path):
    """Claude export with multiple conversations should split into individual chats."""
    memory_path = tmp_path / "memory"
    (memory_path / "chats").mkdir(parents=True)

    export = [
        {
            "uuid": "conv-1",
            "name": "First chat",
            "created_at": "2025-06-01T10:00:00Z",
            "chat_messages": [
                {"sender": "human", "text": "Hello"},
                {"sender": "assistant", "text": "Hi there!"},
            ],
        },
        {
            "uuid": "conv-2",
            "name": "Second chat",
            "created_at": "2025-06-02T14:00:00Z",
            "chat_messages": [
                {"sender": "human", "text": "What is Python?"},
                {"sender": "assistant", "text": "A programming language."},
            ],
        },
    ]
    json_path = tmp_path / "claude_export.json"
    json_path.write_text(json.dumps(export))

    saved = split_export_json(json_path, memory_path)
    assert len(saved) == 2
    assert all(p.exists() for p in saved)

    # Check content of first chat
    content = saved[0].read_text()
    assert "Hello" in content
    assert "Hi there!" in content


def test_split_claude_export_nested(tmp_path):
    """Claude export with conversations key should also work."""
    memory_path = tmp_path / "memory"
    (memory_path / "chats").mkdir(parents=True)

    export = {
        "conversations": [
            {
                "uuid": "conv-1",
                "name": "Test",
                "chat_messages": [
                    {"sender": "human", "text": "Bonjour"},
                    {"sender": "assistant", "text": "Salut!"},
                ],
            },
        ],
    }
    json_path = tmp_path / "export.json"
    json_path.write_text(json.dumps(export))

    saved = split_export_json(json_path, memory_path)
    assert len(saved) == 1


def test_split_chatgpt_export(tmp_path):
    """ChatGPT export with mapping structure should split correctly."""
    memory_path = tmp_path / "memory"
    (memory_path / "chats").mkdir(parents=True)

    export = [
        {
            "title": "GPT Chat",
            "create_time": 1717200000.0,
            "mapping": {
                "node-1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Hello GPT"]},
                        "create_time": 1717200001.0,
                    },
                },
                "node-2": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Hello! How can I help?"]},
                        "create_time": 1717200002.0,
                    },
                },
                "node-3": {
                    "message": {
                        "author": {"role": "system"},
                        "content": {"parts": ["You are a helpful assistant"]},
                        "create_time": 1717200000.0,
                    },
                },
            },
        },
    ]
    json_path = tmp_path / "chatgpt_export.json"
    json_path.write_text(json.dumps(export))

    saved = split_export_json(json_path, memory_path)
    assert len(saved) == 1

    content = saved[0].read_text()
    assert "Hello GPT" in content
    assert "How can I help?" in content
    # System messages should be excluded
    assert "helpful assistant" not in content


def test_split_generic_json_array(tmp_path):
    """Simple role/content JSON array should be treated as single conversation."""
    memory_path = tmp_path / "memory"
    (memory_path / "chats").mkdir(parents=True)

    export = [
        {"role": "user", "content": "Test message"},
        {"role": "assistant", "content": "Test response"},
    ]
    json_path = tmp_path / "simple.json"
    json_path.write_text(json.dumps(export))

    saved = split_export_json(json_path, memory_path)
    assert len(saved) == 1


def test_split_empty_conversations(tmp_path):
    """Export with empty conversations should return empty list."""
    memory_path = tmp_path / "memory"
    (memory_path / "chats").mkdir(parents=True)

    export = [
        {
            "uuid": "conv-1",
            "name": "Empty",
            "chat_messages": [],
        },
    ]
    json_path = tmp_path / "empty.json"
    json_path.write_text(json.dumps(export))

    saved = split_export_json(json_path, memory_path)
    assert len(saved) == 0


def test_split_unrecognized_format(tmp_path):
    """Unrecognized JSON format should return empty list."""
    memory_path = tmp_path / "memory"
    (memory_path / "chats").mkdir(parents=True)

    export = {"some": "random", "data": 42}
    json_path = tmp_path / "unknown.json"
    json_path.write_text(json.dumps(export))

    saved = split_export_json(json_path, memory_path)
    assert len(saved) == 0


def test_split_large_export_many_conversations(tmp_path):
    """Should handle exports with many conversations."""
    memory_path = tmp_path / "memory"
    (memory_path / "chats").mkdir(parents=True)

    export = []
    for i in range(50):
        export.append({
            "uuid": f"conv-{i}",
            "name": f"Chat {i}",
            "chat_messages": [
                {"sender": "human", "text": f"Question {i}"},
                {"sender": "assistant", "text": f"Answer {i}"},
            ],
        })
    json_path = tmp_path / "big_export.json"
    json_path.write_text(json.dumps(export))

    saved = split_export_json(json_path, memory_path)
    assert len(saved) == 50


def test_frontmatter_patched_with_metadata(tmp_path):
    """Chat files should have source metadata in frontmatter."""
    memory_path = tmp_path / "memory"
    (memory_path / "chats").mkdir(parents=True)

    export = [
        {
            "uuid": "conv-1",
            "name": "Important discussion",
            "created_at": "2025-03-15T10:00:00Z",
            "chat_messages": [
                {"sender": "human", "text": "Hello"},
                {"sender": "assistant", "text": "Hi"},
            ],
        },
    ]
    json_path = tmp_path / "export.json"
    json_path.write_text(json.dumps(export))

    saved = split_export_json(json_path, memory_path)
    content = saved[0].read_text()

    assert "source: import" in content
    assert "source_title: Important discussion" in content
