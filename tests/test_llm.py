"""Tests for core/llm.py — all LLM calls are mocked."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import load_config
from src.core.llm import load_prompt, strip_thinking


@pytest.fixture
def config():
    return load_config(project_root=Path(__file__).parent.parent)


def test_strip_thinking():
    text = '<think>some reasoning here</think>{"key": "value"}'
    assert strip_thinking(text) == '{"key": "value"}'


def test_strip_thinking_multiline():
    text = '<think>\nlong\nreasoning\n</think>\n{"result": true}'
    result = strip_thinking(text)
    assert '"result"' in result
    assert "<think>" not in result


def test_strip_thinking_no_tags():
    text = '{"key": "value"}'
    assert strip_thinking(text) == text


def test_load_prompt(config):
    prompt = load_prompt(
        "extract_facts",
        config,
        chat_content="Hello world",
        json_schema='{"type": "object"}',
    )
    assert "Hello world" in prompt
    assert "fr" in prompt  # user_language injected
    assert "{chat_content}" not in prompt  # variable replaced
    assert "{user_language}" not in prompt


def test_load_prompt_missing_file(config):
    with pytest.raises(FileNotFoundError):
        load_prompt("nonexistent_prompt", config)


def test_load_prompt_arbitrate(config):
    prompt = load_prompt(
        "arbitrate_entity",
        config,
        entity_name="Test Entity",
        entity_context="mentioned in a chat",
        candidates="- entity1\n- entity2",
        json_schema='{"type": "object"}',
    )
    assert "Test Entity" in prompt
    assert "mentioned in a chat" in prompt
