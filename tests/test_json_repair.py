"""Tests for JSON repair context manager in llm.py."""

import json
import pytest


def test_valid_json_passes_through():
    """Valid JSON should work normally inside repair context."""
    from src.core.llm import _repaired_json

    with _repaired_json():
        result = json.loads('{"key": "value", "num": 42}')
    assert result == {"key": "value", "num": 42}


def test_unquoted_keys_repaired():
    """Unquoted keys (common small-model error) should be repaired."""
    from src.core.llm import _repaired_json

    malformed = '{name: "Alice", type: "person"}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(malformed)

    with _repaired_json():
        result = json.loads(malformed)
    assert result == {"name": "Alice", "type": "person"}


def test_trailing_comma_repaired():
    """Trailing commas should be repaired."""
    from src.core.llm import _repaired_json

    malformed = '{"entities": [{"name": "X",}]}'
    with _repaired_json():
        result = json.loads(malformed)
    assert result["entities"][0]["name"] == "X"


def test_single_quotes_repaired():
    """Single quotes instead of double quotes should be repaired."""
    from src.core.llm import _repaired_json

    malformed = "{'name': 'Alice', 'type': 'person'}"
    with _repaired_json():
        result = json.loads(malformed)
    assert result == {"name": "Alice", "type": "person"}


def test_json_loads_restored_after_context():
    """json.loads should be restored to original after context exits."""
    from src.core.llm import _repaired_json

    original = json.loads
    with _repaired_json():
        pass
    assert json.loads is original


def test_json_loads_restored_on_exception():
    """json.loads should be restored even if an exception occurs inside context."""
    from src.core.llm import _repaired_json

    original = json.loads
    try:
        with _repaired_json():
            raise ValueError("test error")
    except ValueError:
        pass
    assert json.loads is original


def test_repair_does_not_break_valid_nested_json():
    """Complex nested JSON should pass through without corruption."""
    from src.core.llm import _repaired_json

    nested = '{"entities": [{"name": "X", "obs": [{"cat": "fact"}]}], "relations": []}'
    with _repaired_json():
        result = json.loads(nested)
    assert result["entities"][0]["obs"][0]["cat"] == "fact"
