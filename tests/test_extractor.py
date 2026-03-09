"""Tests for pipeline/extractor.py — LLM calls are mocked."""

from unittest.mock import patch

from src.core.models import RawEntity, RawExtraction, RawObservation, RawRelation
from src.pipeline.extractor import extract_from_chat, _merge_extractions


def test_extract_empty_chat():
    """Empty chat returns empty extraction."""
    from pathlib import Path
    from src.core.config import load_config
    config = load_config(project_root=Path(__file__).parent.parent)
    result = extract_from_chat("", config)
    assert len(result.entities) == 0
    assert len(result.relations) == 0


def test_extract_with_mock():
    """Mocked LLM returns structured extraction."""
    mock_result = RawExtraction(
        entities=[
            RawEntity(
                name="Mal de dos",
                type="health",
                observations=[
                    RawObservation(category="diagnosis", content="Sciatique", importance=0.8)
                ],
            ),
            RawEntity(
                name="Dr Martin",
                type="person",
                observations=[
                    RawObservation(category="fact", content="Médecin traitant", importance=0.5)
                ],
            ),
        ],
        relations=[
            RawRelation(from_name="Dr Martin", to_name="Mal de dos", type="improves", context="traitement"),
        ],
        summary="Patient souffre de sciatique, suivi par Dr Martin",
    )

    with patch("src.pipeline.extractor.call_extraction", return_value=mock_result):
        from pathlib import Path
        from src.core.config import load_config
        config = load_config(project_root=Path(__file__).parent.parent)
        result = extract_from_chat("Some chat content", config)

    assert len(result.entities) == 2
    assert result.entities[0].name == "Mal de dos"
    assert len(result.relations) == 1
    assert result.summary != ""


def test_merge_extractions():
    """_merge_extractions deduplicates entities and relations."""
    ext1 = RawExtraction(
        entities=[
            RawEntity(name="Sophie", type="person", observations=[
                RawObservation(category="fact", content="Nurse", importance=0.5),
            ]),
        ],
        relations=[
            RawRelation(from_name="Alexis", to_name="Sophie", type="lives_with", context=""),
        ],
        summary="Part 1",
    )
    ext2 = RawExtraction(
        entities=[
            RawEntity(name="Sophie", type="person", observations=[
                RawObservation(category="fact", content="Nurse", importance=0.5),  # duplicate
                RawObservation(category="fact", content="Works at Purpan", importance=0.6),  # new
            ]),
        ],
        relations=[
            RawRelation(from_name="Alexis", to_name="Sophie", type="lives_with", context=""),  # duplicate
            RawRelation(from_name="Sophie", to_name="Purpan", type="works_at", context=""),  # new
        ],
        summary="Part 2",
    )

    merged = _merge_extractions([ext1, ext2])

    assert len(merged.entities) == 1
    assert merged.entities[0].name == "Sophie"
    assert len(merged.entities[0].observations) == 2

    assert len(merged.relations) == 2

    assert "Part 1" in merged.summary
    assert "Part 2" in merged.summary


from src.pipeline.extractor import sanitize_extraction


def test_sanitize_fixes_invalid_relation_type():
    """Invalid relation type mapped to linked_to."""
    raw_dict = {
        "entities": [
            {"name": "Dr Martin", "type": "person", "observations": []},
            {"name": "Mal de dos", "type": "health", "observations": []},
        ],
        "relations": [
            {"from_name": "Dr Martin", "to_name": "Mal de dos", "type": "prescrit_par", "context": ""},
        ],
        "summary": None,
    }
    result = sanitize_extraction(raw_dict)
    assert result.relations[0].type == "linked_to"
    assert result.summary == ""


def test_sanitize_fixes_known_french_relation():
    """Known French relation type mapped to correct English type."""
    raw_dict = {
        "entities": [
            {"name": "Alice", "type": "person", "observations": []},
            {"name": "Airbus", "type": "organization", "observations": []},
        ],
        "relations": [
            {"from_name": "Alice", "to_name": "Airbus", "type": "travaille_a", "context": ""},
        ],
        "summary": "test",
    }
    result = sanitize_extraction(raw_dict)
    assert result.relations[0].type == "works_at"


def test_sanitize_drops_empty_relation_refs():
    """Relations with empty from_name or to_name are dropped."""
    raw_dict = {
        "entities": [],
        "relations": [
            {"from_name": "", "to_name": "Mal de dos", "type": "affects", "context": ""},
            {"from_name": "Dr Martin", "to_name": "", "type": "improves", "context": ""},
            {"from_name": "Dr Martin", "to_name": "Mal de dos", "type": "improves", "context": "valid"},
        ],
        "summary": "test",
    }
    result = sanitize_extraction(raw_dict)
    assert len(result.relations) == 1
    assert result.relations[0].context == "valid"


def test_sanitize_fixes_invalid_entity_type():
    """Invalid entity type falls back to interest."""
    raw_dict = {
        "entities": [
            {"name": "Yoga", "type": "activite", "observations": []},
        ],
        "relations": [],
        "summary": "",
    }
    result = sanitize_extraction(raw_dict)
    assert result.entities[0].type == "interest"


def test_sanitize_fixes_invalid_observation_category():
    """Invalid observation category falls back to fact."""
    raw_dict = {
        "entities": [
            {"name": "Yoga", "type": "interest", "observations": [
                {"category": "habitude", "content": "Pratique le matin", "importance": 0.5},
            ]},
        ],
        "relations": [],
        "summary": "",
    }
    result = sanitize_extraction(raw_dict)
    assert result.entities[0].observations[0].category == "fact"


def test_sanitize_drops_empty_entities_and_observations():
    """Entities with empty name and observations with empty content are dropped."""
    raw_dict = {
        "entities": [
            {"name": "", "type": "person", "observations": []},
            {"name": "Yoga", "type": "interest", "observations": [
                {"category": "fact", "content": "", "importance": 0.5},
                {"category": "fact", "content": "Morning practice", "importance": 0.5},
            ]},
        ],
        "relations": [],
        "summary": "",
    }
    result = sanitize_extraction(raw_dict)
    assert len(result.entities) == 1
    assert result.entities[0].name == "Yoga"
    assert len(result.entities[0].observations) == 1


def test_sanitize_clamps_importance():
    """Importance outside [0, 1] is clamped."""
    raw_dict = {
        "entities": [
            {"name": "Test", "type": "interest", "observations": [
                {"category": "fact", "content": "High", "importance": 1.5},
                {"category": "fact", "content": "Low", "importance": -0.3},
            ]},
        ],
        "relations": [],
        "summary": "",
    }
    result = sanitize_extraction(raw_dict)
    assert result.entities[0].observations[0].importance == 1.0
    assert result.entities[0].observations[1].importance == 0.0


def test_sanitize_passthrough_valid_extraction():
    """Valid extraction passes through unchanged."""
    raw = RawExtraction(
        entities=[
            RawEntity(name="Sophie", type="person", observations=[
                RawObservation(category="fact", content="Infirmière", importance=0.5),
            ]),
        ],
        relations=[
            RawRelation(from_name="Alexis", to_name="Sophie", type="lives_with", context="couple"),
        ],
        summary="Discussion about Sophie",
    )
    result = sanitize_extraction(raw)
    assert len(result.entities) == 1
    assert result.entities[0].name == "Sophie"
    assert len(result.relations) == 1
    assert result.relations[0].type == "lives_with"
    assert result.summary == "Discussion about Sophie"
