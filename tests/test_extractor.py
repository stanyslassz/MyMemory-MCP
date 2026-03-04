"""Tests for pipeline/extractor.py — LLM calls are mocked."""

from unittest.mock import patch

from src.core.models import RawEntity, RawExtraction, RawObservation, RawRelation
from src.pipeline.extractor import extract_from_chat


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
                type="sante",
                observations=[
                    RawObservation(category="diagnostic", content="Sciatique", importance=0.8)
                ],
            ),
            RawEntity(
                name="Dr Martin",
                type="personne",
                observations=[
                    RawObservation(category="fait", content="Médecin traitant", importance=0.5)
                ],
            ),
        ],
        relations=[
            RawRelation(from_name="Dr Martin", to_name="Mal de dos", type="ameliore", context="traitement"),
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
