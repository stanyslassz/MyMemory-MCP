"""Tests for core/models.py."""

import json

from pydantic import ValidationError
import pytest

from src.core.models import (
    EntityFrontmatter,
    EntityResolution,
    GraphData,
    GraphEntity,
    GraphRelation,
    RawEntity,
    RawExtraction,
    RawObservation,
    RawRelation,
    Resolution,
)


def test_raw_observation_valid():
    obs = RawObservation(category="fact", content="Test fact", importance=0.5, tags=["test"])
    assert obs.category == "fact"
    assert obs.importance == 0.5


def test_raw_observation_invalid_category():
    with pytest.raises(ValidationError):
        RawObservation(category="invalid_cat", content="Test", importance=0.5)


def test_raw_observation_date_and_valence():
    obs = RawObservation(category="fact", content="Test", importance=0.5,
                         date="2024-03", valence="positive")
    assert obs.date == "2024-03"
    assert obs.valence == "positive"


def test_raw_observation_date_valence_defaults():
    obs = RawObservation(category="fact", content="Test", importance=0.5)
    assert obs.date == ""
    assert obs.valence == ""


def test_raw_observation_invalid_valence():
    with pytest.raises(ValidationError):
        RawObservation(category="fact", content="Test", importance=0.5, valence="happy")


def test_raw_observation_importance_bounds():
    with pytest.raises(ValidationError):
        RawObservation(category="fact", content="Test", importance=1.5)
    with pytest.raises(ValidationError):
        RawObservation(category="fact", content="Test", importance=-0.1)


def test_raw_extraction_roundtrip():
    extraction = RawExtraction(
        entities=[
            RawEntity(
                name="Mal de dos",
                type="health",
                observations=[
                    RawObservation(category="diagnosis", content="Sciatique", importance=0.8)
                ],
            )
        ],
        relations=[
            RawRelation(from_name="Mal de dos", to_name="Natation", type="improves", context="aide")
        ],
        summary="Problème de dos",
    )
    data = extraction.model_dump()
    restored = RawExtraction.model_validate(data)
    assert restored.entities[0].name == "Mal de dos"
    assert restored.relations[0].type == "improves"


def test_entity_resolution():
    res = EntityResolution(action="existing", existing_id="mal-de-dos")
    assert res.action == "existing"
    res2 = EntityResolution(action="new", new_type="health")
    assert res2.new_type == "health"


def test_graph_relation_alias():
    rel = GraphRelation(from_entity="a", to_entity="b", type="affects")
    dumped = rel.model_dump(by_alias=True)
    assert dumped["from"] == "a"
    assert dumped["to"] == "b"

    # Can also construct from alias
    rel2 = GraphRelation(**{"from": "x", "to": "y", "type": "improves"})
    assert rel2.from_entity == "x"


def test_graph_data_serialization():
    gd = GraphData(
        generated="2026-03-03",
        entities={
            "test-entity": GraphEntity(
                file="moi/test.md",
                type="health",
                title="Test",
                score=0.5,
            )
        },
        relations=[GraphRelation(from_entity="a", to_entity="b", type="affects")],
    )
    data = gd.model_dump(by_alias=True)
    assert "test-entity" in data["entities"]
    assert data["relations"][0]["from"] == "a"

    # JSON roundtrip
    json_str = json.dumps(data)
    restored = GraphData.model_validate(json.loads(json_str))
    assert "test-entity" in restored.entities


def test_entity_frontmatter():
    fm = EntityFrontmatter(
        title="Test",
        type="health",
        retention="long_term",
        score=0.88,
        importance=0.85,
        frequency=38,
        last_mentioned="2026-03-03",
        created="2025-09-15",
        aliases=["dos", "sciatique"],
        tags=["santé"],
    )
    assert fm.retention == "long_term"
    assert len(fm.aliases) == 2


def test_resolution():
    r = Resolution(status="resolved", entity_id="test-id")
    assert r.status == "resolved"
    r2 = Resolution(status="ambiguous", candidates=["a", "b"])
    assert len(r2.candidates) == 2
    r3 = Resolution(status="new", suggested_slug="new-entity")
    assert r3.suggested_slug == "new-entity"


def test_graph_entity_new_fields():
    entity = GraphEntity(
        file="self/test.md", type="health", title="Test",
        mention_dates=["2026-03-01", "2026-03-03"],
        monthly_buckets={"2025-01": 5},
        summary="A brief summary.",
        created="2026-01-15",
    )
    assert len(entity.mention_dates) == 2
    assert entity.monthly_buckets["2025-01"] == 5
    assert entity.summary == "A brief summary."
    assert entity.created == "2026-01-15"

def test_graph_entity_defaults_for_new_fields():
    entity = GraphEntity(file="self/test.md", type="health", title="Test")
    assert entity.mention_dates == []
    assert entity.monthly_buckets == {}
    assert entity.created == ""
    assert entity.summary == ""

def test_frontmatter_new_fields():
    fm = EntityFrontmatter(
        title="Test", type="health",
        mention_dates=["2026-03-01"],
        monthly_buckets={"2025-06": 3},
        summary="Summary text.",
    )
    assert len(fm.mention_dates) == 1
    assert fm.summary == "Summary text."

def test_graph_relation_enriched_fields():
    rel = GraphRelation(
        from_entity="a", to_entity="b", type="affects",
        strength=0.7, created="2026-01-01",
        last_reinforced="2026-03-01", mention_count=5,
        context="A affects B because of X",
    )
    assert rel.strength == 0.7
    assert rel.mention_count == 5
    assert rel.context == "A affects B because of X"

def test_graph_relation_defaults():
    rel = GraphRelation(from_entity="a", to_entity="b", type="affects")
    assert rel.strength == 0.5
    assert rel.mention_count == 1
    assert rel.context == ""
    assert rel.created == ""
    assert rel.last_reinforced == ""

def test_graph_relation_serialization_with_new_fields():
    rel = GraphRelation(
        from_entity="a", to_entity="b", type="affects",
        strength=0.8, context="test reason",
    )
    data = rel.model_dump(by_alias=True)
    assert data["from"] == "a"
    assert data["strength"] == 0.8
    assert data["context"] == "test reason"
    restored = GraphRelation.model_validate(data)
    assert restored.strength == 0.8


def test_raw_relation_supersedes_field():
    """RawRelation should have an optional supersedes field."""
    from src.core.models import RawRelation

    rel = RawRelation(from_name="Alice", to_name="Bob", type="parent_of")
    assert rel.supersedes == ""

    rel2 = RawRelation(
        from_name="Alice", to_name="Bob", type="parent_of",
        supersedes="alice:bob:linked_to"
    )
    assert rel2.supersedes == "alice:bob:linked_to"


def test_ai_self_entity_type():
    """ai_self must be a valid entity type."""
    from src.core.models import RawEntity, RawObservation
    entity = RawEntity(
        name="AI Personality",
        type="ai_self",
        observations=[
            RawObservation(category="ai_style", content="Be direct", importance=0.7, tags=["communication"]),
            RawObservation(category="user_reaction", content="Liked tables", importance=0.5, tags=["format"]),
            RawObservation(category="interaction_rule", content="No repeats", importance=0.8, tags=["rules"]),
        ],
    )
    assert entity.type == "ai_self"
    assert entity.observations[0].category == "ai_style"
