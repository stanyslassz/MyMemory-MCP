"""Phase 1 integration: verify new models, enums, and config work end-to-end."""

import json
from src.core.models import (
    GraphData, GraphEntity, GraphRelation, EntityFrontmatter,
    RawExtraction, RawEntity, RawObservation, RawRelation,
)


def test_graph_data_roundtrip_with_new_fields():
    gd = GraphData(
        generated="2026-03-05",
        entities={
            "back-pain": GraphEntity(
                file="self/back-pain.md", type="health", title="Back Pain",
                score=0.75, importance=0.8, frequency=5,
                last_mentioned="2026-03-05", created="2026-01-15",
                mention_dates=["2026-03-01", "2026-03-03", "2026-03-05"],
                monthly_buckets={"2026-01": 3, "2026-02": 2},
                summary="Chronic back pain, improving with physiotherapy.",
                retention="long_term", aliases=["mal de dos"], tags=["health"],
            ),
        },
        relations=[
            GraphRelation(
                from_entity="back-pain", to_entity="physio", type="improves",
                strength=0.7, created="2026-01-15", last_reinforced="2026-03-05",
                mention_count=3, context="Physiotherapy helps with back pain",
            ),
        ],
    )
    data = gd.model_dump(by_alias=True)
    json_str = json.dumps(data, ensure_ascii=False)
    restored = GraphData.model_validate(json.loads(json_str))
    entity = restored.entities["back-pain"]
    assert entity.mention_dates == ["2026-03-01", "2026-03-03", "2026-03-05"]
    assert entity.monthly_buckets == {"2026-01": 3, "2026-02": 2}
    assert entity.summary == "Chronic back pain, improving with physiotherapy."
    rel = restored.relations[0]
    assert rel.strength == 0.7
    assert rel.mention_count == 3


def test_backward_compat_old_graph_json():
    old_data = {
        "generated": "2026-03-03",
        "entities": {
            "test": {
                "file": "self/test.md", "type": "health", "title": "Test",
                "score": 0.5, "importance": 0.5, "frequency": 3,
                "last_mentioned": "2026-03-03", "retention": "short_term",
                "aliases": [], "tags": [],
            }
        },
        "relations": [{"from": "a", "to": "b", "type": "affects"}],
    }
    gd = GraphData.model_validate(old_data)
    entity = gd.entities["test"]
    assert entity.mention_dates == []
    assert entity.monthly_buckets == {}
    assert entity.summary == ""
    rel = gd.relations[0]
    assert rel.strength == 0.5
    assert rel.mention_count == 1


def test_extraction_with_english_enums():
    extraction = RawExtraction(
        entities=[
            RawEntity(
                name="Back Pain", type="health",
                observations=[
                    RawObservation(category="diagnosis", content="Sciatica", importance=0.8),
                    RawObservation(category="vigilance", content="Avoid lifting", importance=0.7),
                ],
            ),
        ],
        relations=[
            RawRelation(from_name="Back Pain", to_name="Swimming", type="improves", context="helps"),
        ],
        summary="Discussion about back pain.",
    )
    assert extraction.entities[0].type == "health"
    assert extraction.relations[0].type == "improves"
