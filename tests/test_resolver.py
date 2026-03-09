"""Tests for pipeline/resolver.py."""

from src.core.models import (
    GraphData,
    GraphEntity,
    RawEntity,
    RawExtraction,
    RawObservation,
    RawRelation,
)
from src.pipeline.resolver import resolve_all, resolve_entity, slugify


def _make_graph():
    graph = GraphData()
    graph.entities["mal-de-dos"] = GraphEntity(
        file="self/mal-de-dos.md", type="health", title="Mal de dos",
        aliases=["dos", "sciatique", "hernie"],
    )
    graph.entities["natation"] = GraphEntity(
        file="interests/natation.md", type="interest", title="Natation",
        aliases=["nager", "piscine"],
    )
    return graph


def test_slugify():
    assert slugify("Mal de dos") == "mal-de-dos"
    assert slugify("Dr. Martin") == "dr-martin"
    assert slugify("Problème articulaire") == "probleme-articulaire"
    assert slugify("  spaces  ") == "spaces"


def test_resolve_exact_match():
    graph = _make_graph()
    res = resolve_entity("mal de dos", graph)
    assert res.status == "resolved"
    assert res.entity_id == "mal-de-dos"


def test_resolve_by_alias():
    graph = _make_graph()
    res = resolve_entity("sciatique", graph)
    assert res.status == "resolved"
    assert res.entity_id == "mal-de-dos"


def test_resolve_by_alias_containment():
    graph = _make_graph()
    res = resolve_entity("mon dos", graph)
    assert res.status == "resolved"
    assert res.entity_id == "mal-de-dos"


def test_resolve_new_entity():
    graph = _make_graph()
    res = resolve_entity("Imprimante 3D", graph)
    assert res.status == "new"
    assert res.suggested_slug == "imprimante-3d"


def test_resolve_all():
    graph = _make_graph()
    extraction = RawExtraction(
        entities=[
            RawEntity(name="sciatique", type="health", observations=[
                RawObservation(category="fact", content="Test", importance=0.5),
            ]),
            RawEntity(name="Yoga", type="interest", observations=[]),
        ],
        relations=[],
        summary="Test",
    )
    resolved = resolve_all(extraction, graph)
    assert len(resolved.resolved) == 2
    assert resolved.resolved[0].resolution.status == "resolved"
    assert resolved.resolved[0].resolution.entity_id == "mal-de-dos"
    assert resolved.resolved[1].resolution.status == "new"


def test_resolve_by_title():
    graph = _make_graph()
    res = resolve_entity("Natation", graph)
    assert res.status == "resolved"
    assert res.entity_id == "natation"


# ── GAP 5: Context-aware resolution ──────────────────────────


def test_resolve_entity_with_context_passes_enriched_query():
    """FAISS search should receive context-enriched query when observation_context provided."""
    graph = _make_graph()
    calls = []

    def mock_faiss(query, top_k=3, threshold=0.75):
        calls.append(query)
        return []

    resolve_entity("Apple", graph, faiss_search_fn=mock_faiss, observation_context="health fruit eating")
    assert len(calls) == 1
    assert "Apple" in calls[0]
    assert "health" in calls[0]
    assert "fruit" in calls[0]


def test_resolve_all_passes_observation_context():
    """resolve_all should build context from first observation and pass it to FAISS."""
    graph = _make_graph()
    calls = []

    def mock_faiss(query, top_k=3, threshold=0.75):
        calls.append(query)
        return []

    extraction = RawExtraction(
        entities=[
            RawEntity(name="Apple", type="interest", observations=[
                RawObservation(category="fact", content="Bought stock in tech company", importance=0.5),
            ]),
        ],
        relations=[],
        summary="Test",
    )
    resolve_all(extraction, graph, faiss_search_fn=mock_faiss)
    assert len(calls) == 1
    # Should contain entity name + category + content prefix
    assert "Apple" in calls[0]
    assert "fact" in calls[0]
    assert "Bought stock" in calls[0]
