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
        file="moi/mal-de-dos.md", type="sante", title="Mal de dos",
        aliases=["dos", "sciatique", "hernie"],
    )
    graph.entities["natation"] = GraphEntity(
        file="interets/natation.md", type="interet", title="Natation",
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
            RawEntity(name="sciatique", type="sante", observations=[
                RawObservation(category="fait", content="Test", importance=0.5),
            ]),
            RawEntity(name="Yoga", type="interet", observations=[]),
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
