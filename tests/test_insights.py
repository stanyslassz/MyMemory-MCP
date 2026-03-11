"""Tests for memory/insights.py."""

from src.core.models import GraphData, GraphEntity, GraphRelation
from src.memory.insights import compute_insights


def _make_entity(title="Test", type_="health", score=0.5, negative_valence_ratio=0.0):
    return GraphEntity(
        file=f"self/{title.lower().replace(' ', '-')}.md",
        type=type_,
        title=title,
        score=score,
        negative_valence_ratio=negative_valence_ratio,
    )


def test_empty_graph():
    graph = GraphData()
    result = compute_insights(graph, today_str="2026-03-11")
    assert result["total_entities"] == 0
    assert result["total_relations"] == 0
    assert result["forgetting_curve"] == []
    assert result["emotional_hotspots"] == []
    assert result["weak_relations"] == []
    assert result["network_hubs"] == []
    assert all(v == 0 for v in result["scoring_distribution"].values())


def test_scoring_distribution():
    graph = GraphData()
    graph.entities["a"] = _make_entity("A", score=0.05)
    graph.entities["b"] = _make_entity("B", score=0.2)
    graph.entities["c"] = _make_entity("C", score=0.4)
    graph.entities["d"] = _make_entity("D", score=0.6)
    graph.entities["e"] = _make_entity("E", score=0.8)

    result = compute_insights(graph, today_str="2026-03-11")
    dist = result["scoring_distribution"]
    assert dist["0.0-0.1"] == 1
    assert dist["0.1-0.3"] == 1
    assert dist["0.3-0.5"] == 1
    assert dist["0.5-0.7"] == 1
    assert dist["0.7-1.0"] == 1


def test_forgetting_curve():
    graph = GraphData()
    graph.entities["fading"] = _make_entity("Fading", score=0.08)
    graph.entities["safe"] = _make_entity("Safe", score=0.5)
    graph.entities["dead"] = _make_entity("Dead", score=0.0)

    result = compute_insights(graph, today_str="2026-03-11")
    fc = result["forgetting_curve"]
    assert len(fc) == 1
    assert fc[0]["entity"] == "fading"
    assert fc[0]["score"] == 0.08


def test_emotional_hotspots():
    graph = GraphData()
    graph.entities["trauma"] = _make_entity("Trauma", negative_valence_ratio=0.5)
    graph.entities["hobby"] = _make_entity("Hobby", negative_valence_ratio=0.1)

    result = compute_insights(graph, today_str="2026-03-11")
    hotspots = result["emotional_hotspots"]
    assert len(hotspots) == 1
    assert hotspots[0]["entity"] == "trauma"
    assert hotspots[0]["ratio"] == 0.5


def test_weak_relations():
    graph = GraphData()
    graph.entities["a"] = _make_entity("A")
    graph.entities["b"] = _make_entity("B")
    graph.relations.append(
        GraphRelation(
            from_entity="a", to_entity="b", type="affects",
            strength=0.15, last_reinforced="2026-01-01",
        )
    )

    result = compute_insights(graph, today_str="2026-03-11")
    weak = result["weak_relations"]
    assert len(weak) == 1
    assert weak[0]["strength"] == 0.15
    assert weak[0]["days_since_reinforced"] == 69  # Jan 1 to Mar 11


def test_weak_relations_excluded_strong():
    graph = GraphData()
    graph.entities["a"] = _make_entity("A")
    graph.entities["b"] = _make_entity("B")
    graph.relations.append(
        GraphRelation(from_entity="a", to_entity="b", type="affects", strength=0.5)
    )

    result = compute_insights(graph, today_str="2026-03-11")
    assert result["weak_relations"] == []


def test_network_hubs():
    graph = GraphData()
    graph.entities["hub"] = _make_entity("Hub")
    graph.entities["a"] = _make_entity("A")
    graph.entities["b"] = _make_entity("B")
    graph.entities["c"] = _make_entity("C")
    graph.relations.append(GraphRelation(from_entity="hub", to_entity="a", type="affects"))
    graph.relations.append(GraphRelation(from_entity="hub", to_entity="b", type="affects"))
    graph.relations.append(GraphRelation(from_entity="hub", to_entity="c", type="affects"))

    result = compute_insights(graph, today_str="2026-03-11")
    hubs = result["network_hubs"]
    assert hubs[0]["entity"] == "hub"
    assert hubs[0]["degree"] == 3


def test_network_hubs_top_10_limit():
    graph = GraphData()
    for i in range(15):
        eid = f"e{i}"
        graph.entities[eid] = _make_entity(f"E{i}")
    # Connect all to each other (fully connected)
    for i in range(15):
        for j in range(i + 1, 15):
            graph.relations.append(
                GraphRelation(from_entity=f"e{i}", to_entity=f"e{j}", type="linked_to")
            )

    result = compute_insights(graph, today_str="2026-03-11")
    assert len(result["network_hubs"]) == 10


def test_insights_default_today():
    """compute_insights works without explicit today_str."""
    graph = GraphData()
    graph.entities["x"] = _make_entity("X", score=0.5)
    result = compute_insights(graph)
    assert result["total_entities"] == 1
