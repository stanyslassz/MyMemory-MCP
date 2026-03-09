"""Tests for graph visualization."""

from pathlib import Path

from src.core.models import GraphData, GraphEntity, GraphRelation
from src.pipeline.visualize import generate_graph_html


def test_generate_graph_html(tmp_path):
    """Generated HTML should contain vis-network and graph data."""
    graph = GraphData(
        generated="2026-03-09",
        entities={
            "alice": GraphEntity(
                file="close_ones/alice.md", type="person", title="Alice",
                score=0.8, importance=0.7, frequency=5,
                last_mentioned="2026-03-01",
            ),
            "swimming": GraphEntity(
                file="interests/swimming.md", type="interest", title="Swimming",
                score=0.6, importance=0.5, frequency=3,
                last_mentioned="2026-02-15",
            ),
        },
        relations=[
            GraphRelation(
                from_entity="alice", to_entity="swimming",
                type="uses", strength=0.7,
            ),
        ],
    )

    output = tmp_path / "test_graph.html"
    result = generate_graph_html(graph, output)

    assert result.exists()
    html = result.read_text()
    assert "vis-network" in html
    assert "Alice" in html
    assert "Swimming" in html
    assert "uses" in html
    assert "2 entities" in html
    assert "1 relations" in html


def test_empty_graph(tmp_path):
    """Empty graph should produce valid HTML."""
    graph = GraphData(generated="2026-03-09", entities={}, relations=[])
    output = tmp_path / "empty.html"
    generate_graph_html(graph, output)

    html = output.read_text()
    assert "0 entities" in html
    assert "vis-network" in html
