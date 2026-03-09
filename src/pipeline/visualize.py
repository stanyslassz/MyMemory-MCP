"""Lightweight graph visualization — generates standalone HTML with vis-network.js."""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

from src.core.models import GraphData

# Color mapping for entity types
_TYPE_COLORS = {
    "person": "#4FC3F7",
    "health": "#EF5350",
    "work": "#66BB6A",
    "project": "#FFA726",
    "interest": "#AB47BC",
    "place": "#26A69A",
    "animal": "#8D6E63",
    "organization": "#5C6BC0",
    "ai_self": "#EC407A",
}

_DEFAULT_COLOR = "#90A4AE"


def generate_graph_html(graph: GraphData, output_path: Path) -> Path:
    """Generate a standalone HTML file with interactive graph visualization."""
    nodes = []
    for eid, entity in graph.entities.items():
        nodes.append({
            "id": eid,
            "label": entity.title,
            "title": (
                f"<b>{entity.title}</b><br>"
                f"Type: {entity.type}<br>"
                f"Score: {entity.score:.2f}<br>"
                f"Frequency: {entity.frequency}"
            ),
            "color": _TYPE_COLORS.get(entity.type, _DEFAULT_COLOR),
            "value": max(1, entity.frequency),
            "font": {"size": 14},
        })

    edges = []
    for rel in graph.relations:
        edges.append({
            "from": rel.from_entity,
            "to": rel.to_entity,
            "label": rel.type,
            "title": f"{rel.type} (strength: {rel.strength:.2f})",
            "width": max(1, rel.strength * 4),
            "arrows": "to",
            "color": {"opacity": 0.6},
        })

    # Build legend HTML
    legend_items = "".join(
        f'<span style="color:{color};margin-right:12px;">&#9679; {etype}</span>'
        for etype, color in _TYPE_COLORS.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>memory-ai — Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  body {{ margin: 0; font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; }}
  #graph {{ width: 100vw; height: 100vh; }}
  #legend {{ position: fixed; top: 10px; left: 10px; background: rgba(0,0,0,0.7);
             padding: 8px 14px; border-radius: 6px; font-size: 13px; z-index: 10; }}
  #info {{ position: fixed; bottom: 10px; left: 10px; background: rgba(0,0,0,0.7);
           padding: 8px 14px; border-radius: 6px; font-size: 12px; z-index: 10; }}
</style>
</head>
<body>
<div id="legend">{legend_items}</div>
<div id="info">{len(nodes)} entities &middot; {len(edges)} relations</div>
<div id="graph"></div>
<script>
var nodes = new vis.DataSet({json.dumps(nodes)});
var edges = new vis.DataSet({json.dumps(edges)});
var container = document.getElementById("graph");
var data = {{ nodes: nodes, edges: edges }};
var options = {{
  physics: {{
    solver: "forceAtlas2Based",
    forceAtlas2Based: {{ gravitationalConstant: -30, springLength: 120, damping: 0.4 }},
    stabilization: {{ iterations: 150 }}
  }},
  interaction: {{ hover: true, tooltipDelay: 100 }},
  nodes: {{ shape: "dot", borderWidth: 2 }},
  edges: {{ smooth: {{ type: "continuous" }}, font: {{ size: 10, color: "#aaa" }} }}
}};
new vis.Network(container, data, options);
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path


def open_graph(graph: GraphData, memory_path: Path) -> Path:
    """Generate graph HTML and open in default browser."""
    output = memory_path / "_graph.html"
    generate_graph_html(graph, output)
    webbrowser.open(f"file://{output.resolve()}")
    return output
