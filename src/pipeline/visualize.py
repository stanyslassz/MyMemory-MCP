"""Enhanced graph visualization — standalone HTML with vis-network.js.

Features: search box, type filter, click-to-expand sidebar, relation strength
legend, Barnes-Hut solver for large graphs, double-click focus.
"""

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


def generate_graph_html(graph: GraphData, output_path: Path, memory_path: Path | None = None) -> Path:
    """Generate a standalone HTML file with interactive graph visualization.

    If memory_path is provided, reads top-5 facts per entity for the sidebar.
    """
    read_entity_fn = None
    if memory_path:
        from src.memory.store import read_entity
        read_entity_fn = read_entity

    nodes = []
    for eid, entity in graph.entities.items():
        # Read facts if memory_path available
        facts: list[str] = []
        if read_entity_fn:
            entity_path = memory_path / entity.file
            if entity_path.exists():
                try:
                    _, sections = read_entity_fn(entity_path)
                    raw_facts = [f for f in sections.get("Facts", []) if "[superseded]" not in f]
                    facts = raw_facts[:5]
                except Exception:
                    pass

        # Collect relations for this entity
        related: list[str] = []
        for rel in graph.relations:
            if rel.from_entity == eid:
                target = graph.entities.get(rel.to_entity)
                if target:
                    related.append(f"\u2192 {rel.type} {target.title}")
            elif rel.to_entity == eid:
                source = graph.entities.get(rel.from_entity)
                if source:
                    related.append(f"\u2190 {rel.type} {source.title}")

        nodes.append({
            "id": eid,
            "label": entity.title,
            "color": _TYPE_COLORS.get(entity.type, _DEFAULT_COLOR),
            "value": max(1, entity.frequency),
            "font": {"size": 14},
            "data": {
                "type": entity.type,
                "score": round(entity.score, 3),
                "frequency": entity.frequency,
                "tags": list(entity.tags or []),
                "summary": entity.summary or "",
                "facts": facts,
                "relations": related,
            },
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
            "relType": rel.type,
        })

    # Choose solver based on graph size
    n_nodes = len(nodes)
    if n_nodes > 50:
        solver_config = """{
      solver: "barnesHut",
      barnesHut: { gravitationalConstant: -3000, springLength: 150, damping: 0.3 },
      stabilization: { iterations: 200 }
    }"""
    else:
        solver_config = """{
      solver: "forceAtlas2Based",
      forceAtlas2Based: { gravitationalConstant: -30, springLength: 120, damping: 0.4 },
      stabilization: { iterations: 150 }
    }"""

    # Build type colors JSON for JS
    type_colors_json = json.dumps(_TYPE_COLORS)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>memory-ai \u2014 Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #1a1a2e; color: #eee; overflow: hidden; }}
  #graph {{ width: 100vw; height: 100vh; }}

  /* Search box */
  #search-box {{ position: fixed; top: 10px; right: 10px; z-index: 20; }}
  #search-box input {{
    background: rgba(0,0,0,0.8); border: 1px solid #555; color: #eee; padding: 8px 12px;
    border-radius: 6px; font-size: 14px; width: 220px; outline: none;
  }}
  #search-box input:focus {{ border-color: #4FC3F7; }}
  #search-box input::placeholder {{ color: #777; }}

  /* Legend / type filter */
  #legend {{
    position: fixed; top: 10px; left: 10px; background: rgba(0,0,0,0.8);
    padding: 10px 14px; border-radius: 6px; font-size: 13px; z-index: 20;
  }}
  .legend-item {{
    display: inline-block; margin-right: 10px; cursor: pointer; user-select: none;
    transition: opacity 0.2s;
  }}
  .legend-item.inactive {{ opacity: 0.3; text-decoration: line-through; }}
  .legend-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }}

  /* Info bar */
  #info {{
    position: fixed; bottom: 10px; left: 10px; background: rgba(0,0,0,0.8);
    padding: 8px 14px; border-radius: 6px; font-size: 12px; z-index: 20;
  }}

  /* Strength legend */
  #strength-legend {{
    position: fixed; bottom: 10px; right: 10px; background: rgba(0,0,0,0.8);
    padding: 8px 14px; border-radius: 6px; font-size: 11px; z-index: 20;
  }}
  .str-line {{ display: inline-block; height: 2px; vertical-align: middle; background: #aaa; margin-right: 4px; }}

  /* Sidebar */
  #sidebar {{
    position: fixed; top: 0; right: -340px; width: 340px; height: 100vh;
    background: rgba(15, 15, 35, 0.95); border-left: 1px solid #333;
    padding: 20px; overflow-y: auto; z-index: 30; transition: right 0.25s ease;
  }}
  #sidebar.open {{ right: 0; }}
  #sidebar-close {{
    position: absolute; top: 10px; right: 14px; cursor: pointer; font-size: 20px;
    color: #999; background: none; border: none;
  }}
  #sidebar-close:hover {{ color: #fff; }}
  #sidebar h2 {{ margin: 0 0 4px 0; font-size: 18px; padding-right: 30px; }}
  #sidebar .meta {{ color: #999; font-size: 12px; margin-bottom: 12px; }}
  #sidebar .section-title {{ color: #4FC3F7; font-size: 13px; font-weight: 600; margin: 14px 0 6px 0; }}
  #sidebar .tag {{ display: inline-block; background: #333; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin: 2px 3px 2px 0; }}
  #sidebar .fact {{ font-size: 12px; color: #ccc; margin: 4px 0; line-height: 1.4; }}
  #sidebar .relation {{ font-size: 12px; color: #aaa; margin: 3px 0; }}
  #sidebar .summary {{ font-size: 12px; color: #bbb; font-style: italic; margin: 8px 0; line-height: 1.4; }}
</style>
</head>
<body>

<div id="search-box"><input type="text" id="search" placeholder="Search entities... (ESC to clear)"></div>

<div id="legend"></div>

<div id="info">{n_nodes} entities &middot; {len(edges)} relations</div>

<div id="strength-legend">
  <span class="str-line" style="width:15px;height:1px;"></span> &lt;0.3
  &nbsp;
  <span class="str-line" style="width:15px;height:2px;"></span> 0.3\u20130.6
  &nbsp;
  <span class="str-line" style="width:15px;height:4px;"></span> &gt;0.6
</div>

<div id="sidebar">
  <button id="sidebar-close">&times;</button>
  <div id="sidebar-content"></div>
</div>

<div id="graph"></div>

<script>
var TYPE_COLORS = {type_colors_json};
var DEFAULT_COLOR = "{_DEFAULT_COLOR}";
var allNodes = {json.dumps(nodes)};
var allEdges = {json.dumps(edges)};

var nodes = new vis.DataSet(allNodes);
var edges = new vis.DataSet(allEdges);
var container = document.getElementById("graph");
var data = {{ nodes: nodes, edges: edges }};
var options = {{
  physics: {solver_config},
  interaction: {{ hover: true, tooltipDelay: 200 }},
  nodes: {{ shape: "dot", borderWidth: 2 }},
  edges: {{ smooth: {{ type: "continuous" }}, font: {{ size: 10, color: "#aaa" }} }}
}};
var network = new vis.Network(container, data, options);

// ── Build legend (clickable type filter) ──
var activeTypes = new Set(Object.keys(TYPE_COLORS).concat(["_other"]));
var legendEl = document.getElementById("legend");

function buildLegend() {{
  legendEl.innerHTML = "";
  for (var t in TYPE_COLORS) {{
    var item = document.createElement("span");
    item.className = "legend-item" + (activeTypes.has(t) ? "" : " inactive");
    item.dataset.type = t;
    item.innerHTML = '<span class="legend-dot" style="background:' + TYPE_COLORS[t] + '"></span>' + t;
    item.addEventListener("click", toggleType);
    legendEl.appendChild(item);
  }}
}}
buildLegend();

function toggleType(e) {{
  var t = e.currentTarget.dataset.type;
  if (activeTypes.has(t)) {{ activeTypes.delete(t); }} else {{ activeTypes.add(t); }}
  e.currentTarget.classList.toggle("inactive");
  applyFilters();
}}

// ── Search ──
var searchInput = document.getElementById("search");
searchInput.addEventListener("input", applyFilters);
searchInput.addEventListener("keydown", function(e) {{
  if (e.key === "Escape") {{ searchInput.value = ""; applyFilters(); }}
}});

function applyFilters() {{
  var query = searchInput.value.toLowerCase().trim();
  var updates = [];
  allNodes.forEach(function(n) {{
    var typeMatch = activeTypes.has(n.data.type);
    var searchMatch = !query || n.label.toLowerCase().includes(query);
    var visible = typeMatch;
    var opacity = (visible && searchMatch) ? 1.0 : (visible ? 0.15 : 0);
    updates.push({{
      id: n.id,
      hidden: !visible,
      opacity: opacity,
      font: {{ size: 14, color: searchMatch && visible ? "#eee" : "rgba(238,238,238,0.15)" }}
    }});
  }});
  nodes.update(updates);
  // Update edge visibility
  var edgeUpdates = [];
  allEdges.forEach(function(e) {{
    var fromNode = nodes.get(e.from);
    var toNode = nodes.get(e.to);
    var hidden = (fromNode && fromNode.hidden) || (toNode && toNode.hidden);
    edgeUpdates.push({{ id: e.id, hidden: hidden }});
  }});
  edges.update(edgeUpdates);
}}

// ── Sidebar ──
var sidebar = document.getElementById("sidebar");
var sidebarContent = document.getElementById("sidebar-content");
var sidebarClose = document.getElementById("sidebar-close");

function escapeHtml(s) {{
  var d = document.createElement("div"); d.textContent = s; return d.innerHTML;
}}

network.on("click", function(params) {{
  if (params.nodes.length === 0) {{
    sidebar.classList.remove("open");
    return;
  }}
  var nodeId = params.nodes[0];
  var node = nodes.get(nodeId);
  if (!node || !node.data) return;
  var d = node.data;
  var html = '<h2 style="color:' + (TYPE_COLORS[d.type] || DEFAULT_COLOR) + '">' + escapeHtml(node.label) + '</h2>';
  html += '<div class="meta">' + d.type + ' &middot; score: ' + d.score + ' &middot; freq: ' + d.frequency + '</div>';
  if (d.summary) {{
    html += '<div class="summary">' + escapeHtml(d.summary) + '</div>';
  }}
  if (d.tags && d.tags.length) {{
    html += '<div style="margin:8px 0">';
    d.tags.forEach(function(t) {{ html += '<span class="tag">' + escapeHtml(t) + '</span>'; }});
    html += '</div>';
  }}
  if (d.facts && d.facts.length) {{
    html += '<div class="section-title">Facts</div>';
    d.facts.forEach(function(f) {{ html += '<div class="fact">' + escapeHtml(f) + '</div>'; }});
  }}
  if (d.relations && d.relations.length) {{
    html += '<div class="section-title">Relations</div>';
    d.relations.forEach(function(r) {{ html += '<div class="relation">' + escapeHtml(r) + '</div>'; }});
  }}
  sidebarContent.innerHTML = html;
  sidebar.classList.add("open");
}});

sidebarClose.addEventListener("click", function() {{
  sidebar.classList.remove("open");
}});

// ── Double-click to focus ──
var focusMode = false;
network.on("doubleClick", function(params) {{
  if (params.nodes.length === 0) {{
    // Double-click on empty: reset
    if (focusMode) {{
      nodes.update(allNodes.map(function(n) {{ return {{ id: n.id, hidden: false, opacity: 1.0, font: {{ size: 14, color: "#eee" }} }}; }}));
      edges.update(allEdges.map(function(e) {{ return {{ id: e.id, hidden: false }}; }}));
      focusMode = false;
    }}
    return;
  }}
  var nodeId = params.nodes[0];
  // Find neighbors
  var neighborIds = new Set();
  neighborIds.add(nodeId);
  allEdges.forEach(function(e) {{
    if (e.from === nodeId) neighborIds.add(e.to);
    if (e.to === nodeId) neighborIds.add(e.from);
  }});
  // Show only neighbors
  nodes.update(allNodes.map(function(n) {{
    var isNeighbor = neighborIds.has(n.id);
    return {{
      id: n.id, hidden: !isNeighbor, opacity: 1.0,
      font: {{ size: 14, color: "#eee" }}
    }};
  }}));
  edges.update(allEdges.map(function(e) {{
    var show = neighborIds.has(e.from) && neighborIds.has(e.to);
    return {{ id: e.id, hidden: !show }};
  }}));
  focusMode = true;
}});
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path


def open_graph(graph: GraphData, memory_path: Path) -> Path:
    """Generate graph HTML and open in default browser."""
    output = memory_path / "_graph.html"
    generate_graph_html(graph, output, memory_path=memory_path)
    webbrowser.open(f"file://{output.resolve()}")
    return output
