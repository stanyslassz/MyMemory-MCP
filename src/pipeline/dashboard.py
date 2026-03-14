"""Interactive dashboard — standalone HTML with vis-network.js.

Generates _dashboard.html with 4 views: Graph, Timeline, Dream replay, Search.
All data embedded as JSON variables. Zero server dependency.
Uses the same vis-network.js + Barnes-Hut physics as visualize.py for the graph tab.
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

from src.core.config import Config
from src.core.models import GraphData
from src.memory.graph import load_graph
from src.memory.store import read_entity

# Color mapping for entity types — matches visualize.py
TYPE_COLORS = {
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


def generate_dashboard(config: Config) -> Path:
    """Generate a standalone HTML dashboard and return its path."""
    memory_path = config.memory_path

    graph = load_graph(memory_path)
    events = _load_events(memory_path)
    entities_detail = _load_entity_details(graph, memory_path)
    dream_sessions = _extract_dream_sessions(events)
    stats = _compute_stats(graph, events)
    vis_data = _graph_to_vis(graph, memory_path)

    n_nodes = len(graph.entities)
    if n_nodes > 50:
        solver_config = '{ solver: "barnesHut", barnesHut: { gravitationalConstant: -3000, springLength: 150, damping: 0.3 }, stabilization: { iterations: 200 } }'
    else:
        solver_config = '{ solver: "forceAtlas2Based", forceAtlas2Based: { gravitationalConstant: -30, springLength: 120, damping: 0.4 }, stabilization: { iterations: 150 } }'

    html = (
        DASHBOARD_TEMPLATE
        .replace("/*__NODES__*/", json.dumps(vis_data["nodes"], ensure_ascii=False))
        .replace("/*__EDGES__*/", json.dumps(vis_data["edges"], ensure_ascii=False))
        .replace("/*__EVENTS__*/", json.dumps(events, ensure_ascii=False))
        .replace("/*__ENTITIES__*/", json.dumps(entities_detail, ensure_ascii=False))
        .replace("/*__DREAM_SESSIONS__*/", json.dumps(dream_sessions, ensure_ascii=False))
        .replace("/*__STATS__*/", json.dumps(stats, ensure_ascii=False))
        .replace("/*__TYPE_COLORS__*/", json.dumps(TYPE_COLORS))
        .replace("/*__SOLVER__*/", solver_config)
        .replace("/*__RELATION_TYPES__*/", json.dumps([
            "affects", "improves", "worsens", "requires", "linked_to",
            "lives_with", "works_at", "parent_of", "friend_of", "uses",
            "part_of", "contrasts_with", "precedes",
        ]))
    )

    output = memory_path / "_dashboard.html"
    output.write_text(html, encoding="utf-8")
    return output


def open_dashboard(config: Config) -> Path:
    """Generate dashboard HTML and open in default browser."""
    output = generate_dashboard(config)
    webbrowser.open(f"file://{output.resolve()}")
    return output


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_events(memory_path: Path) -> list[dict]:
    """Load last 500 events from _event_log.jsonl."""
    log_path = memory_path / "_event_log.jsonl"
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    events: list[dict] = []
    for line in lines[-500:]:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _load_entity_details(graph: GraphData, memory_path: Path) -> dict:
    """Load facts from each entity MD file."""
    details: dict[str, dict] = {}
    for eid, entity in graph.entities.items():
        entity_path = memory_path / entity.file
        if not entity_path.exists():
            continue
        try:
            _fm, sections = read_entity(entity_path)
            raw_facts = [f for f in sections.get("Facts", []) if "[superseded]" not in f]
            details[eid] = {
                "title": entity.title,
                "type": entity.type,
                "score": round(entity.score, 3),
                "frequency": entity.frequency,
                "retention": entity.retention,
                "created": entity.created,
                "last_mentioned": entity.last_mentioned,
                "importance": round(entity.importance, 3),
                "mention_dates": entity.mention_dates[-10:],
                "facts": raw_facts[:20],
                "tags": list(entity.tags or []),
                "aliases": list(entity.aliases or []),
                "summary": entity.summary or "",
                "file": entity.file,
            }
        except Exception:
            continue
    return details


def _graph_to_vis(graph: GraphData, memory_path: Path) -> dict:
    """Convert graph to vis-network format (same as visualize.py)."""
    nodes = []
    for eid, entity in graph.entities.items():
        nodes.append({
            "id": eid,
            "label": entity.title,
            "color": TYPE_COLORS.get(entity.type, _DEFAULT_COLOR),
            "value": max(1, entity.frequency),
            "font": {"size": 14},
            "data": {
                "type": entity.type,
                "score": round(entity.score, 3),
                "frequency": entity.frequency,
                "retention": entity.retention,
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
            "data": {
                "type": rel.type,
                "strength": round(rel.strength, 3),
                "last_reinforced": rel.last_reinforced,
                "mention_count": rel.mention_count,
            },
        })

    return {"nodes": nodes, "edges": edges}


def _extract_dream_sessions(events: list[dict]) -> list[dict]:
    """Group dream_step_* events by day into sessions."""
    sessions: dict[str, dict] = {}
    for evt in events:
        evt_type = evt.get("type", "")
        if not evt_type.startswith("dream_step_"):
            continue
        ts = evt.get("ts", "")
        day = ts[:10] if len(ts) >= 10 else "unknown"
        if day not in sessions:
            sessions[day] = {"date": day, "steps": []}
        sessions[day]["steps"].append(evt)
    return sorted(sessions.values(), key=lambda s: s["date"], reverse=True)


def _compute_stats(graph: GraphData, events: list[dict]) -> dict:
    """Compute global statistics."""
    scores = [e.score for e in graph.entities.values()]
    return {
        "total_entities": len(graph.entities),
        "total_relations": len(graph.relations),
        "avg_score": round(sum(scores) / len(scores), 3) if scores else 0,
        "above_threshold": len([s for s in scores if s >= 0.3]),
        "permanent_count": len([e for e in graph.entities.values() if e.retention == "permanent"]),
        "total_events": len(events),
    }


# ---------------------------------------------------------------------------
# HTML Template — vis-network.js + dark theme dashboard
# ---------------------------------------------------------------------------

DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MyMemory Dashboard</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
:root {
    --bg-primary: #0f0f1a;
    --bg-secondary: #161625;
    --bg-card: #1c1c32;
    --bg-card-hover: #24243d;
    --text-primary: #e8e8f0;
    --text-secondary: #8888a8;
    --accent: #4FC3F7;
    --accent-dim: rgba(79,195,247,0.15);
    --success: #66BB6A;
    --warning: #FFA726;
    --danger: #EF5350;
    --border: #2a2a45;
    --glow: rgba(79,195,247,0.3);
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg-primary); color: var(--text-primary); overflow: hidden; }

/* Nav */
nav {
    display: flex; align-items: center; gap: 6px; padding: 10px 20px;
    background: var(--bg-secondary); border-bottom: 1px solid var(--border);
    backdrop-filter: blur(10px);
}
nav .logo { font-size: 20px; font-weight: 700; margin-right: 24px; letter-spacing: -0.5px; }
nav .logo span { color: var(--accent); }
nav .tab {
    padding: 7px 18px; border: 1px solid transparent; border-radius: 8px;
    background: transparent; color: var(--text-secondary); cursor: pointer;
    font-size: 13px; font-weight: 500; transition: all 0.15s;
}
nav .tab.active { background: var(--accent); color: #000; border-color: var(--accent); font-weight: 600; }
nav .tab:hover:not(.active) { background: var(--bg-card); border-color: var(--border); color: var(--text-primary); }
nav .stats-bar {
    margin-left: auto; display: flex; gap: 16px; font-size: 12px; color: var(--text-secondary);
}
nav .stat-item { display: flex; align-items: center; gap: 4px; }
nav .stat-val { color: var(--accent); font-weight: 600; font-size: 14px; }

/* Views */
.view { display: none; height: calc(100vh - 49px); }
.view.active { display: flex; }

/* Graph view */
#graph-container { flex: 1; background: var(--bg-primary); }
#graph-panel {
    width: 260px; padding: 16px; background: var(--bg-secondary);
    border-left: 1px solid var(--border); overflow-y: auto;
    transition: width 0.2s ease;
}
#graph-panel.search-active { width: 340px; }
#graph-panel h3 { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-secondary); margin: 16px 0 8px 0; }
#graph-panel h3:first-child { margin-top: 0; }
#graph-panel label { display: flex; align-items: center; gap: 6px; margin: 3px 0; font-size: 12px; cursor: pointer; color: var(--text-primary); }
#graph-panel input[type="range"] { width: 100%; accent-color: var(--accent); }
#graph-panel input[type="text"] {
    width: 100%; padding: 7px 10px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--bg-card); color: var(--text-primary); font-size: 12px; outline: none;
}
#graph-panel input[type="text"]:focus { border-color: var(--accent); box-shadow: 0 0 0 2px var(--glow); }
.legend-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; }
.panel-divider { border: none; border-top: 1px solid var(--border); margin: 12px 0; }

/* Timeline view */
#timeline-container { flex: 1; padding: 24px; overflow-y: auto; }
.day-group { margin-bottom: 28px; }
.day-header {
    font-size: 12px; font-weight: 600; color: var(--accent); margin-bottom: 10px;
    padding-bottom: 6px; border-bottom: 1px solid var(--border);
    text-transform: uppercase; letter-spacing: 0.5px;
}
.event {
    display: flex; gap: 10px; padding: 10px 14px; margin: 3px 0; border-radius: 8px;
    background: var(--bg-card); cursor: pointer; font-size: 13px;
    border: 1px solid transparent; transition: all 0.1s;
}
.event:hover { background: var(--bg-card-hover); border-color: var(--border); }
.event-time { color: var(--text-secondary); min-width: 44px; font-size: 11px; font-family: monospace; padding-top: 1px; }
.event-type { color: var(--text-primary); font-weight: 500; }
.event-summary { color: var(--text-secondary); margin-left: 4px; }

/* Dream view */
#dream-container { flex: 1; padding: 24px; overflow-y: auto; }
.dream-session {
    background: var(--bg-card); border-radius: 10px; padding: 20px; margin-bottom: 16px;
    border: 1px solid var(--border);
}
.dream-session h3 { margin-bottom: 14px; font-size: 15px; }
.dream-step {
    display: grid; grid-template-columns: 28px 160px 60px 1fr;
    gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--border);
    font-size: 13px; align-items: start;
}
.dream-step:last-child { border-bottom: none; }
.step-duration { color: var(--text-secondary); font-family: monospace; font-size: 11px; }
.step-summary { color: var(--text-secondary); line-height: 1.4; }

/* Search results in graph panel */
#search-results { margin-top: 8px; }
.search-hit {
    padding: 6px 8px; margin: 3px 0; border-radius: 6px;
    background: var(--bg-card); cursor: pointer; font-size: 11px;
    border: 1px solid transparent; transition: all 0.1s;
}
.search-hit:hover { border-color: var(--accent); background: var(--bg-card-hover); }
.search-hit .hit-title { font-weight: 600; color: var(--text-primary); }
.search-hit .hit-meta { color: var(--text-secondary); font-size: 10px; margin-top: 2px; }
.search-hit .hit-score {
    display: inline-block; background: var(--accent); color: #000; padding: 1px 5px;
    border-radius: 3px; font-size: 9px; font-weight: 700; font-family: monospace; margin-right: 4px;
}
.search-hit .hit-type {
    display: inline-block; padding: 1px 5px; border-radius: 3px;
    font-size: 9px; font-weight: 600; text-transform: uppercase;
}
.search-count { font-size: 10px; color: var(--text-secondary); margin-bottom: 6px; }
mark { background: var(--accent); color: #000; border-radius: 2px; padding: 0 1px; }
.node-counter { font-size: 10px; color: var(--text-secondary); margin-top: 6px; }
.relation-link { cursor: pointer; transition: color 0.1s; }
.relation-link:hover { color: var(--accent); text-decoration: underline; }
.dream-step.clickable { cursor: pointer; transition: background 0.1s; border-radius: 6px; padding: 8px 4px; }
.dream-step.clickable:hover { background: var(--bg-card-hover); }

/* Detail panel (slide-in sidebar) */
#detail-panel {
    position: fixed; right: 0; top: 49px; width: 380px; height: calc(100vh - 49px);
    background: var(--bg-secondary); border-left: 1px solid var(--border);
    transform: translateX(100%); transition: transform 0.2s ease;
    overflow-y: auto; padding: 20px; z-index: 50;
}
#detail-panel.open { transform: translateX(0); }
#detail-panel h2 { margin-bottom: 8px; font-size: 18px; padding-right: 30px; }
#detail-panel .section-title {
    color: var(--accent); font-size: 11px; font-weight: 600; margin: 16px 0 6px 0;
    text-transform: uppercase; letter-spacing: 1px;
}
#detail-panel .fact {
    padding: 5px 0; font-size: 12px; border-bottom: 1px solid var(--border);
    color: var(--text-primary); line-height: 1.5;
}
#detail-panel .meta { color: var(--text-secondary); font-size: 12px; margin-bottom: 12px; line-height: 1.6; }
#detail-panel .tag {
    display: inline-block; background: var(--bg-card); padding: 2px 8px; border-radius: 10px;
    font-size: 10px; margin: 2px 3px 2px 0; color: var(--accent);
}
#detail-panel .summary { font-size: 12px; color: var(--text-secondary); font-style: italic; margin: 8px 0; line-height: 1.5; }
#close-panel {
    position: absolute; top: 16px; right: 16px; background: var(--bg-card); border: 1px solid var(--border);
    color: var(--text-primary); font-size: 14px; cursor: pointer; width: 28px; height: 28px;
    border-radius: 6px; display: flex; align-items: center; justify-content: center;
}
#close-panel:hover { background: var(--bg-card-hover); }

/* Timeline filters */
#timeline-filters { width: 220px; padding: 16px; background: var(--bg-secondary); border-left: 1px solid var(--border); overflow-y: auto; }
#timeline-filters h3 { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-secondary); margin-bottom: 10px; }
#timeline-filters label { display: flex; align-items: center; gap: 6px; margin: 3px 0; font-size: 11px; cursor: pointer; }

/* Raw MD */
#raw-md-container {
    margin-top: 10px; max-height: 300px; overflow-y: auto;
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 10px;
    font-family: monospace; font-size: 11px; white-space: pre-wrap; word-break: break-all;
    color: var(--text-secondary); line-height: 1.5; display: none;
}
#raw-md-container.open { display: block; }

/* Action buttons */
.panel-btn {
    padding: 5px 12px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--bg-card); color: var(--text-primary); cursor: pointer;
    font-size: 11px; font-weight: 500; transition: all 0.15s; margin-right: 6px;
}
.panel-btn:hover { background: var(--bg-card-hover); border-color: var(--accent); }
.panel-btn.active { background: var(--accent); color: #000; border-color: var(--accent); }
.panel-actions { display: flex; gap: 6px; margin: 12px 0; }

/* Add relation form */
#add-relation-form {
    display: none; margin-top: 10px; padding: 10px; background: var(--bg-card);
    border: 1px solid var(--border); border-radius: 6px;
}
#add-relation-form.open { display: block; }
#add-relation-form select, #add-relation-form input {
    width: 100%; padding: 6px 8px; margin: 4px 0; border-radius: 4px;
    border: 1px solid var(--border); background: var(--bg-primary); color: var(--text-primary);
    font-size: 12px; outline: none;
}
#add-relation-form select:focus, #add-relation-form input:focus {
    border-color: var(--accent); box-shadow: 0 0 0 2px var(--glow);
}
#add-relation-form button {
    margin-top: 6px; padding: 5px 16px; border-radius: 4px; border: none;
    background: var(--accent); color: #000; font-weight: 600; font-size: 12px;
    cursor: pointer; transition: opacity 0.15s;
}
#add-relation-form button:hover { opacity: 0.85; }
.autocomplete-list {
    position: absolute; z-index: 100; background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 4px; max-height: 150px; overflow-y: auto; width: calc(100% - 20px);
}
.autocomplete-item {
    padding: 4px 8px; cursor: pointer; font-size: 11px; color: var(--text-primary);
}
.autocomplete-item:hover { background: var(--bg-card-hover); }

/* Delete relation button */
.rel-delete {
    display: inline-block; margin-left: 6px; color: var(--danger); cursor: pointer;
    font-size: 12px; opacity: 0.5; transition: opacity 0.15s;
}
.rel-delete:hover { opacity: 1; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3a3a55; }
</style>
</head>
<body>
<nav>
    <div class="logo"><span>&#9673;</span> MyMemory</div>
    <button class="tab active" onclick="switchView('graph', this)">Graph</button>
    <button class="tab" onclick="switchView('timeline', this)">Timeline</button>
    <button class="tab" onclick="switchView('dream', this)">Dream</button>
    <div class="stats-bar" id="nav-stats"></div>
</nav>

<div id="view-graph" class="view active">
    <div id="graph-container"></div>
    <div id="graph-panel">
        <h3>Score filter</h3>
        <label>Min: <span id="score-val" style="color:var(--accent);font-weight:600;">0.00</span></label>
        <input type="range" id="score-slider" min="0" max="1" step="0.05" value="0" oninput="filterGraph()">
        <label style="margin-top:8px;"><input type="checkbox" id="l1-only" onchange="filterGraph()"> L1 only (&ge; 0.3)</label>
        <hr class="panel-divider">
        <h3>Entity types</h3>
        <div id="type-filters"></div>
        <hr class="panel-divider">
        <h3>Search</h3>
        <input type="text" id="graph-search" placeholder="Name, facts, tags..." oninput="searchGraph()" onkeydown="if(event.key==='Escape'){this.value='';searchGraph();}">
        <div id="search-results"></div>
    </div>
</div>

<div id="view-timeline" class="view">
    <div id="timeline-container"></div>
    <div id="timeline-filters">
        <h3>Event types</h3>
        <div id="event-type-filters"></div>
    </div>
</div>

<div id="view-dream" class="view">
    <div id="dream-container"></div>
</div>

<aside id="detail-panel">
    <button id="close-panel" onclick="closePanel()">&#10005;</button>
    <div id="panel-content"></div>
</aside>

<script>
// ===== KEYBOARD SHORTCUTS =====
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') { closePanel(); }
    if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {
        e.preventDefault();
        var searchInput = document.getElementById('graph-search');
        if (searchInput) { searchInput.focus(); }
    }
});
document.addEventListener('click', function(e) {
    var dd = document.getElementById('autocomplete-dropdown');
    if (dd && !e.target.closest('#add-relation-form')) dd.style.display = 'none';
});

// ===== DATA =====
var allNodes = /*__NODES__*/;
var allEdges = /*__EDGES__*/;
var EVENTS = /*__EVENTS__*/;
var ENTITIES = /*__ENTITIES__*/;
var DREAM_SESSIONS = /*__DREAM_SESSIONS__*/;
var STATS = /*__STATS__*/;
var TYPE_COLORS = /*__TYPE_COLORS__*/;
var RELATION_TYPES = /*__RELATION_TYPES__*/;
var DEFAULT_COLOR = "#90A4AE";

var EVENT_ICONS = {
    entity_created:'\u{1F9E0}', entity_updated:'\u{1F4DD}', entity_merged:'\u{1F517}',
    entity_archived:'\u{1F4E6}', relation_created:'\u{1F517}', relation_reinforced:'\u{1F4AA}',
    chat_ingested:'\u{1F4AC}', scores_recalculated:'\u{1F4CA}', context_rebuilt:'\u{1F4C4}',
    dream_step_started:'\u{1F319}', dream_step_completed:'\u2705', dream_step_failed:'\u274C',
    search_performed:'\u{1F50D}', fact_deleted:'\u{1F5D1}\uFE0F', fact_modified:'\u270F\uFE0F',
    co_retrieval:'\u{1F504}', doc_ingested:'\u{1F4E5}'
};

function escapeHtml(s) {
    if (!s) return '';
    var d = document.createElement('div'); d.textContent = s; return d.innerHTML;
}

function highlightMatch(text, query) {
    if (!query || !text) return escapeHtml(text);
    var escaped = escapeHtml(text);
    var qEscaped = escapeHtml(query);
    var re = new RegExp('(' + qEscaped.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
    return escaped.replace(re, '<mark>$1</mark>');
}

// ===== NAV STATS =====
var statsBar = document.getElementById('nav-stats');
statsBar.innerHTML =
    '<div class="stat-item"><span class="stat-val">' + STATS.total_entities + '</span> entities</div>' +
    '<div class="stat-item"><span class="stat-val">' + STATS.total_relations + '</span> relations</div>' +
    '<div class="stat-item"><span class="stat-val">' + STATS.above_threshold + '</span> in L1</div>' +
    '<div class="stat-item">avg <span class="stat-val">' + STATS.avg_score + '</span></div>';

// ===== NAV =====
function switchView(name, btn) {
    document.querySelectorAll('.view').forEach(function(v) { v.classList.remove('active'); });
    document.querySelectorAll('.tab').forEach(function(t) { t.classList.remove('active'); });
    document.getElementById('view-' + name).classList.add('active');
    btn.classList.add('active');
    closePanel();
    if (name === 'timeline') renderTimeline();
    if (name === 'dream') renderDream();
}

// ===== GRAPH (vis-network.js) =====
var nodes = new vis.DataSet(allNodes);
var edges = new vis.DataSet(allEdges);
var network = new vis.Network(
    document.getElementById('graph-container'),
    { nodes: nodes, edges: edges },
    {
        physics: /*__SOLVER__*/,
        interaction: { hover: true, tooltipDelay: 200 },
        nodes: { shape: 'dot', borderWidth: 2 },
        edges: { smooth: { type: 'continuous' }, font: { size: 10, color: '#666' } }
    }
);

// Build type filters
var typeSet = {};
allNodes.forEach(function(n) { typeSet[n.data.type] = true; });
var activeTypes = new Set(Object.keys(typeSet));
var typeFilterEl = document.getElementById('type-filters');
Object.keys(typeSet).sort().forEach(function(t) {
    var label = document.createElement('label');
    label.innerHTML = '<input type="checkbox" checked onchange="filterGraph()" data-type="' + t + '"> ' +
        '<span class="legend-dot" style="background:' + (TYPE_COLORS[t] || DEFAULT_COLOR) + '"></span> ' + t;
    typeFilterEl.appendChild(label);
});

// Click → detail
network.on('click', function(params) {
    if (params.nodes.length === 0) { closePanel(); return; }
    showEntityDetail(params.nodes[0]);
});

// Double-click → focus neighborhood
var focusMode = false;
network.on('doubleClick', function(params) {
    if (params.nodes.length === 0) {
        if (focusMode) {
            nodes.update(allNodes.map(function(n) { return { id: n.id, hidden: false, opacity: 1.0, font: { size: 14, color: '#eee' } }; }));
            edges.update(allEdges.map(function(e) { return { id: e.id, hidden: false }; }));
            focusMode = false;
        }
        return;
    }
    var nodeId = params.nodes[0];
    var neighborIds = new Set([nodeId]);
    allEdges.forEach(function(e) {
        if (e.from === nodeId) neighborIds.add(e.to);
        if (e.to === nodeId) neighborIds.add(e.from);
    });
    nodes.update(allNodes.map(function(n) {
        return { id: n.id, hidden: !neighborIds.has(n.id), opacity: 1.0, font: { size: 14, color: '#eee' } };
    }));
    edges.update(allEdges.map(function(e) {
        return { id: e.id, hidden: !(neighborIds.has(e.from) && neighborIds.has(e.to)) };
    }));
    focusMode = true;
});

function filterGraph() {
    var minScore = parseFloat(document.getElementById('score-slider').value);
    document.getElementById('score-val').textContent = minScore.toFixed(2);
    var l1Only = document.getElementById('l1-only').checked;
    var checks = document.querySelectorAll('#type-filters input:checked');
    var active = {};
    for (var i = 0; i < checks.length; i++) active[checks[i].dataset.type] = true;

    var nodeUpdates = [];
    var visibleCount = 0;
    allNodes.forEach(function(n) {
        var visible = n.data.score >= minScore && active[n.data.type] && (!l1Only || n.data.score >= 0.3);
        if (visible) visibleCount++;
        nodeUpdates.push({ id: n.id, hidden: !visible });
    });
    nodes.update(nodeUpdates);

    var edgeUpdates = [];
    allEdges.forEach(function(e) {
        var fn = nodes.get(e.from), tn = nodes.get(e.to);
        edgeUpdates.push({ id: e.id, hidden: (fn && fn.hidden) || (tn && tn.hidden) });
    });
    edges.update(edgeUpdates);

    // Update node counter
    var counterEl = document.getElementById('node-counter');
    if (!counterEl) {
        counterEl = document.createElement('div');
        counterEl.id = 'node-counter';
        counterEl.className = 'node-counter';
        document.getElementById('score-slider').parentNode.insertBefore(counterEl, document.getElementById('score-slider').nextSibling);
    }
    counterEl.textContent = visibleCount + '/' + allNodes.length + ' visible';
}

function searchGraph() {
    var q = document.getElementById('graph-search').value.toLowerCase().trim();
    var resultsEl = document.getElementById('search-results');

    var panel = document.getElementById('graph-panel');

    // Reset graph opacity
    if (!q) {
        nodes.update(allNodes.map(function(n) {
            return { id: n.id, opacity: 1.0, font: { size: 14, color: '#eee' } };
        }));
        resultsEl.innerHTML = '';
        panel.classList.remove('search-active');
        return;
    }

    // Full-text search across entities (title, facts, tags, aliases, summary)
    var results = [];
    for (var eid in ENTITIES) {
        var e = ENTITIES[eid];
        var searchable = (e.title + ' ' + (e.summary || '') + ' ' + e.facts.join(' ') + ' ' + e.tags.join(' ') + ' ' + e.aliases.join(' ')).toLowerCase();
        if (searchable.indexOf(q) >= 0) {
            var score = 0;
            if (e.title.toLowerCase().indexOf(q) >= 0) score += 10;
            if ((e.summary || '').toLowerCase().indexOf(q) >= 0) score += 5;
            score += e.score * 3;
            results.push({ eid: eid, e: e, searchScore: score });
        }
    }
    results.sort(function(a, b) { return b.searchScore - a.searchScore; });

    // Matched entity IDs for graph highlighting
    var matchedIds = {};
    results.forEach(function(r) { matchedIds[r.eid] = true; });

    // Highlight matching nodes, fade others
    nodes.update(allNodes.map(function(n) {
        var match = matchedIds[n.id];
        return {
            id: n.id,
            opacity: match ? 1.0 : 0.08,
            font: { size: match ? 16 : 14, color: match ? '#fff' : 'rgba(238,238,238,0.08)' }
        };
    }));

    // Focus on first result
    if (results.length) {
        network.focus(results[0].eid, { scale: 1.2, animation: { duration: 400, easingFunction: 'easeInOutQuad' } });
    }

    // Render results list in panel with highlight
    panel.classList.add('search-active');
    var html = '<div class="search-count">' + results.length + ' result' + (results.length !== 1 ? 's' : '') + '</div>';
    html += results.slice(0, 15).map(function(r) {
        var e = r.e;
        var matchingFact = '';
        for (var i = 0; i < e.facts.length; i++) {
            if (e.facts[i].toLowerCase().indexOf(q) >= 0) { matchingFact = e.facts[i]; break; }
        }
        return '<div class="search-hit" onclick="focusAndShow(\'' + r.eid.replace(/'/g, "\\'") + '\')">' +
            '<div><span class="hit-score">' + e.score.toFixed(2) + '</span>' +
            '<span class="hit-type" style="background:' + (TYPE_COLORS[e.type] || DEFAULT_COLOR) + '22;color:' + (TYPE_COLORS[e.type] || DEFAULT_COLOR) + '">' + e.type + '</span> ' +
            '<span class="hit-title">' + highlightMatch(e.title, q) + '</span></div>' +
            (matchingFact ? '<div class="hit-meta">' + highlightMatch(matchingFact.substring(0, 80), q) + '</div>' : '') +
            '</div>';
    }).join('');
    resultsEl.innerHTML = html;
}

// ===== DETAIL PANEL =====
var _currentEntityId = null;

function showEntityDetail(eid) {
    var e = ENTITIES[eid];
    if (!e) return;
    _currentEntityId = eid;
    var relFrom = allEdges.filter(function(r) { return r.from === eid; });
    var relTo = allEdges.filter(function(r) { return r.to === eid; });

    var c = TYPE_COLORS[e.type] || DEFAULT_COLOR;
    var html = '<h2 style="color:' + c + '">' + escapeHtml(e.title) + '</h2>';
    html += '<div class="meta">';
    html += '<strong>' + e.type + '</strong> &middot; score: ' + e.score + ' &middot; freq: ' + e.frequency + ' &middot; ' + e.retention + '<br>';
    html += 'Created: ' + e.created + ' &middot; Last: ' + e.last_mentioned;
    html += '</div>';
    if (e.summary) html += '<div class="summary">' + escapeHtml(e.summary) + '</div>';
    if (e.aliases && e.aliases.length) html += '<div class="meta">Aliases: ' + e.aliases.map(escapeHtml).join(', ') + '</div>';
    if (e.tags && e.tags.length) {
        html += '<div>';
        e.tags.forEach(function(t) { html += '<span class="tag">' + escapeHtml(t) + '</span>'; });
        html += '</div>';
    }

    // Action buttons
    html += '<div class="panel-actions">';
    html += '<button class="panel-btn" id="btn-raw-md" onclick="toggleRawMd(\'' + eid.replace(/'/g, "\\'") + '\')">Raw MD</button>';
    html += '<button class="panel-btn" id="btn-add-rel" onclick="toggleAddRelation()">+ Relation</button>';
    html += '</div>';
    html += '<div id="raw-md-container"></div>';

    // Add relation form
    html += '<div id="add-relation-form">';
    html += '<div style="position:relative;">';
    html += '<select id="rel-type-select">';
    RELATION_TYPES.forEach(function(t) { html += '<option value="' + t + '">' + t + '</option>'; });
    html += '</select>';
    html += '<input type="text" id="rel-target-input" placeholder="Target entity..." oninput="autocompleteTarget()" onfocus="autocompleteTarget()" onkeydown="acKeydown(event)">';
    html += '<div id="autocomplete-dropdown" class="autocomplete-list" style="display:none;"></div>';
    html += '</div>';
    html += '<button onclick="submitAddRelation()">Add</button>';
    html += '</div>';

    html += '<div class="section-title">Facts (' + e.facts.length + ')</div>';
    e.facts.forEach(function(f) { html += '<div class="fact">' + escapeHtml(f) + '</div>'; });
    if (relFrom.length) {
        html += '<div class="section-title">Relations out (' + relFrom.length + ')</div>';
        relFrom.forEach(function(r) {
            var target = ENTITIES[r.to];
            var tname = target ? target.title : r.to;
            var targetId = r.to.replace(/'/g, "\\'");
            var rtype = r.data ? r.data.type : r.label;
            html += '<div class="fact">&rarr; ' + rtype + ' <strong class="relation-link" onclick="focusAndShow(\'' + targetId + '\')">' + escapeHtml(tname) + '</strong>';
            html += '<span class="rel-delete" onclick="deleteRelation(\'' + eid.replace(/'/g, "\\'") + '\',\'' + targetId + '\',\'' + rtype + '\')" title="Delete relation">&times;</span>';
            html += '</div>';
        });
    }
    if (relTo.length) {
        html += '<div class="section-title">Relations in (' + relTo.length + ')</div>';
        relTo.forEach(function(r) {
            var source = ENTITIES[r.from];
            var sname = source ? source.title : r.from;
            var sourceId = r.from.replace(/'/g, "\\'");
            var rtype = r.data ? r.data.type : r.label;
            html += '<div class="fact">&larr; ' + rtype + ' <strong class="relation-link" onclick="focusAndShow(\'' + sourceId + '\')">' + escapeHtml(sname) + '</strong>';
            html += '<span class="rel-delete" onclick="deleteRelation(\'' + sourceId + '\',\'' + eid.replace(/'/g, "\\'") + '\',\'' + rtype + '\')" title="Delete relation">&times;</span>';
            html += '</div>';
        });
    }
    html += '<div class="section-title">Recent mentions</div>';
    html += '<div class="meta">' + ((e.mention_dates && e.mention_dates.length) ? e.mention_dates.join(', ') : 'None') + '</div>';

    document.getElementById('panel-content').innerHTML = html;
    document.getElementById('detail-panel').classList.add('open');
}

// ===== RAW MD =====
function toggleRawMd(eid) {
    var container = document.getElementById('raw-md-container');
    var btn = document.getElementById('btn-raw-md');
    if (container.classList.contains('open')) {
        container.classList.remove('open');
        btn.classList.remove('active');
        return;
    }
    container.textContent = 'Loading...';
    container.classList.add('open');
    btn.classList.add('active');
    fetch('/api/entity/' + encodeURIComponent(eid) + '/raw')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (data.error) { container.textContent = 'Error: ' + data.error; return; }
            container.textContent = data.content;
        })
        .catch(function(err) { container.textContent = 'Fetch error: ' + err; });
}

// ===== ADD RELATION =====
function toggleAddRelation() {
    var form = document.getElementById('add-relation-form');
    var btn = document.getElementById('btn-add-rel');
    if (form.classList.contains('open')) {
        form.classList.remove('open');
        btn.classList.remove('active');
    } else {
        form.classList.add('open');
        btn.classList.add('active');
    }
}

var _acSelected = -1;
function autocompleteTarget() {
    var input = document.getElementById('rel-target-input');
    var dropdown = document.getElementById('autocomplete-dropdown');
    var q = input.value.toLowerCase().trim();
    _acSelected = -1;
    if (!q) { dropdown.style.display = 'none'; return; }
    var matches = [];
    for (var eid in ENTITIES) {
        if (eid === _currentEntityId) continue;
        var e = ENTITIES[eid];
        if (e.title.toLowerCase().indexOf(q) >= 0 || eid.indexOf(q) >= 0) {
            matches.push({ eid: eid, title: e.title });
        }
    }
    if (!matches.length) { dropdown.style.display = 'none'; return; }
    dropdown.innerHTML = matches.slice(0, 10).map(function(m, i) {
        return '<div class="autocomplete-item" data-idx="' + i + '" data-eid="' + m.eid + '" onmousedown="pickAutocomplete(\'' + m.eid.replace(/'/g, "\\'") + '\',\'' + escapeHtml(m.title).replace(/'/g, "\\'") + '\')">' + escapeHtml(m.title) + ' <span style="color:var(--text-secondary);font-size:9px;">' + m.eid + '</span></div>';
    }).join('');
    dropdown.style.display = 'block';
}

function pickAutocomplete(eid, title) {
    document.getElementById('rel-target-input').value = title;
    document.getElementById('rel-target-input').dataset.eid = eid;
    document.getElementById('autocomplete-dropdown').style.display = 'none';
}

function acKeydown(e) {
    var dropdown = document.getElementById('autocomplete-dropdown');
    var items = dropdown.querySelectorAll('.autocomplete-item');
    if (!items.length) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); _acSelected = Math.min(_acSelected + 1, items.length - 1); highlightAc(items); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); _acSelected = Math.max(_acSelected - 1, 0); highlightAc(items); }
    else if (e.key === 'Enter' && _acSelected >= 0) { e.preventDefault(); items[_acSelected].dispatchEvent(new Event('mousedown')); }
}

function highlightAc(items) {
    for (var i = 0; i < items.length; i++) {
        items[i].style.background = i === _acSelected ? 'var(--bg-card-hover)' : '';
    }
}

function submitAddRelation() {
    var fromId = _currentEntityId;
    var relType = document.getElementById('rel-type-select').value;
    var targetInput = document.getElementById('rel-target-input');
    var toId = targetInput.dataset.eid;
    if (!toId) {
        // Try to match by slug
        var q = targetInput.value.toLowerCase().trim().replace(/\s+/g, '-');
        if (ENTITIES[q]) toId = q;
    }
    if (!fromId || !toId || !relType) { alert('Select a target entity.'); return; }

    fetch('/api/relation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ from: fromId, to: toId, type: relType })
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        if (data.error) { alert('Error: ' + data.error); return; }
        // Add edge to vis.js
        var edgeData = {
            from: fromId, to: toId, label: relType,
            title: relType + ' (strength: 0.50)',
            width: 2, arrows: 'to', color: { opacity: 0.6 },
            data: { type: relType, strength: 0.5, last_reinforced: '', mention_count: 1 }
        };
        edges.add(edgeData);
        allEdges.push(edgeData);
        // Refresh detail panel
        showEntityDetail(fromId);
    })
    .catch(function(err) { alert('Fetch error: ' + err); });
}

// ===== DELETE RELATION =====
function deleteRelation(fromId, toId, relType) {
    if (!confirm('Delete relation: ' + fromId + ' \u2192 ' + relType + ' \u2192 ' + toId + '?')) return;
    fetch('/api/relation', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ from: fromId, to: toId, type: relType })
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        if (data.error) { alert('Error: ' + data.error); return; }
        // Remove edge from vis.js
        var toRemove = edges.get({ filter: function(e) {
            return e.from === fromId && e.to === toId && ((e.data && e.data.type === relType) || e.label === relType);
        }});
        if (toRemove.length) edges.remove(toRemove[0].id);
        allEdges = allEdges.filter(function(e) {
            return !(e.from === fromId && e.to === toId && ((e.data && e.data.type === relType) || e.label === relType));
        });
        // Refresh detail panel
        showEntityDetail(_currentEntityId || fromId);
    })
    .catch(function(err) { alert('Fetch error: ' + err); });
}

function focusAndShow(eid) {
    network.focus(eid, { scale: 1.5, animation: { duration: 400, easingFunction: 'easeInOutQuad' } });
    network.selectNodes([eid]);
    showEntityDetail(eid);
}

function closePanel() { document.getElementById('detail-panel').classList.remove('open'); }

// ===== TIMELINE =====
var timelineRendered = false;
function renderTimeline() {
    var container = document.getElementById('timeline-container');
    var filterContainer = document.getElementById('event-type-filters');

    if (!timelineRendered) {
        var typeSet = {};
        EVENTS.forEach(function(e) { typeSet[e.type] = true; });
        filterContainer.innerHTML = Object.keys(typeSet).sort().map(function(t) {
            return '<label><input type="checkbox" checked onchange="renderTimeline()" data-etype="' + t + '"> ' +
                (EVENT_ICONS[t] || '&middot;') + ' ' + t + '</label>';
        }).join('');
        timelineRendered = true;
    }

    var checks = document.querySelectorAll('#event-type-filters input:checked');
    var active = {};
    for (var i = 0; i < checks.length; i++) active[checks[i].dataset.etype] = true;
    var filtered = EVENTS.filter(function(e) { return active[e.type]; });

    var days = {};
    filtered.forEach(function(evt) {
        var day = evt.ts.substring(0, 10);
        if (!days[day]) days[day] = [];
        days[day].push(evt);
    });

    container.innerHTML = Object.keys(days).sort().reverse().map(function(day) {
        return '<div class="day-group"><div class="day-header">' + day + ' &mdash; ' + days[day].length + ' events</div>' +
            days[day].slice().reverse().map(function(evt, idx) {
                return '<div class="event" data-day="' + day + '" data-idx="' + idx + '" onclick="showEventDetail(this)">' +
                    '<span class="event-time">' + evt.ts.substring(11, 16) + '</span>' +
                    '<span class="event-type">' + (EVENT_ICONS[evt.type] || '') + ' ' + evt.type + '</span>' +
                    '<span class="event-summary">' + escapeHtml(summarizeEvent(evt)) + '</span></div>';
            }).join('') + '</div>';
    }).join('') || '<p style="color:var(--text-secondary);padding:20px;">No events recorded yet.</p>';

    // Store events for click lookup
    window._timelineFiltered = filtered;
    window._timelineDays = days;
}

function summarizeEvent(evt) {
    var d = evt.data || {};
    if (d.entity_id) return d.entity_id;
    if (d.title) return d.title;
    if (d.query) return '"' + d.query + '"';
    if (d.step) return 'Step ' + d.step;
    if (d.from && d.to) return d.from + ' \u2192 ' + d.to;
    return JSON.stringify(d).substring(0, 60);
}

function showEventDetail(el) {
    var day = el.dataset.day;
    var idx = parseInt(el.dataset.idx);
    var evts = (window._timelineDays || {})[day];
    if (!evts) return;
    var evt = evts.slice().reverse()[idx];
    if (!evt) return;

    var html = '<h2>' + (EVENT_ICONS[evt.type] || '') + ' ' + evt.type + '</h2>';
    html += '<div class="meta">Time: ' + evt.ts + '<br>Source: ' + (evt.source || 'unknown') + '</div>';
    html += '<div class="section-title">Data</div>';
    html += '<pre style="font-size:11px;color:var(--text-secondary);white-space:pre-wrap;word-break:break-all;line-height:1.5;">' +
        escapeHtml(JSON.stringify(evt.data, null, 2)) + '</pre>';
    document.getElementById('panel-content').innerHTML = html;
    document.getElementById('detail-panel').classList.add('open');
}

// ===== DREAM =====
function renderDream() {
    var container = document.getElementById('dream-container');
    if (!DREAM_SESSIONS.length) {
        container.innerHTML = '<p style="color:var(--text-secondary);padding:20px;">No dream sessions found in event log.</p>';
        return;
    }
    container.innerHTML = DREAM_SESSIONS.map(function(session, si) {
        return '<div class="dream-session"><h3>\u{1F319} Dream \u2014 ' + session.date + '</h3>' +
            session.steps.map(function(step, sti) {
                var d = step.data || {};
                var status = step.type === 'dream_step_completed' ? '\u2705' :
                             step.type === 'dream_step_failed' ? '\u274C' :
                             step.type === 'dream_step_started' ? '\u23F3' : '\u23ED';
                return '<div class="dream-step clickable" onclick="showDreamStepDetail(' + si + ',' + sti + ')">' +
                    '<span>' + status + '</span>' +
                    '<span>' + (d.step || '') + ' ' + escapeHtml(d.step_name || d.description || '') + '</span>' +
                    '<span class="step-duration">' + (d.duration_s ? d.duration_s.toFixed(1) + 's' : '') + '</span>' +
                    '<span class="step-summary">' + escapeHtml(d.summary || d.error || '') + '</span></div>';
            }).join('') + '</div>';
    }).join('');
}


// ===== DREAM STEP DETAIL =====
function showDreamStepDetail(sessionIdx, stepIdx) {
    var session = DREAM_SESSIONS[sessionIdx];
    if (!session) return;
    var step = session.steps[stepIdx];
    if (!step) return;
    var d = step.data || {};

    var status = step.type === 'dream_step_completed' ? '\u2705 Completed' :
                 step.type === 'dream_step_failed' ? '\u274C Failed' :
                 step.type === 'dream_step_started' ? '\u23F3 Started' : '\u23ED Skipped';

    var html = '<h2>\u{1F319} ' + escapeHtml(d.step_name || d.description || 'Dream Step') + '</h2>';
    html += '<div class="meta">';
    html += '<strong>Status:</strong> ' + status + '<br>';
    html += '<strong>Session:</strong> ' + session.date + '<br>';
    if (d.step) html += '<strong>Step:</strong> ' + d.step + '<br>';
    if (d.duration_s) html += '<strong>Duration:</strong> ' + d.duration_s.toFixed(1) + 's<br>';
    if (step.ts) html += '<strong>Time:</strong> ' + step.ts + '<br>';
    html += '</div>';

    if (d.summary) {
        html += '<div class="section-title">Summary</div>';
        html += '<div class="fact">' + escapeHtml(d.summary) + '</div>';
    }
    if (d.error) {
        html += '<div class="section-title">Error</div>';
        html += '<div class="fact" style="color:var(--danger)">' + escapeHtml(d.error) + '</div>';
    }

    // Show all extra data fields
    var extraKeys = Object.keys(d).filter(function(k) {
        return ['step', 'step_name', 'description', 'duration_s', 'summary', 'error'].indexOf(k) < 0;
    });
    if (extraKeys.length) {
        html += '<div class="section-title">Details</div>';
        html += '<pre style="font-size:11px;color:var(--text-secondary);white-space:pre-wrap;word-break:break-all;line-height:1.5;">';
        var extraData = {};
        extraKeys.forEach(function(k) { extraData[k] = d[k]; });
        html += escapeHtml(JSON.stringify(extraData, null, 2));
        html += '</pre>';
    }

    document.getElementById('panel-content').innerHTML = html;
    document.getElementById('detail-panel').classList.add('open');
}

</script>
</body>
</html>
"""
