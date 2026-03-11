"""MCP Server: 7 tools — get_context, save_chat, search_rag + CRUD tools."""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from datetime import date as date_cls
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.core.config import Config, load_config
from src.core.models import GraphData
from src.core.utils import slugify
from src.memory.graph import load_graph, remove_relation, save_graph
from src.memory.store import (
    read_entity,
    remove_relation_line,
    save_chat as store_save_chat,
    write_entity,
    parse_observation,
    format_observation,
)
from src.pipeline.indexer import search as faiss_search

from src.core.models import SearchResult

logger = logging.getLogger(__name__)

mcp = FastMCP("memory-ai")

# Load config at module level — will be initialized when server starts
_config = None


def _get_config():
    global _config
    if _config is None:
        _config = load_config()
    return _config


@mcp.tool()
def get_context() -> str:
    """Returns the pre-compiled memory context.

    Reads _context.md. Falls back to _index.md if absent.
    """
    config = _get_config()
    memory_path = config.memory_path

    context_file = memory_path / "_context.md"
    if context_file.exists():
        return context_file.read_text(encoding="utf-8")

    index_file = memory_path / "_index.md"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")

    return "No memory context available. Run 'memory run' to process chats and generate context."


@mcp.tool()
def save_chat(messages: list[dict]) -> dict:
    """Saves a conversation for later processing.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        Status dict with 'status' and 'file' keys, or error dict.
    """
    # Validate messages
    if not isinstance(messages, list) or len(messages) == 0:
        return {"status": "error", "message": "messages must be a non-empty list"}

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return {"status": "error", "message": f"message at index {i} must be a dict"}
        if "role" not in msg or not isinstance(msg.get("role"), str):
            return {"status": "error", "message": f"message at index {i} missing valid 'role' (str)"}
        if "content" not in msg or not isinstance(msg.get("content"), str):
            return {"status": "error", "message": f"message at index {i} missing valid 'content' (str)"}

    config = _get_config()
    filepath = store_save_chat(messages, config.memory_path)
    rel_path = filepath.relative_to(config.memory_path)
    return {"status": "saved", "file": str(rel_path)}


def _rrf_fusion(
    faiss_results: list[SearchResult],
    keyword_results,
    graph,
    k: int = 60,
    w_sem: float = 0.5,
    w_kw: float = 0.3,
    w_actr: float = 0.2,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion combining semantic, keyword, and ACT-R signals.

    Returns list of (entity_id, rrf_score) sorted descending.
    """
    sem_ranks = {r.entity_id: i + 1 for i, r in enumerate(faiss_results)}
    kw_ranks = {r.entity_id: i + 1 for i, r in enumerate(keyword_results)}

    all_ids = set(sem_ranks) | set(kw_ranks)

    # ACT-R ranking
    actr_scores = {}
    for eid in all_ids:
        e = graph.entities.get(eid)
        actr_scores[eid] = e.score if e else 0.0
    sorted_actr = sorted(actr_scores.items(), key=lambda x: x[1], reverse=True)
    actr_ranks = {eid: i + 1 for i, (eid, _) in enumerate(sorted_actr)}

    # Default rank for missing entries — penalises absent signals
    default_rank = max(len(faiss_results), len(keyword_results), len(all_ids)) + 10

    scored = []
    for eid in all_ids:
        sr = sem_ranks.get(eid, default_rank)
        kr = kw_ranks.get(eid, default_rank)
        ar = actr_ranks.get(eid, default_rank)

        score = w_sem / (k + sr) + w_kw / (k + kr) + w_actr / (k + ar)
        scored.append((eid, score))

    return sorted(scored, key=lambda x: x[1], reverse=True)


@mcp.tool()
def search_rag(query: str) -> dict:
    """Semantic search across memory.

    Args:
        query: The search query string.

    Returns:
        Dict with search results enriched with entity relations.
    """
    config = _get_config()
    memory_path = config.memory_path

    # FAISS search — graceful fallback if index doesn't exist
    try:
        results = faiss_search(query, config, memory_path)
    except Exception:
        logger.warning("FAISS search failed for query %r, returning empty results", query, exc_info=True)
        return {"query": query, "results": [], "total": 0}

    # Enrich with graph relations — graceful fallback if graph fails
    try:
        graph = load_graph(memory_path)
    except Exception:
        logger.warning("Failed to load graph, returning results without enrichment", exc_info=True)
        enriched_results = [
            {
                "entity_id": r.entity_id,
                "file": r.file,
                "score": r.score,
                "title": r.entity_id,
                "type": "unknown",
                "relations": [],
            }
            for r in results
        ]
        return {"query": query, "results": enriched_results, "total": len(enriched_results)}

    # Hybrid re-ranking: RRF (semantic + keyword + ACT-R) when FTS index exists
    fts_db_path = memory_path / config.search.fts_db_path
    if config.search.hybrid_enabled and fts_db_path.exists():
        from src.pipeline.keyword_index import search_keyword

        kw_results = search_keyword(query, fts_db_path, top_k=config.faiss.top_k * 2)
        if kw_results:
            ranked = _rrf_fusion(
                results,
                kw_results,
                graph,
                k=config.search.rrf_k,
                w_sem=config.search.weight_semantic,
                w_kw=config.search.weight_keyword,
                w_actr=config.search.weight_actr,
            )
            # Build lookup from existing FAISS results
            result_map = {r.entity_id: r for r in results}
            # Add keyword-only results (not found by FAISS)
            for eid, _score in ranked:
                if eid not in result_map and eid in graph.entities:
                    e = graph.entities[eid]
                    result_map[eid] = SearchResult(
                        entity_id=eid,
                        file=e.file,
                        chunk="[keyword match]",
                        score=0.0,
                    )
            reranked = []
            for eid, rrf_score in ranked:
                if eid in result_map:
                    r = result_map[eid]
                    r.score = rrf_score
                    reranked.append(r)
            results = reranked[: config.faiss.top_k]
        else:
            # No keyword matches — fall back to linear re-ranking
            for result in results:
                entity = graph.entities.get(result.entity_id)
                graph_score = entity.score if entity else 0.0
                result.score = result.score * 0.6 + graph_score * 0.4
            results.sort(key=lambda r: r.score, reverse=True)
    else:
        # No FTS index or hybrid disabled — original linear re-ranking
        for result in results:
            entity = graph.entities.get(result.entity_id)
            graph_score = entity.score if entity else 0.0
            result.score = result.score * 0.6 + graph_score * 0.4
        results.sort(key=lambda r: r.score, reverse=True)

    # Build adjacency dict once for O(1) per-entity relation lookup
    adjacency = defaultdict(list)
    for rel in graph.relations:
        adjacency[rel.from_entity].append(("outgoing", rel))
        adjacency[rel.to_entity].append(("incoming", rel))

    enriched_results = []

    for result in results:
        entity = graph.entities.get(result.entity_id)
        relations = []
        if entity:
            for direction, rel in adjacency.get(result.entity_id, []):
                if direction == "outgoing":
                    target = graph.entities.get(rel.to_entity)
                    if target:
                        relations.append({
                            "type": rel.type,
                            "target": target.title,
                            "target_id": rel.to_entity,
                        })
                else:
                    source = graph.entities.get(rel.from_entity)
                    if source:
                        relations.append({
                            "type": rel.type,
                            "source": source.title,
                            "source_id": rel.from_entity,
                        })

        enriched_results.append({
            "entity_id": result.entity_id,
            "file": result.file,
            "score": result.score,
            "title": entity.title if entity else result.entity_id,
            "type": entity.type if entity else "unknown",
            "relations": relations,
        })

    # L2→L1 re-emergence: bump mention_dates for retrieved entities
    # Scores are recalculated during `memory run` / `memory dream`, not per query.
    from datetime import date as date_type
    from src.memory.mentions import add_mention

    today = date_type.today().isoformat()
    promoted = False
    for result in results:
        entity_id = result.entity_id
        if entity_id in graph.entities:
            entity = graph.entities[entity_id]
            entity.mention_dates, entity.monthly_buckets = add_mention(
                today, entity.mention_dates, entity.monthly_buckets,
                window_size=config.scoring.window_size,
            )
            entity.last_mentioned = today
            promoted = True

    if promoted:
        try:
            save_graph(memory_path, graph)
        except RuntimeError:
            logger.warning("Could not save graph after L2→L1 bump (locked by another process)")

    return {
        "query": query,
        "results": enriched_results,
        "total": len(enriched_results),
    }


# ── Helper: resolve entity name → entity_id ────────────────


def _resolve_entity_by_name(name: str, graph: GraphData) -> str | None:
    """Resolve entity name to entity_id via slug, title, or alias match."""
    slug = slugify(name)
    if slug in graph.entities:
        return slug
    for eid, e in graph.entities.items():
        if e.title.lower() == name.lower():
            return eid
    for eid, e in graph.entities.items():
        if any(a.lower() == name.lower() for a in (e.aliases or [])):
            return eid
    return None


# ── Tool implementations (sync, testable without MCP) ──────


def _delete_fact_impl(entity_name: str, fact_content: str, config: Config) -> str:
    """Implementation for delete_fact. Returns JSON string."""
    memory_path = config.memory_path
    graph = load_graph(memory_path)
    entity_id = _resolve_entity_by_name(entity_name, graph)
    if not entity_id:
        return json.dumps({"status": "error", "message": f"Entity '{entity_name}' not found"})

    entity = graph.entities[entity_id]
    filepath = memory_path / entity.file
    if not filepath.exists():
        return json.dumps({"status": "error", "message": f"Entity file not found: {entity.file}"})

    frontmatter, sections = read_entity(filepath)
    facts = sections.get("Facts", [])

    # Find matching fact (case-insensitive substring match on content)
    content_lower = fact_content.lower()
    matched_idx = None
    for i, line in enumerate(facts):
        obs = parse_observation(line)
        if obs and content_lower in obs["content"].lower():
            matched_idx = i
            break

    if matched_idx is None:
        return json.dumps({"status": "error", "message": f"Fact containing '{fact_content}' not found in {entity.title}"})

    deleted_line = facts.pop(matched_idx)
    sections["Facts"] = facts

    # Add history entry
    today = date_cls.today().isoformat()
    history = sections.get("History", [])
    history.append(f"- {today}: Deleted fact: {fact_content[:60]}")
    sections["History"] = history

    write_entity(filepath, frontmatter, sections)

    return json.dumps({
        "status": "deleted",
        "entity": entity.title,
        "deleted_fact": deleted_line,
    })


def _delete_relation_impl(from_entity: str, to_entity: str, relation_type: str, config: Config) -> str:
    """Implementation for delete_relation. Returns JSON string."""
    memory_path = config.memory_path
    graph = load_graph(memory_path)

    from_id = _resolve_entity_by_name(from_entity, graph)
    if not from_id:
        return json.dumps({"status": "error", "message": f"Source entity '{from_entity}' not found"})

    to_id = _resolve_entity_by_name(to_entity, graph)
    if not to_id:
        return json.dumps({"status": "error", "message": f"Target entity '{to_entity}' not found"})

    # Remove from graph
    removed = remove_relation(graph, from_id, to_id, relation_type)
    if not removed:
        return json.dumps({
            "status": "error",
            "message": f"Relation '{relation_type}' from '{from_entity}' to '{to_entity}' not found in graph",
        })

    # Remove from source entity MD file
    from_entity_data = graph.entities[from_id]
    from_path = memory_path / from_entity_data.file
    to_entity_data = graph.entities[to_id]
    if from_path.exists():
        remove_relation_line(from_path, relation_type, to_entity_data.title)

    save_graph(memory_path, graph)

    return json.dumps({
        "status": "deleted",
        "from": from_entity_data.title,
        "to": to_entity_data.title,
        "type": relation_type,
    })


def _modify_fact_impl(entity_name: str, old_content: str, new_content: str, config: Config) -> str:
    """Implementation for modify_fact. Returns JSON string."""
    memory_path = config.memory_path
    graph = load_graph(memory_path)
    entity_id = _resolve_entity_by_name(entity_name, graph)
    if not entity_id:
        return json.dumps({"status": "error", "message": f"Entity '{entity_name}' not found"})

    entity = graph.entities[entity_id]
    filepath = memory_path / entity.file
    if not filepath.exists():
        return json.dumps({"status": "error", "message": f"Entity file not found: {entity.file}"})

    frontmatter, sections = read_entity(filepath)
    facts = sections.get("Facts", [])

    # Find matching fact
    content_lower = old_content.lower()
    matched_idx = None
    for i, line in enumerate(facts):
        obs = parse_observation(line)
        if obs and content_lower in obs["content"].lower():
            matched_idx = i
            break

    if matched_idx is None:
        return json.dumps({"status": "error", "message": f"Fact containing '{old_content}' not found in {entity.title}"})

    # Parse the matched line to preserve metadata
    old_line = facts[matched_idx]
    obs = parse_observation(old_line)
    old_fact_content = obs["content"]
    obs["content"] = new_content
    new_line = format_observation(obs)
    facts[matched_idx] = new_line
    sections["Facts"] = facts

    # Add history entry
    today = date_cls.today().isoformat()
    history = sections.get("History", [])
    history.append(f"- {today}: Modified fact: '{old_fact_content[:40]}' -> '{new_content[:40]}'")
    sections["History"] = history

    write_entity(filepath, frontmatter, sections)

    return json.dumps({
        "status": "modified",
        "entity": entity.title,
        "old_fact": old_line,
        "new_fact": new_line,
    })


def _correct_entity_impl(entity_name: str, field: str, new_value: str, config: Config) -> str:
    """Implementation for correct_entity. Returns JSON string."""
    allowed_fields = {"title", "type", "aliases", "retention"}
    if field not in allowed_fields:
        return json.dumps({
            "status": "error",
            "message": f"Invalid field '{field}'. Allowed: {', '.join(sorted(allowed_fields))}",
        })

    memory_path = config.memory_path
    graph = load_graph(memory_path)
    entity_id = _resolve_entity_by_name(entity_name, graph)
    if not entity_id:
        return json.dumps({"status": "error", "message": f"Entity '{entity_name}' not found"})

    entity = graph.entities[entity_id]
    filepath = memory_path / entity.file
    if not filepath.exists():
        return json.dumps({"status": "error", "message": f"Entity file not found: {entity.file}"})

    frontmatter, sections = read_entity(filepath)
    old_value = getattr(frontmatter, field)
    changes = {"field": field, "old": str(old_value), "new": new_value}

    if field == "title":
        frontmatter.title = new_value
        entity.title = new_value
    elif field == "type":
        frontmatter.type = new_value
        entity.type = new_value
        # Move file to correct folder
        new_folder = config.get_folder_for_type(new_value)
        new_filepath = memory_path / new_folder / filepath.name
        if new_filepath != filepath:
            new_filepath.parent.mkdir(parents=True, exist_ok=True)
            write_entity(filepath, frontmatter, sections)  # Save before move
            shutil.move(str(filepath), str(new_filepath))
            new_rel = str(new_filepath.relative_to(memory_path))
            entity.file = new_rel
            filepath = new_filepath  # Update for final write below
            changes["moved_to"] = new_rel
    elif field == "aliases":
        aliases = [a.strip() for a in new_value.split(",") if a.strip()]
        frontmatter.aliases = aliases
        entity.aliases = aliases
    elif field == "retention":
        frontmatter.retention = new_value

    # Add history entry
    today = date_cls.today().isoformat()
    history = sections.get("History", [])
    history.append(f"- {today}: Corrected {field}: '{old_value}' -> '{new_value}'")
    sections["History"] = history

    write_entity(filepath, frontmatter, sections)
    save_graph(memory_path, graph)

    return json.dumps({"status": "updated", "entity": entity.title, "changes": changes})


# ── MCP tool wrappers ───────────────────────────────────────


@mcp.tool()
def delete_fact(entity_name: str, fact_content: str) -> str:
    """Delete a specific fact from an entity's memory.

    Args:
        entity_name: Name of the entity (title, slug, or alias)
        fact_content: Content of the fact to delete (partial match supported)
    """
    config = _get_config()
    return _delete_fact_impl(entity_name, fact_content, config)


@mcp.tool()
def delete_relation(from_entity: str, to_entity: str, relation_type: str) -> str:
    """Delete a relation between two entities.

    Args:
        from_entity: Source entity name
        to_entity: Target entity name
        relation_type: Type of relation (affects, parent_of, friend_of, etc.)
    """
    config = _get_config()
    return _delete_relation_impl(from_entity, to_entity, relation_type, config)


@mcp.tool()
def modify_fact(entity_name: str, old_content: str, new_content: str) -> str:
    """Modify a fact's content while preserving its metadata (category, date, valence).

    Args:
        entity_name: Name of the entity
        old_content: Content to find (partial match)
        new_content: New content to replace with
    """
    config = _get_config()
    return _modify_fact_impl(entity_name, old_content, new_content, config)


@mcp.tool()
def correct_entity(entity_name: str, field: str, new_value: str) -> str:
    """Correct an entity's metadata (title, type, aliases, retention).

    Args:
        entity_name: Name of the entity
        field: Field to correct (title, type, aliases, retention)
        new_value: New value (for aliases, comma-separated list)
    """
    config = _get_config()
    return _correct_entity_impl(entity_name, field, new_value, config)


def run_server(config=None, transport_override: str | None = None):
    """Start the MCP server.

    Args:
        config: Pre-loaded Config instance (honors CLI --config). Falls back to _get_config().
        transport_override: Override transport from CLI --transport flag. If None, uses config.
    """
    if config is not None:
        global _config
        _config = config
    else:
        config = _get_config()

    transport = transport_override or config.mcp_transport

    # Apply host/port from config for SSE transport (LAN-reachable)
    mcp.settings.host = config.mcp_host
    mcp.settings.port = config.mcp_port

    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse")
