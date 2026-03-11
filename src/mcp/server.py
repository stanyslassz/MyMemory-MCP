"""MCP Server: 3 tools — get_context, save_chat, search_rag."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.core.config import load_config
from src.memory.graph import load_graph, save_graph
from src.memory.store import save_chat as store_save_chat
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
    from datetime import date as date_type
    from src.memory.mentions import add_mention
    from src.memory.scoring import recalculate_all_scores

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
        graph = recalculate_all_scores(graph, config)
        save_graph(memory_path, graph)

    return {
        "query": query,
        "results": enriched_results,
        "total": len(enriched_results),
    }


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
