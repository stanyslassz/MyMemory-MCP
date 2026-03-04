"""MCP Server: 3 tools — get_context, save_chat, search_rag."""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.core.config import load_config
from src.memory.graph import load_graph
from src.memory.store import save_chat as store_save_chat
from src.pipeline.indexer import search as faiss_search

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
        Status dict with 'status' and 'file' keys.
    """
    config = _get_config()
    filepath = store_save_chat(messages, config.memory_path)
    rel_path = filepath.relative_to(config.memory_path)
    return {"status": "saved", "file": str(rel_path)}


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

    # FAISS search
    results = faiss_search(query, config, memory_path)

    # Enrich with graph relations
    graph = load_graph(memory_path)
    enriched_results = []

    for result in results:
        entity = graph.entities.get(result.entity_id)
        relations = []
        if entity:
            for rel in graph.relations:
                if rel.from_entity == result.entity_id:
                    target = graph.entities.get(rel.to_entity)
                    if target:
                        relations.append({
                            "type": rel.type,
                            "target": target.title,
                            "target_id": rel.to_entity,
                        })
                elif rel.to_entity == result.entity_id:
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

    return {
        "query": query,
        "results": enriched_results,
        "total": len(enriched_results),
    }


def run_server():
    """Start the MCP server."""
    config = _get_config()
    transport = config.mcp_transport
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse")
