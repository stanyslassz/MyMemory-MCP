"""MCP Server: 7 tools — get_context, save_chat, search_rag + CRUD tools."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import date as date_cls
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.core.config import Config, load_config
from src.core.models import GraphData
from src.memory.graph import find_entity_by_name, load_graph, remove_relation, save_graph
from src.memory.store import (
    read_entity,
    remove_relation_line,
    save_chat as store_save_chat,
    write_entity,
    parse_observation,
    format_observation,
)
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
    """Retrieve the user's complete memory context — their identity, preferences, relationships, and recent history.

    USE THIS WHEN:
    - Starting a new conversation — call this first to understand who the user is
    - You need background on the user's life, work, health, or relationships
    - You want to personalize your response based on what you know about them

    DO NOT USE WHEN:
    - Looking for a specific fact or detail — use search_rag instead
    - The context was already loaded earlier in this conversation

    Returns:
        Pre-compiled markdown context organized into sections: AI personality,
        identity, work, personal life, top-of-mind topics, vigilances, and history.
        Falls back to a simpler entity index if context hasn't been generated yet.
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
    """Save this conversation to the user's memory for later processing and knowledge extraction.

    USE THIS WHEN:
    - The conversation contains new information worth remembering (personal updates,
      decisions, preferences, health changes, project progress)
    - The user explicitly asks to save or remember something
    - At the end of a meaningful conversation

    DO NOT USE WHEN:
    - The conversation is purely technical/coding with no personal information
    - The conversation only contains small talk with no new facts
    - The chat has already been saved

    Args:
        messages: List of message dicts, each with 'role' (str: 'user' or 'assistant')
                  and 'content' (str). Example: [{"role": "user", "content": "I started a new job at Acme"}]

    Returns:
        Dict with 'status' ('saved' or 'error') and 'file' (relative path to saved chat).
        The chat is saved with processed=false and will be extracted on next 'memory run'.
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


@mcp.tool()
def search_rag(query: str) -> dict:
    """Search the user's memory for specific information using semantic + keyword search.

    USE THIS WHEN:
    - Looking for a specific fact not in the current context (e.g., "what medication does the user take?")
    - The user asks about something from a past conversation
    - You need details about a specific entity, event, or relationship
    - Context from get_context() doesn't contain what you need

    DO NOT USE WHEN:
    - You need general background — use get_context() first
    - The information is already available in the current context

    Good queries: "back pain treatment", "project deadline", "wife's birthday"
    Bad queries: "tell me everything", "what do you know" (too vague — use get_context instead)

    Args:
        query: Natural language search query. Be specific for best results.
               Shorter, focused queries work better than long sentences.

    Returns:
        Dict with 'query', 'total' result count, and 'results' list. Each result has:
        entity_id, file, score, title, type, and related entities (relations list).
        Results are ranked by hybrid score (semantic similarity + keyword match + memory strength).
        Also promotes retrieved entities in memory (L2→L1 re-emergence).
    """
    config = _get_config()
    memory_path = config.memory_path

    from src.memory.rag import search as rag_search, SearchOptions
    results = rag_search(query, config, memory_path, SearchOptions(
        expand_relations=True,
        bump_mentions=True,
    ))

    # Format results for MCP output — load graph once for title/type enrichment
    try:
        from src.memory.graph import load_graph as _load_graph
        graph = _load_graph(memory_path)
    except Exception:
        graph = None

    enriched_results = []
    for result in results:
        entity_data = {"entity_id": result.entity_id, "file": result.file, "score": result.score}
        entity = graph.entities.get(result.entity_id) if graph else None
        entity_data["title"] = entity.title if entity else result.entity_id
        entity_data["type"] = entity.type if entity else "unknown"
        entity_data["relations"] = getattr(result, "relations", [])
        enriched_results.append(entity_data)

    return {
        "query": query,
        "results": enriched_results,
        "total": len(enriched_results),
    }


def _find_fact_line(facts: list[str], content: str) -> int | None:
    """Find a fact line by case-insensitive substring match on content. Returns index or None."""
    content_lower = content.lower()
    for i, line in enumerate(facts):
        obs = parse_observation(line)
        if obs and content_lower in obs["content"].lower():
            return i
    return None


# ── Tool implementations (sync, testable without MCP) ──────


def _delete_fact_impl(entity_name: str, fact_content: str, config: Config) -> str:
    """Implementation for delete_fact. Returns JSON string."""
    memory_path = config.memory_path
    graph = load_graph(memory_path)
    entity_id = find_entity_by_name(entity_name, graph)
    if not entity_id:
        return json.dumps({"status": "error", "message": f"Entity '{entity_name}' not found"})

    entity = graph.entities[entity_id]
    filepath = memory_path / entity.file
    if not filepath.exists():
        return json.dumps({"status": "error", "message": f"Entity file not found: {entity.file}"})

    frontmatter, sections = read_entity(filepath)
    facts = sections.get("Facts", [])

    matched_idx = _find_fact_line(facts, fact_content)
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

    from_id = find_entity_by_name(from_entity, graph)
    if not from_id:
        return json.dumps({"status": "error", "message": f"Source entity '{from_entity}' not found"})

    to_id = find_entity_by_name(to_entity, graph)
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
    entity_id = find_entity_by_name(entity_name, graph)
    if not entity_id:
        return json.dumps({"status": "error", "message": f"Entity '{entity_name}' not found"})

    entity = graph.entities[entity_id]
    filepath = memory_path / entity.file
    if not filepath.exists():
        return json.dumps({"status": "error", "message": f"Entity file not found: {entity.file}"})

    frontmatter, sections = read_entity(filepath)
    facts = sections.get("Facts", [])

    matched_idx = _find_fact_line(facts, old_content)
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
    entity_id = find_entity_by_name(entity_name, graph)
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
    """Permanently delete a specific fact from an entity's memory.

    USE THIS WHEN:
    - The user says a stored fact is wrong and wants it removed entirely
    - A fact is outdated and should be deleted (not just updated)
    - Removing sensitive or incorrect information

    DO NOT USE WHEN:
    - The fact needs correction — use modify_fact to update it instead
    - You want to update the entity itself — use correct_entity

    Args:
        entity_name: Name of the entity (exact title, slug like 'back-pain', or known alias).
        fact_content: Partial content match — e.g., "started physiotherapy" will match
                      a fact containing that phrase. Case-insensitive.

    Returns:
        JSON string with 'status' ('deleted' or 'error'), entity name, and the deleted fact line.
    """
    config = _get_config()
    return _delete_fact_impl(entity_name, fact_content, config)


@mcp.tool()
def delete_relation(from_entity: str, to_entity: str, relation_type: str) -> str:
    """Remove a relationship between two entities.

    USE THIS WHEN:
    - A relationship no longer exists (e.g., user no longer works_at a company)
    - A relationship was incorrectly created
    - The user explicitly asks to remove a connection between two things

    DO NOT USE WHEN:
    - You want to change the relationship type — delete the old one, then the pipeline
      will create the new one from future conversations

    Args:
        from_entity: Source entity name (title, slug, or alias).
        to_entity: Target entity name (title, slug, or alias).
        relation_type: One of: affects, improves, worsens, requires, linked_to, lives_with,
                       works_at, parent_of, friend_of, uses, part_of, contrasts_with, precedes.

    Returns:
        JSON string with 'status' ('deleted' or 'error') and the removed relation details.
    """
    config = _get_config()
    return _delete_relation_impl(from_entity, to_entity, relation_type, config)


@mcp.tool()
def modify_fact(entity_name: str, old_content: str, new_content: str) -> str:
    """Update a fact's content while keeping its metadata (category, date, valence, tags).

    USE THIS WHEN:
    - A fact needs correction (e.g., wrong date, misspelling, outdated detail)
    - The user provides updated information about an existing fact
    - Partial information needs to be completed or refined

    DO NOT USE WHEN:
    - The fact should be removed entirely — use delete_fact
    - You want to add a new fact — save a chat with the new information instead
    - You want to change the entity's metadata — use correct_entity

    Args:
        entity_name: Name of the entity (title, slug, or alias).
        old_content: Partial content to find the fact (case-insensitive substring match).
        new_content: Replacement content. Category, date, valence, and tags are preserved.

    Returns:
        JSON string with 'status' ('modified' or 'error'), old and new fact lines.
    """
    config = _get_config()
    return _modify_fact_impl(entity_name, old_content, new_content, config)


@mcp.tool()
def correct_entity(entity_name: str, field: str, new_value: str) -> str:
    """Correct an entity's metadata — title, type, aliases, or retention level.

    USE THIS WHEN:
    - An entity has the wrong name or type (e.g., 'interest' should be 'health')
    - Adding or updating aliases for better future matching
    - Changing retention (short_term, long_term, permanent) to control how long it's remembered
    - The user says "that's not a person, it's an organization"

    DO NOT USE WHEN:
    - You want to change a fact about the entity — use modify_fact or delete_fact
    - You want to remove a relationship — use delete_relation

    Args:
        entity_name: Current name of the entity (title, slug, or alias).
        field: One of 'title', 'type', 'aliases', 'retention'.
        new_value: New value. For aliases, provide comma-separated list (e.g., "back pain, sciatica, lumbar").
                   For type, use: person, health, work, project, interest, place, animal, organization, ai_self.
                   For retention: short_term, long_term, permanent.

    Returns:
        JSON string with 'status' ('updated' or 'error') and change details.
        If type changes, the entity file is automatically moved to the correct folder.
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
