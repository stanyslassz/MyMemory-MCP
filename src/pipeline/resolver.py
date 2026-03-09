"""Step 2: Resolve extracted entity names to existing entities (deterministic, 0 LLM tokens)."""

from __future__ import annotations

from typing import Callable, Optional

from src.core.models import (
    GraphData,
    RawExtraction,
    Resolution,
    ResolvedEntity,
    ResolvedExtraction,
)
from src.core.utils import slugify


def resolve_entity(
    name: str,
    graph: GraphData,
    faiss_search_fn: Optional[Callable] = None,
    observation_context: str = "",
) -> Resolution:
    """Resolve a free-form entity name against the graph.

    Resolution order:
    1. Exact match by slug ID
    2. Alias containment check
    3. FAISS similarity search (if available, context-enriched query)
    4. New entity

    observation_context: optional snippet from observations to disambiguate
    homonyms (e.g., "Apple" as fruit vs company).
    """
    slug = slugify(name)

    # 1. Exact match by ID
    if slug in graph.entities:
        return Resolution(status="resolved", entity_id=slug)

    # 2. Match by alias (containment check)
    name_lower = name.lower()
    for entity_id, meta in graph.entities.items():
        for alias in meta.aliases:
            alias_lower = alias.lower()
            if alias_lower in name_lower or name_lower in alias_lower:
                return Resolution(status="resolved", entity_id=entity_id)
        # Also check title
        if meta.title.lower() in name_lower or name_lower in meta.title.lower():
            return Resolution(status="resolved", entity_id=entity_id)

    # 3. FAISS similarity search (if available)
    # Context-aware: enrich query with observation context for better disambiguation
    if faiss_search_fn is not None:
        try:
            query = f"{name} {observation_context}".strip() if observation_context else name
            similar = faiss_search_fn(query, top_k=3, threshold=0.75)
            if similar:
                candidates = [s["entity_id"] for s in similar if "entity_id" in s]
                if candidates:
                    return Resolution(status="ambiguous", candidates=candidates)
        except Exception:
            pass  # FAISS not available, skip

    # 4. New entity
    return Resolution(status="new", suggested_slug=slug)


def resolve_all(
    raw_extraction: RawExtraction,
    graph: GraphData,
    faiss_search_fn: Optional[Callable] = None,
) -> ResolvedExtraction:
    """Resolve all entities from a raw extraction."""
    resolved_entities = []

    for entity in raw_extraction.entities:
        # Build observation context for FAISS disambiguation
        obs_context = ""
        if entity.observations:
            obs = entity.observations[0]
            obs_context = f"{obs.category} {obs.content[:50]}"
        resolution = resolve_entity(entity.name, graph, faiss_search_fn, obs_context)
        resolved_entities.append(ResolvedEntity(raw=entity, resolution=resolution))

    return ResolvedExtraction(
        resolved=resolved_entities,
        relations=raw_extraction.relations,
        summary=raw_extraction.summary,
    )
