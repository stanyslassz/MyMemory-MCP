"""Step 3: LLM arbitrates ambiguous entity resolutions (micro-prompt, ~50 tokens)."""

from __future__ import annotations

from src.core.config import Config
from src.core.llm import call_arbitration
from src.core.models import EntityResolution, GraphData


def arbitrate_entity(
    name: str,
    context: str,
    candidates: list[str],
    graph: GraphData,
    config: Config,
) -> EntityResolution:
    """Arbitrate a single ambiguous entity by asking the LLM."""
    candidates_data = []
    for cid in candidates:
        if cid in graph.entities:
            entity = graph.entities[cid]
            candidates_data.append({
                "id": cid,
                "title": entity.title,
                "type": entity.type,
                "aliases": entity.aliases,
            })

    if not candidates_data:
        return EntityResolution(action="new", new_type=None)

    return call_arbitration(name, context, candidates_data, config)


def arbitrate_all(
    ambiguous_list: list[dict],
    graph: GraphData,
    config: Config,
) -> list[EntityResolution]:
    """Arbitrate all ambiguous entities. Each dict has 'name', 'context', 'candidates'."""
    results = []
    for item in ambiguous_list:
        result = arbitrate_entity(
            item["name"],
            item.get("context", ""),
            item["candidates"],
            graph,
            config,
        )
        results.append(result)
    return results
