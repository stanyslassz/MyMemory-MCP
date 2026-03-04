"""Score calculation for memory entities."""

from __future__ import annotations

import math
from datetime import date, datetime

from src.core.config import Config
from src.core.models import GraphData, GraphEntity


def calculate_score(entity: GraphEntity, config: Config, today: date | None = None) -> float:
    """Calculate entity score using the formula:
    score = (importance × W_imp) + (min(freq/cap, 1.0) × W_freq) + (e^(-days/halflife) × W_rec)
    """
    if today is None:
        today = date.today()

    s = config.scoring

    # Importance component
    imp_component = entity.importance * s.weight_importance

    # Frequency component (capped)
    freq_norm = min(entity.frequency / s.frequency_cap, 1.0) if s.frequency_cap > 0 else 0.0
    freq_component = freq_norm * s.weight_frequency

    # Recency component (exponential decay)
    days_since = 0.0
    if entity.last_mentioned:
        try:
            last = datetime.fromisoformat(entity.last_mentioned).date()
            days_since = (today - last).days
        except (ValueError, TypeError):
            days_since = 365  # fallback: treat as old if unparseable

    recency = math.exp(-days_since / s.recency_halflife_days) if s.recency_halflife_days > 0 else 0.0
    rec_component = recency * s.weight_recency

    return round(imp_component + freq_component + rec_component, 4)


def recalculate_all_scores(graph: GraphData, config: Config, today: date | None = None) -> GraphData:
    """Recalculate scores for all entities in the graph."""
    for entity_id, entity in graph.entities.items():
        entity.score = calculate_score(entity, config, today)
    return graph


def get_top_entities(
    graph: GraphData,
    n: int,
    include_permanent: bool = True,
    min_score: float = 0.0,
) -> list[tuple[str, GraphEntity]]:
    """Get top N entities by score, always including permanent ones."""
    permanent = []
    scored = []

    for entity_id, entity in graph.entities.items():
        if include_permanent and entity.retention == "permanent":
            permanent.append((entity_id, entity))
        elif entity.score >= min_score:
            scored.append((entity_id, entity))

    # Sort by score descending
    scored.sort(key=lambda x: x[1].score, reverse=True)

    # Permanent entities always included, then top N from scored
    perm_ids = {eid for eid, _ in permanent}
    result = list(permanent)
    for item in scored:
        if item[0] not in perm_ids and len(result) < n + len(permanent):
            result.append(item)

    return result
