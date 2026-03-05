"""ACT-R based scoring with spreading activation for memory entities."""

from __future__ import annotations

import math
from datetime import date, datetime
from collections import defaultdict

from src.core.config import Config
from src.core.models import GraphData, GraphEntity


def _sigmoid(x: float) -> float:
    """Standard sigmoid: 1 / (1 + e^(-x)) -> maps to (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))


def calculate_actr_base(
    mention_dates: list[str],
    monthly_buckets: dict[str, int],
    decay_factor: float,
    today: date,
) -> float:
    """ACT-R base-level activation: B = ln(sum(t_j^(-d))).

    t_j = days since each mention (minimum 0.5 to avoid div-by-zero).
    For monthly_buckets: convert each bucket to estimated dates (mid-month,
    spread uniformly).
    If no mentions at all, return -5.0 (very low activation).
    """
    summation = 0.0

    # Direct mention dates
    for ds in mention_dates:
        try:
            d = datetime.fromisoformat(ds).date() if "T" in ds else date.fromisoformat(ds)
            days = max((today - d).days, 0) + 0.5  # minimum 0.5
            summation += days ** (-decay_factor)
        except (ValueError, TypeError):
            continue

    # Monthly buckets: spread mentions uniformly across the month
    for bucket_key, count in monthly_buckets.items():
        try:
            # Parse "YYYY-MM" format
            parts = bucket_key.split("-")
            year, month = int(parts[0]), int(parts[1])
            # Use mid-month (15th) as representative date
            mid = date(year, month, 15)
            days = max((today - mid).days, 0) + 0.5
            summation += count * (days ** (-decay_factor))
        except (ValueError, TypeError, IndexError):
            continue

    if summation <= 0:
        return -5.0

    return math.log(summation)


def calculate_score(
    entity: GraphEntity,
    config: Config,
    today: date | None = None,
    spreading_bonus: float = 0.0,
) -> float:
    """Final score = sigmoid(B + beta + spreading_weight * S).

    B = ACT-R base from calculate_actr_base
    beta = entity.importance * config.scoring.importance_weight
    S = spreading_bonus (passed externally)
    Uses decay_factor_short_term for short_term retention entities.
    Enforces permanent_min_score for permanent retention entities.
    Returns round(score, 4).
    """
    if today is None:
        today = date.today()

    s = config.scoring

    # Pick decay factor based on retention
    decay = s.decay_factor_short_term if entity.retention == "short_term" else s.decay_factor

    # ACT-R base-level activation
    B = calculate_actr_base(entity.mention_dates, entity.monthly_buckets, decay, today)

    # Importance boost
    beta = entity.importance * s.importance_weight

    # Combined activation
    activation = B + beta + s.spreading_weight * spreading_bonus

    score = _sigmoid(activation)

    # Enforce permanent minimum
    if entity.retention == "permanent":
        score = max(score, s.permanent_min_score)

    return round(score, 4)


def spreading_activation(
    graph: GraphData,
    config: Config,
    today: date | None = None,
) -> dict[str, float]:
    """Compute spreading activation bonus for all entities.

    First pass: compute base ACT-R scores (sigmoid of B + beta) for all entities.
    Compute effective relation strengths with time decay.
    Build bidirectional adjacency list.
    Second pass: S_i = sum(w_ij * A_j) where w_ij = effective_strength / total_outgoing.
    Returns dict of entity_id -> spreading_bonus.
    """
    if today is None:
        today = date.today()

    s = config.scoring

    # First pass: base scores for all entities
    base_scores: dict[str, float] = {}
    for eid, entity in graph.entities.items():
        decay = s.decay_factor_short_term if entity.retention == "short_term" else s.decay_factor
        B = calculate_actr_base(entity.mention_dates, entity.monthly_buckets, decay, today)
        beta = entity.importance * s.importance_weight
        base_scores[eid] = _sigmoid(B + beta)

    # Build bidirectional adjacency with effective strengths
    # adjacency[target] = list of (source, effective_strength)
    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for rel in graph.relations:
        # Compute time-decayed strength
        days_since = 0.0
        if rel.last_reinforced:
            try:
                d = date.fromisoformat(rel.last_reinforced)
                days_since = max((today - d).days, 0)
            except (ValueError, TypeError):
                days_since = 365.0

        effective_strength = rel.strength * math.exp(
            -days_since / s.relation_decay_halflife
        )

        # Bidirectional
        adjacency[rel.to_entity].append((rel.from_entity, effective_strength))
        adjacency[rel.from_entity].append((rel.to_entity, effective_strength))

    # Second pass: compute spreading bonus
    spreading: dict[str, float] = {}
    for eid in graph.entities:
        if eid not in adjacency:
            spreading[eid] = 0.0
            continue

        neighbors = adjacency[eid]
        total_strength = sum(eff for _, eff in neighbors)
        if total_strength <= 0:
            spreading[eid] = 0.0
            continue

        bonus = 0.0
        for neighbor_id, eff in neighbors:
            if neighbor_id in base_scores:
                bonus += eff * base_scores[neighbor_id]

        spreading[eid] = bonus

    return spreading


def recalculate_all_scores(
    graph: GraphData,
    config: Config,
    today: date | None = None,
) -> GraphData:
    """Recalculate scores for all entities using ACT-R + spreading activation."""
    if today is None:
        today = date.today()

    bonuses = spreading_activation(graph, config, today)

    for entity_id, entity in graph.entities.items():
        bonus = bonuses.get(entity_id, 0.0)
        entity.score = calculate_score(entity, config, today, spreading_bonus=bonus)

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
