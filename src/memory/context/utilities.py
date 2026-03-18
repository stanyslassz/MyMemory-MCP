"""Context utilities: stopwords, similarity, dedup, sorting, grouping helpers."""

from __future__ import annotations

from src.memory.store import parse_observation


_STOPWORDS = frozenset(
    # French
    "le la les un une des de du d l et en pour aux avec sans par sur dans "
    "qui que qu est sont a au ce cette ces sa son ses se ne pas ni si "
    "on nous vous il elle ils elles "
    # English
    "the a an is are was were be been being have has had do does did "
    "will would shall should can could may might must "
    "i me my we our you your he him his she her it its they them their "
    "and or but not no nor so if then than that this these those "
    "in on at to for of by from with as into about up out".split()
)


def _trigrams(text: str) -> set[str]:
    """Return the set of character trigrams from text."""
    return {text[i:i + 3] for i in range(len(text) - 2)}


def _content_similarity(text_a: str, text_b: str) -> float:
    """Blended similarity: 50% stopword-filtered word Jaccard + 50% trigram Jaccard."""
    words_a = {w for w in text_a.lower().split() if w not in _STOPWORDS and len(w) > 1}
    words_b = {w for w in text_b.lower().split() if w not in _STOPWORDS and len(w) > 1}
    if not words_a or not words_b:
        return 0.0
    word_jaccard = len(words_a & words_b) / len(words_a | words_b)
    tri_a = _trigrams(" ".join(sorted(words_a)))
    tri_b = _trigrams(" ".join(sorted(words_b)))
    tri_union = tri_a | tri_b
    tri_jaccard = len(tri_a & tri_b) / len(tri_union) if tri_union else 0.0
    return 0.5 * word_jaccard + 0.5 * tri_jaccard


def _deduplicate_facts_for_context(
    facts: list[str], threshold: float = 0.35, max_per_category: int = 5,
) -> list[str]:
    """Drop near-duplicate facts within same category using blended similarity.

    Uses stopword-filtered word Jaccard + character trigram overlap.
    After dedup, caps each category to max_per_category (first occurrence wins).
    Preserves input order — only removes later duplicates.
    """
    kept_by_cat: dict[str, list[dict]] = {}
    cat_counts: dict[str, int] = {}
    result = []
    for line in facts:
        obs = parse_observation(line)
        if not obs:
            result.append(line)
            continue
        cat = obs["category"]
        content = obs["content"]
        # Check duplicate against kept facts in same category
        is_dup = False
        for kept_obs in kept_by_cat.get(cat, []):
            if _content_similarity(content, kept_obs["content"]) > threshold:
                is_dup = True
                break
        if is_dup:
            continue
        # Check category cap
        count = cat_counts.get(cat, 0)
        if count >= max_per_category:
            continue
        result.append(line)
        kept_by_cat.setdefault(cat, []).append(obs)
        cat_counts[cat] = count + 1
    return result


def _sort_facts_by_date(facts: list[str]) -> list[str]:
    """Sort facts by date (chronological). Undated facts go last, preserving order."""
    dated: list[tuple[str, str]] = []
    undated: list[str] = []
    for fact in facts:
        parsed = parse_observation(fact)
        if parsed and parsed["date"]:
            dated.append((parsed["date"], fact))
        else:
            undated.append(fact)
    dated.sort(key=lambda x: x[0])
    return [f for _, f in dated] + undated


def _group_facts_by_category(facts: list[str]) -> dict[str, list[str]]:
    """Group fact lines by their [category] prefix, stripping the prefix from content.

    Returns an ordered dict of category -> list of content strings (without category prefix).
    Non-parseable lines go under '_other'.
    """
    from collections import OrderedDict
    grouped: dict[str, list[str]] = OrderedDict()
    for line in facts:
        obs = parse_observation(line)
        if obs:
            cat = obs["category"]
            # Rebuild display content: (date) content [valence] #tags
            parts = []
            if obs.get("date"):
                parts.append(f"({obs['date']})")
            parts.append(obs["content"])
            if obs.get("valence"):
                markers = {"positive": "[+]", "negative": "[-]", "neutral": "[~]"}
                if obs["valence"] in markers:
                    parts.append(markers[obs["valence"]])
            for tag in obs.get("tags", []):
                parts.append(f"#{tag}")
            grouped.setdefault(cat, []).append(" ".join(parts))
        else:
            grouped.setdefault("_other", []).append(line.lstrip("- "))
    return grouped


def _sort_by_cluster(
    entities: list[tuple[str, "GraphEntity"]],
    graph: "GraphData",
) -> list[tuple[str, "GraphEntity"]]:
    """Sort entities so that members of the same connected component are adjacent.

    Within each cluster, preserves the original order (score-descending).
    """
    if len(entities) <= 1:
        return entities

    entity_ids = {eid for eid, _ in entities}

    # Build adjacency restricted to these entities
    adj: dict[str, set[str]] = {eid: set() for eid in entity_ids}
    for rel in graph.relations:
        if rel.from_entity in entity_ids and rel.to_entity in entity_ids:
            adj[rel.from_entity].add(rel.to_entity)
            adj[rel.to_entity].add(rel.from_entity)

    # BFS connected components
    visited: set[str] = set()
    cluster_map: dict[str, int] = {}
    cluster_id = 0
    for eid in [e for e, _ in entities]:  # iterate in score order
        if eid in visited:
            continue
        queue = [eid]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            cluster_map[node] = cluster_id
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        cluster_id += 1

    # Sort: primary by first-appearance order of cluster, secondary by original order
    cluster_first_idx: dict[int, int] = {}
    for i, (eid, _) in enumerate(entities):
        cid = cluster_map.get(eid, 0)
        if cid not in cluster_first_idx:
            cluster_first_idx[cid] = i

    original_pos = {eid: i for i, (eid, _) in enumerate(entities)}
    return sorted(entities, key=lambda x: (cluster_first_idx.get(cluster_map.get(x[0], 0), 0), original_pos.get(x[0], 0)))


