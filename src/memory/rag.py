"""Unified RAG search facade — single entry point for all memory search.

Orchestrates FAISS, FTS5, RRF merge, ACT-R reranking, entity deduplication,
GraphRAG expansion, and L2→L1 mention bump.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from src.core.config import Config
from src.core.models import GraphData, GraphRelation, SearchResult
from src.pipeline.indexer import search as faiss_search
from src.pipeline.keyword_index import search_keyword

logger = logging.getLogger(__name__)


@dataclass
class SearchOptions:
    """Options controlling search behavior."""
    top_k: int = 5
    expand_relations: bool = False
    expand_max: int = 10
    include_chunk_text: bool = True
    deduplicate_entities: bool = True
    use_fts5: bool = True
    rerank_actr: bool = True
    bump_mentions: bool = False
    bump_rate_limit: bool = True
    threshold: float = 0.0
    context_for_extraction: bool = False


def search(
    query: str,
    config: Config,
    memory_path: Path,
    options: SearchOptions | None = None,
) -> list[SearchResult]:
    """Single entry point for all memory search."""
    if options is None:
        options = SearchOptions()

    # Step 1: FAISS search
    try:
        faiss_results = faiss_search(query, config, memory_path, top_k=options.top_k * 2)
    except Exception:
        logger.warning("FAISS search failed for query %r", query, exc_info=True)
        return []

    if not faiss_results:
        return []

    # Load graph (needed for multiple steps)
    graph = _load_graph_safe(memory_path)

    results = list(faiss_results)

    # Step 2+3: FTS5 search + RRF merge
    if options.use_fts5:
        results = _try_hybrid_reranking(results, query, config, memory_path, graph)
    elif options.rerank_actr and graph:
        results = _linear_rerank(results, graph, config)

    # Step 5: Entity deduplication
    if options.deduplicate_entities:
        results = _deduplicate_by_entity(results)

    # Step 6: GraphRAG expansion
    if options.expand_relations and graph:
        results = _expand_by_relations(results, graph, options.expand_max)

    # Step 7: Relations enrichment
    if graph:
        _enrich_with_relations(results, graph)

    # Step 8: Mention bump
    if options.bump_mentions and graph:
        _bump_mentions(results, graph, config, memory_path, options.bump_rate_limit, query)

    # Step 9: Final filter
    if options.threshold > 0:
        results = [r for r in results if r.score >= options.threshold]
    results = results[:options.top_k]

    return results


def _load_graph_safe(memory_path: Path) -> GraphData | None:
    """Load graph, returning None on failure."""
    try:
        from src.memory.graph import load_graph
        return load_graph(memory_path)
    except Exception:
        logger.warning("Failed to load graph for search enrichment", exc_info=True)
        return None


def _try_hybrid_reranking(
    faiss_results: list[SearchResult],
    query: str,
    config: Config,
    memory_path: Path,
    graph: GraphData | None,
) -> list[SearchResult]:
    """Try FTS5 keyword search + RRF fusion. Falls back to FAISS-only on failure."""
    fts_db_path = memory_path / config.search.fts_db_path
    if not config.search.hybrid_enabled or not fts_db_path.exists():
        if graph:
            return _linear_rerank(faiss_results, graph, config)
        return faiss_results

    try:
        kw_results = search_keyword(query, fts_db_path, top_k=len(faiss_results) * 2)
    except Exception:
        logger.warning("FTS5 search failed, using FAISS results only", exc_info=True)
        if graph:
            return _linear_rerank(faiss_results, graph, config)
        return faiss_results

    if not kw_results:
        if graph:
            return _linear_rerank(faiss_results, graph, config)
        return faiss_results

    if not graph:
        return _rrf_merge_no_actr(faiss_results, kw_results, config)

    return _rrf_fusion(faiss_results, kw_results, graph, config)


def _rrf_fusion(
    faiss_results: list[SearchResult],
    keyword_results,
    graph: GraphData,
    config: Config,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion combining semantic, keyword, and ACT-R signals."""
    k = config.search.rrf_k
    w_sem = config.search.weight_semantic
    w_kw = config.search.weight_keyword
    w_actr = config.search.weight_actr

    sem_ranks = {r.entity_id: i + 1 for i, r in enumerate(faiss_results)}
    kw_ranks = {r.entity_id: i + 1 for i, r in enumerate(keyword_results)}

    all_ids = set(sem_ranks) | set(kw_ranks)

    actr_scores = {}
    for eid in all_ids:
        e = graph.entities.get(eid)
        actr_scores[eid] = e.score if e else 0.0
    sorted_actr = sorted(actr_scores.items(), key=lambda x: x[1], reverse=True)
    actr_ranks = {eid: i + 1 for i, (eid, _) in enumerate(sorted_actr)}

    default_rank = max(len(faiss_results), len(keyword_results), len(all_ids)) + 10

    result_map = {r.entity_id: r for r in faiss_results}

    scored = []
    for eid in all_ids:
        sr = sem_ranks.get(eid, default_rank)
        kr = kw_ranks.get(eid, default_rank)
        ar = actr_ranks.get(eid, default_rank)

        rrf_score = w_sem / (k + sr) + w_kw / (k + kr) + w_actr / (k + ar)

        if eid in result_map:
            result = result_map[eid]
            result.score = rrf_score
            scored.append(result)
        elif eid in graph.entities:
            e = graph.entities[eid]
            scored.append(SearchResult(
                entity_id=eid, file=e.file,
                chunk="[keyword match]", score=rrf_score,
            ))

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored


def _rrf_merge_no_actr(
    faiss_results: list[SearchResult],
    keyword_results,
    config: Config,
) -> list[SearchResult]:
    """RRF merge with semantic + keyword only (no graph available)."""
    k = config.search.rrf_k
    w_sem = config.search.weight_semantic
    w_kw = config.search.weight_keyword

    sem_ranks = {r.entity_id: i + 1 for i, r in enumerate(faiss_results)}
    kw_ranks = {r.entity_id: i + 1 for i, r in enumerate(keyword_results)}
    all_ids = set(sem_ranks) | set(kw_ranks)
    default_rank = max(len(faiss_results), len(keyword_results), len(all_ids)) + 10

    result_map = {r.entity_id: r for r in faiss_results}
    scored = []
    for eid in all_ids:
        sr = sem_ranks.get(eid, default_rank)
        kr = kw_ranks.get(eid, default_rank)
        rrf_score = w_sem / (k + sr) + w_kw / (k + kr)

        if eid in result_map:
            result = result_map[eid]
            result.score = rrf_score
            scored.append(result)

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored


def _linear_rerank(
    results: list[SearchResult],
    graph: GraphData,
    config: Config,
) -> list[SearchResult]:
    """Linear fallback reranking: faiss_weight * FAISS + actr_weight * ACT-R."""
    w_faiss = config.search.linear_faiss_weight
    w_actr = config.search.linear_actr_weight

    for result in results:
        entity = graph.entities.get(result.entity_id)
        graph_score = entity.score if entity else 0.0
        result.score = result.score * w_faiss + graph_score * w_actr

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def _deduplicate_by_entity(results: list[SearchResult]) -> list[SearchResult]:
    """Keep best-scoring chunk per entity_id."""
    seen: dict[str, SearchResult] = {}
    for result in results:
        if result.entity_id not in seen or result.score > seen[result.entity_id].score:
            seen[result.entity_id] = result
    return sorted(seen.values(), key=lambda r: r.score, reverse=True)


def _expand_by_relations(
    results: list[SearchResult],
    graph: GraphData,
    expand_max: int,
) -> list[SearchResult]:
    """GraphRAG: expand top results with depth-1 neighbors."""
    from src.memory.graph import get_related

    existing_ids = {r.entity_id for r in results}
    expanded = list(results)

    for result in results[:3]:
        try:
            neighbors = get_related(graph, result.entity_id, depth=1)
        except Exception:
            continue

        for neighbor_id in neighbors:
            if neighbor_id in existing_ids:
                continue
            neighbor = graph.entities.get(neighbor_id)
            if not neighbor:
                continue

            eff_strength = 0.5
            for rel in graph.relations:
                if (rel.from_entity == result.entity_id and rel.to_entity == neighbor_id) or \
                   (rel.to_entity == result.entity_id and rel.from_entity == neighbor_id):
                    eff_strength = rel.strength
                    break

            neighbor_score = result.score * eff_strength * 0.5
            expanded.append(SearchResult(
                entity_id=neighbor_id,
                file=neighbor.file,
                chunk=f"[expanded from {result.entity_id}]",
                score=neighbor_score,
            ))
            existing_ids.add(neighbor_id)

    expanded.sort(key=lambda r: r.score, reverse=True)
    return expanded[:expand_max] if len(expanded) > expand_max else expanded


def _enrich_with_relations(results: list[SearchResult], graph: GraphData) -> None:
    """Attach directional relations to each result (mutates in place)."""
    adjacency = defaultdict(list)
    for rel in graph.relations:
        adjacency[rel.from_entity].append(("outgoing", rel))
        adjacency[rel.to_entity].append(("incoming", rel))

    for result in results:
        relations = []
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
        result.relations = relations


def _bump_mentions(
    results: list[SearchResult],
    graph: GraphData,
    config: Config,
    memory_path: Path,
    rate_limit: bool,
    query: str = "",
) -> None:
    """L2→L1 re-emergence: bump mention_dates for retrieved entities."""
    from src.memory.mentions import add_mention
    from src.memory.graph import save_graph
    from src.memory.event_log import append_event

    today = date.today().isoformat()
    promoted = False

    for result in results:
        entity_id = result.entity_id
        if entity_id not in graph.entities:
            continue
        entity = graph.entities[entity_id]

        if rate_limit and today in entity.mention_dates:
            continue

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
            logger.warning("Could not save graph after L2→L1 bump (locked)")

        try:
            append_event(memory_path, "search_performed", "rag", {"query": query[:100]})
        except Exception:
            pass
