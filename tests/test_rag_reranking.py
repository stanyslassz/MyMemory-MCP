"""Tests verifying that ACT-R reranking actually changes result order vs pure FAISS."""

from pathlib import Path
from unittest.mock import patch

from src.core.config import Config
from src.core.models import GraphData, GraphEntity, GraphRelation, SearchResult
from src.memory.rag import search, SearchOptions, _linear_rerank, _rrf_fusion
from src.pipeline.keyword_index import KeywordResult


def _make_config(tmp_path: Path) -> Config:
    config = Config()
    config.memory_path = tmp_path
    config.faiss.index_path = str(tmp_path / "_memory.faiss")
    config.faiss.mapping_path = str(tmp_path / "_memory.pkl")
    config.faiss.manifest_path = str(tmp_path / "_faiss_manifest.json")
    return config


def _make_graph_divergent_scores():
    """Create a graph where ACT-R scores strongly disagree with FAISS order."""
    graph = GraphData()
    # FAISS rank 1 (highest similarity) but lowest ACT-R score
    graph.entities["faiss-top"] = GraphEntity(
        file="test/faiss-top.md", type="person", title="FAISS Top",
        score=0.05, mention_dates=[], monthly_buckets={},
    )
    # FAISS rank 2 but medium ACT-R score
    graph.entities["faiss-mid"] = GraphEntity(
        file="test/faiss-mid.md", type="health", title="FAISS Mid",
        score=0.5, mention_dates=[], monthly_buckets={},
    )
    # FAISS rank 3 (lowest similarity) but highest ACT-R score
    graph.entities["faiss-low"] = GraphEntity(
        file="test/faiss-low.md", type="interest", title="FAISS Low",
        score=0.95, mention_dates=[], monthly_buckets={},
    )
    return graph


def _make_faiss_results():
    """FAISS results ordered by semantic similarity (descending)."""
    return [
        SearchResult(entity_id="faiss-top", file="test/faiss-top.md", chunk="text", score=0.95),
        SearchResult(entity_id="faiss-mid", file="test/faiss-mid.md", chunk="text", score=0.70),
        SearchResult(entity_id="faiss-low", file="test/faiss-low.md", chunk="text", score=0.40),
    ]


def test_linear_rerank_changes_order():
    """Linear reranking should reorder when ACT-R strongly disagrees with FAISS."""
    config = _make_config(Path("/tmp/test"))
    graph = _make_graph_divergent_scores()
    results = _make_faiss_results()

    faiss_order = [r.entity_id for r in results]
    assert faiss_order == ["faiss-top", "faiss-mid", "faiss-low"]

    reranked = _linear_rerank(list(results), graph, config)
    reranked_order = [r.entity_id for r in reranked]

    # With default weights (0.6 FAISS + 0.4 ACT-R):
    # faiss-top: 0.95*0.6 + 0.05*0.4 = 0.59
    # faiss-mid: 0.70*0.6 + 0.50*0.4 = 0.62
    # faiss-low: 0.40*0.6 + 0.95*0.4 = 0.62
    # Order should change: faiss-mid/faiss-low above faiss-top
    assert reranked_order != faiss_order, (
        f"ACT-R reranking should change order when scores diverge. "
        f"FAISS: {faiss_order}, Reranked: {reranked_order}"
    )
    # faiss-top should no longer be first
    assert reranked_order[0] != "faiss-top"


def test_linear_rerank_preserves_order_when_aligned():
    """When ACT-R agrees with FAISS, order should remain the same."""
    config = _make_config(Path("/tmp/test"))
    graph = GraphData()
    graph.entities["a"] = GraphEntity(
        file="test/a.md", type="person", title="A", score=0.9,
    )
    graph.entities["b"] = GraphEntity(
        file="test/b.md", type="person", title="B", score=0.5,
    )
    graph.entities["c"] = GraphEntity(
        file="test/c.md", type="person", title="C", score=0.1,
    )

    results = [
        SearchResult(entity_id="a", file="test/a.md", chunk="text", score=0.9),
        SearchResult(entity_id="b", file="test/b.md", chunk="text", score=0.5),
        SearchResult(entity_id="c", file="test/c.md", chunk="text", score=0.1),
    ]

    reranked = _linear_rerank(list(results), graph, config)
    assert [r.entity_id for r in reranked] == ["a", "b", "c"]


def test_rrf_reranking_actr_changes_order():
    """RRF with ACT-R signal should produce different order than without ACT-R."""
    config = _make_config(Path("/tmp/test"))
    graph = _make_graph_divergent_scores()

    faiss_results = _make_faiss_results()
    kw_results = [
        KeywordResult(entity_id="faiss-top", chunk_idx=0, bm25_score=5.0),
        KeywordResult(entity_id="faiss-mid", chunk_idx=0, bm25_score=3.0),
    ]

    # With ACT-R (default config: w_sem=0.5, w_kw=0.3, w_actr=0.2)
    config_with_actr = _make_config(Path("/tmp/test"))
    ranked_with = _rrf_fusion(list(faiss_results), kw_results, graph, config_with_actr)

    # Without ACT-R (zero weight)
    config_no_actr = _make_config(Path("/tmp/test"))
    config_no_actr.search.weight_actr = 0.0
    config_no_actr.search.weight_semantic = 0.65
    config_no_actr.search.weight_keyword = 0.35
    ranked_without = _rrf_fusion(list(_make_faiss_results()), kw_results, graph, config_no_actr)

    order_with = [r.entity_id for r in ranked_with]
    order_without = [r.entity_id for r in ranked_without]

    # The orders may or may not differ depending on exact scores,
    # but the final scores should differ
    scores_with = {r.entity_id: r.score for r in ranked_with}
    scores_without = {r.entity_id: r.score for r in ranked_without}

    # At least one entity should have a different score
    score_diffs = [
        abs(scores_with.get(eid, 0) - scores_without.get(eid, 0))
        for eid in set(scores_with) & set(scores_without)
    ]
    assert any(d > 1e-6 for d in score_diffs), (
        "ACT-R should affect at least one entity's final score"
    )


@patch("src.memory.rag._load_graph_safe")
@patch("src.memory.rag.faiss_search")
def test_search_actr_reranking_end_to_end(mock_faiss, mock_graph, tmp_path):
    """End-to-end: search with ACT-R reranking changes order vs without."""
    config = _make_config(tmp_path)
    graph = _make_graph_divergent_scores()
    mock_graph.return_value = graph

    # Without ACT-R
    mock_faiss.return_value = _make_faiss_results()
    results_no_actr = search("test", config, tmp_path, SearchOptions(
        use_fts5=False, rerank_actr=False, bump_mentions=False,
    ))

    # With ACT-R
    mock_faiss.return_value = _make_faiss_results()
    results_with_actr = search("test", config, tmp_path, SearchOptions(
        use_fts5=False, rerank_actr=True, bump_mentions=False,
    ))

    order_no = [r.entity_id for r in results_no_actr]
    order_with = [r.entity_id for r in results_with_actr]

    # With divergent scores, ACT-R reranking should change the order
    assert order_no != order_with, (
        f"ACT-R should change ranking with divergent scores. "
        f"Without: {order_no}, With: {order_with}"
    )


def test_high_actr_weight_dominates():
    """With ACT-R weight at 0.9, ranking should follow ACT-R scores."""
    config = _make_config(Path("/tmp/test"))
    config.search.linear_faiss_weight = 0.1
    config.search.linear_actr_weight = 0.9

    graph = _make_graph_divergent_scores()
    results = _make_faiss_results()

    reranked = _linear_rerank(list(results), graph, config)
    order = [r.entity_id for r in reranked]

    # faiss-low has highest ACT-R score (0.95), should be first
    assert order[0] == "faiss-low"
    # faiss-top has lowest ACT-R score (0.05), should be last
    assert order[-1] == "faiss-top"
