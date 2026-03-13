"""Tests for RRF (Reciprocal Rank Fusion) in memory/rag.py."""

from src.core.config import Config
from src.core.models import GraphData, GraphEntity, SearchResult
from src.memory.rag import _rrf_fusion
from src.pipeline.keyword_index import KeywordResult


def _make_graph(entities_dict):
    """Create a minimal GraphData with given entities."""
    entities = {}
    for eid, score in entities_dict.items():
        entities[eid] = GraphEntity(
            file=f"test/{eid}.md",
            type="person",
            title=eid.replace("-", " ").title(),
            score=score,
        )
    return GraphData(entities=entities)


def _make_config(w_sem=0.5, w_kw=0.3, w_actr=0.2, rrf_k=60):
    config = Config()
    config.search.weight_semantic = w_sem
    config.search.weight_keyword = w_kw
    config.search.weight_actr = w_actr
    config.search.rrf_k = rrf_k
    return config


def test_rrf_fusion_combines_signals():
    """Entity present in both FAISS and keyword lists should rank higher."""
    faiss_results = [
        SearchResult(entity_id="alpha", file="a.md", chunk="c", score=0.9),
        SearchResult(entity_id="beta", file="b.md", chunk="c", score=0.7),
        SearchResult(entity_id="gamma", file="g.md", chunk="c", score=0.5),
    ]
    kw_results = [
        KeywordResult(entity_id="beta", chunk_idx=0, bm25_score=5.0),
        KeywordResult(entity_id="delta", chunk_idx=0, bm25_score=3.0),
    ]
    graph = _make_graph({"alpha": 0.8, "beta": 0.6, "gamma": 0.3, "delta": 0.5})

    ranked = _rrf_fusion(faiss_results, kw_results, graph, _make_config())
    ids = [r.entity_id for r in ranked]

    assert ids[0] == "beta"
    assert set(ids) == {"alpha", "beta", "gamma", "delta"}


def test_rrf_fusion_keyword_only_results_included():
    """Entities found only by keyword search should appear in RRF output."""
    faiss_results = [
        SearchResult(entity_id="alpha", file="a.md", chunk="c", score=0.9),
    ]
    kw_results = [
        KeywordResult(entity_id="keyword-only", chunk_idx=0, bm25_score=10.0),
    ]
    graph = _make_graph({"alpha": 0.5, "keyword-only": 0.8})

    ranked = _rrf_fusion(faiss_results, kw_results, graph, _make_config())
    ids = [r.entity_id for r in ranked]
    assert "keyword-only" in ids


def test_rrf_fusion_empty_keyword():
    """With no keyword results, only FAISS entities should appear."""
    faiss_results = [
        SearchResult(entity_id="alpha", file="a.md", chunk="c", score=0.9),
        SearchResult(entity_id="beta", file="b.md", chunk="c", score=0.7),
    ]
    kw_results = []
    graph = _make_graph({"alpha": 0.8, "beta": 0.6})

    ranked = _rrf_fusion(faiss_results, kw_results, graph, _make_config())
    ids = [r.entity_id for r in ranked]
    assert set(ids) == {"alpha", "beta"}


def test_rrf_fusion_weights_affect_ranking():
    """Heavy keyword weight should boost keyword-only entities."""
    faiss_results = [
        SearchResult(entity_id="semantic-hit", file="s.md", chunk="c", score=0.95),
    ]
    kw_results = [
        KeywordResult(entity_id="keyword-hit", chunk_idx=0, bm25_score=10.0),
    ]
    graph = _make_graph({"semantic-hit": 0.3, "keyword-hit": 0.3})

    ranked_kw = _rrf_fusion(
        faiss_results, kw_results, graph, _make_config(w_sem=0.1, w_kw=0.8, w_actr=0.1),
    )
    ranked_sem = _rrf_fusion(
        faiss_results, kw_results, graph, _make_config(w_sem=0.8, w_kw=0.1, w_actr=0.1),
    )

    assert ranked_kw[0].entity_id == "keyword-hit"
    assert ranked_sem[0].entity_id == "semantic-hit"


def test_rrf_fusion_low_actr_entity_still_appears():
    """Entities in graph with zero ACT-R score should still appear via RRF."""
    faiss_results = [
        SearchResult(entity_id="known", file="k.md", chunk="c", score=0.9),
    ]
    kw_results = [
        KeywordResult(entity_id="low-score", chunk_idx=0, bm25_score=5.0),
    ]
    graph = _make_graph({"known": 0.8, "low-score": 0.0})

    ranked = _rrf_fusion(faiss_results, kw_results, graph, _make_config())
    ids = [r.entity_id for r in ranked]
    assert "low-score" in ids
