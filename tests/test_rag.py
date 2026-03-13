"""Tests for the unified RAG search facade."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.config import Config
from src.core.models import GraphData, GraphEntity, GraphRelation, SearchResult
from src.memory.rag import search, SearchOptions


def _make_config(tmp_path: Path) -> Config:
    config = Config()
    config.memory_path = tmp_path
    config.faiss.index_path = str(tmp_path / "_memory.faiss")
    config.faiss.mapping_path = str(tmp_path / "_memory.pkl")
    config.faiss.manifest_path = str(tmp_path / "_faiss_manifest.json")
    return config


def _make_search_results():
    return [
        SearchResult(entity_id="alpha", file="test/alpha.md", chunk="Alpha text", score=0.9),
        SearchResult(entity_id="beta", file="test/beta.md", chunk="Beta text", score=0.7),
        SearchResult(entity_id="gamma", file="test/gamma.md", chunk="Gamma text", score=0.5),
    ]


def _make_graph():
    graph = GraphData()
    graph.entities["alpha"] = GraphEntity(
        file="test/alpha.md", type="person", title="Alpha", score=0.8,
        mention_dates=[], monthly_buckets={},
    )
    graph.entities["beta"] = GraphEntity(
        file="test/beta.md", type="health", title="Beta", score=0.6,
        mention_dates=[], monthly_buckets={},
    )
    graph.entities["gamma"] = GraphEntity(
        file="test/gamma.md", type="interest", title="Gamma", score=0.3,
        mention_dates=[], monthly_buckets={},
    )
    graph.entities["delta"] = GraphEntity(
        file="test/delta.md", type="person", title="Delta", score=0.4,
        mention_dates=[], monthly_buckets={},
    )
    graph.relations.append(GraphRelation(
        from_entity="alpha", to_entity="delta", type="friend_of",
        strength=0.8, last_reinforced=date.today().isoformat(),
    ))
    return graph


@patch("src.memory.rag._load_graph_safe")
@patch("src.memory.rag.faiss_search")
def test_search_basic(mock_faiss, mock_graph, tmp_path):
    """Basic search returns FAISS results."""
    config = _make_config(tmp_path)
    mock_faiss.return_value = _make_search_results()
    mock_graph.return_value = None

    results = search("test query", config, tmp_path, SearchOptions(
        use_fts5=False, rerank_actr=False, bump_mentions=False,
    ))

    assert len(results) == 3
    assert results[0].entity_id == "alpha"
    mock_faiss.assert_called_once()


@patch("src.memory.rag._load_graph_safe")
@patch("src.memory.rag.faiss_search")
def test_search_threshold_filters(mock_faiss, mock_graph, tmp_path):
    """Results below threshold should be filtered out."""
    config = _make_config(tmp_path)
    mock_faiss.return_value = _make_search_results()
    mock_graph.return_value = None

    results = search("test", config, tmp_path, SearchOptions(
        threshold=0.6, use_fts5=False, rerank_actr=False, bump_mentions=False,
    ))

    assert len(results) == 2


@patch("src.memory.rag._load_graph_safe")
@patch("src.memory.rag.faiss_search")
def test_search_top_k_limits(mock_faiss, mock_graph, tmp_path):
    """top_k should limit result count."""
    config = _make_config(tmp_path)
    mock_faiss.return_value = _make_search_results()
    mock_graph.return_value = None

    results = search("test", config, tmp_path, SearchOptions(
        top_k=2, use_fts5=False, rerank_actr=False, bump_mentions=False,
    ))

    assert len(results) == 2


@patch("src.memory.rag._load_graph_safe")
@patch("src.memory.rag.faiss_search")
def test_search_expand_relations(mock_faiss, mock_graph, tmp_path):
    """With expand_relations, neighbors should appear in results."""
    config = _make_config(tmp_path)
    mock_faiss.return_value = [
        SearchResult(entity_id="alpha", file="test/alpha.md", chunk="Alpha text", score=0.9),
    ]
    mock_graph.return_value = _make_graph()

    results = search("test", config, tmp_path, SearchOptions(
        expand_relations=True, use_fts5=False, rerank_actr=False, bump_mentions=False,
    ))

    entity_ids = [r.entity_id for r in results]
    assert "alpha" in entity_ids
    assert "delta" in entity_ids


@patch("src.memory.rag._load_graph_safe")
@patch("src.memory.rag.faiss_search")
def test_search_empty_faiss_returns_empty(mock_faiss, mock_graph, tmp_path):
    config = _make_config(tmp_path)
    mock_faiss.return_value = []
    mock_graph.return_value = None
    results = search("test", config, tmp_path)
    assert results == []


@patch("src.memory.rag._load_graph_safe")
@patch("src.memory.rag.faiss_search")
def test_search_faiss_exception_returns_empty(mock_faiss, mock_graph, tmp_path):
    config = _make_config(tmp_path)
    mock_faiss.side_effect = FileNotFoundError("No index")
    mock_graph.return_value = None
    results = search("test", config, tmp_path)
    assert results == []


@patch("src.memory.rag._load_graph_safe")
@patch("src.memory.rag.faiss_search")
def test_search_dedup_entities(mock_faiss, mock_graph, tmp_path):
    config = _make_config(tmp_path)
    mock_faiss.return_value = [
        SearchResult(entity_id="alpha", file="test/alpha.md", chunk="chunk1", score=0.9),
        SearchResult(entity_id="alpha", file="test/alpha.md", chunk="chunk2", score=0.7),
        SearchResult(entity_id="beta", file="test/beta.md", chunk="chunk1", score=0.6),
    ]
    mock_graph.return_value = None

    results = search("test", config, tmp_path, SearchOptions(
        deduplicate_entities=True, use_fts5=False, rerank_actr=False, bump_mentions=False,
    ))

    assert len(results) == 2
    alpha_result = [r for r in results if r.entity_id == "alpha"][0]
    assert alpha_result.score == 0.9


@patch("src.memory.rag._load_graph_safe")
@patch("src.memory.rag.faiss_search")
def test_search_default_options(mock_faiss, mock_graph, tmp_path):
    config = _make_config(tmp_path)
    mock_faiss.return_value = _make_search_results()
    mock_graph.return_value = _make_graph()
    results = search("test", config, tmp_path)
    assert len(results) >= 1


# ── FTS5 / RRF tests ────────────────────────────────────────────


@patch("src.memory.rag._load_graph_safe")
@patch("src.memory.rag.faiss_search")
def test_search_hybrid_rrf_merge(mock_faiss, mock_graph, tmp_path):
    config = _make_config(tmp_path)
    mock_faiss.return_value = [
        SearchResult(entity_id="alpha", file="test/alpha.md", chunk="Alpha text", score=0.9),
        SearchResult(entity_id="beta", file="test/beta.md", chunk="Beta text", score=0.7),
    ]
    mock_graph.return_value = _make_graph()

    fts_db = tmp_path / config.search.fts_db_path
    fts_db.touch()

    kw_results = [
        SearchResult(entity_id="beta", file="test/beta.md", chunk="Beta kw", score=0.8),
        SearchResult(entity_id="gamma", file="test/gamma.md", chunk="Gamma kw", score=0.6),
    ]

    with patch("src.memory.rag.search_keyword", return_value=kw_results):
        results = search("test", config, tmp_path, SearchOptions(
            use_fts5=True, rerank_actr=True, bump_mentions=False,
        ))

    entity_ids = [r.entity_id for r in results]
    assert "beta" in entity_ids
    assert "alpha" in entity_ids


@patch("src.memory.rag._load_graph_safe")
@patch("src.memory.rag.faiss_search")
def test_search_hybrid_no_graph_rrf_two_signals(mock_faiss, mock_graph, tmp_path):
    config = _make_config(tmp_path)
    mock_faiss.return_value = [
        SearchResult(entity_id="alpha", file="test/alpha.md", chunk="Alpha text", score=0.9),
    ]
    mock_graph.return_value = None

    fts_db = tmp_path / config.search.fts_db_path
    fts_db.touch()

    kw_results = [
        SearchResult(entity_id="alpha", file="test/alpha.md", chunk="Alpha kw", score=0.8),
        SearchResult(entity_id="beta", file="test/beta.md", chunk="Beta kw", score=0.6),
    ]

    with patch("src.memory.rag.search_keyword", return_value=kw_results):
        results = search("test", config, tmp_path, SearchOptions(
            use_fts5=True, rerank_actr=True, bump_mentions=False,
        ))

    assert any(r.entity_id == "alpha" for r in results)


@patch("src.memory.rag._load_graph_safe")
@patch("src.memory.rag.faiss_search")
def test_search_fts5_unavailable_falls_back(mock_faiss, mock_graph, tmp_path):
    config = _make_config(tmp_path)
    mock_faiss.return_value = _make_search_results()
    mock_graph.return_value = _make_graph()

    results = search("test", config, tmp_path, SearchOptions(
        use_fts5=True, rerank_actr=True, bump_mentions=False,
    ))

    assert len(results) >= 1


@patch("src.memory.rag._load_graph_safe")
@patch("src.memory.rag.faiss_search")
def test_search_linear_fallback_reranking(mock_faiss, mock_graph, tmp_path):
    config = _make_config(tmp_path)
    mock_faiss.return_value = [
        SearchResult(entity_id="alpha", file="test/alpha.md", chunk="Alpha text", score=0.3),
        SearchResult(entity_id="gamma", file="test/gamma.md", chunk="Gamma text", score=0.9),
    ]
    graph = _make_graph()
    mock_graph.return_value = graph

    results = search("test", config, tmp_path, SearchOptions(
        use_fts5=False, rerank_actr=True, bump_mentions=False,
    ))

    assert len(results) >= 2
    for r in results:
        if r.entity_id == "alpha":
            assert r.score != 0.3
