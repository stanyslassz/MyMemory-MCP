"""Tests for pipeline/indexer.py — uses mock embeddings to avoid model downloads."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.core.config import Config, EmbeddingsConfig, FAISSConfig, ScoringConfig
from src.core.models import EntityFrontmatter
from src.memory.store import init_memory_structure, write_entity
from src.pipeline.indexer import _chunk_text, _file_hash, build_index, search


def _make_config(tmp_path):
    """Create a test config with FAISS paths pointing to tmp_path."""
    config = Config.__new__(Config)
    config.memory_path = tmp_path
    config.embeddings = EmbeddingsConfig(
        provider="sentence-transformers",
        model="all-MiniLM-L6-v2",
        chunk_size=400,
        chunk_overlap=80,
    )
    config.faiss = FAISSConfig(
        index_path=str(tmp_path / "_memory.faiss"),
        mapping_path=str(tmp_path / "_memory.pkl"),
        manifest_path=str(tmp_path / "_faiss_manifest.json"),
        top_k=5,
    )
    config.scoring = ScoringConfig()
    return config


def _mock_embed(texts):
    """Mock embedding function: returns random 384-dim vectors."""
    np.random.seed(42)
    vecs = np.random.randn(len(texts), 384).astype(np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def test_chunk_text():
    text = " ".join(["word"] * 1000)
    chunks = _chunk_text(text, chunk_size=400, overlap=80)
    assert len(chunks) > 1
    # Each chunk should have content
    for chunk in chunks:
        assert len(chunk.strip()) > 0


def test_chunk_text_short():
    text = "Short text here"
    chunks = _chunk_text(text, chunk_size=400, overlap=80)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_empty():
    chunks = _chunk_text("", chunk_size=400, overlap=80)
    assert len(chunks) == 0


def test_file_hash(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    h1 = _file_hash(f)
    assert len(h1) == 64  # SHA256 hex

    f.write_text("hello world modified")
    h2 = _file_hash(f)
    assert h1 != h2


def test_build_index(tmp_path):
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    # Create some entity files
    fm = EntityFrontmatter(
        title="Test Entity", type="sante",
        created="2026-01-01", last_mentioned="2026-03-03",
    )
    write_entity(
        tmp_path / "moi" / "test-entity.md", fm,
        {"Faits": ["- [fait] Test fact one", "- [diagnostic] Test diagnosis"],
         "Relations": [], "Historique": []},
    )

    with patch("src.pipeline.indexer._get_embedding_fn", return_value=_mock_embed):
        manifest = build_index(tmp_path, config)

    assert "moi/test-entity.md" in manifest["indexed_files"]
    assert Path(config.faiss.index_path).exists()
    assert Path(config.faiss.mapping_path).exists()


def test_search(tmp_path):
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    fm = EntityFrontmatter(
        title="Natation", type="interet",
        created="2026-01-01", last_mentioned="2026-03-03",
    )
    write_entity(
        tmp_path / "interets" / "natation.md", fm,
        {"Faits": ["- [fait] Nage trois fois par semaine"],
         "Relations": [], "Historique": []},
    )

    with patch("src.pipeline.indexer._get_embedding_fn", return_value=_mock_embed):
        build_index(tmp_path, config)
        results = search("natation", config, tmp_path, top_k=3)

    assert len(results) >= 1
    assert results[0].entity_id == "natation"


def test_build_empty_index(tmp_path):
    config = _make_config(tmp_path)
    init_memory_structure(tmp_path)

    with patch("src.pipeline.indexer._get_embedding_fn", return_value=_mock_embed):
        manifest = build_index(tmp_path, config)

    assert len(manifest["indexed_files"]) == 0
