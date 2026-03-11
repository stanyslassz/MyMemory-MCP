"""Tests for pipeline/doc_ingest.py — document normalize → chunk → embed → index."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.core.config import Config, EmbeddingsConfig, FAISSConfig, FeaturesConfig, IngestConfig, ScoringConfig
from src.core.models import IngestKey, CHUNK_POLICY_VERSION
from src.memory.store import init_memory_structure
from src.pipeline.doc_ingest import _normalize_text, ingest_document


def _make_config(tmp_path) -> Config:
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
    config.features = FeaturesConfig(doc_pipeline=True)
    config.ingest = IngestConfig(
        recovery_threshold_seconds=300,
        max_retries=3,
        jobs_path=str(tmp_path / "_ingest_jobs.json"),
    )
    config.scoring = ScoringConfig()
    return config


def _mock_embed(texts):
    np.random.seed(42)
    vecs = np.random.randn(len(texts), 384).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


class TestNormalizeText:
    def test_strips_frontmatter(self):
        text = "---\ntitle: Test\ntype: sante\n---\n\n# Content\nHello"
        result = _normalize_text(text)
        assert not result.startswith("---")
        assert "# Content" in result

    def test_normalizes_whitespace(self):
        text = "Line 1\n\n\n\n\nLine 2"
        result = _normalize_text(text)
        assert "\n\n\n" not in result

    def test_strips_trailing_whitespace(self):
        text = "hello   \nworld   "
        result = _normalize_text(text)
        lines = result.split("\n")
        for line in lines:
            assert line == line.rstrip()

    def test_empty_text(self):
        assert _normalize_text("") == ""
        assert _normalize_text("   ") == ""


class TestDocumentIngest:
    def test_ingest_creates_index(self, tmp_path):
        config = _make_config(tmp_path)
        init_memory_structure(tmp_path)

        key = IngestKey(source_id="test.md", content_hash="abc123", chunk_policy_version=CHUNK_POLICY_VERSION)
        content = "# Test Document\n\nThis is a test document with enough content to index."

        with patch("src.pipeline.doc_ingest.get_embedding_fn", return_value=_mock_embed):
            result = ingest_document("test.md", content, key, tmp_path, config)

        assert result["chunks_indexed"] >= 1
        assert Path(config.faiss.index_path).exists()

    def test_ingest_idempotent_same_hash(self, tmp_path):
        config = _make_config(tmp_path)
        init_memory_structure(tmp_path)

        content = "# Test\n\nSame content"
        key = IngestKey(source_id="test.md", content_hash="same_hash", chunk_policy_version=CHUNK_POLICY_VERSION)

        with patch("src.pipeline.doc_ingest.get_embedding_fn", return_value=_mock_embed):
            r1 = ingest_document("test.md", content, key, tmp_path, config)
            assert r1["chunks_indexed"] >= 1

            # Second ingest with same hash should skip
            r2 = ingest_document("test.md", content, key, tmp_path, config)
            assert r2.get("skipped", False) is True

    def test_ingest_empty_content(self, tmp_path):
        config = _make_config(tmp_path)
        init_memory_structure(tmp_path)

        key = IngestKey(source_id="empty.md", content_hash="empty", chunk_policy_version=CHUNK_POLICY_VERSION)

        with patch("src.pipeline.doc_ingest.get_embedding_fn", return_value=_mock_embed):
            result = ingest_document("empty.md", "", key, tmp_path, config)

        assert result["chunks_indexed"] == 0

    def test_ingest_updates_manifest(self, tmp_path):
        import json
        config = _make_config(tmp_path)
        init_memory_structure(tmp_path)

        key = IngestKey(source_id="doc.md", content_hash="hash123", chunk_policy_version=CHUNK_POLICY_VERSION)
        content = "# Document\n\nContent for manifest test."

        with patch("src.pipeline.doc_ingest.get_embedding_fn", return_value=_mock_embed):
            ingest_document("doc.md", content, key, tmp_path, config)

        manifest = json.loads(Path(config.faiss.manifest_path).read_text())
        assert "_doc/doc.md" in manifest["indexed_files"]
        assert manifest["indexed_files"]["_doc/doc.md"]["source_type"] == "document"
        assert manifest["indexed_files"]["_doc/doc.md"]["content_hash"] == "hash123"
