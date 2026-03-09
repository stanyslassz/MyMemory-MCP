"""Tests for dream step: extract entities from RAG documents."""

import json
from pathlib import Path

from src.pipeline.indexer import list_unextracted_docs


def _make_manifest(manifest_path, docs):
    """Create a FAISS manifest with document entries."""
    manifest = {
        "indexed_files": {},
        "embedding_model": "test/model",
    }
    for source_id, extracted in docs.items():
        manifest["indexed_files"][f"_doc/{source_id}"] = {
            "hash": "abc123",
            "content_hash": "abc123",
            "chunks": 3,
            "source_type": "document",
            "entity_extracted": extracted,
        }
    Path(manifest_path).write_text(json.dumps(manifest))
    return manifest


def test_list_unextracted_documents(tmp_path):
    """Should find documents not yet entity-extracted."""
    manifest_path = tmp_path / "manifest.json"
    _make_manifest(str(manifest_path), {
        "chat_export": False,
        "already_done": True,
        "new_doc": False,
    })

    docs = list_unextracted_docs(str(manifest_path))
    assert len(docs) == 2
    source_ids = [d["source_id"] for d in docs]
    assert "chat_export" in source_ids
    assert "new_doc" in source_ids
    assert "already_done" not in source_ids


def test_list_unextracted_empty_manifest(tmp_path):
    """Should return empty list for empty manifest."""
    manifest_path = tmp_path / "manifest.json"
    (manifest_path).write_text('{"indexed_files":{}}')

    docs = list_unextracted_docs(str(manifest_path))
    assert len(docs) == 0


def test_list_unextracted_no_documents(tmp_path):
    """Should skip non-document entries."""
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "indexed_files": {
            "interests/python.md": {"hash": "x", "chunks": 2},
        },
        "embedding_model": "test/model",
    }
    (manifest_path).write_text(json.dumps(manifest))

    docs = list_unextracted_docs(str(manifest_path))
    assert len(docs) == 0
