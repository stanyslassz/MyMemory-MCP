"""Document ingest pipeline: normalize → chunk → embed → index (immediate retrieval)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.core.config import Config
from src.core.metrics import metrics
from src.core.models import IngestKey
from src.pipeline.indexer import _chunk_text, _get_embedding_fn, _load_manifest, _save_manifest

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    """Normalize document text for chunking.

    - Strip YAML frontmatter
    - Normalize excessive whitespace
    - Strip trailing whitespace per line
    """
    # Strip YAML frontmatter if present
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3:].lstrip("\n")

    # Normalize multiple blank lines to max 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def ingest_document(
    source_id: str,
    content: str,
    ingest_key: IngestKey,
    memory_path: Path,
    config: Config,
) -> dict:
    """Ingest a document: normalize → chunk → embed → add to FAISS index.

    Returns dict with chunks_indexed count and manifest info.
    """
    import faiss
    import numpy as np
    import pickle

    # Normalize
    normalized = _normalize_text(content)
    if not normalized:
        return {"chunks_indexed": 0, "source_id": source_id}

    # Chunk
    chunks = _chunk_text(normalized, config.embeddings.chunk_size, config.embeddings.chunk_overlap)
    if not chunks:
        return {"chunks_indexed": 0, "source_id": source_id}

    # Embed
    embed_fn = _get_embedding_fn(config)
    embeddings = embed_fn(chunks)

    # Load existing index or create new
    index_path = Path(config.faiss.index_path)
    mapping_path = Path(config.faiss.mapping_path)
    manifest = _load_manifest(config.faiss.manifest_path)

    if index_path.exists() and mapping_path.exists():
        index = faiss.read_index(str(index_path))
        with open(mapping_path, "rb") as f:
            chunk_mapping = pickle.load(f)
    else:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        chunk_mapping = []

    # Upsert guard: remove old vectors for same source if re-ingesting
    doc_key = f"_doc/{source_id}"
    existing_entry = manifest.get("indexed_files", {}).get(doc_key)
    if existing_entry and existing_entry.get("content_hash") == ingest_key.content_hash:
        # Already indexed with same content — skip
        logger.info("Document %s already indexed with same hash, skipping", source_id)
        return {"chunks_indexed": 0, "source_id": source_id, "skipped": True}

    # Add new vectors
    start_idx = index.ntotal
    index.add(embeddings.astype(np.float32))

    for i, chunk in enumerate(chunks):
        chunk_mapping.append({
            "file": doc_key,
            "entity_id": f"doc:{source_id}",
            "chunk_idx": i,
            "chunk_text": chunk[:200],  # store preview for search results
        })

    # Update manifest
    if "indexed_files" not in manifest:
        manifest["indexed_files"] = {}
    manifest["indexed_files"][doc_key] = {
        "hash": ingest_key.content_hash,
        "content_hash": ingest_key.content_hash,
        "chunks": len(chunks),
        "ids": list(range(start_idx, start_idx + len(chunks))),
        "source_type": "document",
        "chunk_policy_version": ingest_key.chunk_policy_version,
    }
    manifest["embedding_model"] = f"{config.embeddings.provider}/{config.embeddings.model}"

    # Save
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(mapping_path, "wb") as f:
        pickle.dump(chunk_mapping, f)
    _save_manifest(config.faiss.manifest_path, manifest)

    metrics.record_ingest_success(source_id, len(chunks))
    logger.info("Indexed document %s: %d chunks", source_id, len(chunks))

    return {"chunks_indexed": len(chunks), "source_id": source_id}
