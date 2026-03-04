"""FAISS indexing: incremental vector index for semantic search."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.core.config import Config
from src.core.models import SearchResult

logger = logging.getLogger(__name__)

# Lazy-loaded globals
_embedding_model = None
_embedding_model_name = None


def _get_embedding_fn(config: Config):
    """Get an embedding function based on config."""
    global _embedding_model, _embedding_model_name

    provider = config.embeddings.provider
    model_name = config.embeddings.model

    if provider == "sentence-transformers":
        if _embedding_model is None or _embedding_model_name != model_name:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(model_name)
            _embedding_model_name = model_name

        def embed(texts: list[str]) -> np.ndarray:
            return _embedding_model.encode(texts, normalize_embeddings=True)

        return embed

    elif provider == "ollama":
        import litellm

        def embed(texts: list[str]) -> np.ndarray:
            response = litellm.embedding(
                model=f"ollama/{model_name}",
                input=texts,
                api_base=config.embeddings.api_base or "http://localhost:11434",
            )
            return np.array([d["embedding"] for d in response.data], dtype=np.float32)

        return embed

    elif provider == "openai":
        import litellm

        def embed(texts: list[str]) -> np.ndarray:
            response = litellm.embedding(model=f"openai/{model_name}", input=texts)
            return np.array([d["embedding"] for d in response.data], dtype=np.float32)

        return embed

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Split text into overlapping chunks by approximate token count (words/1.3)."""
    words = text.split()
    approx_tokens_per_word = 1.3
    words_per_chunk = int(chunk_size / approx_tokens_per_word)
    words_overlap = int(overlap / approx_tokens_per_word)

    if len(words) <= words_per_chunk:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(words):
        end = start + words_per_chunk
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - words_overlap
        if start >= len(words):
            break

    return chunks if chunks else ([text] if text.strip() else [])


def _file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


def _load_manifest(manifest_path: str) -> dict:
    """Load FAISS manifest from JSON."""
    path = Path(manifest_path)
    if not path.exists():
        return {"embedding_model": "", "last_build": "", "indexed_files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(manifest_path: str, manifest: dict) -> None:
    """Save FAISS manifest to JSON."""
    Path(manifest_path).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _get_entity_files(memory_path: Path) -> list[Path]:
    """Get all entity MD files (exclude _* files and chats/)."""
    files = []
    for md_file in sorted(memory_path.rglob("*.md")):
        rel = md_file.relative_to(memory_path)
        parts = rel.parts
        if any(p.startswith("_") for p in parts) or (parts and parts[0] == "chats"):
            continue
        files.append(md_file)
    return files


def build_index(memory_path: Path, config: Config) -> dict:
    """Build FAISS index from scratch. Returns manifest."""
    import faiss

    embed_fn = _get_embedding_fn(config)
    files = _get_entity_files(memory_path)

    all_chunks: list[str] = []
    chunk_mapping: list[dict] = []  # [{file, entity_id, chunk_idx}, ...]
    manifest = {
        "embedding_model": f"{config.embeddings.provider}/{config.embeddings.model}",
        "last_build": datetime.now().isoformat(),
        "indexed_files": {},
    }

    for md_file in files:
        text = md_file.read_text(encoding="utf-8")
        file_hash = _file_hash(md_file)
        rel_path = str(md_file.relative_to(memory_path))
        entity_id = md_file.stem

        chunks = _chunk_text(text, config.embeddings.chunk_size, config.embeddings.chunk_overlap)

        start_idx = len(all_chunks)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_mapping.append({
                "file": rel_path,
                "entity_id": entity_id,
                "chunk_idx": i,
            })

        manifest["indexed_files"][rel_path] = {
            "hash": file_hash,
            "chunks": len(chunks),
            "ids": list(range(start_idx, start_idx + len(chunks))),
        }

    if not all_chunks:
        # Create empty index
        dim = 384  # default dimension for all-MiniLM-L6-v2
        index = faiss.IndexFlatIP(dim)
        _save_index(config, index, chunk_mapping, manifest)
        return manifest

    # Embed all chunks
    embeddings = embed_fn(all_chunks)
    dim = embeddings.shape[1]

    # Build FAISS index (Inner Product for cosine similarity on normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    _save_index(config, index, chunk_mapping, manifest)
    return manifest


def incremental_update(memory_path: Path, config: Config) -> dict:
    """Incrementally update FAISS index — only re-index modified files."""
    manifest = _load_manifest(config.faiss.manifest_path)

    # Check if model changed → full rebuild
    current_model = f"{config.embeddings.provider}/{config.embeddings.model}"
    if manifest.get("embedding_model") != current_model:
        logger.info("Embedding model changed, performing full rebuild")
        return build_index(memory_path, config)

    files = _get_entity_files(memory_path)
    changed_files = []

    for md_file in files:
        rel_path = str(md_file.relative_to(memory_path))
        file_hash = _file_hash(md_file)
        indexed = manifest.get("indexed_files", {}).get(rel_path)
        if not indexed or indexed.get("hash") != file_hash:
            changed_files.append(md_file)

    if not changed_files:
        return manifest  # Nothing to update

    # For simplicity in v1: full rebuild if anything changed
    # A true incremental update would selectively replace vectors
    return build_index(memory_path, config)


def search(query: str, config: Config, memory_path: Path, top_k: int | None = None) -> list[SearchResult]:
    """Search the FAISS index for similar chunks."""
    import faiss

    if top_k is None:
        top_k = config.faiss.top_k

    index_path = Path(config.faiss.index_path)
    mapping_path = Path(config.faiss.mapping_path)

    if not index_path.exists() or not mapping_path.exists():
        # Auto-rebuild if missing
        logger.info("FAISS index missing, building from scratch")
        build_index(memory_path, config)

    if not index_path.exists():
        return []

    index = faiss.read_index(str(index_path))
    with open(mapping_path, "rb") as f:
        chunk_mapping = pickle.load(f)

    if index.ntotal == 0:
        return []

    embed_fn = _get_embedding_fn(config)
    query_vec = embed_fn([query]).astype(np.float32)

    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunk_mapping):
            continue
        mapping = chunk_mapping[idx]
        results.append(SearchResult(
            entity_id=mapping["entity_id"],
            file=mapping["file"],
            chunk=f"[chunk {mapping['chunk_idx']}]",
            score=float(score),
        ))

    return results


def _save_index(config: Config, index, chunk_mapping: list[dict], manifest: dict) -> None:
    """Save FAISS index, mapping, and manifest to disk."""
    import faiss

    index_path = Path(config.faiss.index_path)
    mapping_path = Path(config.faiss.mapping_path)

    index_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    with open(mapping_path, "wb") as f:
        pickle.dump(chunk_mapping, f)
    _save_manifest(config.faiss.manifest_path, manifest)
