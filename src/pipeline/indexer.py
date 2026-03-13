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

# ── spaCy integration for sentence-aware chunking ──────────────
_spacy_nlp_cache: dict[str, Any] = {}

SPACY_MODELS = {
    "fr": "fr_core_news_sm",
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "es": "es_core_news_sm",
    "it": "it_core_news_sm",
    "pt": "pt_core_news_sm",
    "nl": "nl_core_news_sm",
    "zh": "zh_core_web_sm",
}


def _get_spacy_nlp(language: str = "en"):
    """Load spaCy model with auto-download fallback. Returns None if spaCy not installed."""
    if language in _spacy_nlp_cache:
        return _spacy_nlp_cache[language]
    try:
        import spacy
        model_name = SPACY_MODELS.get(language, "en_core_web_sm")
        try:
            nlp = spacy.load(model_name)
        except OSError:
            logger.info("spaCy model %s not found, downloading...", model_name)
            spacy.cli.download(model_name)
            nlp = spacy.load(model_name)
        _spacy_nlp_cache[language] = nlp
        return nlp
    except ImportError:
        return None
    except Exception:
        # Network failure during download, air-gapped env, etc.
        return None


def _split_sentences_regex(text: str) -> list[str]:
    """Fallback sentence splitter using regex."""
    import re
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p for p in parts if p.strip()]


# Lazy-loaded globals
_embedding_model = None
_embedding_model_name = None


def _normalize_l2(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors for cosine similarity with IndexFlatIP."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return (vectors / norms).astype(np.float32)


def get_embedding_fn(config: Config):
    """Get an embedding function based on config.

    All providers return L2-normalized vectors so that FAISS IndexFlatIP
    computes cosine similarity correctly.
    """
    global _embedding_model, _embedding_model_name

    provider = config.embeddings.provider
    model_name = config.embeddings.model

    if provider == "sentence-transformers":
        if _embedding_model is None or _embedding_model_name != model_name:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(model_name)
            _embedding_model_name = model_name

        def embed(texts: list[str]) -> np.ndarray:
            vecs = _embedding_model.encode(texts, normalize_embeddings=True)
            return _normalize_l2(np.asarray(vecs, dtype=np.float32))

        return embed

    elif provider == "ollama":
        import litellm

        def embed(texts: list[str]) -> np.ndarray:
            response = litellm.embedding(
                model=f"ollama/{model_name}",
                input=texts,
                api_base=config.embeddings.api_base or "http://localhost:11434",
            )
            vecs = np.array([d["embedding"] for d in response.data], dtype=np.float32)
            return _normalize_l2(vecs)

        return embed

    elif provider == "openai":
        import litellm

        def embed(texts: list[str]) -> np.ndarray:
            kwargs: dict[str, Any] = {"model": f"openai/{model_name}", "input": texts}
            if config.embeddings.api_base:
                kwargs["api_base"] = config.embeddings.api_base
            response = litellm.embedding(**kwargs)
            vecs = np.array([d["embedding"] for d in response.data], dtype=np.float32)
            return _normalize_l2(vecs)

        return embed

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80,
               language: str = "en") -> list[str]:
    """Split text into overlapping chunks by sentence boundaries.

    Uses spaCy for sentence segmentation when available, falls back to regex.
    """
    if not text.strip():
        return []

    def _tok(t: str) -> int:
        """Estimate token count: ~4 chars per token."""
        return max(1, len(t) // 4)

    # Get sentences (spaCy or regex fallback)
    nlp = _get_spacy_nlp(language)
    if nlp is not None:
        try:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except Exception:
            sentences = _split_sentences_regex(text)
    else:
        sentences = _split_sentences_regex(text)

    if not sentences:
        return [text] if text.strip() else []

    # If entire text fits in one chunk, return as-is
    if _tok(text) <= chunk_size:
        return [text]

    # Accumulate sentences into chunks
    chunks = []
    current_chunk: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _tok(sentence)
        # If a single sentence is larger than chunk_size, split it by words
        if sentence_tokens > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            words = sentence.split()
            word_start = 0
            while word_start < len(words):
                word_end = word_start
                word_tokens = 0
                while word_end < len(words):
                    wt = _tok(words[word_end])
                    if word_tokens + wt >= chunk_size and word_end > word_start:
                        break
                    word_tokens += wt
                    word_end += 1
                sub = " ".join(words[word_start:word_end])
                if sub.strip():
                    chunks.append(sub)
                # Overlap in words
                overlap_words = max(0, word_end - max(1, int(overlap / max(1, _tok(words[0])))))
                word_start = max(word_start + 1, overlap_words)
            continue
        if current_tokens + sentence_tokens >= chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Overlap: keep trailing sentences that fit within overlap budget
            overlap_tokens = 0
            overlap_start = len(current_chunk)
            for i in range(len(current_chunk) - 1, -1, -1):
                overlap_tokens += _tok(current_chunk[i])
                if overlap_tokens >= overlap:
                    overlap_start = i
                    break
            current_chunk = current_chunk[overlap_start:]
            current_tokens = sum(_tok(s) for s in current_chunk)
        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else ([text] if text.strip() else [])


def _file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


def load_manifest(manifest_path: str) -> dict:
    """Load FAISS manifest from JSON."""
    path = Path(manifest_path)
    if not path.exists():
        return {"embedding_model": "", "last_build": "", "indexed_files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(manifest_path: str, manifest: dict) -> None:
    """Save FAISS manifest to JSON."""
    Path(manifest_path).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# Backward-compatible aliases
_chunk_text = chunk_text
_get_embedding_fn = get_embedding_fn
_load_manifest = load_manifest
_save_manifest = save_manifest


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

    embed_fn = get_embedding_fn(config)
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
        # Strip YAML frontmatter to avoid indexing metadata
        if text.startswith("---"):
            parts = text.split("---", 2)
            text = parts[2] if len(parts) >= 3 else text
        file_hash = _file_hash(md_file)
        rel_path = str(md_file.relative_to(memory_path))
        entity_id = md_file.stem

        chunks = chunk_text(text, config.embeddings.chunk_size, config.embeddings.chunk_overlap,
                            language=config.user_language)

        start_idx = len(all_chunks)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_mapping.append({
                "file": rel_path,
                "entity_id": entity_id,
                "chunk_idx": i,
                "chunk_text": chunk,
            })

        manifest["indexed_files"][rel_path] = {
            "hash": file_hash,
            "chunks": len(chunks),
            "ids": list(range(start_idx, start_idx + len(chunks))),
        }

    if not all_chunks:
        # Probe embedding function to get actual dimension
        try:
            probe = embed_fn(["test"])
            dim = probe.shape[1]
        except Exception:
            dim = 384  # fallback for all-MiniLM-L6-v2
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
    manifest = load_manifest(config.faiss.manifest_path)

    # Check if model changed → full rebuild
    current_model = f"{config.embeddings.provider}/{config.embeddings.model}"
    if manifest.get("embedding_model") != current_model:
        logger.info("Embedding model changed, performing full rebuild")
        manifest = build_index(memory_path, config)
        try:
            from src.pipeline.keyword_index import build_keyword_index

            fts_db_path = memory_path / config.search.fts_db_path
            build_keyword_index(memory_path, fts_db_path)
        except Exception:
            logger.warning("Failed to build FTS5 keyword index", exc_info=True)
        return manifest

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
    manifest = build_index(memory_path, config)

    # Rebuild FTS5 keyword index alongside FAISS
    try:
        from src.pipeline.keyword_index import build_keyword_index

        fts_db_path = memory_path / config.search.fts_db_path
        build_keyword_index(memory_path, fts_db_path)
    except Exception:
        logger.warning("Failed to build FTS5 keyword index", exc_info=True)

    return manifest


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

    embed_fn = get_embedding_fn(config)
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
            chunk=mapping.get("chunk_text", f"[chunk {mapping['chunk_idx']}]"),
            score=float(score),
        ))

    # Deduplicate by entity: keep best-scoring chunk per entity
    seen: dict[str, SearchResult] = {}
    for result in results:
        if result.entity_id not in seen or result.score > seen[result.entity_id].score:
            seen[result.entity_id] = result
    return sorted(seen.values(), key=lambda r: r.score, reverse=True)[:top_k]


def list_unextracted_docs(manifest_path: str) -> list[dict]:
    """List documents in FAISS manifest that haven't been entity-extracted."""
    manifest = load_manifest(manifest_path)
    docs = []
    for key, entry in manifest.get("indexed_files", {}).items():
        if entry.get("source_type") != "document":
            continue
        if entry.get("entity_extracted", False):
            continue
        source_id = key.replace("_doc/", "", 1)
        docs.append({"source_id": source_id, "key": key, **entry})
    return docs


def mark_doc_extracted(manifest_path: str, doc_key: str) -> None:
    """Mark a document as entity-extracted in the manifest."""
    manifest = load_manifest(manifest_path)
    if doc_key in manifest.get("indexed_files", {}):
        manifest["indexed_files"][doc_key]["entity_extracted"] = True
        save_manifest(manifest_path, manifest)


def _save_index(config: Config, index, chunk_mapping: list[dict], manifest: dict) -> None:
    """Save FAISS index, mapping, and manifest to disk."""
    import faiss

    index_path = Path(config.faiss.index_path)
    mapping_path = Path(config.faiss.mapping_path)

    index_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    with open(mapping_path, "wb") as f:
        pickle.dump(chunk_mapping, f)
    save_manifest(config.faiss.manifest_path, manifest)
