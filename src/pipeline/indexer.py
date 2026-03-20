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

logger = logging.getLogger(__name__)
from src.core.models import SearchResult
from src.core.utils import is_entity_file

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
    except Exception as e:
        # Network failure during download, air-gapped env, etc.
        logger.debug("spaCy model load failed: %s", e)
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
        except Exception as e:
            logger.debug("spaCy sentence split failed, using regex fallback: %s", e)
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
        if not is_entity_file(rel.parts):
            continue
        files.append(md_file)
    return files


def build_index(memory_path: Path, config: Config) -> dict:
    """Build FAISS index from scratch using IndexIDMap for delta support. Returns manifest."""
    import faiss

    embed_fn = get_embedding_fn(config)
    files = _get_entity_files(memory_path)

    all_chunks: list[str] = []
    # Dict keyed by FAISS ID for O(1) lookup during search and delta updates
    chunk_mapping: dict[int, dict] = {}
    manifest = {
        "embedding_model": f"{config.embeddings.provider}/{config.embeddings.model}",
        "last_build": datetime.now().isoformat(),
        "indexed_files": {},
        "next_id": 0,
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
            vec_id = start_idx + i
            all_chunks.append(chunk)
            chunk_mapping[vec_id] = {
                "file": rel_path,
                "entity_id": entity_id,
                "chunk_idx": i,
                "chunk_text": chunk,
            }

        manifest["indexed_files"][rel_path] = {
            "hash": file_hash,
            "chunks": len(chunks),
            "ids": list(range(start_idx, start_idx + len(chunks))),
        }

    manifest["next_id"] = len(all_chunks)

    if not all_chunks:
        # Probe embedding function to get actual dimension
        try:
            probe = embed_fn(["test"])
            dim = probe.shape[1]
        except Exception as e:
            logger.debug("Embedding probe failed, using default dim=384: %s", e)
            dim = 384  # fallback for all-MiniLM-L6-v2
        base_index = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap(base_index)
        _save_index(config, index, chunk_mapping, manifest)
        return manifest

    # Embed all chunks
    embeddings = embed_fn(all_chunks)
    dim = embeddings.shape[1]

    # Build FAISS IndexIDMap wrapping IndexFlatIP for delta update support
    base_index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(base_index)
    ids = np.arange(len(all_chunks), dtype=np.int64)
    index.add_with_ids(embeddings.astype(np.float32), ids)

    _save_index(config, index, chunk_mapping, manifest)
    return manifest


def _rebuild_keyword_index(memory_path: Path, config: Config) -> None:
    """Rebuild FTS5 keyword index alongside FAISS. Swallows errors."""
    try:
        from src.pipeline.keyword_index import build_keyword_index

        fts_db_path = memory_path / config.search.fts_db_path
        build_keyword_index(memory_path, fts_db_path, chunk_size=config.embeddings.chunk_size, chunk_overlap=config.embeddings.chunk_overlap)
    except Exception:
        logger.warning("Failed to build FTS5 keyword index", exc_info=True)


def incremental_update(memory_path: Path, config: Config) -> dict:
    """Incrementally update FAISS index — delta re-indexing for modified files.

    Uses IndexIDMap to selectively remove and re-add vectors for changed files.
    Falls back to full rebuild if >30% of files changed or if the index format
    doesn't support delta updates (legacy IndexFlatIP without IDMap).
    """
    import faiss

    manifest = load_manifest(config.faiss.manifest_path)

    # Check if model changed → full rebuild
    current_model = f"{config.embeddings.provider}/{config.embeddings.model}"
    if manifest.get("embedding_model") != current_model:
        logger.info("Embedding model changed, performing full rebuild")
        manifest = build_index(memory_path, config)
        _rebuild_keyword_index(memory_path, config)
        return manifest

    files = _get_entity_files(memory_path)
    current_file_paths = {str(f.relative_to(memory_path)) for f in files}
    indexed_files = manifest.get("indexed_files", {})

    # Detect changes: modified, new, and deleted files
    changed_files: list[Path] = []
    new_files: list[Path] = []
    deleted_paths: list[str] = []

    for md_file in files:
        rel_path = str(md_file.relative_to(memory_path))
        file_hash = _file_hash(md_file)
        indexed = indexed_files.get(rel_path)
        if not indexed:
            new_files.append(md_file)
        elif indexed.get("hash") != file_hash:
            changed_files.append(md_file)

    for rel_path in indexed_files:
        # Skip non-entity entries (e.g., _doc/ entries from doc_ingest)
        if indexed_files[rel_path].get("source_type") == "document":
            continue
        if rel_path not in current_file_paths:
            deleted_paths.append(rel_path)

    total_changes = len(changed_files) + len(new_files) + len(deleted_paths)

    if total_changes == 0:
        return manifest  # Nothing to update

    total_files = max(len(files), len(indexed_files), 1)
    change_ratio = total_changes / total_files

    # Fall back to full rebuild if >30% changed or no next_id (legacy format)
    if change_ratio > 0.3 or "next_id" not in manifest:
        logger.info(
            "Delta ratio %.0f%% (changed=%d, new=%d, deleted=%d) — full rebuild",
            change_ratio * 100, len(changed_files), len(new_files), len(deleted_paths),
        )
        manifest = build_index(memory_path, config)
        _rebuild_keyword_index(memory_path, config)
        return manifest

    # ── Delta update ──────────────────────────────────────────
    logger.info(
        "Delta update: %d changed, %d new, %d deleted (%.0f%% of %d files)",
        len(changed_files), len(new_files), len(deleted_paths),
        change_ratio * 100, total_files,
    )

    index_path = Path(config.faiss.index_path)
    mapping_path = Path(config.faiss.mapping_path)

    if not index_path.exists() or not mapping_path.exists():
        manifest = build_index(memory_path, config)
        _rebuild_keyword_index(memory_path, config)
        return manifest

    # Load existing index and mapping
    index = faiss.read_index(str(index_path))
    with open(mapping_path, "rb") as f:
        raw_mapping = pickle.load(f)

    # Normalize mapping to dict format (backwards compat with list format)
    if isinstance(raw_mapping, list):
        chunk_mapping: dict[int, dict] = {i: cm for i, cm in enumerate(raw_mapping)}
    else:
        chunk_mapping = raw_mapping

    # Verify we have an IndexIDMap (supports remove_ids)
    if not hasattr(index, "remove_ids"):
        logger.info("Index does not support remove_ids, performing full rebuild")
        manifest = build_index(memory_path, config)
        _rebuild_keyword_index(memory_path, config)
        return manifest

    embed_fn = get_embedding_fn(config)
    next_id = manifest.get("next_id", max(chunk_mapping.keys(), default=-1) + 1)

    # Step 1: Collect IDs to remove (changed + deleted files)
    ids_to_remove: list[int] = []
    files_to_remove = {str(f.relative_to(memory_path)) for f in changed_files} | set(deleted_paths)

    for rel_path in files_to_remove:
        entry = indexed_files.get(rel_path)
        if entry and "ids" in entry:
            ids_to_remove.extend(entry["ids"])

    # Remove old vectors from index
    if ids_to_remove:
        remove_array = np.array(ids_to_remove, dtype=np.int64)
        index.remove_ids(remove_array)

    # Remove old entries from chunk_mapping
    for vec_id in ids_to_remove:
        chunk_mapping.pop(vec_id, None)

    # Remove from manifest
    for rel_path in files_to_remove:
        indexed_files.pop(rel_path, None)
    for rel_path in deleted_paths:
        indexed_files.pop(rel_path, None)

    # Step 2: Add new/changed files
    files_to_add = changed_files + new_files
    new_chunks: list[str] = []
    new_ids: list[int] = []

    for md_file in files_to_add:
        text = md_file.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", 2)
            text = parts[2] if len(parts) >= 3 else text

        file_hash = _file_hash(md_file)
        rel_path = str(md_file.relative_to(memory_path))
        entity_id = md_file.stem

        chunks = chunk_text(
            text, config.embeddings.chunk_size, config.embeddings.chunk_overlap,
            language=config.user_language,
        )

        file_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = next_id
            next_id += 1
            new_chunks.append(chunk)
            new_ids.append(chunk_id)
            file_ids.append(chunk_id)
            chunk_mapping[chunk_id] = {
                "file": rel_path,
                "entity_id": entity_id,
                "chunk_idx": i,
                "chunk_text": chunk,
            }

        indexed_files[rel_path] = {
            "hash": file_hash,
            "chunks": len(chunks),
            "ids": file_ids,
        }

    # Embed and add new vectors
    if new_chunks:
        embeddings = embed_fn(new_chunks)
        ids_array = np.array(new_ids, dtype=np.int64)
        index.add_with_ids(embeddings.astype(np.float32), ids_array)

    # Update manifest
    manifest["last_build"] = datetime.now().isoformat()
    manifest["next_id"] = next_id

    _save_index(config, index, chunk_mapping, manifest)
    _rebuild_keyword_index(memory_path, config)

    return manifest


def search(query: str, config: Config, memory_path: Path, top_k: int | None = None) -> list[SearchResult]:
    """Search the FAISS index for similar chunks.

    Supports both legacy list-based and new dict-based chunk mappings.
    With IndexIDMap, returned indices are custom IDs used as dict keys.
    With legacy IndexFlatIP, returned indices are positional into the list.
    """
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
        raw_mapping = pickle.load(f)

    if index.ntotal == 0:
        return []

    # Normalize mapping: dict (new format) or list (legacy format)
    if isinstance(raw_mapping, list):
        chunk_mapping: dict[int, dict] = {i: cm for i, cm in enumerate(raw_mapping)}
    else:
        chunk_mapping = raw_mapping

    embed_fn = get_embedding_fn(config)
    query_vec = embed_fn([query]).astype(np.float32)

    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        mapping = chunk_mapping.get(int(idx))
        if mapping is None:
            continue

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


def _save_index(config: Config, index, chunk_mapping: dict[int, dict] | list[dict], manifest: dict) -> None:
    """Save FAISS index, mapping, and manifest to disk."""
    import faiss

    index_path = Path(config.faiss.index_path)
    mapping_path = Path(config.faiss.mapping_path)

    index_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    with open(mapping_path, "wb") as f:
        pickle.dump(chunk_mapping, f)
    save_manifest(config.faiss.manifest_path, manifest)
