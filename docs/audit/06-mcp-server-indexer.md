# Audit 06 — MCP Server & FAISS Indexer

Deep audit of the MCP server (`src/mcp/server.py`), FAISS indexer (`src/pipeline/indexer.py`), standalone launcher (`mcp_stdio.py`), and the `SearchResult` model (`src/core/models.py`).

---

## Table of Contents

1. [MCP Server (`src/mcp/server.py`)](#1-mcp-server)
2. [FAISS Indexer (`src/pipeline/indexer.py`)](#2-faiss-indexer)
3. [Standalone Launcher (`mcp_stdio.py`)](#3-standalone-launcher)
4. [SearchResult Model (`src/core/models.py`)](#4-searchresult-model)
5. [Missing MCP Tools](#5-missing-mcp-tools)
6. [Keyword / FTS5 Hybrid Search](#6-keyword--fts5-hybrid-search)
7. [Reciprocal Rank Fusion (RRF)](#7-reciprocal-rank-fusion-rrf)
8. [Summary of Findings](#8-summary-of-findings)

---

## 1. MCP Server

**File**: `src/mcp/server.py` (215 lines)

### 1.1 Module-Level Setup (lines 1-18)

```python
mcp = FastMCP("memory-ai")
_config = None
```

A single `FastMCP` instance is created at import time. Config is lazily loaded on first tool call via `_get_config()`. The `FastMCP` name `"memory-ai"` is the server identity reported to MCP clients.

**Imports**: `load_graph`, `save_graph` from `src.memory.graph`; `save_chat` (aliased `store_save_chat`) from `src.memory.store`; `search` (aliased `faiss_search`) from `src.pipeline.indexer`.

### 1.2 `_get_config()` (lines 24-28)

```python
def _get_config() -> Config
```

- **Purpose**: Lazy singleton config loader.
- **Logic**: Checks global `_config`; if `None`, calls `load_config()` (no arguments — uses default `config.yaml` in cwd).
- **Thread safety**: None. If two MCP requests arrive simultaneously before config is loaded, `load_config()` could be called twice. In practice, FastMCP stdio is single-threaded, so this is not a real issue in the default transport. For SSE transport with concurrent requests, this is a latent race condition.
- **Finding**: No issue for stdio; potential double-init under SSE.

---

### 1.3 Tool: `get_context()` (lines 31-48)

```python
@mcp.tool()
def get_context() -> str
```

**Input schema**: No parameters.

**Output schema**: Raw string (the context markdown).

**Logic flow**:
1. Load config via `_get_config()`.
2. Try `memory_path / "_context.md"` — if exists, read and return full text.
3. Fallback: try `memory_path / "_index.md"` — if exists, read and return.
4. Final fallback: return a static "No memory context available" message.

**Error handling**: No explicit try/except. If `read_text()` raises (e.g., encoding error), the exception propagates to the MCP framework, which returns an error to the client.

**Performance**: Single file read, O(file_size). No caching — every call re-reads the file from disk. For a typical `_context.md` of 5-20 KB, this is negligible.

**Finding**: No caching, but acceptable for the use case. The file is re-read on every call, which ensures freshness after `memory run` updates context.

---

### 1.4 Tool: `save_chat(messages)` (lines 51-76)

```python
@mcp.tool()
def save_chat(messages: list[dict]) -> dict
```

**Input schema**: `messages` — list of dicts, each requiring `role` (str) and `content` (str).

**Output schema**: Dict with either `{"status": "saved", "file": "<relative_path>"}` or `{"status": "error", "message": "<reason>"}`.

**Logic flow**:
1. Validate `messages` is a non-empty list.
2. Per-message validation: each must be a dict with string `role` and string `content`.
3. Delegates to `store_save_chat(messages, config.memory_path)` (line 74).
4. Returns relative path of saved file.

**Error handling**: Explicit validation returns error dicts (lines 63-71). If `store_save_chat` raises (disk error, permission), exception propagates unhandled.

**What `store_save_chat` does** (`src/memory/store.py:256-276`): Creates `chats/YYYY-MM-DD_HHhMM.md` with YAML frontmatter (`date`, `participants`, `processed: false`), collision avoidance via counter suffix, then writes message body as markdown.

**Performance**: Single file write. Collision counter is linear scan but practically never exceeds 1-2 iterations.

**Finding**: Validation is thorough for shape but does not check role values (e.g., allows arbitrary strings like "banana"). No size limit on messages — a malicious or buggy client could send enormous payloads. No sanitization of content for path traversal (though the filename is generated server-side from timestamp, so this is not exploitable).

---

### 1.5 Tool: `search_rag(query)` (lines 79-189)

```python
@mcp.tool()
def search_rag(query: str) -> dict
```

**Input schema**: `query` — a search string.

**Output schema**:
```json
{
  "query": "...",
  "results": [
    {
      "entity_id": "slug",
      "file": "relative/path.md",
      "score": 0.85,
      "title": "Entity Name",
      "type": "person",
      "relations": [
        {"type": "affects", "target": "...", "target_id": "..."},
        {"type": "linked_to", "source": "...", "source_id": "..."}
      ]
    }
  ],
  "total": 5
}
```

**Full logic flow**:

#### Phase 1 — FAISS Search (lines 93-97)
- Calls `faiss_search(query, config, memory_path)`.
- On any exception: logs warning, returns `{"query": query, "results": [], "total": 0}`. Graceful degradation.

#### Phase 2 — Graph Loading (lines 100-115)
- Calls `load_graph(memory_path)`.
- On failure: returns results without enrichment (no relations, type="unknown", title=entity_id). Graceful degradation.

#### Phase 3 — Hybrid Re-ranking (lines 117-122)
```python
result.score = result.score * 0.6 + graph_score * 0.4
```
- **Formula**: `final_score = 0.6 * FAISS_cosine_similarity + 0.4 * ACT-R_graph_score`.
- FAISS scores are cosine similarities in [0, 1] (normalized embeddings with IndexFlatIP).
- Graph scores are sigmoid outputs in [0, 1] from ACT-R calculation.
- Mutates `result.score` in-place, then sorts descending.
- **Finding**: The weights are hardcoded (0.6/0.4). No config knob to tune the balance. If an entity is not in the graph, `graph_score = 0.0`, so it gets penalized by 40%.

#### Phase 4 — Relation Enrichment (lines 124-161)
- Builds an adjacency dict from all graph relations: `adjacency[entity_id] = [("outgoing"/"incoming", rel), ...]`. This is O(R) where R = total relations, done once.
- Per result: looks up adjacency, resolves target/source entity titles from graph. Skips relations where the other entity is missing from graph.
- Outgoing relations produce `{"type", "target", "target_id"}`.
- Incoming relations produce `{"type", "source", "source_id"}`.

#### Phase 5 — L2-to-L1 Re-emergence (lines 163-183)
```python
from datetime import date as date_type
from src.memory.mentions import add_mention
from src.memory.scoring import recalculate_all_scores

today = date_type.today().isoformat()
```
- For every result entity found in the graph:
  - Calls `add_mention(today, mention_dates, monthly_buckets, window_size)` — appends today's date to `mention_dates`, consolidates to monthly buckets if window exceeded.
  - Updates `last_mentioned` to today.
- If any entity was promoted: calls `recalculate_all_scores(graph, config)` and `save_graph(memory_path, graph)`.
- **Mechanism**: By bumping `mention_dates`, the ACT-R base-level activation `B = ln(sum(t_j^(-d)))` increases because there's now a very recent mention (t_j close to 0.5). This raises the entity's score, potentially above `min_score_for_context` (default 0.3), causing it to appear in `_context.md` on the next context rebuild.
- **Finding — Performance concern**: Every `search_rag` call triggers `recalculate_all_scores()` (O(E + R) where E = entities, R = relations) and `save_graph()` (serializes entire graph to JSON + atomic write). For a graph with thousands of entities, this adds latency to every search. The imports are also inside the function body (lines 164-166), incurring import overhead on every call.
- **Finding — Score inflation**: Every search bumps mentions for ALL returned results (default top_k=5). Frequent searching inflates scores for popular entities, creating a feedback loop. No rate limiting or deduplication of mention bumps (searching the same entity twice in one day adds two mentions).
- **Finding — Stale context**: The L2-to-L1 mechanism only updates the graph scores. It does NOT rebuild `_context.md`. The entity won't actually appear in L1 context until the next `memory run` or `memory context` command rebuilds the context file.

#### Return (lines 185-189)
Returns the enriched results dict.

---

### 1.6 `run_server()` (lines 192-215)

```python
def run_server(config=None, transport_override: str | None = None) -> None
```

- **Purpose**: Entry point to start the MCP server.
- **Logic**: If `config` is provided, sets global `_config`. Otherwise loads via `_get_config()`. Applies `mcp_host` and `mcp_port` to FastMCP settings. Runs with either `"stdio"` or `"sse"` transport.
- **Config defaults**: `mcp_transport="stdio"`, `mcp_host="127.0.0.1"`, `mcp_port=8000` (`src/core/config.py:113-115`).
- **Finding**: Host/port settings are applied even for stdio transport (no effect but no harm).

---

## 2. FAISS Indexer

**File**: `src/pipeline/indexer.py` (294 lines)

### 2.1 `_get_embedding_fn(config)` (lines 25-69)

```python
def _get_embedding_fn(config: Config) -> Callable[[list[str]], np.ndarray]
```

**Purpose**: Factory that returns an embedding function based on the configured provider.

**Three providers supported**:

| Provider | Import | Model format | Normalization | API base |
|----------|--------|-------------|---------------|----------|
| `sentence-transformers` (line 32) | `SentenceTransformer` (lazy) | Direct model name | `normalize_embeddings=True` | N/A (local) |
| `ollama` (line 43) | `litellm` | `ollama/{model_name}` | **None** — raw embeddings | `config.embeddings.api_base` or `http://localhost:11434` |
| `openai` (line 56) | `litellm` | `openai/{model_name}` | **None** — raw embeddings | Optional `config.embeddings.api_base` |

**Caching**: `sentence-transformers` model is cached in module-level globals `_embedding_model` and `_embedding_model_name`. Reloads only if model name changes. Ollama and OpenAI have no caching (stateless API calls).

**Finding — Normalization gap**: `sentence-transformers` explicitly normalizes embeddings (`normalize_embeddings=True`), which is required for `IndexFlatIP` to compute cosine similarity. Ollama and OpenAI providers do **not** normalize. This means:
- With Ollama/OpenAI, `IndexFlatIP` computes raw inner product, not cosine similarity.
- Scores are not bounded to [0, 1] and may be arbitrarily large.
- The re-ranking formula in `search_rag` (0.6 * score + 0.4 * graph_score) assumes scores in [0, 1]. Unnormalized scores break this assumption.
- **Fix**: Add `embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)` after the Ollama/OpenAI embedding calls.

**Finding — No batching for API providers**: Ollama and OpenAI embed all texts in a single API call. For large corpora (hundreds of chunks), this could hit API limits or timeouts. Consider batching (e.g., 100 texts per call).

---

### 2.2 `_chunk_text(text, chunk_size, overlap)` (lines 72-93)

```python
def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]
```

**Purpose**: Split text into overlapping chunks by approximate token count.

**Algorithm**:
1. Split text on whitespace (`text.split()`).
2. Approximate tokens per word: `1.3` (hardcoded constant, line 75).
3. `words_per_chunk = int(400 / 1.3) = 307` words per chunk.
4. `words_overlap = int(80 / 1.3) = 61` words overlap.
5. Sliding window: start at word 0, take 307 words, advance by `307 - 61 = 246` words.
6. Skip empty chunks.
7. If total words <= words_per_chunk, return entire text as single chunk.
8. If no chunks produced (edge case), return [text] if non-empty, else [].

**Finding — Naive tokenization**: Word-based splitting with a fixed 1.3 multiplier is a rough approximation. For French text (the default domain), the ratio may differ. For entity files with YAML frontmatter, the frontmatter is included in chunks, diluting semantic content with metadata.

**Finding — No semantic boundaries**: Chunks split mid-sentence without regard for section headers (`## Facts`, `## Relations`). A chunk could span the end of Facts and the beginning of Relations, mixing different semantic contexts.

**Finding — Overlap can cause loop**: If `words_overlap >= words_per_chunk`, the window never advances. Currently `61 < 307` so this is safe, but there is no guard against misconfiguration.

---

### 2.3 `_file_hash(filepath)` (lines 96-98)

```python
def _file_hash(filepath: Path) -> str
```

Simple SHA256 of file bytes. Used for change detection in manifest.

---

### 2.4 `_load_manifest(manifest_path)` / `_save_manifest(manifest_path, manifest)` (lines 101-113)

```python
def _load_manifest(manifest_path: str) -> dict
def _save_manifest(manifest_path: str, manifest: dict) -> None
```

- Load: Returns default dict `{"embedding_model": "", "last_build": "", "indexed_files": {}}` if file doesn't exist.
- Save: JSON dump with indent=2 and `ensure_ascii=False`.
- **Finding**: No atomic write. If the process crashes mid-write, manifest is corrupted. Compare with `save_graph()` which uses temp file + `os.replace()`.

---

### 2.5 `_get_entity_files(memory_path)` (lines 116-125)

```python
def _get_entity_files(memory_path: Path) -> list[Path]
```

- Recursively globs `*.md` files under `memory_path`.
- Excludes: any file whose path contains a directory starting with `_` (e.g., `_inbox/`, `_archive/`, `_context.md`), or files under `chats/`.
- Returns sorted list.

**Finding**: `_context.md` and `_index.md` are excluded by the `_` prefix rule. This is correct.

---

### 2.6 `build_index(memory_path, config)` (lines 128-182)

```python
def build_index(memory_path: Path, config: Config) -> dict
```

**Purpose**: Full FAISS index rebuild from all entity MD files.

**Logic flow**:
1. Get embedding function via `_get_embedding_fn(config)`.
2. Collect all entity files via `_get_entity_files()`.
3. For each file:
   - Read full text (including YAML frontmatter).
   - Compute SHA256 hash.
   - Chunk text with configured chunk_size/overlap.
   - Append chunks to `all_chunks` list.
   - Record chunk-to-entity mapping: `{file, entity_id, chunk_idx}`.
   - Record in manifest: `{hash, chunks_count, ids}`.
4. If no chunks: create empty index with dimension 384 (default for `all-MiniLM-L6-v2`).
5. Embed all chunks in one batch.
6. Create `faiss.IndexFlatIP(dim)` and add all embeddings.
7. Save index, mapping, and manifest via `_save_index()`.

**FAISS index type — `IndexFlatIP`**:
- Flat = brute-force, no approximation. Exact search.
- IP = Inner Product. With normalized vectors, IP equals cosine similarity.
- **Why this choice**: Exact results, no training needed, simple to build. Suitable for small-to-medium corpora (< 100K vectors). Memory: ~4 bytes * dim * N vectors.
- For the typical memory-ai corpus (hundreds of entities, ~1-5K chunks), this is optimal. An `IndexIVFFlat` or `IndexHNSW` would only help at 100K+ vectors.

**Performance**: Embedding is the bottleneck. For `sentence-transformers` on CPU, ~100-500 chunks/sec. For API providers, limited by network latency and batch size.

**Finding — Hardcoded dimension**: Empty index uses dim=384 (line 168). If the actual model produces different dimensions (e.g., OpenAI `text-embedding-3-small` = 1536), this creates a dimension mismatch error when vectors are later added. The hardcoded fallback should at minimum log a warning or probe the model dimension.

**Finding — Full text indexed**: YAML frontmatter is included in the indexed text. Search queries matching frontmatter fields (like `score: 0.72`) will return false-positive results.

---

### 2.7 `incremental_update(memory_path, config)` (lines 185-210)

```python
def incremental_update(memory_path: Path, config: Config) -> dict
```

**Purpose**: Only re-index files that have changed since last build.

**Logic flow**:
1. Load manifest.
2. If embedding model changed: full rebuild.
3. For each entity file: compare SHA256 hash to manifest.
4. If any files changed: **full rebuild** (line 210).
5. If no changes: return existing manifest.

**Finding — Not actually incremental**: Despite the name, this function always does a full rebuild when anything changes (line 209 comment: "For simplicity in v1"). It's effectively a change-detection wrapper around `build_index()`. A true incremental update would:
- Remove old vectors for changed files (tracked by `ids` in manifest).
- Add new vectors for changed files only.
- `IndexFlatIP` supports `remove_ids()` via `faiss.IDSelectorRange`, but the current index doesn't use `IndexIDMap`, so selective removal isn't possible without refactoring.

**Finding — Deleted files not detected**: The function checks for new/changed files but does not detect files that were deleted since the last build. Deleted entities remain in the index until a full rebuild is triggered by some other change. This is a latent correctness issue.

---

### 2.8 `search(query, config, memory_path, top_k)` (lines 213-256)

```python
def search(query: str, config: Config, memory_path: Path, top_k: int | None = None) -> list[SearchResult]
```

**Purpose**: Query the FAISS index for semantically similar chunks.

**Logic flow**:
1. Default `top_k` to `config.faiss.top_k` (default 5).
2. If index or mapping file missing: auto-rebuild via `build_index()`.
3. Load FAISS index from disk (`faiss.read_index`).
4. Load chunk mapping from pickle.
5. If index is empty: return [].
6. Embed query (single text).
7. Clamp k to `min(top_k, index.ntotal)`.
8. `index.search(query_vec, k)` returns (scores, indices) arrays.
9. For each result: skip invalid indices (idx < 0 or out of range), build `SearchResult`.

**Output**: List of `SearchResult(entity_id, file, chunk, score, relations=[])`. Note: `chunk` is always `"[chunk N]"` — the actual chunk text is NOT returned. Only the chunk index is included.

**Finding — Chunk text not returned**: The `SearchResult.chunk` field contains `"[chunk 0]"` rather than the actual text. The MCP client (Claude) receives entity IDs and scores but cannot see the matched text. This limits the usefulness of search results for direct answer generation. The client must separately read the entity file.

**Finding — No deduplication**: If multiple chunks from the same entity match, they appear as separate results. The MCP `search_rag` doesn't deduplicate either. If entity "back-pain" has 3 matching chunks in top-5, it appears 3 times in results, crowding out other entities.

**Finding — Index loaded on every search**: `faiss.read_index()` and `pickle.load()` are called on every search. No in-memory caching. For stdio transport (single process), caching the index would reduce latency. For SSE transport, caching with file-mtime invalidation would be appropriate.

**Finding — Auto-rebuild on missing index**: If index is missing, `build_index()` is called (line 226-227). But the check at line 228 (`if not index_path.exists()`) could still fail if `build_index()` produced an empty index with no files. In that case, returns [].

---

### 2.9 `list_unextracted_docs(manifest_path)` (lines 259-270)

```python
def list_unextracted_docs(manifest_path: str) -> list[dict]
```

**Purpose**: Find documents in FAISS manifest that were ingested via `doc_ingest` but haven't been entity-extracted yet (used by dream mode step 2).

**Logic**: Filters manifest entries where `source_type == "document"` and `entity_extracted` is falsy. Strips `_doc/` prefix from key to recover `source_id`.

---

### 2.10 `mark_doc_extracted(manifest_path, doc_key)` (lines 273-278)

```python
def mark_doc_extracted(manifest_path: str, doc_key: str) -> None
```

Sets `entity_extracted: True` on a manifest entry. Used after dream mode extracts entities from a previously-ingested document.

---

### 2.11 `_save_index(config, index, chunk_mapping, manifest)` (lines 281-293)

```python
def _save_index(config: Config, index, chunk_mapping: list[dict], manifest: dict) -> None
```

- Creates parent directory for index path.
- Writes FAISS index via `faiss.write_index()`.
- Writes chunk mapping via `pickle.dump()`.
- Writes manifest via `_save_manifest()`.

**Finding — No atomic writes**: Three files are written sequentially without atomicity. If the process crashes after writing the index but before the manifest, the manifest is stale. On next run, `incremental_update` would detect hash mismatches and rebuild, so this is self-healing but not ideal.

**Finding — Pickle security**: `chunk_mapping` is serialized with `pickle`. If an attacker can modify the `.pkl` file, they can execute arbitrary code when the mapping is loaded. In a local-first system this is low risk, but using JSON instead of pickle would be safer.

---

## 3. Standalone Launcher

**File**: `mcp_stdio.py` (29 lines)

```python
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)
from src.mcp.server import run_server
run_server(transport_override="stdio")
```

**Purpose**: Self-contained launcher for Claude Desktop integration. Allows running the MCP server without `pip install` or `uv run`.

**Logic**:
1. Resolves project root from script location.
2. Inserts project root into `sys.path` so `import src.*` works.
3. Changes cwd to project root so `config.yaml`, `memory/`, `prompts/` resolve correctly.
4. Imports and calls `run_server(transport_override="stdio")`.

**Finding**: Straightforward and correct. The `os.chdir()` is essential because `load_config()` resolves paths relative to cwd. The `if __name__ == "__main__"` guard is present but the import + chdir happen at module level, which means importing this module as a library would have side effects (cwd change, sys.path mutation).

---

## 4. SearchResult Model

**File**: `src/core/models.py` (lines 160-165)

```python
class SearchResult(BaseModel):
    entity_id: str
    file: str
    chunk: str
    score: float
    relations: list[dict] = Field(default_factory=list)
```

**Fields**:
- `entity_id`: Entity slug (e.g., `"back-pain"`).
- `file`: Relative path to entity MD file.
- `chunk`: Chunk identifier (set to `"[chunk N]"` by indexer, not actual text).
- `score`: Initially FAISS similarity, later mutated by re-ranking in `search_rag`.
- `relations`: Empty list from indexer; populated by `search_rag` enrichment? Actually no — `search_rag` builds its own `enriched_results` dicts, it doesn't populate this field.

**Finding**: The `relations` field on `SearchResult` is never populated. The MCP server builds its own enriched dicts with relation data instead of using the model's `relations` field. The field is effectively dead code.

---

## 5. Missing MCP Tools

The MCP server exposes exactly 3 tools. Several important operations are NOT available via MCP:

### 5.1 Fact/Observation Modification
- **No `add_fact(entity_id, observation)` tool**: Cannot add a fact to an existing entity without going through the full extraction pipeline (save_chat + run).
- **No `update_fact(entity_id, fact_index, new_content)` tool**: Cannot correct or update individual facts.
- **No `delete_fact(entity_id, fact_index)` tool**: Cannot remove incorrect facts.

### 5.2 Relation Modification
- **No `add_relation(from_id, to_id, type, context)` tool**: Cannot manually create relations.
- **No `delete_relation(from_id, to_id, type)` tool**: Cannot remove incorrect relations.

### 5.3 Entity Management
- **No `create_entity(name, type, observations)` tool**: Cannot create entities directly.
- **No `delete_entity(entity_id)` tool**: Cannot remove entities (only dream mode prunes).
- **No `merge_entities(source_id, target_id)` tool**: Cannot manually merge duplicates.
- **No `update_entity_metadata(entity_id, ...)` tool**: Cannot change type, retention, aliases.

### 5.4 System Operations
- **No `run_pipeline()` tool**: Cannot trigger `memory run` from MCP.
- **No `get_entity(entity_id)` tool**: Cannot read a specific entity's full content.
- **No `list_entities(filter)` tool**: Cannot browse the knowledge base.
- **No `get_stats()` tool**: Cannot query memory statistics.

### 5.5 Impact
The current tool set is read-heavy: get context, search, and save raw chat. The LLM client cannot make targeted corrections to memory without going through the full chat-save-then-extract pipeline. This means:
- Correcting a wrong fact requires saving a synthetic chat that mentions the correction, then running the pipeline.
- The user cannot say "remove that fact about X" and have it take effect immediately.
- Entity management operations require CLI access.

---

## 6. Keyword / FTS5 Hybrid Search

### Current State
Search is purely vector-based (FAISS cosine similarity). There is no keyword/lexical search component.

### Where to Add FTS5

**Option A — SQLite FTS5** (recommended):
- Create `_memory_fts.db` alongside `_memory.faiss`.
- In `build_index()` (line 128), after chunking, also insert chunks into an FTS5 virtual table:
  ```sql
  CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
      entity_id, file, chunk_idx, content,
      tokenize='porter unicode61'
  );
  ```
- In `search()`, add a parallel FTS5 query:
  ```sql
  SELECT entity_id, file, chunk_idx, rank FROM chunks WHERE content MATCH ? ORDER BY rank LIMIT ?
  ```
- FTS5's BM25 ranking provides complementary signal to vector similarity.

**Option B — Whoosh / Tantivy**: Heavier dependencies, more features (faceting, phrase queries). Overkill for this use case.

**Integration point**: The `search()` function (line 213) would run both FAISS and FTS5 queries, then combine results via RRF (see below).

### Benefits
- Exact term matching (e.g., searching for "sciatique" — a specific French medical term — benefits from exact match, not just semantic similarity).
- Handles out-of-vocabulary terms better than embeddings.
- Near-zero latency for FTS5 on small corpora.

---

## 7. Reciprocal Rank Fusion (RRF)

### Current Re-ranking
The current re-ranking (line 121) is a linear weighted sum:
```python
result.score = result.score * 0.6 + graph_score * 0.4
```

This has problems:
- Assumes both scores are on the same scale [0, 1].
- Ollama/OpenAI embeddings produce unnormalized scores (see finding in 2.1).
- Graph scores cluster around certain ranges due to sigmoid, making the 0.4 weight's actual impact variable.

### RRF Implementation

Reciprocal Rank Fusion is rank-based, not score-based, making it robust to score scale differences.

**Formula**:
```
RRF_score(d) = sum(1 / (k + rank_i(d))) for each ranker i
```
Where `k` is a constant (typically 60).

**How to implement** in `search_rag()`:

```python
def _rrf_merge(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge multiple ranked lists via Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of lists, each containing entity_ids in rank order.
        k: Smoothing constant (default 60).

    Returns:
        List of (entity_id, rrf_score) tuples sorted by score descending.
    """
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, entity_id in enumerate(ranked, start=1):
            scores[entity_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Integration** in `search_rag()` (replacing lines 117-122):
```python
# Rank list 1: FAISS similarity order (already sorted by FAISS)
faiss_ranked = [r.entity_id for r in results]

# Rank list 2: ACT-R graph score order
graph_ranked = sorted(
    [r.entity_id for r in results if r.entity_id in graph.entities],
    key=lambda eid: graph.entities[eid].score,
    reverse=True,
)

# Rank list 3 (future): FTS5 BM25 order
# fts5_ranked = [r.entity_id for r in fts5_results]

# Merge
merged = _rrf_merge([faiss_ranked, graph_ranked], k=60)
```

**Benefits over current approach**:
- Scale-invariant: works regardless of whether FAISS scores are normalized.
- Extensible: adding a third ranker (FTS5, BM25, recency) requires only appending to `ranked_lists`.
- Well-studied: RRF consistently outperforms linear combination in information retrieval benchmarks.

---

## 8. Summary of Findings

### Critical Issues

| # | Location | Issue |
|---|----------|-------|
| 1 | `indexer.py:46-64` | **Ollama/OpenAI embeddings not normalized.** `IndexFlatIP` requires normalized vectors for cosine similarity. Raw inner product scores break the 0.6/0.4 re-ranking formula. |
| 2 | `server.py:163-183` | **`search_rag` triggers full score recalculation and graph save on every call.** O(E+R) computation + disk I/O on every search. |

### Significant Issues

| # | Location | Issue |
|---|----------|-------|
| 3 | `indexer.py:185-210` | **`incremental_update` always does full rebuild.** Despite the name, any change triggers complete re-indexing. |
| 4 | `server.py:170-178` | **No rate limiting on mention bumps.** Repeated searches inflate entity scores without bound. |
| 5 | `indexer.py:168` | **Hardcoded dimension 384 for empty index.** Breaks if using a model with different dimensions. |
| 6 | `indexer.py:249` | **Chunk text not returned in SearchResult.** Only `"[chunk N]"` is stored; MCP clients cannot see matched text. |
| 7 | `server.py:117-122` | **Hardcoded re-ranking weights (0.6/0.4).** No config knob to tune the balance. |

### Minor Issues

| # | Location | Issue |
|---|----------|-------|
| 8 | `indexer.py:109-113` | No atomic write for manifest file. |
| 9 | `indexer.py:232-233` | Index loaded from disk on every search (no caching). |
| 10 | `indexer.py:143-144` | YAML frontmatter indexed alongside content, causing false positive matches. |
| 11 | `models.py:165` | `SearchResult.relations` field is never populated (dead field). |
| 12 | `server.py:62-71` | `save_chat` accepts arbitrary role values without validation. |
| 13 | `indexer.py:72-93` | Chunk boundaries ignore sentence/section structure. |
| 14 | `server.py:163-183` | L2-to-L1 updates graph scores but does NOT rebuild `_context.md`. Entity won't appear in L1 until next `memory run`. |

### Architectural Gaps

| # | Area | Gap |
|---|------|-----|
| 15 | MCP tools | No tools for fact/relation/entity CRUD operations. |
| 16 | MCP tools | No `run_pipeline` or `get_entity` tools. |
| 17 | Search | No keyword/lexical search component (FTS5). |
| 18 | Search | No entity-level deduplication in results. |
| 19 | Re-ranking | Linear weighted sum instead of rank-based fusion (RRF). |
