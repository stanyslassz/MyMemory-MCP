# Audit 08 — Pipeline Orchestration & CLI

Deep audit of the pipeline orchestration layer, CLI commands, fallback pipelines, inbox routing, chat splitting, and ingest state machine.

---

## Table of Contents

1. [Pipeline Orchestrator (`src/pipeline/orchestrator.py`)](#1-pipeline-orchestrator)
2. [CLI Commands (`src/cli.py`)](#2-cli-commands)
3. [Document Ingest Pipeline (`src/pipeline/doc_ingest.py`)](#3-document-ingest-pipeline)
4. [Inbox Pipeline (`src/pipeline/inbox.py`)](#4-inbox-pipeline)
5. [Chat Splitter (`src/pipeline/chat_splitter.py`)](#5-chat-splitter)
6. [Ingest State Machine (`src/pipeline/ingest_state.py`)](#6-ingest-state-machine)
7. [Router (`src/pipeline/router.py`)](#7-router)
8. [Full Pipeline Chain — End to End](#8-full-pipeline-chain)
9. [Where `sanitize_extraction()` Is Called](#9-where-sanitize_extraction-is-called)
10. [Correction Interception Points](#10-correction-interception-points)
11. [Error Handling & Recovery Mechanisms](#11-error-handling--recovery-mechanisms)
12. [Findings & Issues](#12-findings--issues)

---

## 1. Pipeline Orchestrator

**File**: `src/pipeline/orchestrator.py`

The orchestrator contains the core business logic, extracted from `cli.py` to keep CLI wrappers thin. It has no Click dependency — all functions accept a `console: rich.console.Console` parameter for output.

### 1.1 `is_timeout_error(exc: Exception) -> bool` (line 19)

**Purpose**: Classify whether an exception is timeout-related to decide fallback behavior.

**Logic**:
- Checks if `exc` is an instance of `StallError` (imported from `src.core.llm`)
- Checks exception string and type name against indicators: `"timeout"`, `"timed out"`, `"ReadTimeout"`, `"ConnectTimeout"`, `"stall"`
- Case-insensitive comparison

**Returns**: `True` if any indicator matches.

### 1.2 `make_faiss_fn(config, memory_path) -> Callable` (line 34)

**Purpose**: Adapter between the resolver's expected FAISS interface and the indexer's actual API.

**Signature bridge**:
- Resolver expects: `fn(query: str, top_k: int, threshold: float) -> list[dict]`
- Indexer provides: `search(query, config, memory_path, top_k) -> list[SearchResult]`

**Logic**:
- Creates a closure that calls `indexer.search()` and filters results by `threshold` (default 0.85)
- Returns list of `{"entity_id": ..., "score": ...}` dicts

**Note**: The threshold parameter (0.85) is set at call site, not configurable via `config.yaml`. The resolver itself uses a separate 0.75 threshold for its own FAISS calls.

### 1.3 `fallback_to_doc_ingest(chat_path, content, reason, memory_path, config, console) -> None` (line 53)

**Purpose**: Fallback path when extraction fails — chunks and indexes the raw content via FAISS instead of creating entities.

**Flow**:
1. Computes `IngestKey` from `chat_path.name` + `content`
2. **Idempotency guard**: checks `has_been_ingested(key, config)` — if already ingested, just marks chat as fallback-processed and returns (line 64-66)
3. Creates ingest job via `create_job(key, config, route="fallback_doc_ingest")` (line 70)
4. Transitions job to `"running"` (line 71)
5. Calls `ingest_document(source_id, content, key, memory_path, config)` (line 72)
6. On success: transitions to `"succeeded"` with `chunks_indexed` count (line 73-76)
7. On failure: prints error but does NOT transition job (bug — see Findings) (line 78-79)
8. **Always** marks chat as fallback-processed via `mark_chat_fallback()` regardless of ingest outcome (line 82)

### 1.4 `discover_batch_relations(touched_ids, graph, config, memory_path, console) -> int` (line 89)

**Purpose**: Post-enrichment deterministic relation discovery. Uses FAISS similarity + tag overlap heuristics. Zero LLM calls.

**Signature**: `(touched_ids: list[str], graph, config, memory_path, console) -> int`

**Algorithm**:
1. Early return if `touched_ids` is empty (line 100)
2. Build set of existing relation pairs `(from, to)` + `(to, from)` for bidirectional dedup (lines 108-111)
3. Deduplicate `touched_ids` preserving order via `dict.fromkeys()` (line 114)
4. For each touched entity:
   - FAISS search with `top_k=3` using entity title as query (line 122)
   - For each result: skip self, skip missing entities, skip existing pairs (lines 127-130)
   - **Discovery criterion**: 2+ shared tags AND FAISS score >= 0.8 (line 134)
   - Creates `GraphRelation(type="linked_to", context="tag overlap: ...")` (lines 135-138)
   - Calls `add_relation()` with Hebbian `strength_growth` from config (line 140)
   - Adds pair to `existing` set to prevent intra-batch duplicates (lines 141-142)
5. If any discovered: `save_graph()` and print count (lines 145-148)

**Returns**: Count of newly discovered relations.

### 1.5 `auto_consolidate(memory_path, config, console, min_facts=8) -> None` (line 155)

**Purpose**: Auto-consolidate entities with too many facts. Called by `memory run` but NOT by `memory run-light`.

**Flow**:
1. Loads graph from `_graph.json` (line 160)
2. Iterates all entities in graph (line 163)
3. For each entity with existing file:
   - Gets `max_facts` from `config.get_max_facts(entity.type)` — type-specific caps (line 168)
   - Reads entity MD, counts live facts (excludes `[superseded]`) (lines 169-171)
   - If live facts exceed `max_facts`: calls `consolidate_entity_facts(entity_path, config, max_facts=max_facts)` (line 174)
   - This triggers an LLM call via `call_fact_consolidation()` in `store.py` (line 193-214 of store.py)
4. Reports consolidated count

**Key detail**: The threshold is `max_facts` from config (per entity type), NOT the `min_facts=8` default parameter. The `min_facts` parameter is unused in `auto_consolidate` — it only exists on the function signature for compatibility but the comparison is `len(live_facts) > max_facts` (line 172).

### 1.6 `consolidate_facts(config, console, dry_run, min_facts) -> None` (line 189)

**Purpose**: CLI-facing fact consolidation logic for `memory consolidate --facts`.

**Flow**:
1. Loads graph and scans all entities (lines 194-195)
2. For each entity: reads facts, counts live facts, applies threshold as `min(min_facts, max_facts)` (line 209) — uses the stricter of CLI arg or config cap
3. If `dry_run`: lists candidates and returns (line 223)
4. Otherwise: calls `consolidate_entity_facts()` per candidate (line 230)

### 1.7 `run_pipeline(config, console, *, consolidate=True) -> None` (line 245)

**Purpose**: The main pipeline entry point shared by `memory run` and `memory run-light`.

**Parameter**: `consolidate: bool` — `True` for `run`, `False` for `run-light`

**Constants**:
- `EXTRACTION_MAX_RETRIES = 2` (line 263)

**Full flow — per-chat loop**:

#### Chat loading and capping (lines 265-272)
1. `list_unprocessed_chats(memory_path)` — scans `chats/*.md`, filters by `processed: false` in frontmatter, returns sorted list
2. Caps to `config.job_max_chats_per_run` via slice `chats[:max_chats]` (line 271)
3. Tracks `all_touched_ids: list[str]` across all chats (line 275)

#### Step 1 — Extraction (lines 279-304)
1. `get_chat_content(chat_path)` — reads MD body (strips frontmatter)
2. Empty content: marks processed with empty entity lists, continues (lines 281-284)
3. `extract_from_chat(content, config)` — LLM extraction (may split long content into segments)
4. `sanitize_extraction(extraction)` — post-extraction cleanup (line 288)
5. **On failure**:
   - Checks `is_timeout_error(e)` (line 291)
   - `increment_extraction_retries(chat_path)` — bumps counter in chat frontmatter (line 292)
   - Fallback if timeout OR retries >= 2 (line 293):
     - `fallback_to_doc_ingest()` (line 298)
     - `record_failure()` to retry ledger (lines 300-301)
   - Otherwise: prints retry count, continues to next chat (line 303)

#### Step 2 — Resolution (lines 307-308)
1. Reloads graph (fresh state after potential prior enrichment) (line 307)
2. `resolve_all(extraction, graph, faiss_search_fn=make_faiss_fn(config, memory_path))` (line 308)
3. Returns `ResolvedExtraction` with status per entity: `"resolved"`, `"new"`, or `"ambiguous"`

#### Step 3 — Arbitration (lines 311-338)
1. Iterates `resolved.resolved` items
2. Only processes `status == "ambiguous"` (line 312)
3. Calls `arbitrate_entity(name, extraction.summary, candidates, graph, config)` — LLM call (lines 314-319)
4. Result `action == "existing"`: resolves to existing entity ID (lines 321-324)
5. Result `action == "new"`: creates new entity with slugified name (lines 326-329)
6. **On failure**: falls back to creating new entity (lines 332-338) — graceful degradation

#### Step 4 — Enrichment (lines 341-354)
1. `enrich_memory(resolved, config)` → `EnrichmentReport` (line 342)
2. Collects `entities_updated` and `entities_created` into `all_touched_ids` (lines 344-345)
3. Prints any warnings from `report.errors` (lines 346-348)
4. **On failure**: prints error and continues to next chat (skips `mark_chat_processed`) (lines 349-351) — **BUG**: chat remains unprocessed, will be retried indefinitely on enrichment errors
5. `mark_chat_processed(chat_path, report.entities_updated, report.entities_created)` (line 354)

#### Post-loop: Step 5a — Batch Relation Discovery (lines 357-361)
1. Reloads graph (line 358)
2. `discover_batch_relations(all_touched_ids, graph, config, memory_path, console)` (line 359)
3. Wrapped in try/except — failure is non-fatal (line 361)

#### Post-loop: Step 5b — Auto-Consolidation (lines 363-368)
1. Only if `consolidate=True` (i.e., `memory run`, not `run-light`) (line 364)
2. `auto_consolidate(memory_path, config, console)` (line 366)
3. Wrapped in try/except — failure is non-fatal (line 368)

#### Post-loop: Step 7 — Context Generation (lines 371-388)
(Note: numbering skips Step 6 in the code — comment says "Step 7")
1. Reloads graph again (line 372)
2. If `consolidate=True` AND `config.context_llm_sections` is true: uses `build_context_with_llm()` (LLM per-section) (lines 373-377)
3. Otherwise: uses `build_context()` (deterministic template, zero LLM) (lines 379-381)
4. Writes `_context.md` via `write_context()` (line 383)
5. Wrapped in try/except — failure is non-fatal (line 388)

#### Post-loop: Step 8 — FAISS Indexing (lines 391-396)
1. `incremental_update(memory_path, config)` — compares file hashes, rebuilds changed entries (line 393)
2. Wrapped in try/except — failure is non-fatal (line 396)

---

## 2. CLI Commands

**File**: `src/cli.py`

All commands are Click wrappers. The CLI group initializes config and memory structure on every invocation.

### 2.1 `cli()` — Group (line 36)

**Options**: `--verbose/-v` (debug logging), `--config/-c` (config path override)

**Setup**:
1. `_setup_logging(verbose)` (line 38)
2. Sets `project_root = Path.cwd()` (line 40)
3. `load_config(config_path, project_root)` (line 41)
4. `init_memory_structure(config.memory_path)` — creates all subdirectories (line 45)

### 2.2 `run` (line 48)

```
memory run
```
Calls `run_pipeline(config, console, consolidate=True)`. Full pipeline with auto-consolidation and optional LLM context.

### 2.3 `run-light` (line 55)

```
memory run-light
```
Calls `run_pipeline(config, console, consolidate=False)`. Skips auto-consolidation; context generation is always deterministic.

### 2.4 `rebuild-graph` (line 62)

```
memory rebuild-graph
```
1. `rebuild_from_md(config.memory_path)` — scans all entity MDs, rebuilds graph
2. `save_graph()` — atomic write

### 2.5 `rebuild-faiss` (line 75)

```
memory rebuild-faiss
```
1. `build_index(config.memory_path, config)` — full FAISS rebuild from all entity MDs

### 2.6 `rebuild-all` (line 88)

```
memory rebuild-all
```
1. `rebuild_from_md()` → graph
2. `recalculate_all_scores()` → rescored graph
3. **Repair**: fixes entities with empty `mention_dates` by backfilling from `created` or `last_mentioned` (lines 105-114)
4. `save_graph()`
5. `write_index()` → `_index.md`
6. `build_context()` → `_context.md` (deterministic only)
7. `build_index()` → FAISS

**Note**: `rebuild-all` always uses deterministic context, never LLM per-section, regardless of config.

### 2.7 `validate` (line 136)

```
memory validate
```
Loads graph, runs `validate_graph()`, prints warnings about orphan relations and missing files.

### 2.8 `stats` (line 154)

```
memory stats
```
Rich table showing: entity counts (graph vs files), relations, pending chats, type breakdown, status of `_context.md` / `_index.md` / FAISS.

### 2.9 `inbox` (line 196)

```
memory inbox
```
Calls `process_inbox(memory_path, config)`. Returns list of processed filenames. Prints hint to run `memory run` afterward.

### 2.10 `clean` (line 211)

```
memory clean [--all] [--artifacts] [--dry-run]
```
**Targets**:
- `--artifacts`: `_context.md`, `_index.md`, `_graph.json` (+`.bak`, `.lock`), FAISS files, ingest jobs
- `--all`: adds `__pycache__` dirs and `_inbox/_processed/`

**Safety**: Creates timestamped tarball backup in `backups/` before deletion (lines 271-285).

### 2.11 `serve` (line 302)

```
memory serve [--transport stdio|sse]
```
Starts MCP server. Transport from CLI arg or `config.mcp_transport`.

### 2.12 `replay` (line 319)

```
memory replay [--list]
```
Replays failed extractions from the retry ledger.

**Flow**:
1. `list_retriable(config)` — gets pending entries from `_retry_ledger.json` (line 331)
2. `--list`: displays table of failures (file, error, attempts, recorded) (lines 336-349)
3. Without `--list`: replays each entry through the pipeline:
   - `extract_from_chat()` + `sanitize_extraction()` (lines 373-374)
   - `resolve_all()` (line 378) — **Note**: no arbitration step for ambiguous entities (see Findings)
   - `enrich_memory()` (line 380)
   - `mark_chat_processed()` (line 381)
   - `mark_replayed(success=True)` (line 382)
4. On failure: `mark_replayed(success=False, error=str(e))` — after 3 total attempts, status becomes `"exhausted"` (line 386)

### 2.13 `consolidate` (line 392)

```
memory consolidate [--dry-run] [--facts] [--min-facts N]
```
Two modes:
- `--facts`: delegates to `consolidate_facts()` in orchestrator (LLM-based fact consolidation)
- Default: name-based duplicate entity detection using `title.lower()` + aliases (lines 405-436). Groups entities sharing a name/alias. Dry-run only preview.

### 2.14 `dream` (line 438)

```
memory dream [--dry-run] [--step N]
```
Delegates to `run_dream(config, console, dry_run, step)`. Prints dream report metrics.

### 2.15 `context` (line 475)

```
memory context
```
1. Loads graph
2. `recalculate_all_scores()` — rescores all entities
3. `save_graph()` — persists rescored graph
4. `build_context()` → `_context.md` (always deterministic)
5. `write_index()` → `_index.md`

### 2.16 `graph` (line 496)

```
memory graph
```
Opens interactive graph visualization in browser via `open_graph()`.

---

## 3. Document Ingest Pipeline

**File**: `src/pipeline/doc_ingest.py`

### 3.1 `_normalize_text(text: str) -> str` (line 17)

**Purpose**: Prepare document text for chunking.
- Strips YAML frontmatter (lines 25-28)
- Normalizes 3+ blank lines to 2 (line 31)
- Strips trailing whitespace per line (lines 33-35)

### 3.2 `ingest_document(source_id, content, ingest_key, memory_path, config) -> dict` (line 38)

**Signature**: `(source_id: str, content: str, ingest_key: IngestKey, memory_path: Path, config: Config) -> dict`

**Flow**:
1. **Normalize**: `_normalize_text(content)` (line 54)
2. **Chunk**: `_chunk_text(normalized, chunk_size, chunk_overlap)` — reuses indexer's chunking (line 59)
3. **Embed**: `_get_embedding_fn(config)` → `embed_fn(chunks)` (lines 64-65)
4. **Load/create FAISS index**: reads existing or creates new `IndexFlatIP` (lines 68-79)
5. **Upsert guard**: if document already indexed with same content hash, skips (lines 82-87)
6. **Add vectors**: `index.add(embeddings)` (line 91)
7. **Update chunk mapping**: stores `file`, `entity_id` (prefixed `doc:`), `chunk_idx`, `chunk_text` (200-char preview) (lines 93-99)
8. **Update manifest**: records hash, chunk count, vector IDs, source type, chunk policy version (lines 102-113)
9. **Save**: writes FAISS index, pickle mapping, JSON manifest (lines 115-119)
10. Records metric and returns `{"chunks_indexed": N, "source_id": ...}` (lines 121-124)

**Key detail**: Document entities get `entity_id = f"doc:{source_id}"` — these are NOT graph entities. They exist only in FAISS for RAG search.

---

## 4. Inbox Pipeline

**File**: `src/pipeline/inbox.py`

### 4.1 `process_inbox(memory_path: Path, config: Config) -> list[str]` (line 25)

**Purpose**: Process files dropped into `_inbox/`. Routes to conversation or document pipeline.

**Accepted file types**: `.md`, `.txt`, `.json` (line 50)

**Flow per file**:
1. **JSON files** (line 55):
   - `split_export_json(filepath, memory_path)` from chat_splitter (line 57)
   - If no conversations found: falls back to document pipeline (lines 59-64)
   - Legacy fallback if `features.doc_pipeline` disabled (line 63-64)
2. **MD/TXT files** (lines 66-76):
   - If `features.doc_pipeline` disabled: `_legacy_ingest()` — wraps as pseudo-chat (lines 71-73)
   - If enabled: `_routed_ingest()` — deterministic routing (line 76)
3. **Post-processing**: moves file to `_inbox/_processed/` with timestamp prefix (lines 79-80)
4. Errors are logged but do not halt processing of other files (lines 83-84)

### 4.2 `_legacy_ingest(content, memory_path) -> None` (line 89)

Wraps content as `[{"role": "user", "content": content}]` and calls `save_chat()`.

### 4.3 `_routed_ingest(filename, content, memory_path, config) -> None` (line 95)

**Flow**:
1. `compute_ingest_key(filename, content)` (line 99)
2. **Idempotency guard**: `has_been_ingested(key, config)` (line 102)
3. `classify(content, source_filename=filename)` — deterministic routing (line 109)
4. **Conversation route**: `save_chat()` as unprocessed chat (lines 118-121)
5. **Document/uncertain route**: creates ingest job, transitions to running, calls `ingest_document()` (lines 123-143)
6. On failure: transitions job to `"failed"`, re-raises exception (lines 140-143)

---

## 5. Chat Splitter

**File**: `src/pipeline/chat_splitter.py`

### 5.1 `split_export_json(filepath: Path, memory_path: Path) -> list[Path]` (line 19)

**Purpose**: Detect JSON export format and split into individual chat files.

**Detection order** (line 27-31):
1. `_parse_claude_export(data)` — Claude.ai format
2. `_parse_chatgpt_export(data)` — ChatGPT format
3. `_parse_generic_json_array(data)` — simple `[{role, content}]`

Uses short-circuit OR: first parser that returns non-None wins.

**Per conversation** (lines 39-53):
1. `save_chat(messages, memory_path)` — creates unprocessed chat file
2. If metadata available (`date` or `title`): `_patch_chat_frontmatter()` adds `source: import`, `source_title`, `date`

### 5.2 `_parse_claude_export(data) -> list[dict] | None` (line 57)

**Detects**:
- Top-level array where first item has `chat_messages` or `uuid` (lines 68-71)
- Object with `conversations`, `chats`, or `data` key containing similar items (lines 73-79)

**Message mapping**:
- `sender` field → role: `"human"/"user"/"User"` → `"user"`, else `"assistant"` (lines 97-104)
- Text from `text` or `content` field (line 98)
- Date from `created_at` or `created` field, ISO-parsed (lines 109-116)
- Title from `name` or `title` field (line 108)

### 5.3 `_parse_chatgpt_export(data) -> list[dict] | None` (line 127)

**Detects**: Array where first item has `mapping` key (lines 134-137)

**Message extraction**:
- Flattens mapping tree (not a linear walk — iterates all nodes) (lines 153-168)
- Filters to `role in ("user", "assistant")` (lines 160-161)
- Text from `content.parts` array, joined with newlines (line 164)
- Sorted by `create_time` (line 172)
- Date from conversation-level `create_time` (epoch timestamp) (lines 177-184)

### 5.4 `_parse_generic_json_array(data) -> list[dict] | None` (line 195)

**Detects**: Array where first item has `role` and `content` keys (line 201)
**Output**: Single conversation with all messages.

### 5.5 `_patch_chat_frontmatter(path, conv) -> None` (line 213)

Updates existing chat frontmatter with `date`, `source_title`, and `source: "import"`.

---

## 6. Ingest State Machine

**File**: `src/pipeline/ingest_state.py`

### 6.1 Retry Ledger (lines 20-83)

Separate from the ingest job state machine. Tracks extraction failures for `memory replay`.

**Storage**: `_retry_ledger.json` in `memory_path` (line 22)

#### `record_failure(chat_path, error, config) -> None` (line 44)
- Appends entry with `status: "pending"`, `attempts: 1`, timestamps
- Dedup by file path — won't add if same file already pending (line 48)

#### `list_retriable(config) -> list[dict]` (line 63)
- Returns entries with `status == "pending"`

#### `mark_replayed(chat_path, success, config, error=None) -> None` (line 68)
- Increments `attempts`, updates `last_attempt`
- Success: `status = "succeeded"` (line 76)
- Failure with `attempts >= 3`: `status = "exhausted"` (lines 77-79)
- Otherwise: stays `"pending"` for future retry

### 6.2 Ingest Key (lines 86-93)

#### `compute_ingest_key(source_id, content) -> IngestKey` (line 86)
- SHA-256 of content → `content_hash`
- Bundles with `source_id` and `CHUNK_POLICY_VERSION` constant

### 6.3 Job State Machine (lines 96-206)

**Storage**: `_ingest_jobs.json` at `config.ingest.jobs_path`

**States and valid transitions** (lines 158-164):
```
pending → running
running → succeeded | failed | retriable
retriable → running
failed → (terminal)
succeeded → (terminal)
```

#### `has_been_ingested(key, config) -> bool` (line 115)
- Scans all jobs for matching canonical key with `status == "succeeded"`
- Canonical: `"{source_id}::{content_hash}::{chunk_policy_version}"`

#### `create_job(key, config, route=None) -> IngestJob` (line 127)
- UUID job ID, status `"pending"`, respects `config.ingest.max_retries`
- Persists immediately to jobs file

#### `transition_job(job_id, new_status, config, error=None, chunks_indexed=None) -> IngestJob | None` (line 147)
- Validates transition against allowed state map (line 166)
- **Auto-escalation**: if transitioning to `"retriable"` and retries >= `max_retries`, auto-escalates to `"failed"` (lines 179-183)
- Returns `None` if job not found or transition invalid

#### `get_job(job_id, config) -> IngestJob | None` (line 190)
Simple lookup by ID.

#### `list_jobs(config, status=None) -> list[IngestJob]` (line 198)
List all jobs, optionally filtered.

#### `recover_stale_jobs(config) -> list[str]` (line 208)
- Finds jobs stuck in `"running"` beyond `config.ingest.recovery_threshold_seconds`
- Transitions them to `"retriable"` (with auto-escalation to `"failed"` if max retries exceeded)
- Returns list of recovered job IDs

**Note**: `recover_stale_jobs()` is defined but never called from any CLI command or pipeline. It requires explicit integration (see Findings).

---

## 7. Router

**File**: `src/pipeline/router.py`

### `classify(text, source_filename=None) -> RouteDecision` (line 84)

**Purpose**: Deterministic heuristic classification. Zero LLM calls.

**Signals scored**:
- **Conversation**: speaker-turn markers (`User:`, `Assistant:`, etc.), timestamp-speaker patterns, `"role"` JSON markers, dialogue density
- **Document**: markdown headings, long paragraphs (>200 chars), bullet/numbered lists, single-author style

**Decision thresholds**:
- `conv_score >= 0.4` and `conv_score > doc_score` → `"conversation"` (line 161)
- `doc_score >= 0.3` and `doc_score > conv_score` → `"document"` (line 165)
- Otherwise → `"uncertain"` (line 170)

**Special case**: JSON chat array (`[{role, content}]`) → immediately returns `"conversation"` with 0.98 confidence (lines 97-102)

**Note**: The `source_filename` parameter is accepted but never used in the classification logic (line 84). Potential for filename-based heuristics (e.g., `.chat.json` vs `.doc.md`) is untapped.

---

## 8. Full Pipeline Chain — End to End

### `memory run` chain

```
CLI cli() → load_config() + init_memory_structure()
  → run_pipeline(consolidate=True)
    → list_unprocessed_chats() [capped at job_max_chats_per_run]
    Per chat:
      → get_chat_content()
      → extract_from_chat() [LLM, stall-aware streaming]
      → sanitize_extraction() [deterministic cleanup]
      → load_graph()
      → resolve_all() [deterministic: slug/alias/FAISS match]
      → arbitrate_entity() [LLM, ambiguous only]
      → enrich_memory() [MD write + graph + ACT-R scoring]
      → mark_chat_processed()
    Post-loop:
      → discover_batch_relations() [deterministic, FAISS + tags]
      → auto_consolidate() [LLM fact consolidation]
      → build_context() or build_context_with_llm() [context generation]
      → incremental_update() [FAISS indexing]
```

### `memory run-light` chain

Same as above but `consolidate=False`:
- Skips `auto_consolidate()`
- Always uses deterministic `build_context()` (never LLM per-section)

### `memory inbox` chain

```
CLI cli() → process_inbox()
  Per file in _inbox/:
    JSON → split_export_json() → save_chat() per conversation
    MD/TXT:
      doc_pipeline enabled → classify() → conversation|document|uncertain
        conversation → save_chat()
        document/uncertain → create_job() → ingest_document() [FAISS only]
      doc_pipeline disabled → _legacy_ingest() → save_chat()
    → move to _processed/
```

### `memory replay` chain

```
CLI cli() → list_retriable()
  Per entry:
    → extract_from_chat() [LLM]
    → sanitize_extraction()
    → resolve_all() [deterministic]
    → enrich_memory() [NO arbitration step]
    → mark_chat_processed()
    → mark_replayed()
```

---

## 9. Where `sanitize_extraction()` Is Called

`sanitize_extraction()` is defined in `src/pipeline/extractor.py:42` and called in exactly two places:

1. **`run_pipeline()`** in `src/pipeline/orchestrator.py:288` — during the main `memory run` / `run-light` flow, immediately after `extract_from_chat()`
2. **`replay()`** in `src/cli.py:374` — during replay of failed extractions, immediately after `extract_from_chat()`

**What it sanitizes** (extractor.py:42-131):
- Null summary → empty string (line 54)
- Empty entity names → dropped (line 63)
- Invalid entity types → `"interest"` (line 70)
- Empty observation content → dropped (line 78)
- Invalid observation categories → `"fact"` (line 83)
- Importance clamped to [0.0, 1.0] (line 89)
- Null valence/date/supersedes/tags → defaults (lines 92-99)
- Empty relation from/to names → dropped (line 114)
- Invalid relation types → fuzzy-mapped via `_RELATION_FALLBACK` dict, default `"linked_to"` (lines 118-121)
- Null relation context → empty string (line 125)

---

## 10. Correction Interception Points

Places where corrections or transformations could be inserted:

1. **Post-extraction, pre-sanitization** (`orchestrator.py:287-288`): Between `extract_from_chat()` and `sanitize_extraction()` — could inspect raw LLM output before cleanup
2. **Post-sanitization, pre-resolution** (`orchestrator.py:288-308`): After sanitization but before entity resolution — could override entity names/types
3. **Post-resolution, pre-arbitration** (`orchestrator.py:308-311`): After deterministic matching but before LLM disambiguation — could force resolution decisions
4. **Post-arbitration, pre-enrichment** (`orchestrator.py:338-342`): After all entities resolved — could filter/modify before writing to disk
5. **In `sanitize_extraction()` itself** (`extractor.py:42-131`): The `_RELATION_FALLBACK` dict and type mappings are hardcoded — adding entries changes behavior
6. **In `discover_batch_relations()`** (`orchestrator.py:134`): The tag overlap threshold (2 shared tags, FAISS >= 0.8) could be parameterized
7. **In `auto_consolidate()`** (`orchestrator.py:172`): The fact count threshold comes from `config.get_max_facts()` — configurable per entity type

---

## 11. Error Handling & Recovery Mechanisms

### Extraction Failures
- **Retry tracking**: `increment_extraction_retries()` in chat frontmatter (orchestrator.py:292)
- **Timeout detection**: `is_timeout_error()` checks exception type and message (orchestrator.py:19-31)
- **Fallback**: After timeout or 2 retries → `fallback_to_doc_ingest()` (orchestrator.py:298)
- **Retry ledger**: `record_failure()` enables future `memory replay` (orchestrator.py:300-301)

### Enrichment Failures
- Chat is NOT marked processed → will be retried on next `memory run` (orchestrator.py:349-351)
- No retry counter for enrichment — can loop indefinitely if enrichment consistently fails

### Batch Relation / Consolidation / Context / FAISS
- All wrapped in try/except with warning output (orchestrator.py:357-396)
- Non-fatal: pipeline completes even if post-loop steps fail

### Ingest Job State Machine
- `recover_stale_jobs()` available but never called automatically
- Jobs stuck in `"running"` stay there until manual recovery
- `transition_job()` validates state transitions, rejects invalid ones

### Replay Mechanism
- `mark_replayed()` tracks attempts; after 3 attempts → `"exhausted"` (ingest_state.py:77)
- Exhausted entries are permanently skipped by `list_retriable()`

### Graph Atomicity
- `save_graph()` uses lockfile + temp file + `os.replace()` (referenced in code, actual impl in `graph.py`)
- Corruption recovery: `load_graph()` tries `.bak` file, then `rebuild_from_md()`

---

## 12. Findings & Issues

### F1: Missing job state transition on doc_ingest failure (orchestrator.py:78-79)

In `fallback_to_doc_ingest()`, when `ingest_document()` raises an exception (line 78), the job is never transitioned to `"failed"`. It was transitioned to `"running"` at line 71 and stays there permanently. The `mark_chat_fallback()` at line 82 still runs (always-block), so the chat is marked processed, but the ingest job is orphaned in `"running"` state.

**Fix**: Add `transition_job(job.job_id, "failed", config, error=str(e2))` in the except block.

### F2: Replay skips arbitration (cli.py:362-388)

The `replay()` command runs `extract_from_chat()` → `sanitize_extraction()` → `resolve_all()` → `enrich_memory()` but never calls `arbitrate_entity()` for ambiguous resolutions. Ambiguous entities go to enrichment with `status="ambiguous"`, which may cause unexpected behavior in the enricher.

**Impact**: Entities that would need LLM disambiguation during normal `run` are silently handled differently during `replay`.

### F3: Enrichment failure leaves chat unprocessed with no retry cap (orchestrator.py:349-351)

If `enrich_memory()` fails, the chat is not marked processed. On the next `memory run`, it will be re-extracted (burning LLM tokens) and fail again at enrichment. There is no retry counter for enrichment failures, unlike extraction which has `EXTRACTION_MAX_RETRIES = 2`.

### F4: `recover_stale_jobs()` never called (ingest_state.py:208)

The function exists to recover jobs stuck in `"running"` state beyond a timeout threshold, but it is never called from any CLI command or automated pipeline. Stale jobs accumulate silently.

**Fix**: Call at start of `process_inbox()` or add a CLI command.

### F5: `auto_consolidate()` `min_facts` parameter is unused (orchestrator.py:155)

The function signature accepts `min_facts: int = 8` but the actual comparison uses `config.get_max_facts(entity.type)` (line 168-172). The `min_facts` parameter has no effect.

### F6: Step numbering gap in `run_pipeline()` (orchestrator.py:370)

Comments jump from "Step 5b" to "Step 7", skipping Step 6. This is cosmetic but confusing.

### F7: `source_filename` parameter unused in router (router.py:84)

`classify()` accepts `source_filename` but never uses it in the heuristic logic. Filename-based signals (e.g., detecting `.chat.json` vs `.notes.md`) could improve routing accuracy.

### F8: Graph reloaded 3 times in post-loop (orchestrator.py:358,366,372)

After the per-chat loop, the graph is loaded from disk 3 times:
- Line 358 for batch relation discovery
- Line 366 for auto-consolidation (inside `auto_consolidate()` at line 160)
- Line 372 for context generation

Each reload reads `_graph.json` from disk. The graph modified by `discover_batch_relations()` is saved (line 146) then immediately reloaded by `auto_consolidate()`.

### F9: `consolidate` CLI command without `--facts` is view-only (cli.py:405-436)

Running `memory consolidate` (without `--facts` and without `--dry-run`) detects and PRINTS duplicates but does not merge them. There is no non-dry-run merge path in the default mode. The absence of `--dry-run` has no additional effect over including it — both just print.

### F10: Chat timestamp collision handling (store.py:266-269)

`save_chat()` uses minute-resolution timestamps (`%Y-%m-%d_%Hh%M`) with a counter suffix for collisions. When importing large JSON exports via `split_export_json()`, many chats can be created in the same minute, leading to sequential suffixes (`_1`, `_2`, ..., `_N`). This works but produces non-informative filenames. The original conversation title is only stored in frontmatter metadata, not the filename.
