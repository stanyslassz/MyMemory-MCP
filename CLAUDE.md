# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

memory-ai is a personal persistent memory system for LLMs. It extracts structured knowledge (entities, relations, observations) from chat conversations, stores them as Markdown files, scores them with ACT-R cognitive models, and serves them via MCP (Model Context Protocol). Local-first, Markdown-based, French-language domain by default (`user_language: fr`).

## Commands

```bash
# Install / sync
uv sync --extra dev

# Run CLI (global flags: -v/--verbose, -c/--config <path>)
uv run memory run              # Process pending chats through full pipeline (+ auto-consolidation + LLM context if enabled)
uv run memory run-light        # Same as run but skips auto-consolidation, deterministic context only
uv run memory serve            # Start MCP server (stdio default)
uv run memory stats            # Display memory metrics
uv run memory rebuild-all      # Rebuild graph + scores + context + FAISS
uv run memory rebuild-graph    # Rebuild _graph.json from MD files
uv run memory rebuild-faiss    # Full FAISS index rebuild
uv run memory validate         # Check graph consistency
uv run memory inbox            # Process files in _inbox/
uv run memory replay           # Retry failed extractions (--list to preview)
uv run memory consolidate      # Detect duplicate entities (--dry-run to preview)
uv run memory clean            # Remove generated artifacts (--dry-run, --artifacts, --all)
uv run memory replay --list    # Show failed extractions
uv run memory context           # Rebuild _context.md (no extraction, no LLM)
uv run memory dream            # Brain-like memory reorganization (--dry-run, --step N, --resume, --reset, --report)
uv run memory graph            # Open interactive graph visualization in browser
uv run memory search "query"   # Search memory via RAG (--top-k N, --expand/--no-expand)
uv run memory actions          # Show centralized action history (--last N, --entity, --action)
uv run memory relations        # Discover new relations (--entity, --dry-run)
uv run memory insights         # Show ACT-R cognitive insights (--format text|json)
uv run memory health           # Memory health analysis (--format text|json)

# Tests (no conftest.py — tests are standalone, no shared fixtures)
uv run pytest tests/ -v                    # All tests
uv run pytest tests/test_graph.py -v       # Single test file
uv run pytest tests/test_graph.py::test_name -v  # Single test
```

## Source Layout

```
src/                    # Package root (imports: from src.core.config import ...)
  cli.py                # Click CLI — thin wrappers over orchestrator
  core/                 # Config, LLM abstraction, models, utils, action_log, event_log, metrics
  pipeline/             # Extraction, resolution, enrichment, indexing, inbox, router, visualize
    dream/                # Dream mode package (split from monolithic dream.py)
      __init__.py         # Re-exports: run_dream, DreamReport, decide_dream_steps
      coordinator.py      # run_dream(), decide_dream_steps(), checkpoints, validation, reporting
      consolidator.py     # Fact consolidation, summaries, document extraction steps
      merger.py           # Entity merging, FAISS dedup candidates
      discovery.py        # Relation discovery, transitive relations
      maintenance.py      # Load, prune, rebuild steps
    dashboard.py        # Interactive HTML dashboard (graph, timeline, dream, search views)
    dashboard_server.py # Micro HTTP server for dashboard with entity/relation API
    keyword_index.py    # SQLite FTS5 keyword index for hybrid search
    nlp.py              # Optional spaCy NLP (NER, dedup, date extraction)
  memory/               # Graph, scoring, store (MD I/O), mentions, insights (+ analyze_memory_health())
    rag.py              # Unified RAG search facade (FAISS + FTS5 + RRF + ACT-R reranking)
    context/            # Context generation package: builder, formatter, utilities
  mcp/                  # FastMCP server (stdio)
prompts/                # LLM prompt templates (Markdown with {variables})
tests/                  # Standalone tests (no conftest.py, no shared fixtures)
memory/                 # Runtime data directory (gitignored entity MDs, chats, FAISS)
mcp_stdio.py            # Standalone MCP launcher for Claude Desktop
scripts/                # Utility scripts
  benchmark_scoring.py  # ACT-R scoring benchmark (correctness + performance regression)
config.yaml.example     # Template for config.yaml (gitignored)
.github/workflows/ci.yml # GitHub Actions CI (pytest on push/PR)
```

Entry point: `memory = "src.cli:cli"` (pyproject.toml). All imports use `src.` prefix (e.g. `from src.core.config import load_config`).

## Architecture

### Full Pipeline Chain (`memory run`)

```
Chat text (from chats/ with processed: false)
  → Extractor (LLM, stall-aware streaming)
  → Resolver (deterministic slug/alias/FAISS match)
  → Arbitrator (LLM, ambiguous entities only)
  → Enricher (MD + graph write + ACT-R scoring)
  → Batch relation discovery (deterministic FAISS + tag overlap, zero LLM)
  → Context builder (deterministic template, zero LLM)
  → FAISS indexer (incremental)
```

**Entry point**: `cli.py:run()` → loads unprocessed chats (capped at `job_max_chats_per_run`), runs pipeline per chat, then rebuilds context + FAISS once.

#### Step 1 — Extraction (`pipeline/extractor.py` → `core/llm.py`)

- `extract_from_chat(chat_content, config) → RawExtraction`
- If content > 70% of `context_window`: splits into overlapping segments, calls LLM per segment, merges via `_merge_extractions()` (dedup by slug/relation tuple)
- LLM call: `call_extraction()` uses **stall-aware streaming** — watchdog thread monitors token progress, raises `StallError` if no tokens for `timeout` seconds
- Prompt: `prompts/extract_facts.md` — injects `{chat_content}`, `{user_language}`, `{categories_observations}`, `{categories_entity_types}`, `{categories_relation_types}`
- Output: `RawExtraction(entities: list[RawEntity], relations: list[RawRelation], summary: str)`
- **Post-extraction sanitization**: `sanitize_extraction()` fixes invalid types from small LLMs — fuzzy maps invented relation types (e.g. `prescrit_par` → `linked_to`), falls back invalid entity types to `interest`, invalid observation categories to `fact`, clamps importance to [0,1], drops empty refs, coerces `None` → defaults. Called in both `_run_pipeline()` and `replay()` before `resolve_all()`.
- **Fallback on timeout**: if extraction fails or retries >= 2, falls back to `doc_ingest` (chunk + FAISS index raw content). Failures recorded in `_retry_ledger.json`.

#### Step 2 — Resolution (`pipeline/resolver.py`)

- `resolve_all(raw_extraction, graph, faiss_search_fn) → ResolvedExtraction`
- Per entity: `resolve_entity(name, graph, faiss_search_fn) → Resolution`
- Resolution order: exact slug match → alias containment → FAISS similarity (threshold 0.75, context-enriched query) → new
- **Context-aware FAISS**: query enriched with first observation's category + content prefix for disambiguation
- **Zero LLM tokens** — purely deterministic

#### Step 3 — Arbitration (`pipeline/arbitrator.py` → `core/llm.py`)

- Only for entities with `status="ambiguous"` (FAISS found multiple candidates)
- `arbitrate_entity(name, context, candidates, graph, config) → EntityResolution`
- LLM call: `call_arbitration()` uses `_call_structured()` (no stall detection, shorter call)
- Prompt: `prompts/arbitrate_entity.md`
- Output: `EntityResolution(action: "existing"|"new", existing_id?, new_type?)`

#### Step 4 — Enrichment (`pipeline/enricher.py`)

- `enrich_memory(resolved, config, today) → EnrichmentReport`
- **Existing entities**: reads MD, appends new observations (dedup check), bumps `frequency`, updates `mention_dates` (windowed), running-average `importance`
- **New entities**: creates MD file in folder mapped by `config.get_folder_for_type()`, adds to graph
- **Relations**: resolves slugs, creates stub entities for forward references, calls `add_relation()` with Hebbian reinforcement
- **Scoring**: `recalculate_all_scores(graph, config, today)` — ACT-R + spreading activation
- **Persist**: saves `_graph.json` (atomic write with `.bak` backup and lockfile) + `_index.md`

#### Step 5 — Context Generation (`memory/context/builder.py`)

- `build_context(graph, memory_path, config) → str` (alias: `build_deterministic_context`)
- **Zero LLM** — deterministic template + token budgeting
- Template: `prompts/context_template.md` with variables: `{date}`, `{user_language_name}`, `{ai_personality}`, `{sections}`, `{available_entities}`, `{custom_instructions}`
- Gets top 50 entities above `min_score_for_context`, categorizes into sections:
  - **AI Personality**: type=ai_self
  - **Identity**: entities in `self/` folder
  - **Work**: type in (work, organization)
  - **Personal**: type in (person, animal, place)
  - **Top of Mind**: remaining high-score entities
  - **Vigilances**: scans shown entities for [vigilance]/[diagnosis]/[treatment] facts
  - **Active Memory Management**: pending corrections queued via `suggest_correction` MCP tool
  - **Brief History**: split by recency (recent/earlier/longterm)
- Each section has a token budget from `config.context_budget` (percentage of `context_max_tokens`)
- Entity dossiers enriched via `_enrich_entity()`: reads facts from MD, collects BFS depth-1 relations, sorts facts chronologically
- **Natural mode** (alternative, `context_format: natural`): generates context via LLM per-section calls instead

#### Step 6 — FAISS Indexing (`pipeline/indexer.py`)

- `incremental_update(memory_path, config)` — compares file hashes against manifest, rebuilds if changed
- `build_index()`: reads all entity MDs → chunks (400 tokens, 80 overlap) → embeds → `faiss.IndexFlatIP` (cosine similarity on normalized vectors)
- `search(query, config, memory_path, top_k) → list[SearchResult]`

### 3-Level Memory

| Level | Storage | Purpose |
|-------|---------|---------|
| **L1** | `_context.md` | Active memories injected into every conversation |
| **L2** | FAISS index | Searchable via RAG, promotes back to L1 via mention bump |
| **L3** | Markdown files | Full knowledge base, source of truth |

**L2→L1 Re-emergence**: `search_rag()` in MCP server bumps `mention_dates` for retrieved entities → ACT-R score rises → entity naturally re-enters context on next rebuild. Rate-limited by `mention_bump_cooldown_days` (default 3) to prevent positive feedback loops. Uses RRF hybrid re-ranking (semantic + keyword + ACT-R) when FTS5 index exists, otherwise linear re-ranking (60% vector + 40% ACT-R).

### RAG Search Pipeline (`memory/rag.py`)

9-step pipeline: FAISS search (top_k×2 candidates) → FTS5 keyword search → RRF fusion (`rrf_score = w_sem/(k+sem_rank) + w_kw/(k+kw_rank) + w_actr/(k+actr_rank)`, defaults: w_sem=0.5, w_kw=0.3, w_actr=0.2, k=60) → entity dedup (best chunk per entity) → GraphRAG expansion (top 3 with depth-1 neighbors) → relations enrichment → mention bump (L2→L1 with cooldown) → threshold + top_k limit. Fallback: if FTS5 unavailable uses linear reranking (60% FAISS + 40% ACT-R).

### MCP Server (`mcp/server.py`)

8 tools exposed via FastMCP:

| Tool | Input | Logic |
|------|-------|-------|
| `get_context()` | None | Returns `_context.md` (fallback: `_index.md`) |
| `save_chat(messages)` | list[dict] | Saves to `chats/` with `processed: false` |
| `search_rag(query)` | str | FAISS + FTS5 keyword search → RRF hybrid re-ranking (semantic + keyword + ACT-R) → L2→L1 bump |
| `delete_fact(entity_name, fact_content)` | str, str | Find entity → match fact by content → remove from MD → history entry |
| `delete_relation(from_entity, to_entity, relation_type)` | str, str, str | Resolve entities → remove from graph + MD → save graph |
| `modify_fact(entity_name, old_content, new_content)` | str, str, str | Find fact → replace content preserving metadata → history entry |
| `correct_entity(entity_name, field, new_value)` | str, str, str | Update entity metadata (title, type, aliases, retention) → move file if type changed |
| `suggest_correction(entity_name, field, suggested_value, reason)` | str, str, str, str | Queue a correction suggestion for user review → stored as pending correction → shown in "Pending Corrections" context section |

`mcp_stdio.py` (project root, not under `src/`) — Standalone launcher for Claude Desktop (injects sys.path, sets cwd, no pip install needed).

**Thread safety**: Per-entity locks (`_entity_locks: dict[str, threading.Lock]`) prevent concurrent CRUD race conditions on the same entity file.

### Fallback Pipeline (`pipeline/doc_ingest.py`)

When extraction fails: `ingest_document()` normalizes text → chunks → embeds → adds to FAISS directly (no entity creation). State tracked in `_ingest_jobs.json` via `pipeline/ingest_state.py` state machine (pending → running → succeeded/failed/retriable).

### Inbox Pipeline (`pipeline/inbox.py`)

`process_inbox()` handles files dropped in `memory/_inbox/`:
- Routes via `classify()` (deterministic heuristics, no LLM)
- Conversation → saves as unprocessed chat
- Document → `ingest_document()` with idempotency guard (`IngestKey` = source_id + content_hash)

### Dream Mode (`pipeline/dream/`)

Primarily reorganizes existing knowledge. Step 2 may extract entities from previously unprocessed RAG documents. `memory dream` runs a 10-step pipeline:

**Coordinator**: Deterministic coordinator (`decide_dream_steps()`) plans which steps to run based on memory stats and `analyze_memory_health()` output for step prioritization. Critical steps validated deterministically (`validate_dream_step()`).

1. **Load** — `load_graph()` + scan entity MDs
2. **Extract docs** (LLM) — scan FAISS manifest for unextracted RAG documents → `extract_from_chat()` + full pipeline
3. **Fact consolidation** (LLM) — entities with 8+ facts → `call_fact_consolidation()`
4. **Entity merging** (deterministic + LLM) — slug/alias overlap (deterministic) + FAISS similarity with LLM confirmation via `call_dedup_check()` (confidence ≥ `dedup_confidence_threshold`)
5. **Relation discovery** (FAISS + LLM) — FAISS top-5 similar → `call_relation_discovery()` validates new relations
6. **Transitive relations** (deterministic) — infer relations via transitive rules, capped at 20. Inferred strength = `min(strength_AB, strength_BC) × 0.5`. Rules: affects→affects=affects, part_of→part_of=part_of, requires→requires=requires, improves→affects=improves, worsens→affects=worsens, uses→part_of=uses
7. **Dead pruning** (deterministic) — archive entities where `score < 0.1`, `freq ≤ 1`, `age > 90d`, no relations → `_archive/`
8. **Summary generation** (LLM) — entities without summary → `call_entity_summary()`
9. **Rescore** — `recalculate_all_scores()`
10. **Rebuild** — `build_context()` + `build_index()`

**Validation**: `validate_dream_step()` checks post-conditions (e.g., step 3 must not increase fact count, step 4 must not increase entity count, step 5 adding >50 relations is suspicious). Failed validation triggers rollback (graph snapshot restored).
**Checkpointing**: `_save_checkpoint()` after each successful step; `--resume` continues from last checkpoint (`_dream_checkpoint.json`); `--reset` clears checkpoint.
**Event logging**: Emits 6 event types to `_event_log.jsonl` via `append_event()`: `dream_session_start`, `dream_step_start`, `dream_step_done`, `dream_step_failed`, `dream_step_skipped`, `dream_session_end`. Each event includes `dream_id` for session correlation.
**Persist-per-iteration**: Steps 4 (merge), 5 (relation discovery), and 8 (summary) call `save_graph()` after each individual merge/relation/summary to ensure zero progress loss on interruption.
**Dream report**: `_dream_report.md` generated after each run (`--report` flag or automatic). Summarizes steps executed, entities affected, relations created, and errors encountered.
**Dashboard**: Rich Live terminal UI (`dream_dashboard.py`) shows real-time step progress (pending/running/done/failed/skipped). Dream tab groups events by `dream_id`, displays session header with timestamps, and shows a details grid per step.
**LLM config**: Optional `llm.dream` in `config.yaml`. Falls back to `llm.context` via `config.llm_dream_effective`.
**Prompts**: `discover_relations.md`.

### Inbox JSON Import (`pipeline/chat_splitter.py`)

`process_inbox()` now accepts `.json` files. Multi-conversation exports are auto-split:
- **Claude.ai**: `chat_messages` with `sender: human/assistant`
- **ChatGPT**: `mapping` tree with `author.role` and `content.parts`
- **Generic**: `[{role, content}]` arrays
Each conversation → individual chat file with `processed: false`. Metadata preserved in frontmatter (`source: import`, `source_title`).

## Scoring Algorithm (ACT-R + Spreading Activation)

### ACT-R Base-Level Activation

```
B = ln(Σ t_j^(-d))
```
- `t_j` = days since each mention (minimum 0.5)
- `d` = `decay_factor` (0.5 for long_term, 0.8 for short_term)
- Uses both `mention_dates` (recent, high-resolution) and `monthly_buckets` (old, aggregated)

### Final Score

```
score = sigmoid(B + importance × importance_weight + spreading_weight × S + emotional_boost)
```
- `S` = spreading activation bonus (see below)
- `emotional_boost` = `negative_valence_ratio × emotional_boost_weight` (amygdala modulation)
- Permanent retention entities: `score >= permanent_min_score` (floor 0.5)
- **Retrieval threshold**: if `score < retrieval_threshold` (default 0.05) and not permanent → score = 0.0 (true forgetting, ACT-R retrieval failure)

### Spreading Activation (Two-Pass)

**Pass 1**: Compute base scores for all entities: `base[e] = sigmoid(B + importance × weight)`

**Pass 2**: For each relation, compute power-law time-decayed effective strength:
```
eff_strength = rel.strength × (days_since + 0.5)^(-relation_decay_power)
```
Build bidirectional adjacency, then for each entity:
```
spreading_bonus = Σ(eff_strength_i × base_score_neighbor_i) / total_strength
```

### Hebbian Learning + LTD

`add_relation()` in `graph.py` — when two entities co-occur (LTP, multiplicative Hebbian):
```
mention_count += 1
activity_factor = max(0.2, mean(from_score, to_score))
delta = relation_strength_growth × (1.0 - strength) × activity_factor
strength = min(1.0, strength + delta)   # default growth +0.05, self-limiting
last_reinforced = now
```
Growth is self-limiting (slows as strength→1.0) and modulated by entity activity (scores of both endpoints).

`_apply_ltd()` in `scoring.py` — during `recalculate_all_scores()` (LTD):
```
onset_factor = min(1.0, days / ltd_onset_days)    # linear fade-in 0→1 over 90 days
decay = exp(-onset_factor × days / relation_ltd_halflife)  # halflife 360 days
strength = max(min_relation_strength, strength × decay)    # floor 0.1
```
Combined with power-law decay in spreading: "neurons that fire together wire together", unused connections actively weaken.

### Emotional Modulation

`negative_valence_ratio` computed during `rebuild_from_md()` from entity facts:
- Facts with `[-]` valence, or categories `vigilance`/`diagnosis`/`treatment` count as "emotional"
- Ratio = emotional_facts / total_facts → boosts activation via `emotional_boost_weight` (default 0.15)
- Models amygdala-hippocampal consolidation: emotional memories persist longer

### Mention Windowing (`memory/mentions.py`)

- `add_mention()`: appends date to `mention_dates`
- When list exceeds `window_size` (default 50): `consolidate_window()` moves oldest dates into `monthly_buckets` (YYYY-MM → count)
- Both fed into `calculate_actr_base()` for scoring

## Scoring Details

### Retention Upgrade Rules (`_upgrade_retention()` in `scoring.py`)

| Entity type | Condition | Upgrade to |
|---|---|---|
| `ai_self` | Always | `permanent` |
| `person`, `animal` | `frequency >= 3` | `long_term` |
| `health` | `frequency >= 2` | `long_term` |
| Any (already `short_term`) | `frequency >= 10` AND age > 30 days | `long_term` |

Never downgrades retention.

### Supersession Mechanism

Observations and relations support a `supersedes` field:
- **Observations**: `supersedes: "ski à Toulouse"` — marks that this fact replaces a previous one
- **Relations**: `supersedes: "user:louise:linked_to"` — format: `from_slug:to_slug:old_relation_type`
- The enricher checks for supersession and handles fact/relation replacement, not just addition.

### Fact TTL (Time-to-Live)

`config.yaml` → `fact_ttl` section controls how long facts appear in L1 context:
- TTL is per observation category (e.g., `emotion: 30` days, `context: 60` days, `fact: 0` = never expires)
- Facts remain in MD files and RAG — TTL **only** affects `_context.md`
- Filtered via `filter_live_facts()` in `core/utils.py`

## Data Models (`core/models.py`)

### Closed Literal Types (must sync with `categories` in `config.yaml`)

| Type | Values |
|------|--------|
| `ObservationCategory` | fact, preference, diagnosis, treatment, progression, technique, vigilance, decision, emotion, interpersonal, skill, project, context, rule, ai_style, user_reaction, interaction_rule |
| `EntityType` | person, health, work, project, interest, place, animal, organization, ai_self |
| `RelationType` | affects, improves, worsens, requires, linked_to, lives_with, works_at, parent_of, friend_of, uses, part_of, contrasts_with, precedes |

### Key Models

- **RawObservation**: `category`, `content`, `importance` (0-1), `tags`, `date` (optional ISO), `valence` ("positive"|"negative"|"neutral"|"")
- **RawEntity**: `name`, `type`, `observations`
- **RawRelation**: `from_name`, `to_name`, `type`, `context`
- **RawExtraction**: `entities`, `relations`, `summary`
- **Resolution**: `status` ("resolved"|"new"|"ambiguous"), `entity_id`, `candidates`, `suggested_slug`
- **GraphEntity**: `file`, `type`, `title`, `score`, `importance`, `frequency`, `last_mentioned`, `retention`, `aliases`, `tags`, `mention_dates`, `monthly_buckets`, `created`, `summary`, `negative_valence_ratio`
- **GraphRelation**: `from_entity` (alias "from"), `to_entity` (alias "to"), `type`, `strength` (0.5), `created`, `last_reinforced`, `mention_count` (1), `context` — uses `Field(alias="from")` for JSON serialization (Python reserved word workaround)
- **GraphData**: `generated`, `entities` (dict[str, GraphEntity]), `relations` (list[GraphRelation]), `_adjacency` (PrivateAttr, lazy bidirectional index via `get_adjacency()` → `dict[str, list[GraphRelation]]`, invalidated by `invalidate_adjacency()` on relation mutations, O(1) lookups instead of O(R) scans)
- **EntityFrontmatter**: YAML frontmatter schema for MD files — mirrors GraphEntity fields
- **SearchResult**: `entity_id`, `file`, `chunk`, `score`, `relations`
- **DreamPlan**: `steps` (list[int]), `reasoning` (str) — deterministic coordinator output
- **DreamValidation**: `approved` (bool), `issues` (list[str]) — deterministic post-step validation

## Entity File Format (`memory/store.py`)

### Structure
```markdown
---
title: Entity Name
type: health
retention: long_term
score: 0.72
importance: 0.85
frequency: 12
last_mentioned: "2026-03-07"
created: "2025-09-15"
aliases: [back pain, sciatica]
tags: [health, chronic]
mention_dates: ["2026-03-01", "2026-03-07"]
monthly_buckets: {"2025-06": 3, "2025-09": 5}
summary: "Chronic back issue with sciatica."
---
## Facts
- [diagnosis] (2024-03) Chronic sciatica [-]
- [treatment] (2025-11) Started physiotherapy [+]
- [fact] Regular monitoring needed

## Relations
- affects [[Daily routine]]
- improves [[Swimming]]

## History
- 2025-09-15: Created
```

### Observation Format
```
- [category] (date) content [valence] #tag1 #tag2
```
- Date: optional, YYYY-MM or YYYY-MM-DD
- Valence: `[+]` positive, `[-]` negative, `[~]` neutral (optional)
- Tags: optional, prefixed with `#`

### Key Functions
- `_format_observation(obs: dict) → str` — dict to markdown line
- `_parse_observation(line: str) → dict | None` — markdown line to dict
- `_is_duplicate_observation()` — matches on category + content (ignores date/valence)
- `read_entity(path) → (EntityFrontmatter, sections_dict)`
- `write_entity(path, frontmatter, sections)` — sections order: Facts, Relations, History
- `update_entity(path, new_observations, new_relations, ...) → EntityFrontmatter` — dedup, bump frequency
- `create_entity(memory_path, folder, slug, frontmatter, observations) → Path`
- `create_stub_entity()` — for forward references (default: short_term, importance 0.3)

## Configuration (`core/config.py`)

### Dataclass Hierarchy
- **Config** — master, loaded from `config.yaml` + `.env`
  - **LLMStepConfig** — per-step: `model`, `temperature`, `max_retries`, `timeout`, `api_base`, `context_window`
  - **ScoringConfig** — ACT-R params: `decay_factor` (0.5), `decay_factor_short_term` (0.8), `importance_weight` (0.3), `spreading_weight` (0.2), `permanent_min_score` (0.5), `relation_strength_base` (0.5), `relation_decay_halflife` (180), `relation_strength_growth` (0.05), `relation_ltd_halflife` (360), `relation_decay_power` (0.3), `retrieval_threshold` (0.05), `emotional_boost_weight` (0.15), `window_size` (50), `min_score_for_context` (0.3), `batch_relation_threshold` (0.8), `relation_discovery_threshold` (0.75), `relation_discovery_type_threshold` (0.80)
  - **EmbeddingsConfig**: `provider`, `model`, `chunk_size` (400), `chunk_overlap` (80)
  - **FAISSConfig**: `index_path`, `mapping_path`, `manifest_path`, `top_k` (5)
  - **CategoriesConfig**: `observations`, `entity_types`, `relation_types`, `folders`
  - **NLPConfig**: `enabled`, `model`, `dedup_threshold`, `date_extraction`, `pre_ner`
  - **SearchConfig**: `hybrid_enabled`, `fts_db_path`, `rrf_k`, `weight_semantic`, `weight_keyword`, `weight_actr`
  - **FeaturesConfig**: `doc_pipeline` (bool)
  - **IngestConfig**: `recovery_threshold_seconds`, `max_retries`, `jobs_path`
  - **DreamConfig**: `faiss_merge_threshold` (0.80), `faiss_merge_max_candidates` (20), `dossier_max_facts` (3), `transitive_min_strength` (0.4), `transitive_max_new` (20), `dedup_confidence_threshold` (0.7)
  - **ExtractionConfig**: `prompt_overhead` (1500), `split_threshold` (0.7), `segment_ratio` (0.5), `overlap_tokens` (200)
  - `context_format` — `structured` | `natural` (determines context generation mode)
  - `context_llm_sections` — bool (enables LLM per-section context generation)
  - `max_facts` — per entity type limit on stored facts

### Key Methods
- `load_config(config_path, project_root) → Config` — loads `.env` first, then YAML, resolves all paths relative to project_root
- `config.user_language_name` — maps "fr" → "French", "en" → "English"
- `config.get_folder_for_type(entity_type) → str` — folder from `categories.folders`, defaults to "interests"

## LLM Abstraction (`core/llm.py`)

### Prompt Loading
- `load_prompt(name, config, **variables) → str` — loads `prompts/{name}.md`, auto-injects `{user_language}`, `{categories_*}`, then applies `**variables` via `str.replace()` (not f-strings, to avoid JSON brace conflicts)

### Two Call Modes
1. **`_call_structured(step_config, prompt, response_model)`** — standard Instructor call, for short steps (arbitration, summary)
2. **`_call_with_stall_detection(step_config, prompt, response_model, stall_timeout)`** — worker thread streams tokens, watchdog checks every 2s, raises `StallError` if no progress for `stall_timeout` seconds. Used for extraction (long calls). Timeout set to `config.timeout × 3` to allow slow-but-progressing responses.

### Instructor MD_JSON Mode
Extracts JSON from markdown code blocks — avoids `response_format` param incompatible with some local models (Ollama, LM Studio).

### Thinking Model Support
`strip_thinking(text)` removes `<think>...</think>` tags from reasoning models (Qwen3, DeepSeek-R1). These models may need higher timeouts as `<think>` tokens aren't visible to Instructor streaming.

## Graph Management (`memory/graph.py`)

- `load_graph()` — loads `_graph.json` with corruption recovery (tries `.bak`, then `rebuild_from_md()`)
- `save_graph()` — atomic write with lockfile (`_graph.lock`, 5-minute timeout) and `.bak` backup
- `add_relation()` — dedup by (from, to, type) tuple; reinforces existing (Hebbian) or creates new
- `find_entity_by_name(name, graph)` — resolve entity name to entity_id (slug → title → alias match)
- `get_related(graph, entity_id, depth=1)` — BFS traversal, bidirectional
- `rebuild_from_md(memory_path)` — scans all entity MDs, rebuilds graph from frontmatter + relation sections
- `validate_graph()` — returns warnings for missing files or orphan relations

## Data Files (all gitignored)

| File | Purpose |
|------|---------|
| `memory/` | Source of truth. Subfolders: self, close_ones, projects, work, interests, chats, _inbox |
| `memory/chats/` | Chat transcripts with YAML frontmatter (`processed: true/false`) |
| `memory/_inbox/` | Drop zone for file ingestion |
| `memory/_archive/` | Pruned entities (dream mode, reversible) |
| `_graph.json` | Entity/relation index derived from MD files |
| `_context.md` | Token-budgeted context served by MCP `get_context()` |
| `_index.md` | Visual entity list (fallback for context) |
| `_retry_ledger.json` | Failed extraction tracking for `memory replay` |
| `_ingest_jobs.json` | Document ingest state machine |
| `_memory.faiss` / `_memory.pkl` | FAISS vector index + chunk mapping |
| `_faiss_manifest.json` | Index manifest (hashes, model, timestamp) |
| `_graph.html` | Interactive graph visualization (generated, gitignored) |
| `_dream_report.md` | Post-dream run report (generated after each `memory dream`) |
| `_dream_checkpoint.json` | Dream session checkpoint for `--resume` |
| `_event_log.jsonl` | Structured event log (dream sessions, step progress, errors) |
| `_memory_fts.db` | SQLite FTS5 keyword index |
| `_graph.lock` | Graph lockfile (atomic creation, 5-min stale timeout) |

## Prompt Files

| File | Used By | Variables |
|------|---------|-----------|
| `extract_facts.md` | Step 1 extraction | `{chat_content}`, `{user_language}`, `{categories_*}` |
| `arbitrate_entity.md` | Step 3 arbitration | `{entity_name}`, `{entity_context}`, `{candidates}`, `{json_schema}` |
| `context_template.md` | Step 5 context | `{date}`, `{user_language_name}`, `{ai_personality}`, `{sections}`, `{available_entities}`, `{custom_instructions}` |
| `context_instructions.md` | Step 5 context | None (user-editable custom rules) |
| `summarize_entity.md` | Entity summary | `{entity_title}`, `{entity_type}`, `{entity_facts}`, `{entity_relations}`, `{entity_tags}` |
| `consolidate_facts.md` | Fact consolidation | `{entity_title}`, `{entity_type}`, `{facts_text}` |
| `context_section.md` | LLM per-section context | `{section_name}`, `{entities_dossier}`, `{rag_context}`, `{budget_tokens}` |
| `discover_relations.md` | Dream step 5 | `{entity_a_title}`, `{entity_a_type}`, `{entity_a_dossier}`, `{entity_b_*}` |
| `context_natural_section.md` | Natural context generation | Used by `call_natural_context_section()` for natural context per-section |
| `context_natural.md` | Natural context format | Template for natural context format |
| `dedup_check.md` | Duplicate entity check | Used by `call_dedup_check()` for FAISS-based duplicate entity verification |

## Key Conventions

- **Language convention**: All code, constants, keys, section names, and labels are in **English**. The `user_language` config only controls what the LLM generates for the user. The context template stays in English — the receiving LLM translates via `{user_language_name}`.
- **Closed category lists** in `models.py` must stay in sync with `categories` in `config.yaml`. Adding a new observation category, entity type, or relation type requires updating both.
- **Folder mapping**: Entity type → subfolder configured in `config.yaml` under `categories.folders` (e.g., person → close_ones, health → self, ai_self → self).
- **Config** is `config.yaml` (gitignored, copy from `config.yaml.example`). Auth via `.env` (e.g., `OPENAI_API_KEY=lm-studio` for LM Studio).
- **Context LLM per-section mode**: `context_llm_sections: true` in config.yaml enables `build_context_with_llm()` — generates each section (ai_personality, identity, work, personal, top_of_mind) via LLM with RAG pre-fetch, while vigilances stay deterministic. Used by `memory run` (not `run-light`).
- **Context natural mode**: `context_format: natural` switches from deterministic template to LLM-generated prose per section.
- **Context dispatch helper**: `build_context_for_config(graph, memory_path, config, use_llm=True)` in `memory/context/builder.py` is the single entry point for all 4 context modes (structured/natural × deterministic/llm). All CLI, orchestrator, and dream call sites use this helper.
- **Path traversal guard**: `_enrich_entity()` in context.py validates entity file paths stay within `memory_path` via `is_relative_to()`.
- **Graph atomicity**: `save_graph()` uses temp file + `os.replace()` + lockfile to prevent corruption.
- **Centralized utilities**: `slugify()`, `estimate_tokens()`, `parse_frontmatter()`, `atomic_write_text()`, `filter_live_facts()`, `is_entity_file()` live in `core/utils.py` — import from there, not from individual modules.
- **Auto-consolidation**: `memory run` auto-consolidates entities with 8+ facts via LLM (`auto_consolidate()` in `pipeline/orchestrator.py`). `run-light` skips this.
- **Pipeline orchestration**: Business logic lives in `pipeline/orchestrator.py` (extracted from cli.py). CLI is thin Click wrappers only.
- **Observation guard-rails**: max 150 chars post-consolidation in `store.py`, max 3 tags per fact.
- **JSON repair**: `_repaired_json()` context manager in `llm.py` patches `json.loads` to auto-repair malformed JSON from small models via `json-repair` library. Wraps both `_call_structured()` and `_call_with_stall_detection()`. Restores original `json.loads` during repair to avoid recursion (json-repair internally calls json.loads).
- **Directory initialization**: `init_memory_structure()` called on every CLI startup — all folders (`_inbox/`, `_archive/`, entity subfolders) created automatically.
- Python 3.11+, managed with `uv`, built with `hatchling`.

## Unused / Orphan Code

None currently tracked. Last audit: 2026-03-19 v3 (full codebase — dream/ package split, adjacency index, multiplicative Hebbian, memory health, bidirectional corrections, CI).

## Known Issues

None currently tracked.
