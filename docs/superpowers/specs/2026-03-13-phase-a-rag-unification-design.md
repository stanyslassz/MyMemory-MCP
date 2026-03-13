# Phase A — RAG Unification + Config Externalization

**Date**: 2026-03-13
**Scope**: Workstream 1 (unified RAG facade) + Workstream 6 (externalize hardcoded constants)
**Approach**: Facade pattern — orchestrate existing modules without deep rewrites

## 1. Unified RAG Facade (`src/memory/rag.py`)

### Responsibility

Single entry point for all memory search. Orchestrates FAISS, FTS5, RRF merge, ACT-R reranking, entity deduplication, GraphRAG expansion, and L2→L1 mention bump.

### Public Interface

```python
from dataclasses import dataclass, field
from pathlib import Path
from src.core.config import Config
from src.core.models import SearchResult

@dataclass
class SearchOptions:
    top_k: int = 5
    expand_relations: bool = False       # GraphRAG depth-1
    expand_max: int = 10                 # cap total after expansion
    include_chunk_text: bool = True      # actual chunk text in results
    deduplicate_entities: bool = True    # one result per entity
    use_fts5: bool = True               # hybrid keyword search
    rerank_actr: bool = True            # ACT-R score reranking
    bump_mentions: bool = False          # L2→L1 mention bump
    bump_rate_limit: bool = True         # max 1 bump per entity per day
    threshold: float = 0.0              # minimum score to return
    context_for_extraction: bool = False # compact mode for extraction context
    # Note: RRF weights come from config.search (weight_semantic, weight_keyword, weight_actr)
    # Linear fallback weights come from config.search (linear_faiss_weight, linear_actr_weight)
    # No duplication of weight config in SearchOptions — single source of truth in Config

def search(
    query: str,
    config: Config,
    memory_path: Path,
    options: SearchOptions | None = None,
) -> list[SearchResult]:
    """Single entry point for all memory search."""
```

### Internal Pipeline (9 steps)

1. **FAISS search** — `indexer.search(query, config, memory_path, top_k=options.top_k * 2)` (over-fetch for RRF merge headroom). Note: `indexer.search()` already deduplicates by entity internally, so the over-fetch provides headroom for RRF re-ranking rather than dedup. Step 5 below handles the case where FTS5 introduces duplicate entities not caught by the indexer's internal dedup.
2. **FTS5 search** (if `use_fts5` and `_memory_fts.db` exists) — `keyword_index.search_keyword(query, top_k * 2)`. Skip silently if unavailable.
3. **RRF merge** (if FTS5 active) — `score = w1/(k+rank_faiss) + w2/(k+rank_fts5)`, k = `config.search.rrf_k`, weights from `config.search.weight_semantic/weight_keyword/weight_actr`. If no FTS5: FAISS results only, with optional linear fallback (`config.search.linear_faiss_weight` × FAISS + `config.search.linear_actr_weight` × ACT-R) when ACT-R reranking is active but FTS5 is not.
4. **ACT-R reranking** (if `rerank_actr`) — load graph, add `w3/(k+rank_actr)` to RRF score. Skip silently if graph unavailable.
5. **Entity deduplication** (if `deduplicate_entities`) — keep best chunk per `entity_id`. This catches cross-source duplicates (same entity found by both FAISS and FTS5 after RRF merge).
6. **GraphRAG expansion** (if `expand_relations`) — for top-3 results, `get_related(graph, entity_id, depth=1)`. For each unseen neighbor: `score = result.score * effective_strength * 0.5`. Re-sort descending, cap at `expand_max`.
7. **Relations enrichment** — attach directional relations to each result.
8. **Mention bump** (if `bump_mentions`) — for each result in graph: rate-limit check (today already in `mention_dates`?), if not: `add_mention()` + update MD frontmatter + save graph. Lightweight: bump only, no `recalculate_all_scores()` — score recalculation happens during `memory run`. Emit `search_performed` event.
9. **Final filter** — apply `threshold`, truncate to `top_k`, return.

### Error handling

- If FAISS index does not exist: return empty results (no auto-build). Callers should ensure index exists via `indexer.incremental_update()` or `memory rebuild-faiss`.
- If graph loading fails: skip ACT-R reranking, GraphRAG expansion, and mention bump silently. Return FAISS-only results.
- All exceptions from individual pipeline steps are caught and logged. The pipeline degrades gracefully rather than failing entirely.

### Logic extracted from `server.py`

- `_rrf_fusion()` (lines 118-156) → moves to `rag.py`
- L2→L1 bump logic (lines 303-326) → moves to `rag.py`
- Relations enrichment (lines 264-301) → moves to `rag.py`
- `server.py::search_rag()` becomes a thin wrapper calling `rag.search()` + formatting

### What stays unchanged

- `indexer.search()` — remains as internal FAISS component
- `keyword_index.search_keyword()` — remains as internal FTS5 component
- `graph.get_related()` — remains as internal graph traversal

## 2. Consumer Rebranchement

### Migration table

| Consumer | File | Before | After |
|----------|------|--------|-------|
| MCP search_rag | `server.py` | 170 lines inline | `rag.search(query, config, path, SearchOptions(expand_relations=True, bump_mentions=True))` + format |
| Batch relations | `orchestrator.py` | `make_faiss_fn()` wrapper | `rag.search(title, config, path, SearchOptions(top_k=5, bump_mentions=False))` |
| Entity resolver | `resolver.py` | `faiss_search_fn(query, top_k=3, threshold=0.75)` → `list[dict]` | `rag.search(query, config, path, SearchOptions(top_k=3, use_fts5=False, rerank_actr=False, threshold=config.search.resolver_threshold, deduplicate_entities=False))` → `list[SearchResult]` |
| Dream merge | `dream/merger.py` | `indexer.search(title, config, path, top_k=5)` | `rag.search(title, config, path, SearchOptions(top_k=max_candidates, threshold=merge_threshold, bump_mentions=False))` |
| Dream discovery | `dream/discovery.py` | `indexer.search(title, config, path, top_k=5)` | `rag.search(title, config, path, SearchOptions(top_k=5, bump_mentions=False))` |

### Resolver signature change

```python
# Before
def resolve_entity(name, graph, faiss_search_fn) -> Resolution
def resolve_all(raw_extraction, graph, faiss_search_fn) -> ResolvedExtraction

# After
def resolve_entity(name, graph, config, memory_path) -> Resolution
def resolve_all(raw_extraction, graph, config, memory_path) -> ResolvedExtraction
```

The resolver imports `rag.search` directly and works with `SearchResult` instead of `dict`. Internal code changes from dict key access (`s["entity_id"]`, `s["score"]`) to attribute access (`s.entity_id`, `s.score`). Callers of `resolve_all()` pass `config, memory_path` instead of `make_faiss_fn(config, memory_path)`.

**Note on threshold**: The resolver hardcodes `threshold=0.75` when calling the search function. `make_faiss_fn()` has a default of `0.85` but this is dead code — the resolver always passes `0.75` explicitly. The canonical value is `0.75`, externalized as `config.search.resolver_threshold`.

### Deletion

- `make_faiss_fn()` removed from `orchestrator.py` (lines 34-50)

## 3. Sentence-Aware Chunking

### spaCy integration in `indexer.py`

**Module-level cache** (shared with future NLP features):

```python
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
            import logging
            logging.getLogger(__name__).info(
                "spaCy model %s not found, downloading...", model_name
            )
            spacy.cli.download(model_name)
            nlp = spacy.load(model_name)
        _spacy_nlp_cache[language] = nlp
        return nlp
    except ImportError:
        return None
    except Exception:
        # Network failure during download, air-gapped env, etc.
        return None
```

**Modified `chunk_text()`**:

```python
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80,
               language: str = "en") -> list[str]:
```

- Attempts `_get_spacy_nlp(language)` → segment via `doc.sents`
- If spaCy unavailable → fallback `re.split(r'(?<=[.!?])\s+', text)`
- Accumulates sentences until `chunk_size` tokens reached, overlap by trailing sentences

**Callers**: `build_index()` passes `language=config.user_language` to `chunk_text()`.

### Dependency

```toml
# pyproject.toml
[project.optional-dependencies]
nlp = ["spacy>=3.5"]
```

### config.yaml.example update

```yaml
# Language for LLM-generated content and NLP processing
# Supported: fr, en, de, es, it, pt, nl, zh
# This determines: LLM output language, spaCy model for sentence tokenization and NER,
# token estimation ratio (1.5 for fr/de/es/it, 1.3 for en)
user_language: fr
```

## 4. Config Externalization

All new fields have default values matching the current hardcoded constants. Zero behavior change.

### ScoringConfig — 2 additions

```python
ltd_onset_days: int = 90            # threshold in _apply_ltd()
min_relation_strength: float = 0.1  # floor in _apply_ltd()
```

### SearchConfig — 2 additions

```python
resolver_threshold: float = 0.75    # in resolver.py (canonical value, make_faiss_fn's 0.85 default was dead code)
linear_faiss_weight: float = 0.6    # fallback linear reranking in rag.py (moved from server.py)
linear_actr_weight: float = 0.4     # fallback linear reranking in rag.py (moved from server.py)
```

### DreamConfig — 5 additions

```python
prune_score_threshold: float = 0.1       # step 7 in dream.py
prune_min_age_days: int = 90             # step 7 minimum age
prune_max_frequency: int = 1             # step 7 max frequency
transitive_min_strength: float = 0.4     # step 6 in discovery.py (_step_transitive_relations)
transitive_max_new: int = 20             # step 6 in discovery.py (_step_transitive_relations)
```

Note: `discovery_*` renamed to `transitive_*` — these constants are used in `_step_transitive_relations()` (step 6), not `_step_discover_relations()` (step 5).

Existing fields (`faiss_merge_threshold`, `faiss_merge_max_candidates`, `dossier_max_facts`) unchanged.

### ContextConfig — 1 addition

```python
history_recent_days: int = 30       # in formatter.py
```

### Module replacements

| Module | Hardcoded | Replaced by |
|--------|-----------|-------------|
| `scoring.py::_apply_ltd()` | `90` | `config.scoring.ltd_onset_days` |
| `scoring.py::_apply_ltd()` | `0.1` | `config.scoring.min_relation_strength` |
| `resolver.py` | `0.75` | `config.search.resolver_threshold` (via SearchOptions) |
| `rag.py` linear fallback | `0.6`, `0.4` | `config.search.linear_faiss_weight`, `linear_actr_weight` |
| `dream.py` step 7 | `0.1`, `90`, `1` | `config.dream.prune_score_threshold`, `prune_min_age_days`, `prune_max_frequency` |
| `discovery.py` step 6 | `0.4`, `20` | `config.dream.transitive_min_strength`, `transitive_max_new` |
| `formatter.py` | `30` | `config.ctx.history_recent_days` |

## 5. Implementation Order

```
1. Config externalization (WS6)        — independent, zero risk
2. spaCy chunking (indexer.py)         — independent, zero risk
3. rag.py facade                       — core piece
4. Consumer rebranchement              — depends on rag.py
5. Delete make_faiss_fn()              — after all consumers migrated
6. Tests + validation
```

Steps 1 and 2 can be parallelized (different files).

## 6. Files Touched

| File | Change |
|------|--------|
| `src/memory/rag.py` | **NEW** — search facade |
| `src/core/config.py` | Add fields to ScoringConfig, SearchConfig, DreamConfig, ContextConfig |
| `config.yaml.example` | Document new fields + user_language comment |
| `src/pipeline/indexer.py` | spaCy chunking + cache + `language` param |
| `src/mcp/server.py` | Extract search logic → thin wrapper over rag.search() |
| `src/pipeline/resolver.py` | Signature change: `faiss_search_fn` → `config, memory_path` |
| `src/pipeline/orchestrator.py` | Delete `make_faiss_fn()`, use rag.search() |
| `src/pipeline/dream.py` | Hardcoded constants → config fields |
| `src/pipeline/dream/merger.py` | `indexer.search` → `rag.search` |
| `src/pipeline/dream/discovery.py` | `indexer.search` → `rag.search` |
| `src/memory/scoring.py` | `90` / `0.1` → config fields |
| `src/memory/context/formatter.py` | `30` → config field |
| `pyproject.toml` | Extra `nlp = ["spacy>=3.5"]` |
| `tests/test_rag.py` | **NEW** — rag.search() tests |

### Unchanged

`graph.py`, `store.py`, `enricher.py`, `llm.py`, `models.py`, `event_log.py`, `mentions.py`

## 7. Test Strategy

- `tests/test_rag.py` (new): search() with various SearchOptions combos, mocked FAISS + FTS5 + graph
- Chunking tests: spaCy installed + model available, spaCy + missing model (auto-download mocked), spaCy absent (ImportError mocked), network failure during download (fallback to regex), cache verification
- **Resolver migration tests**: update existing resolver tests to use new signature (`config, memory_path` instead of `faiss_search_fn`), verify attribute access on `SearchResult` works correctly
- Existing tests: must all pass unchanged (default config values = old constants)
- No conftest.py — standalone tests per project convention
