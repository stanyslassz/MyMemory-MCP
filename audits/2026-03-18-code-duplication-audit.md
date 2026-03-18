# Code Duplication Audit — 2026-03-18

**Scope**: Full `src/`, `tests/`, `prompts/` directories
**Method**: Automated exploration (3 parallel agents) + manual line-number verification
**Files analyzed**: 43 source files, 17 test files, 8 prompt files

---

## Summary

| Category | Findings | Estimated LOC Savings |
|----------|----------|-----------------------|
| Path filtering logic | 4 identical copies | ~12 |
| Test factory helpers (`_make_config`) | 17 copies | ~200 |
| Test entity factories | 7+ variants | ~120 |
| LLM call wrappers | 2 near-identical functions | ~30 |
| Chat frontmatter update cycle | 3 copies | ~25 |
| JSONL log modules | 2 parallel implementations | ~40 |
| Keyword index rebuild | 2 exact copies | ~8 |
| Superseded-fact filtering | 15+ copies of same comprehension | ~15 |
| MCP entity-lookup boilerplate | 4 identical blocks | ~16 |
| Context `section_budget()` | 2 identical nested functions | ~6 |
| Entity enrichment functions | 3 variants | ~80 |
| Prompt "Respond ONLY with JSON" | 5 copies | — (prompt text) |
| Template loading + fallback | 2 copies | ~10 |
| **Total** | | **~560** |

---

## Findings

### 1. Path Filtering Logic (4 copies)

**Importance: 8/10** — A change to the filter rule (e.g. adding a new excluded prefix) requires updating 4 files.

**Locations** (exact same expression):
- `src/memory/graph.py:217`
- `src/memory/store.py:190`
- `src/pipeline/indexer.py:246`
- `src/pipeline/keyword_index.py:42`

**Duplicated code**:
```python
if any(p.startswith("_") for p in parts) or (parts and parts[0] == "chats"):
    continue
```

**Remediation**: Extract to `src/core/utils.py`:
```python
def is_entity_file(rel_path: Path) -> bool:
    """Return True if rel_path points to an entity MD (not chats, _archive, etc.)."""
    parts = rel_path.parts
    return not (any(p.startswith("_") for p in parts) or (parts and parts[0] == "chats"))
```
Replace all 4 call sites with `if not is_entity_file(rel): continue`.

**Effort**: 15 min

---

### 2. Test `_make_config()` Factories (17 copies)

**Importance: 7/10** — Every Config API change must be replicated across 17 test files.

**Locations** (one per file):
`test_cli.py:19`, `test_context.py:12`, `test_doc_ingest.py:17`, `test_dream_events.py:16`, `test_dream_report.py:15`, `test_e2e.py:97`, `test_indexer.py:16`, `test_ingest_invariants.py:26`, `test_mcp_smoke.py:11`, `test_mcp_tools.py:73`, `test_mcp_transport.py:22`, `test_rag.py:13`, `test_robustness.py:23`, `test_rrf.py:22`, `test_scoring.py:13`, `test_sprint_p0p1.py:99`, `test_timeout_fallback.py:21`

**Three implementation styles**:
1. `Config.__new__(Config)` + manual attribute assignment (most files)
2. `load_config(project_root=...)` + overrides (`test_e2e.py`)
3. `ScoringConfig(**overrides)` only (`test_scoring.py`)

**Remediation**: Create `tests/helpers.py` (no conftest.py per project convention):
```python
def make_test_config(tmp_path: Path, **overrides) -> Config:
    """Minimal valid Config for tests. Override any field via kwargs."""
    config = Config.__new__(Config)
    config.memory_path = tmp_path
    config.scoring = ScoringConfig()
    config.context_max_tokens = 3000
    config.context_budget = {"identity": 10, "top_of_mind": 25, ...}
    config.prompts_path = Path(__file__).parent.parent / "prompts"
    config.faiss = FAISSConfig(
        index_path=str(tmp_path / "_memory.faiss"),
        mapping_path=str(tmp_path / "_memory.pkl"),
        manifest_path=str(tmp_path / "_faiss_manifest.json"),
    )
    for k, v in overrides.items():
        setattr(config, k, v)
    return config
```
Each test file: `from tests.helpers import make_test_config`.

**Effort**: 1-2 hours (touch all 17 files)

---

### 3. Test Entity Factories (7+ variants)

**Importance: 6/10** — Inconsistent defaults make tests harder to reason about.

**Locations**:
- `tests/test_graph.py:21` — `_make_entity(title, type_, score, aliases) -> GraphEntity`
- `tests/test_mcp_tools.py:28` — `_make_entity(tmp_path, slug, ...) -> Path` (writes to disk)
- `tests/test_natural_context.py:30` — `_make_entity(**kwargs) -> GraphEntity`
- `tests/test_insights.py:7` — `_make_entity(title, type_, score, negative_valence_ratio)`
- `tests/test_resolver.py:17` — `_make_graph()` (inline GraphEntity construction)
- `tests/test_rag.py:30` — `_make_graph()` (inline construction)
- `tests/test_rrf.py:9` — `_make_graph(entities_dict)`

**Remediation**: Add to `tests/helpers.py`:
```python
def make_graph_entity(**kwargs) -> GraphEntity:
    defaults = dict(file="self/test.md", type="health", title="Test",
                    score=0.5, importance=0.7, frequency=3, ...)
    defaults.update(kwargs)
    return GraphEntity(**defaults)
```
Keep disk-writing variant in `test_mcp_tools.py` since it serves a different purpose.

**Effort**: 1 hour

---

### 4. LLM Call Wrappers — `call_context_section` / `call_natural_context_section` (~95% identical)

**Importance: 7/10** — Near-identical 30-line functions that differ only in prompt filename.

**Locations**:
- `src/core/llm.py:295` — `call_context_section()`
- `src/core/llm.py:328` — `call_natural_context_section()`

**Both functions do**:
1. `load_prompt(PROMPT_NAME, config, section_name=..., entities_dossier=..., ...)`
2. Build `kwargs` dict with model/messages/temperature/timeout/api_base
3. `litellm.completion(**kwargs)`
4. `strip_thinking(text).strip()`

**Only difference**: prompt name (`"context_section"` vs `"context_natural_section"`).

**Remediation**:
```python
def _call_context_llm(prompt_name: str, section_name: str, entities_dossier: str,
                      rag_context: str, budget_tokens: int, config: Config) -> str:
    prompt = load_prompt(prompt_name, config, section_name=section_name,
                         entities_dossier=entities_dossier,
                         rag_context=rag_context or "No additional context available.",
                         budget_tokens=str(budget_tokens))
    step_config = config.llm_context
    kwargs: dict[str, Any] = {"model": step_config.model,
                               "messages": [{"role": "user", "content": prompt}],
                               "temperature": step_config.temperature}
    if step_config.timeout: kwargs["timeout"] = step_config.timeout
    if step_config.api_base: kwargs["api_base"] = step_config.api_base
    response = litellm.completion(**kwargs)
    return strip_thinking(response.choices[0].message.content or "").strip()

def call_context_section(...): return _call_context_llm("context_section", ...)
def call_natural_context_section(...): return _call_context_llm("context_natural_section", ...)
```

**Effort**: 20 min

---

### 5. Chat Frontmatter Update Cycle (3 copies)

**Importance: 6/10** — Same read-parse-modify-write pattern repeated.

**Locations**:
- `src/memory/store.py:365` — `mark_chat_processed()`
- `src/memory/store.py:383` — `mark_chat_fallback()`
- `src/memory/store.py:407` — `increment_extraction_retries()`

**Repeated pattern** (5 lines each):
```python
text = filepath.read_text(encoding="utf-8")
fm_data, body = _shared_parse_frontmatter(text)
# ... modify fm_data ...
fm_yaml = yaml.safe_dump(fm_data, default_flow_style=False, allow_unicode=True)
_atomic_write_text(filepath, f"---\n{fm_yaml}---\n{body}")
```

**Remediation**:
```python
def _update_chat_frontmatter(filepath: Path, updater: Callable[[dict], None]) -> dict:
    text = filepath.read_text(encoding="utf-8")
    fm_data, body = _shared_parse_frontmatter(text)
    updater(fm_data)
    fm_yaml = yaml.safe_dump(fm_data, default_flow_style=False, allow_unicode=True)
    _atomic_write_text(filepath, f"---\n{fm_yaml}---\n{body}")
    return fm_data
```

**Effort**: 20 min

---

### 6. JSONL Log Modules — Parallel Implementations

**Importance: 5/10** — Two modules with identical append+read patterns but no shared base.

**Locations**:
- `src/core/action_log.py` (53 lines) — `log_action()` + `read_actions()`
- `src/memory/event_log.py` (96 lines) — `append_event()` + `read_events()`

**Shared logic**:
- JSON serialize → append newline to `.jsonl` file
- Read lines → parse JSON → optional filter → return last N

**Differences**: `event_log.py` adds `threading.Lock`; field names differ.

**Remediation**: Create `src/core/jsonl_store.py`:
```python
class JSONLStore:
    def __init__(self, filename: str, thread_safe: bool = False): ...
    def append(self, memory_path: Path, **fields): ...
    def read(self, memory_path: Path, last: int = 0, **filters) -> list[dict]: ...
```
Then `action_log = JSONLStore("_actions.jsonl")` and `event_log = JSONLStore("_event_log.jsonl", thread_safe=True)`.

**Effort**: 45 min

---

### 7. Keyword Index Rebuild — Exact Duplicate

**Importance: 8/10** — Identical 4-line block copy-pasted within the same file.

**Locations**:
- `src/pipeline/indexer.py:329-332` (inside `build_index()`)
- `src/pipeline/indexer.py:356-359` (inside `incremental_update()`)

**Exact duplicate**:
```python
from src.pipeline.keyword_index import build_keyword_index
fts_db_path = memory_path / config.search.fts_db_path
build_keyword_index(memory_path, fts_db_path,
    chunk_size=config.embeddings.chunk_size,
    chunk_overlap=config.embeddings.chunk_overlap)
```

**Remediation**: Extract local helper at module level:
```python
def _rebuild_keyword_index(memory_path: Path, config: Config) -> None:
    from src.pipeline.keyword_index import build_keyword_index
    fts_db_path = memory_path / config.search.fts_db_path
    build_keyword_index(memory_path, fts_db_path,
        chunk_size=config.embeddings.chunk_size,
        chunk_overlap=config.embeddings.chunk_overlap)
```

**Effort**: 5 min

---

### 8. Superseded-Fact Filtering (15+ copies)

**Importance: 7/10** — The same list comprehension scattered across the codebase.

**Expression**: `[f for f in facts if "[superseded]" not in f]`

**Locations** (source files only, excluding tests):
- `src/memory/context/formatter.py`: lines 156, 234, 341
- `src/memory/store.py`: lines 83, 230
- `src/pipeline/orchestrator.py`: lines 155, 191
- `src/pipeline/enricher.py`: line 220
- `src/pipeline/dream.py`: lines 484, 728, 1097, 1321
- `src/pipeline/dashboard.py`: line 94
- `src/pipeline/visualize.py`: line 35

**Remediation**: Add to `src/core/utils.py`:
```python
def live_facts(facts: list[str]) -> list[str]:
    """Return facts excluding superseded ones."""
    return [f for f in facts if "[superseded]" not in f]
```

**Effort**: 30 min (many call sites but trivial replacement)

---

### 9. MCP Entity-Lookup Boilerplate (4 copies)

**Importance: 6/10** — Same 4-line pattern repeated in every MCP tool that operates on entities.

**Locations** in `src/mcp/server.py`:
- Line 188 (`delete_fact`)
- Line 227 + 231 (`delete_relation` — twice, for from/to)
- Line 264 (`modify_fact`)
- Line 316 (`correct_entity`)

**Repeated pattern**:
```python
entity_id = find_entity_by_name(entity_name, graph)
if not entity_id:
    return json.dumps({"status": "error", "message": f"Entity '{entity_name}' not found"})
entity = graph.entities[entity_id]
```

**Remediation**:
```python
def _resolve_entity(graph, name: str) -> tuple[str, GraphEntity] | str:
    """Resolve entity or return JSON error string."""
    entity_id = find_entity_by_name(name, graph)
    if not entity_id:
        return json.dumps({"status": "error", "message": f"Entity '{name}' not found"})
    return entity_id, graph.entities[entity_id]
```

**Effort**: 20 min

---

### 10. Context `section_budget()` — Duplicate Nested Function

**Importance: 5/10** — Identical 3-line nested function defined twice in the same file.

**Locations**:
- `src/memory/context/builder.py:185` (inside `build_context()`)
- `src/memory/context/builder.py:369` (inside `build_context_with_llm()`)

**Identical code**:
```python
def section_budget(key: str) -> int:
    pct = budget.get(key, config.ctx.default_budget_pct)
    return int(total_budget * pct / 100)
```

**Remediation**: Move to `src/memory/context/utilities.py` as a module-level function:
```python
def calculate_section_budget(budget: dict, key: str, total: int, default_pct: int) -> int:
    return int(total * budget.get(key, default_pct) / 100)
```

**Effort**: 10 min

---

### 11. Entity Enrichment — 3 Variants

**Importance: 6/10** — Three functions that read entity facts + relations with overlapping logic.

**Locations** in `src/memory/context/formatter.py`:
- `_enrich_entity()` (line 312) — full dossier with scores, relations, BFS depth-1
- `_enrich_entity_natural()` (line 212) — compact dossier for natural LLM context
- `_read_entity_facts()` (implied, line ~72) — minimal fact extraction

**Shared sub-operations**:
1. Path validation (`is_relative_to(memory_path)`)
2. `read_entity()` call with exception handling
3. Fact filtering (superseded, expired, dedup)
4. Relation collection from graph

**Remediation**: Extract shared core:
```python
def _load_entity_data(entity_id, entity, graph, memory_path, config):
    """Validate path, read MD, filter facts, collect relations. Returns (facts, relations) or None."""
    ...
```
Then `_enrich_entity()` and `_enrich_entity_natural()` become thin formatters over the same data.

**Effort**: 45 min

---

### 12. Template Loading + Fallback (2 copies)

**Importance: 4/10** — Same template-load-or-fallback block in the same file.

**Locations** in `src/memory/context/builder.py`:
- Lines 168-172 (inside `build_context()`)
- Lines 351-356 (inside `build_context_with_llm()`)

**Repeated**:
```python
template_path = config.prompts_path / "context_template.md"
if template_path.exists():
    template = template_path.read_text(encoding="utf-8")
else:
    template = "# Personal Memory — {date}\n\n{sections}\n\n{available_entities}\n{custom_instructions}"
```

**Remediation**:
```python
def _load_template(config: Config) -> str:
    path = config.prompts_path / "context_template.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "# Personal Memory — {date}\n\n{sections}\n\n{available_entities}\n{custom_instructions}"
```

**Effort**: 10 min

---

### 13. Prompt "Respond ONLY with valid JSON" Instruction (5 copies)

**Importance: 3/10** — Prompt text, not code. Each prompt has slightly different framing.

**Locations**:
- `prompts/extract_facts.md:57`
- `prompts/arbitrate_entity.md:4`
- `prompts/consolidate_facts.md:3`
- `prompts/summarize_entity.md:5`
- `prompts/discover_relations.md:3`

**Note**: This is intentional — prompts are self-contained documents. Extracting a shared "JSON instruction include" would add complexity without meaningful benefit, since prompt templates don't support includes. **No action recommended.**

---

### 14. YAML Frontmatter Stripping (2 variants)

**Importance: 3/10** — Two slightly different implementations for the same operation.

**Locations**:
- `src/pipeline/doc_ingest.py` (~line 24):
  ```python
  if text.startswith("---"):
      end = text.find("---", 3)
      if end != -1:
          text = text[end + 3:].lstrip("\n")
  ```
- `src/pipeline/indexer.py` (~line 270):
  ```python
  if text.startswith("---"):
      parts = text.split("---", 2)
      text = parts[2] if len(parts) >= 3 else text
  ```

**Remediation**: Use `parse_frontmatter()` from `src/core/utils.py` if it exists, or add a `strip_frontmatter(text) -> str` utility.

**Effort**: 10 min

---

### 15. Mock Extraction Fixtures in Tests (3 copies)

**Importance: 4/10** — Same entity names and structures across test files.

**Locations**:
- `tests/test_extractor.py:19-53`
- `tests/test_e2e.py:31-94` (`_mock_extract_fixture()`)
- `tests/test_phase1_integration.py:68-85`

**Shared entities**: "Mal de dos", "Dr Martin", "Sophie" with near-identical observations.

**Remediation**: Add to `tests/helpers.py`:
```python
def make_raw_extraction(**overrides) -> RawExtraction:
    """Standard mock extraction with Mal de dos + Dr Martin."""
    ...
```

**Effort**: 30 min

---

## Priority Matrix

| Priority | Finding | Impact if left | Fix effort |
|----------|---------|----------------|------------|
| **P0** | #7 Keyword index exact dup | Bug risk (divergence) | 5 min |
| **P1** | #1 Path filtering (4 copies) | Filter rule drift | 15 min |
| **P1** | #4 LLM call wrappers | Maintenance burden | 20 min |
| **P1** | #8 Superseded-fact filter | Inconsistency risk | 30 min |
| **P2** | #2 Test `_make_config` (17 copies) | Config API changes painful | 1-2 hours |
| **P2** | #5 Chat frontmatter cycle | Low (same file) | 20 min |
| **P2** | #9 MCP entity-lookup | Low (same file) | 20 min |
| **P2** | #11 Entity enrichment variants | Fact-filtering drift | 45 min |
| **P3** | #3 Test entity factories | Test readability | 1 hour |
| **P3** | #6 JSONL log modules | Low (stable code) | 45 min |
| **P3** | #10 section_budget dup | Low (same file) | 10 min |
| **P3** | #12 Template loading | Low (same file) | 10 min |
| **Skip** | #13 Prompt JSON instruction | Intentional | — |
| **P3** | #14 Frontmatter stripping | Low | 10 min |
| **P3** | #15 Mock extraction fixtures | Test maintenance | 30 min |

---

## Recommended Action Order

1. **Quick wins (< 30 min total)**: #7, #1, #4, #10, #12 — immediate dedup with zero risk
2. **Medium batch (~ 1 hour)**: #8, #5, #9 — extract shared helpers
3. **Test infrastructure (~ 2-3 hours)**: #2, #3, #15 — create `tests/helpers.py`
4. **Optional refactors**: #6, #11, #14 — lower ROI, schedule when touching those files

**Total estimated effort**: ~6-7 hours for all findings
