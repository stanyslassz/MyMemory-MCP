# Audit 01 — Extraction Prompt & Execution Chain

**Date**: 2026-03-10
**Scope**: `prompts/extract_facts.md`, `src/pipeline/extractor.py`, `src/core/llm.py` (extraction path), `src/core/models.py` (extraction models)
**Goal**: Exhaustive traceability of Step 1 — from raw chat text to `RawExtraction` object.

---

## Table of Contents

1. [Prompt Analysis](#1-prompt-analysis)
2. [Data Models](#2-data-models)
3. [Extractor Functions](#3-extractor-functions)
4. [LLM Call Chain](#4-llm-call-chain)
5. [Orchestrator Integration](#5-orchestrator-integration)
6. [Findings & Recommendations](#6-findings--recommendations)

---

## 1. Prompt Analysis

**File**: `prompts/extract_facts.md` (77 lines)

### 1.1 Structure

The prompt uses a `# SYSTEM` / `# USER` split (lines 1, 41). Instructor with `MD_JSON` mode does **not** use the OpenAI `system` role — the entire prompt is sent as a single `user` message. The `# SYSTEM` / `# USER` headers are cosmetic only and have no functional effect on how the LLM processes them.

### 1.2 Line-by-Line Analysis

| Lines | Content | Assessment |
|-------|---------|------------|
| 3-4 | "memory extraction engine" role + "Do NOT extract small talk" | Clear. Good filter. |
| 7 | "Only extract what is explicitly stated or strongly implied. Never invent." | Core safety rule. Well placed. |
| 8 | Importance scale: 0.1 trivial, 0.5 notable, 0.9 critical | Only 3 anchor points given. No guidance for 0.2-0.4 or 0.6-0.8. Small LLMs tend to cluster at 0.5 or round to 0.1/0.5/0.9. |
| 9-11 | Allowed categories injected via `{categories_observations}`, `{categories_entity_types}`, `{categories_relation_types}` | Flat comma-separated lists. No definitions or examples of when to use each category. A small LLM seeing `progression` vs `fact` vs `context` has no semantic anchor. |
| 12 | "Use entity names as they appear in the conversation" | Good, prevents normalization errors. |
| 13-14 | Language rule: content MUST be in `{user_language}` | Critical for French-first system. But the example (line 57) shows French content — good alignment. |
| 15-16 | Date extraction: YYYY-MM or YYYY-MM-DD, empty if unknown | No guidance on relative dates ("last week", "yesterday"). LLM must infer from chat context with no anchor date provided. |
| 17-18 | Valence: "positive", "negative", "neutral", empty if uncertain | Reasonable. The "leave empty if uncertain" is good for recall over precision. |
| 19-23 | Supersedes field for corrections/contradictions | **Weakest section.** Only one example given ("La Rosiere not Toulouse"). No guidance on: partial corrections, updates vs contradictions, graduated changes ("it's getting better" — does this supersede a diagnosis?). Small LLMs will almost always leave this empty. |
| 24-29 | AI Personality extraction (ai_self, max 3 per conversation) | The "NOVEL, enduring" qualifier is vague. Small LLMs may still extract one-off instructions. Cap of 3 is good. |
| 30-34 | User-about-themselves rule: use specialized entities, not "person" | Well explained with concrete example (Airbus). Multiple negative examples help. |
| 35-36 | Medical techniques are observations, not entities | Domain-specific guardrail. Good. |
| 37 | Max ~120 chars per observation | "~120" is vague — the `~` gives no hard boundary. `sanitize_extraction()` does not enforce this limit (it is enforced only post-consolidation in `store.py` at 150 chars). |
| 38 | "Respond ONLY with valid JSON. No text before or after." | **Contradicted by Instructor MD_JSON mode**, which expects JSON inside markdown code blocks. This instruction tells the LLM to emit bare JSON, but Instructor's `MD_JSON` mode extracts from ````json ... ``` `` blocks. If the LLM obeys the prompt literally and emits bare JSON, MD_JSON parsing may still work (it falls back), but the instruction is misleading. |
| 42-44 | `{chat_content}` injection | No length warning to the LLM. If content is long (split segments), there's no "this is segment X of Y" context. |
| 48-74 | Example JSON output | Good: shows two entity types (person + ai_self), relations, summary. **Problem**: example shows `"date": "2024-09"` and `"date": ""` — good empty handling. But no example of `supersedes` being used (always empty in example). Small LLMs learn from examples more than rules. |
| 77 | Final instruction: "Return ONLY a JSON object" | Reinforces line 38. Same MD_JSON tension. |

### 1.3 Variables Injected

| Variable | Source | Example Value |
|----------|--------|---------------|
| `{user_language}` | `config.user_language` | `fr` |
| `{categories_observations}` | `config.categories.observations` joined with `, ` | `fact, preference, diagnosis, treatment, ...` |
| `{categories_entity_types}` | `config.categories.entity_types` joined with `, ` | `person, health, work, project, ...` |
| `{categories_relation_types}` | `config.categories.relation_types` joined with `, ` | `affects, improves, worsens, ...` |
| `{chat_content}` | Raw chat text | Full conversation |
| `{json_schema}` | `RawExtraction.model_json_schema()` | **UNUSED** — injected by `call_extraction()` (llm.py:250-251) but no `{json_schema}` placeholder exists in the prompt. The schema is silently dropped. |

### 1.4 What a Small LLM (Qwen3-30B) Would Struggle With

1. **Category disambiguation**: 17 observation categories listed as a flat comma-separated string with no definitions. `fact` vs `context` vs `progression` are ambiguous. The LLM must guess.
2. **Supersedes logic**: The single example is insufficient. Corrections like "actually, she works at Thales now, not Airbus" require understanding temporal override semantics.
3. **Importance calibration**: With only 3 anchor points (0.1, 0.5, 0.9), small LLMs produce bimodal distributions. Missing mid-range guidance.
4. **Relation type selection**: 13 relation types with no definitions. `affects` vs `linked_to` vs `requires` are all plausible for many real relationships. `linked_to` becomes a catch-all.
5. **Date inference from relative expressions**: "last month", "two years ago" with no anchor date in the prompt. The LLM has no way to compute absolute dates.
6. **JSON schema complexity**: The nested structure (entities containing observations arrays) is challenging. Small models often produce flat structures or miss nested array boundaries.
7. **Thinking model interference**: Qwen3 may emit `<think>` tokens that Instructor's streaming doesn't count as progress, potentially causing stall detection to fire. The `strip_thinking()` function exists but is only applied to non-streaming calls.

### 1.5 Missing Guidance

1. **No anchor date**: The prompt doesn't inject today's date. Relative date expressions ("yesterday", "last week") cannot be resolved.
2. **No deduplication guidance**: If the same entity appears multiple times in a conversation with different phrasings, the prompt gives no instruction on whether to merge or create multiple entries.
3. **No entity relationship directionality guidance**: `from_name` and `to_name` imply direction, but the prompt doesn't explain which entity should be "from" vs "to" for asymmetric relations like `parent_of` or `improves`.
4. **No guidance on observation granularity**: Should "She moved to Paris and started a new job at Thales" be one observation or two? The prompt is silent.
5. **No negative examples**: The prompt says what to extract but gives few examples of what NOT to extract (beyond "small talk"). Common over-extraction targets: timestamps, filler, meta-discussion.
6. **No relation context guidance**: The `context` field on relations has no explanation of what should go there.
7. **No tags guidance**: The `tags` field is shown in the example but never explained. What vocabulary? Free-form? From a controlled list?

---

## 2. Data Models

**File**: `src/core/models.py`

### 2.1 RawObservation (models.py:32-39)

```python
class RawObservation(BaseModel):
    category: ObservationCategory       # Literal of 17 values
    content: str                         # Free text
    importance: float = Field(ge=0, le=1)  # Constrained 0-1
    tags: list[str] = Field(default_factory=list)
    date: str = ""                       # Optional ISO date
    valence: Literal["positive", "negative", "neutral", ""] = ""
    supersedes: str = ""                 # Old fact description
```

**Notes**:
- `category` uses Pydantic `Literal` validation — if the LLM invents a category, Instructor's `max_retries` will force re-generation. If all retries fail, the entire extraction fails. `sanitize_extraction()` handles this post-hoc if the raw dict path is used.
- `importance` has `ge=0, le=1` — Pydantic rejects out-of-range values at parse time. Combined with `sanitize_extraction()` clamping, there is double protection.
- `date` is a plain `str` with no format validation. Any string is accepted: "yesterday", "2024", "sometime in March". No downstream validation either.
- `valence` allows empty string `""` as a valid value alongside the three named options. This is correct per the prompt.
- `supersedes` is a free-text field with no structure. Downstream, nothing in the pipeline currently acts on this field — it is stored in the observation but never used for fact replacement logic.

### 2.2 RawEntity (models.py:42-44)

```python
class RawEntity(BaseModel):
    name: str
    type: EntityType  # Literal of 9 values
    observations: list[RawObservation] = Field(default_factory=list)
```

**Notes**:
- `name` has no validation — any string accepted. Empty strings are caught by `sanitize_extraction()` (extractor.py:62-63).
- `type` uses Literal validation. Invalid types cause Instructor retry or are caught by sanitization.
- An entity with zero observations is valid (empty list default). This can happen and will create stub entities in enrichment.

### 2.3 RawRelation (models.py:48-52)

```python
class RawRelation(BaseModel):
    from_name: str
    to_name: str
    type: RelationType  # Literal of 13 values
    context: str = ""
```

**Notes**:
- `from_name` / `to_name` are free strings that must match entity names from the extraction. No referential integrity check at the model level — this is deferred to the resolver.
- Relations can reference entities not in the `entities` list. The enricher creates stub entities for forward references (enricher.py).

### 2.4 RawExtraction (models.py:55-58)

```python
class RawExtraction(BaseModel):
    entities: list[RawEntity] = Field(default_factory=list)
    relations: list[RawRelation] = Field(default_factory=list)
    summary: str = ""
```

**Notes**:
- All fields have defaults — an empty extraction `{}` is valid.
- The `summary` field is not used downstream except for display and merged summaries in segment splitting.

### 2.5 JSON Schema Complexity

`RawExtraction.model_json_schema()` produces a deeply nested schema with `$defs` for each sub-model, Literal enums as `enum` arrays, and nested `items` definitions. This is ~80 lines of JSON. It is generated in `call_extraction()` (llm.py:250) but **never injected into the prompt** (no `{json_schema}` placeholder in `extract_facts.md`). The schema is only used by Instructor internally for validation/retry.

---

## 3. Extractor Functions

**File**: `src/pipeline/extractor.py` (223 lines)

### 3.1 Module-Level Constants (extractor.py:20-39)

```python
_VALID_RELATION_TYPES: set[str] = set(get_args(RelationType))
_VALID_ENTITY_TYPES: set[str] = set(get_args(EntityType))
_VALID_OBSERVATION_CATEGORIES: set[str] = set(get_args(ObservationCategory))
```

Derived from Literal types at import time. Used by `sanitize_extraction()` for validation.

```python
_RELATION_FALLBACK: dict[str, str]  # 10 entries mapping common LLM inventions
```

Maps French and English invented relation types to valid ones. Coverage is limited — only 10 mappings for a potentially infinite space of LLM inventions. Any unmapped type falls through to `"linked_to"` (extractor.py:119).

### 3.2 `sanitize_extraction()` (extractor.py:42-131)

**Signature**: `sanitize_extraction(raw: Union[RawExtraction, dict]) -> RawExtraction`

**Purpose**: Post-hoc cleanup of LLM output. Accepts either a validated `RawExtraction` (from successful Instructor parse) or a raw `dict` (for cases where Pydantic would reject the output).

**Flow**:
1. Convert to dict if needed (line 48-51)
2. Fix `summary`: `None` → `""` (line 54-56)
3. For each entity:
   - Drop entities with empty `name` (line 62-63)
   - Fix invalid `type` → `"interest"` (line 68-70)
   - For each observation:
     - Drop observations with empty `content` (line 76-78)
     - Fix invalid `category` → `"fact"` (line 81-83)
     - Clamp `importance` to [0.0, 1.0], default 0.5 if None (line 86-89)
     - Coerce `None` → `""` for `valence`, `date`, `supersedes` (line 92-97)
     - Coerce `None` → `[]` for `tags` (line 98-99)
4. For each relation:
   - Drop relations with empty `from_name` or `to_name` (line 112-114)
   - Fix invalid `type` via `_RELATION_FALLBACK` or default `"linked_to"` (line 117-121)
   - Coerce `None` context → `""` (line 124)
5. Construct and return `RawExtraction(**data)` (line 131)

**Edge cases handled**:
- Null/None fields at every level
- Empty strings for names/content
- Out-of-vocabulary types for entities, observations, relations
- Out-of-range importance values

**Edge cases NOT handled**:
- Duplicate entities within the same extraction (same name, different observations) — not merged
- Observation content exceeding 120 chars — not truncated here (only at store.py level, 150 chars)
- `date` format validation — any string accepted
- `valence` values outside the Literal set — if Pydantic already parsed it, it's validated; if raw dict, no check
- `tags` content validation — any strings accepted
- Self-referencing relations (`from_name == to_name`) — not caught

**Failure mode**: The final `RawExtraction(**data)` call (line 131) will raise `ValidationError` if the sanitized data still doesn't conform to Pydantic constraints. This exception propagates up to the orchestrator's `try/except` block.

### 3.3 `_merge_extractions()` (extractor.py:134-166)

**Signature**: `_merge_extractions(extractions: list[RawExtraction]) -> RawExtraction`

**Purpose**: Merge multiple segment extractions (from split long chats) into one, deduplicating by slug (entities) and by (from_slug, to_slug, type) tuple (relations).

**Flow**:
1. Merge entities: keyed by `slugify(entity.name)`. For existing slug, append observations not already seen (dedup by `content.lower()`). For new slug, deep copy (line 141-149).
2. Merge relations: dedup by `(slugify(from_name), slugify(to_name), type)` (line 151-158).
3. Merge summaries: space-joined (line 160).

**Edge cases handled**:
- Observation-level dedup by lowercase content match
- Relation dedup by slugified tuple

**Edge cases NOT handled**:
- Entity type conflicts: if segment 1 extracts "Marie" as `person` and segment 2 extracts "Marie" as `work`, the first-seen type wins. No conflict resolution.
- Observation importance conflicts: same content from different segments with different importance scores — the first-seen wins (second is deduped out).
- Partial name overlap: "Dr. Martin" and "Martin" get different slugs and are not merged.
- Summary quality: concatenation with spaces can produce incoherent summaries.

### 3.4 `_split_text()` (extractor.py:169-188)

**Signature**: `_split_text(text: str, segment_tokens: int, overlap_tokens: int) -> list[str]`

**Purpose**: Split text into overlapping segments by approximate word count.

**Flow**:
1. Split text by whitespace (line 171)
2. Convert token count to word count using 1.3 ratio (line 172-173)
3. Slide window with overlap (line 179-188)

**Edge cases handled**:
- Text shorter than one segment returns as-is (line 175-176)
- Loop termination when start exceeds text length (line 185-186)

**Edge cases NOT handled**:
- The 1.3 tokens-per-word ratio is a rough English approximation. For French, this may underestimate (French words tend to be longer, but tokenizers vary). No language-aware adjustment.
- Splits can break mid-sentence or mid-entity-mention. No sentence boundary detection.
- If overlap is >= segment size, infinite loop. No guard (though in practice overlap=200 << segment_tokens).

### 3.5 `extract_from_chat()` (extractor.py:191-223)

**Signature**: `extract_from_chat(chat_content: str, config: Config) -> RawExtraction`

**Purpose**: Main entry point. Extracts structured info from chat, splitting if needed.

**Flow**:
1. Empty check: return empty extraction (line 197-198)
2. Estimate tokens (line 200)
3. Compare against 70% of `context_window` minus 500-token overhead (line 201-205)
4. If under threshold: single `call_extraction()` call (line 206)
5. If over: split into segments at 50% of context_window with 200-token overlap, extract each, merge (line 209-223)

**Edge cases handled**:
- Empty/whitespace-only input
- Content exceeding context window

**Edge cases NOT handled**:
- `prompt_overhead = 500` is hardcoded (line 201). The actual prompt template is ~1200 tokens (77 lines of instructions + example JSON). This underestimate means the 70% threshold triggers too late — the LLM may receive prompts exceeding its context window.
- No error handling: if `call_extraction()` raises on any segment, the entire function fails. No per-segment retry or fallback.
- No segment context: the LLM sees each segment in isolation with no "this is part X of Y" indicator. Cross-segment entity references may produce inconsistent extractions.

---

## 4. LLM Call Chain

**File**: `src/core/llm.py`

### 4.1 `load_prompt()` (llm.py:77-100)

**Signature**: `load_prompt(name: str, config: Config, **variables: Any) -> str`

**Purpose**: Load prompt template and inject variables.

**Flow**:
1. Read `prompts/{name}.md` (line 83-87)
2. Auto-inject `user_language`, `categories_observations`, `categories_entity_types`, `categories_relation_types` (line 90-95)
3. Replace each `{key}` with `str(value)` via `str.replace()` (line 97-98)

**Edge cases handled**:
- Missing prompt file raises `FileNotFoundError` (line 85)
- Uses `str.replace()` not `str.format()` to avoid JSON brace conflicts

**Edge cases NOT handled**:
- Unreplaced variables: if a `{placeholder}` exists in the template but no matching variable is provided, it remains as literal text in the prompt. No warning logged. In the case of `{json_schema}` in `call_extraction()`, the variable is passed but the placeholder doesn't exist in the template — the variable is silently ignored.
- Variable injection order: variables are replaced sequentially. If a variable's value contains `{another_var}`, it is NOT expanded (no recursive replacement). This is safe but could surprise.
- `chat_content` may contain `{braces}` that could match other variable names. Since `str.replace()` is used after auto-injection, user content containing `{user_language}` would already have been replaced. **Security concern**: if chat content contains `{categories_observations}`, it would be replaced with the category list during the loop. However, since `chat_content` is replaced last (it's in `**variables`, processed after `setdefault` calls), and `str.replace` replaces all occurrences, the auto-injected variables are replaced first in the template, then `{chat_content}` is replaced with the actual chat. The chat content itself won't trigger further replacements because the loop is one-pass per variable. This is safe.

### 4.2 `call_extraction()` (llm.py:244-260)

**Signature**: `call_extraction(chat_content: str, config: Config) -> RawExtraction`

**Purpose**: Build prompt, call LLM with stall detection, return validated extraction.

**Flow**:
1. Generate JSON schema from `RawExtraction` (line 250) — **unused by prompt**
2. Load prompt with `chat_content` and `json_schema` variables (line 251-256)
3. Set `stall_timeout` = `config.llm_extraction.timeout` (line 257)
4. Call `_call_with_stall_detection()` (line 258-260)

**Key observation**: The `json_schema` variable is passed to `load_prompt()` but `extract_facts.md` has no `{json_schema}` placeholder. The schema generation on line 250 is wasted computation. Instructor uses its own internal schema for validation, so this is not a functional bug, but it is dead code and a misleading signal that the schema is part of the prompt.

### 4.3 `_call_with_stall_detection()` (llm.py:147-241)

**Signature**: `_call_with_stall_detection(step_config, prompt, response_model, stall_timeout=30) -> T`

**Purpose**: Stream LLM response with a watchdog thread that detects stalls.

**Architecture**:
- **Worker thread** (`_do_call`, line 171-208): creates Instructor client, calls `create_partial()` for streaming, iterates chunks, updates `last_activity` timestamp.
- **Watchdog** (main thread, line 216-235): polls every 2 seconds, checks idle time against threshold.

**Flow**:
1. Initialize shared state: `last_activity`, `first_token_received`, `lock`, `done` event (line 164-169)
2. Start worker thread (line 210-211)
3. Watchdog loop:
   - Wait 2s on `done` event (line 217)
   - If done, break (line 218-219)
   - Check idle time under lock (line 220-222)
   - First-token grace: `2x stall_timeout` before first token (line 223)
   - If idle > threshold: set `StallError`, signal done (line 224-235)
4. After loop: raise error if any, return result (line 237-241)

**Edge cases handled**:
- First-token grace period for thinking models (2x timeout) (line 223)
- Worker thread is daemon — won't block process exit (line 210)
- Lock protects shared `last_activity` and `first_token_received` (line 168, 192-196)
- `done.set()` from both worker (normal completion) and watchdog (stall) (line 208, 234)

**Edge cases NOT handled**:
- **Thread leak on stall**: When the watchdog detects a stall, it sets `done` and raises `StallError`, but the worker thread continues running in the background (it's still blocked on `for chunk in partial`). The daemon flag means it won't prevent exit, but during a run with multiple chats, leaked threads accumulate. There is no cancellation mechanism.
- **Race condition window**: Between `done.wait(timeout=2.0)` returning and acquiring the lock, the worker could update `last_activity`. This is benign (the watchdog checks idle time, which would now be small, so no false stall).
- **`result = chunk` on every iteration** (line 203): The last partial chunk is the final validated result. If the stream is interrupted mid-object, `result` holds an incomplete partial. However, `StallError` is raised in this case, so the incomplete result is never returned.
- **`timeout` parameter to litellm** (line 187): Set to `step_config.timeout * 3`. With default timeout=60, this gives 180s connection timeout. The stall detection at 60s would fire first for mid-stream stalls, but the connection timeout could fire first for initial connection failures.

### 4.4 `_get_client()` (llm.py:103-115)

**Signature**: `_get_client(step_config: LLMStepConfig) -> instructor.Instructor`

**Purpose**: Create Instructor client in `MD_JSON` mode.

**Notes**:
- Uses `instructor.Mode.MD_JSON` (line 113): extracts JSON from markdown code blocks. This allows the LLM to wrap JSON in ` ```json ... ``` ` blocks, which many models do naturally.
- The `api_base` from `step_config` is passed to `kwargs` but not used in the `from_litellm()` call (line 106-107, 111-114). It's passed later in the actual completion call. This is correct — `api_base` is a per-request parameter.
- A new client is created on every call. No connection pooling.

### 4.5 `_repaired_json()` Context Manager (llm.py:30-69)

**Signature**: `@contextmanager _repaired_json()`

**Purpose**: Monkey-patch `json.loads` to auto-repair malformed JSON from small LLMs.

**Flow**:
1. If `json_repair` not installed, yield immediately (line 38-39)
2. Save original `json.loads` (line 42)
3. Replace with `_patched` that tries original first, then repairs on `JSONDecodeError` (line 45-63)
4. Restore original on exit (line 68)

**Edge cases handled**:
- Recursion guard: restores original `json.loads` during repair, then re-patches (line 58-62). `json_repair` internally calls `json.loads`, which would cause infinite recursion without this.
- Graceful degradation if `json_repair` not installed (line 38-39)
- Logging: first repair attempt is `warning`, subsequent are `debug` (line 52-55)

**Edge cases NOT handled**:
- **Thread safety**: `json.loads` is a global function. If two extraction calls run concurrently (unlikely in current architecture, but possible), they would both try to patch/unpatch the same global. This could cause `json.loads` to be permanently patched or restored to the wrong function.
- **Repair failure**: If `_repair_json()` returns invalid JSON that still fails `original()`, the `JSONDecodeError` propagates up. This is correct behavior but means the patching only helps with fixable errors.

### 4.6 `strip_thinking()` (llm.py:72-74)

**Signature**: `strip_thinking(text: str) -> str`

Removes `<think>...</think>` tags. **Not called in the extraction path** — only used by `call_context_generation()`, `call_context_section()`, and `call_entity_summary()`. For extraction, `<think>` content would be part of the streamed chunks and could interfere with JSON parsing. Instructor's `MD_JSON` mode should ignore non-JSON content, but `<think>` blocks containing partial JSON could confuse the parser.

---

## 5. Orchestrator Integration

**File**: `src/pipeline/orchestrator.py` (lines 262-304)

### 5.1 Per-Chat Extraction Flow

```
content = get_chat_content(chat_path)          # line 280
  → extract_from_chat(content, config)          # line 287
  → sanitize_extraction(extraction)             # line 288
```

### 5.2 Error Handling

On any exception from `extract_from_chat()` or `sanitize_extraction()` (line 290):
1. Check if timeout via `is_timeout_error(e)` (line 291) — recognizes `StallError`, plus string matching for "timeout", "timed out", "ReadTimeout", etc.
2. Increment retry counter in chat frontmatter (line 292)
3. If timeout OR retries >= 2: fall back to `doc_ingest` (line 293-298)
4. Otherwise: log and continue to next chat (line 302-303)

**Failure cascade**: extraction failure → doc_ingest fallback → chat content chunked into FAISS directly (no entities created). The chat is marked with `fallback: doc_ingest` in frontmatter and recorded in `_retry_ledger.json`.

### 5.3 Retry Semantics

`EXTRACTION_MAX_RETRIES = 2` (line 263). The counter is incremented BEFORE the check (line 292-293), so:
- 1st failure: retries=1, < 2 → retry later
- 2nd failure: retries=2, >= 2 → fallback

This means each chat gets at most 2 extraction attempts before fallback. Timeout errors get immediate fallback on first occurrence.

---

## 6. Findings & Recommendations

### 6.1 Critical Issues

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| C1 | **`{json_schema}` dead code**: `call_extraction()` generates and passes `json_schema` but the prompt has no placeholder for it. The LLM never sees the schema. | llm.py:250-251, extract_facts.md | Low functional impact (Instructor validates internally), but wasted computation and misleading code. |
| C2 | **Prompt overhead underestimate**: Hardcoded `prompt_overhead = 500` tokens, but actual prompt is ~1200+ tokens (instructions + example + injected categories). The split threshold triggers too late. | extractor.py:201 | Chats near the context window limit may cause truncation or LLM errors. |
| C3 | **No anchor date in prompt**: Relative date expressions ("yesterday", "last week") cannot be resolved without knowing today's date. | extract_facts.md | Dates from relative expressions will be wrong or empty. |
| C4 | **Thread leak on stall**: Worker thread continues running after `StallError` is raised. No cancellation mechanism. | llm.py:210-234 | Leaked threads accumulate during long runs with repeated stalls. |
| C5 | **`supersedes` field is write-only**: Extracted and stored but never consumed by any downstream pipeline step. No fact replacement logic exists. | models.py:39, prompt line 19-23 | Users expect corrections to replace old facts. They don't — both old and new facts coexist. |

### 6.2 Moderate Issues

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| M1 | **No category definitions in prompt**: 17 observation categories, 9 entity types, 13 relation types listed as flat strings without definitions. | extract_facts.md:9-11 | Small LLMs misclassify frequently. `linked_to` becomes catch-all for relations. |
| M2 | **Bare JSON instruction conflicts with MD_JSON mode**: Prompt says "Respond ONLY with valid JSON. No text before or after" but Instructor expects markdown code blocks. | extract_facts.md:38, llm.py:113 | Ambiguous signal to LLM. Works due to MD_JSON fallback but is a latent parsing risk. |
| M3 | **No segment context for split chats**: Each segment is extracted independently with no "this is part X of Y" or "entities already seen" context. | extractor.py:218-220 | Cross-segment entities may get different types/names. Merge relies solely on slug matching. |
| M4 | **Entity type conflict in merge**: If two segments assign different types to the same slug, first-seen wins silently. | extractor.py:141-149 | Wrong entity type persists without warning. |
| M5 | **`_split_text()` breaks mid-sentence**: Word-boundary splitting with no sentence detection. | extractor.py:169-188 | Entities or facts spanning a split boundary may be lost or duplicated. |
| M6 | **Observation length not enforced at extraction**: Prompt says "~120 chars" but neither `sanitize_extraction()` nor models enforce it. Only `store.py` truncates at 150 chars post-consolidation. | extract_facts.md:37, extractor.py | Long observations silently stored, then later truncated possibly mid-sentence. |
| M7 | **`<think>` tags not stripped in extraction path**: `strip_thinking()` is not called for streaming extraction. Thinking model output could interfere with JSON parsing. | llm.py:72-74, not called in extraction | Qwen3/DeepSeek-R1 `<think>` content in streamed chunks may cause partial parse failures. |

### 6.3 Low-Priority / Cosmetic

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| L1 | **`_RELATION_FALLBACK` limited coverage**: Only 10 French/English mappings. Any unmapped invention goes to `linked_to`. | extractor.py:26-39 | Over-reliance on `linked_to` catch-all degrades relation graph quality. |
| L2 | **No duplicate entity detection in `sanitize_extraction()`**: Same entity name appearing twice in one extraction is not merged. | extractor.py:42-131 | Rare in practice but produces duplicate entities in enrichment. |
| L3 | **Self-referencing relations not caught**: `from_name == to_name` is valid. | extractor.py:110-127 | Meaningless relations like "X affects X" can be created. |
| L4 | **Summary concatenation in merge**: Space-joined summaries from segments produce incoherent text. | extractor.py:160 | Minor — summary is only used for display. |
| L5 | **New Instructor client per call**: No connection reuse or caching. | llm.py:103-115 | Performance overhead, especially for multi-segment extractions. |
| L6 | **`valence` in raw dict path not validated**: `sanitize_extraction()` coerces `None` → `""` but doesn't validate against the Literal set. | extractor.py:92-93 | If LLM returns `"valence": "très positif"`, it passes sanitization but fails Pydantic at line 131. |

### 6.4 Recommendations (Prioritized)

1. **Add today's date to prompt** (C3): Inject `{today_date}` into `extract_facts.md`. Add "Today's date is {today_date}. Use this to resolve relative dates." One-line fix in `call_extraction()`.

2. **Fix prompt overhead constant** (C2): Replace `prompt_overhead = 500` with a dynamic estimate: `prompt_overhead = _estimate_tokens(prompt_template_without_chat)`. Or increase to a safe static value like 1500.

3. **Remove dead `json_schema` generation** (C1): Delete line 250 in `llm.py` and remove `json_schema=schema` from the `load_prompt()` call. Or, add `{json_schema}` to the prompt if schema guidance is desired.

4. **Add category definitions to prompt** (M1): At minimum, add one-line descriptions for the most confused categories. Example: `fact = objective information, preference = personal taste, context = situational/temporary info`.

5. **Implement `supersedes` consumption** (C5): Either build fact replacement logic in the enricher (find existing observation matching `supersedes` description and mark/remove it), or remove the field from the prompt to avoid false expectations.

6. **Harmonize JSON instruction with MD_JSON mode** (M2): Change prompt line 38 to: "Respond with a JSON object inside a markdown code block (```json ... ```)."

7. **Add segment context for split chats** (M3): Prepend to each segment: "This is segment {i}/{n} of a conversation. Previously seen entities: {list}."

8. **Strip thinking tags in streaming path** (M7): Apply `strip_thinking()` to chunk content before Instructor processes it, or configure Instructor to ignore `<think>` blocks.
