# Audit 09 — Core Infrastructure

Deep audit of the four foundational modules: `config.py`, `models.py`, `utils.py`, `llm.py`.

---

## 1. `src/core/config.py` — Configuration System

### 1.1 Helper: `_resolve_path` (line 14)

```python
def _resolve_path(base: Path, p: str) -> Path
```

Resolves a path relative to `project_root`. Absolute paths pass through unchanged. All YAML-sourced paths flow through this function.

### 1.2 `LLMStepConfig` (line 23)

Per-pipeline-step LLM configuration. Plain dataclass (no validation).

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `model` | `str` | (required) | LiteLLM model string, e.g. `ollama/llama3.1:8b` |
| `temperature` | `float` | `0.0` | No clamping — negative or >2.0 values silently pass through |
| `max_retries` | `int` | `3` | Instructor retry count |
| `timeout` | `int` | `60` | Seconds. Used as stall threshold in extraction, `*3` as connection timeout |
| `api_base` | `str \| None` | `None` | Override endpoint for local servers |
| `context_window` | `int` | `8192` | Used by extractor to decide chunking threshold |

**Gap**: No validation on any field. `temperature` can be negative, `timeout` can be zero (would cause instant stall detection), `context_window` can be zero (division risk in extractor chunking logic).

### 1.3 `EmbeddingsConfig` (line 33)

| Field | Type | Default |
|-------|------|---------|
| `provider` | `str` | `"sentence-transformers"` |
| `model` | `str` | `"all-MiniLM-L6-v2"` |
| `api_base` | `str \| None` | `None` |
| `chunk_size` | `int` | `400` |
| `chunk_overlap` | `int` | `80` |

**Gap**: No validation that `chunk_overlap < chunk_size`. If `chunk_overlap >= chunk_size`, the chunking loop in `indexer.py` would produce overlapping-only or infinite chunks.

### 1.4 `ScoringConfig` (line 42)

17 fields controlling ACT-R scoring. All `float` or `int` with sensible defaults. No validation constraints.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `model` | `str` | `"act_r"` | Scoring model identifier (only `act_r` implemented) |
| `decay_factor` | `float` | `0.5` | Long-term ACT-R decay exponent |
| `decay_factor_short_term` | `float` | `0.8` | Short-term decay exponent |
| `importance_weight` | `float` | `0.3` | Weight of importance in final score sigmoid |
| `spreading_weight` | `float` | `0.2` | Weight of spreading activation bonus |
| `permanent_min_score` | `float` | `0.5` | Floor score for permanent-retention entities |
| `relation_strength_base` | `float` | `0.5` | Initial strength for new relations |
| `relation_decay_halflife` | `int` | `180` | Days for effective strength halflife (spreading calc) |
| `relation_strength_growth` | `float` | `0.05` | Hebbian LTP increment per co-occurrence |
| `relation_ltd_halflife` | `int` | `360` | Days for stored strength LTD decay |
| `relation_decay_power` | `float` | `0.3` | Power-law exponent for effective strength |
| `retrieval_threshold` | `float` | `0.05` | Below this, score collapses to 0.0 (true forgetting) |
| `emotional_boost_weight` | `float` | `0.15` | Amygdala modulation weight |
| `window_size` | `int` | `50` | Max recent mention_dates before consolidation to buckets |
| `min_score_for_context` | `float` | `0.3` | Minimum score to appear in `_context.md` |

**Gap**: `model` field is vestigial — only `act_r` is ever checked. No enum/literal constraint. If set to anything else, scoring silently uses ACT-R anyway (no branching on this value in `scoring.py`).

### 1.5 `FAISSConfig` (line 61)

| Field | Type | Default |
|-------|------|---------|
| `index_path` | `str` | `"./memory/_memory.faiss"` |
| `mapping_path` | `str` | `"./memory/_memory.pkl"` |
| `manifest_path` | `str` | `"./memory/_faiss_manifest.json"` |
| `top_k` | `int` | `5` |

Paths are `str` here but resolved to absolute in `load_config` (line 232-234). Inconsistent with `memory_path`/`prompts_path` which are `Path` objects on `Config`.

### 1.6 `CategoriesConfig` (line 69)

| Field | Type | Default |
|-------|------|---------|
| `observations` | `list[str]` | `[]` |
| `entity_types` | `list[str]` | `[]` |
| `relation_types` | `list[str]` | `[]` |
| `folders` | `dict[str, str]` | `{}` |

These are the runtime lists loaded from `config.yaml`. They are injected into LLM prompts via `load_prompt()` (line 93-95 of `llm.py`). The closed `Literal` types in `models.py` are compile-time fixed and must be manually kept in sync with these lists.

**Critical sync requirement**: If a new observation category, entity type, or relation type is added to `config.yaml`, it will be injected into prompts but Pydantic validation in `models.py` will reject it. The `sanitize_extraction()` function in `extractor.py` handles this by fuzzy-mapping invalid types, but this is a silent lossy fallback rather than proper sync.

### 1.7 `FeaturesConfig` (line 77), `IngestConfig` (line 82), `NLPConfig` (line 88)

Small configs. `IngestConfig.jobs_path` is `str`, resolved in `load_config`. No validation issues.

### 1.8 `Config` (line 98)

Master config with 25 fields. Key properties and methods:

| Property/Method | Line | Returns | Purpose |
|----------------|------|---------|---------|
| `user_language_name` | 128 | `str` | Maps 2-letter code to full name. Only 6 languages mapped — unknown codes return the raw code. |
| `llm_dream_effective` | 134 | `LLMStepConfig` | Falls back `llm_dream` -> `llm_context` |
| `get_folder_for_type(entity_type)` | 138 | `str` | Maps entity type to subfolder, defaults to `"interests"` |
| `get_max_facts(entity_type)` | 142 | `int` | Max facts per entity type, defaults to `max_facts["default"]` or 50 |

### 1.9 `_build_llm_step` (line 147)

Factory function for `LLMStepConfig` from a YAML dict. Duplicates all defaults from the dataclass — any default change must happen in both places.

### 1.10 `load_config` (line 158)

```python
def load_config(config_path: str | Path | None = None, project_root: Path | None = None) -> Config
```

Loads `.env` first (for API keys), then `config.yaml`. Returns default `Config` if no YAML file exists. All paths resolved relative to `project_root`.

**Findings**:
- **F-CFG-1**: No config validation layer. Invalid YAML keys are silently ignored (e.g., a typo `scoring.decayfactor` would be silently dropped). No warning emitted.
- **F-CFG-2**: `context_budget` defaults to `{}` (line 108/210). When empty, context builder falls back to hardcoded percentages. This implicit fallback chain is undocumented.
- **F-CFG-3**: `max_facts` is loaded from top-level `raw.get("max_facts", ...)` (line 213) but `context_narrative` and `context_llm_sections` are loaded from `mem` (the `memory:` section, lines 211-212). Inconsistent nesting.
- **F-CFG-4**: All dataclass configs use plain `dataclass` — no `__post_init__` validation exists anywhere. Invalid combinations (e.g., `chunk_overlap > chunk_size`, `retrieval_threshold > 1.0`) pass silently.
- **F-CFG-5**: `FAISSConfig` stores paths as `str` while `Config.memory_path` and `Config.prompts_path` are `Path`. Inconsistent types.

---

## 2. `src/core/models.py` — Data Models

### 2.1 Closed Literal Types (lines 11-27)

Three literal type aliases constrain the domain vocabulary:

```python
ObservationCategory = Literal[
    "fact", "preference", "diagnosis", "treatment", "progression",
    "technique", "vigilance", "decision", "emotion",
    "interpersonal", "skill", "project", "context", "rule",
    "ai_style", "user_reaction", "interaction_rule",
]  # 17 values

EntityType = Literal[
    "person", "health", "work", "project", "interest",
    "place", "animal", "organization", "ai_self",
]  # 9 values

RelationType = Literal[
    "affects", "improves", "worsens", "requires", "linked_to",
    "lives_with", "works_at", "parent_of", "friend_of", "uses",
    "part_of", "contrasts_with", "precedes",
]  # 13 values
```

**Sync points**: These Literals must match `categories` in `config.yaml.example` (lines 129-171). Currently in sync. However, the Literals are compile-time constants while the YAML lists are runtime-loaded into `CategoriesConfig`. There is no runtime cross-check that verifies they match.

### 2.2 `RawObservation` (line 32)

```python
class RawObservation(BaseModel):
    category: ObservationCategory
    content: str
    importance: float = Field(ge=0, le=1)
    tags: list[str] = Field(default_factory=list)
    date: str = ""
    valence: Literal["positive", "negative", "neutral", ""] = ""
    supersedes: str = ""
```

**Validation**: Only `importance` is constrained (0.0 to 1.0 via `Field(ge=0, le=1)`).

**Gaps**:
- **F-MOD-1**: `content` has no `max_length`. LLMs can produce arbitrarily long observation content. The 150-char guard is only in `store.py` post-consolidation, not at the model level.
- **F-MOD-2**: `tags` has no `max_length` on the list or individual strings. The 3-tag limit is enforced only in `store.py` formatting.
- **F-MOD-3**: `date` accepts any string — no regex validation for ISO format (YYYY-MM or YYYY-MM-DD).
- **F-MOD-4**: `supersedes` is present on `RawObservation` but absent from `RawRelation`. Relations cannot express supersession (e.g., "used to work at X, now works at Y").

### 2.3 `RawEntity` (line 42)

```python
class RawEntity(BaseModel):
    name: str
    type: EntityType
    observations: list[RawObservation] = Field(default_factory=list)
```

**Gap**: `name` has no constraints. Empty string `""` is valid. No `min_length`.

### 2.4 `RawRelation` (line 48)

```python
class RawRelation(BaseModel):
    from_name: str
    to_name: str
    type: RelationType
    context: str = ""
```

**Gap**: No `supersedes` field (noted in F-MOD-4). No `strength` hint from extraction — initial strength always comes from `ScoringConfig.relation_strength_base`.

### 2.5 `RawExtraction` (line 55)

```python
class RawExtraction(BaseModel):
    entities: list[RawEntity] = Field(default_factory=list)
    relations: list[RawRelation] = Field(default_factory=list)
    summary: str = ""
```

Top-level LLM output model. `summary` is extracted but only stored in chat frontmatter (not used downstream).

### 2.6 `Resolution` (line 63)

```python
class Resolution(BaseModel):
    status: Literal["resolved", "new", "ambiguous"]
    entity_id: Optional[str] = None
    candidates: list[str] = Field(default_factory=list)
    suggested_slug: Optional[str] = None
```

**Invariant not enforced**: When `status="resolved"`, `entity_id` should be non-None. When `status="ambiguous"`, `candidates` should be non-empty. These are logical invariants but not validated by Pydantic.

### 2.7 `ResolvedEntity` (line 70) and `ResolvedExtraction` (line 75)

Wrappers pairing raw data with resolution results. No additional validation.

### 2.8 `EntityResolution` (line 83)

```python
class EntityResolution(BaseModel):
    action: Literal["existing", "new"]
    existing_id: Optional[str] = None
    new_type: Optional[EntityType] = None
```

LLM arbitration output. Same invariant gap: `action="existing"` should require `existing_id` non-None; `action="new"` should require `new_type` non-None.

### 2.9 `GraphEntity` (line 91)

```python
class GraphEntity(BaseModel):
    file: str
    type: EntityType
    title: str
    score: float = 0.0
    importance: float = 0.0
    frequency: int = 0
    last_mentioned: str = ""
    retention: Literal["short_term", "long_term", "permanent"] = "short_term"
    aliases: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    mention_dates: list[str] = Field(default_factory=list)
    monthly_buckets: dict[str, int] = Field(default_factory=dict)
    created: str = ""
    summary: str = ""
    negative_valence_ratio: float = 0.0
```

Central index model stored in `_graph.json`. 16 fields.

**Findings**:
- **F-MOD-5**: `negative_valence_ratio` is computed from facts during `rebuild_from_md()` in `graph.py` but is not persisted in entity MD frontmatter. It exists only in the graph JSON and is recomputed on every rebuild. The `EntityFrontmatter` model (line 132) deliberately omits it.
- **F-MOD-6**: `score` is stored but always recomputed by `recalculate_all_scores()`. The persisted value is the last-computed snapshot — stale after any graph mutation.
- **F-MOD-7**: `importance` on `GraphEntity` is a running average updated by `enricher.py`, but has no `ge=0, le=1` constraint (unlike `RawObservation.importance`).

### 2.10 `GraphRelation` (line 111)

```python
class GraphRelation(BaseModel):
    from_entity: str = Field(alias="from", serialization_alias="from")
    to_entity: str = Field(alias="to", serialization_alias="to")
    type: RelationType
    strength: float = 0.5
    created: str = ""
    last_reinforced: str = ""
    mention_count: int = 1
    context: str = ""

    model_config = {"populate_by_name": True}
```

Uses `Field(alias="from")` because `from` is a Python reserved word. `populate_by_name = True` allows both `from_entity=` and `from=` in constructors. `serialization_alias="from"` ensures JSON output uses `"from"` key.

**Gap**: `strength` has no `ge=0, le=1` constraint. The `min(1.0, ...)` and `max(0.1, ...)` clamping happens only in `graph.py:add_relation()` and `scoring.py:_apply_ltd()` respectively.

### 2.11 `GraphData` (line 124)

```python
class GraphData(BaseModel):
    generated: str = ""
    entities: dict[str, GraphEntity] = Field(default_factory=dict)
    relations: list[GraphRelation] = Field(default_factory=list)
```

Top-level graph container. `generated` is an ISO timestamp string.

### 2.12 `EntityFrontmatter` (line 132)

```python
class EntityFrontmatter(BaseModel):
    title: str
    type: EntityType
    retention: Literal["short_term", "long_term", "permanent"] = "short_term"
    score: float = 0.0
    importance: float = 0.0
    frequency: int = 0
    last_mentioned: str = ""
    created: str = ""
    aliases: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    mention_dates: list[str] = Field(default_factory=list)
    monthly_buckets: dict[str, int] = Field(default_factory=dict)
    summary: str = ""
```

Mirrors `GraphEntity` but omits `file`, `score` (stored but read-only in MD), and `negative_valence_ratio`. 13 fields vs GraphEntity's 16.

**Finding F-MOD-8**: `EntityFrontmatter` and `GraphEntity` are structurally near-identical but share no base class. Adding a field to one requires remembering to add it to the other. The conversion between them happens in `graph.py:rebuild_from_md()` and `enricher.py`, both doing manual field-by-field mapping.

### 2.13 `EnrichmentReport` (line 151)

```python
class EnrichmentReport(BaseModel):
    entities_updated: list[str] = Field(default_factory=list)
    entities_created: list[str] = Field(default_factory=list)
    relations_added: int = 0
    errors: list[str] = Field(default_factory=list)
```

Pipeline output tracking. No issues.

### 2.14 `SearchResult` (line 160)

```python
class SearchResult(BaseModel):
    entity_id: str
    file: str
    chunk: str
    score: float
    relations: list[dict] = Field(default_factory=list)
```

**Gap**: `relations` is `list[dict]` rather than `list[GraphRelation]`. This is a typed-dict escape hatch — the dicts contain ad-hoc keys (`from`, `to`, `type`, etc.) but without schema enforcement.

### 2.15 `RouteDecision` (line 173), `IngestKey` (line 186), `IngestJob` (line 196)

Route/ingest models. `RouteDecision.confidence` has `Field(ge=0.0, le=1.0)` — one of the few validated fields.

`IngestKey` has a `canonical` property (line 192) producing a composite key string. `IngestJob` tracks the full state machine for document ingestion.

### 2.16 `ConsolidatedFact` (line 211) and `FactConsolidation` (line 220)

```python
class ConsolidatedFact(BaseModel):
    category: ObservationCategory
    content: str
    date: str = ""
    valence: Literal["positive", "negative", "neutral", ""] = ""
    tags: list[str] = Field(default_factory=list)
    replaces_indices: list[int] = Field(default_factory=list)
```

LLM-produced consolidated facts. `replaces_indices` references positions in the original facts list.

**Gap**: Same `content` max_length gap as `RawObservation`. The 150-char post-consolidation trim in `store.py` is the only guardrail.

### 2.17 `DreamPlan` (line 226) and `DreamValidation` (line 231)

```python
class DreamPlan(BaseModel):
    steps: list[int] = Field(description="Ordered list of step numbers to execute (1-10)")
    reasoning: str = Field(description="Brief explanation of why these steps were chosen")

class DreamValidation(BaseModel):
    approved: bool = Field(description="Whether the results look correct")
    issues: list[str] = Field(default_factory=list, description="List of issues found, if any")
```

**Gap**: `DreamPlan.steps` has no constraint on valid step numbers. The dream pipeline has steps 1-10 but any `int` is accepted. Invalid step numbers would be silently skipped by the step dispatcher.

### 2.18 Model Relationship Map

```
RawExtraction
  ├── RawEntity[]
  │     ├── name, type (EntityType)
  │     └── RawObservation[]
  │           └── category (ObservationCategory), supersedes
  └── RawRelation[]
        └── type (RelationType)

ResolvedExtraction
  ├── ResolvedEntity[]
  │     ├── raw: RawEntity
  │     └── resolution: Resolution
  └── RawRelation[] (unchanged)

EntityResolution (arbitration output)

GraphData
  ├── GraphEntity{} (keyed by slug)
  │     └── type (EntityType), retention (3 values)
  └── GraphRelation[]
        └── type (RelationType)

EntityFrontmatter ≈ GraphEntity (MD YAML)

SearchResult → entity_id links to GraphData.entities key

ConsolidatedFact → replaces_indices → original fact positions
FactConsolidation → consolidated: ConsolidatedFact[]

DreamPlan → steps: int[] (dream pipeline)
DreamValidation → approved: bool
```

---

## 3. `src/core/utils.py` — Utility Functions

### 3.1 `slugify` (line 11)

```python
def slugify(text: str) -> str
```

Converts a title to a filesystem-safe slug. Steps:
1. NFKD Unicode normalization
2. ASCII encoding (drops non-ASCII)
3. Remove non-word/space/hyphen characters
4. Collapse whitespace/hyphens to single hyphen
5. Strip leading/trailing hyphens

**Findings**:
- **F-UTL-1**: Pure-ASCII output means accented names lose information: "Helene" and "Helene" (from "Helene" and "Helene") produce the same slug. For French-domain usage, "Clement" and "Clement" would collide.
- **F-UTL-2**: Empty input returns empty string `""`. No guard against empty slugs, which would create unnamed files.
- **F-UTL-3**: Entirely non-ASCII input (e.g., Chinese characters) returns `""` — a more significant version of the empty slug problem.

### 3.2 `estimate_tokens` (line 20)

```python
def estimate_tokens(text: str) -> int
```

Rough estimator: `word_count * 1.3`. Acceptable for English, likely underestimates for French (more multi-syllabic words) and overestimates for CJK.

**Gap**: Empty string returns 0 (correct). But the 1.3 multiplier is hardcoded — no language-awareness despite `user_language` config existing.

### 3.3 `parse_frontmatter` (line 25)

```python
def parse_frontmatter(text: str) -> tuple[dict, str]
```

Regex-based YAML frontmatter parser. Returns `({}, text)` if no frontmatter found.

**Findings**:
- **F-UTL-4**: The regex `^---\n(.*?\n)---\n(.*)` requires `\n` after the closing `---`. A file ending with `---` on the last line (no trailing newline) would fail to parse.
- **F-UTL-5**: No error handling around `yaml.safe_load()`. Malformed YAML in frontmatter would raise an unhandled `yaml.YAMLError`, propagating up to callers.
- **F-UTL-6**: Uses `re.DOTALL` which makes `.` match newlines, but the `.*?` in group 1 is non-greedy. This works correctly but depends on the regex engine finding the first `---\n` boundary.

---

## 4. `src/core/llm.py` — LLM Abstraction Layer

### 4.1 `_repaired_json` Context Manager (line 31)

```python
@contextmanager
def _repaired_json()
```

Monkey-patches `json.loads` globally to auto-repair malformed JSON from small LLMs. Uses `json-repair` library if available; no-ops if not installed.

**Implementation details**:
- Saves original `json.loads` (line 42)
- Patches with `_patched()` that tries original first, falls back to `_repair_json()` on `JSONDecodeError`
- During repair, temporarily restores original `json.loads` to avoid recursion (json-repair internally calls `json.loads`) (lines 58-62)
- Logs first repair as `WARNING`, subsequent as `DEBUG` (streaming produces many partial chunks)
- Restores original in `finally` block (line 69)

**Finding F-LLM-1**: This is a global monkey-patch on `json.loads`. In a multi-threaded context (stall detection uses threads), two concurrent `_repaired_json()` contexts could race on `json.loads`. The `_call_with_stall_detection` function enters `_repaired_json()` on the main thread, then spawns a worker thread that also uses `json.loads` via Instructor streaming. The worker thread sees the patched `json.loads` (intentional) but if the context manager exits while the worker is still running, `json.loads` is restored to original, potentially causing the worker to lose repair capability mid-stream.

### 4.2 `strip_thinking` (line 72)

```python
def strip_thinking(text: str) -> str
```

Removes `<think>...</think>` tags from reasoning models (Qwen3, DeepSeek-R1). Uses `re.DOTALL` for multiline think blocks. Returns stripped and `.strip()`-ed text.

No issues. Clean utility.

### 4.3 `load_prompt` (line 77)

```python
def load_prompt(name: str, config: Config, **variables: Any) -> str
```

Loads `prompts/{name}.md` and performs variable substitution.

**Auto-injected variables** (always available):
- `{user_language}` — from `config.user_language`
- `{categories_observations}` — comma-joined list from `config.categories.observations`
- `{categories_entity_types}` — comma-joined list from `config.categories.entity_types`
- `{categories_relation_types}` — comma-joined list from `config.categories.relation_types`

Uses `str.replace()` per variable (line 98) — avoids f-string/format-string issues with JSON braces in prompts.

**Findings**:
- **F-LLM-2**: No warning for unreplaced variables. If a prompt contains `{foo}` but no `foo` variable is passed, the literal `{foo}` remains in the output. Silent failure.
- **F-LLM-3**: Variable values are converted via `str(value)` (line 98). Complex objects would produce repr-strings. No type checking.

### 4.4 `_get_client` (line 103)

```python
def _get_client(step_config: LLMStepConfig) -> instructor.Instructor
```

Creates an Instructor-patched LiteLLM client using `instructor.Mode.MD_JSON`.

**MD_JSON mode**: Extracts JSON from markdown code blocks in LLM output. This avoids the `response_format` parameter which some local model servers (Ollama, LM Studio) do not support. The LLM is prompted to emit JSON inside a markdown code block, and Instructor parses it out.

**Finding F-LLM-4**: `step_config.api_base` is extracted into `kwargs` (line 107) but never passed anywhere — the `kwargs` dict is built but not used. The `api_base` is instead passed in the `create()` call kwargs inside `_call_structured` and `_call_with_stall_detection`. This `kwargs` construction in `_get_client` is dead code.

### 4.5 `_call_structured` (line 118)

```python
def _call_structured(
    step_config: LLMStepConfig,
    prompt: str,
    response_model: type[T],
) -> T
```

Standard Instructor call for short steps (arbitration, summary, consolidation, dream planning). Non-streaming.

**Call kwargs**: `model`, single `user` message, `response_model`, `max_retries`, `temperature`, optional `timeout` and `api_base`.

Wrapped in `_repaired_json()` context.

### 4.6 `StallError` (line 142)

```python
class StallError(TimeoutError): pass
```

Custom exception for stall detection. Subclasses `TimeoutError`.

### 4.7 `_call_with_stall_detection` (line 147)

```python
def _call_with_stall_detection(
    step_config: LLMStepConfig,
    prompt: str,
    response_model: type[T],
    stall_timeout: int = 30,
) -> T
```

Streaming LLM call with watchdog thread. Used only for extraction (long calls).

**Architecture**:
1. Main thread enters `_repaired_json()` context
2. Worker thread (`_do_call`, line 171) creates client, streams via `create_partial()`
3. Each chunk updates `last_activity` timestamp under lock (line 193-196)
4. Main thread polls every 2 seconds (line 217)
5. First-token grace: 2x stall_timeout before any token arrives (line 223)
6. On stall detection, sets `error` and `done` event (lines 230-235)

**Findings**:
- **F-LLM-5**: The worker thread is a daemon thread (line 210: `daemon=True`). On stall detection, the main thread raises the error but never joins the worker. The worker thread continues running in background until the process exits or the HTTP connection times out. This is a resource leak — the LLM connection stays open.
- **F-LLM-6**: `timeout` is set to `step_config.timeout * 3` (line 187) as the overall connection timeout. With default timeout=60, this means 180-second hard limit. But `stall_timeout` defaults to 30, meaning a slow-but-progressing response would be killed at 180s even though individual chunks arrive within 30s windows. In practice, `call_extraction` sets `stall_timeout = step_config.timeout` (line 257), making the connection timeout = 3x stall threshold.
- **F-LLM-7**: Race condition window: if `done.set()` is called by both the worker (line 208) and the watchdog (line 234) near-simultaneously, the error from the watchdog may overwrite a successful result. However, the `done.is_set()` check at line 218 means the watchdog exits the loop once the worker sets `done`, so this race is narrow.

### 4.8 Public `call_*` Functions

#### `call_extraction` (line 244)

```python
def call_extraction(chat_content: str, config: Config) -> RawExtraction
```

Step 1: Extracts facts from chat. Uses `_call_with_stall_detection`. Injects `{chat_content}` and `{json_schema}` into `extract_facts.md` prompt. `stall_timeout = config.llm_extraction.timeout`.

#### `call_arbitration` (line 263)

```python
def call_arbitration(
    entity_name: str, entity_context: str,
    candidates: list[dict], config: Config,
) -> EntityResolution
```

Step 3: Resolves ambiguous entities. Uses `_call_structured`. Formats candidates as bullet list. Injects into `arbitrate_entity.md`.

#### `call_context_generation` (line 286)

```python
def call_context_generation(enriched_data: str, config: Config) -> str
```

Legacy narrative mode. Returns free-text markdown (not Instructor-structured). Uses raw `litellm.completion()`. Strips thinking tags.

**Note**: Only used when `config.context_narrative = True`. Identified as legacy in CLAUDE.md.

#### `call_context_section` (line 316)

```python
def call_context_section(
    section_name: str, entities_dossier: str,
    rag_context: str, budget_tokens: int, config: Config,
) -> str
```

Per-section LLM context generation. Returns free-text markdown. Uses raw `litellm.completion()`. Used when `config.context_llm_sections = True`.

#### `call_fact_consolidation` (line 349)

```python
def call_fact_consolidation(
    entity_title: str, entity_type: str,
    facts_text: str, config: Config, max_facts: int = 50,
) -> FactConsolidation
```

Consolidates redundant observations. Uses `_call_structured` with `config.llm_context` step. Returns structured `FactConsolidation`.

**Finding F-LLM-8**: Uses `llm_context` config rather than `llm_consolidation`. This means the consolidation-specific model/temperature settings in `config.llm_consolidation` are ignored. The dedicated `llm_consolidation` config (line 103 of config.py) exists but this function does not use it.

#### `call_entity_summary` (line 368)

```python
def call_entity_summary(
    title: str, entity_type: str,
    facts: list[str], relations: list[str],
    tags: list[str], config: Config,
) -> str
```

Generates 1-3 sentence summary. Free-text output via raw `litellm.completion()`. Uses `config.llm_context`.

#### `call_relation_discovery` (line 413)

```python
def call_relation_discovery(
    entity_a_title, entity_a_type, entity_a_dossier,
    entity_b_title, entity_b_type, entity_b_dossier,
    config: Config,
) -> RelationProposal
```

Dream step 5: discover relations between entity pairs. Uses `_call_structured` with `config.llm_dream_effective`.

**Note**: `RelationProposal` is defined in `llm.py` (line 407), not in `models.py`. This is the only model defined outside `models.py`.

```python
class RelationProposal(BaseModel):
    action: str  # "relate" or "none"
    relation_type: str | None = None
    context: str = ""
```

**Finding F-LLM-9**: `RelationProposal.action` is `str` not `Literal["relate", "none"]`. `relation_type` is `str` not `RelationType`. No Pydantic validation on either field — LLM can return any string and it will be accepted. This model should be in `models.py` with proper typing.

#### `call_dream_plan` (line 436)

```python
def call_dream_plan(memory_stats: str, config: Config) -> DreamPlan
```

Lazy-imports `DreamPlan` from models (avoids circular import). Passes placeholder strings for candidate counts (`"(see stats)"`).

#### `call_dream_validate` (line 456)

```python
def call_dream_validate(step_name: str, changes_summary: str, config: Config) -> DreamValidation
```

Post-step validation. Lazy-imports `DreamValidation`. Uses `_call_structured` with `config.llm_dream_effective`.

### 4.9 LLM Call Pattern Summary

| Function | Step Config Used | Call Method | Output Type |
|----------|-----------------|-------------|-------------|
| `call_extraction` | `llm_extraction` | `_call_with_stall_detection` (streaming) | `RawExtraction` |
| `call_arbitration` | `llm_arbitration` | `_call_structured` | `EntityResolution` |
| `call_context_generation` | `llm_context` | Raw `litellm.completion` | `str` |
| `call_context_section` | `llm_context` | Raw `litellm.completion` | `str` |
| `call_fact_consolidation` | `llm_context` (!) | `_call_structured` | `FactConsolidation` |
| `call_entity_summary` | `llm_context` | Raw `litellm.completion` | `str` |
| `call_relation_discovery` | `llm_dream_effective` | `_call_structured` | `RelationProposal` |
| `call_dream_plan` | `llm_dream_effective` | `_call_structured` | `DreamPlan` |
| `call_dream_validate` | `llm_dream_effective` | `_call_structured` | `DreamValidation` |

**Pattern**: Structured outputs use Instructor (`_call_structured` or `_call_with_stall_detection`). Free-text outputs use raw `litellm.completion()` without `_repaired_json()` wrapping.

---

## 5. Consolidated Findings

### Critical

| ID | Location | Finding |
|----|----------|---------|
| F-LLM-8 | `llm.py:365` | `call_fact_consolidation` uses `config.llm_context` instead of `config.llm_consolidation`. The dedicated consolidation LLM config is defined in Config but never consumed. |
| F-LLM-1 | `llm.py:31-69` | Global `json.loads` monkey-patch in `_repaired_json()` is not thread-safe. Worker thread in stall detection shares the patched function, and early context exit could de-patch while worker still runs. |

### Structural

| ID | Location | Finding |
|----|----------|---------|
| F-MOD-8 | `models.py:91,132` | `GraphEntity` and `EntityFrontmatter` are near-duplicate classes with no shared base. Field additions must be manually synchronized. |
| F-LLM-9 | `llm.py:407-410` | `RelationProposal` defined in `llm.py` with untyped `action: str` and `relation_type: str`. Should be in `models.py` with `Literal` constraints matching `RelationType`. |
| F-CFG-1 | `config.py:158-266` | No config validation. Typos in YAML keys silently ignored. No schema validation or warning for unknown keys. |
| F-MOD-4 | `models.py:48-52` | `RawRelation` has no `supersedes` field. Relations cannot express temporal replacement (e.g., job change). |

### Validation Gaps

| ID | Location | Finding |
|----|----------|---------|
| F-MOD-1 | `models.py:34` | `RawObservation.content` has no `max_length`. 150-char limit enforced only in `store.py` post-consolidation. |
| F-MOD-2 | `models.py:35` | `RawObservation.tags` has no list length constraint. 3-tag limit enforced only in `store.py`. |
| F-MOD-3 | `models.py:37` | `RawObservation.date` is unconstrained `str`. No ISO format validation. |
| F-MOD-7 | `models.py:96` | `GraphEntity.importance` has no `[0,1]` constraint (unlike `RawObservation.importance`). |
| F-CFG-4 | `config.py:33-38` | `EmbeddingsConfig` has no check that `chunk_overlap < chunk_size`. |

### Minor / Cosmetic

| ID | Location | Finding |
|----|----------|---------|
| F-CFG-3 | `config.py:213` | `max_facts` loaded from top-level YAML, while `context_narrative`/`context_llm_sections` loaded from `memory:` section. Inconsistent nesting. |
| F-CFG-5 | `config.py:61-65` | `FAISSConfig` paths are `str` while `Config.memory_path`/`prompts_path` are `Path`. |
| F-LLM-4 | `llm.py:105-107` | `_get_client()` builds unused `kwargs` dict with `api_base`. Dead code. |
| F-LLM-5 | `llm.py:210` | Worker daemon thread never joined on stall. LLM connection leaks until process exit. |
| F-UTL-1 | `utils.py:13` | `slugify()` drops non-ASCII, causing collisions for accented names (French domain). |
| F-UTL-2 | `utils.py:11-17` | `slugify("")` and `slugify("...only non-ASCII...")` return `""`. No empty-slug guard. |
| F-UTL-4 | `utils.py:28` | `parse_frontmatter` regex requires trailing newline after closing `---`. |
| F-UTL-5 | `utils.py:32` | No `yaml.YAMLError` handling in `parse_frontmatter`. Malformed YAML propagates as unhandled exception. |

### Where New Models Would Be Needed

1. **Temporal relation supersession**: A `RelationSupersession` model or `supersedes` field on `RawRelation` to express "replaces previous relation" semantics (e.g., changing jobs).
2. **Config validation model**: A Pydantic model for `config.yaml` parsing would catch typos and enforce constraints at load time, replacing the manual `dict.get()` approach.
3. **Base entity model**: A shared `BaseEntityFields` class for both `GraphEntity` and `EntityFrontmatter` to eliminate field duplication.
4. **Typed SearchResult.relations**: Replace `list[dict]` with `list[GraphRelation]` or a dedicated `SearchRelation` model.
5. **Pipeline step result**: A unified `StepResult` model for tracking per-step outcomes in the pipeline (currently ad-hoc).

---

## 6. Category Sync Verification

Current state (2026-03-10): All three Literal types in `models.py` exactly match `config.yaml.example` categories. Verified:

- `ObservationCategory` (17 values) matches `categories.observations` (17 items)
- `EntityType` (9 values) matches `categories.entity_types` (9 items)
- `RelationType` (13 values) matches `categories.relation_types` (13 items)
- `categories.folders` covers all 9 entity types

No drift detected.
