# Audit 04 — Graph Management & ACT-R Scoring

Deep audit of `src/memory/graph.py`, `src/memory/scoring.py`, `src/memory/mentions.py`, and the graph-related models in `src/core/models.py`.

---

## 1. Data Models (`src/core/models.py`)

### 1.1 GraphEntity (lines 91-108)

```python
class GraphEntity(BaseModel):
    file: str                    # relative path from memory root, e.g. "self/sciatica.md"
    type: EntityType             # closed literal — 9 values
    title: str
    score: float = 0.0
    importance: float = 0.0      # [0, 1] — running average set by enricher
    frequency: int = 0           # mention counter, bumped on each enrichment
    last_mentioned: str = ""     # ISO date string
    retention: Literal["short_term", "long_term", "permanent"] = "short_term"
    aliases: list[str] = []
    tags: list[str] = []
    mention_dates: list[str] = []      # recent ISO dates (windowed)
    monthly_buckets: dict[str, int] = {}  # "YYYY-MM" → count (aggregated old mentions)
    created: str = ""
    summary: str = ""
    negative_valence_ratio: float = 0.0  # [0, 1] — ratio of emotional facts
```

**Notes:**
- `score` is ephemeral — recomputed on every `recalculate_all_scores()` call. It is persisted in the graph JSON and MD frontmatter, but treated as a cache, not source of truth.
- `negative_valence_ratio` is computed from the markdown body during `rebuild_from_md()` (graph.py:220) but is NOT recomputed during normal enrichment. It only updates on full graph rebuild.
- `importance` is a running average maintained by the enricher, not the scoring module. The scoring module reads it as a constant.

### 1.2 GraphRelation (lines 111-121)

```python
class GraphRelation(BaseModel):
    from_entity: str = Field(alias="from", serialization_alias="from")
    to_entity: str = Field(alias="to", serialization_alias="to")
    type: RelationType           # closed literal — 13 values
    strength: float = 0.5        # [0.1, 1.0] — Hebbian stored strength
    created: str = ""            # ISO datetime
    last_reinforced: str = ""    # ISO datetime — reset on each co-occurrence
    mention_count: int = 1       # co-occurrence counter
    context: str = ""            # human-readable context from extraction

    model_config = {"populate_by_name": True}
```

**Field alias note:** `from_entity` serializes to/from JSON key `"from"` (Python reserved word workaround). `populate_by_name = True` allows construction via either `from_entity=` or `from=` keyword argument.

**Edge case:** `strength` defaults to 0.5 and is clamped to `[0.1, 1.0]` — the upper bound is enforced by `add_relation()` via `min(1.0, ...)`, the lower bound by `_apply_ltd()` via `max(0.1, ...)`. However, there is no validation constraint on the Pydantic model itself, so a deserialized value could theoretically be outside this range if the JSON was hand-edited.

### 1.3 GraphData (lines 124-127)

```python
class GraphData(BaseModel):
    generated: str = ""                          # ISO datetime of last save
    entities: dict[str, GraphEntity] = {}        # slug → GraphEntity
    relations: list[GraphRelation] = []          # flat list, no index
```

**Performance note:** Relations are stored as a flat list. Every lookup (duplicate check in `add_relation`, neighbor scan in `get_related`, spreading activation adjacency build) requires a full linear scan of all relations. For a graph with R relations, `add_relation` is O(R) per call, and `get_related` at depth 1 is O(R) per BFS step. This is acceptable for personal knowledge bases (typically < 1000 relations) but would not scale.

---

## 2. Graph Operations (`src/memory/graph.py`)

### 2.1 load_graph (lines 23-59)

```python
def load_graph(memory_path: Path) -> GraphData
```

**Purpose:** Load `_graph.json` with three-tier graceful degradation.

**Algorithm:**
1. If `_graph.json` does not exist → return empty `GraphData` with current timestamp.
2. Try to parse `_graph.json` via `json.loads` → `GraphData.model_validate()`.
3. On `JSONDecodeError | ValueError | KeyError` → log warning, try `_graph.json.bak`.
4. If `.bak` parses successfully → atomically overwrite the corrupt primary file with `.bak` content, return graph.
5. If `.bak` also fails → full rebuild from markdown files via `rebuild_from_md()`, then `save_graph()`.

**Edge cases:**
- **Empty graph:** Returns `GraphData(generated=<now>)` with no entities/relations. Safe for all downstream consumers.
- **Race condition:** No lock is acquired during load. If another process is mid-write, `json.loads` could see a truncated file. The three-tier fallback handles this — it will fall through to `.bak` or rebuild.
- **No .bak file:** If primary is corrupt and no backup exists, goes straight to MD rebuild.

**Observation:** The `.bak` restore path (line 50) calls `_atomic_write()` to overwrite the corrupt primary, but does NOT acquire the lockfile first. This is a minor inconsistency — `save_graph()` always acquires the lock, but this restore path bypasses it. In practice this is benign because it only triggers on corruption recovery, but it could race with a concurrent `save_graph()`.

### 2.2 save_graph (lines 62-77)

```python
def save_graph(memory_path: Path, graph: GraphData) -> None
```

**Purpose:** Persist `_graph.json` atomically with backup and lock protection.

**Algorithm:**
1. Acquire lockfile `_graph.lock` (see `_acquire_lock` below).
2. If `_graph.json` exists → `shutil.copy2()` to `_graph.json.bak` (preserves metadata).
3. Update `graph.generated` to current ISO timestamp.
4. Serialize via `graph.model_dump(by_alias=True)` — ensures `from_entity` serializes as `"from"`.
5. Write via `_atomic_write()` — temp file + `os.replace()`.
6. Release lockfile in `finally` block (always runs, even on exception).

**Atomicity guarantees:**
- **Lockfile:** `_graph.lock` prevents concurrent writes. Uses `O_CREAT | O_EXCL` for atomic creation (line 269). Stale locks older than 300 seconds are auto-removed (line 259).
- **Atomic write:** `_atomic_write()` (line 235) creates a temp file in the same directory (same filesystem), writes content, then `os.replace()` — which is atomic on POSIX. If the write fails mid-way, the temp file is cleaned up and the original `_graph.json` remains intact.
- **Backup:** `.bak` is created BEFORE the write, so if the process crashes during `_atomic_write()`, the previous version survives as `.bak`.

**Edge case — backup race:** `shutil.copy2()` is NOT atomic. If the process crashes between `copy2` start and `_atomic_write`, the `.bak` could be a partial copy. However, `os.replace` on the primary file is atomic, so the primary always remains valid or gets fully replaced.

### 2.3 _acquire_lock / _release_lock (lines 254-281)

```python
def _acquire_lock(lock_path: Path) -> None
def _release_lock(lock_path: Path) -> None
```

**Algorithm:**
1. If lock file exists and age > `LOCK_TIMEOUT_SECONDS` (300s) → delete it (stale).
2. If lock file exists and age <= 300s → raise `RuntimeError`.
3. Create lock atomically via `os.open(O_CREAT | O_EXCL | O_WRONLY)`.
4. Write PID and timestamp to lock file for debugging.
5. Release: simple `unlink()`, swallowing `FileNotFoundError`.

**Race condition:** Between the staleness check (line 256-263) and the `os.open` (line 269), another process could acquire the lock. The `O_CREAT | O_EXCL` ensures only one process wins — the loser gets `FileExistsError` and raises `RuntimeError`. This is correct.

**Missing feature:** There is no retry/wait mechanism. If the lock is held by another process, the caller gets an immediate exception rather than a backoff loop. For a single-user CLI tool this is acceptable.

### 2.4 _atomic_write (lines 235-251)

```python
def _atomic_write(filepath: Path, content: str) -> None
```

Creates temp file in the same directory as `filepath` (ensuring same-filesystem for atomic `os.replace`). Uses raw `os.write()` + `os.close()` + `os.replace()`. On any failure, cleans up the temp file and re-raises.

**Note:** Does not call `os.fsync()` before `os.replace()`. On a power failure between `os.write` and `os.replace`, the temp file contents might not be flushed to disk. For a CLI tool on modern filesystems with journaling, this is a negligible risk.

### 2.5 add_entity (lines 80-83)

```python
def add_entity(graph: GraphData, entity_id: str, entity: GraphEntity) -> GraphData
```

Simple dict assignment. Overwrites if entity_id already exists — no merge logic. Returns the mutated graph (in-place mutation + return for chaining).

### 2.6 update_entity (lines 86-93)

```python
def update_entity(graph: GraphData, entity_id: str, **updates) -> GraphData
```

Uses `setattr()` to patch fields. Silently ignores unknown keys (checked via `hasattr`). Silently no-ops if `entity_id` not in graph — no error, no warning.

### 2.7 add_relation — Hebbian Learning (lines 96-116)

```python
def add_relation(graph: GraphData, relation: GraphRelation, *, strength_growth: float = 0.05) -> GraphData
```

**Purpose:** Add a new relation or reinforce an existing one (Hebbian LTP — Long-Term Potentiation).

**Algorithm:**

1. **Duplicate detection:** Linear scan of all relations. Match criteria: `(from_entity, to_entity, type)` must all be equal.
2. **If match found — Hebbian reinforcement (LTP):**
   ```
   existing.mention_count += 1
   existing.last_reinforced = now()
   existing.strength = min(1.0, existing.strength + strength_growth)
   ```
   Where `strength_growth` defaults to 0.05 (configurable via `config.scoring.relation_strength_growth`).
   Context is only set if the existing relation had no context (line 107-108) — first-write-wins semantics.
3. **If no match — new relation:**
   Sets `created` and `last_reinforced` to current time if not already set. Appends to `graph.relations`.

**Hebbian learning formula:**
```
strength_new = min(1.0, strength_old + delta)
```
Where `delta = 0.05` by default. This is **linear growth with a hard ceiling** — not the classic Hebb rule (which is multiplicative). Starting from the default 0.5, it takes 10 co-occurrences to reach 1.0.

**Edge cases:**
- **Self-references:** Nothing prevents `from_entity == to_entity`. A self-referential relation would be accepted and reinforced normally. The spreading activation would then give an entity a bonus from its own base score.
- **Directionality:** The duplicate check is directional — `(A, B, affects)` and `(B, A, affects)` are different relations. However, `get_related()` and spreading activation treat relations bidirectionally.
- **No validation of entity existence:** `add_relation` does not verify that `from_entity` and `to_entity` exist in `graph.entities`. Orphan relations can be created and must be cleaned up later via `remove_orphan_relations()`.

### 2.8 remove_orphan_relations (lines 119-125)

```python
def remove_orphan_relations(graph: GraphData) -> GraphData
```

List comprehension filter: keeps only relations where both `from_entity` and `to_entity` exist in `graph.entities`. O(R) with O(1) dict lookups.

### 2.9 get_related — BFS Traversal (lines 128-153)

```python
def get_related(graph: GraphData, entity_id: str, depth: int = 1) -> list[str]
```

**Algorithm:** Standard BFS up to `depth` hops. Bidirectional — follows both `from_entity → to_entity` and `to_entity → from_entity` for each relation.

**Implementation detail:** Uses `queue.pop(0)` which is O(n) on a list. Should use `collections.deque` for O(1) popleft. Not a practical concern for small graphs.

**Edge cases:**
- Entity not in graph → returns empty list (line 130-131).
- `depth=0` → returns empty list (the starting entity is excluded from results).
- Cycles → handled by `visited` set.

**Performance:** For each node in the BFS, scans ALL relations (line 147). With E entities at depth D and R relations, worst case is O(E * R). Acceptable for small graphs.

### 2.10 get_aliases_lookup (lines 156-164)

```python
def get_aliases_lookup(graph: GraphData) -> dict[str, str]
```

Builds a case-insensitive lookup from entity ID, title, and all aliases to entity ID. Used by the resolver for fast alias matching.

**Edge case — collision:** If two entities share an alias (or one entity's alias matches another's title), the last one in iteration order wins. `dict[str, GraphEntity]` is insertion-ordered in Python 3.7+, so the winner is deterministic but depends on the order entities were added to the graph.

### 2.11 validate_graph (lines 167-184)

```python
def validate_graph(graph: GraphData, memory_path: Path) -> list[str]
```

Returns a list of warning strings. Checks:
1. Each entity's `file` field points to an existing file on disk.
2. Each relation's `from_entity` and `to_entity` exist in `graph.entities`.

Does NOT check: relation type validity, score ranges, date format correctness, or self-references.

### 2.12 rebuild_from_md (lines 187-230)

```python
def rebuild_from_md(memory_path: Path) -> GraphData
```

**Purpose:** Full graph reconstruction from markdown source files. Last-resort recovery mechanism.

**Algorithm:**
1. Recursively glob all `*.md` files under `memory_path`, sorted alphabetically.
2. Skip files in directories starting with `_` (e.g., `_archive/`, `_inbox/`) or in `chats/`.
3. For each file: parse YAML frontmatter, require `title` and `type` fields.
4. Create `GraphEntity` from frontmatter fields, using file stem as slug.
5. Compute `negative_valence_ratio` from the markdown body (line 220).
6. Parse `## Relations` section for wikilink patterns: `- relation_type [[Target]]`.
7. Convert target title to slug, create `GraphRelation`, add via `add_relation()`.

**`_compute_negative_valence_ratio` (lines 312-340):**
Scans the `## Facts` section. For each line starting with `- [`:
- Increment `total` counter.
- If line contains `[-]` → increment `negative_count`.
- Else if category (extracted via regex) is in `{vigilance, diagnosis, treatment}` → increment `negative_count`.
- Returns `round(negative_count / total, 4)` or 0.0 if no facts.

**Note:** Treatment facts with `[+]` valence (e.g., `[treatment] Started physiotherapy [+]`) are still counted as "emotional" because the category match (line 336) is checked in the `else` branch — only when `[-]` is NOT present. However, a treatment fact with `[-]` would be counted via the `[-]` branch, not double-counted. This means a treatment fact with `[+]` valence is treated as emotional, which models the amygdala's response to health-salient information regardless of positive/negative valence.

**`_parse_relations_from_body` (lines 289-309):**
- Regex: `r"- (\w+) \[\[(.+?)\]\]"` — requires exactly one space between relation type and `[[`. A line like `- affects  [[Target]]` (double space) would NOT match.
- Exceptions during `GraphRelation` construction (e.g., invalid `RelationType`) are silently swallowed (line 308).

---

## 3. ACT-R Scoring (`src/memory/scoring.py`)

### 3.1 _sigmoid (line 13-15)

```python
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
```

Standard logistic sigmoid mapping R → (0, 1). No overflow protection — for `x < -709`, `math.exp(-x)` would overflow to `inf`, returning 0.0 (correct). For `x > 709`, `math.exp(-x)` underflows to 0.0, returning 1.0 (correct). So it is numerically stable in practice despite no explicit clamping.

### 3.2 calculate_actr_base (lines 18-58)

```python
def calculate_actr_base(
    mention_dates: list[str],
    monthly_buckets: dict[str, int],
    decay_factor: float,
    today: date,
) -> float
```

**Purpose:** Compute ACT-R base-level activation B.

**Mathematical formula:**
```
B = ln( Σ_j  t_j^(-d) )
```
Where:
- `t_j` = days since mention `j`, floored at 0.5 (to avoid division by zero)
- `d` = decay factor (0.5 for long_term/permanent, 0.8 for short_term)
- Summation is over all mentions from both `mention_dates` and `monthly_buckets`

**Algorithm detail for `mention_dates` (lines 34-40):**
```python
for ds in mention_dates:
    d = parse_date(ds)
    days = max((today - d).days, 0) + 0.5   # minimum 0.5
    summation += days ** (-decay_factor)
```
- Handles both ISO date strings (`"2026-03-07"`) and datetime strings with `T` separator.
- Invalid/unparseable dates are silently skipped.

**Algorithm detail for `monthly_buckets` (lines 43-53):**
```python
for bucket_key, count in monthly_buckets.items():
    year, month = parse("YYYY-MM")
    mid = date(year, month, 15)         # mid-month representative
    days = max((today - mid).days, 0) + 0.5
    summation += count * (days ** (-decay_factor))
```
- Each bucket contributes `count` mentions, all assumed to occur on the 15th.
- This is an approximation — for a month with 30 mentions, they'd ideally be spread across 30 different days. The mid-month approximation underestimates the contribution of recent days and overestimates distant ones. The error is small for buckets more than a few months old (which is when consolidation happens).

**Edge case — no mentions at all:** Returns `-5.0` (line 57). This yields `sigmoid(-5.0) ≈ 0.0067`, which is below the default retrieval threshold of 0.05, so the entity will be forgotten (score → 0.0) unless it has very high importance or spreading bonus.

**Edge case — future dates:** If a mention date is in the future, `(today - d).days` is negative, `max(..., 0)` clamps to 0, so `days = 0.5`. The mention contributes `0.5^(-d)` ≈ 1.41 (for d=0.5) or 1.74 (for d=0.8). This is the maximum possible contribution per mention — a reasonable handling of clock skew.

### 3.3 calculate_score (lines 61-108)

```python
def calculate_score(
    entity: GraphEntity,
    config: Config,
    today: date | None = None,
    spreading_bonus: float = 0.0,
) -> float
```

**Full formula:**
```
score = sigmoid(B + β + w_s × S + E)
```
Where:
- `B` = `calculate_actr_base(mention_dates, monthly_buckets, d, today)`
- `d` = `decay_factor_short_term` (0.8) if retention is short_term, else `decay_factor` (0.5)
- `β` = `importance × importance_weight` (default: importance × 0.3)
- `w_s` = `spreading_weight` (default: 0.2)
- `S` = `spreading_bonus` (passed in from spreading activation pass)
- `E` = `negative_valence_ratio × emotional_boost_weight` (default: ratio × 0.15)

**Post-sigmoid adjustments (lines 100-107):**

1. **Permanent floor:** If `retention == "permanent"` → `score = max(score, permanent_min_score)` (default 0.5). Permanent entities never fall below 0.5.
2. **Retrieval threshold (true forgetting):** If `retention != "permanent"` AND `score < retrieval_threshold` (default 0.05) → `score = 0.0`. This models ACT-R's retrieval failure — memories below the threshold become completely inaccessible, not just weak.

**Numerical example — typical entity:**
- Mentioned 3 times in the last week: B ≈ ln(3 × 3.5^(-0.5)) ≈ ln(1.60) ≈ 0.47
- importance = 0.7, β = 0.7 × 0.3 = 0.21
- No relations: S = 0
- No emotional facts: E = 0
- activation = 0.47 + 0.21 = 0.68
- score = sigmoid(0.68) ≈ 0.66

**Numerical example — forgotten entity:**
- Last mentioned 365 days ago, once: B ≈ ln(365.5^(-0.5)) ≈ ln(0.052) ≈ -2.96
- importance = 0.3, β = 0.09
- activation ≈ -2.87
- score = sigmoid(-2.87) ≈ 0.054 → above threshold (barely)
- At 500 days: B ≈ ln(500.5^(-0.5)) ≈ -3.11, activation ≈ -3.02, score ≈ 0.047 → BELOW threshold → score = 0.0 (forgotten)

### 3.4 Spreading Activation — Two-Pass Algorithm (lines 111-177)

```python
def spreading_activation(
    graph: GraphData,
    config: Config,
    today: date | None = None,
) -> dict[str, float]
```

**Purpose:** Compute a spreading activation bonus for each entity, modeling how activation propagates through the associative network.

#### Pass 1 — Base Scores (lines 130-135)

For every entity, compute:
```
base_score[e] = sigmoid(B_e + β_e)
```
This is the score WITHOUT spreading activation and WITHOUT emotional boost. It represents the entity's intrinsic activation level.

#### Effective Relation Strength (lines 141-155)

For each relation, compute time-decayed effective strength using power-law decay:
```
eff_strength = strength × (days_since_reinforced + 0.5)^(-relation_decay_power)
```

Where:
- `strength` = stored Hebbian strength (0.1 to 1.0)
- `days_since_reinforced` = days since `last_reinforced` date
- `relation_decay_power` = 0.3 (default)

**Fallback:** If `last_reinforced` is empty or unparseable → `days_since = 365.0` (line 149).

**Mathematical behavior:**
- A relation reinforced today: `eff = strength × 0.5^(-0.3) ≈ strength × 1.23`
- A relation reinforced 30 days ago: `eff = strength × 30.5^(-0.3) ≈ strength × 0.33`
- A relation reinforced 365 days ago: `eff = strength × 365.5^(-0.3) ≈ strength × 0.15`

The power-law decay is much slower than exponential — old relations maintain influence for a long time, consistent with ACT-R's rational analysis of memory.

Relations are added **bidirectionally** to the adjacency list (lines 154-155). A relation `A → B` creates edges in both directions: `adjacency[B].append((A, eff))` and `adjacency[A].append((B, eff))`.

#### Pass 2 — Spreading Bonus (lines 158-177)

For each entity `i`:
```
S_i = Σ_j (eff_strength_ij × base_score_j) / Σ_j (eff_strength_ij)
```

This is a **weighted average** of neighbor base scores, weighted by effective relation strength. The normalization by `total_strength` ensures the bonus is in [0, 1] (since base scores are sigmoid outputs in [0, 1]).

**Edge cases:**
- Entity with no relations: `spreading[eid] = 0.0` (line 161).
- Entity with relations but all neighbors missing from `base_scores` (orphan relations): the numerator stays 0, result is 0.0. Note: `neighbor_id in base_scores` check (line 172) prevents KeyError but means orphan neighbors contribute nothing.
- All effective strengths sum to 0 (theoretically impossible since strength >= 0.1 and power-law decay never reaches 0): returns 0.0 (line 167).
- **Self-referential relations:** If an entity has a self-loop, it would appear in its own adjacency list. Its own base score would be included in the weighted average. This inflates its spreading bonus — a minor bug if self-references exist.

**Interaction with final score:** The spreading bonus is multiplied by `spreading_weight` (0.2) in `calculate_score`. Since `S_i` is in [0, 1] and `spreading_weight = 0.2`, the maximum contribution to activation is +0.2. Through the sigmoid, this shifts the score by at most ~0.05 (at the steepest part of the sigmoid curve).

### 3.5 Long-Term Depression (_apply_ltd) (lines 180-199)

```python
def _apply_ltd(graph: GraphData, config: Config, today: date) -> None
```

**Purpose:** Weaken stored relation strengths for relations that haven't been reinforced recently. This is the "forgetting" counterpart to Hebbian reinforcement in `add_relation()`.

**Formula:**
```
if days_since_reinforced > 90:
    strength_new = max(0.1, strength_old × exp(-days / halflife))
```

Where:
- `days` = days since `last_reinforced`
- `halflife` = `relation_ltd_halflife` (default: 360 days)
- Floor: 0.1 (relations never fully disappear)

**Trigger condition:** Only applies when `days > 90`. Relations reinforced within the last 90 days retain their full stored strength.

**Decay behavior:**
- At 90 days: `decay_factor = exp(-90/360) ≈ 0.78` → strength × 0.78
- At 180 days: `decay_factor = exp(-180/360) ≈ 0.61` → strength × 0.61
- At 360 days: `decay_factor = exp(-360/360) ≈ 0.37` → strength × 0.37
- At 720 days: `decay_factor = exp(-720/360) ≈ 0.14` → approaches floor of 0.1

**Important nuance — LTD vs. effective strength decay:**
There are TWO independent decay mechanisms on relation strength:
1. **LTD (this function):** Modifies the **stored** `strength` field permanently. Exponential decay. Only triggers after 90 days. Applied once per `recalculate_all_scores()` call.
2. **Power-law decay in spreading activation:** Computes a **transient** `effective_strength` that does NOT modify the stored value. Power-law decay (`(t+0.5)^(-0.3)`). Applied on every spreading activation computation.

These compound: a relation reinforced 200 days ago has:
- Stored strength (after LTD): `0.5 × exp(-200/360) ≈ 0.29`
- Effective strength (in spreading): `0.29 × 200.5^(-0.3) ≈ 0.29 × 0.18 ≈ 0.052`

**Mutation warning:** `_apply_ltd` modifies `graph.relations` in place. It is called at the START of `recalculate_all_scores()` (line 215), before spreading activation. This means spreading activation uses the freshly-decayed strengths, which is the intended behavior.

**Edge case — repeated scoring:** If `recalculate_all_scores()` is called multiple times in the same run (e.g., during dream mode steps), LTD is applied each time. However, `exp(-days/halflife)` is idempotent for the same `days` value — re-applying the same decay factor on already-decayed strength just decays further. This could over-decay if called repeatedly. In practice, `recalculate_all_scores()` is typically called once per pipeline run.

### 3.6 recalculate_all_scores (lines 202-223)

```python
def recalculate_all_scores(
    graph: GraphData,
    config: Config,
    today: date | None = None,
) -> GraphData
```

**Orchestration function:**
1. Apply LTD to all relations (`_apply_ltd`).
2. Compute spreading activation bonuses for all entities.
3. For each entity: `entity.score = calculate_score(entity, config, today, spreading_bonus=bonuses[eid])`.
4. Return the mutated graph.

**Note:** This function mutates the graph in place (both relation strengths via LTD and entity scores). The return value is the same object, provided for chaining convenience.

### 3.7 get_top_entities (lines 226-252)

```python
def get_top_entities(
    graph: GraphData,
    n: int,
    include_permanent: bool = True,
    min_score: float = 0.0,
) -> list[tuple[str, GraphEntity]]
```

Returns permanent entities (always included) plus the top N non-permanent entities above `min_score`, sorted by score descending.

**Edge case:** The cap is `n + len(permanent)` total results (line 249). So if `n=50` and there are 5 permanent entities, the result can have up to 55 entries. This means permanent entities do NOT count against the `n` limit.

---

## 4. Mention Windowing (`src/memory/mentions.py`)

### 4.1 add_mention (lines 6-18)

```python
def add_mention(
    date_iso: str,
    mention_dates: list[str],
    monthly_buckets: dict[str, int],
    window_size: int = 50,
) -> tuple[list[str], dict[str, int]]
```

**Algorithm:**
1. Append `date_iso` to `mention_dates`.
2. If `len(mention_dates) > window_size` → call `consolidate_window()`.
3. Return the (possibly consolidated) lists.

**Note:** Mutates `mention_dates` in place (line 13: `mention_dates.append()`), then potentially replaces it via `consolidate_window()` return value. Callers MUST use the returned tuple, not the original references.

### 4.2 consolidate_window (lines 21-39)

```python
def consolidate_window(
    mention_dates: list[str],
    monthly_buckets: dict[str, int],
    window_size: int = 50,
) -> tuple[list[str], dict[str, int]]
```

**Algorithm:**
1. If `len(mention_dates) <= window_size` → return unchanged.
2. Sort `mention_dates` chronologically (lexicographic sort works for ISO dates).
3. `overflow = len - window_size` → number of oldest dates to consolidate.
4. Take the `overflow` oldest dates, move each into `monthly_buckets` by extracting `d[:7]` (the `"YYYY-MM"` prefix).
5. Return the `window_size` most recent dates plus the updated buckets.

**Design rationale:** This creates a two-tier temporal resolution:
- **Recent mentions (mention_dates):** Day-level precision, used for fine-grained ACT-R recency scoring.
- **Old mentions (monthly_buckets):** Month-level precision, used for long-tail activation via mid-month approximation.

The window_size of 50 means the 50 most recent mentions retain full date precision. For an entity mentioned daily, this covers ~7 weeks of precise history.

**Edge case — duplicate dates:** If the same date appears multiple times, each occurrence is counted separately both in `mention_dates` and when consolidated into `monthly_buckets`. This is intentional — multiple mentions on the same day should increase activation.

**Edge case — date format:** Relies on `d[:7]` to extract the month key. Works for `"2026-03-07"` → `"2026-03"` and for `"2026-03-07T14:30:00"` → `"2026-03"`. Would break for non-ISO formats.

---

## 5. Cross-Cutting Analysis

### 5.1 Hebbian Learning + LTD Lifecycle

The full lifecycle of a relation's strength:

```
Creation:        strength = 0.5  (default in GraphRelation model)
Co-occurrence:   strength = min(1.0, strength + 0.05)   [add_relation, Hebbian LTP]
Aging (>90d):    strength = max(0.1, strength × e^(-days/360))   [_apply_ltd, LTD]
Activation use:  eff = strength × (days + 0.5)^(-0.3)   [spreading_activation, transient]
```

**Steady-state analysis:**
- A relation reinforced every 30 days: strength grows by +0.05 per reinforcement, reaching 1.0 after 10 reinforcements. LTD never triggers (always < 90 days since last reinforcement). Effective strength stays high.
- A relation reinforced once then abandoned: strength starts at 0.5, decays to 0.39 at 90d, 0.30 at 180d, 0.18 at 360d, floors at 0.1. Effective strength compounds further via power-law decay.

### 5.2 Emotional Modulation (Amygdala Model)

**Computation:** `negative_valence_ratio` is computed in `_compute_negative_valence_ratio()` (graph.py:312-340) during `rebuild_from_md()`.

**What counts as "emotional":**
1. Any fact with `[-]` valence marker.
2. Any fact in categories `{vigilance, diagnosis, treatment}` WITHOUT `[-]` (to avoid double-counting).

**Effect on scoring:**
```
emotional_boost = negative_valence_ratio × emotional_boost_weight (0.15)
```

Maximum boost: ratio = 1.0 → boost = +0.15 to activation → approximately +0.037 to score at the sigmoid midpoint. This is a subtle but meaningful effect — it can be the difference between a score above or below the retrieval threshold.

**When it updates:** ONLY during `rebuild_from_md()`. Normal enrichment (adding new observations) does NOT recompute the ratio. This means the emotional modulation is stale between full rebuilds. A newly added `[-]` fact won't affect scoring until the next `rebuild-all` or corruption recovery.

**Recommendation:** Consider computing `negative_valence_ratio` during enrichment as well, not just during full rebuild.

### 5.3 Retrieval Threshold — True Forgetting

The retrieval threshold (default 0.05) implements ACT-R's concept of retrieval failure:

```python
if entity.retention != "permanent" and score < retrieval_threshold:
    score = 0.0
```

This creates a **discontinuity** in the score function. An entity with a true score of 0.049 gets mapped to 0.0, while one at 0.051 keeps its score. This is by design — it models the all-or-nothing nature of memory retrieval in ACT-R theory.

**Implications:**
- Entities at score 0.0 are invisible to `get_top_entities()` with any positive `min_score`.
- They still exist in the graph and can be "resurrected" by new mentions (which reset their mention_dates, boosting B).
- The context builder uses `min_score_for_context = 0.3`, well above the threshold.

### 5.4 Power-Law vs. Exponential Decay

The system uses two distinct decay functions for different purposes:

| Mechanism | Function | Formula | Purpose |
|-----------|----------|---------|---------|
| ACT-R base activation | Power-law | `t^(-d)` | Individual mention recency |
| Effective relation strength | Power-law | `(t+0.5)^(-p)` | Transient activation spread |
| LTD stored strength | Exponential | `e^(-t/τ)` | Permanent synapse weakening |
| Mention window | None | Exact dates → monthly buckets | Temporal resolution trade-off |

Power-law decay is characteristic of ACT-R and models the empirical finding that memory follows a power law of forgetting (Anderson & Schooler, 1991). Exponential decay for LTD models synaptic depression in neuroscience — a different mechanism operating on a different timescale.

### 5.5 Why There Is No `remove_relation()` Function

The graph module provides `add_relation()`, `remove_orphan_relations()`, and `add_entity()`, but conspicuously lacks a `remove_relation()` function. This is an intentional architectural decision rooted in the cognitive model:

1. **Memories fade, they don't delete.** The ACT-R + Hebbian model handles relation weakening through two natural mechanisms: LTD (stored strength decay) and power-law effective strength decay. A relation that hasn't been reinforced for years will have near-zero effective strength, contributing essentially nothing to spreading activation. There's no need to explicitly delete it.

2. **The strength floor of 0.1 preserves traces.** Even fully decayed relations maintain a minimum stored strength of 0.1. This models the psychological phenomenon of "savings" — relearning something forgotten is faster than learning it anew. A dormant relation can be rapidly restrengthened by a single co-occurrence via `add_relation()`.

3. **Orphan cleanup is structural, not semantic.** `remove_orphan_relations()` handles the only case where deletion is necessary: when an entity is removed and its relations become dangling pointers. This is a data integrity operation, not a forgetting operation.

4. **Dream mode archives, not deletes.** The dead-pruning step in dream mode moves low-score entities to `_archive/` rather than deleting them. Relations to archived entities become orphans and are cleaned up by `remove_orphan_relations()`.

5. **Practical consideration:** A `remove_relation()` would require specifying the exact `(from, to, type)` tuple, and there's currently no pipeline step that would need to explicitly remove a relation. All relation changes are additive (reinforcement) or passive (decay).

### 5.6 Potential Issues and Observations

1. **LTD idempotency:** `_apply_ltd` is NOT idempotent across multiple calls in the same session. If `recalculate_all_scores()` is called twice, LTD is applied twice, compounding the decay. The `round(..., 4)` on line 199 prevents floating-point drift but not logical over-decay. In the dream pipeline, `recalculate_all_scores` is called in step 9 (rescore), and the enricher also calls it — potentially double-decaying within one dream run.

2. **`negative_valence_ratio` staleness:** As noted in 5.2, this value is only recomputed during `rebuild_from_md()`, not during normal enrichment. New emotional facts don't affect scoring until the next full rebuild.

3. **No relation indexing:** Relations are stored as a flat list with no secondary index. `add_relation()` does a full scan for duplicate detection. For graphs with thousands of relations, this could become a bottleneck.

4. **`load_graph` backup restore skips lock:** Line 50 calls `_atomic_write()` directly without acquiring `_graph.lock`, unlike `save_graph()`. This is a minor consistency issue.

5. **Self-referential relations:** No validation prevents `from_entity == to_entity`. A self-loop would cause the entity to boost its own spreading activation score, creating a feedback artifact.

6. **`update_entity` silent failures:** Lines 88-93 silently ignore both missing entity IDs and invalid field names. No logging or error signaling.

7. **Monthly bucket approximation:** Using mid-month (15th) for all consolidated dates introduces a systematic bias. For a bucket with 30 mentions spread across a month, the approximation treats them all as occurring on the 15th. For recent months this undervalues early-month mentions and overvalues late-month ones. The error diminishes for buckets several months old.

---

## 6. File Reference Index

| File | Key Lines | Content |
|------|-----------|---------|
| `src/core/models.py:91-108` | `GraphEntity` model definition |
| `src/core/models.py:111-121` | `GraphRelation` model definition with alias workaround |
| `src/core/models.py:124-127` | `GraphData` container model |
| `src/core/config.py:42-57` | `ScoringConfig` defaults |
| `src/memory/graph.py:23-59` | `load_graph` — three-tier recovery |
| `src/memory/graph.py:62-77` | `save_graph` — atomic write + lock + backup |
| `src/memory/graph.py:96-116` | `add_relation` — Hebbian LTP |
| `src/memory/graph.py:128-153` | `get_related` — BFS traversal |
| `src/memory/graph.py:187-230` | `rebuild_from_md` — full MD reconstruction |
| `src/memory/graph.py:235-251` | `_atomic_write` — temp file + os.replace |
| `src/memory/graph.py:254-273` | `_acquire_lock` — O_CREAT\|O_EXCL atomic lock |
| `src/memory/graph.py:312-340` | `_compute_negative_valence_ratio` — emotional modulation source |
| `src/memory/scoring.py:18-58` | `calculate_actr_base` — B = ln(Σ t^(-d)) |
| `src/memory/scoring.py:61-108` | `calculate_score` — full scoring formula |
| `src/memory/scoring.py:111-177` | `spreading_activation` — two-pass algorithm |
| `src/memory/scoring.py:180-199` | `_apply_ltd` — Long-Term Depression |
| `src/memory/scoring.py:202-223` | `recalculate_all_scores` — orchestrator |
| `src/memory/scoring.py:226-252` | `get_top_entities` — top-N with permanent inclusion |
| `src/memory/mentions.py:6-18` | `add_mention` — append + overflow check |
| `src/memory/mentions.py:21-39` | `consolidate_window` — mention_dates → monthly_buckets |
