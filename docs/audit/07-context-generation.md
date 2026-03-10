# Audit 07 -- Context Generation

**Date**: 2026-03-10
**Files audited**:
- `src/memory/context.py` (669 lines)
- `src/core/llm.py` (lines 286-346: `call_context_generation`, `call_context_section`)
- `prompts/context_template.md`
- `prompts/context_section.md`
- `prompts/context_instructions.md`
- `prompts/generate_context.md` (legacy)
- `src/pipeline/orchestrator.py` (lines 370-388: mode selection)
- `src/cli.py` (lines 88-133: `rebuild-all`, `context` commands)

---

## 1. Three Context Modes

### 1.1 Mode Selection Logic

**`src/pipeline/orchestrator.py:373`**:
```python
use_llm = consolidate and getattr(config, "context_llm_sections", False)
```

| Mode | Config | Active by default | Entry point |
|------|--------|-------------------|-------------|
| **Deterministic template** | (default) | **Yes** | `build_context()` at context.py:270 |
| **LLM per-section** | `context_llm_sections: true` | No | `build_context_with_llm()` at context.py:454 |
| **Narrative (legacy)** | `context_narrative: true` | No | `build_context_input()` + `generate_context()` at context.py:608-625 |

**Finding: Narrative mode is dead code.** The `context_narrative` flag is stored in `Config` (config.py:123) and loaded from YAML (config.py:211), but **no code path ever checks it**. Neither `orchestrator.py` nor `cli.py` branches on `context_narrative`. The functions `build_context_input()` (context.py:608) and `generate_context()` (context.py:623) are never called from any pipeline path. `call_context_generation()` in llm.py:286 is imported by context.py:9 but only used by the unreachable `generate_context()` wrapper.

**Finding: `rebuild-all` and `context` CLI commands always use deterministic mode.** `cli.py:124` and `cli.py:489` both hardcode `build_context()` -- they never check `context_llm_sections`. Only the orchestrator's `_finalize_pipeline()` (orchestrator.py:373) respects `context_llm_sections`, and only when `consolidate=True` (i.e., `memory run`, not `run-light`).

---

## 2. Function-by-Function Analysis

### 2.1 `_sort_facts_by_date(facts)` -- context.py:17-28

- **Signature**: `(facts: list[str]) -> list[str]`
- **Purpose**: Sort fact lines chronologically; undated facts appended at end.
- **Mechanism**: Calls `_parse_observation()` per line, sorts dated tuples by date string (ISO format, so lexicographic = chronological), appends undated in original order.
- **Note**: Stable sort for dated facts with identical dates (Python `sort` is stable).

### 2.2 `_content_similarity(text_a, text_b)` -- context.py:50-61

- **Signature**: `(text_a: str, text_b: str) -> float`
- **Purpose**: Blended similarity for fact deduplication.
- **Algorithm**: 50% stopword-filtered word Jaccard + 50% character trigram Jaccard.
- **Stopwords**: Bilingual French+English set (context.py:31-42), hardcoded as `frozenset`.
- **Trigrams**: Computed on sorted, joined, stopword-filtered words (context.py:57-58). This means trigrams operate on a canonical form, not the raw text.
- **Edge case**: Returns 0.0 if either filtered word set is empty (context.py:54-55).

### 2.3 `_deduplicate_facts_for_context(facts, threshold=0.35, max_per_category=5)` -- context.py:64-98

- **Signature**: `(facts: list[str], threshold: float = 0.35, max_per_category: int = 5) -> list[str]`
- **Purpose**: Remove near-duplicate facts within same category, then cap each category.
- **Algorithm**:
  1. For each fact line, parse its category and content.
  2. Compare content against all previously kept facts **in the same category** via `_content_similarity()`.
  3. If similarity > 0.35 (threshold), drop as duplicate.
  4. If category already has `max_per_category` facts, drop.
  5. Non-parseable lines always kept (context.py:78-80).
- **Order**: Preserves input order; later duplicates are removed.
- **Note**: Called from `_enrich_entity()` at context.py:223 with `max_per_category=3` for `ai_self` entities, `5` for all others.

### 2.4 `_group_facts_by_category(facts)` -- context.py:101-127

- **Signature**: `(facts: list[str]) -> dict[str, list[str]]`
- **Purpose**: Group fact lines by `[category]` prefix for cleaner output display.
- **Output**: `OrderedDict` of category -> list of display strings (date + content + valence + tags, without category prefix).
- **Non-parseable lines**: Grouped under `"_other"` key (context.py:126).

### 2.5 `_sort_by_cluster(entities, graph)` -- context.py:130-177

- **Signature**: `(entities: list[tuple[str, GraphEntity]], graph: GraphData) -> list[tuple[str, GraphEntity]]`
- **Purpose**: Sort entities so that members of the same connected component are adjacent (for "Top of Mind" section).
- **Algorithm**:
  1. Build adjacency restricted to the input entity set (context.py:144-148).
  2. BFS connected components (context.py:152-167).
  3. Sort by (cluster first-appearance index, original position) (context.py:177).
- **Effect**: Related entities cluster together while preserving score-descending order within clusters.

### 2.6 `_estimate_tokens(text)` -- context.py:180-182

- **Signature**: `(text: str) -> int`
- **Delegates to**: `src.core.utils.estimate_tokens` (words * 1.3).

### 2.7 `_enrich_entity(entity_id, entity, graph, memory_path)` -- context.py:185-248

- **Signature**: `(entity_id: str, entity: GraphEntity, graph: GraphData, memory_path: Path) -> str`
- **Purpose**: Build an enriched dossier string for a single entity.
- **Returns**: Multi-line string with header, tags, facts (grouped by category), related entities, and directional relations.

#### Path Traversal Guard (context.py:193-195)

```python
entity_path = (memory_path / entity.file).resolve()
if entity_path.is_relative_to(memory_path.resolve()) and entity_path.exists():
```

Validates that the resolved entity file path stays within `memory_path`. If a `GraphEntity.file` contained `../../etc/passwd`, the guard would reject it. Only reads facts if both conditions pass.

#### Fact Processing Pipeline (context.py:218-229)

1. Filter out `[superseded]` facts (context.py:221).
2. Sort chronologically via `_sort_facts_by_date()` (context.py:222).
3. Deduplicate via `_deduplicate_facts_for_context()` with `max_per_category=3` for ai_self, `5` otherwise (context.py:219-223).
4. Group by category via `_group_facts_by_category()` (context.py:225).

#### BFS Depth-1 Relations (context.py:203-208)

```python
related_ids = get_related(graph, entity_id, depth=1)
```

Calls `graph.get_related()` (graph.py:128-153) which does bidirectional BFS traversal over **all** graph relations. Collects neighboring entity titles and types as `"Title (type)"` strings. Displayed as a comma-separated "Related:" line.

#### Directional Relations (context.py:234-246)

Separately iterates **all** `graph.relations` to find edges from/to this entity. Displays them as:
- `-> relation_type TargetTitle` (outgoing)
- `<- relation_type SourceTitle` (incoming)

**Critical observation**: Both the "Related:" line (BFS) and the "Relations:" block (direct scan) show relation data. The BFS "Related:" line only shows titles, while the "Relations:" block shows direction and type. There is some redundancy.

### 2.8 `_collect_section(entities, graph, memory_path, token_budget)` -- context.py:251-267

- **Signature**: `(entities: list[tuple[str, GraphEntity]], graph: GraphData, memory_path: Path, token_budget: int) -> str`
- **Purpose**: Collect enriched dossiers for a list of entities, stopping when the token budget is exhausted.
- **Budget enforcement**: Iterates entities in order, estimates token cost per dossier, breaks when cumulative cost exceeds budget (but always includes at least one entity -- context.py:263: `if used + cost > token_budget and parts`).

### 2.9 `build_context(graph, memory_path, config)` -- context.py:270-411 (MAIN DETERMINISTIC FUNCTION)

- **Signature**: `(graph: GraphData, memory_path: Path, config: Config) -> str`
- **Returns**: Complete `_context.md` string.

#### Token Budget Calculation (context.py:292-297)

```python
reserved = 500
total_budget = max(config.context_max_tokens - reserved, 1000)
```

Default: `context_max_tokens=3000`, so `total_budget = 2500`. Each section gets a percentage:

```python
def section_budget(key: str) -> int:
    pct = budget.get(key, 10)  # default 10% if key missing
    return int(total_budget * pct / 100)
```

With example config budgets (ai_personality=8, identity=10, work=10, personal=10, top_of_mind=17, vigilances=10, history_recent=12, history_earlier=8, history_longterm=5, instructions=10):

| Section | % | Tokens (of 2500) |
|---------|---|-------------------|
| ai_personality | 8 | 200 |
| identity | 10 | 250 |
| work | 10 | 250 |
| personal | 10 | 250 |
| top_of_mind | 17 | 425 |
| vigilances | 10 | 250 |
| history_recent | 12 | 300 |
| history_earlier | 8 | 200 |
| history_longterm | 5 | 125 |
| instructions | 10 | 250 |
| **Total** | **100** | **2500** |

**Note**: The `instructions` budget is defined in config but never used programmatically -- custom instructions are injected verbatim without truncation.

#### Entity Selection (context.py:300-301)

```python
min_score = config.scoring.min_score_for_context  # default 0.3
all_top = get_top_entities(graph, n=50, include_permanent=True, min_score=min_score)
```

Calls `scoring.get_top_entities()` (scoring.py:226-252):
- All permanent-retention entities are **always** included regardless of score.
- Then top N entities with `score >= min_score` are added.
- Result can exceed 50 if there are many permanent entities (permanent count + up to 50 scored).

#### Section Categorization (context.py:305-389)

Entities are assigned to sections in this priority order:

1. **AI Personality** (context.py:305-311): `entity.type == "ai_self"` -- enriched inline, not via `_collect_section()`.
2. **Identity** (context.py:314-317): `entity.file.startswith("self/") and entity.type != "ai_self"` -- entities stored in `self/` folder (typically health entities).
3. **Work** (context.py:320-322): `entity.type in ("work", "organization")`.
4. **Personal** (context.py:325-329): `entity.type in ("person", "animal", "place")`.
5. **Top of Mind** (context.py:332-339): All remaining entities not yet assigned, capped at 10, sorted by score, then cluster-sorted.
6. **Vigilances** (context.py:342-357): Cross-cuts previously shown entities (except ai_self). Scans their facts for `[vigilance]`, `[diagnosis]`, `[treatment]` categories. Max 2 facts per entity.
7. **Brief History** (context.py:359-369): Entities still remaining after Top of Mind, split by recency:
   - Recent: `last_mentioned >= 30 days ago`
   - Earlier: `30 days ago > last_mentioned >= 1 year ago`
   - Long-term: `last_mentioned < 1 year ago`

**Finding: Assignment is mutually exclusive for sections 1-5.** Once an entity is added to `shown_ids`, it is excluded from later sections. But **Vigilances is additive** -- it re-scans `shown_ids` entities to extract safety-critical facts.

**Finding: An entity with `type="person"` stored in `self/` folder would be assigned to Identity (rule 2), not Personal (rule 4).** The folder-based check happens before the type-based check.

#### Template Assembly (context.py:402-411)

Six variables replaced via `str.replace()`:
- `{date}` -> ISO date
- `{user_language_name}` -> e.g., "French"
- `{ai_personality}` -> enriched ai_self dossiers
- `{sections}` -> all assembled section blocks
- `{available_entities}` -> pipe-separated list of up to 30 remaining entities
- `{custom_instructions}` -> verbatim content of `prompts/context_instructions.md`

### 2.10 `_rag_prefetch(entity_ids, graph, config, memory_path, max_results_per_entity=2)` -- context.py:414-451

- **Signature**: `(entity_ids: list[str], graph: GraphData, config: Config, memory_path: Path, max_results_per_entity: int = 2) -> str`
- **Purpose**: Pre-fetch related facts from FAISS for LLM per-section mode.
- **Output**: Up to 15 RAG result lines, each `[Title] chunk_preview[:200]`.
- **Self-reference filter**: Skips results whose `entity_id` is in the input set (context.py:442).
- **Graceful degradation**: Returns empty string if FAISS import fails (context.py:428-429).

### 2.11 `build_context_with_llm(graph, memory_path, config)` -- context.py:454-605

- **Signature**: `(graph: GraphData, memory_path: Path, config: Config) -> str`
- **Purpose**: LLM per-section mode. Same entity selection and section categorization as `build_context()`, but each section's dossier is passed through `call_context_section()` for LLM cleanup.

**Key differences from `build_context()`**:

1. Each section calls `_llm_section()` (context.py:506-535) which:
   - Builds raw dossier via `_enrich_entity()` (same as deterministic)
   - Gets RAG context via `_rag_prefetch()`
   - Calls `call_context_section()` (llm.py:316-346)
   - Falls back to raw dossier if LLM fails (context.py:534)

2. **No Brief History section** -- LLM mode stops at Vigilances, no history split.

3. **Vigilances remain deterministic** (context.py:555) -- explicitly noted as "no LLM needed for safety-critical data".

4. **Entity assignment for Personal section** (context.py:500-503) differs slightly: it explicitly removes entities already assigned to ai/identity/work from the personal list, while deterministic mode uses `shown_ids` incrementally. The effect is the same.

5. **`shown_ids` tracking bug**: In `_llm_section()`, entities are added to `shown_ids` (context.py:514) inside the closure. But for top-of-mind at context.py:547, the `shown_ids` filter works because `_llm_section` mutates the outer set. However, the top-of-mind entities are then re-added to `shown_ids` at context.py:553 (outside `_llm_section`), which is redundant but harmless.

### 2.12 `build_context_input(graph, memory_path, config)` -- context.py:608-620

- **Signature**: `(graph: GraphData, memory_path: Path, config: Config) -> str`
- **Purpose**: Build enriched dossier for narrative mode input. Only top 15 entities.
- **Status**: **Dead code** -- never called from any pipeline path.

### 2.13 `generate_context(enriched_input, config)` -- context.py:623-625

- **Signature**: `(enriched_input: str, config: Config) -> str`
- **Purpose**: Thin wrapper around `call_context_generation()`.
- **Status**: **Dead code** -- never called from any pipeline path.

### 2.14 `write_context(memory_path, content)` -- context.py:628-630

- **Signature**: `(memory_path: Path, content: str) -> None`
- **Purpose**: Write `_context.md` to disk. No atomic write pattern (unlike `save_graph()`).

### 2.15 `generate_index(graph)` -- context.py:633-662

- **Signature**: `(graph: GraphData) -> str`
- **Purpose**: Generate `_index.md` with entity tables grouped by type and all relations listed.
- **No token budget** -- outputs everything.

### 2.16 `write_index(memory_path, graph)` -- context.py:665-668

- **Signature**: `(memory_path: Path, graph: GraphData) -> None`

---

## 3. LLM Functions in `llm.py`

### 3.1 `call_context_generation(enriched_data, config)` -- llm.py:286-313

- **Prompt**: `prompts/generate_context.md`
- **Variables**: `{context_max_tokens}`, `{enriched_data}`, `{context_budget}`, `{date}`, `{user_language}`
- **Call**: Direct `litellm.completion()` (no Instructor, no streaming, no stall detection). Returns free text.
- **Config**: Uses `config.llm_context` step config.
- **Post-processing**: `strip_thinking()` applied.
- **Status**: Only reachable via dead-code `generate_context()` wrapper.

### 3.2 `call_context_section(section_name, entities_dossier, rag_context, budget_tokens, config)` -- llm.py:316-346

- **Prompt**: `prompts/context_section.md`
- **Variables**: `{section_name}`, `{entities_dossier}`, `{rag_context}`, `{budget_tokens}`, `{user_language}`
- **Call**: Direct `litellm.completion()` (no Instructor, no streaming, no stall detection). Returns free text.
- **Config**: Uses `config.llm_context` step config.
- **Post-processing**: `strip_thinking()` + `strip()`.
- **Note**: If `rag_context` is empty, substitutes `"No additional context available."` (llm.py:329).

---

## 4. Prompt Templates

### 4.1 `prompts/context_template.md`

The deterministic template with 6 variable slots. Used by both `build_context()` and `build_context_with_llm()`. Contains:
- Header with date and language instruction
- `{ai_personality}` section
- `{sections}` placeholder for all generated sections
- "Available in memory" section with `{available_entities}`
- "Extended memory access" section pointing to `search_rag` tool
- `{custom_instructions}` at the end

### 4.2 `prompts/context_section.md`

LLM per-section prompt. Instructs the LLM to:
- Merge duplicate entities
- Shorten facts > 120 chars
- Remove contradictory/redundant/nonsensical facts
- Correct type/tag errors
- Respect token budget
- Never invent information
- Keep content in `{user_language}`

### 4.3 `prompts/context_instructions.md`

User-editable custom instructions. Default content is a placeholder comment. Injected verbatim at the end of `_context.md` without token truncation.

### 4.4 `prompts/generate_context.md` (legacy)

Narrative mode prompt. Instructs LLM to write fluid prose with section budgets. **Dead code** -- only reachable via unreachable `generate_context()`.

---

## 5. How Wrong Relations End Up in Context (Louise/Anais Bug Propagation)

### 5.1 The Problem

If the graph contains an incorrect relation (e.g., `louise -> parent_of -> anais` when no such relationship exists), this bad relation appears in context through **three independent display paths** in `_enrich_entity()`:

#### Path A: BFS "Related:" line (context.py:203-208)

```python
related_ids = get_related(graph, entity_id, depth=1)
```

`get_related()` (graph.py:128-153) does **unfiltered** BFS over all graph relations. It collects every neighbor regardless of relation type, strength, or age. If Louise has a `parent_of` relation to Anais, Anais will appear in Louise's "Related:" line as `Anais (person)`.

#### Path B: Directional "Relations:" block (context.py:234-246)

```python
for rel in graph.relations:
    if rel.from_entity == entity_id:
        ...
    elif rel.to_entity == entity_id:
        ...
```

This is a full scan of **all** graph relations. No filtering by strength, age, or validity. The wrong relation appears as `-> parent_of Anais`.

#### Path C: Cluster sorting in "Top of Mind" (context.py:130-177)

Wrong relations influence entity grouping. If Louise and Anais are both in "Top of Mind", the wrong `parent_of` relation creates an adjacency that makes them cluster together, reinforcing the false connection.

### 5.2 Where Filtering Could Be Added

**No relation filtering exists anywhere in the context generation pipeline.** There are several insertion points where filters would help:

1. **In `_enrich_entity()` at context.py:234**: Filter relations by strength threshold (e.g., `rel.strength >= 0.3`) or by age (e.g., skip relations not reinforced in > 180 days). This would prevent weak/stale relations from appearing.

2. **In `get_related()` at graph.py:128**: Add optional `min_strength` parameter. Currently accepts no filtering parameters.

3. **In `_sort_by_cluster()` at context.py:144-148**: Filter adjacency by strength when building the cluster graph.

4. **In `build_context()` section assembly**: A post-processing step could validate relations before template assembly.

### 5.3 Root Cause Chain

The root cause is upstream (extraction or resolution creating a wrong relation), but the context layer **amplifies** the error because:
- It displays the relation in multiple formats (Related + Relations)
- It never validates relation plausibility
- It never filters by strength or recency
- The receiving LLM treats `_context.md` as ground truth

### 5.4 Relation Strength as a Natural Filter

The scoring system already computes `eff_strength = rel.strength * (days + 0.5)^(-decay_power)` for spreading activation. A wrong relation that is never reinforced will have:
- `strength = 0.5` (base, never grown via Hebbian)
- `mention_count = 1`
- Decaying effective strength

But this decay only affects the **score computation**, not the **context display**. The relation still appears in context at full visual prominence.

---

## 6. Detailed Token Budget Analysis

### 6.1 Budget Gaps

- **AI Personality**: Budget key is `"ai_personality"` but `_collect_section()` is NOT used for this section. Dossiers are collected without budget enforcement (context.py:306-311). The budget is only checked implicitly by the number of ai_self entities.

- **Vigilances**: No budget enforcement. All matching facts from all shown entities are included (context.py:342-357). Could grow unbounded with many health entities.

- **Available entities**: No budget. Up to 30 entities listed (context.py:399).

- **Custom instructions**: No budget. Injected verbatim (context.py:409).

- **Template overhead** (headers, separators): Covered by the 500-token `reserved` (context.py:292), which may be insufficient if there are many sections.

### 6.2 Budget Not Enforced Globally

Each section independently tracks its budget via `_collect_section()`, but there is no global check that the total output stays within `context_max_tokens`. The template, section headers, available entities, custom instructions, and vigilances are all outside budget control. The actual output can significantly exceed `context_max_tokens`.

---

## 7. Additional Findings

### 7.1 `_enrich_entity()` Reads Facts Twice for Vigilance Entities

For entities shown in context, `_enrich_entity()` reads the MD file once (context.py:197), and then the Vigilances loop reads the same file again (context.py:348-349). No caching layer exists.

### 7.2 Entity in Multiple Type Categories

An entity can only match one section due to the `shown_ids` exclusion set. But the priority order means:
- A `person` entity in `self/` folder goes to **Identity**, not **Personal**.
- A `work` entity in `self/` folder goes to **Identity**, not **Work** (folder check at line 314 happens before type check at line 320).

### 7.3 `get_top_entities()` Can Return More Than N

`scoring.get_top_entities(graph, n=50)` returns all permanent entities **plus** up to 50 scored entities (scoring.py:249: `len(result) < n + len(permanent)`). If there are 10 permanent entities, the result can have up to 60 entries.

### 7.4 Non-Atomic Write

`write_context()` (context.py:628-630) uses `Path.write_text()` directly, unlike `save_graph()` which uses temp file + `os.replace()` + lockfile. A crash during write could leave a truncated `_context.md`.

### 7.5 LLM Per-Section Mode Missing Brief History

`build_context_with_llm()` does not generate Brief History sections (Recent/Earlier/Long-term). These sections exist only in `build_context()` (context.py:359-389). Any entities that would have appeared in history are simply lost in LLM mode.

### 7.6 Stopword List Gaps

The `_STOPWORDS` set (context.py:31-42) covers common French and English words but is hardcoded. The system's `user_language` is not consulted. If the language were changed to German or Spanish, the deduplication quality would degrade.

### 7.7 Legacy Narrative Mode Prompt Language Mismatch

`prompts/generate_context.md` instructs "Write ALL content in {user_language}" and "Section headers in {user_language}" -- but the deterministic mode uses English section headers (`## Identity`, `## Work context`, etc.). If the user switches between modes, the language style would be inconsistent. Moot since narrative mode is dead code.

---

## 8. Summary of Issues by Severity

### High

| # | Issue | Location |
|---|-------|----------|
| 1 | **No relation filtering in context** -- wrong/stale/weak relations propagated to LLM without any strength, age, or validity check | context.py:203-246 |
| 2 | **Global token budget not enforced** -- AI personality, vigilances, available entities, custom instructions, and template overhead are outside budget control; output can exceed `context_max_tokens` | context.py:270-411 |

### Medium

| # | Issue | Location |
|---|-------|----------|
| 3 | **Narrative mode is dead code** -- `context_narrative` config flag loaded but never checked; `build_context_input()`, `generate_context()`, and `call_context_generation()` are unreachable | context.py:608-625, llm.py:286-313 |
| 4 | **LLM per-section mode missing Brief History** -- entities beyond Top of Mind are silently dropped | context.py:454-605 |
| 5 | **AI Personality section has no budget enforcement** -- all ai_self entities included regardless of token cost | context.py:305-311 |
| 6 | **Vigilances section has no budget cap** -- can grow unbounded | context.py:342-357 |
| 7 | **`rebuild-all` and `context` CLI commands ignore `context_llm_sections`** -- always use deterministic mode | cli.py:124, cli.py:489 |

### Low

| # | Issue | Location |
|---|-------|----------|
| 8 | **Double file read for vigilance entities** -- no fact caching between `_enrich_entity()` and vigilance scan | context.py:197, 348-349 |
| 9 | **Non-atomic context write** -- crash during write truncates `_context.md` | context.py:628-630 |
| 10 | **Hardcoded stopwords** -- dedup quality degrades for non-French/English languages | context.py:31-42 |
| 11 | **Folder-based Identity assignment overrides type-based section** -- can cause unexpected categorization | context.py:314 |
| 12 | **`get_top_entities()` can exceed N** -- permanent entities are additive, not counted against the N limit | scoring.py:226-252 |
