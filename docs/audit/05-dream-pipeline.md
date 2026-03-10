# Audit 05 — Dream Pipeline

Deep audit of the dream mode pipeline: brain-like memory reorganization during idle time.

**Files audited:**
- `src/pipeline/dream.py` (main pipeline, all 10 steps)
- `src/pipeline/dream_dashboard.py` (Rich Live UI)
- `src/core/llm.py:349-468` (LLM call functions)
- `src/core/models.py:211-233` (data models)
- `src/memory/store.py:184-253` (fact consolidation persistence)
- `src/pipeline/indexer.py:259-293` (unextracted doc helpers)
- `prompts/dream_plan.md`, `prompts/dream_validate.md`, `prompts/discover_relations.md`, `prompts/consolidate_facts.md`

---

## 1. Pipeline Entry Point

**Function:** `run_dream()` at `dream.py:44-147`

**Signature:** `run_dream(config, console, *, dry_run=False, step=None) -> DreamReport`

**Flow:**
1. Always calls `_step_load()` first to get graph + entity paths.
2. If `step` is provided, runs only that single step. Otherwise, invokes the LLM coordinator to plan which steps to run.
3. Wraps execution in a `DreamDashboard` context manager (Rich Live terminal UI).
4. Iterates steps 1-10 sequentially. Steps not in `steps_to_run` are marked "skipped" on the dashboard.
5. Each step wrapped in try/except: on failure, dashboard shows `fail_step()` and error is appended to `report.errors`.

**DreamReport** (`dream.py:30-41`): mutable accumulator with counters for each step type plus an `errors` list.

---

## 2. Coordinator

### 2.1 `_collect_dream_stats()` — `dream.py:153-254`

**Purpose:** Gather memory state metrics to feed the LLM planner.

**Algorithm:**
1. Counts `total_entities` and `total_relations` from the graph.
2. **Unextracted docs:** Calls `indexer.list_unextracted_docs(manifest_path)` — scans FAISS manifest for entries with `source_type == "document"` and `entity_extracted == False`. Silently catches all exceptions.
3. **Consolidation candidates:** For each entity path, reads the MD file via `read_entity()`, filters out `[superseded]` facts, checks if `len(live_facts) > config.get_max_facts(entity.type)`. The threshold is per-type from `config.max_facts` with a default of 50 (`config.py:142-144`).
4. **Merge candidates:** O(n^2) pairwise comparison of all entity slugs. Two entities are merge candidates if they share the same `type` AND have overlapping alias sets (lowercased title + aliases). Deduped via sorted-pair set.
5. **Prune candidates:** Entities with `score < 0.1`, `frequency <= 1`, `retention != "permanent"`, no relations, and `age > 90 days`. Age parsed from `entity.created` with silent exception handling.
6. **Summary candidates:** Entities where `entity.summary` is falsy.
7. **Cluster analysis:** BFS connected components over the relation graph. Records `clusters` count and `largest_cluster` size.
8. Formats all counts into a newline-separated string like `"- Total Entities: 42"`.

**Returns:** `(formatted_stats_string, counts_dict)`

**Observations:**
- Cluster stats are computed but only passed through the stats string — the LLM planner sees them but no step directly uses them.
- The O(n^2) merge candidate scan is duplicated identically in `_step_merge_entities()` (lines 414-429). This is redundant work.
- Exception handling on prune candidate date parsing means entities with malformed `created` dates silently pass the age check and become prune candidates. This is arguably too permissive.

### 2.2 LLM Plan Generation — `dream.py:68-78`

**Flow:**
1. Calls `call_dream_plan(stats_text, config)` from `llm.py:436-453`.
2. `call_dream_plan` loads `prompts/dream_plan.md`, injects `{memory_stats}`, `{json_schema}` (from `DreamPlan.model_json_schema()`), and placeholder strings `"(see stats)"` for all candidate count variables.
3. Uses `_call_structured(config.llm_dream_effective, prompt, DreamPlan)` — standard Instructor call, no stall detection.
4. Returns `DreamPlan(steps: list[int], reasoning: str)`.

**Prompt analysis** (`dream_plan.md`):
- The prompt lists steps 1-9 (Load through Rebuild) but the code has 10 steps (1-10). Step numbering mismatch: the prompt calls Rescore "step 8" and Rebuild "step 9", but the code uses steps 9 and 10 respectively. **Bug: the prompt is out of sync with the code. Steps 6 (Transitive) is missing from the prompt entirely.**
- The prompt injects `{unextracted_docs}`, `{consolidation_candidates}` etc. directly into step descriptions, but `call_dream_plan()` passes `"(see stats)"` for all of these. The actual counts are in `{memory_stats}`. This means the LLM sees duplicate information: once in the stats block with real numbers, once in the step descriptions as `"(see stats)"`.
- Rules say "Step 1 (Load) is always included" and "Steps 8 and 9 should be included if ANY other step runs" — but the code enforces step 1 inclusion independently (`dream.py:81-82`), so the prompt rule is redundant for step 1. Steps 8/9 (which are actually 9/10 in code) are NOT enforced by code — they rely entirely on the LLM following the prompt instruction.

**Fallback:** On any exception from `call_dream_plan`, falls back to `list(range(1, 11))` — all 10 steps (`dream.py:77-78`).

**Step 1 enforcement:** `dream.py:81-82` always inserts step 1 at the front if not in the plan. This is the only hard-coded step enforcement.

### 2.3 Post-Step Validation — `dream.py:257-271`

**Function:** `_validate_step(step_num, summary, config, report)`

**Applied to:** Steps 3 (Consolidation), 4 (Merge), 5 (Relations) — only when changes were made and not in dry_run mode.

**Algorithm:**
1. Maps step number to human name via `step_names` dict (only 3, 4, 5 defined).
2. Calls `call_dream_validate(step_name, summary, config)` from `llm.py:456-468`.
3. `call_dream_validate` loads `prompts/dream_validate.md`, injects `{step_name}`, `{changes_summary}`, `{json_schema}`.
4. Returns `DreamValidation(approved: bool, issues: list[str])`.
5. If not approved: appends issues to `report.errors` and returns modified summary with `[!validated: ...]`.
6. If approved: appends `[validated]` to summary.
7. On exception: silently returns the original summary unchanged.

**Critical weakness:** Validation is purely informational. Even if `approved == False`, the changes are NOT rolled back. The step has already persisted its changes to disk. The validation result only affects the dashboard display text and the report's error list.

---

## 3. Step-by-Step Analysis

### Step 1: Load — `_step_load()` at `dream.py:277-287`

| Property | Value |
|----------|-------|
| **Type** | Deterministic |
| **Input** | `memory_path` |
| **Algorithm** | Calls `load_graph(memory_path)` from `memory/graph.py`. Then iterates all graph entities, mapping entity IDs to file paths (only includes paths that exist on disk). |
| **Output** | `(GraphData, dict[str, Path])` — graph object and entity_id-to-path mapping |
| **Persistence** | None (read-only) |
| **Failure** | Unhandled — will propagate up and abort the entire dream run since it happens before the dashboard loop |

**Note:** This step always runs regardless of the plan. The dashboard marks it complete with entity/relation counts.

---

### Step 2: Extract Docs — `_step_extract_documents()` at `dream.py:290-359`

| Property | Value |
|----------|-------|
| **Type** | LLM (extraction) + deterministic (resolution/enrichment) |
| **Input** | FAISS manifest (unextracted documents), FAISS chunk mapping (pickle) |
| **Validation** | None (not in the `_validate_step` set) |

**Algorithm:**
1. Calls `list_unextracted_docs(manifest_path)` — filters FAISS manifest for `source_type == "document"` entries without `entity_extracted == True`.
2. Returns 0 immediately if no docs found.
3. For each unextracted doc:
   a. In dry_run: just prints and counts.
   b. Loads the FAISS pickle mapping, filters chunks matching `doc["key"]`, sorts by `chunk_idx`, joins text.
   c. Calls `extract_from_chat(text, config)` — full LLM extraction with stall-aware streaming.
   d. Calls `sanitize_extraction(extraction)` — fixes invalid types from small LLMs.
   e. If entities found: runs `resolve_all()` (deterministic resolver) then `enrich_memory()` (writes to MD + graph).
   f. Marks doc as extracted via `mark_doc_extracted()`.
4. Per-doc errors are caught and appended to report; processing continues.

**Persistence:** Entity MD files and `_graph.json` are written by `enrich_memory()`. FAISS manifest is updated by `mark_doc_extracted()`.

**Issue:** The graph object passed in is the one loaded in step 1. `enrich_memory()` mutates it, but there is no explicit `save_graph()` call in this step. The graph save happens inside `enrich_memory()` via the enricher pipeline. If the process crashes after enriching some docs but before finishing, the manifest marks may be out of sync with the graph state.

**Issue:** The FAISS pickle mapping is loaded fresh for every document (`dream.py:327-328`). For N documents, this means N redundant pickle loads of the same file.

---

### Step 3: Consolidate Facts — `_step_consolidate_facts()` at `dream.py:362-397`

| Property | Value |
|----------|-------|
| **Type** | LLM (fact consolidation) |
| **Input** | Entity MD files, `config.get_max_facts(entity.type)` threshold |
| **Validation** | Yes — `_validate_step(3, ...)` if changes made |

**Algorithm:**
1. Iterates all entity paths.
2. For each entity: reads MD, filters out `[superseded]` facts, checks if `len(live_facts) > max_facts`.
3. If over threshold: calls `consolidate_entity_facts(path, config, max_facts=max_facts)` from `store.py:184-253`.
4. `consolidate_entity_facts()`:
   a. Reads entity MD.
   b. Filters live vs superseded facts.
   c. Skips if `< 3` live facts.
   d. Builds indexed text (e.g., `"0: - [fact] content"`).
   e. Calls `call_fact_consolidation(title, type, indexed_text, config, max_facts)` — `_call_structured` with `FactConsolidation` response model.
   f. Builds new fact list from consolidated results, truncating content to 150 chars, capping tags at 3.
   g. Preserves superseded facts unchanged.
   h. Writes updated entity MD.
   i. Appends history entry.

**Persistence:** Entity MD files rewritten in-place by `write_entity()`.

**Failure:** Per-entity try/except; errors appended to report, processing continues.

**LLM config:** Uses `config.llm_context` (NOT `llm_dream_effective`). This is inconsistent with steps 5 and coordinator which use `llm_dream_effective`. See `llm.py:365`: `return _call_structured(config.llm_context, prompt, FactConsolidation)`.

---

### Step 4: Merge Entities — `_step_merge_entities()` at `dream.py:400-456`

| Property | Value |
|----------|-------|
| **Type** | Deterministic (no LLM) |
| **Input** | Graph entities (slug + alias comparison) |
| **Validation** | Yes — `_validate_step(4, ...)` if merges happened |

**Algorithm — Candidate detection (lines 410-429):**
1. O(n^2) pairwise comparison of all entity slugs.
2. Two entities are merge candidates if:
   - Same `type` (exact match).
   - Overlapping alias sets (lowercased title + aliases).
3. Deduped via sorted-pair set.

**Algorithm — Merge execution via `_do_merge()` (lines 459-536):**
1. Keep entity = higher score; drop entity = lower score.
2. Read both entity MDs.
3. Merge aliases: union of both alias sets, excluding keep entity's title.
4. Merge facts: set-based dedup (exact string match on formatted fact lines).
5. Merge tags: union.
6. Keep higher importance, sum frequencies.
7. Merge mention_dates: union + sort.
8. Append history entry about the merge.
9. Write merged entity MD.
10. Retarget all relations from drop to keep (in-place mutation of `graph.relations`).
11. Delete drop entity from `graph.entities`.
12. Remove self-referencing relations (where retargeting created A->A loops).
13. Move drop MD file to `_archive/`.
14. Update keep entity's graph entry with merged metadata.

**Persistence:** Keep entity MD rewritten, drop entity archived, graph mutated in memory (no `save_graph()` call in this step — relies on step 10 to persist).

**Issues:**
- **No LLM dedup:** Merge detection is purely alias overlap. Two entities about the same concept with different names and no alias overlap will never be detected. This is the natural insertion point for LLM-based semantic deduplication.
- **No FAISS similarity check:** Despite the docstring saying "slug similarity + FAISS", the actual code only does alias overlap. FAISS is never queried in step 4.
- **Fact dedup is exact string match** (`dream.py:488-490`): Two facts with identical meaning but slightly different formatting (e.g., different date format, different valence marker) will both be kept.
- **No duplicate relation detection after retargeting:** When relations from drop are retargeted to keep, duplicates can be created (e.g., if both keep and drop had `affects -> X`). Only self-loops are removed (line 521-524); duplicate (keep, X, affects) pairs are not.
- **Graph not saved:** The graph is mutated in memory but never saved in this step. If the pipeline crashes between steps 4 and 10, all merge changes to the graph are lost (though MD file changes and archive moves are persisted, creating inconsistency).
- **monthly_buckets not merged:** `mention_dates` are merged but `monthly_buckets` are completely ignored. Historical mention data from the drop entity is silently lost.

---

### Step 5: Discover Relations — `_step_discover_relations()` at `dream.py:538-628`

| Property | Value |
|----------|-------|
| **Type** | FAISS (candidate generation) + LLM (validation) |
| **Input** | All graph entities, FAISS index |
| **Validation** | Yes — `_validate_step(5, ...)` if relations discovered |

**Algorithm — Candidate generation (lines 550-576):**
1. Build existing relation set (bidirectional) for fast lookup.
2. For each entity in the graph:
   a. FAISS search for entity title with `top_k=5`.
   b. Filter results: skip self, skip entities not in graph, skip already-related pairs.
   c. Dedup candidate pairs via sorted tuple set.

**Performance concern:** Line 575 rebuilds a set comprehension `{tuple(sorted(c)) for c in candidates}` on every inner iteration. This is O(n*k*m) where m is the growing candidates list. Should be a persistent set.

**Algorithm — LLM evaluation (lines 586-628):**
1. For each candidate pair:
   a. Build dossiers via `_build_dossier()` (title, type, tags, first 10 live facts, summary).
   b. Call `call_relation_discovery(...)` — `_call_structured` with `RelationProposal` response model, uses `config.llm_dream_effective`.
   c. If `action == "relate"` and `relation_type` is valid (checked against `_VALID_RELATION_TYPES`):
      - Create `GraphRelation` and add via `add_relation(graph, ...)` with Hebbian reinforcement.
2. After all candidates processed: `save_graph()` if any discovered.

**Persistence:** `_graph.json` saved explicitly after all discoveries.

**Issues:**
- **No conflict detection:** If the LLM proposes `A -> improves -> B` but `A -> worsens -> B` already exists, both relations will coexist. There is no check for contradictory relation types between the same entity pair.
- **Direction is always A->B:** The prompt says "specify the direction (A->B)" but `RelationProposal` has no direction field. The code always creates `from_entity=eid_a, to_entity=eid_b`. The LLM cannot propose B->A.
- **No cap on LLM calls:** If FAISS returns many candidate pairs, every single one triggers an LLM call. There is no batching or limit on the number of LLM calls in this step.

---

### Step 6: Transitive Relations — `_step_transitive_relations()` at `dream.py:667-749`

| Property | Value |
|----------|-------|
| **Type** | Deterministic (no LLM) |
| **Input** | Graph relations with `strength >= 0.4` |
| **Validation** | None |

**Transitive rules** (`dream.py:657-664`):
```
(affects, affects)   -> affects
(part_of, part_of)   -> part_of
(requires, requires) -> requires
(improves, affects)  -> improves
(worsens, affects)   -> worsens
(uses, part_of)      -> uses
```

**Algorithm:**
1. Build forward-only adjacency from relations with `strength >= min_strength` (default 0.4).
2. Build bidirectional existing relation set.
3. For each triple `(A -> rel1 -> B, B -> rel2 -> C)`:
   a. Skip if `C == A` (cycle).
   b. Skip if `(A, C)` already exists.
   c. Check `_TRANSITIVE_RULES[(rel1.type, rel2.type)]`.
   d. If rule matches: inferred strength = `min(strength_AB, strength_BC) * 0.5`.
   e. Create `GraphRelation` with context documenting the transitive chain.
   f. Call `add_relation(graph, new_rel, strength_growth=0.0)` — no Hebbian reinforcement.
4. Capped at `max_new=20` total inferred relations.
5. `save_graph()` if any discovered.

**Persistence:** `_graph.json` saved explicitly.

**Issues:**
- **Forward-only adjacency:** Only considers `from_entity -> to_entity` direction. If a relation is semantically bidirectional (e.g., `linked_to`), the reverse direction is never traversed. This means `A -> linked_to -> B, B -> linked_to -> C` would be found, but `B -> linked_to -> A` would not pair with `B -> linked_to -> C` to infer `A -> linked_to -> C`.
- **No validation step:** Unlike steps 3, 4, 5, transitive relations are not validated by the LLM. Given that transitive inference can produce spurious connections (e.g., `A affects B, B affects C` does not always mean `A affects C`), this is a gap.
- **Strength floor:** Inferred relations start at `min * 0.5`, so minimum possible strength is `0.4 * 0.5 = 0.2`. This is above the default `relation_strength_base` of 0.5 for new relations. Wait — actually it is below 0.5. So inferred relations start weaker than explicitly created ones, which is appropriate.

---

### Step 7: Prune Dead — `_step_prune_dead()` at `dream.py:752-822`

| Property | Value |
|----------|-------|
| **Type** | Deterministic (no LLM) |
| **Input** | Graph entities, relations, today's date |
| **Validation** | None |

**Prune criteria (ALL must be true):**
1. `score < 0.1` (default, configurable via param)
2. `frequency <= 1`
3. `retention != "permanent"`
4. Entity has zero relations (not in `related_entities` set)
5. `age > 90 days` (from `entity.created`) — but if `created` is missing or unparseable, the entity **passes** this check (the `continue` is inside the try block, so on exception the loop continues to `prune_candidates.append`).

**Algorithm:**
1. Build set of entities with any relations.
2. Filter entities matching all criteria above.
3. For each candidate:
   a. Move MD file to `_archive/`.
   b. Delete from `graph.entities`.
4. After all pruning: `remove_orphan_relations(graph)` + `save_graph()`.

**Persistence:** Files moved to `_archive/`, graph saved.

**Issue — Date parsing bug:** `dream.py:783-789` — If `entity.created` is set but malformed (not ISO format), the `except` catches the error and the entity is NOT skipped. The `continue` at line 788 is inside the `if age_days < min_age_days` check, not in the except. So a malformed date means the age check is skipped entirely, making the entity a prune candidate regardless of actual age. This could prune recently-created entities with bad date formats.

**Actually, re-reading more carefully:** The `try/except` block at lines 784-789: if `date.fromisoformat()` raises, the except at 789 passes (does nothing), and execution falls through to `prune_candidates.append(eid)` at line 792. So yes, entities with invalid `created` dates are always eligible for pruning. This matches the same pattern in `_collect_dream_stats()` (line 218).

---

### Step 8: Generate Summaries — `_step_generate_summaries()` at `dream.py:825-882`

| Property | Value |
|----------|-------|
| **Type** | LLM (summary generation) |
| **Input** | Entity MD files (facts, relations, tags) |
| **Validation** | None |

**Algorithm:**
1. Iterates all entity paths.
2. Skips entities that already have a summary (`entity.summary` is truthy).
3. Skips entities not in graph (may have been pruned in step 7).
4. Reads entity MD, filters live facts (excluding `[superseded]`).
5. Skips entities with no live facts.
6. Collects relations for the entity (both directions) with human-readable formatting.
7. Calls `call_entity_summary(title, type, live_facts, relations, tags, config)` from `llm.py:368-404`.
8. `call_entity_summary` does NOT use `_call_structured` — it makes a raw `litellm.completion()` call and returns free text (stripped of `<think>` tags).
9. If summary returned: updates both the frontmatter object and the graph entity object, writes MD.

**LLM config:** Uses `config.llm_context` (via `step_config = config.llm_context` at `llm.py:391`). Same inconsistency as step 3 — does not use `llm_dream_effective`.

**Persistence:** Entity MD files rewritten with updated summary. Graph entity `.summary` updated in memory (persisted by step 10).

**Issue:** Only entities without summaries are processed. There is no mechanism to refresh stale summaries (e.g., after facts have changed significantly in step 3).

---

### Step 9: Rescore — `dream.py:132-136`

| Property | Value |
|----------|-------|
| **Type** | Deterministic (ACT-R + spreading activation) |
| **Input** | Graph |
| **Validation** | None |

**Algorithm:** Calls `recalculate_all_scores(graph, config)` from `memory/scoring.py`. This runs the full ACT-R base-level activation + two-pass spreading activation + emotional modulation + retrieval threshold + LTD on relations.

**Persistence:** Graph object updated in memory. NOT saved to disk in this step — relies on step 10.

**Skipped in dry_run mode.**

---

### Step 10: Rebuild — `_step_rebuild()` at `dream.py:885-911`

| Property | Value |
|----------|-------|
| **Type** | Deterministic |
| **Input** | Graph, memory path, config |
| **Validation** | None |

**Algorithm:**
1. `save_graph(memory_path, graph)` — atomic write with lockfile and `.bak` backup.
2. `build_context(graph, memory_path, config)` — deterministic context generation.
3. Write `_context.md` if non-empty.
4. Write `_index.md`.
5. `build_index(memory_path, config)` — full FAISS rebuild (not incremental).

**Persistence:** `_graph.json`, `_context.md`, `_index.md`, FAISS index files.

**Failure:** FAISS rebuild failure is caught and shown as a warning; other failures propagate.

**Skipped entirely in dry_run mode.**

---

## 4. Dashboard — `dream_dashboard.py`

### Step Definitions

10 steps defined in `DREAM_STEPS` dict (lines 12-23): Load, Extract docs, Consolidate, Merge, Relations, Transitive, Prune, Summaries, Rescore, Rebuild.

### Status States

5 states defined in `_STATUS_STYLE` (lines 25-31):

| Status | Icon | Style |
|--------|------|-------|
| `pending` | `○` | dim |
| `running` | `◉` | cyan bold |
| `done` | `✓` | green |
| `skipped` | `⊘` | dim |
| `failed` | `✗` | red bold |

### Progress Tracking

- `DreamDashboard.__init__()`: All 10 steps initialized to `pending` status.
- `start_step(n)`: Sets status to `running`, triggers `_refresh()`.
- `complete_step(n, summary)`: Sets status to `done` with summary text.
- `skip_step(n)`: Sets status to `skipped` with reason "skipped by plan".
- `fail_step(n, error)`: Sets status to `failed` with error text (truncated to 50 chars).
- `render()`: Builds a Rich `Table` with 3 columns (status icon, step name, detail). Summary text truncated to 50 chars.
- Uses `Rich.Live` with `refresh_per_second=2` and `transient=False` (stays on screen after completion).

### Lifecycle

- Context manager: `__enter__` starts `Live`, `__exit__` stops it.
- `_refresh()` calls `self._live.update(self.render())` to redraw the table.

---

## 5. LLM Call Functions

### `call_fact_consolidation()` — `llm.py:349-365`

- **Prompt:** `prompts/consolidate_facts.md`
- **Variables:** `entity_title`, `entity_type`, `facts_text`, `max_facts`
- **LLM config:** `config.llm_context` (NOT `llm_dream_effective`)
- **Call mode:** `_call_structured` (Instructor, no stall detection)
- **Response model:** `FactConsolidation` -> list of `ConsolidatedFact(category, content, date, valence, tags, replaces_indices)`

### `call_entity_summary()` — `llm.py:368-404`

- **Prompt:** `prompts/summarize_entity.md`
- **Variables:** `entity_title`, `entity_type`, `entity_facts`, `entity_relations`, `entity_tags`
- **LLM config:** `config.llm_context` (NOT `llm_dream_effective`)
- **Call mode:** Raw `litellm.completion()` — NOT `_call_structured`. Returns free text, not structured. Applies `strip_thinking()` for reasoning models.
- **No Instructor validation, no JSON repair.**

### `call_relation_discovery()` — `llm.py:413-433`

- **Prompt:** `prompts/discover_relations.md`
- **Variables:** `entity_a_title`, `entity_a_type`, `entity_a_dossier`, `entity_b_title`, `entity_b_type`, `entity_b_dossier`
- **LLM config:** `config.llm_dream_effective`
- **Call mode:** `_call_structured` (Instructor)
- **Response model:** `RelationProposal(action, relation_type, context)`

### `call_dream_plan()` — `llm.py:436-453`

- **Prompt:** `prompts/dream_plan.md`
- **Variables:** `memory_stats`, `json_schema`, plus 6 candidate count placeholders all set to `"(see stats)"`
- **LLM config:** `config.llm_dream_effective`
- **Call mode:** `_call_structured` (Instructor)
- **Response model:** `DreamPlan(steps: list[int], reasoning: str)`

### `call_dream_validate()` — `llm.py:456-468`

- **Prompt:** `prompts/dream_validate.md`
- **Variables:** `step_name`, `changes_summary`, `json_schema`
- **LLM config:** `config.llm_dream_effective`
- **Call mode:** `_call_structured` (Instructor)
- **Response model:** `DreamValidation(approved: bool, issues: list[str])`

---

## 6. Findings and Analysis

### 6.1 Where LLM Deduplication Could Be Inserted in Step 4

Step 4 (`_step_merge_entities`) currently uses only alias overlap for merge detection. LLM-based semantic deduplication could be inserted in two places:

1. **Candidate expansion** — After the alias-overlap scan at `dream.py:414-429`, add a FAISS similarity search for each entity. Entity pairs with high FAISS similarity (e.g., > 0.85) but no alias overlap become additional merge candidates. This would catch entities like "React" and "React.js" that share no exact alias but are semantically identical.

2. **Merge confirmation** — Before calling `_do_merge()` at `dream.py:452`, add an LLM confirmation step similar to arbitration. Pass both entity dossiers to the LLM and ask "are these the same entity?" This would prevent false merges from coincidental alias overlap (e.g., "Python" the language vs "Python" the pet snake sharing the alias "python").

The most impactful insertion point is **candidate expansion with FAISS** — it addresses the biggest gap (missed duplicates) while the existing alias overlap already provides reasonable precision for the cases it catches.

### 6.2 Where Relation Conflict Detection Is Missing

**Step 5 (`_step_discover_relations`):** Lines 550-554 check if a relation already exists between two entities, but only checks for existence, not type. If `A -> improves -> B` exists and the LLM proposes `A -> worsens -> B`, both would coexist. The `existing_rels` set at line 551 is checked bidirectionally for any relation, so this specific scenario is actually blocked (the pair is skipped). However, this means if `A -> improves -> B` exists, a new `B -> affects -> A` would also be blocked, even though it is a distinct relation in a different direction.

**Step 6 (`_step_transitive_relations`):** Same bidirectional check. If `A -> improves -> C` already exists, `A -> affects -> C` (transitive) will be blocked. But if only `C -> linked_to -> A` exists, the code would allow `A -> affects -> C` to be created, resulting in potentially contradictory relations.

**Step 4 (`_do_merge`):** Lines 511-524 retarget relations from drop entity to keep entity. If both entities had relations to the same target (e.g., `drop -> affects -> X` and `keep -> improves -> X`), both relations now point from keep to X with different types. No conflict detection or deduplication of relation types occurs. Only exact self-loops are removed.

### 6.3 Steps That Could Be Made Deterministic

**Step 3 (Consolidate Facts):** Currently uses LLM to merge redundant facts. Could be made partially deterministic using:
- String similarity (Levenshtein/Jaccard) to detect near-duplicate facts.
- Rule-based merging: keep most recent date, strongest valence, union tags.
- LLM would only be needed for semantic equivalence judgment on borderline cases.
However, the quality of LLM consolidation is significantly better for understanding semantic overlap, so this is a tradeoff.

**Step 5 (Discover Relations):** The FAISS candidate generation is already deterministic. The LLM validation could theoretically be replaced by:
- Tag overlap scoring.
- Co-occurrence frequency analysis.
- Rule-based relation inference from entity types (e.g., person + organization -> works_at candidate).
But relation discovery fundamentally requires semantic understanding, so making it fully deterministic would severely reduce quality.

**Step 8 (Generate Summaries):** Summarization is inherently an LLM task. Cannot be made deterministic without significant quality loss.

**Conclusion:** Steps 3 and 5 could have deterministic pre-filters to reduce LLM calls (e.g., skip consolidation candidates where no two facts share > 60% word overlap; skip relation candidates with zero tag overlap), but the core LLM calls should remain.

### 6.4 Ordering Dependencies Between Steps

| Step | Depends on | Reason |
|------|-----------|--------|
| 1 (Load) | None | Must run first |
| 2 (Extract docs) | 1 | Needs graph for resolution |
| 3 (Consolidate) | 1 | Needs entity paths |
| 4 (Merge) | 1, optionally 3 | Needs graph; better after consolidation so merged entities have clean facts |
| 5 (Relations) | 1, optionally 4 | Needs graph; better after merge to avoid discovering relations between entities that will be merged |
| 6 (Transitive) | 5 | Operates on relations; should run after new relations are discovered |
| 7 (Prune) | 9 (Rescore) ideally | Pruning uses `entity.score` which may be stale; should ideally rescore first. Currently runs BEFORE rescore. |
| 8 (Summaries) | 3, 4 | Should run after facts are consolidated and entities merged |
| 9 (Rescore) | 2-8 | Should run after all entity/relation changes |
| 10 (Rebuild) | 9 | Should run last |

**Key ordering issue:** Step 7 (Prune) runs at position 7 but uses `entity.score` which is calculated at step 9 (Rescore). This means pruning decisions are based on potentially stale scores from the last run, not scores that reflect changes made in steps 2-6 of the current run. Entities that would be rescored above 0.1 (and thus saved from pruning) might be incorrectly archived.

**Independent steps:** Steps 2, 3, and 8 are independent of each other and could theoretically run in parallel. Steps 3 and 4 could also run in parallel since they operate on different aspects (facts vs entity identity), though running 3 before 4 produces cleaner results.

### 6.5 Additional Findings

**LLM config inconsistency:** Steps 3 and 8 use `config.llm_context` while steps 5, coordinator, and validator use `config.llm_dream_effective`. This means if a user configures a separate dream LLM (e.g., a smaller/cheaper model for dream-time reorganization), fact consolidation and summary generation will still use the main context LLM. This is likely a bug — all dream steps should use `llm_dream_effective`. Affected lines:
- `llm.py:365` (`call_fact_consolidation` uses `config.llm_context`)
- `llm.py:391` (`call_entity_summary` uses `config.llm_context`)

**Prompt step numbering mismatch:** `dream_plan.md` lists 9 steps (1-9) but the code has 10 steps (1-10). Step 6 (Transitive Relations) is entirely absent from the prompt. The prompt's "step 6" is Prune, "step 7" is Summaries, "step 8" is Rescore, "step 9" is Rebuild. The LLM planner may output step numbers according to the prompt (1-9) rather than the code (1-10), causing incorrect step selection.

**Graph persistence gaps:** Steps 4 and 9 mutate the graph in memory but do not call `save_graph()`. They rely on step 10 to persist. If the pipeline crashes between these steps and step 10, graph changes are lost while MD file changes persist, creating inconsistency.

**Validation is non-blocking:** `_validate_step()` at `dream.py:257-271` only appends to `report.errors` and modifies the dashboard summary text. Even if the LLM says the changes are wrong, they are already persisted. There is no rollback mechanism.

**`entity_paths` is stale after step 2:** `entity_paths` is built in step 1 from the graph loaded at that time. If step 2 creates new entities (via `enrich_memory()`), they will not appear in `entity_paths`. Steps 3 and 8, which iterate `entity_paths`, will miss any entities created during document extraction.

**Dry run is incomplete for step 9:** Step 9 skips rescoring in dry_run mode, but step 10 also skips everything. Steps 2, 3, 4, 5, 6, 7, 8 all have per-item dry_run checks. This is consistent but worth noting: dry_run affects nothing for steps 9 and 10.

**No progress within steps:** The dashboard only tracks step-level progress (pending/running/done). For steps that process many items (e.g., step 5 evaluating 50 relation candidates via LLM), there is no sub-step progress indicator. The dashboard shows "running" with "..." until the step completes or fails.
