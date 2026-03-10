# Audit 02 — Entity Resolution & Arbitration

**Date**: 2026-03-10
**Scope**: `src/pipeline/resolver.py`, `src/pipeline/arbitrator.py`, `src/core/llm.py` (call_arbitration, _call_structured), `src/core/models.py` (Resolution, EntityResolution), `src/pipeline/orchestrator.py` (make_faiss_fn, arbitration loop), `prompts/arbitrate_entity.md`

---

## 1. Function-by-Function Analysis

### 1.1 `resolve_entity()` — `src/pipeline/resolver.py:17-65`

**Signature**:
```python
def resolve_entity(
    name: str,
    graph: GraphData,
    faiss_search_fn: Optional[Callable] = None,
    observation_context: str = "",
) -> Resolution
```

**Purpose**: Resolve a free-form entity name against the existing knowledge graph. Zero LLM tokens — purely deterministic.

**Input**: Entity name (string), loaded graph, optional FAISS search callable, optional observation context string for disambiguation.

**Output**: `Resolution` with `status` in `{"resolved", "new", "ambiguous"}`.

#### Resolution Order (4 stages):

**Stage 1 — Exact slug match** (line 37-38):
```python
slug = slugify(name)
if slug in graph.entities:
    return Resolution(status="resolved", entity_id=slug)
```
- Calls `slugify()` from `src/core/utils.py:11-17` which normalizes to ASCII lowercase with hyphens.
- Instant O(1) dict lookup. No ambiguity possible.

**Stage 2 — Alias containment check** (lines 41-49):
```python
name_lower = name.lower()
for entity_id, meta in graph.entities.items():
    for alias in meta.aliases:
        alias_lower = alias.lower()
        if alias_lower in name_lower or name_lower in alias_lower:
            return Resolution(status="resolved", entity_id=entity_id)
    if meta.title.lower() in name_lower or name_lower in meta.title.lower():
        return Resolution(status="resolved", entity_id=entity_id)
```
- **Bidirectional substring containment**: checks if alias is IN name OR name is IN alias.
- Also checks entity title (not just aliases).
- Returns the **first match found** — iteration order is dict insertion order, which is non-deterministic across graph rebuilds.

**Stage 3 — FAISS similarity search** (lines 53-62):
```python
query = f"{name} {observation_context}".strip() if observation_context else name
similar = faiss_search_fn(query, top_k=3, threshold=0.75)
if similar:
    candidates = [s["entity_id"] for s in similar if "entity_id" in s]
    if candidates:
        return Resolution(status="ambiguous", candidates=candidates)
```
- Context-enriched: prepends first observation's category + content prefix (see `resolve_all`, line 81).
- Threshold **0.75** passed to the search function.
- Returns `status="ambiguous"` with candidate list — never auto-resolves from FAISS alone.
- Exception silently caught (line 61-62): if FAISS fails, falls through to "new".

**Stage 4 — New entity** (line 65):
```python
return Resolution(status="new", suggested_slug=slug)
```

---

### 1.2 `resolve_all()` — `src/pipeline/resolver.py:68-89`

**Signature**:
```python
def resolve_all(
    raw_extraction: RawExtraction,
    graph: GraphData,
    faiss_search_fn: Optional[Callable] = None,
) -> ResolvedExtraction
```

**Purpose**: Iterate over all extracted entities and resolve each one.

**FAISS context enrichment** (lines 78-82):
```python
if entity.observations:
    obs = entity.observations[0]
    obs_context = f"{obs.category} {obs.content[:50]}"
```
- Uses **only the first observation** — category name + first 50 chars of content.
- This is the disambiguation signal: e.g., "Apple preference I love their MacBooks" vs "Apple fact A common fruit".

**Key detail**: Relations are passed through unmodified (`raw_extraction.relations`). Entity names in relations are NOT resolved here — that happens later in the enricher via `_find_entity_slug()`.

---

### 1.3 `make_faiss_fn()` — `src/pipeline/orchestrator.py:34-50`

**Signature**:
```python
def make_faiss_fn(config, memory_path):
    def fn(query: str, top_k: int = 3, threshold: float = 0.85):
        ...
    return fn
```

**Critical finding**: The wrapper has a **default threshold of 0.85**, but the resolver calls it with **threshold=0.75** (resolver.py:56). The resolver's explicit argument wins. However, if `make_faiss_fn` were called without the threshold argument from elsewhere, 0.85 would apply. The mismatch in defaults is confusing but currently harmless.

**The wrapper**:
- Calls `indexer.search()` which does raw FAISS inner-product search with no threshold.
- Applies threshold **post-hoc** by filtering: `if r.score >= threshold`.
- Returns `list[dict]` with keys `entity_id` and `score`.

---

### 1.4 `arbitrate_entity()` — `src/pipeline/arbitrator.py:10-32`

**Signature**:
```python
def arbitrate_entity(
    name: str,
    context: str,
    candidates: list[str],
    graph: GraphData,
    config: Config,
) -> EntityResolution
```

**Purpose**: When FAISS returns multiple candidates (ambiguous), ask the LLM to pick one or declare a new entity.

**Candidate formatting** (lines 19-27):
```python
candidates_data = []
for cid in candidates:
    if cid in graph.entities:
        entity = graph.entities[cid]
        candidates_data.append({
            "id": cid,
            "title": entity.title,
            "type": entity.type,
            "aliases": entity.aliases,
        })
```
- Includes: slug ID, title, type, aliases.
- **Does NOT include**: observations/facts, relations, score, summary, tags.
- If a candidate ID is not in the graph (e.g., stale FAISS index), it is silently dropped.
- If ALL candidates are invalid, returns `EntityResolution(action="new")` without calling the LLM (line 29-30).

---

### 1.5 `call_arbitration()` — `src/core/llm.py:263-283`

**Signature**:
```python
def call_arbitration(
    entity_name: str,
    entity_context: str,
    candidates: list[dict],
    config: Config,
) -> EntityResolution
```

**Candidate string formatting** (lines 271-274):
```python
candidates_str = "\n".join(
    f"- {c['id']}: {c['title']} (type: {c['type']}, aliases: {c.get('aliases', [])})"
    for c in candidates
)
```
- Example output: `- louise: Louise (type: person, aliases: [Lou])`
- **No observation/fact context about candidates** is provided to the LLM. The LLM sees only name, type, and aliases.

**LLM call**: Uses `_call_structured()` (line 283) — no stall detection, standard Instructor call. Uses `config.llm_arbitration` step config.

---

### 1.6 `_call_structured()` — `src/core/llm.py:118-139`

**Signature**:
```python
def _call_structured(
    step_config: LLMStepConfig,
    prompt: str,
    response_model: type[T],
) -> T
```

- Uses Instructor `MD_JSON` mode (extracts JSON from markdown code blocks).
- Wrapped in `_repaired_json()` context manager for malformed JSON recovery.
- Respects `max_retries` from step config (Instructor retry with validation feedback).
- No streaming, no stall detection — appropriate for short arbitration calls.

---

### 1.7 Prompt: `prompts/arbitrate_entity.md`

```
New mention detected: "{entity_name}"
Context of appearance: "{entity_context}"
Existing entities that might match:
{candidates}
Does this mention correspond to an existing entity?
If yes, indicate which one. If no, indicate the type of the new entity.
```

**Variables injected**: `entity_name`, `entity_context` (extraction summary), `candidates` (formatted string), `json_schema` (EntityResolution schema), `categories_entity_types` (auto-injected).

---

### 1.8 Data Models — `src/core/models.py`

**Resolution** (line 63-67):
```python
class Resolution(BaseModel):
    status: Literal["resolved", "new", "ambiguous"]
    entity_id: Optional[str] = None
    candidates: list[str] = Field(default_factory=list)
    suggested_slug: Optional[str] = None
```

**EntityResolution** (lines 83-86):
```python
class EntityResolution(BaseModel):
    action: Literal["existing", "new"]
    existing_id: Optional[str] = None
    new_type: Optional[EntityType] = None
```

---

### 1.9 Orchestrator Arbitration Loop — `src/pipeline/orchestrator.py:310-338`

```python
for item in resolved.resolved:
    if item.resolution.status == "ambiguous":
        try:
            arb_result = arbitrate_entity(
                item.raw.name,
                extraction.summary,      # context = full chat summary
                item.resolution.candidates,
                graph,
                config,
            )
            if arb_result.action == "existing" and arb_result.existing_id:
                item.resolution = Resolution(status="resolved", entity_id=arb_result.existing_id)
            else:
                item.resolution = Resolution(status="new", suggested_slug=slugify(item.raw.name))
        except Exception as e:
            console.print(f"  [yellow]Arbitration failed for {item.raw.name}: {e}[/yellow]")
            item.resolution = Resolution(status="new", suggested_slug=slugify(item.raw.name))
```

**Key behaviors**:
- Context passed to LLM is `extraction.summary` (the whole chat's summary), not per-entity context.
- On ANY exception (LLM failure, timeout, validation error): **defaults to creating a new entity**. This is a safe fallback but can cause duplicates.
- If `arb_result.action == "existing"` but `existing_id` is None/empty: treated as new entity. No validation that `existing_id` is actually in the graph.

---

## 2. Critical Findings

### 2.1 Alias Containment is Overly Aggressive (resolver.py:42-49)

The bidirectional substring check creates false positives:

| Name | Alias | Match? | Correct? |
|------|-------|--------|----------|
| "Ana" | "Banana" | YES (Ana in Banana) | NO |
| "Art" | "Martial Art" | YES (Art in Martial Art) | MAYBE |
| "AI" | "Claire" | YES (AI in clAIre) | NO — but .lower() prevents this |
| "Jo" | "John" | YES (jo in john) | MAYBE |
| "Dr Martin" | "Martin" | YES (martin in dr martin) | YES |

**Risk**: Short names (2-3 chars) are especially prone to false alias matches. The check is case-insensitive but has no minimum-length guard.

**Recommendation**: Add minimum match length (e.g., 4 chars) or switch to word-boundary matching. Consider Jaccard similarity on tokens instead of raw substring.

### 2.2 First-Match Wins in Alias Scan (resolver.py:42)

The alias scan iterates `graph.entities.items()` and returns on the **first match**. If multiple entities have overlapping aliases, the one that appears first in dict iteration order wins. This is:
- Non-deterministic across graph rebuilds (dict ordering depends on insertion order in `rebuild_from_md`).
- Not scored — a weak substring match beats a perfect match that appears later.

**Recommendation**: Collect all alias matches, score them (e.g., by match length ratio), return the best or mark as ambiguous if tied.

### 2.3 FAISS Threshold Discrepancy (resolver.py:56 vs orchestrator.py:42)

- Resolver calls with `threshold=0.75`
- `make_faiss_fn` default is `threshold=0.85`
- The explicit argument wins, but the inconsistency is confusing.

**The 0.75 threshold**: For cosine similarity on normalized sentence-transformer embeddings, 0.75 is reasonable for semantic similarity but may be too permissive for entity identity. Two different entities about similar topics (e.g., "Swimming" and "Pool") could exceed 0.75.

**Recommendation**: Consider raising to 0.80 or making configurable via `config.yaml`.

### 2.4 FAISS Always Returns "ambiguous", Never "resolved" (resolver.py:60)

Even if FAISS returns a single candidate with score 0.99, the status is still `"ambiguous"`, triggering an LLM arbitration call. This wastes tokens for high-confidence matches.

**Recommendation**: If FAISS returns exactly 1 candidate with score >= 0.90 (high threshold), auto-resolve without LLM.

### 2.5 Candidate Information Starvation (arbitrator.py:19-27)

The LLM arbitrator receives only: `id`, `title`, `type`, `aliases`. It does NOT receive:
- Entity observations/facts
- Relations
- Summary
- Tags
- Score or frequency

This means the LLM is making disambiguation decisions with minimal context. For entities with similar names but different facts (e.g., two "Martin" entities — one a doctor, one a colleague), the LLM has almost no signal beyond the type field.

**Recommendation**: Include the entity `summary` field and top 3 facts in the candidate data.

### 2.6 No Correction Detection (systemic gap)

**Can resolution detect CORRECTIONS?** (e.g., "Louise is not my wife, she's my daughter")

**No.** The pipeline has no mechanism to:
1. Detect negation or contradiction in new observations
2. Reclassify an entity's type based on new information
3. Remove or invalidate existing relations

The `supersedes` field on `RawObservation` (models.py:39) handles fact-level replacement:
- `mark_observation_superseded()` in `store.py:487-508` marks old facts with `[superseded]`
- But this only works within a single entity's facts, matching by category + content substring
- It does NOT handle relation corrections (e.g., changing `parent_of` to something else)
- It does NOT handle entity type changes (e.g., changing `type: person` from one role to another)
- The supersedes field must be explicitly populated by the LLM during extraction — there is no automatic contradiction detection

**Example failure**: User says "Louise is not my wife, she's my daughter."
1. Extractor may produce a new observation with `supersedes: "wife"` — depends on LLM quality
2. Even if supersession works for the fact, the `lives_with` or `parent_of` relation remains unchanged
3. No existing relation is deleted or modified
4. The entity type stays `person` (correct), but wrong relations persist

### 2.7 Contradictory Relations Are Never Detected

The system has no mechanism to detect or resolve contradictory relations:
- `add_relation()` in `graph.py` deduplicates by `(from, to, type)` tuple and reinforces existing relations
- But it does NOT check for semantic conflicts (e.g., `A improves B` vs `A worsens B`)
- Both contradictory relations can coexist in the graph indefinitely
- The `contrasts_with` relation type exists but is never auto-generated from contradiction detection

**Recommendation**: Add a contradiction check in `add_relation()` that detects opposing relation types (improves/worsens, requires/contrasts_with) between the same entity pair.

### 2.8 No Validation of Arbitration Result (orchestrator.py:321)

When the LLM returns `action="existing"` with an `existing_id`, the orchestrator does NOT verify that:
- `existing_id` exists in `graph.entities`
- `existing_id` was one of the original candidates

The LLM could hallucinate an entity ID, and it would be accepted. The enricher would then fail silently at line 120-121:
```python
if entity_id not in graph.entities:
    return
```

**Recommendation**: Validate `arb_result.existing_id in item.resolution.candidates` before accepting.

### 2.9 Race Condition: Graph Loaded Twice (orchestrator.py:307 + enricher.py:41)

```python
# orchestrator.py:307
graph = load_graph(memory_path)
resolved = resolve_all(extraction, graph, ...)
# ... arbitration using same graph ...

# enricher.py:41 (called later)
graph = load_graph(memory_path)  # RELOADS from disk
```

If another process modifies `_graph.json` between resolution and enrichment (e.g., concurrent `memory run`), the enricher operates on a different graph state than what resolution saw. This could cause:
- Resolved entity IDs no longer exist
- New entity slugs collide with entities created by the other process

The lockfile in `save_graph()` protects writes but not the read-resolve-write span.

**Recommendation**: Pass the graph object through the pipeline instead of reloading, or use a read-lock.

### 2.10 Entity Processing Order Matters (resolver.py:76)

Entities are resolved in extraction order. If the same entity appears twice in one extraction with different names (e.g., "Louise" and "Lou"), the first one creates a slug, but the second one won't find it because the graph hasn't been updated yet (resolution is read-only).

Result: both get `status="new"` and the enricher creates two separate entities.

**Recommendation**: Build a local resolution cache within `resolve_all()` that tracks newly-resolved slugs.

### 2.11 Silent FAISS Failure (resolver.py:61-62)

```python
except Exception:
    pass  # FAISS not available, skip
```

Any FAISS error — corrupted index, embedding model mismatch, OOM — is silently swallowed. The entity falls through to "new", potentially creating duplicates.

**Recommendation**: Log the exception at WARNING level. Differentiate between "FAISS not built yet" (expected) and actual errors.

### 2.12 `_call_structured` Has No Stall Detection

`call_arbitration()` uses `_call_structured()` which has no stall detection (unlike extraction which uses `_call_with_stall_detection`). If the arbitration LLM hangs, the only protection is the `timeout` in `step_config`. If `timeout` is not set (it's optional), the call can hang indefinitely.

**Recommendation**: Ensure `llm_arbitration.timeout` always has a default value (e.g., 30s).

---

## 3. Data Flow Summary

```
Chat text
  |
  v
extract_from_chat() --> RawExtraction {entities, relations, summary}
  |
  v
sanitize_extraction() --> fixes types, clamps values
  |
  v
resolve_all(raw, graph, faiss_fn)
  |
  +-- per entity: resolve_entity(name, graph, faiss_fn, obs_context)
  |     |
  |     +-- slug match? --> resolved
  |     +-- alias match? --> resolved (first match wins)
  |     +-- FAISS match? --> ambiguous (with candidates)
  |     +-- none? --> new
  |
  v
ResolvedExtraction {resolved: [ResolvedEntity], relations, summary}
  |
  v
orchestrator loop: for each ambiguous entity:
  |
  +-- arbitrate_entity(name, summary, candidates, graph, config)
  |     |
  |     +-- format candidates as {id, title, type, aliases}
  |     +-- call_arbitration() --> LLM via _call_structured()
  |     +-- prompt: arbitrate_entity.md
  |     |
  |     +-- returns EntityResolution {action, existing_id?, new_type?}
  |
  +-- on success: update Resolution to resolved or new
  +-- on failure: default to new
  |
  v
enrich_memory(resolved, config, today)
  |
  +-- resolved entities: update MD + graph
  +-- new entities: create MD + add to graph
  +-- relations: resolve names, create stubs, add to graph
```

---

## 4. Threshold Analysis

| Threshold | Location | Value | Purpose |
|-----------|----------|-------|---------|
| FAISS similarity | resolver.py:56 | 0.75 | Minimum cosine similarity to consider a FAISS match |
| FAISS default | orchestrator.py:42 | 0.85 | Default in wrapper (overridden by resolver) |
| Extraction retry | orchestrator.py:263 | 2 | Max retries before doc_ingest fallback |

The 0.75 FAISS threshold is on the permissive side for entity identity matching. Sentence-transformer cosine similarity of 0.75 means "semantically related" but not necessarily "the same entity." For example:
- "Swimming" vs "Pool swimming" might score ~0.80 (correct match)
- "Swimming" vs "Diving" might score ~0.76 (incorrect match)
- "Dr. Martin" vs "Martin Luther King" might score ~0.78 (incorrect match)

The context enrichment (category + content prefix) helps but only uses 50 chars of the first observation.

---

## 5. Risk Matrix

| Issue | Severity | Likelihood | Impact |
|-------|----------|------------|--------|
| 2.1 Alias false positives (short names) | Medium | Medium | Wrong entity updated |
| 2.2 First-match non-determinism | Low | Low | Inconsistent resolution |
| 2.4 FAISS single-candidate still triggers LLM | Low | High | Wasted tokens |
| 2.5 Candidate info starvation | Medium | Medium | Wrong arbitration decisions |
| 2.6 No correction detection | High | Medium | Stale/wrong facts persist |
| 2.7 No contradictory relation detection | High | Medium | Contradictions accumulate |
| 2.8 No validation of arbitrated ID | Medium | Low | Silent enrichment failure |
| 2.9 Graph race condition | Medium | Low | Data loss on concurrent runs |
| 2.10 Intra-batch duplicate creation | Medium | Medium | Duplicate entities |
| 2.11 Silent FAISS failure | Low | Low | Unexpected new entities |
| 2.12 No arbitration timeout default | Low | Low | Hung pipeline |

---

## 6. Recommendations (Prioritized)

1. **Add intra-batch resolution cache** (fixes 2.10) — Track slugs resolved within current `resolve_all()` call to prevent duplicate creation.
2. **Enrich arbitration candidates** (fixes 2.5) — Include `summary` and top 3 facts in candidate data sent to LLM.
3. **Validate arbitrated ID** (fixes 2.8) — Check `existing_id in candidates` before accepting.
4. **Add contradiction detection** (fixes 2.7) — Check for opposing relation types on same entity pair in `add_relation()`.
5. **Guard alias matching** (fixes 2.1) — Require minimum 4-char match length, or switch to token-based matching.
6. **Auto-resolve high-confidence FAISS** (fixes 2.4) — If single candidate with score >= 0.90, skip arbitration.
7. **Log FAISS errors** (fixes 2.11) — Replace bare `except: pass` with `logger.warning()`.
8. **Pass graph through pipeline** (fixes 2.9) — Avoid reloading graph between resolution and enrichment.
9. **Make FAISS threshold configurable** (fixes 2.3) — Add `scoring.faiss_resolution_threshold` to config.
10. **Design correction protocol** (fixes 2.6) — Requires extraction-level changes to detect negation and relation invalidation.
