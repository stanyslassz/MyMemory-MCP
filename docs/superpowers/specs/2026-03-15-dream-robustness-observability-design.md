# Dream Pipeline — Robustness & Observability

**Date**: 2026-03-15
**Status**: Approved
**Scope**: Intra-step checkpointing, post-dream reporting, HTML dashboard integration

## Problem Statement

The dream pipeline has 3 issues:

1. **Lost progress on interruption**: Checkpoint saves only between steps, not within. Killing mid-step (e.g., step 5 relations with 200 entities) loses all step progress. Unacceptable for 1h+ dreams.
2. **No post-dream visibility**: The Rich terminal dashboard disappears after the run. No way to review what happened.
3. **Empty dream tab in HTML dashboard**: `dashboard.py` already has a Dream tab with session/step rendering, but `dream.py` emits zero events to the event log.

## Design Decisions

- **Option B chosen for intra-step recovery**: No explicit progress counter. Instead, each iteration persists its result immediately (graph + MD), and steps detect already-done work via filesystem state on restart. The filesystem IS the checkpoint.
- **Approach 1 chosen**: Persist-per-iteration + event bridge. Single source of truth (filesystem for progress, event log for history).

## Part 1: Persist-Per-Iteration (Intra-Step Checkpointing)

### Principle

Each long step persists its work after every iteration that modifies graph or MD files. On `--resume`, the step restarts from the beginning but skips already-done work via natural idempotency checks.

The inter-step checkpoint (`_dream_checkpoint.json`) remains unchanged — it tracks which step to resume from. Intra-step progress is tracked by the filesystem state itself.

### Per-Step Changes

#### Step 3 — Consolidate (`_step_consolidate_facts` in `dream.py`)

- **Already quasi-idempotent**: `write_entity()` is called inside the loop per entity
- **Detection**: consolidated facts have `[superseded]` markers. On restart, `len(live_facts) <= max_facts` naturally skips them
- **No `save_graph()` needed**: consolidation only touches MD files, not the graph
- **Change**: verify `write_entity()` is inside the loop (not batched after). Return `(summary, details)` tuple

#### Step 4 — Merge (`_step_merge_entities` in `dream.py`)

> **Note**: `_step_merge_entities` exists in both `dream.py` and `dream/merger.py`. `run_dream()` calls the **`dream.py` copy** (line 178). All changes target `dream.py`. The `dream/merger.py` copy is dead code (to be cleaned up separately).

- **Change**: call `save_graph()` after each `_do_merge(slug_a, slug_b, ...)`
- **Detection**: if `slug_b` not in `graph.entities` → pair already merged, skip
- **Impact**: 0-20 merges max (FAISS candidate cap), so 0-20 extra save_graph calls. Negligible
- **Change**: return `(summary, details)` tuple with `{candidates, merged, rejected}`

**Interaction with snapshot-rollback**: Steps 4 and 5 have a `snapshot = graph.model_copy(deep=True)` + `validate_dream_step()` mechanism that rolls back the in-memory graph on validation failure and calls `save_graph()` with the snapshot. Per-iteration `save_graph()` means intermediate states are written to disk during the step, but the rollback `save_graph()` at lines 186/204 **overwrites** them with the snapshot, so the final on-disk state is still correct. The only risk is a kill during the rollback write itself — a window of ~10ms. This is acceptable; the existing `.bak` mechanism in `save_graph()` covers this case.

#### Step 5 — Relations (`_step_discover_relations` in `dream.py`)

> **Note**: `_step_discover_relations` exists in both `dream.py` and `dream/discovery.py`. `run_dream()` calls the **`dream.py` copy** (line 196). All changes target `dream.py`. The `dream/discovery.py` copy is dead code (to be cleaned up separately).

- **Change**: call `save_graph()` after each successful `add_relation()`
- **Idempotency on restart**: the candidate collection phase (FAISS search) rebuilds `existing_rels` as a set of `(from, to)` pairs from the reloaded graph. Pairs that already have a relation are excluded during collection, so they never reach the LLM call loop. Per-iteration `save_graph()` ensures the reloaded graph has the latest relations
- **Within-run guard**: `add_relation()` already deduplicates via Hebbian reinforcement, so no extra check needed in the loop body
- **Trade-off**: candidates rejected by LLM (action != "relate") are re-evaluated on restart. This costs extra LLM calls but avoids maintaining a "rejected pairs" state
- **Note**: the FAISS candidate collection phase reruns on restart (~seconds, no LLM). Acceptable
- **Impact**: ~5-30 save_graph calls in worst case
- **Rollback**: same snapshot-rollback interaction as Step 4 (see above)
- **Change**: return `(summary, details)` tuple with `{candidates, created, rejected}`

#### Step 8 — Summaries (`_step_generate_summaries` in `dream.py`)

- **Change**: add `save_graph()` after each summary write. The in-memory `entity.summary = summary` and `write_entity()` calls already exist per-iteration; only the `save_graph()` call is new
- **Detection**: `if entity.summary: continue` — already in the code
- **Impact**: potentially many save_graph calls on first dream, but each is ~10ms. No snapshot-rollback in this step, so per-iteration saves are safe
- **Change**: return `(summary, details)` tuple with `{generated, skipped, fallback}`

#### Steps 6, 7 (deterministic, fast)

- **No change needed**: these complete in seconds. A kill mid-step = full re-run of the step, acceptable
- **Change**: return `(summary, details)` tuple for event consistency

#### Steps 1, 2, 9, 10

- **No change for persistence**: step 1 is read-only, step 2 uses existing pipeline (already persists), steps 9-10 are single operations
- **Change**: return `(summary, details)` tuple for event consistency

### save_graph() Frequency

| Step | save_graph calls per run | Acceptable? |
|------|--------------------------|-------------|
| 3. Consolidate | 0 (MD only) | Yes |
| 4. Merge | 0-20 | Yes |
| 5. Relations | 5-30 | Yes |
| 8. Summaries | 0-N (N = entities without summary) | Yes (~10ms each) |

### Return Value Convention

All `_step_*` functions change from returning `int` or `str` to returning `tuple[str, dict]`:
- `str`: human-readable summary (e.g., "12 entities consolidated")
- `dict`: structured details (e.g., `{"processed": 12, "skipped": 45, "errors": 0}`)

## Part 2: Event Logging + Post-Dream Report

### Events Emitted

All events use `append_event(memory_path, event_type, "dream", data)`.

#### Event Types

| Event | When | Data Fields |
|-------|------|-------------|
| `dream_session_started` | Before step loop | `dream_id`, `steps_planned`, `resumed` (bool), `entity_count`, `relation_count` |
| `dream_step_started` | After `dashboard.start_step(s)` | `dream_id`, `step`, `step_name` |
| `dream_step_completed` | After `dashboard.complete_step(s)` | `dream_id`, `step`, `step_name`, `duration_s`, `summary`, `details` |
| `dream_step_failed` | In `except` block | `dream_id`, `step`, `step_name`, `duration_s`, `error` |
| `dream_step_skipped` | After `dashboard.skip_step(s)` | `dream_id`, `step`, `step_name` |
| `dream_session_completed` | After step loop | `dream_id`, `duration_s`, `steps_completed`, `steps_failed` |

#### Placement in `run_dream()`

```
dream_session_started          # before `with DreamDashboard`
  for s in range(1, 11):
    if s not in steps_to_run:
      dream_step_skipped
      continue
    t0 = time.monotonic()
    dream_step_started
    try:
      ... execute step ...
      dream_step_completed     # elapsed = time.monotonic() - t0
    except:
      dream_step_failed
dream_session_completed        # after loop, before _clear_checkpoint()
```

#### `details` Dict Per Step

| Step | Details Keys |
|------|-------------|
| 1. Load | `entities`, `relations` |
| 2. Extract | `docs_found`, `docs_extracted` |
| 3. Consolidate | `processed`, `skipped`, `errors` |
| 4. Merge | `candidates`, `merged`, `rejected` |
| 5. Relations | `candidates`, `created`, `rejected` |
| 6. Transitive | `triples_checked`, `created` |
| 7. Prune | `candidates`, `archived` |
| 8. Summaries | `generated`, `skipped`, `fallback` |
| 9. Rescore | `entities_scored` |
| 10. Rebuild | `context_tokens` |

### Dream Report (`_dream_report.md`)

Generated at the end of `run_dream()` by reading events from the current session (`read_events(source="dream", after=session_start_ts)`).

#### Format

```markdown
# Dream Report — 2026-03-15

**Session**: 2026-03-15T14:23:45 | **Duration**: 12m 34s | **Status**: completed

## Steps

| # | Step | Status | Duration | Summary |
|---|------|--------|----------|---------|
| 1 | Load | done | 0.2s | 156 entities, 89 relations |
| 2 | Extract docs | skipped | — | skipped by plan |
| 3 | Consolidate | done | 3m 12s | 12 entities consolidated |
| 5 | Relations | done | 6m 45s | 9 new relations from 28 candidates |
| 9 | Rescore | done | 1.1s | 156 entities scored |
| 10 | Rebuild | done | 0.8s | context rebuilt |

## Details

### 3. Consolidate
- Processed: 12
- Skipped: 45
- Errors: 0

### 5. Relations
- Candidates: 28
- Created: 9
- Rejected: 19
```

#### Generation

A function `_generate_dream_report(memory_path, dream_id, session_start_ts) -> Path` in `dream.py`:
1. Read events with `read_events(memory_path, source="dream", after=session_start_ts, limit=10_000)` — high limit since the `after=` filter already scopes to the current session (~62 events per run). Note: `read_events` returns `events[-limit:]` (oldest events dropped first), so the limit must exceed the total session event count
2. Filter by `dream_id`
3. Format as markdown table + detail sections
4. Write to `memory/_dream_report.md`

### CLI Flag

Add `--report` flag to the `dream` command in `cli.py`:
- `memory dream --report` reads and prints `memory/_dream_report.md` to stdout
- If no report exists, prints "No dream report found. Run `memory dream` first."
- Does not trigger a dream run

## Part 3: HTML Dashboard Dream View

### What Works Automatically

Once events are emitted (Part 2), `dashboard.py` works out of the box:
- `_extract_dream_sessions(events)` groups `dream_step_*` events by date
- Steps render with status badges (checkmark/x/skip/spinner)
- Click on a step shows `summary`, `duration_s`, `error` in the detail panel

### Enrichments (Minor)

#### 1. Display `details` in step detail panel (~10 lines JS)

In the JS function that renders dream step details, iterate over `event.data.details` and display each key/value pair:

```
Consolidate — 3m 12s ✅
  processed: 12, skipped: 45, errors: 0
```

#### 2. Session header with duration and status (~15 lines Python, ~5 lines JS)

In `_extract_dream_sessions()`:
1. Broaden the filter from `evt_type.startswith("dream_step_")` to also include `dream_session_*` events
2. **Change grouping key from calendar day to `dream_id`** (read from `evt["data"]["dream_id"]`). The current day-based grouping collides when multiple dreams run on the same day. Use `ts[:10]` only as the display label
3. Extract total duration and step counts from `dream_session_completed`:

```
Session 2026-03-15 — 12m 34s — 8/10 steps completed
```

#### 3. No new tabs, pages, or server routes

Everything stays in the existing Dream tab. No new HTML files, no new API endpoints.

### Dashboard Changes Summary

| Change | Location | Effort |
|--------|----------|--------|
| Display `details` dict in step detail | `dashboard.py` (JS template) | ~10 lines |
| Broaden filter + group by `dream_id` + session header | `dashboard.py` (Python + JS) | ~25 lines |
| **Total** | | ~35 lines |

## File Change Summary

| File | Changes |
|------|---------|
| `src/pipeline/dream.py` | Emit 6 event types, add `t0`/elapsed timing, call `save_graph()` in steps 4/5/8 iteration loops, generate `_dream_report.md`, all `_step_*` return `(summary, details)`. Steps 4 and 5 changes are in this file (the `dream/merger.py` and `dream/discovery.py` copies are dead code — not called by `run_dream()`) |
| `src/pipeline/dashboard.py` | Broaden `_extract_dream_sessions()` filter to include `dream_session_*` events, change grouping from day to `dream_id`, display `details` in step panel, session header (~35 lines) |
| `src/cli.py` | Add `--report` flag to `dream` command |

**No new files. No new dependencies.**

## What Is NOT in Scope

- No changes to `_dream_checkpoint.json` format
- No changes to the Rich terminal dashboard (`dream_dashboard.py`) beyond what it already does
- No intra-step progress bar (the terminal dashboard shows step-level status, not entity-level)
- No new MCP tools
- No changes to scoring, context generation, or extraction
- Event log rotation out of scope (personal local-first tool, `_event_log.jsonl` growth is manageable)
- Dead code cleanup of `dream/merger.py` and `dream/discovery.py` duplicates out of scope (separate task)
