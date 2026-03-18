# Dream Pipeline Robustness & Observability — Implementation Plan

> **COMPLETED** (2026-03-18): All 10 tasks implemented across 7 commits. Post-implementation audit (CLAUDE.md update, dead code cleanup, config bypass fixes) completed separately.

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the dream pipeline interruptible at any point with zero progress loss, and provide post-dream visibility via markdown reports and an HTML dashboard.

**Architecture:** Three changes to `dream.py`: (1) persist graph/MD after each LLM iteration in long steps so the filesystem IS the intra-step checkpoint, (2) emit structured events to `_event_log.jsonl` via the existing `append_event()` API, (3) generate a `_dream_report.md` at end of run. Minor enrichments to `dashboard.py` for the existing Dream tab.

**Tech Stack:** Python 3.11+, existing `event_log.py` API, existing `dashboard.py` HTML template, Click CLI.

**Spec:** `docs/superpowers/specs/2026-03-15-dream-robustness-observability-design.md`

---

## Chunk 1: Event Logging Infrastructure

### Task 1: Add event emission to `run_dream()` main loop

**Files:**
- Modify: `src/pipeline/dream.py:95-250` (run_dream function)
- Test: `tests/test_dream_events.py` (new)

- [ ] **Step 1: Write failing test for dream session events**

Create `tests/test_dream_events.py`:

```python
"""Tests for dream event logging."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from rich.console import Console
from io import StringIO

from src.pipeline.dream import run_dream
from src.core.config import Config
from src.memory.event_log import read_events


def _make_config(tmp_path: Path) -> Config:
    """Create a minimal config for testing."""
    memory_path = tmp_path / "memory"
    memory_path.mkdir()
    (memory_path / "chats").mkdir()
    (memory_path / "_inbox").mkdir()
    # Minimal config — we patch LLM calls, so no real API needed
    from unittest.mock import MagicMock
    config = MagicMock(spec=Config)
    config.memory_path = memory_path
    config.dream = MagicMock()
    config.dream.prune_score_threshold = 0.1
    config.dream.prune_max_frequency = 1
    config.dream.prune_min_age_days = 90
    config.scoring = MagicMock()
    config.scoring.relation_strength_growth = 0.05
    return config


def test_dream_emits_session_events(tmp_path):
    """run_dream should emit dream_session_started and dream_session_completed events."""
    config = _make_config(tmp_path)
    console = Console(file=StringIO())

    from src.core.models import GraphData
    empty_graph = GraphData(generated=datetime.now().isoformat(), entities={}, relations=[])

    with patch("src.pipeline.dream._step_load", return_value=(empty_graph, {})):
        run_dream(config, console, dry_run=True, step=1)

    events = read_events(config.memory_path, source="dream", limit=10_000)
    types = [e["type"] for e in events]
    assert "dream_session_started" in types
    assert "dream_session_completed" in types

    # Check session_started data
    started = next(e for e in events if e["type"] == "dream_session_started")
    assert "dream_id" in started["data"]
    assert "steps_planned" in started["data"]
    assert started["data"]["entity_count"] == 0


def test_dream_emits_step_events(tmp_path):
    """Each executed step should emit started + completed/skipped events."""
    config = _make_config(tmp_path)
    console = Console(file=StringIO())

    from src.core.models import GraphData
    empty_graph = GraphData(generated=datetime.now().isoformat(), entities={}, relations=[])

    with patch("src.pipeline.dream._step_load", return_value=(empty_graph, {})):
        # Run only step 1 — should get started + completed for step 1, skipped for rest
        run_dream(config, console, dry_run=True, step=1)

    events = read_events(config.memory_path, source="dream", limit=10_000)
    step_events = [e for e in events if e["type"].startswith("dream_step_")]

    # Step 1 should have started + completed
    step1_types = [e["type"] for e in step_events if e["data"].get("step") == 1]
    assert "dream_step_started" in step1_types
    assert "dream_step_completed" in step1_types

    # Other steps should be skipped
    skipped = [e for e in step_events if e["type"] == "dream_step_skipped"]
    assert len(skipped) == 9  # steps 2-10


def test_dream_step_completed_has_duration(tmp_path):
    """Completed step events should have duration_s field."""
    config = _make_config(tmp_path)
    console = Console(file=StringIO())

    from src.core.models import GraphData
    empty_graph = GraphData(generated=datetime.now().isoformat(), entities={}, relations=[])

    with patch("src.pipeline.dream._step_load", return_value=(empty_graph, {})):
        run_dream(config, console, dry_run=True, step=1)

    events = read_events(config.memory_path, source="dream", limit=10_000)
    completed = [e for e in events if e["type"] == "dream_step_completed"]
    assert len(completed) >= 1
    assert "duration_s" in completed[0]["data"]
    assert isinstance(completed[0]["data"]["duration_s"], (int, float))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dream_events.py -v`
Expected: FAIL — no events emitted yet

- [ ] **Step 3: Implement event emission in `run_dream()`**

In `src/pipeline/dream.py`, add the import at the top (after existing imports):

```python
import time
from src.memory.event_log import append_event
```

Then modify `run_dream()` to emit events. The changes are:

1. Before `with DreamDashboard(console) as dashboard:` (after line 141), add:

```python
    session_start_ts = datetime.now().isoformat()
    append_event(memory_path, "dream_session_started", "dream", {
        "dream_id": dream_id,
        "steps_planned": steps_to_run,
        "resumed": bool(checkpoint and resume),
        "entity_count": len(graph.entities),
        "relation_count": len(graph.relations),
    })
```

2. Inside the `for s in range(1, 11):` loop, after `dashboard.skip_step(s)` (line 145), add:

```python
                append_event(memory_path, "dream_step_skipped", "dream", {
                    "dream_id": dream_id, "step": s,
                    "step_name": DREAM_STEPS[s]["name"],
                })
```

(Import `DREAM_STEPS` from `src.pipeline.dream_dashboard` — it's already imported at line 109 inside the function. Move the import to top-level or import `DREAM_STEPS` separately.)

3. After `dashboard.start_step(s)` (line 148), add:

```python
                t0 = time.monotonic()
                append_event(memory_path, "dream_step_started", "dream", {
                    "dream_id": dream_id, "step": s,
                    "step_name": DREAM_STEPS[s]["name"],
                })
```

4. After each `dashboard.complete_step(s, summary)` call (there are multiple — one per step), add the completed event. The cleanest approach: extract a helper called just before `dashboard.complete_step`:

Instead of modifying every step individually, refactor the loop to emit events at the two convergence points. After the entire try block succeeds (just before `_save_checkpoint` on line 240), add:

```python
                # (inside the try, after the step's dashboard.complete_step call)
                elapsed = round(time.monotonic() - t0, 1)
                append_event(memory_path, "dream_step_completed", "dream", {
                    "dream_id": dream_id, "step": s,
                    "step_name": DREAM_STEPS[s]["name"],
                    "duration_s": elapsed,
                    "summary": dashboard._steps[s]["summary"],
                    "details": {},  # Will be enriched in Task 2
                })
```

Note: read the summary back from `dashboard._steps[s]["summary"]` since each step sets it differently via `dashboard.complete_step(s, summary_str)`.

5. In the `except` block (line 243), after `dashboard.fail_step`:

```python
                elapsed = round(time.monotonic() - t0, 1)
                append_event(memory_path, "dream_step_failed", "dream", {
                    "dream_id": dream_id, "step": s,
                    "step_name": DREAM_STEPS[s]["name"],
                    "duration_s": elapsed,
                    "error": str(e)[:200],
                })
```

6. After the `with DreamDashboard` block (after line 248 `_clear_checkpoint`), add:

```python
    steps_completed = sum(1 for s in range(1, 11) if s in steps_to_run)
    steps_failed = len(report.errors)
    total_elapsed = round(time.monotonic() - (t0_session := getattr(run_dream, '_t0', time.monotonic())), 1)
```

Actually, simpler — capture `t0_session` right before the dashboard block:

```python
    t0_session = time.monotonic()
```

Then after `_clear_checkpoint`:

```python
    append_event(memory_path, "dream_session_completed", "dream", {
        "dream_id": dream_id,
        "duration_s": round(time.monotonic() - t0_session, 1),
        "steps_completed": sum(
            1 for e_type in [dashboard._steps[s]["status"] for s in range(1, 11)]
            if e_type == "done"
        ),
        "steps_failed": sum(
            1 for e_type in [dashboard._steps[s]["status"] for s in range(1, 11)]
            if e_type == "failed"
        ),
    })
```

Also need to import `DREAM_STEPS` from `dream_dashboard`. Since the import is already done lazily inside `run_dream` at line 109, move the `DREAM_STEPS` dict to be accessible. The simplest: import it at the top of `run_dream` alongside `DreamDashboard`:

```python
    from src.pipeline.dream_dashboard import DreamDashboard, DREAM_STEPS
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dream_events.py -v`
Expected: PASS

- [ ] **Step 5: Run existing dream tests to verify no regressions**

Run: `uv run pytest tests/test_dream_deterministic.py tests/test_dream_dashboard.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/dream.py tests/test_dream_events.py
git commit -m "feat(dream): emit structured events to event log during dream runs"
```

---

### Task 2: Add `details` dict to step events via `(summary, details)` return convention

**Files:**
- Modify: `src/pipeline/dream.py` (all `_step_*` functions + `run_dream` loop)
- Test: `tests/test_dream_events.py` (extend)

This task changes each `_step_*` function to return `tuple[str, dict]` and wires the `details` into the event emission.

- [ ] **Step 1: Write failing test for details in events**

Add to `tests/test_dream_events.py`:

```python
def test_dream_step_completed_has_details(tmp_path):
    """Completed step events should have a details dict with step-specific counters."""
    config = _make_config(tmp_path)
    console = Console(file=StringIO())

    from src.core.models import GraphData, GraphEntity
    # Create a graph with one entity so step 1 reports non-trivial counts
    entity = GraphEntity(
        file="self/test.md", type="person", title="Test",
        score=0.5, importance=0.5, frequency=1,
        last_mentioned="2026-03-15", retention="long_term",
        aliases=[], tags=[], mention_dates=["2026-03-15"],
        monthly_buckets={}, created="2026-03-15", summary="",
        negative_valence_ratio=0.0,
    )
    graph = GraphData(
        generated=datetime.now().isoformat(),
        entities={"test": entity},
        relations=[],
    )

    with patch("src.pipeline.dream._step_load", return_value=(graph, {})):
        run_dream(config, console, dry_run=True, step=1)

    events = read_events(config.memory_path, source="dream", limit=10_000)
    completed = [e for e in events if e["type"] == "dream_step_completed" and e["data"].get("step") == 1]
    assert len(completed) == 1
    details = completed[0]["data"]["details"]
    assert "entities" in details
    assert details["entities"] == 1
    assert "relations" in details
    assert details["relations"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dream_events.py::test_dream_step_completed_has_details -v`
Expected: FAIL — details is `{}` or missing keys

- [ ] **Step 3: Implement (summary, details) return for each step**

In `src/pipeline/dream.py`, modify the `run_dream()` loop to capture details from each step and pass them to the event. The approach: each step branch computes a `step_details` dict before calling `dashboard.complete_step`.

**Step 1 (Load)** — already inline, just add details:

Replace the step 1 block:
```python
                if s == 1:
                    step_details = {"entities": len(graph.entities), "relations": len(graph.relations)}
                    dashboard.complete_step(s, f"{len(graph.entities)} entities, {len(graph.relations)} relations")
```

**Step 2 (Extract docs)** — `_step_extract_documents` already returns `int`:

```python
                elif s == 2:
                    n = _step_extract_documents(graph, memory_path, config, console, report, dry_run)
                    step_details = {"docs_found": len(list_unextracted_docs(config.faiss.manifest_path)) + n if not dry_run else n, "docs_extracted": n}
```

Simpler: count docs before the call:

```python
                elif s == 2:
                    from src.pipeline.indexer import list_unextracted_docs
                    docs_before = len(list_unextracted_docs(config.faiss.manifest_path))
                    n = _step_extract_documents(graph, memory_path, config, console, report, dry_run)
                    step_details = {"docs_found": docs_before, "docs_extracted": n}
                    dashboard.complete_step(s, f"{n} docs extracted" if n else "no docs pending")
```

**Step 3 (Consolidate)** — report already tracks `facts_consolidated`:

```python
                elif s == 3:
                    facts_before = report.facts_consolidated
                    # ... existing code ...
                    step_details = {"processed": report.facts_consolidated - facts_before, "skipped": 0, "errors": len([e for e in report.errors if "Fact consolidation" in e])}
```

Actually, cleaner: count before and after. The `_step_consolidate_facts` writes directly to `report.facts_consolidated`. We can diff:

```python
                    consolidated_before = report.facts_consolidated
                    errors_before = len(report.errors)
                    # ... existing step 3 code ...
                    step_details = {
                        "processed": report.facts_consolidated - consolidated_before,
                        "skipped": len(entity_paths) - (report.facts_consolidated - consolidated_before),
                        "errors": len(report.errors) - errors_before,
                    }
```

Apply the same pattern for all steps:

**Step 4 (Merge):**
```python
                    merged_before = report.entities_merged
                    errors_before = len(report.errors)
                    # ... existing step 4 code ...
                    step_details = {
                        "candidates": 0,  # Not easily available without modifying _step_merge_entities
                        "merged": report.entities_merged - merged_before,
                        "rejected": 0,
                        "errors": len(report.errors) - errors_before,
                    }
```

For `candidates` count: modify `_step_merge_entities` to return the count. Change its signature to return `int` (number of candidates evaluated):

At the end of `_step_merge_entities` (before the implicit `return`), add:
```python
    return len(merge_candidates)
```

Then in the loop:
```python
                    candidates_count = _step_merge_entities(graph, memory_path, config, console, report, dry_run, entity_paths) or 0
                    step_details = {
                        "candidates": candidates_count,
                        "merged": report.entities_merged - merged_before,
                        "rejected": candidates_count - (report.entities_merged - merged_before),
                    }
```

**Step 5 (Relations):** same pattern. `_step_discover_relations` already prints candidate count. Add return:

At the end of `_step_discover_relations`, return `len(candidates)`:
```python
    return len(candidates)
```

Then:
```python
                    candidates_count = _step_discover_relations(graph, memory_path, config, console, report, dry_run) or 0
                    step_details = {
                        "candidates": candidates_count,
                        "created": report.relations_discovered - discovered_before,
                        "rejected": candidates_count - (report.relations_discovered - discovered_before),
                    }
```

**Step 6 (Transitive):** `_step_transitive_relations` doesn't expose triples_checked. Add return:

At the end, return a tuple `(checked, created)`. Actually, keep it simpler — just return the discovered count (already in `report.transitive_relations`):

```python
                    transitive_before = report.transitive_relations
                    _step_transitive_relations(graph, memory_path, config, console, report, dry_run)
                    step_details = {"created": report.transitive_relations - transitive_before}
```

**Step 7 (Prune):**
```python
                    pruned_before = report.entities_pruned
                    # ... existing step 7 code ...
                    step_details = {"archived": report.entities_pruned - pruned_before}
```

**Step 8 (Summaries):**

Note: to track `skipped` accurately, count entities that already have a summary before calling the step:
```python
                    already_have_summary = sum(1 for e in graph.entities.values() if e.summary)
                    summaries_before = report.summaries_generated
                    errors_before = len(report.errors)
                    _step_generate_summaries(graph, entity_paths, config, console, report, dry_run)
                    generated = report.summaries_generated - summaries_before
                    step_details = {
                        "generated": generated,
                        "skipped": already_have_summary,
                        "errors": len(report.errors) - errors_before,
                    }
```

**Step 9 (Rescore):**
```python
                    step_details = {"entities_scored": len(graph.entities)}
```

**Step 10 (Rebuild):**
```python
                    step_details = {}  # No meaningful counters
```

Then in the completed event emission, replace `"details": {}` with `"details": step_details`:

```python
                append_event(memory_path, "dream_step_completed", "dream", {
                    "dream_id": dream_id, "step": s,
                    "step_name": DREAM_STEPS[s]["name"],
                    "duration_s": elapsed,
                    "summary": dashboard._steps[s]["summary"],
                    "details": step_details,
                })
```

Initialize `step_details = {}` at the start of each loop iteration (right after `t0 = ...`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dream_events.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/test_dream_deterministic.py tests/test_dream_dashboard.py tests/test_dream_events.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/dream.py tests/test_dream_events.py
git commit -m "feat(dream): add step details to dream events"
```

---

## Chunk 2: Persist-Per-Iteration

### Task 3: Add `save_graph()` per merge in step 4

**Files:**
- Modify: `src/pipeline/dream.py:696-701` (inside `_step_merge_entities`, after each successful merge)
- Test: `tests/test_dream_persist.py` (new)

- [ ] **Step 1: Write failing test**

Create `tests/test_dream_persist.py`:

```python
"""Tests for dream per-iteration persistence."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from src.core.models import GraphData, GraphEntity, GraphRelation
from src.pipeline.dream import _step_merge_entities, DreamReport


def _make_graph_with_duplicates():
    """Create a graph with two entities that share aliases (deterministic merge)."""
    entity_a = GraphEntity(
        file="close_ones/alice.md", type="person", title="Alice",
        score=0.8, importance=0.7, frequency=5,
        last_mentioned="2026-03-15", retention="long_term",
        aliases=["alice dupont"], tags=["family"],
        mention_dates=["2026-03-15"], monthly_buckets={},
        created="2025-01-01", summary="Alice.",
        negative_valence_ratio=0.0,
    )
    entity_b = GraphEntity(
        file="close_ones/alice-dupont.md", type="person", title="Alice Dupont",
        score=0.3, importance=0.4, frequency=2,
        last_mentioned="2026-01-01", retention="short_term",
        aliases=["alice"], tags=[],
        mention_dates=["2026-01-01"], monthly_buckets={},
        created="2025-06-01", summary="",
        negative_valence_ratio=0.0,
    )
    graph = GraphData(
        generated=datetime.now().isoformat(),
        entities={"alice": entity_a, "alice-dupont": entity_b},
        relations=[],
    )
    return graph


def test_merge_saves_graph_per_iteration(tmp_path):
    """save_graph should be called after each merge, not just at the end."""
    graph = _make_graph_with_duplicates()
    memory_path = tmp_path / "memory"
    memory_path.mkdir()

    # Create entity files
    for eid, entity in graph.entities.items():
        p = memory_path / entity.file
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"---\ntitle: {entity.title}\ntype: {entity.type}\nretention: long_term\n---\n## Facts\n- [fact] test\n")

    config = MagicMock()
    config.memory_path = memory_path
    config.dream = MagicMock()
    config.dream.faiss_merge_threshold = 0.80
    config.dream.faiss_merge_max_candidates = 20
    console = MagicMock()
    report = DreamReport()

    with patch("src.memory.graph.save_graph") as mock_save, \
         patch("src.pipeline.dream._find_faiss_dedup_candidates", return_value=[]):
        _step_merge_entities(graph, memory_path, config, console, report, dry_run=False, entity_paths={
            "alice": memory_path / "close_ones/alice.md",
            "alice-dupont": memory_path / "close_ones/alice-dupont.md",
        })

    # save_graph should have been called at least once (per merge)
    assert mock_save.call_count >= 1
    assert report.entities_merged >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dream_persist.py::test_merge_saves_graph_per_iteration -v`
Expected: FAIL — save_graph not called (step 4 doesn't call it per iteration today)

- [ ] **Step 3: Implement per-merge save_graph**

In `src/pipeline/dream.py`, in the `_step_merge_entities` function, add `save_graph` import at the top of the function and call after each successful merge.

After line 698 (`report.entities_merged += 1`), add:

```python
            from src.memory.graph import save_graph
            save_graph(memory_path, graph)
```

The full block becomes:

```python
        try:
            _do_merge(keep, drop, graph, memory_path, config, entity_paths)
            report.entities_merged += 1
            from src.memory.graph import save_graph
            save_graph(memory_path, graph)
        except Exception as e:
            report.errors.append(f"Merge failed {drop} -> {keep}: {e}")
            console.print(f"    [yellow]Failed: {e}[/yellow]")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dream_persist.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/dream.py tests/test_dream_persist.py
git commit -m "feat(dream): save_graph after each merge in step 4"
```

---

### Task 4: Add `save_graph()` per relation in step 5

**Files:**
- Modify: `src/pipeline/dream.py:868-876` (inside `_step_discover_relations`)
- Test: `tests/test_dream_persist.py` (extend)

- [ ] **Step 1: Write failing test**

Add to `tests/test_dream_persist.py`:

```python
def test_discover_relations_saves_graph_per_relation(tmp_path):
    """save_graph should be called after each new relation, not just at the end."""
    entity_a = GraphEntity(
        file="interests/cooking.md", type="interest", title="Cooking",
        score=0.7, importance=0.6, frequency=3,
        last_mentioned="2026-03-15", retention="long_term",
        aliases=[], tags=["hobby"], mention_dates=["2026-03-15"],
        monthly_buckets={}, created="2025-01-01", summary="Cooking hobby.",
        negative_valence_ratio=0.0,
    )
    entity_b = GraphEntity(
        file="interests/nutrition.md", type="interest", title="Nutrition",
        score=0.6, importance=0.5, frequency=2,
        last_mentioned="2026-03-10", retention="long_term",
        aliases=[], tags=["health"], mention_dates=["2026-03-10"],
        monthly_buckets={}, created="2025-01-01", summary="Nutrition interest.",
        negative_valence_ratio=0.0,
    )
    graph = GraphData(
        generated=datetime.now().isoformat(),
        entities={"cooking": entity_a, "nutrition": entity_b},
        relations=[],
    )

    memory_path = tmp_path / "memory"
    memory_path.mkdir()
    for eid, entity in graph.entities.items():
        p = memory_path / entity.file
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"---\ntitle: {entity.title}\ntype: {entity.type}\nretention: long_term\n---\n## Facts\n- [fact] test\n")

    config = MagicMock()
    config.memory_path = memory_path
    config.scoring = MagicMock()
    config.scoring.relation_strength_growth = 0.05
    console = MagicMock()
    report = DreamReport()

    # Mock FAISS to return entity_b when searching for entity_a
    mock_result = MagicMock()
    mock_result.entity_id = "nutrition"
    mock_result.score = 0.9

    # Mock LLM to approve the relation
    mock_proposal = MagicMock()
    mock_proposal.action = "relate"
    mock_proposal.relation_type = "linked_to"
    mock_proposal.context = "test context"

    with patch("src.memory.rag.search", return_value=[mock_result]), \
         patch("src.core.llm.call_relation_discovery", return_value=mock_proposal), \
         patch("src.memory.graph.save_graph") as mock_save:
        _step_discover_relations(graph, memory_path, config, console, report, dry_run=False)

    # save_graph called after each relation (not just once at end)
    # With 1 relation created, should be called at least 1 time
    assert mock_save.call_count >= 1
    assert report.relations_discovered >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dream_persist.py::test_discover_relations_saves_graph_per_relation -v`
Expected: FAIL — save_graph currently called once at the end (line 876), not per relation

- [ ] **Step 3: Implement per-relation save_graph**

In `_step_discover_relations`, move the `save_graph` call from the end (lines 875-876) to inside the loop, right after `add_relation`:

Replace lines 868-876:
```python
                add_relation(graph, new_rel, strength_growth=config.scoring.relation_strength_growth)
                discovered += 1
                console.print(f"    [green]{entity_a.title} -> {rel_type} -> {entity_b.title}[/green]")
        except Exception as e:
            report.errors.append(f"Relation discovery failed for {eid_a}/{eid_b}: {e}")

    report.relations_discovered = discovered
    if discovered and not dry_run:
        save_graph(memory_path, graph)
        console.print(f"  [green]Discovered {discovered} new relation(s)[/green]")
```

With:
```python
                add_relation(graph, new_rel, strength_growth=config.scoring.relation_strength_growth)
                save_graph(memory_path, graph)
                discovered += 1
                console.print(f"    [green]{entity_a.title} -> {rel_type} -> {entity_b.title}[/green]")
        except Exception as e:
            report.errors.append(f"Relation discovery failed for {eid_a}/{eid_b}: {e}")

    report.relations_discovered = discovered
    if discovered and not dry_run:
        console.print(f"  [green]Discovered {discovered} new relation(s)[/green]")
```

Note: `save_graph` is already imported at the top of `_step_discover_relations` (line 795).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dream_persist.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/dream.py tests/test_dream_persist.py
git commit -m "feat(dream): save_graph after each relation in step 5"
```

---

### Task 5: Add `save_graph()` per summary in step 8

**Files:**
- Modify: `src/pipeline/dream.py:1143-1149` (inside `_step_generate_summaries`)
- Test: `tests/test_dream_persist.py` (extend)

- [ ] **Step 1: Write failing test**

Add to `tests/test_dream_persist.py`:

```python
def test_summaries_saves_graph_per_entity(tmp_path):
    """save_graph should be called after each summary is generated."""
    entity = GraphEntity(
        file="interests/cooking.md", type="interest", title="Cooking",
        score=0.7, importance=0.6, frequency=3,
        last_mentioned="2026-03-15", retention="long_term",
        aliases=[], tags=[], mention_dates=["2026-03-15"],
        monthly_buckets={}, created="2025-01-01", summary="",  # No summary yet
        negative_valence_ratio=0.0,
    )
    graph = GraphData(
        generated=datetime.now().isoformat(),
        entities={"cooking": entity},
        relations=[],
    )

    memory_path = tmp_path / "memory"
    memory_path.mkdir()
    p = memory_path / "interests/cooking.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("---\ntitle: Cooking\ntype: interest\nretention: long_term\n---\n## Facts\n- [fact] likes cooking\n")

    config = MagicMock()
    config.memory_path = memory_path
    console = MagicMock()
    report = DreamReport()

    with patch("src.core.llm.call_entity_summary", return_value="A cooking enthusiast."), \
         patch("src.memory.graph.save_graph") as mock_save:
        from src.pipeline.dream import _step_generate_summaries
        _step_generate_summaries(graph, {"cooking": p}, config, console, report, dry_run=False)

    assert mock_save.call_count >= 1
    assert report.summaries_generated >= 1
    assert graph.entities["cooking"].summary == "A cooking enthusiast."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dream_persist.py::test_summaries_saves_graph_per_entity -v`
Expected: FAIL — no save_graph call in step 8

- [ ] **Step 3: Implement per-summary save_graph**

In `_step_generate_summaries`, after line 1146 (`write_entity(path, fm, sections)`), add:

```python
                from src.memory.graph import save_graph
                save_graph(config.memory_path, graph)
```

The full block (lines 1143-1149) becomes:

```python
            if summary:
                fm.summary = summary
                entity.summary = summary
                write_entity(path, fm, sections)
                from src.memory.graph import save_graph
                save_graph(config.memory_path, graph)
                report.summaries_generated += 1
                display = f"{summary[:60]}..." if len(summary) > 60 else summary
                console.print(f"  [green]{entity.title}: {display}[/green]")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dream_persist.py -v`
Expected: PASS

- [ ] **Step 5: Run all dream tests**

Run: `uv run pytest tests/test_dream_deterministic.py tests/test_dream_dashboard.py tests/test_dream_events.py tests/test_dream_persist.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/dream.py tests/test_dream_persist.py
git commit -m "feat(dream): save_graph after each summary in step 8"
```

---

## Chunk 3: Dream Report + CLI

### Task 6: Generate `_dream_report.md` after each dream run

**Files:**
- Modify: `src/pipeline/dream.py` (add `_generate_dream_report` function, call from `run_dream`)
- Test: `tests/test_dream_report.py` (new)

- [ ] **Step 1: Write failing test**

Create `tests/test_dream_report.py`:

```python
"""Tests for dream report generation."""

from datetime import datetime
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

from rich.console import Console

from src.pipeline.dream import run_dream, _generate_dream_report
from src.core.models import GraphData
from src.memory.event_log import append_event, read_events


def _make_config(tmp_path: Path):
    config = MagicMock()
    memory_path = tmp_path / "memory"
    memory_path.mkdir()
    (memory_path / "chats").mkdir()
    (memory_path / "_inbox").mkdir()
    config.memory_path = memory_path
    config.dream = MagicMock()
    config.dream.prune_score_threshold = 0.1
    config.dream.prune_max_frequency = 1
    config.dream.prune_min_age_days = 90
    config.scoring = MagicMock()
    config.scoring.relation_strength_growth = 0.05
    return config


def test_dream_generates_report_file(tmp_path):
    """run_dream should generate _dream_report.md after completion."""
    config = _make_config(tmp_path)
    console = Console(file=StringIO())

    empty_graph = GraphData(generated=datetime.now().isoformat(), entities={}, relations=[])

    with patch("src.pipeline.dream._step_load", return_value=(empty_graph, {})):
        run_dream(config, console, dry_run=True, step=1)

    report_path = config.memory_path / "_dream_report.md"
    assert report_path.exists()
    content = report_path.read_text()
    assert "# Dream Report" in content
    assert "Load" in content


def test_generate_dream_report_format(tmp_path):
    """Report should have session header, steps table, and details sections."""
    memory_path = tmp_path / "memory"
    memory_path.mkdir()

    dream_id = "2026-03-15T14:00:00"
    ts_before = "2026-03-15T13:59:59"

    # Simulate events
    append_event(memory_path, "dream_session_started", "dream", {
        "dream_id": dream_id, "steps_planned": [1, 9, 10],
        "resumed": False, "entity_count": 42, "relation_count": 10,
    })
    append_event(memory_path, "dream_step_started", "dream", {
        "dream_id": dream_id, "step": 1, "step_name": "Load",
    })
    append_event(memory_path, "dream_step_completed", "dream", {
        "dream_id": dream_id, "step": 1, "step_name": "Load",
        "duration_s": 0.2, "summary": "42 entities, 10 relations",
        "details": {"entities": 42, "relations": 10},
    })
    append_event(memory_path, "dream_step_skipped", "dream", {
        "dream_id": dream_id, "step": 2, "step_name": "Extract docs",
    })
    append_event(memory_path, "dream_session_completed", "dream", {
        "dream_id": dream_id, "duration_s": 1.5,
        "steps_completed": 1, "steps_failed": 0,
    })

    report_path = _generate_dream_report(memory_path, dream_id, ts_before)

    assert report_path.exists()
    content = report_path.read_text()
    assert "# Dream Report" in content
    assert "2026-03-15" in content
    assert "| 1 | Load |" in content or "Load" in content
    assert "42" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dream_report.py -v`
Expected: FAIL — `_generate_dream_report` doesn't exist

- [ ] **Step 3: Implement `_generate_dream_report`**

Add to `src/pipeline/dream.py`, after `_clear_checkpoint`:

```python
def _generate_dream_report(memory_path: Path, dream_id: str, session_start_ts: str) -> Path:
    """Generate a markdown report from dream session events."""
    from src.memory.event_log import read_events

    events = read_events(memory_path, source="dream", after=session_start_ts, limit=10_000)
    # Filter to this session's dream_id
    events = [e for e in events if e.get("data", {}).get("dream_id") == dream_id]

    # Extract session info
    session_completed = next(
        (e for e in events if e["type"] == "dream_session_completed"), None
    )
    duration_s = session_completed["data"].get("duration_s", 0) if session_completed else 0
    steps_completed = session_completed["data"].get("steps_completed", 0) if session_completed else 0
    steps_failed = session_completed["data"].get("steps_failed", 0) if session_completed else 0
    status = "completed" if steps_failed == 0 else f"{steps_failed} failed"

    # Format duration
    if duration_s >= 60:
        mins = int(duration_s // 60)
        secs = int(duration_s % 60)
        duration_str = f"{mins}m {secs}s"
    else:
        duration_str = f"{duration_s:.1f}s"

    date_str = dream_id[:10]
    lines = [
        f"# Dream Report — {date_str}",
        "",
        f"**Session**: {dream_id} | **Duration**: {duration_str} | **Status**: {status}",
        "",
        "## Steps",
        "",
        "| # | Step | Status | Duration | Summary |",
        "|---|------|--------|----------|---------|",
    ]

    # Collect step events
    step_completed = {e["data"]["step"]: e for e in events if e["type"] == "dream_step_completed"}
    step_failed = {e["data"]["step"]: e for e in events if e["type"] == "dream_step_failed"}
    step_skipped = {e["data"]["step"]: e for e in events if e["type"] == "dream_step_skipped"}

    details_sections = []

    for s in range(1, 11):
        if s in step_completed:
            d = step_completed[s]["data"]
            dur = f"{d['duration_s']:.1f}s" if d.get("duration_s") else "—"
            lines.append(f"| {s} | {d.get('step_name', '')} | done | {dur} | {d.get('summary', '')} |")
            if d.get("details"):
                detail_lines = [f"### {s}. {d.get('step_name', '')}"]
                for k, v in d["details"].items():
                    detail_lines.append(f"- {k.replace('_', ' ').title()}: {v}")
                details_sections.append("\n".join(detail_lines))
        elif s in step_failed:
            d = step_failed[s]["data"]
            dur = f"{d['duration_s']:.1f}s" if d.get("duration_s") else "—"
            lines.append(f"| {s} | {d.get('step_name', '')} | FAILED | {dur} | {d.get('error', '')} |")
        elif s in step_skipped:
            d = step_skipped[s]["data"]
            lines.append(f"| {s} | {d.get('step_name', '')} | skipped | — | — |")

    if details_sections:
        lines.append("")
        lines.append("## Details")
        lines.append("")
        lines.extend(details_sections)

    lines.append("")

    report_path = memory_path / "_dream_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
```

Then in `run_dream()`, after `_clear_checkpoint(memory_path)` and after the `dream_session_completed` event:

```python
    if not dry_run:
        _clear_checkpoint(memory_path)

    # Always generate report (even dry_run)
    _generate_dream_report(memory_path, dream_id, session_start_ts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dream_report.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/dream.py tests/test_dream_report.py
git commit -m "feat(dream): generate _dream_report.md after each dream run"
```

---

### Task 7: Add `--report` flag to CLI

**Files:**
- Modify: `src/cli.py:502-538` (dream command)
- Test: manual (CLI flag)

- [ ] **Step 1: Add `--report` option to the dream command**

In `src/cli.py`, modify the `dream` command decorator and function:

Add the option:
```python
@click.option("--report", "show_report", is_flag=True, help="Show last dream report and exit")
```

Update the function signature:
```python
def dream(ctx, dry_run, step, resume, reset, show_report):
```

At the start of the function body (before any other logic), add:

```python
    if show_report:
        config = ctx.obj["config"]
        report_path = config.memory_path / "_dream_report.md"
        if report_path.exists():
            click.echo(report_path.read_text(encoding="utf-8"))
        else:
            click.echo("No dream report found. Run `memory dream` first.")
        return
```

- [ ] **Step 2: Test manually**

Run: `uv run memory dream --report`
Expected: either the report content or "No dream report found."

- [ ] **Step 3: Commit**

```bash
git add src/cli.py
git commit -m "feat(cli): add --report flag to dream command"
```

---

## Chunk 4: Dashboard Enrichments

### Task 8: Fix `_extract_dream_sessions` grouping + filter

**Files:**
- Modify: `src/pipeline/dashboard.py:170-182`
- Test: `tests/test_dashboard_dream.py` (new)

- [ ] **Step 1: Write failing test**

Create `tests/test_dashboard_dream.py`:

```python
"""Tests for dashboard dream session extraction."""

from src.pipeline.dashboard import _extract_dream_sessions


def test_extract_dream_sessions_groups_by_dream_id():
    """Sessions should be grouped by dream_id, not by calendar day."""
    events = [
        {"type": "dream_session_started", "ts": "2026-03-15T10:00:00", "data": {"dream_id": "2026-03-15T10:00:00", "steps_planned": [1, 9, 10]}},
        {"type": "dream_step_completed", "ts": "2026-03-15T10:00:01", "data": {"dream_id": "2026-03-15T10:00:00", "step": 1, "step_name": "Load", "duration_s": 0.2, "summary": "ok"}},
        {"type": "dream_session_completed", "ts": "2026-03-15T10:00:05", "data": {"dream_id": "2026-03-15T10:00:00", "duration_s": 5.0, "steps_completed": 1}},
        # Second session same day
        {"type": "dream_session_started", "ts": "2026-03-15T14:00:00", "data": {"dream_id": "2026-03-15T14:00:00", "steps_planned": [1, 3]}},
        {"type": "dream_step_completed", "ts": "2026-03-15T14:00:01", "data": {"dream_id": "2026-03-15T14:00:00", "step": 1, "step_name": "Load", "duration_s": 0.1, "summary": "ok"}},
        {"type": "dream_session_completed", "ts": "2026-03-15T14:00:10", "data": {"dream_id": "2026-03-15T14:00:00", "duration_s": 10.0, "steps_completed": 1}},
    ]
    sessions = _extract_dream_sessions(events)
    # Should have 2 sessions, not 1 (because they have different dream_ids)
    assert len(sessions) == 2


def test_extract_dream_sessions_includes_session_events():
    """Session events (dream_session_*) should not be filtered out."""
    events = [
        {"type": "dream_session_started", "ts": "2026-03-15T10:00:00", "data": {"dream_id": "d1", "steps_planned": [1]}},
        {"type": "dream_step_completed", "ts": "2026-03-15T10:00:01", "data": {"dream_id": "d1", "step": 1, "step_name": "Load"}},
        {"type": "dream_session_completed", "ts": "2026-03-15T10:00:05", "data": {"dream_id": "d1", "duration_s": 5.0, "steps_completed": 1}},
    ]
    sessions = _extract_dream_sessions(events)
    assert len(sessions) == 1
    session = sessions[0]
    # Should have session-level metadata
    assert "duration_s" in session or any(e["type"] == "dream_session_completed" for e in session.get("steps", []))


def test_extract_dream_sessions_has_date_label():
    """Each session should have a date label for display."""
    events = [
        {"type": "dream_session_started", "ts": "2026-03-15T10:00:00", "data": {"dream_id": "2026-03-15T10:00:00"}},
        {"type": "dream_step_completed", "ts": "2026-03-15T10:00:01", "data": {"dream_id": "2026-03-15T10:00:00", "step": 1, "step_name": "Load"}},
    ]
    sessions = _extract_dream_sessions(events)
    assert len(sessions) == 1
    assert sessions[0]["date"] == "2026-03-15"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dashboard_dream.py -v`
Expected: FAIL — current code groups by day and filters out `dream_session_*`

- [ ] **Step 3: Implement the fix**

Replace `_extract_dream_sessions` in `src/pipeline/dashboard.py` (lines 170-182):

```python
def _extract_dream_sessions(events: list[dict]) -> list[dict]:
    """Group dream events by dream_id into sessions."""
    sessions: dict[str, dict] = {}
    for evt in events:
        evt_type = evt.get("type", "")
        if not (evt_type.startswith("dream_step_") or evt_type.startswith("dream_session_")):
            continue
        dream_id = evt.get("data", {}).get("dream_id", "")
        if not dream_id:
            # Fallback for legacy events without dream_id: group by day
            ts = evt.get("ts", "")
            dream_id = ts[:10] if len(ts) >= 10 else "unknown"
        if dream_id not in sessions:
            ts = evt.get("ts", "")
            sessions[dream_id] = {
                "date": ts[:10] if len(ts) >= 10 else "unknown",
                "dream_id": dream_id,
                "steps": [],
                "duration_s": None,
                "steps_completed": None,
                "steps_failed": None,
            }
        if evt_type == "dream_session_completed":
            d = evt.get("data", {})
            sessions[dream_id]["duration_s"] = d.get("duration_s")
            sessions[dream_id]["steps_completed"] = d.get("steps_completed")
            sessions[dream_id]["steps_failed"] = d.get("steps_failed")
        elif evt_type.startswith("dream_step_"):
            sessions[dream_id]["steps"].append(evt)
    return sorted(sessions.values(), key=lambda s: s["date"], reverse=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dashboard_dream.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/dashboard.py tests/test_dashboard_dream.py
git commit -m "fix(dashboard): group dream sessions by dream_id, include session events"
```

---

### Task 9: Display `details` dict and session header in dashboard JS

**Files:**
- Modify: `src/pipeline/dashboard.py` (JS template sections: `renderDream` and `showDreamStepDetail`)

- [ ] **Step 1: Update `renderDream()` to show session header**

In `src/pipeline/dashboard.py`, replace the `renderDream()` JS function (lines 993-1013):

```javascript
function renderDream() {
    var container = document.getElementById('dream-container');
    if (!DREAM_SESSIONS.length) {
        container.innerHTML = '<p style="color:var(--text-secondary);padding:20px;">No dream sessions found in event log.</p>';
        return;
    }
    container.innerHTML = DREAM_SESSIONS.map(function(session, si) {
        var header = '\u{1F319} Dream \u2014 ' + session.date;
        if (session.duration_s != null) {
            var d = session.duration_s;
            header += ' \u2014 ' + (d >= 60 ? Math.floor(d/60) + 'm ' + Math.floor(d%60) + 's' : d.toFixed(1) + 's');
        }
        if (session.steps_completed != null) {
            header += ' \u2014 ' + session.steps_completed + ' completed';
            if (session.steps_failed) header += ', ' + session.steps_failed + ' failed';
        }
        return '<div class="dream-session"><h3>' + header + '</h3>' +
            session.steps.map(function(step, sti) {
                var d = step.data || {};
                var status = step.type === 'dream_step_completed' ? '\u2705' :
                             step.type === 'dream_step_failed' ? '\u274C' :
                             step.type === 'dream_step_started' ? '\u23F3' : '\u23ED';
                return '<div class="dream-step clickable" onclick="showDreamStepDetail(' + si + ',' + sti + ')">' +
                    '<span>' + status + '</span>' +
                    '<span>' + (d.step || '') + ' ' + escapeHtml(d.step_name || d.description || '') + '</span>' +
                    '<span class="step-duration">' + (d.duration_s ? d.duration_s.toFixed(1) + 's' : '') + '</span>' +
                    '<span class="step-summary">' + escapeHtml(d.summary || d.error || '') + '</span></div>';
            }).join('') + '</div>';
    }).join('');
}
```

- [ ] **Step 2: Update `showDreamStepDetail()` to render details**

In `src/pipeline/dashboard.py`, in the `showDreamStepDetail()` function, the existing code at lines 1046-1057 renders "extra data fields" as raw JSON. The `details` dict is NOT excluded by the filter list (line 1048 only excludes `step`, `step_name`, `description`, `duration_s`, `summary`, `error`), so `details` currently appears as a raw JSON blob. We improve this to render a clean key-value grid instead.

Replace lines 1046-1057:

```javascript
    // Show details dict with friendly formatting
    if (d.details && Object.keys(d.details).length) {
        html += '<div class="section-title">Details</div>';
        html += '<div style="display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:13px;padding:8px 0;">';
        Object.keys(d.details).forEach(function(k) {
            html += '<span style="color:var(--text-secondary)">' + escapeHtml(k.replace(/_/g, ' ')) + ':</span>';
            html += '<span style="color:var(--text-primary);font-weight:500">' + d.details[k] + '</span>';
        });
        html += '</div>';
    }

    // Show remaining extra data fields
    var extraKeys = Object.keys(d).filter(function(k) {
        return ['step', 'step_name', 'description', 'duration_s', 'summary', 'error', 'dream_id', 'details'].indexOf(k) < 0;
    });
    if (extraKeys.length) {
        html += '<div class="section-title">Extra</div>';
        html += '<pre style="font-size:11px;color:var(--text-secondary);white-space:pre-wrap;word-break:break-all;line-height:1.5;">';
        var extraData = {};
        extraKeys.forEach(function(k) { extraData[k] = d[k]; });
        html += escapeHtml(JSON.stringify(extraData, null, 2));
        html += '</pre>';
    }
```

- [ ] **Step 3: Test manually**

Run: `uv run memory dream --step 1` then `uv run memory graph`
Navigate to the Dream tab in the browser. Verify:
- Session header shows date + duration + step counts
- Clicking a step shows the details grid
- Multiple sessions on the same day appear separately

- [ ] **Step 4: Commit**

```bash
git add src/pipeline/dashboard.py
git commit -m "feat(dashboard): enrich dream view with session header and details grid"
```

---

### Task 10: Final integration test

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: End-to-end manual test**

```bash
# Run a dream (dry run to avoid LLM calls)
uv run memory dream --dry-run

# Check report was generated
cat memory/_dream_report.md

# Check events were logged
tail -20 memory/_event_log.jsonl | python -m json.tool

# View dashboard
uv run memory graph
# → Navigate to Dream tab, verify sessions appear

# Test --report flag
uv run memory dream --report
```

- [ ] **Step 3: Final commit**

If any fixes were needed during integration testing, commit them:

```bash
git add -u
git commit -m "fix(dream): integration test fixes for dream observability"
```
