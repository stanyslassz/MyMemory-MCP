# MyMemory — Plan d'Implémentation Globale

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 10 critical bugs, add MCP CRUD tools, hybrid RRF search, simplify prompts for small LLMs, and add observability features (action log, ACT-R insights, LLM dedup).

**Architecture:** 5-phase incremental improvement. Phase 1 (bug fixes) is the prerequisite. Phases 2-5 can be parallelized after Phase 1. Each task follows TDD: write failing test, implement, verify, commit.

**Tech Stack:** Python 3.11+, Pydantic, FAISS, SQLite FTS5, FastMCP, Instructor, Click CLI, uv

**Design doc:** `docs/plans/2026-03-10-memory-improvements-design.md`
**Audit files:** `docs/audit/01-*.md` through `docs/audit/10-*.md`

---

## Phase 1 — Correctifs Critiques

### Task 1: `remove_relation()` dans graph.py

**Files:**
- Modify: `src/memory/graph.py:96-125`
- Test: `tests/test_graph.py`

**Step 1: Write the failing test**

```python
# tests/test_graph.py — append to existing file

def test_remove_relation():
    """remove_relation() should delete a relation by (from, to, type) tuple."""
    from src.memory.graph import remove_relation
    from src.core.models import GraphData, GraphEntity, GraphRelation

    graph = GraphData(
        generated="2026-03-10",
        entities={
            "alice": GraphEntity(file="close_ones/alice.md", type="person", title="Alice", score=0.5),
            "bob": GraphEntity(file="close_ones/bob.md", type="person", title="Bob", score=0.5),
        },
        relations=[
            GraphRelation(from_entity="alice", to_entity="bob", type="parent_of"),
            GraphRelation(from_entity="alice", to_entity="bob", type="friend_of"),
        ],
    )

    result = remove_relation(graph, "alice", "bob", "parent_of")
    assert result is True
    assert len(graph.relations) == 1
    assert graph.relations[0].type == "friend_of"


def test_remove_relation_not_found():
    """remove_relation() returns False if no matching relation exists."""
    from src.memory.graph import remove_relation
    from src.core.models import GraphData, GraphRelation

    graph = GraphData(generated="2026-03-10", entities={}, relations=[
        GraphRelation(from_entity="a", to_entity="b", type="affects"),
    ])

    result = remove_relation(graph, "a", "b", "linked_to")
    assert result is False
    assert len(graph.relations) == 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph.py::test_remove_relation tests/test_graph.py::test_remove_relation_not_found -v`
Expected: FAIL with ImportError (remove_relation not defined)

**Step 3: Write minimal implementation**

Add to `src/memory/graph.py` after `add_relation()` (~line 117):

```python
def remove_relation(graph: GraphData, from_entity: str, to_entity: str, rel_type: str) -> bool:
    """Remove a specific relation by (from, to, type) tuple. Returns True if found and removed."""
    before = len(graph.relations)
    graph.relations = [
        r for r in graph.relations
        if not (r.from_entity == from_entity and r.to_entity == to_entity and r.type == rel_type)
    ]
    return len(graph.relations) < before
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_graph.py::test_remove_relation tests/test_graph.py::test_remove_relation_not_found -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/memory/graph.py tests/test_graph.py
git commit -m "feat: add remove_relation() to graph.py for relation deletion"
```

---

### Task 2: `supersedes` field on `RawRelation` model

**Files:**
- Modify: `src/core/models.py:48-52`
- Test: `tests/test_models.py` (create if needed)

**Step 1: Write the failing test**

```python
# tests/test_models.py

def test_raw_relation_supersedes_field():
    """RawRelation should have an optional supersedes field."""
    from src.core.models import RawRelation

    rel = RawRelation(from_name="Alice", to_name="Bob", type="parent_of")
    assert rel.supersedes == ""

    rel2 = RawRelation(
        from_name="Alice", to_name="Bob", type="parent_of",
        supersedes="alice:bob:linked_to"
    )
    assert rel2.supersedes == "alice:bob:linked_to"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py::test_raw_relation_supersedes_field -v`
Expected: FAIL (supersedes not a field)

**Step 3: Write minimal implementation**

In `src/core/models.py:48-52`, add `supersedes` field to `RawRelation`:

```python
class RawRelation(BaseModel):
    from_name: str
    to_name: str
    type: RelationType
    context: str = ""
    supersedes: str = ""  # Format: "from_slug:to_slug:relation_type"
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py::test_raw_relation_supersedes_field -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/core/models.py tests/test_models.py
git commit -m "feat: add supersedes field to RawRelation for relation corrections"
```

---

### Task 3: `remove_relation_line()` in store.py

**Files:**
- Modify: `src/memory/store.py`
- Test: `tests/test_store.py` (or create)

**Step 1: Write the failing test**

```python
# tests/test_store.py

import tempfile
from pathlib import Path

def test_remove_relation_line():
    """remove_relation_line() should remove a specific relation from MD file."""
    from src.memory.store import remove_relation_line

    content = """---
title: Alice
type: person
---
## Facts
- [fact] Likes coffee

## Relations
- parent_of [[Bob]]
- friend_of [[Carol]]

## History
- 2026-03-10: Created
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        path = Path(f.name)

    result = remove_relation_line(path, "parent_of", "Bob")
    assert result is True

    text = path.read_text()
    assert "parent_of [[Bob]]" not in text
    assert "friend_of [[Carol]]" in text
    path.unlink()


def test_remove_relation_line_not_found():
    """remove_relation_line() returns False if relation not found."""
    from src.memory.store import remove_relation_line

    content = """---
title: Alice
type: person
---
## Facts
## Relations
- friend_of [[Carol]]
## History
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        path = Path(f.name)

    result = remove_relation_line(path, "parent_of", "Bob")
    assert result is False
    path.unlink()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_store.py::test_remove_relation_line tests/test_store.py::test_remove_relation_line_not_found -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Add to `src/memory/store.py` after `mark_observation_superseded()` (~line 509):

```python
def remove_relation_line(entity_path: Path, relation_type: str, target_title: str) -> bool:
    """Remove a specific relation line from ## Relations section.

    Looks for lines matching '- {relation_type} [[{target_title}]]' (case-insensitive on title).
    Returns True if a line was removed.
    """
    frontmatter, sections = read_entity(entity_path)
    relations_text = sections.get("Relations", "")
    if not relations_text:
        return False

    lines = relations_text.strip().split("\n")
    target_lower = target_title.lower()
    new_lines = []
    removed = False
    for line in lines:
        stripped = line.strip()
        # Match pattern: - relation_type [[Target Title]]
        if (stripped.startswith(f"- {relation_type} [[")
                and target_lower in stripped.lower()):
            removed = True
        else:
            new_lines.append(line)

    if removed:
        sections["Relations"] = "\n".join(new_lines)
        write_entity(entity_path, frontmatter, sections)
    return removed
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_store.py::test_remove_relation_line tests/test_store.py::test_remove_relation_line_not_found -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/memory/store.py tests/test_store.py
git commit -m "feat: add remove_relation_line() for MD relation cleanup"
```

---

### Task 4: Relation supersession in enricher.py

**Files:**
- Modify: `src/pipeline/enricher.py:61-95` (relation processing section)
- Modify: `src/memory/graph.py` (import remove_relation)
- Modify: `src/memory/store.py` (import remove_relation_line)
- Test: `tests/test_enricher.py`

**Step 1: Write the failing test**

```python
# tests/test_enricher.py

def test_relation_supersession():
    """When a relation has supersedes set, the old relation should be removed."""
    import tempfile, json
    from pathlib import Path
    from src.core.models import (
        GraphData, GraphEntity, GraphRelation,
        RawRelation, ResolvedExtraction, Resolution,
    )
    from src.pipeline.enricher import enrich_memory
    from src.core.config import load_config

    # Setup: create a minimal graph with an existing wrong relation
    # alice parent_of bob (WRONG — should be friend_of)
    # New extraction says: alice friend_of bob, supersedes alice:bob:parent_of

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        # Create entity files and graph...
        # (setup code depends on existing test patterns — adapt from existing tests)

        graph = GraphData(
            generated="2026-03-10",
            entities={
                "alice": GraphEntity(file="close_ones/alice.md", type="person", title="Alice", score=0.5),
                "bob": GraphEntity(file="close_ones/bob.md", type="person", title="Bob", score=0.5),
            },
            relations=[
                GraphRelation(from_entity="alice", to_entity="bob", type="parent_of"),
            ],
        )

        # Create minimal MD files
        (memory_path / "close_ones").mkdir(parents=True)
        for slug, title in [("alice", "Alice"), ("bob", "Bob")]:
            (memory_path / f"close_ones/{slug}.md").write_text(
                f"---\ntitle: {title}\ntype: person\n---\n## Facts\n\n## Relations\n- parent_of [[{'Bob' if slug == 'alice' else 'Alice'}]]\n\n## History\n"
            )

        # Save graph
        graph_path = memory_path / "_graph.json"
        graph_path.write_text(graph.model_dump_json(indent=2, by_alias=True))

        # The new relation with supersedes
        resolved = ResolvedExtraction(
            entities=[],
            relations=[
                RawRelation(
                    from_name="Alice", to_name="Bob", type="friend_of",
                    supersedes="alice:bob:parent_of"
                )
            ],
            resolutions={"alice": Resolution(status="resolved", entity_id="alice"),
                         "bob": Resolution(status="resolved", entity_id="bob")},
            summary="correction",
        )

        config = load_config(None, memory_path)
        report = enrich_memory(resolved, config, "2026-03-10")

        # Reload graph and check
        updated_graph_text = graph_path.read_text()
        updated_graph = GraphData.model_validate_json(updated_graph_text)

        # Old relation should be gone
        parent_rels = [r for r in updated_graph.relations if r.type == "parent_of"]
        assert len(parent_rels) == 0, f"Old parent_of relation should be removed, found {parent_rels}"

        # New relation should exist
        friend_rels = [r for r in updated_graph.relations if r.type == "friend_of"]
        assert len(friend_rels) == 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_enricher.py::test_relation_supersession -v`
Expected: FAIL (supersedes not handled)

**Step 3: Write minimal implementation**

In `src/pipeline/enricher.py`, in the relation processing section (~line 61-95), add supersession logic before `add_relation()`:

```python
# Before adding the new relation, handle supersession
if hasattr(relation, 'supersedes') and relation.supersedes:
    parts = relation.supersedes.split(":")
    if len(parts) == 3:
        old_from, old_to, old_type = parts
        remove_relation(graph, old_from, old_to, old_type)
        # Also clean MD files
        from_entity_file = graph.entities.get(old_from)
        if from_entity_file:
            from_path = memory_path / from_entity_file.file
            if from_path.exists():
                old_target_title = graph.entities.get(old_to, GraphEntity()).title
                remove_relation_line(from_path, old_type, old_target_title)
```

Add imports at top of enricher.py:
```python
from src.memory.graph import remove_relation
from src.memory.store import remove_relation_line
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_enricher.py::test_relation_supersession -v`
Expected: PASS

**Step 5: Run all existing tests to verify no regressions**

Run: `uv run pytest tests/ -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/pipeline/enricher.py tests/test_enricher.py
git commit -m "feat: implement relation supersession in enricher pipeline"
```

---

### Task 5: Update extract_facts.md with relation correction guidance

**Files:**
- Modify: `prompts/extract_facts.md:19-23` (after observation supersedes section)

**Step 1: Read current prompt**

Run: `cat prompts/extract_facts.md` to see exact current content

**Step 2: Add relation correction section**

After the observation supersedes instructions (around line 23), add:

```markdown
### Relation corrections
If the user corrects a relationship between entities (e.g., "Louise is not my wife, she's my daughter"):
- Extract the NEW correct relation with proper type
- Set `supersedes` field to the old relation in format "from_slug:to_slug:old_relation_type"
  - Use slugified names (lowercase, hyphens): "Jean-Pierre" → "jean-pierre"
- Example: user says "Louise est ma fille, pas ma femme"
  → relation: {"from_name": "User", "to_name": "Louise", "type": "parent_of", "context": "correction", "supersedes": "user:louise:linked_to"}

IMPORTANT: Only set `supersedes` when the user EXPLICITLY corrects a previous statement. Do not guess.
```

**Step 3: Commit**

```bash
git add prompts/extract_facts.md
git commit -m "feat: add relation correction guidance to extraction prompt"
```

---

### Task 6: Fix context.py — filter weak/stale relations

**Files:**
- Modify: `src/memory/context.py:234-246` (_enrich_entity relation gathering)
- Test: `tests/test_context.py`

**Step 1: Write the failing test**

```python
# tests/test_context.py

def test_enrich_entity_filters_weak_relations():
    """_enrich_entity should exclude relations with strength < 0.3."""
    from src.core.models import GraphData, GraphEntity, GraphRelation
    from datetime import date

    graph = GraphData(
        generated="2026-03-10",
        entities={
            "alice": GraphEntity(file="close_ones/alice.md", type="person", title="Alice", score=0.7),
            "bob": GraphEntity(file="close_ones/bob.md", type="person", title="Bob", score=0.5),
            "carol": GraphEntity(file="close_ones/carol.md", type="person", title="Carol", score=0.5),
        },
        relations=[
            GraphRelation(from_entity="alice", to_entity="bob", type="friend_of", strength=0.8),
            GraphRelation(from_entity="alice", to_entity="carol", type="linked_to", strength=0.1),  # Weak — should be filtered
        ],
    )

    # Call _enrich_entity for alice and check relations section
    # (exact test depends on _enrich_entity signature — adapt)
    # The weak relation to carol should NOT appear in the enriched output
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_context.py::test_enrich_entity_filters_weak_relations -v`
Expected: FAIL (weak relations currently included)

**Step 3: Write minimal implementation**

In `src/memory/context.py:234-246`, add filtering before building the relations section:

```python
# Filter weak/stale relations before inclusion
from datetime import date, timedelta

today = date.today()
min_strength = config.scoring.min_score_for_context if hasattr(config.scoring, 'min_score_for_context') else 0.3
max_age_days = 365

filtered_relations = []
for r in entity_relations:
    if r.strength < min_strength:
        continue
    if r.last_reinforced:
        try:
            last = date.fromisoformat(r.last_reinforced)
            if (today - last).days > max_age_days:
                continue
        except (ValueError, TypeError):
            pass
    filtered_relations.append(r)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_context.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/memory/context.py tests/test_context.py
git commit -m "fix: filter weak/stale relations from context output"
```

---

### Task 7: Fix dream_plan.md step numbering

**Files:**
- Modify: `prompts/dream_plan.md`

**Step 1: Read current prompt**

Read `prompts/dream_plan.md` to see exact step numbering.

**Step 2: Update to match code's 10-step structure**

Update the step list in `dream_plan.md` to include all 10 steps with step 6 (Transitive Relations):

```markdown
Steps:
1. Load — Load graph and scan entity files
2. Extract docs — Extract knowledge from unextracted RAG documents
3. Fact consolidation — Merge redundant facts in entities with many observations
4. Entity merging — Merge duplicate entities detected by slug/alias overlap
5. Relation discovery — Find new relations between similar entities via FAISS + LLM
6. Transitive relations — Infer relations via transitive rules (A→B, B→C → A→C)
7. Dead pruning — Archive low-score, rarely-mentioned entities with no relations
8. Summary generation — Generate summaries for entities without one
9. Rescore — Recalculate all ACT-R scores
10. Rebuild — Rebuild context and FAISS index
```

**Step 3: Commit**

```bash
git add prompts/dream_plan.md
git commit -m "fix: correct dream_plan.md step numbering (add step 6 transitive)"
```

---

### Task 8: Fix call_fact_consolidation LLM config

**Files:**
- Modify: `src/core/llm.py:349-365`

**Step 1: Read current code**

Read `src/core/llm.py:349-365` to see the exact config usage.

**Step 2: Fix config path**

Change `config.llm_context` to `config.llm_dream_effective` (or `config.llm_consolidation` if it exists) in `call_fact_consolidation()`.

Look for the step_config assignment line and change it.

**Step 3: Run existing tests**

Run: `uv run pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add src/core/llm.py
git commit -m "fix: use correct LLM config for fact consolidation"
```

---

### Task 9: Fix prompt_overhead constant

**Files:**
- Modify: `src/pipeline/extractor.py:201`

**Step 1: Read current code**

Read `src/pipeline/extractor.py:195-210` to see the constant and its usage.

**Step 2: Fix the constant**

Change `prompt_overhead = 500` to `prompt_overhead = 1500` (or compute dynamically):

```python
# Dynamic computation from actual prompt
prompt_text = load_prompt("extract_facts", config)
prompt_overhead = estimate_tokens(prompt_text) + 200  # 200 token margin
```

**Step 3: Run existing tests**

Run: `uv run pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add src/pipeline/extractor.py
git commit -m "fix: use dynamic prompt overhead in extractor split threshold"
```

---

### Task 10: Fix vector normalization for all embedding providers

**Files:**
- Modify: `src/pipeline/indexer.py:25-69` (_get_embedding_fn)
- Test: `tests/test_indexer.py`

**Step 1: Write the failing test**

```python
# tests/test_indexer.py

def test_embedding_normalization():
    """All embedding providers should return L2-normalized vectors."""
    import numpy as np

    # Mock an unnormalized embedding
    raw_vec = [3.0, 4.0, 0.0]  # norm = 5.0
    vec = np.array(raw_vec, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)

    np.testing.assert_almost_equal(np.linalg.norm(vec), 1.0, decimal=5)
    np.testing.assert_almost_equal(vec[0], 0.6, decimal=5)
    np.testing.assert_almost_equal(vec[1], 0.8, decimal=5)
```

**Step 2: Write minimal implementation**

In `src/pipeline/indexer.py:_get_embedding_fn()`, add normalization wrapper around all providers:

```python
import numpy as np

def _normalize_vec(vec):
    """L2-normalize a vector for cosine similarity with IndexFlatIP."""
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()
```

Wrap the return of each embedding provider's function to call `_normalize_vec()` on the result.

**Step 3: Run tests**

Run: `uv run pytest tests/test_indexer.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/pipeline/indexer.py tests/test_indexer.py
git commit -m "fix: normalize embeddings for all providers (FAISS IndexFlatIP)"
```

---

### Task 11: Fix search_rag scoring performance

**Files:**
- Modify: `src/mcp/server.py:163-183`

**Step 1: Read current code**

Read `src/mcp/server.py:155-190` to see the full search_rag re-ranking and score bump logic.

**Step 2: Replace full graph recalculation with targeted update**

Replace `recalculate_all_scores(graph, config, today)` with per-entity score update:

```python
# Instead of recalculating ALL scores, only bump retrieved entities
from src.memory.mentions import add_mention

for result in results:
    entity = graph.entities.get(result.entity_id)
    if entity:
        add_mention(entity, today)
        # Score will be recalculated on next full pipeline run
        # For now, just use the existing score for re-ranking
```

Remove the `recalculate_all_scores()` call from search_rag. Scores are recalculated during `memory run` or `memory dream`.

**Step 3: Run existing tests**

Run: `uv run pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add src/mcp/server.py
git commit -m "perf: remove full graph recalculation from search_rag"
```

---

### Task 12: Fix tags lost during entity creation

**Files:**
- Modify: `src/pipeline/enricher.py:222-225`

**Step 1: Read current code**

Read `src/pipeline/enricher.py:187-247` (_create_new_entity) to see how tags are handled.

**Step 2: Fix tag passthrough**

Ensure tags from observations are collected and passed to `create_entity()`:

```python
# Collect tags from all observations
all_tags = set()
for obs in observations:
    if obs.get("tags"):
        all_tags.update(obs["tags"])

# Pass tags to entity frontmatter
frontmatter.tags = sorted(all_tags)
```

**Step 3: Run existing tests**

Run: `uv run pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add src/pipeline/enricher.py
git commit -m "fix: preserve observation tags during entity creation"
```

---

### Task 13: Fix recover_stale_jobs never called

**Files:**
- Modify: `src/pipeline/orchestrator.py:245-250` (start of run_pipeline)

**Step 1: Read current code**

Read `src/pipeline/orchestrator.py:245-260` and find `recover_stale_jobs` definition.

**Step 2: Add call at pipeline start**

At the beginning of `run_pipeline()`, add:

```python
from src.pipeline.ingest_state import recover_stale_jobs
recover_stale_jobs(config)
```

**Step 3: Run existing tests**

Run: `uv run pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add src/pipeline/orchestrator.py
git commit -m "fix: call recover_stale_jobs at pipeline start"
```

---

### Task 14: Phase 1 integration test + full test run

**Step 1: Run ALL tests**

Run: `uv run pytest tests/ -v`
Expected: All PASS

**Step 2: Test the full pipeline manually**

Run: `uv run memory validate` to check graph consistency.

**Step 3: Commit phase 1 completion marker**

```bash
git commit --allow-empty -m "milestone: Phase 1 complete — 10 critical bug fixes"
```

---

## Phase 2 — MCP CRUD & Correction Interactive

### Task 15: `delete_fact` MCP tool

**Files:**
- Modify: `src/mcp/server.py`
- Modify: `src/memory/store.py` (use mark_observation_superseded)
- Test: `tests/test_mcp_tools.py`

**Step 1: Write the failing test**

```python
# tests/test_mcp_tools.py

def test_delete_fact():
    """delete_fact MCP tool should mark a fact as superseded."""
    import tempfile
    from pathlib import Path
    from src.memory.store import read_entity

    # Create entity with facts
    content = """---
title: Alice
type: person
retention: long_term
score: 0.5
importance: 0.5
frequency: 3
last_mentioned: "2026-03-10"
created: "2026-01-01"
aliases: []
tags: []
mention_dates: []
monthly_buckets: {}
---
## Facts
- [fact] Likes coffee
- [fact] Lives in Paris
- [diagnosis] Has back pain [-]

## Relations

## History
- 2026-01-01: Created
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        entity_path = memory_path / "close_ones" / "alice.md"
        entity_path.parent.mkdir(parents=True)
        entity_path.write_text(content)

        # Setup minimal graph
        from src.core.models import GraphData, GraphEntity
        import json
        graph = GraphData(
            generated="2026-03-10",
            entities={"alice": GraphEntity(file="close_ones/alice.md", type="person", title="Alice", score=0.5)},
            relations=[],
        )
        (memory_path / "_graph.json").write_text(graph.model_dump_json(indent=2, by_alias=True))

        # Call delete_fact logic
        from src.mcp.server import _delete_fact_impl
        result = _delete_fact_impl(memory_path, graph, "Alice", "Lives in Paris")

        assert result["deleted"] is True
        # Verify fact is gone or superseded
        text = entity_path.read_text()
        assert "Lives in Paris" not in text or "[superseded]" in text
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp_tools.py::test_delete_fact -v`
Expected: FAIL (function not defined)

**Step 3: Write implementation**

In `src/mcp/server.py`, add implementation function and MCP tool:

```python
def _resolve_entity_by_name(name: str, graph: GraphData) -> str | None:
    """Resolve entity name to entity_id via exact slug, title, or alias match."""
    from src.core.utils import slugify
    slug = slugify(name)
    if slug in graph.entities:
        return slug
    # Title match
    for eid, e in graph.entities.items():
        if e.title.lower() == name.lower():
            return eid
    # Alias match
    for eid, e in graph.entities.items():
        if any(a.lower() == name.lower() for a in (e.aliases or [])):
            return eid
    return None


def _find_matching_fact(facts_text: str, target_content: str) -> str | None:
    """Find best matching fact line using substring match."""
    target_lower = target_content.lower()
    for line in facts_text.strip().split("\n"):
        if target_lower in line.lower():
            return line.strip()
    return None


def _delete_fact_impl(memory_path: Path, graph: GraphData, entity_name: str, fact_content: str) -> dict:
    """Delete a fact from an entity. Returns {deleted: bool, entity_id, fact}."""
    entity_id = _resolve_entity_by_name(entity_name, graph)
    if not entity_id:
        return {"deleted": False, "error": f"Entity '{entity_name}' not found"}

    entity = graph.entities[entity_id]
    entity_path = memory_path / entity.file
    if not entity_path.exists():
        return {"deleted": False, "error": f"Entity file not found: {entity.file}"}

    from src.memory.store import read_entity, write_entity
    frontmatter, sections = read_entity(entity_path)
    facts_text = sections.get("Facts", "")
    matching_line = _find_matching_fact(facts_text, fact_content)

    if not matching_line:
        return {"deleted": False, "error": f"Fact not found matching: {fact_content}"}

    # Remove the line
    lines = facts_text.strip().split("\n")
    new_lines = [l for l in lines if l.strip() != matching_line]
    sections["Facts"] = "\n".join(new_lines)

    # Add history entry
    from datetime import date
    history = sections.get("History", "")
    sections["History"] = history.rstrip() + f"\n- {date.today().isoformat()}: Fact deleted: {matching_line[:60]}..."

    write_entity(entity_path, frontmatter, sections)
    return {"deleted": True, "entity_id": entity_id, "fact": matching_line}


# MCP tool registration
@mcp.tool()
async def delete_fact(entity_name: str, fact_content: str) -> str:
    """Delete a specific fact from an entity's memory.

    Args:
        entity_name: Name of the entity (title, slug, or alias)
        fact_content: Content of the fact to delete (partial match supported)
    """
    graph = load_graph(memory_path)
    result = _delete_fact_impl(memory_path, graph, entity_name, fact_content)
    if result["deleted"]:
        save_graph(graph, memory_path)
        return json.dumps({"status": "deleted", "entity": result["entity_id"], "fact": result["fact"]})
    return json.dumps({"status": "not_found", "error": result.get("error", "")})
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_mcp_tools.py::test_delete_fact -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mcp/server.py tests/test_mcp_tools.py
git commit -m "feat: add delete_fact MCP tool"
```

---

### Task 16: `delete_relation` MCP tool

**Files:**
- Modify: `src/mcp/server.py`
- Test: `tests/test_mcp_tools.py`

**Step 1: Write the failing test**

```python
def test_delete_relation():
    """delete_relation MCP tool should remove a relation from graph and MD."""
    import tempfile
    from pathlib import Path
    from src.core.models import GraphData, GraphEntity, GraphRelation

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        (memory_path / "close_ones").mkdir(parents=True)

        # Create entity with relation
        (memory_path / "close_ones/alice.md").write_text(
            "---\ntitle: Alice\ntype: person\n---\n## Facts\n\n## Relations\n- parent_of [[Bob]]\n\n## History\n"
        )
        (memory_path / "close_ones/bob.md").write_text(
            "---\ntitle: Bob\ntype: person\n---\n## Facts\n\n## Relations\n\n## History\n"
        )

        graph = GraphData(
            generated="2026-03-10",
            entities={
                "alice": GraphEntity(file="close_ones/alice.md", type="person", title="Alice", score=0.5),
                "bob": GraphEntity(file="close_ones/bob.md", type="person", title="Bob", score=0.5),
            },
            relations=[GraphRelation(from_entity="alice", to_entity="bob", type="parent_of")],
        )
        (memory_path / "_graph.json").write_text(graph.model_dump_json(indent=2, by_alias=True))

        from src.mcp.server import _delete_relation_impl
        result = _delete_relation_impl(memory_path, graph, "Alice", "Bob", "parent_of")

        assert result["deleted"] is True
        assert len(graph.relations) == 0
        assert "parent_of [[Bob]]" not in (memory_path / "close_ones/alice.md").read_text()
```

**Step 2: Run test, verify fail**

**Step 3: Implement**

```python
def _delete_relation_impl(memory_path: Path, graph: GraphData, from_name: str, to_name: str, relation_type: str) -> dict:
    """Delete a relation between two entities."""
    from_id = _resolve_entity_by_name(from_name, graph)
    to_id = _resolve_entity_by_name(to_name, graph)

    if not from_id:
        return {"deleted": False, "error": f"Entity '{from_name}' not found"}
    if not to_id:
        return {"deleted": False, "error": f"Entity '{to_name}' not found"}

    from src.memory.graph import remove_relation
    removed = remove_relation(graph, from_id, to_id, relation_type)

    if removed:
        # Clean MD file
        from src.memory.store import remove_relation_line
        from_entity = graph.entities[from_id]
        from_path = memory_path / from_entity.file
        to_title = graph.entities[to_id].title
        if from_path.exists():
            remove_relation_line(from_path, relation_type, to_title)
        return {"deleted": True, "from": from_id, "to": to_id, "type": relation_type}

    return {"deleted": False, "error": f"Relation {from_name} → {relation_type} → {to_name} not found"}


@mcp.tool()
async def delete_relation(from_entity: str, to_entity: str, relation_type: str) -> str:
    """Delete a relation between two entities.

    Args:
        from_entity: Source entity name
        to_entity: Target entity name
        relation_type: Type of relation (affects, improves, parent_of, etc.)
    """
    graph = load_graph(memory_path)
    result = _delete_relation_impl(memory_path, graph, from_entity, to_entity, relation_type)
    if result["deleted"]:
        save_graph(graph, memory_path)
    return json.dumps(result)
```

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```bash
git add src/mcp/server.py tests/test_mcp_tools.py
git commit -m "feat: add delete_relation MCP tool"
```

---

### Task 17: `modify_fact` MCP tool

**Files:**
- Modify: `src/mcp/server.py`
- Test: `tests/test_mcp_tools.py`

**Step 1: Write the failing test**

```python
def test_modify_fact():
    """modify_fact should replace a fact's content while preserving metadata."""
    import tempfile
    from pathlib import Path
    from src.core.models import GraphData, GraphEntity

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        (memory_path / "close_ones").mkdir(parents=True)
        (memory_path / "close_ones/alice.md").write_text(
            "---\ntitle: Alice\ntype: person\n---\n## Facts\n- [fact] (2026-03) Likes coffee\n\n## Relations\n\n## History\n"
        )
        graph = GraphData(
            generated="2026-03-10",
            entities={"alice": GraphEntity(file="close_ones/alice.md", type="person", title="Alice", score=0.5)},
            relations=[],
        )

        from src.mcp.server import _modify_fact_impl
        result = _modify_fact_impl(memory_path, graph, "Alice", "Likes coffee", "Prefers tea")

        assert result["modified"] is True
        text = (memory_path / "close_ones/alice.md").read_text()
        assert "Prefers tea" in text
        assert "Likes coffee" not in text
        assert "[fact]" in text  # Category preserved
        assert "(2026-03)" in text  # Date preserved
```

**Step 2-5: Same pattern as above — fail, implement, pass, commit**

Implementation: parse the matching line, extract `[category] (date)` prefix, replace content, write back.

```bash
git commit -m "feat: add modify_fact MCP tool"
```

---

### Task 18: `correct_entity` MCP tool

**Files:**
- Modify: `src/mcp/server.py`
- Test: `tests/test_mcp_tools.py`

Supports corrections to: title, type, aliases, retention. If type changes, moves file to correct folder.

Follow same TDD pattern. Key implementation:

```python
def _correct_entity_impl(memory_path, graph, entity_name, corrections: dict) -> dict:
    """Apply corrections to entity metadata."""
    entity_id = _resolve_entity_by_name(entity_name, graph)
    if not entity_id:
        return {"corrected": False, "error": f"Entity '{entity_name}' not found"}

    entity = graph.entities[entity_id]
    entity_path = memory_path / entity.file
    frontmatter, sections = read_entity(entity_path)

    changes = {}
    for field, value in corrections.items():
        if field == "title":
            frontmatter.title = value
            entity.title = value
            changes["title"] = value
        elif field == "type":
            old_type = frontmatter.type
            frontmatter.type = value
            entity.type = value
            # Move file to correct folder
            new_folder = config.get_folder_for_type(value)
            new_path = memory_path / new_folder / entity_path.name
            new_path.parent.mkdir(parents=True, exist_ok=True)
            entity_path.rename(new_path)  # Move first
            entity_path = new_path
            entity.file = f"{new_folder}/{entity_path.name}"
            changes["type"] = f"{old_type} → {value}"
        elif field == "aliases":
            frontmatter.aliases = value if isinstance(value, list) else [value]
            entity.aliases = frontmatter.aliases
            changes["aliases"] = frontmatter.aliases
        elif field == "retention":
            frontmatter.retention = value
            entity.retention = value
            changes["retention"] = value

    write_entity(entity_path, frontmatter, sections)
    return {"corrected": True, "entity_id": entity_id, "changes": changes}
```

```bash
git commit -m "feat: add correct_entity MCP tool"
```

---

### Task 19: Phase 2 integration test

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All PASS

**Step 2: Commit milestone**

```bash
git commit --allow-empty -m "milestone: Phase 2 complete — 4 MCP CRUD tools"
```

---

## Phase 3 — Recherche Hybride RRF

### Task 20: SQLite FTS5 keyword index

**Files:**
- Create: `src/pipeline/keyword_index.py`
- Test: `tests/test_keyword_index.py`

**Step 1: Write the failing test**

```python
# tests/test_keyword_index.py

import tempfile
from pathlib import Path

def test_build_and_search_keyword_index():
    """Build FTS5 index from entity MDs and search by keyword."""
    from src.pipeline.keyword_index import build_keyword_index, search_keyword

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        (memory_path / "close_ones").mkdir()

        # Create entities
        (memory_path / "close_ones/dr-martin.md").write_text(
            "---\ntitle: Dr. Martin\ntype: person\n---\n## Facts\n- [fact] Prescribes ACT-R therapy\n- [fact] Works at CHU Mondor\n"
        )
        (memory_path / "close_ones/alice.md").write_text(
            "---\ntitle: Alice\ntype: person\n---\n## Facts\n- [fact] Likes swimming\n"
        )

        db_path = memory_path / "_memory_fts.db"
        build_keyword_index(memory_path, db_path)

        # Exact keyword search should find Dr. Martin
        results = search_keyword("Dr. Martin", db_path, top_k=5)
        assert len(results) > 0
        assert results[0].entity_id == "dr-martin"

        # Keyword "ACT-R" should find Dr. Martin
        results2 = search_keyword("ACT-R", db_path, top_k=5)
        assert len(results2) > 0
        assert results2[0].entity_id == "dr-martin"

        # Keyword "swimming" should find Alice
        results3 = search_keyword("swimming", db_path, top_k=5)
        assert len(results3) > 0
        assert results3[0].entity_id == "alice"
```

**Step 2: Run test, verify fail**

**Step 3: Write implementation**

```python
# src/pipeline/keyword_index.py

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.core.utils import parse_frontmatter


@dataclass
class KeywordResult:
    entity_id: str
    chunk_idx: int
    bm25_score: float


def build_keyword_index(memory_path: Path, db_path: Path) -> int:
    """Build SQLite FTS5 index from entity MD files.

    Returns number of chunks indexed.
    """
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE VIRTUAL TABLE memory_fts USING fts5(
            entity_id,
            chunk_idx,
            content,
            tokenize='unicode61 remove_diacritics 2'
        )
    """)

    count = 0
    # Scan all MD files (same folders as FAISS indexer)
    for md_file in sorted(memory_path.rglob("*.md")):
        if md_file.name.startswith("_"):
            continue
        rel = md_file.relative_to(memory_path)
        if str(rel).startswith("chats/") or str(rel).startswith("_"):
            continue

        entity_id = md_file.stem
        text = md_file.read_text(encoding="utf-8")

        # Extract content (title + facts)
        try:
            frontmatter, sections = parse_frontmatter(text)
        except Exception:
            continue

        title = frontmatter.get("title", entity_id) if isinstance(frontmatter, dict) else entity_id
        facts = sections.get("Facts", "") if isinstance(sections, dict) else ""
        full_text = f"{title}\n{facts}"

        # Chunk same as FAISS (400 tokens, 80 overlap) — simplified word-based
        words = full_text.split()
        chunk_size = 300  # words (approx 400 tokens)
        overlap = 60
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap
        if not chunks:
            chunks = [full_text]

        for idx, chunk in enumerate(chunks):
            conn.execute(
                "INSERT INTO memory_fts (entity_id, chunk_idx, content) VALUES (?, ?, ?)",
                (entity_id, idx, chunk),
            )
            count += 1

    conn.commit()
    conn.close()
    return count


def search_keyword(query: str, db_path: Path, top_k: int = 10) -> list[KeywordResult]:
    """Search FTS5 index with BM25 ranking."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT entity_id, chunk_idx, bm25(memory_fts) as score
            FROM memory_fts
            WHERE memory_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (query, top_k),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()

    return [
        KeywordResult(entity_id=row[0], chunk_idx=row[1], bm25_score=-row[2])
        for row in rows
    ]
```

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add src/pipeline/keyword_index.py tests/test_keyword_index.py
git commit -m "feat: add SQLite FTS5 keyword index for hybrid search"
```

---

### Task 21: Integrate keyword index into incremental_update

**Files:**
- Modify: `src/pipeline/indexer.py:185-210`
- Modify: `src/pipeline/keyword_index.py` (import)

**Step 1: Add keyword index build alongside FAISS rebuild**

In `incremental_update()`, after FAISS index is built, add:

```python
from src.pipeline.keyword_index import build_keyword_index

fts_db_path = memory_path / "_memory_fts.db"
build_keyword_index(memory_path, fts_db_path)
```

**Step 2: Run existing tests**

**Step 3: Commit**

```bash
git add src/pipeline/indexer.py
git commit -m "feat: build keyword index alongside FAISS during incremental_update"
```

---

### Task 22: RRF fusion in search_rag

**Files:**
- Modify: `src/mcp/server.py:80-189`
- Modify: `src/core/config.py` (add search config)
- Test: `tests/test_rrf.py`

**Step 1: Write the failing test**

```python
# tests/test_rrf.py

def test_rrf_fusion():
    """RRF should combine semantic, keyword, and ACT-R rankings."""
    from src.mcp.server import _rrf_fusion
    from src.core.models import SearchResult, GraphData, GraphEntity

    faiss_results = [
        SearchResult(entity_id="a", file="a.md", chunk="", score=0.9),
        SearchResult(entity_id="b", file="b.md", chunk="", score=0.7),
    ]
    keyword_results = [
        {"entity_id": "b", "bm25_score": 5.0},  # b ranks first in keyword
        {"entity_id": "c", "bm25_score": 3.0},  # c only in keyword
    ]
    graph = GraphData(
        generated="2026-03-10",
        entities={
            "a": GraphEntity(file="a.md", type="person", title="A", score=0.8),
            "b": GraphEntity(file="b.md", type="person", title="B", score=0.6),
            "c": GraphEntity(file="c.md", type="person", title="C", score=0.3),
        },
        relations=[],
    )

    ranked = _rrf_fusion(faiss_results, keyword_results, graph)
    entity_ids = [r[0] for r in ranked]

    # b should rank highly (present in both semantic AND keyword)
    assert "b" in entity_ids[:2]
    # All 3 entities should be present
    assert set(entity_ids) == {"a", "b", "c"}
```

**Step 2: Implement RRF fusion**

```python
def _rrf_fusion(faiss_results, keyword_results, graph, k=60,
                w_sem=0.5, w_kw=0.3, w_actr=0.2):
    """Reciprocal Rank Fusion combining 3 signals."""
    sem_ranks = {r.entity_id: i + 1 for i, r in enumerate(faiss_results)}
    kw_ranks = {r["entity_id"] if isinstance(r, dict) else r.entity_id: i + 1
                for i, r in enumerate(keyword_results)}

    all_ids = set(sem_ranks) | set(kw_ranks)

    # ACT-R ranking
    actr_scores = {eid: graph.entities[eid].score for eid in all_ids if eid in graph.entities}
    sorted_actr = sorted(actr_scores.items(), key=lambda x: x[1], reverse=True)
    actr_ranks = {eid: i + 1 for i, (eid, _) in enumerate(sorted_actr)}

    scored = []
    for eid in all_ids:
        sr = sem_ranks.get(eid, len(faiss_results) + 10)
        kr = kw_ranks.get(eid, len(keyword_results) + 10)
        ar = actr_ranks.get(eid, len(all_ids) + 10)

        score = w_sem / (k + sr) + w_kw / (k + kr) + w_actr / (k + ar)
        scored.append((eid, score))

    return sorted(scored, key=lambda x: x[1], reverse=True)
```

**Step 3: Update search_rag to use RRF**

In `search_rag()`, add keyword search and call `_rrf_fusion()`:

```python
# After FAISS search
from src.pipeline.keyword_index import search_keyword
fts_db_path = memory_path / "_memory_fts.db"
keyword_results = search_keyword(query, fts_db_path, top_k=config.faiss.top_k)

if keyword_results:
    ranked = _rrf_fusion(faiss_results, keyword_results, graph)
    # Reorder results by RRF score
    ...
else:
    # Fallback to current linear re-ranking
    ...
```

**Step 4: Add search config to config.py**

```python
@dataclass
class SearchConfig:
    hybrid_enabled: bool = True
    rrf_k: int = 60
    weight_semantic: float = 0.5
    weight_keyword: float = 0.3
    weight_actr: float = 0.2
```

**Step 5: Run tests, commit**

```bash
git add src/mcp/server.py src/core/config.py tests/test_rrf.py
git commit -m "feat: implement RRF hybrid search (semantic + keyword + ACT-R)"
```

---

### Task 23: Phase 3 integration + milestone

Run all tests, commit milestone.

```bash
git commit --allow-empty -m "milestone: Phase 3 complete — hybrid RRF search"
```

---

## Phase 4 — Prompts & Petits LLMs

### Task 24: Dream coordinator — deterministic rules

**Files:**
- Modify: `src/pipeline/dream.py:44-78` (replace LLM coordinator)
- Modify: `src/core/llm.py:436-453` (keep call_dream_plan but stop calling it)
- Test: `tests/test_dream.py`

**Step 1: Write the failing test**

```python
# tests/test_dream.py

def test_decide_dream_steps_with_candidates():
    """Deterministic step selection based on candidate counts."""
    from src.pipeline.dream import decide_dream_steps

    stats = {
        "unextracted_docs": 0,
        "consolidation_candidates": 5,
        "merge_candidates": 0,
        "relation_candidates": 10,
        "transitive_candidates": 0,
        "prune_candidates": 2,
        "summary_candidates": 0,
    }
    steps = decide_dream_steps(stats)
    assert 1 in steps  # Load always
    assert 3 in steps  # consolidation_candidates >= 3
    assert 5 in steps  # relation_candidates >= 5
    assert 7 in steps  # prune_candidates >= 1
    assert 9 in steps  # Rescore (because other steps run)
    assert 10 in steps  # Rebuild
    assert 2 not in steps  # No unextracted docs
    assert 4 not in steps  # No merge candidates
    assert 6 not in steps  # No transitive candidates


def test_decide_dream_steps_nothing_to_do():
    """When no candidates, only load step runs."""
    from src.pipeline.dream import decide_dream_steps

    stats = {
        "unextracted_docs": 0,
        "consolidation_candidates": 0,
        "merge_candidates": 0,
        "relation_candidates": 0,
        "transitive_candidates": 0,
        "prune_candidates": 0,
        "summary_candidates": 0,
    }
    steps = decide_dream_steps(stats)
    assert steps == [1]
```

**Step 2: Run test, verify fail**

**Step 3: Implement**

```python
def decide_dream_steps(stats: dict) -> list[int]:
    """Deterministic dream step selection based on candidate counts.

    Replaces LLM coordinator (call_dream_plan) for 100% reproducibility.
    """
    steps = [1]  # Load always
    if stats.get("unextracted_docs", 0) > 0:
        steps.append(2)
    if stats.get("consolidation_candidates", 0) >= 3:
        steps.append(3)
    if stats.get("merge_candidates", 0) >= 2:
        steps.append(4)
    if stats.get("relation_candidates", 0) >= 5:
        steps.append(5)
    if stats.get("transitive_candidates", 0) >= 3:
        steps.append(6)
    if stats.get("prune_candidates", 0) >= 1:
        steps.append(7)
    if stats.get("summary_candidates", 0) >= 3:
        steps.append(8)
    if any(s in steps for s in [2, 3, 4, 5, 6, 7, 8]):
        steps.extend([9, 10])
    return sorted(set(steps))
```

Replace the `call_dream_plan()` call in `run_dream()` with `decide_dream_steps()`.

**Step 4: Run tests, commit**

```bash
git add src/pipeline/dream.py tests/test_dream.py
git commit -m "feat: replace LLM dream coordinator with deterministic rules"
```

---

### Task 25: Dream validate — deterministic checks

**Files:**
- Modify: `src/pipeline/dream.py:257-271`
- Test: `tests/test_dream.py`

**Step 1: Write the failing test**

```python
def test_validate_dream_step_consolidation():
    """Consolidation should not increase fact count."""
    from src.pipeline.dream import validate_dream_step

    before = {"total_facts": 100, "total_entities": 50}
    after = {"total_facts": 80, "total_entities": 50}  # Reduced — good
    ok, issues = validate_dream_step(3, before, after)
    assert ok is True

    after_bad = {"total_facts": 120, "total_entities": 50}  # Increased — bad
    ok2, issues2 = validate_dream_step(3, before, after_bad)
    assert ok2 is False
    assert len(issues2) > 0
```

**Step 2-5: Implement, test, commit**

```python
def validate_dream_step(step: int, before: dict, after: dict) -> tuple[bool, list[str]]:
    """Deterministic validation of dream step results."""
    issues = []
    if step == 3:  # Consolidation
        if after.get("total_facts", 0) > before.get("total_facts", 0):
            issues.append("Consolidation increased fact count")
    elif step == 4:  # Merge
        if after.get("total_entities", 0) > before.get("total_entities", 0):
            issues.append("Merge increased entity count")
    elif step == 5:  # Relation discovery
        new_rels = after.get("total_relations", 0) - before.get("total_relations", 0)
        if new_rels > 50:
            issues.append(f"Relation discovery added {new_rels} relations (suspiciously high)")
    return len(issues) == 0, issues
```

```bash
git commit -m "feat: replace LLM dream validation with deterministic checks"
```

---

### Task 26: Simplify extract_facts.md

**Files:**
- Modify: `prompts/extract_facts.md`
- Modify: `src/pipeline/extractor.py` (inject {today} variable)

**Step 1: Read current prompt**

**Step 2: Apply simplifications**

1. Add `{today}` injection for date anchor
2. Add category descriptions table (not just names)
3. Add concrete examples per category
4. Remove supersedes from relation JSON (keep only for observations)
5. Simplify observation JSON to 5 fields

**Step 3: Update extractor.py to inject today**

In `extract_from_chat()`, add `today=date.today().isoformat()` to the `load_prompt()` call.

**Step 4: Run existing extraction tests**

**Step 5: Commit**

```bash
git add prompts/extract_facts.md src/pipeline/extractor.py
git commit -m "feat: simplify extract_facts prompt for small LLMs"
```

---

### Task 27: Simplify consolidate_facts.md

**Files:**
- Modify: `prompts/consolidate_facts.md`
- Modify: `src/memory/store.py:184-253` (add deterministic dedup phase)

**Step 1: Add deterministic Levenshtein pre-dedup**

Before calling LLM consolidation, add a deterministic dedup pass:

```python
from difflib import SequenceMatcher

def _dedup_facts_deterministic(facts: list[str], threshold: float = 0.85) -> list[str]:
    """Remove near-duplicate facts using Levenshtein similarity."""
    kept = []
    for fact in facts:
        is_dup = False
        for existing in kept:
            ratio = SequenceMatcher(None, fact.lower(), existing.lower()).ratio()
            if ratio >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(fact)
    return kept
```

**Step 2: Simplify prompt**

Reduce from 8+ instructions to 3:
1. Merge semantically identical facts into one
2. Keep max {max_facts} facts, prioritize most recent and most important
3. Preserve category, date, and valence exactly

**Step 3: Run tests, commit**

```bash
git add prompts/consolidate_facts.md src/memory/store.py
git commit -m "feat: add deterministic fact dedup + simplify consolidation prompt"
```

---

### Task 28: Simplify discover_relations.md

**Files:**
- Modify: `prompts/discover_relations.md`
- Modify: `src/pipeline/dream.py:631-653` (_build_dossier — trim to 3 facts)

**Step 1: Trim dossier to title + type + 3 facts + tags**

In `_build_dossier()`, limit facts to top 3 by importance (or first 3):

```python
facts = facts[:3]  # Only top 3 facts instead of all
```

**Step 2: Simplify prompt** — reduce to ~100 tokens of instructions

**Step 3: Commit**

```bash
git add prompts/discover_relations.md src/pipeline/dream.py
git commit -m "feat: reduce relation discovery context for small LLMs"
```

---

### Task 29: Enrich arbitrate_entity.md + structured summary output

**Files:**
- Modify: `prompts/arbitrate_entity.md`
- Modify: `src/core/llm.py:263-283` (enrich candidates with facts)
- Modify: `prompts/summarize_entity.md`
- Modify: `src/core/models.py` (add EntitySummary model)

**Step 1: Add EntitySummary model**

```python
class EntitySummary(BaseModel):
    summary: str = Field(max_length=150)
```

**Step 2: Update call_entity_summary to use structured output**

**Step 3: Enrich arbitration candidates with summary/facts**

**Step 4: Use [EXISTING_N] format in arbitration prompt**

**Step 5: Run tests, commit**

```bash
git add prompts/arbitrate_entity.md prompts/summarize_entity.md src/core/llm.py src/core/models.py
git commit -m "feat: enrich arbitration context + structured summary output"
```

---

### Task 30: Phase 4 milestone

```bash
git commit --allow-empty -m "milestone: Phase 4 complete — prompt simplifications for small LLMs"
```

---

## Phase 5 — Nice-to-Have

### Task 31: Centralized action log

**Files:**
- Create: `src/core/action_log.py`
- Modify: `src/pipeline/enricher.py` (add hooks)
- Modify: `src/pipeline/dream.py` (add hooks)
- Modify: `src/cli.py` (add `actions` command)
- Test: `tests/test_action_log.py`

**Step 1: Write the failing test**

```python
# tests/test_action_log.py

import tempfile, json
from pathlib import Path

def test_log_action():
    """log_action should append a JSON line to _actions.jsonl."""
    from src.core.action_log import log_action

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        log_action(memory_path, "create", entity_id="alice", details={"type": "person"})
        log_action(memory_path, "update", entity_id="alice", details={"facts_added": 3})

        log_path = memory_path / "_actions.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        assert entry1["action"] == "create"
        assert entry1["entity_id"] == "alice"
        assert "timestamp" in entry1


def test_read_actions():
    """read_actions should return filtered action entries."""
    from src.core.action_log import log_action, read_actions

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_path = Path(tmpdir)
        log_action(memory_path, "create", entity_id="alice")
        log_action(memory_path, "update", entity_id="bob")
        log_action(memory_path, "delete", entity_id="alice")

        all_actions = read_actions(memory_path)
        assert len(all_actions) == 3

        alice_actions = read_actions(memory_path, entity_id="alice")
        assert len(alice_actions) == 2

        delete_actions = read_actions(memory_path, action="delete")
        assert len(delete_actions) == 1
```

**Step 2: Implement**

```python
# src/core/action_log.py

import json
from datetime import datetime
from pathlib import Path


def log_action(memory_path: Path, action: str, entity_id: str = "",
               details: dict = None, source: str = "pipeline"):
    """Append action to centralized log (_actions.jsonl)."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "entity_id": entity_id,
        "source": source,
        "details": details or {},
    }
    log_path = memory_path / "_actions.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_actions(memory_path: Path, entity_id: str = None,
                 action: str = None, last_n: int = 0) -> list[dict]:
    """Read and filter action log entries."""
    log_path = memory_path / "_actions.jsonl"
    if not log_path.exists():
        return []

    entries = []
    for line in log_path.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        entry = json.loads(line)
        if entity_id and entry.get("entity_id") != entity_id:
            continue
        if action and entry.get("action") != action:
            continue
        entries.append(entry)

    if last_n > 0:
        entries = entries[-last_n:]
    return entries
```

**Step 3: Add hooks in enricher.py and dream.py** (just add `log_action()` calls after key operations)

**Step 4: Add CLI command**

```python
@cli.command()
@click.option("--last", default=20, help="Show last N actions")
@click.option("--entity", default=None, help="Filter by entity name")
@click.option("--action", "action_type", default=None, help="Filter by action type")
def actions(last, entity, action_type):
    """Show centralized action history."""
    from src.core.action_log import read_actions
    entries = read_actions(memory_path, entity_id=entity, action=action_type, last_n=last)
    for e in entries:
        click.echo(f"[{e['timestamp'][:19]}] {e['action']:15s} {e.get('entity_id', ''):20s} {json.dumps(e.get('details', {}))}")
```

**Step 5: Run tests, commit**

```bash
git add src/core/action_log.py tests/test_action_log.py src/pipeline/enricher.py src/pipeline/dream.py src/cli.py
git commit -m "feat: add centralized action log (_actions.jsonl) with CLI command"
```

---

### Task 32: ACT-R Insights command

**Files:**
- Modify: `src/cli.py` (add `insights` command)
- Create: `src/memory/insights.py`
- Test: `tests/test_insights.py`

**Step 1: Write the failing test**

```python
# tests/test_insights.py

def test_compute_insights():
    """compute_insights should analyze graph and return structured insights."""
    from src.memory.insights import compute_insights
    from src.core.models import GraphData, GraphEntity, GraphRelation

    graph = GraphData(
        generated="2026-03-10",
        entities={
            "alice": GraphEntity(file="a.md", type="person", title="Alice", score=0.8, frequency=10,
                                 negative_valence_ratio=0.0),
            "bob": GraphEntity(file="b.md", type="person", title="Bob", score=0.04, frequency=1,
                               negative_valence_ratio=0.0),  # Below retrieval threshold
            "health": GraphEntity(file="h.md", type="health", title="Back Pain", score=0.6, frequency=5,
                                  negative_valence_ratio=0.7),  # Emotional hotspot
        },
        relations=[
            GraphRelation(from_entity="alice", to_entity="bob", type="friend_of", strength=0.15,
                          last_reinforced="2025-06-01"),  # Approaching LTD
        ],
    )

    insights = compute_insights(graph)

    assert len(insights["forgetting_curve"]) >= 1  # Bob approaching threshold
    assert len(insights["emotional_hotspots"]) >= 1  # Back Pain
    assert len(insights["weak_relations"]) >= 1  # alice→bob weak
    assert "scoring_distribution" in insights
```

**Step 2: Implement**

```python
# src/memory/insights.py

from datetime import date, timedelta
from src.core.models import GraphData


def compute_insights(graph: GraphData, today: str = None) -> dict:
    """Analyze graph for ACT-R-aware insights. Zero LLM."""
    if today is None:
        today = date.today().isoformat()
    today_date = date.fromisoformat(today)

    insights = {
        "forgetting_curve": [],
        "emotional_hotspots": [],
        "weak_relations": [],
        "network_hubs": [],
        "scoring_distribution": {"0-0.1": 0, "0.1-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-1.0": 0},
    }

    # Scoring distribution + forgetting curve + emotional hotspots
    for eid, e in graph.entities.items():
        s = e.score
        if s < 0.1:
            insights["scoring_distribution"]["0-0.1"] += 1
        elif s < 0.3:
            insights["scoring_distribution"]["0.1-0.3"] += 1
        elif s < 0.5:
            insights["scoring_distribution"]["0.3-0.5"] += 1
        elif s < 0.7:
            insights["scoring_distribution"]["0.5-0.7"] += 1
        else:
            insights["scoring_distribution"]["0.7-1.0"] += 1

        if 0.0 < s <= 0.1:
            insights["forgetting_curve"].append({"entity": eid, "title": e.title, "score": s})

        nvr = getattr(e, "negative_valence_ratio", 0.0) or 0.0
        if nvr > 0.3:
            insights["emotional_hotspots"].append({"entity": eid, "title": e.title, "ratio": nvr})

    # Weak relations (approaching LTD)
    for r in graph.relations:
        if r.strength < 0.2:
            age = 0
            if r.last_reinforced:
                try:
                    age = (today_date - date.fromisoformat(r.last_reinforced)).days
                except ValueError:
                    pass
            insights["weak_relations"].append({
                "from": r.from_entity, "to": r.to_entity, "type": r.type,
                "strength": r.strength, "days_since_reinforced": age,
            })

    # Network hubs (degree centrality)
    degree = {}
    for r in graph.relations:
        degree[r.from_entity] = degree.get(r.from_entity, 0) + 1
        degree[r.to_entity] = degree.get(r.to_entity, 0) + 1
    top_hubs = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]
    insights["network_hubs"] = [
        {"entity": eid, "title": graph.entities.get(eid, type("", (), {"title": eid})).title, "degree": d}
        for eid, d in top_hubs
    ]

    return insights
```

**Step 3: Add CLI command**

```python
@cli.command()
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def insights(fmt):
    """Show ACT-R cognitive insights about memory state."""
    from src.memory.insights import compute_insights
    from src.memory.graph import load_graph
    graph = load_graph(memory_path)
    result = compute_insights(graph)
    if fmt == "json":
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Formatted text output
        ...
```

**Step 4: Run tests, commit**

```bash
git add src/memory/insights.py tests/test_insights.py src/cli.py
git commit -m "feat: add ACT-R insights command (forgetting curve, emotional hotspots, weak relations)"
```

---

### Task 33: Dream Step 4 LLM deduplication

**Files:**
- Modify: `src/pipeline/dream.py:400-456` (_step_merge_entities)
- Create: `prompts/dedup_check.md`
- Modify: `src/core/llm.py` (add call_dedup_check)
- Modify: `src/core/models.py` (add DedupVerdict)
- Test: `tests/test_dream_dedup.py`

**Step 1: Add DedupVerdict model**

```python
class DedupVerdict(BaseModel):
    is_duplicate: bool
    confidence: float = Field(ge=0, le=1, default=0.5)
    reason: str = ""
```

**Step 2: Create dedup_check.md prompt**

```markdown
Are these two entities the same thing?

Entity A: {title_a} ({type_a})
Summary: {summary_a}

Entity B: {title_b} ({type_b})
Summary: {summary_b}

Answer as JSON: {"is_duplicate": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}
```

**Step 3: Add call_dedup_check to llm.py**

```python
def call_dedup_check(entity_a_title, entity_a_type, entity_a_summary,
                     entity_b_title, entity_b_type, entity_b_summary,
                     config) -> DedupVerdict:
    prompt = load_prompt("dedup_check", config,
                         title_a=entity_a_title, type_a=entity_a_type, summary_a=entity_a_summary or "N/A",
                         title_b=entity_b_title, type_b=entity_b_type, summary_b=entity_b_summary or "N/A")
    return _call_structured(config.llm_dream_effective, prompt, DedupVerdict)
```

**Step 4: Enhance _step_merge_entities in dream.py**

After deterministic slug/alias merges, add FAISS-based candidate expansion:

```python
# After deterministic merges, try FAISS-based dedup
if faiss_search_fn:
    for entity_id, entity in list(graph.entities.items()):
        if entity_id in already_merged:
            continue
        results = faiss_search_fn(entity.title, top_k=5)
        for result in results:
            if (result.entity_id != entity_id
                    and result.score > 0.8
                    and result.entity_id in graph.entities
                    and graph.entities[result.entity_id].type == entity.type
                    and result.entity_id not in already_merged):
                verdict = call_dedup_check(
                    entity.title, entity.type, entity.summary,
                    graph.entities[result.entity_id].title,
                    graph.entities[result.entity_id].type,
                    graph.entities[result.entity_id].summary,
                    config,
                )
                if verdict.is_duplicate and verdict.confidence >= 0.7:
                    merge_candidates.append((entity_id, result.entity_id))
```

**Step 5: Write test, run, commit**

```bash
git add src/pipeline/dream.py src/core/llm.py src/core/models.py prompts/dedup_check.md tests/test_dream_dedup.py
git commit -m "feat: add LLM-based deduplication in dream step 4"
```

---

### Task 34: Phase 5 milestone + final integration

**Step 1: Run ALL tests**

Run: `uv run pytest tests/ -v`
Expected: All PASS

**Step 2: Run validate**

Run: `uv run memory validate`

**Step 3: Commit milestone**

```bash
git commit --allow-empty -m "milestone: Phase 5 complete — action log, insights, dream LLM dedup"
```

---

### Task 35: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

Update the following sections:
- Add `remove_relation()` to graph.py function list
- Add new MCP tools (delete_fact, delete_relation, modify_fact, correct_entity)
- Add `memory insights` and `memory actions` CLI commands
- Document hybrid RRF search config
- Update dream mode description (deterministic coordinator)
- Add `src/pipeline/keyword_index.py` and `src/core/action_log.py` to source layout
- Add `_actions.jsonl` and `_memory_fts.db` to data files table

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with all new features and changes"
```

---

## Dependency Graph Summary

```
Task 1-13  (Phase 1: bugs)     → sequential, each builds on previous
Task 14    (Phase 1: verify)   → depends on 1-13
Task 15-19 (Phase 2: MCP)      → depends on Phase 1 (uses remove_relation)
Task 20-23 (Phase 3: RRF)      → independent after Phase 1
Task 24-30 (Phase 4: prompts)  → independent after Phase 1
Task 31-34 (Phase 5: extras)   → independent after Phase 1
Task 35    (CLAUDE.md)         → depends on all phases
```

**Phases 2, 3, 4, 5 can be executed in parallel after Phase 1 is complete.**
