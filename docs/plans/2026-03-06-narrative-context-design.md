# Narrative Context Pipeline — Design Document

**Date:** 2026-03-06
**Status:** Validated

## Problem

`build_deterministic_context()` generates `_context.md` as structured blocks (entity name, score, retention, tagged facts, relations). Functional but verbose:
- Tokens wasted on metadata (scores, retention, `[fact]` prefixes)
- "AI Personality" section = static list, no contextual priorities
- No signal to the receiving LLM about what matters right now

## Goal

Generate `_context.md` in natural prose, section by section, combining ACT-R graph data + RAG enrichment, via small local LLM calls (no tool calling, pure text).

## Architecture

### Two paths, one flag

```
cli.py (run / rebuild_all)
  ├─ context_narrative: false → build_context()              [existing, unchanged]
  └─ context_narrative: true  → build_narrative_context()    [new orchestrator]
```

`build_context()` remains the deterministic fallback, untouched.

### Narrative pipeline flow

```
build_narrative_context(graph, memory_path, config)
  │
  ├─ 1. get_top_entities(graph, n=50)
  ├─ 2. Group by section via SECTION_MAP (entity type → section name)
  │
  ├─ 3. For each section (in SECTION_ORDER):
  │     ├─ a. Select section entities
  │     ├─ b. _enrich_entity() for each                      [existing]
  │     ├─ c. For top 3-5 by score:
  │     │     ├─ Build RAG query (title + top 3 facts)
  │     │     ├─ search() from indexer.py                     [direct Python call]
  │     │     └─ NLP dedup: compute_similarity() on chunks vs existing facts
  │     ├─ d. Assemble section_input (dossiers + RAG chunks)
  │     └─ e. call_section_generation(section_input, section_name, config)
  │           → prose paragraph
  │
  ├─ 4. Final pass "How to interact":
  │     ├─ Input: all prose sections + ai_self entities
  │     └─ call_interaction_generation(sections_prose, ai_self_data, config)
  │       → "How to interact" paragraph
  │
  ├─ 5. Assemble via context_template_narrative.md
  │     (static header + prose sections + static footer)
  │
  └─ return complete markdown
```

### Entity type → section mapping

```python
SECTION_MAP = {
    "person": "family",
    "animal": "family",
    "health": "health",
    "work": "work",
    "organization": "work",
    "project": "work",
    "interest": "hobbies",
    "place": "hobbies",
    "ai_self": "interaction",  # handled in final pass
}

SECTION_ORDER = ["identity", "hobbies", "health", "work", "family", "vigilances", "interaction"]
```

- **identity**: entities in `self/` folder, type is not `health` nor `ai_self`
- **vigilances**: transversal scan of `[vigilance]`/`[diagnosis]`/`[treatment]` markers across all shown entities → passed through LLM for prose formatting (consistent with other sections)

### RAG query strategy

For each section's top 3-5 entities by score:
- Query = `"{entity_title} — {top_3_facts_by_importance}"` (title + top 3 facts)
- Provides enough semantic signal for FAISS without overwhelming the embedding
- `search(query, config, memory_path, top_k=3)` called directly (not via MCP)

### NLP dedup integration

In `_build_section_input()`, after RAG results return:

1. **Chunk vs facts**: if `compute_similarity(chunk, facts_text) >= config.nlp.dedup_threshold` (default 0.85) → filter out redundant chunk
2. **Chunk vs chunk**: if two RAG chunks are >= threshold similar (from different entity queries in same section) → keep highest FAISS score

Uses existing `compute_similarity()` from `nlp_prefilter.py` and `dedup_threshold` from `NLPConfig`.

### Cross-section richness

Richness comes from three mechanisms, not from complex entity routing:

1. **Graph relations** (BFS depth=1 via `_enrich_entity()`) — naturally bring related entities from other types
2. **RAG chunks** — can return content from related entities across types
3. **"How to interact" final pass** — receives ALL sections, synthesizes cross-cutting priorities

No multi-section entities, no custom tags, no complex routing.

## Prompt structure

### File layout

```
prompts/
├── generate_section.md              # NEW — base prompt for each section
├── generate_interaction.md          # NEW — final pass prompt
├── context_template_narrative.md    # NEW — prose assembly template
├── sections/                        # NEW — per-section instruction snippets
│   ├── identity.md
│   ├── hobbies.md
│   ├── health.md
│   ├── work.md
│   ├── family.md
│   └── vigilances.md
├── context_template.md              # existing, unchanged (deterministic mode)
├── context_instructions.md          # existing, unchanged (custom user rules)
├── extract_facts.md                 # existing, unchanged
├── arbitrate_entity.md              # existing, unchanged
├── summarize_entity.md              # existing, unchanged
└── consolidate.md                   # existing, unchanged
```

`generate_context.md` is deleted (replaced by `generate_section.md` + `generate_interaction.md`).

### generate_section.md

Variables: `{user_language}`, `{token_budget}`, `{section_instructions}`, `{enriched_data}`, `{rag_context}`

Rules: write in user language, natural prose, respect token budget, bold critical items, no metadata in output, weave relations naturally.

### Section snippets (3-5 lines each)

- **identity.md**: Core identity, professional situation, family overview
- **hobbies.md**: Concrete interests, active projects if hobby-related, current engagement level
- **health.md**: Active diagnoses, current treatments, health trends, impact on daily life, non-alarmist tone
- **work.md**: Role, employer, active projects (pro + personal), deadlines, tech stacks, career events
- **family.md**: Key people, behavioral nuances, relationship dynamics
- **vigilances.md**: Transform markers into short actionable list, merge related items, max 6-8 items

### generate_interaction.md

Variables: `{user_language}`, `{ai_self_data}`, `{sections_prose}`, `{date}`

Rules: 5-6 lines max, combine stable preferences with today's priorities, specific and actionable, imperative tone.

### context_template_narrative.md

Static template with section placeholders:
`{date}`, `{user_language_name}`, `{user_name}`, `{section_identity}`, `{section_hobbies}`, `{section_health}`, `{section_work}`, `{section_family}`, `{section_vigilances}`, `{section_interaction}`, `{custom_instructions}`

## CLI wiring

Both `run()` and `rebuild_all()` in `cli.py` branch on `config.context_narrative`:

```python
if config.context_narrative:
    context_text = build_narrative_context(graph, memory_path, config)
else:
    context_text = build_context(graph, memory_path, config)
```

## Config changes

### Legacy fields to remove

From `ScoringConfig`:
- `weight_importance`, `weight_frequency`, `weight_recency`, `frequency_cap`, `recency_halflife_days`

From `Config` (if present):
- `job_schedule`, `job_idle_trigger_minutes`

### context_budget update

New keys for narrative mode:
```yaml
context_budget:
  identity: 12
  hobbies: 10
  health: 15
  work: 15
  family: 12
  vigilances: 8
  interaction: 10
```

## Files summary

### Modified

| File | Changes |
|---|---|
| `src/memory/context.py` | Add `build_narrative_context()`, `_build_section_input()`, `_build_rag_query()`, `_collect_vigilances()`. Remove `build_context_input()`, `generate_context()`. |
| `src/core/llm.py` | Replace `call_context_generation()` with `call_section_generation()` + `call_interaction_generation()`. |
| `src/cli.py` | Wire `context_narrative` flag at both call sites. |
| `src/core/config.py` | Remove legacy fields from `ScoringConfig` and `Config`. |
| `config.yaml.example` | Update `context_budget` keys, remove legacy fields. |

### Created

| File | Purpose |
|---|---|
| `prompts/generate_section.md` | Base prompt for section generation |
| `prompts/generate_interaction.md` | Final pass "How to interact" prompt |
| `prompts/context_template_narrative.md` | Prose assembly template |
| `prompts/sections/identity.md` | Section snippet |
| `prompts/sections/hobbies.md` | Section snippet |
| `prompts/sections/health.md` | Section snippet |
| `prompts/sections/work.md` | Section snippet |
| `prompts/sections/family.md` | Section snippet |
| `prompts/sections/vigilances.md` | Section snippet |

### Deleted

| File | Reason |
|---|---|
| `prompts/generate_context.md` | Replaced by generate_section.md + generate_interaction.md |

### Unchanged

All scoring, graph, extraction, enricher, MCP server, indexer code. `build_context()` deterministic path intact.

## Function inventory

### New

| Function | File | Role |
|---|---|---|
| `build_narrative_context(graph, memory_path, config)` | context.py | Main orchestrator |
| `_build_section_input(entities, graph, memory_path, config)` | context.py | Enrich entities + RAG query + NLP dedup |
| `_build_rag_query(entity_id, entity, memory_path)` | context.py | Build "title + top 3 facts" query |
| `_collect_vigilances(shown_entities, graph, memory_path)` | context.py | Transversal vigilance marker scan |
| `call_section_generation(section_input, section_name, config)` | llm.py | LLM call for one section |
| `call_interaction_generation(sections_prose, ai_self_data, config)` | llm.py | LLM call for final pass |

### Removed

| Function | File | Replaced by |
|---|---|---|
| `generate_context()` | context.py | Loop in `build_narrative_context()` |
| `build_context_input()` | context.py | `_build_section_input()` (per-section, with RAG + dedup) |
| `call_context_generation()` | llm.py | `call_section_generation()` + `call_interaction_generation()` |

### Wired (previously orphaned)

| Function/Field | Now used in |
|---|---|
| `compute_similarity()` | `_build_section_input()` for NLP dedup |
| `dedup_threshold` | `_build_section_input()` via `config.nlp.dedup_threshold` |
| `context_narrative` | `cli.py` run + rebuild_all branching |
