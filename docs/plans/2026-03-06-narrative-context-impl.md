# Narrative Context Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add LLM-powered narrative prose context generation as an alternative to the existing deterministic context builder.

**Architecture:** Section-by-section LLM calls with RAG enrichment. Each section gets its primary entities (by type), discovers additional related entities via FAISS search, deduplicates with NLP similarity, then generates prose via a small local model. A final pass synthesizes "How to interact" from all sections. Controlled by `context_narrative` config flag.

**Tech Stack:** LiteLLM, FAISS (existing indexer.search), spaCy (existing compute_similarity), Pydantic config

**Design doc:** `docs/plans/2026-03-06-narrative-context-design.md`

---

### Task 1: Config cleanup — remove legacy fields

**Files:**
- Modify: `src/core/config.py:53-58` (ScoringConfig legacy fields)
- Modify: `src/core/config.py:116-117` (Config scheduler fields)
- Modify: `src/core/config.py:211-215` (load_config legacy loading)
- Modify: `src/core/config.py:233-234` (load_config scheduler loading)
- Modify: `config.yaml.example:49-59` (context_budget keys)
- Modify: `config.yaml.example:74-78` (job section)
- Modify: `tests/test_config.py` (update assertions)

**Step 1: Write a failing test**

Add test in `tests/test_config.py` confirming legacy fields are gone:

```python
def test_legacy_fields_removed():
    """Legacy scoring fields should no longer exist."""
    from src.core.config import ScoringConfig
    sc = ScoringConfig()
    assert not hasattr(sc, 'weight_importance')
    assert not hasattr(sc, 'weight_frequency')
    assert not hasattr(sc, 'weight_recency')
    assert not hasattr(sc, 'frequency_cap')
    assert not hasattr(sc, 'recency_halflife_days')
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_legacy_fields_removed -v`
Expected: FAIL (fields still exist)

**Step 3: Remove legacy fields**

In `src/core/config.py`:

1. Remove lines 53-58 from `ScoringConfig`:
```python
    # DELETE these 5 lines:
    weight_importance: float = 0.4
    weight_frequency: float = 0.3
    weight_recency: float = 0.3
    frequency_cap: int = 20
    recency_halflife_days: int = 30
```

2. Remove lines 116-117 from `Config`:
```python
    # DELETE these 2 lines:
    job_schedule: str = "0 3 * * *"
    job_idle_trigger_minutes: int = 10
```

3. Remove lines 211-215 from `load_config()` (the scoring legacy loading):
```python
    # DELETE these 5 lines:
    weight_importance=scoring.get("weight_importance", 0.4),
    weight_frequency=scoring.get("weight_frequency", 0.3),
    weight_recency=scoring.get("weight_recency", 0.3),
    frequency_cap=scoring.get("frequency_cap", 20),
    recency_halflife_days=scoring.get("recency_halflife_days", 30),
```

4. Remove lines 233-234 from `load_config()` (scheduler loading):
```python
    # DELETE these 2 lines:
    job_schedule=job.get("schedule", "0 3 * * *"),
    job_idle_trigger_minutes=job.get("idle_trigger_minutes", 10),
```

5. Update `config.yaml.example` — replace `context_budget` section (lines 49-59) with:
```yaml
  context_budget:
    # Deterministic mode keys
    ai_personality: 8
    identity: 10
    work: 10
    personal: 10
    top_of_mind: 17
    vigilances: 10
    history_recent: 12
    history_earlier: 8
    history_longterm: 5
    instructions: 10
    # Narrative mode keys (used when context_narrative: true)
    # identity: 12
    # hobbies: 10
    # health: 15
    # work: 15
    # family: 12
    # vigilances: 8
    # interaction: 10
```

6. Remove `job` section from `config.yaml.example` (lines 74-78). Keep `max_chats_per_run` in Config (still used in cli.py:64).

7. Update existing test in `tests/test_config.py` — the assertion `assert config.scoring.weight_importance == 0.4` must be removed. Also remove `weight_importance` etc from the test config fixture YAML if present.

**Step 4: Run all tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/core/config.py config.yaml.example tests/test_config.py
git commit -m "chore: remove legacy scoring fields and scheduler config"
```

---

### Task 2: Create prompt files

**Files:**
- Create: `prompts/generate_section.md`
- Create: `prompts/generate_interaction.md`
- Create: `prompts/context_template_narrative.md`
- Create: `prompts/sections/identity.md`
- Create: `prompts/sections/hobbies.md`
- Create: `prompts/sections/health.md`
- Create: `prompts/sections/work.md`
- Create: `prompts/sections/family.md`
- Create: `prompts/sections/vigilances.md`
- Delete: `prompts/generate_context.md`

**Step 1: Create directory and all prompt files**

Create `prompts/sections/` directory.

`prompts/generate_section.md`:
```markdown
# SYSTEM

You condense structured memory data into a fluent prose paragraph
for a personal AI assistant's context file.

Rules:
- Write in {user_language}. Natural prose, not bullet points.
- Maximum ~{token_budget} tokens for this section.
- Prioritize high-score entities. Mention all provided entities at least once.
- Weave relations naturally: if entity A affects entity B, say so inline.
- Bold **critical items** (diagnoses, deadlines, important dates).
- Do not invent information. Only use what is provided below.
- No metadata (scores, retention levels, tags) in the output.
- Start directly with the content. No section header.

## Section-specific instructions

{section_instructions}

# USER

## Entities and facts

{enriched_data}

## Additional related context from memory

{rag_context}

Write the paragraph now.
```

`prompts/generate_interaction.md`:
```markdown
# SYSTEM

You write the "How to interact" section for a personal AI assistant.
This section tells the assistant HOW to behave in the current context.

Rules:
- Write in {user_language}. 5-6 lines maximum.
- Combine stable interaction preferences with TODAY's priorities.
- Reason about what matters NOW: health status, upcoming deadlines,
  emotional context, recent events.
- Be specific and actionable: "ask about X", "don't push Y",
  "remind about Z deadline".
- Do not repeat information already in the sections — reference it.
- Direct, imperative tone. This is instructions TO the assistant.

# USER

## Interaction style preferences (stable)

{ai_self_data}

## Current context (all sections generated today, {date})

{sections_prose}

Write the "How to interact" instructions now.
```

`prompts/context_template_narrative.md`:
```markdown
# Personal Memory — {date}

You are a personal assistant with persistent memory.
Use this information to personalize your responses.

**Language: respond in {user_language_name}.**

---

## Who is {user_name}

{section_identity}

## Passions & hobbies

{section_hobbies}

## Health

{section_health}

## Work & projects

{section_work}

## Family & close ones

{section_family}

## Vigilances

{section_vigilances}

## How to interact

{section_interaction}

---

## Extended memory access

If you need more details about any topic the user mentions,
use the `search_rag` tool with a relevant query.

{custom_instructions}
```

`prompts/sections/identity.md`:
```markdown
Summarize core identity: name, age, location, professional role, family situation.
Keep it concise — this is the overview, other sections have details.
Mention key life circumstances (remote work, disabilities, living situation) if present.
```

`prompts/sections/hobbies.md`:
```markdown
List active interests and passions with concrete details.
Mention current engagement level and recent activity if known.
Include technical hobbies with specifics (tools, frameworks, hardware).
```

`prompts/sections/health.md`:
```markdown
Focus on active diagnoses, current treatments, and health evolution.
Bold **critical vigilances** and **active treatments** inline.
Mention impact on daily life and work if relations exist.
Factual, non-alarmist tone. Recent changes first.
```

`prompts/sections/work.md`:
```markdown
Cover professional role, employer, and active projects (personal included).
Highlight **deadlines**, current blockers, and tech stacks for dev projects.
Mention career events (reviews, salary discussions) if present.
Group professional and personal projects in the same flow.
```

`prompts/sections/family.md`:
```markdown
Describe key people and their relationship to the user.
Include behavioral nuances important for interactions (sensitivities, dynamics).
Mention pets with their personality if present.
Keep respectful tone — these are real people.
```

`prompts/sections/vigilances.md`:
```markdown
Transform vigilance markers into a short, actionable list for the assistant.
Each item = one direct instruction. Format: "- Instruction courte."
No categories, no prefixes, no entity names as headers.
Merge related items (same diagnosis + treatment = one line).
Maximum 6-8 items. Most critical first.
```

**Step 2: Delete old prompt**

Delete `prompts/generate_context.md`.

**Step 3: Commit**

```bash
git add prompts/generate_section.md prompts/generate_interaction.md prompts/context_template_narrative.md prompts/sections/
git rm prompts/generate_context.md
git commit -m "feat: add narrative context prompt files, remove old generate_context.md"
```

---

### Task 3: Add LLM functions — call_section_generation + call_interaction_generation

**Files:**
- Modify: `src/core/llm.py:225-252` (replace call_context_generation)
- Test: `tests/test_llm_context.py` (new file)

**Step 1: Write failing tests**

Create `tests/test_llm_context.py`:

```python
"""Tests for narrative context LLM functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.config import Config, LLMStepConfig


def _make_config(tmp_path):
    config = Config.__new__(Config)
    config.llm_context = LLMStepConfig(model="test/model", temperature=0.3, timeout=60)
    config.user_language = "fr"
    config.context_max_tokens = 3000
    config.context_budget = {"health": 15}
    config.prompts_path = tmp_path / "prompts"
    config.categories = MagicMock()
    config.categories.observations = ["fact", "diagnosis"]
    config.categories.entity_types = ["health", "person"]
    config.categories.relation_types = ["affects"]
    return config


def test_call_section_generation(tmp_path):
    """call_section_generation loads prompt + snippet and returns LLM text."""
    config = _make_config(tmp_path)

    # Create prompt files
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "sections").mkdir()
    (prompts / "generate_section.md").write_text(
        "Instructions: {section_instructions}\nData: {enriched_data}\nRAG: {rag_context}\nBudget: {token_budget}\nLang: {user_language}",
        encoding="utf-8",
    )
    (prompts / "sections" / "health.md").write_text("Focus on health.", encoding="utf-8")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Prose paragraph about health."

    with patch("src.core.llm.litellm.completion", return_value=mock_response) as mock_llm:
        from src.core.llm import call_section_generation
        result = call_section_generation("entity data", "rag data", "health", config)

    assert result == "Prose paragraph about health."
    # Verify prompt was composed correctly
    call_args = mock_llm.call_args
    prompt_text = call_args[1]["messages"][0]["content"]
    assert "Focus on health." in prompt_text
    assert "entity data" in prompt_text
    assert "rag data" in prompt_text


def test_call_interaction_generation(tmp_path):
    """call_interaction_generation loads prompt and returns LLM text."""
    config = _make_config(tmp_path)

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "generate_interaction.md").write_text(
        "Style: {ai_self_data}\nSections: {sections_prose}\nDate: {date}\nLang: {user_language}",
        encoding="utf-8",
    )

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Direct, motivant, zero filler."

    with patch("src.core.llm.litellm.completion", return_value=mock_response) as mock_llm:
        from src.core.llm import call_interaction_generation
        result = call_interaction_generation("all sections", "ai style prefs", config)

    assert result == "Direct, motivant, zero filler."
    prompt_text = mock_llm.call_args[1]["messages"][0]["content"]
    assert "ai style prefs" in prompt_text
    assert "all sections" in prompt_text


def test_call_section_generation_strips_thinking(tmp_path):
    """Thinking tags from reasoning models should be stripped."""
    config = _make_config(tmp_path)

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "sections").mkdir()
    (prompts / "generate_section.md").write_text("{section_instructions} {enriched_data} {rag_context} {token_budget}", encoding="utf-8")
    (prompts / "sections" / "work.md").write_text("Work focus.", encoding="utf-8")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "<think>reasoning</think>Clean output."

    with patch("src.core.llm.litellm.completion", return_value=mock_response):
        from src.core.llm import call_section_generation
        result = call_section_generation("data", "rag", "work", config)

    assert result == "Clean output."
    assert "<think>" not in result
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm_context.py -v`
Expected: FAIL (functions don't exist yet)

**Step 3: Implement the functions**

In `src/core/llm.py`, replace `call_context_generation` (lines 225-252) with:

```python
def call_section_generation(
    enriched_data: str,
    rag_context: str,
    section_name: str,
    config: Config,
) -> str:
    """Generate one narrative section from enriched data + RAG context.

    Loads generate_section.md as base prompt, injects section-specific
    instructions from prompts/sections/{section_name}.md.
    Returns free-text prose.
    """
    from datetime import date

    # Load section-specific instructions snippet
    snippet_path = config.prompts_path / "sections" / f"{section_name}.md"
    section_instructions = ""
    if snippet_path.exists():
        section_instructions = snippet_path.read_text(encoding="utf-8")

    token_budget = config.context_budget.get(section_name, 10)
    budget_tokens = int(max(config.context_max_tokens - 500, 1000) * token_budget / 100)

    prompt = load_prompt(
        "generate_section",
        config,
        section_instructions=section_instructions,
        enriched_data=enriched_data,
        rag_context=rag_context or "No additional context.",
        token_budget=str(budget_tokens),
    )

    step_config = config.llm_context
    kwargs: dict[str, Any] = {
        "model": step_config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": step_config.temperature,
    }
    if step_config.timeout:
        kwargs["timeout"] = step_config.timeout
    if step_config.api_base:
        kwargs["api_base"] = step_config.api_base

    response = litellm.completion(**kwargs)
    text = response.choices[0].message.content or ""
    return strip_thinking(text)


def call_interaction_generation(
    sections_prose: str,
    ai_self_data: str,
    config: Config,
) -> str:
    """Generate the 'How to interact' section from all prose sections + ai_self data.

    This is the final pass that synthesizes cross-cutting priorities.
    Returns free-text prose (5-6 lines).
    """
    from datetime import date

    prompt = load_prompt(
        "generate_interaction",
        config,
        ai_self_data=ai_self_data,
        sections_prose=sections_prose,
        date=date.today().isoformat(),
    )

    step_config = config.llm_context
    kwargs: dict[str, Any] = {
        "model": step_config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": step_config.temperature,
    }
    if step_config.timeout:
        kwargs["timeout"] = step_config.timeout
    if step_config.api_base:
        kwargs["api_base"] = step_config.api_base

    response = litellm.completion(**kwargs)
    text = response.choices[0].message.content or ""
    return strip_thinking(text)
```

Also update the import in `src/memory/context.py` line 9: change `from src.core.llm import call_context_generation` to `from src.core.llm import call_section_generation, call_interaction_generation`.

**Step 4: Run tests**

Run: `uv run pytest tests/test_llm_context.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/core/llm.py tests/test_llm_context.py src/memory/context.py
git commit -m "feat: add call_section_generation and call_interaction_generation"
```

---

### Task 4: Implement _build_rag_query helper

**Files:**
- Modify: `src/memory/context.py` (add function)
- Test: `tests/test_context_narrative.py` (new file)

**Step 1: Write failing test**

Create `tests/test_context_narrative.py`:

```python
"""Tests for narrative context pipeline (context.py additions)."""

from pathlib import Path

from src.core.config import Config, ScoringConfig
from src.core.models import EntityFrontmatter, GraphData, GraphEntity
from src.memory.store import write_entity


def _make_config(tmp_path):
    config = Config.__new__(Config)
    config.scoring = ScoringConfig(min_score_for_context=0.0)
    config.memory_path = tmp_path
    config.context_max_tokens = 3000
    config.context_budget = {"identity": 12, "hobbies": 10, "health": 15, "work": 15, "family": 12, "vigilances": 8, "interaction": 10}
    config.user_language = "fr"
    config.user_language_name  # property, needs full Config
    config.prompts_path = tmp_path / "prompts"
    config.context_narrative = True
    return config


def _make_full_config(tmp_path):
    """Create a proper Config instance (not __new__)."""
    from src.core.config import NLPConfig
    config = _make_config(tmp_path)
    config.nlp = NLPConfig(enabled=True, dedup_threshold=0.85)
    config.faiss = None  # Will be mocked
    return config


def test_build_rag_query_with_facts(tmp_path):
    """RAG query should include entity title + top 3 facts."""
    (tmp_path / "self").mkdir()
    fm = EntityFrontmatter(
        title="Mal de dos", type="health", score=0.8, importance=0.9,
    )
    write_entity(
        tmp_path / "self" / "mal-de-dos.md", fm,
        {"Facts": [
            "- [diagnosis] Hernie discale L5-S1",
            "- [treatment] Kiné 2x/semaine",
            "- [progression] Amélioration lente",
            "- [fact] Douleur depuis 2024",
        ], "Relations": [], "History": []},
    )

    entity = GraphEntity(
        file="self/mal-de-dos.md", type="health", title="Mal de dos",
        score=0.8, importance=0.9,
    )

    from src.memory.context import _build_rag_query
    query = _build_rag_query("mal-de-dos", entity, tmp_path)

    assert "Mal de dos" in query
    # Should have max 3 facts
    assert query.count(" — ") <= 1  # title — facts joined
    lines = query.split(" — ", 1)
    assert lines[0] == "Mal de dos"


def test_build_rag_query_no_facts(tmp_path):
    """RAG query with no MD file should return just the title."""
    entity = GraphEntity(
        file="self/missing.md", type="health", title="Unknown",
        score=0.5,
    )

    from src.memory.context import _build_rag_query
    query = _build_rag_query("unknown", entity, tmp_path)
    assert query == "Unknown"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_context_narrative.py::test_build_rag_query_with_facts -v`
Expected: FAIL (function doesn't exist)

**Step 3: Implement _build_rag_query**

Add to `src/memory/context.py` after `_collect_section` (after line 93):

```python
def _build_rag_query(entity_id: str, entity: GraphEntity, memory_path: Path) -> str:
    """Build a RAG search query from entity title + top 3 facts.

    Returns "Title — fact1, fact2, fact3" or just "Title" if no facts found.
    """
    entity_path = (memory_path / entity.file).resolve()
    facts: list[str] = []
    if entity_path.is_relative_to(memory_path.resolve()) and entity_path.exists():
        try:
            _, sections = read_entity(entity_path)
            raw_facts = sections.get("Facts", [])
            # Strip markdown list prefix and category tags for cleaner query
            for f in raw_facts[:3]:
                clean = f.lstrip("- ").strip()
                # Remove [category] prefix
                if clean.startswith("["):
                    close = clean.find("]")
                    if close != -1:
                        clean = clean[close + 1:].strip()
                if clean:
                    facts.append(clean)
        except Exception:
            pass

    if facts:
        return f"{entity.title} — {', '.join(facts)}"
    return entity.title
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_context_narrative.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/memory/context.py tests/test_context_narrative.py
git commit -m "feat: add _build_rag_query helper for RAG search queries"
```

---

### Task 5: Implement _build_section_input with RAG + NLP dedup

**Files:**
- Modify: `src/memory/context.py` (add function)
- Modify: `tests/test_context_narrative.py` (add tests)

**Step 1: Write failing tests**

Add to `tests/test_context_narrative.py`:

```python
from unittest.mock import patch, MagicMock
from src.core.models import SearchResult


def test_build_section_input_enriches_entities(tmp_path):
    """Section input should contain enriched dossiers for section entities."""
    config = _make_full_config(tmp_path)

    (tmp_path / "close_ones").mkdir()
    fm = EntityFrontmatter(title="Alice", type="person", score=0.7, importance=0.6)
    write_entity(tmp_path / "close_ones" / "alice.md", fm,
                 {"Facts": ["- [fact] Best friend"], "Relations": [], "History": []})

    graph = GraphData()
    graph.entities["alice"] = GraphEntity(
        file="close_ones/alice.md", type="person", title="Alice",
        score=0.7, importance=0.6,
    )
    entities = [("alice", graph.entities["alice"])]

    with patch("src.memory.context.search", return_value=[]):
        from src.memory.context import _build_section_input
        enriched, rag = _build_section_input(entities, graph, tmp_path, config)

    assert "Alice" in enriched
    assert "Best friend" in enriched


def test_build_section_input_rag_adds_related(tmp_path):
    """RAG results should add related entities not already in section."""
    config = _make_full_config(tmp_path)

    # Primary entity
    (tmp_path / "self").mkdir()
    fm1 = EntityFrontmatter(title="Sciatique", type="health", score=0.8)
    write_entity(tmp_path / "self" / "sciatique.md", fm1,
                 {"Facts": ["- [diagnosis] Hernie L5-S1"], "Relations": [], "History": []})

    # Related entity (found by RAG)
    (tmp_path / "self" / "kine.md").parent.mkdir(exist_ok=True)
    fm2 = EntityFrontmatter(title="Kinésithérapie", type="health", score=0.6)
    write_entity(tmp_path / "self" / "kine.md", fm2,
                 {"Facts": ["- [treatment] 2 séances/semaine"], "Relations": [], "History": []})

    graph = GraphData()
    graph.entities["sciatique"] = GraphEntity(
        file="self/sciatique.md", type="health", title="Sciatique", score=0.8,
    )
    graph.entities["kine"] = GraphEntity(
        file="self/kine.md", type="health", title="Kinésithérapie", score=0.6,
    )

    entities = [("sciatique", graph.entities["sciatique"])]

    rag_result = SearchResult(entity_id="kine", file="self/kine.md", chunk="[chunk 0]", score=0.9)

    with patch("src.memory.context.search", return_value=[rag_result]):
        from src.memory.context import _build_section_input
        enriched, rag = _build_section_input(entities, graph, tmp_path, config)

    assert "Sciatique" in enriched
    assert "Kinésithérapie" in rag


def test_build_section_input_dedup_filters_similar(tmp_path):
    """RAG entities too similar to primary entities should be filtered."""
    config = _make_full_config(tmp_path)
    config.nlp.dedup_threshold = 0.85

    (tmp_path / "self").mkdir()
    fm1 = EntityFrontmatter(title="Sciatique", type="health", score=0.8)
    write_entity(tmp_path / "self" / "sciatique.md", fm1,
                 {"Facts": ["- [diagnosis] Hernie discale L5-S1"], "Relations": [], "History": []})

    # Duplicate-ish entity
    fm2 = EntityFrontmatter(title="Hernie", type="health", score=0.5)
    write_entity(tmp_path / "self" / "hernie.md", fm2,
                 {"Facts": ["- [diagnosis] Hernie discale L5-S1"], "Relations": [], "History": []})

    graph = GraphData()
    graph.entities["sciatique"] = GraphEntity(
        file="self/sciatique.md", type="health", title="Sciatique", score=0.8,
    )
    graph.entities["hernie"] = GraphEntity(
        file="self/hernie.md", type="health", title="Hernie", score=0.5,
    )

    entities = [("sciatique", graph.entities["sciatique"])]
    rag_result = SearchResult(entity_id="hernie", file="self/hernie.md", chunk="[chunk 0]", score=0.8)

    # Mock compute_similarity to return high similarity (above threshold)
    with patch("src.memory.context.search", return_value=[rag_result]), \
         patch("src.memory.context.compute_similarity", return_value=0.95), \
         patch("src.memory.context.is_available", return_value=True):
        from src.memory.context import _build_section_input
        enriched, rag = _build_section_input(entities, graph, tmp_path, config)

    # RAG should be empty since the entity was too similar
    assert rag == "" or "Hernie" not in rag
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_context_narrative.py::test_build_section_input_enriches_entities -v`
Expected: FAIL

**Step 3: Implement _build_section_input**

Add to `src/memory/context.py`, adding necessary imports at top:

At the top of the file, add imports:
```python
from src.pipeline.indexer import search
from src.pipeline.nlp_prefilter import compute_similarity, is_available
```

Add function after `_build_rag_query`:

```python
def _build_section_input(
    entities: list[tuple[str, GraphEntity]],
    graph: GraphData,
    memory_path: Path,
    config: Config,
) -> tuple[str, str]:
    """Build enriched input for one narrative section.

    1. Enriches primary entities with _enrich_entity()
    2. For top 5 entities, queries RAG to discover related entities
    3. Deduplicates RAG results with NLP similarity
    4. Returns (enriched_data, rag_context) as two text blocks
    """
    # 1. Enrich primary entities
    primary_ids = {eid for eid, _ in entities}
    dossiers = []
    for eid, ent in entities:
        dossiers.append(_enrich_entity(eid, ent, graph, memory_path))
    enriched_data = "\n\n".join(dossiers)

    # 2. RAG search for top entities
    rag_entity_ids: dict[str, float] = {}  # entity_id -> best score
    for eid, ent in entities[:5]:
        query = _build_rag_query(eid, ent, memory_path)
        try:
            results = search(query, config, memory_path, top_k=3)
            for r in results:
                if r.entity_id not in primary_ids:
                    if r.entity_id not in rag_entity_ids or r.score > rag_entity_ids[r.entity_id]:
                        rag_entity_ids[r.entity_id] = r.score
        except Exception:
            pass  # FAISS not available or index missing

    # 3. Enrich RAG-discovered entities
    rag_dossiers = []
    for rel_id, _score in sorted(rag_entity_ids.items(), key=lambda x: x[1], reverse=True):
        if rel_id in graph.entities:
            dossier = _enrich_entity(rel_id, graph.entities[rel_id], graph, memory_path)
            rag_dossiers.append((rel_id, dossier))

    # 4. NLP dedup: filter RAG entities too similar to primary entities
    if rag_dossiers and is_available() and config.nlp.dedup_threshold > 0:
        filtered = []
        for rel_id, dossier in rag_dossiers:
            sim = compute_similarity(dossier, enriched_data)
            if sim < config.nlp.dedup_threshold:
                filtered.append(dossier)
        rag_context = "\n\n".join(filtered)
    else:
        rag_context = "\n\n".join(d for _, d in rag_dossiers)

    return enriched_data, rag_context
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_context_narrative.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/memory/context.py tests/test_context_narrative.py
git commit -m "feat: add _build_section_input with RAG enrichment and NLP dedup"
```

---

### Task 6: Implement _collect_vigilances helper

**Files:**
- Modify: `src/memory/context.py` (add function)
- Modify: `tests/test_context_narrative.py` (add test)

**Step 1: Write failing test**

Add to `tests/test_context_narrative.py`:

```python
def test_collect_vigilances(tmp_path):
    """Should collect vigilance/diagnosis/treatment markers across entities."""
    (tmp_path / "self").mkdir()
    fm = EntityFrontmatter(title="Sciatique", type="health", score=0.8)
    write_entity(tmp_path / "self" / "sciatique.md", fm,
                 {"Facts": [
                     "- [diagnosis] Hernie discale L5-S1",
                     "- [treatment] Kiné 2x/semaine",
                     "- [fact] Douleur chronique",
                     "- [vigilance] Ne pas forcer sur le sport",
                 ], "Relations": [], "History": []})

    graph = GraphData()
    graph.entities["sciatique"] = GraphEntity(
        file="self/sciatique.md", type="health", title="Sciatique", score=0.8,
    )

    shown = [("sciatique", graph.entities["sciatique"])]

    from src.memory.context import _collect_vigilances
    result = _collect_vigilances(shown, graph, tmp_path)

    assert "Hernie discale" in result
    assert "Kiné" in result
    assert "Ne pas forcer" in result
    # Regular facts should NOT be included
    assert "Douleur chronique" not in result
```

**Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_context_narrative.py::test_collect_vigilances -v`
Expected: FAIL

**Step 3: Implement**

Add to `src/memory/context.py`:

```python
def _collect_vigilances(
    shown_entities: list[tuple[str, GraphEntity]],
    graph: GraphData,
    memory_path: Path,
) -> str:
    """Scan all shown entities for vigilance/diagnosis/treatment markers.

    Returns a structured text block suitable as LLM input for the
    vigilances section. Not prose yet — will be passed through
    call_section_generation for formatting.
    """
    markers = []
    for eid, entity in shown_entities:
        if entity.type == "ai_self":
            continue
        entity_path = (memory_path / entity.file).resolve()
        if not entity_path.exists() or not entity_path.is_relative_to(memory_path.resolve()):
            continue
        try:
            _, sections = read_entity(entity_path)
            facts = sections.get("Facts", [])
            for fact in facts:
                fact_lower = fact.lower()
                if any(tag in fact_lower for tag in ("[vigilance]", "[diagnosis]", "[treatment]")):
                    markers.append(f"- {entity.title}: {fact.lstrip('- ').strip()}")
        except Exception:
            pass

    return "\n".join(markers) if markers else "No vigilance markers found."
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_context_narrative.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/memory/context.py tests/test_context_narrative.py
git commit -m "feat: add _collect_vigilances transversal marker scanner"
```

---

### Task 7: Implement build_narrative_context orchestrator

**Files:**
- Modify: `src/memory/context.py` (add main function, remove old functions)
- Modify: `tests/test_context_narrative.py` (add integration test)

**Step 1: Write failing test**

Add to `tests/test_context_narrative.py`:

```python
def test_build_narrative_context_integration(tmp_path):
    """Full narrative pipeline should produce markdown with all sections."""
    config = _make_full_config(tmp_path)
    config.user_language = "fr"

    # Create prompt files
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "sections").mkdir()
    (prompts / "context_template_narrative.md").write_text(
        "# Memory — {date}\nLang: {user_language_name}\nUser: {user_name}\n\n"
        "## Identity\n{section_identity}\n\n## Hobbies\n{section_hobbies}\n\n"
        "## Health\n{section_health}\n\n## Work\n{section_work}\n\n"
        "## Family\n{section_family}\n\n## Vigilances\n{section_vigilances}\n\n"
        "## How to interact\n{section_interaction}\n\n{custom_instructions}",
        encoding="utf-8",
    )
    (prompts / "context_instructions.md").write_text("Be helpful.", encoding="utf-8")
    for name in ("identity", "hobbies", "health", "work", "family", "vigilances"):
        (prompts / "sections" / f"{name}.md").write_text(f"Focus on {name}.", encoding="utf-8")
    (prompts / "generate_section.md").write_text(
        "{section_instructions} {enriched_data} {rag_context} {token_budget}",
        encoding="utf-8",
    )
    (prompts / "generate_interaction.md").write_text(
        "{ai_self_data} {sections_prose} {date}",
        encoding="utf-8",
    )

    # Create entities of various types
    (tmp_path / "self").mkdir()
    (tmp_path / "close_ones").mkdir()
    (tmp_path / "work").mkdir()

    graph = GraphData(generated="2026-03-06")

    # Identity entity
    graph.entities["me"] = GraphEntity(
        file="self/me.md", type="health", title="Moi", score=0.9,
    )
    fm_me = EntityFrontmatter(title="Moi", type="health", score=0.9)
    write_entity(tmp_path / "self" / "me.md", fm_me,
                 {"Facts": ["- [fact] Alexis, 35 ans"], "Relations": [], "History": []})

    # AI self
    graph.entities["ai"] = GraphEntity(
        file="self/ai.md", type="ai_self", title="AI Personality", score=0.8,
    )
    fm_ai = EntityFrontmatter(title="AI Personality", type="ai_self", score=0.8)
    write_entity(tmp_path / "self" / "ai.md", fm_ai,
                 {"Facts": ["- [ai_style] Direct et motivant"], "Relations": [], "History": []})

    # Work entity
    graph.entities["job"] = GraphEntity(
        file="work/job.md", type="work", title="BNP", score=0.7,
    )
    fm_job = EntityFrontmatter(title="BNP", type="work", score=0.7)
    write_entity(tmp_path / "work" / "job.md", fm_job,
                 {"Facts": ["- [fact] Offering Manager"], "Relations": [], "History": []})

    # Person entity
    graph.entities["alice"] = GraphEntity(
        file="close_ones/alice.md", type="person", title="Alice", score=0.6,
    )
    fm_alice = EntityFrontmatter(title="Alice", type="person", score=0.6)
    write_entity(tmp_path / "close_ones" / "alice.md", fm_alice,
                 {"Facts": [], "Relations": [], "History": []})

    # Mock LLM calls
    call_count = {"n": 0}
    def mock_section_gen(enriched, rag, name, cfg):
        call_count["n"] += 1
        return f"[prose for {name}]"

    def mock_interaction_gen(sections, ai_data, cfg):
        return "[interaction prose]"

    with patch("src.memory.context.call_section_generation", side_effect=mock_section_gen), \
         patch("src.memory.context.call_interaction_generation", side_effect=mock_interaction_gen), \
         patch("src.memory.context.search", return_value=[]):
        from src.memory.context import build_narrative_context
        result = build_narrative_context(graph, tmp_path, config)

    # Should have all sections
    assert "## Identity" in result
    assert "[prose for identity]" in result
    assert "## Health" in result or "[prose for health]" in result
    assert "## Work" in result
    assert "[prose for work]" in result
    assert "## Family" in result
    assert "[prose for family]" in result
    assert "## Vigilances" in result
    assert "## How to interact" in result
    assert "[interaction prose]" in result
    assert "Be helpful." in result
    # LLM was called for each non-empty section + vigilances
    assert call_count["n"] >= 2  # At least identity + work + vigilances
```

**Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_context_narrative.py::test_build_narrative_context_integration -v`
Expected: FAIL

**Step 3: Implement build_narrative_context**

Replace the old `build_context_input` and `generate_context` functions (lines 242-259) with:

```python
# ── Section mapping for narrative mode ──────────────────────

SECTION_MAP: dict[str, str] = {
    "person": "family",
    "animal": "family",
    "health": "health",
    "work": "work",
    "organization": "work",
    "project": "work",
    "interest": "hobbies",
    "place": "hobbies",
    "ai_self": "interaction",
}

SECTION_ORDER = ["identity", "hobbies", "health", "work", "family", "vigilances", "interaction"]


def build_narrative_context(
    graph: GraphData, memory_path: Path, config: Config,
) -> str:
    """Build _context.md using LLM-generated prose, section by section.

    Each section gets its entities enriched + RAG-discovered related entities,
    then a small LLM generates a prose paragraph. A final pass generates
    "How to interact" from all sections combined.
    """
    today = date.today()
    today_str = today.isoformat()

    # Load template
    template_path = config.prompts_path / "context_template_narrative.md"
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
    else:
        template = "# Personal Memory — {date}\n\n{section_identity}\n{section_work}\n{section_interaction}"

    # Load custom instructions
    instructions_path = config.prompts_path / "context_instructions.md"
    custom_instructions = ""
    if instructions_path.exists():
        custom_instructions = instructions_path.read_text(encoding="utf-8")

    # Get all scored entities
    min_score = config.scoring.min_score_for_context
    all_top = get_top_entities(graph, n=50, include_permanent=True, min_score=min_score)

    # Group entities by section
    section_entities: dict[str, list[tuple[str, GraphEntity]]] = {s: [] for s in SECTION_ORDER}
    ai_self_entities: list[tuple[str, GraphEntity]] = []

    for eid, entity in all_top:
        if entity.type == "ai_self":
            ai_self_entities.append((eid, entity))
            continue

        # Identity: entities in self/ folder that aren't health or ai_self
        if entity.file.startswith("self/") and entity.type not in ("health", "ai_self"):
            section_entities["identity"].append((eid, entity))
        else:
            section_name = SECTION_MAP.get(entity.type)
            if section_name and section_name in section_entities:
                section_entities[section_name].append((eid, entity))

    # Generate prose for each section
    section_prose: dict[str, str] = {}
    all_shown: list[tuple[str, GraphEntity]] = []

    for section_name in SECTION_ORDER:
        if section_name == "interaction":
            continue  # handled in final pass
        if section_name == "vigilances":
            continue  # handled after other sections

        entities = section_entities.get(section_name, [])
        if not entities:
            section_prose[section_name] = ""
            continue

        all_shown.extend(entities)
        enriched_data, rag_context = _build_section_input(entities, graph, memory_path, config)
        prose = call_section_generation(enriched_data, rag_context, section_name, config)
        section_prose[section_name] = prose

    # Vigilances: transversal scan + LLM formatting
    vigilance_input = _collect_vigilances(all_shown, graph, memory_path)
    if vigilance_input and vigilance_input != "No vigilance markers found.":
        section_prose["vigilances"] = call_section_generation(
            vigilance_input, "", "vigilances", config,
        )
    else:
        section_prose["vigilances"] = ""

    # Final pass: "How to interact"
    all_sections_text = "\n\n".join(
        f"## {name}\n{prose}" for name, prose in section_prose.items() if prose
    )
    ai_self_data = "\n\n".join(
        _enrich_entity(eid, ent, graph, memory_path) for eid, ent in ai_self_entities
    ) if ai_self_entities else "No interaction style data."
    section_prose["interaction"] = call_interaction_generation(
        all_sections_text, ai_self_data, config,
    )

    # Extract user name from identity entities (first entity title, or "the user")
    identity_ents = section_entities.get("identity", [])
    user_name = identity_ents[0][1].title if identity_ents else "the user"

    # Assemble template
    result = template
    result = result.replace("{date}", today_str)
    result = result.replace("{user_language_name}", config.user_language_name)
    result = result.replace("{user_name}", user_name)
    result = result.replace("{custom_instructions}", custom_instructions)
    for section_name in SECTION_ORDER:
        placeholder = f"{{section_{section_name}}}"
        result = result.replace(placeholder, section_prose.get(section_name, ""))

    return result
```

Also update the import line at the top of context.py to include the new LLM functions:
```python
from src.core.llm import call_section_generation, call_interaction_generation
```

And remove the old `build_context_input` (lines 242-254) and `generate_context` (lines 257-259) functions entirely.

**Step 4: Run all tests**

Run: `uv run pytest tests/test_context_narrative.py tests/test_context.py -v`
Expected: ALL PASS (context_narrative.py passes, existing context.py tests still pass since build_context is untouched)

**Step 5: Update test_context.py imports**

The existing `test_context.py` imports `build_context_input` and `generate_index` on line 8. Remove `build_context_input` from the import. Also remove the `test_build_context_input` and `test_path_traversal_entity_file_blocked` tests since they test the removed function.

**Step 6: Run all context tests again**

Run: `uv run pytest tests/test_context.py tests/test_context_narrative.py -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add src/memory/context.py tests/test_context.py tests/test_context_narrative.py
git commit -m "feat: add build_narrative_context orchestrator, remove old unwired functions"
```

---

### Task 8: Wire context_narrative flag in cli.py

**Files:**
- Modify: `src/cli.py:158-170` (run command context section)
- Modify: `src/cli.py:209-254` (rebuild_all command)

**Step 1: Write failing test**

Add to `tests/test_context_narrative.py`:

```python
def test_cli_run_uses_narrative_when_flag_set():
    """cli.py run should call build_narrative_context when context_narrative=True."""
    # This is a wiring check — verified by reading the code after modification
    import ast
    source = Path("src/cli.py").read_text()
    tree = ast.parse(source)
    # Check that 'context_narrative' appears in the source
    assert "context_narrative" in source
    assert "build_narrative_context" in source
```

**Step 2: Modify cli.py run command**

Replace lines 158-170 in `src/cli.py`:

```python
    # Step 7: Generate context
    try:
        mode = "narrative" if config.context_narrative else "deterministic"
        console.print(f"\n[bold]Generating context ({mode})...[/bold]")
        graph = load_graph(memory_path)
        if config.context_narrative:
            from src.memory.context import build_narrative_context
            context_text = build_narrative_context(graph, memory_path, config)
        else:
            from src.memory.context import build_deterministic_context
            context_text = build_deterministic_context(graph, memory_path, config)
        if context_text.strip():
            write_context(memory_path, context_text)
            console.print(f"  [green]_context.md updated ({mode})[/green]")
        else:
            console.print("  [dim]No entities for context[/dim]")
    except Exception as e:
        console.print(f"  [yellow]Context generation warning: {e}[/yellow]")
```

**Step 3: Modify cli.py rebuild_all command**

Update the import on line 215:
```python
    from src.memory.context import build_deterministic_context, build_narrative_context, write_context, write_index
```

Replace lines 244-248:
```python
    # Context
    mode = "narrative" if config.context_narrative else "deterministic"
    console.print(f"  Generating context ({mode})...")
    if config.context_narrative:
        context_text = build_narrative_context(graph, config.memory_path, config)
    else:
        context_text = build_deterministic_context(graph, config.memory_path, config)
    if context_text.strip():
        write_context(config.memory_path, context_text)
        console.print(f"  _context.md updated ({mode})")
```

**Step 4: Run the wiring test + all tests**

Run: `uv run pytest tests/test_context_narrative.py tests/test_context.py tests/test_config.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/cli.py
git commit -m "feat: wire context_narrative flag in cli.py run + rebuild_all"
```

---

### Task 9: Full integration test + final cleanup

**Files:**
- Run: full test suite
- Verify: no orphan imports, no broken references

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

**Step 2: Check for orphan references to removed functions**

Run: `grep -r "call_context_generation\|build_context_input\|generate_context" src/ tests/ --include="*.py"`
Expected: No matches (all references cleaned up). If `generate_index` still exists (it should — it's a different function), that's fine.

**Step 3: Check for import issues**

Run: `uv run python -c "from src.memory.context import build_context, build_narrative_context, write_context; print('OK')"`
Expected: `OK`

Run: `uv run python -c "from src.core.llm import call_section_generation, call_interaction_generation; print('OK')"`
Expected: `OK`

**Step 4: Verify prompt files exist**

Run: `ls prompts/sections/ prompts/generate_section.md prompts/generate_interaction.md prompts/context_template_narrative.md`
Expected: All files listed

**Step 5: Final commit if any cleanup was needed**

```bash
git add -A
git commit -m "chore: final cleanup for narrative context pipeline"
```

---

## Task Dependency Graph

```
Task 1 (config cleanup) ──────────────────────┐
Task 2 (prompt files) ────────────────────────┤
Task 3 (LLM functions) ──────────────────────┤
                                               ├──→ Task 5 (section input + RAG + dedup)
Task 4 (RAG query helper) ───────────────────┤
                                               │
Task 6 (vigilances) ──────────────────────────┤
                                               ├──→ Task 7 (orchestrator) ──→ Task 8 (CLI wiring) ──→ Task 9 (integration)
```

Tasks 1-4 and 6 are independent and can be done in parallel.
Task 5 depends on 3 and 4.
Task 7 depends on 5 and 6.
Task 8 depends on 7.
Task 9 depends on 8.
