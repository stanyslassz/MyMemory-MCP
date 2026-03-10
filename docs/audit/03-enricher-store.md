# Audit 03 -- Enrichment Pipeline & Entity Storage

Audit date: 2026-03-10
Files examined:
- `src/pipeline/enricher.py` (265 lines)
- `src/memory/store.py` (509 lines)
- `src/memory/graph.py` (346 lines) -- for relation handling
- `src/core/models.py` -- data models

---

## 1. Enricher (`src/pipeline/enricher.py`)

### 1.1 `enrich_memory()` -- Main entry point

**Signature** (line 23):
```python
def enrich_memory(
    resolved: ResolvedExtraction,
    config: Config,
    today: str | None = None,
) -> EnrichmentReport
```

**Purpose**: Apply a fully resolved extraction (entities + relations) to the Markdown file store and graph.

**Full flow**:
1. **Line 37-38**: Defaults `today` to `date.today().isoformat()` if None.
2. **Line 41**: Loads `_graph.json` via `load_graph()` (with corruption recovery).
3. **Lines 45-59**: Iterates `resolved.resolved` (list of `ResolvedEntity`):
   - `status == "resolved"` with an `entity_id` -> calls `_update_existing_entity()`
   - `status == "new"` -> calls `_create_new_entity()` with `suggested_slug` or a freshly computed slug
   - `status == "ambiguous"` -> silently skipped (comment says "should have been arbitrated already"), **no error raised**
   - All exceptions caught per-entity and appended to `report.errors`
4. **Lines 62-95**: Iterates `resolved.relations` (list of `RawRelation`):
   - Resolves `from_slug` and `to_slug` via `_find_entity_slug()`
   - If `to_slug` not found: creates a **stub entity** (type `interest`, importance 0.3, retention `short_term`) and adds it to graph
   - If both slugs resolved: creates `GraphRelation`, calls `add_relation()` on graph (Hebbian reinforcement or new), increments `report.relations_added`
   - **Also writes relation text** (`- type [[Target]]`) to the source entity's MD via `update_entity(filepath, new_relations=[rel_line])`
   - All exceptions caught per-relation and appended to `report.errors`
5. **Line 98**: Recalculates all ACT-R scores.
6. **Lines 101-102**: Saves graph atomically, regenerates `_index.md`.

**Key observation**: Relations are written to **two places** -- `_graph.json` (structured, via `add_relation()`) and the source entity's `.md` file (text line, via `update_entity()`). These can drift apart.

### 1.2 `_update_existing_entity()`

**Signature** (line 107):
```python
def _update_existing_entity(
    entity_id: str,
    raw_entity,          # untyped, actually RawEntity
    graph,               # untyped, actually GraphData
    memory_path: Path,
    today: str,
    report: EnrichmentReport,
    config: Config | None = None,
) -> None
```

**Full flow**:
1. **Line 120-121**: Bail if `entity_id` not in graph.
2. **Line 126-127**: Bail with error if the MD file doesn't exist on disk.
3. **Lines 131-140 -- Supersession**: Filters observations with `obs.supersedes` truthy, reads the entity file, calls `mark_observation_superseded()` for each, then writes back. This is a **separate read-write cycle** before the main update.
4. **Lines 143-156 -- Pre-consolidation gate**: If `config` is provided and `max_facts` is configured for the entity type, reads the file *again*, counts live facts, and if `live_facts + new_observations > max_facts`, calls `consolidate_entity_facts()` (LLM call). Failures are caught and logged as warnings.
5. **Lines 159-163**: Builds observation dicts from `raw_entity.observations` (drops `importance` and `supersedes` fields -- only keeps category, content, tags, date, valence).
6. **Line 166**: Calls `update_entity()` on the MD file with new observations and `last_mentioned`.
7. **Lines 169-177**: Updates in-memory graph metadata: bumps `frequency += 1`, sets `last_mentioned`, calls `add_mention()` for windowed mention tracking.
8. **Lines 180-183**: Running-average importance: `(old + new_avg) / 2`.

**Bug -- Triple file read**: The file is read up to 3 times in sequence:
- Once at line 133 for supersession
- Once at line 146 for pre-consolidation check
- Once inside `update_entity()` at store.py:69

Each read-write is independent. If supersession writes the file, the pre-consolidation check reads the updated version, which is correct sequentially but wasteful.

**Bug -- `config=None` fallback in line 199**: When `_create_new_entity()` falls back to `_update_existing_entity()` (line 199), it passes **no `config`** argument. This means the pre-consolidation gate (lines 143-156) is entirely skipped for entities that were thought to be new but already existed. The `max_facts` enforcement is silently bypassed.

### 1.3 `_create_new_entity()`

**Signature** (line 187):
```python
def _create_new_entity(
    slug: str,
    raw_entity,          # RawEntity
    graph,               # GraphData
    memory_path: Path,
    config: Config,
    today: str,
    report: EnrichmentReport,
) -> None
```

**Flow**:
1. **Lines 197-199**: If slug already in graph, **falls back** to `_update_existing_entity()` -- but without passing `config` (see bug above).
2. **Lines 202-206**: Computes folder from entity type, average importance from observations (default 0.3 if no observations).
3. **Lines 208-220**: Builds `EntityFrontmatter` with `retention="short_term"`, `score=0.0`, `frequency=1`, tags aggregated from all observations (deduplicated via set).
4. **Lines 222-225**: Builds observation dicts -- note: **drops `tags`** from individual observations (unlike `_update_existing_entity` which keeps them). Tags only go to frontmatter.
5. **Line 228**: Calls `create_entity()`.
6. **Lines 231-245**: Adds `GraphEntity` to graph.

**Bug -- Tags lost on creation**: At line 222-225, the observation dicts for `create_entity()` do not include `"tags"`. This means when the observation is formatted via `_format_observation()`, the `obs.get("tags")` at store.py:413 returns None and no `#tag` suffixes are written. Tags are only stored in frontmatter. Compare with `_update_existing_entity()` (line 159-163) which **does** include `"tags"` in the observation dicts.

### 1.4 `_find_entity_slug()`

**Signature** (line 250):
```python
def _find_entity_slug(name: str, graph) -> str | None
```

**Flow**: Tries exact slug match, then case-insensitive title match, then case-insensitive alias match. Returns first match or None.

**Note**: Does NOT use FAISS -- this is a lighter lookup than the full resolution pipeline. This means relation targets can fail to resolve even if the entity exists under a slightly different name that FAISS would catch.

---

## 2. Entity Store (`src/memory/store.py`)

### 2.1 `read_entity()`

**Signature** (line 23):
```python
def read_entity(filepath: Path) -> tuple[EntityFrontmatter, dict[str, list[str]]]
```

**Flow**:
1. Reads entire file as UTF-8.
2. Delegates frontmatter parsing to `_parse_frontmatter()` -> `_shared_parse_frontmatter()` (from `core/utils.py`).
3. Validates frontmatter via `EntityFrontmatter.model_validate(fm_data)` -- Pydantic v2 strict validation. Will **raise** on invalid `type`, `retention`, etc.
4. Parses body into sections via `_parse_sections()`.

**Returns**: `(EntityFrontmatter, {"Facts": [...], "Relations": [...], "History": [...]})`

### 2.2 `write_entity()`

**Signature** (line 35):
```python
def write_entity(filepath: Path, frontmatter: EntityFrontmatter, sections: dict[str, list[str]]) -> None
```

**Flow**:
1. Creates parent directories.
2. Dumps frontmatter via `model_dump()` -> `yaml.safe_dump()` (no flow style, unicode allowed, keys NOT sorted).
3. Writes: `---\n{yaml}\n---\n\n# {title}\n\n` then for each section in **fixed order** `["Facts", "Relations", "History"]`: `## {name}\n{items}\n\n`.
4. Items that are whitespace-only are skipped (line 50: `if item.strip()`).

**Note**: Any custom sections in the original file are **silently dropped** -- only Facts, Relations, History survive a read-write cycle.

### 2.3 `update_entity()`

**Signature** (line 57):
```python
def update_entity(
    filepath: Path,
    new_observations: list[dict[str, str]] | None = None,
    new_relations: list[str] | None = None,
    frequency_increment: int = 1,
    last_mentioned: str | None = None,
    max_facts: int | None = None,
) -> EntityFrontmatter
```

**Flow**:
1. **Line 69**: Reads current entity (full read + parse).
2. **Lines 72-89 -- Observation dedup + append**:
   - For each new observation, formats it via `_format_observation()`, then checks `_is_duplicate_observation()` against existing facts.
   - If not duplicate, appends to `existing_facts`.
   - **Hard cap safety net** (lines 79-88): If `max_facts` is set and live fact count exceeds `max_facts * 2`, truncates to the last `max_facts * 2` live facts (most recently added) plus all superseded facts. Logs a warning.
3. **Lines 92-97 -- Relation dedup + append**:
   - For each new relation string, checks `if rel not in existing_rels` (exact string match).
   - If not present, appends. **Relations are never removed.**
4. **Lines 100-101**: Bumps `frontmatter.frequency` by `frequency_increment` (default 1).
5. **Line 104**: Calls `write_entity()` to persist.

**Double frequency bump bug**: `update_entity()` increments `frequency` by 1 at line 100. But `_update_existing_entity()` in enricher.py also increments `graph.entities[entity_id].frequency` at line 169. These are **different objects** -- the frontmatter in the MD file and the graph entity in memory. The MD file gets `+1` from `update_entity()`, and the graph gets `+1` from enricher.py. On next `rebuild_from_md()`, the MD file's value wins. So effectively the frequency is incremented once per update (from the MD), but the in-memory graph also gets a separate `+1` that is transient. **This is correct** because the graph is saved from its in-memory state (not re-read from MD), but the two increments happen independently and could diverge if the code path changes.

### 2.4 `create_entity()`

**Signature** (line 108):
```python
def create_entity(
    memory_path: Path,
    folder: str,
    slug: str,
    frontmatter: EntityFrontmatter,
    observations: list[dict[str, str]] | None = None,
    relations: list[str] | None = None,
) -> Path
```

**Flow**:
1. Computes filepath: `memory_path / folder / slug.md`.
2. Initializes sections with empty Facts, empty Relations, and History containing creation date.
3. Formats each observation via `_format_observation()` and appends to Facts.
4. If `relations` provided, sets the Relations section.
5. Calls `write_entity()`.
6. **No dedup check**: Does not check if file already exists. Will silently **overwrite** an existing entity file.

### 2.5 `create_stub_entity()`

**Signature** (line 136):
```python
def create_stub_entity(
    memory_path: Path,
    folder: str,
    slug: str,
    title: str,
    entity_type: str,
    today: str,
) -> Path
```

**Flow**: Creates a minimal entity with `retention="short_term"`, `importance=0.3`, `frequency=1`, empty facts/relations, history noting "Created by forward reference". Calls `write_entity()`.

**Note**: `entity_type` is `str`, not `EntityType`. Pydantic validation in `EntityFrontmatter` will reject invalid types at runtime.

### 2.6 `consolidate_entity_facts()`

**Signature** (line 184):
```python
def consolidate_entity_facts(
    filepath: Path,
    config,             # untyped, actually Config
    max_facts: int | None = None,
) -> dict
```

**Flow**:
1. Reads entity. Separates live facts from superseded facts.
2. If fewer than 3 live facts, returns early (nothing to consolidate).
3. Builds indexed text (`"0: fact_line\n1: fact_line\n..."`) for LLM.
4. Calls `call_fact_consolidation()` with `effective_max = max_facts or 50`.
5. **Lines 218-230 -- Guard rails applied here**:
   - **150-char cap**: If `cf.content` exceeds 150 chars, truncates to 147 + `"..."` (line 221-222).
   - **3-tag cap**: `cf.tags[:3]` (line 228).
6. Formats each consolidated fact via `_format_observation()`.
7. Appends preserved superseded facts.
8. Logs consolidation in History.
9. Writes entity back.

**Important**: The 150-char and 3-tag caps are **only enforced during consolidation**. Normal observation creation via `_format_observation()` has **no length or tag cap**. An LLM extraction can produce arbitrarily long content or many tags.

### 2.7 `mark_observation_superseded()`

**Signature** (line 487):
```python
def mark_observation_superseded(
    existing_facts: list[str],
    category: str,
    supersedes_text: str,
) -> list[str]
```

**Flow**:
1. Iterates all existing fact lines.
2. Parses each via `_parse_observation()`.
3. If the parsed fact: is not already superseded, matches the same `category`, and its content contains `supersedes_text` (case-insensitive substring match):
   - Sets `obs["superseded"] = True`
   - Re-formats via `_format_observation()` (wraps content in `~~strikethrough~~`, appends `[superseded]`)
4. Returns the updated list.

**Edge case**: The substring match (`supersedes_lower in obs["content"].lower()`) is very loose. A supersedes_text of `"a"` would match nearly every fact. The LLM must produce precise supersedes strings for this to work correctly.

**Edge case**: If multiple facts match the same category + substring, ALL of them get superseded. There's no "first match only" guard.

---

## 3. Observation Format Parser/Formatter

### 3.1 `_format_observation()` (store.py:394)

**Input**: `obs: dict` with keys `category`, `content`, `date` (optional), `valence` (optional), `tags` (optional), `superseded` (optional bool).

**Output format**: `- [category] (date) content [+/-/~] #tag1 #tag2 [superseded]`

**Logic**:
1. If `superseded`, wraps content in `~~...~~`.
2. Starts with `- [{category}]`.
3. Appends `({date})` if date is truthy.
4. Appends content.
5. Appends valence marker if valence is in `_VALENCE_MARKERS` (`positive`->`[+]`, `negative`->`[-]`, `neutral`->`[~]`).
6. Appends tags with `#` prefix (skips adding `#` if tag already starts with it).
7. Appends `[superseded]` if superseded.

### 3.2 `_parse_observation()` (store.py:424)

**Input**: A markdown line string.

**Regex** (line 429): `r"- \[(\w+)\]\s*(?:\(([^)]+)\)\s*)?(.+)"`
- Group 1: category (word chars only)
- Group 2: date (anything inside parens, optional)
- Group 3: rest (everything after)

**Logic**:
1. Detects `[superseded]` in rest, strips it.
2. Extracts valence by looking for ` [+]`, ` [-]`, ` [~]` in rest (first match wins, replaces only first occurrence).
3. Extracts tags via `re.findall(r"#(\S+)", rest)`.
4. Strips tags from content via `re.sub(r"\s*#\S+", "", rest)`.
5. If superseded, strips `~` from content edges.

**Bug -- Valence marker ambiguity**: The parser looks for ` [+]` etc. anywhere in the content string. If the actual content text contains the literal string ` [+]` (e.g., `"Blood type is O [+]"`), it would be incorrectly interpreted as a positive valence marker. The marker is not anchored to end-of-content.

**Bug -- Strikethrough stripping is greedy**: Line 455 uses `content.strip("~")`. The `strip()` method removes ALL leading and trailing `~` characters, not just the `~~` pair. Content like `"~~~test~~~"` would become `"test"` instead of `"~test~"`.

**Bug -- Tag extraction can capture valence markers**: If the valence marker extraction fails to match (e.g., due to extra whitespace), and the content contains `[+]`, the `#` in tag extraction won't interfere. However, the regex `#(\S+)` will match things like `#tag[+]` as a single tag `tag[+]` if they're adjacent without space.

### 3.3 `_is_duplicate_observation()` (store.py:464)

**Input**: `new_line: str`, `existing_lines: list[str]`

**Logic**:
1. Parses the new line.
2. For each existing line: parses it, skips superseded lines, compares category (exact) and content (case-insensitive **substring** in either direction).
3. Returns True if `new_content in ex_content or ex_content in new_content`.

**Edge case -- Substring dedup is asymmetric and loose**: A new fact `"back pain"` would match existing `"chronic back pain"` (substring). But also, a new fact `"chronic lower back pain and sciatica"` would NOT match `"back pain"` because neither is a substring of the other. This can lead to near-duplicates accumulating.

### 3.4 `_parse_sections()` (store.py:373)

**Logic**:
1. Splits body by newlines.
2. Lines starting with `## ` start a new section.
3. Non-empty lines (after strip) are appended to the current section.
4. Lines starting with `# ` (title) are skipped.

**Bug -- Leading whitespace stripped from section content**: Line 386 does `sections[current_section].append(line.strip())`. This means indented content (e.g., nested lists, code blocks) loses its indentation. A fact like `  - [fact] nested item` becomes `- [fact] nested item`.

---

## 4. Relation Handling -- Why Relations Are Never Deleted

### 4.1 The Append-Only Pattern

Relations exist in two stores:

**Store A -- `_graph.json`** (`GraphData.relations: list[GraphRelation]`):
- Added via `graph.add_relation()` (graph.py:96-116)
- Dedup: exact match on `(from_entity, to_entity, type)` tuple -> Hebbian reinforcement instead of duplicate
- **No delete function exists** in `graph.py` except `remove_orphan_relations()` (line 119) which only removes relations pointing to non-existent entities
- The list only grows (or shrinks when entities are deleted)

**Store B -- Entity `.md` files** (Relations section):
- Added via `update_entity(filepath, new_relations=[rel_line])` (store.py:92-97)
- Dedup: exact string match `if rel not in existing_rels` (line 95)
- **No delete path**: `update_entity()` only appends. `write_entity()` writes whatever is in the sections dict, but nothing ever removes items from the Relations list.

### 4.2 Tracing the Code Path

When a relation is processed in `enrich_memory()` (enricher.py:62-95):

```
enricher.py:84  -> graph.add_relation(graph, graph_rel, ...)   # Store A: append or reinforce
enricher.py:93  -> update_entity(entity_file, new_relations=[rel_line])  # Store B: append
```

Neither path has any concept of removing or superseding a relation. Compare with observations:
- Observations have `supersedes` field on `RawObservation` (models.py:39)
- Observations have `mark_observation_superseded()` (store.py:487)
- Observations have `[superseded]` marker in the formatted text

**Relations have none of these mechanisms.**

### 4.3 The Only "Deletion" Paths

1. **`remove_orphan_relations()`** (graph.py:119-125): Removes relations where either entity no longer exists in the graph. Called during `rebuild_from_md()` implicitly (relations pointing to missing entities are simply not rebuilt).

2. **Dream mode dead pruning** (pipeline/dream.py): Archives low-score entities. When an entity is archived, `remove_orphan_relations()` cleans up its relations from the graph. But the archived entity's `.md` file still contains the relation text in its Relations section.

3. **Manual editing**: A user can manually edit the `.md` file to remove relation lines. On next `rebuild_from_md()`, those relations will disappear from the graph.

### 4.4 Where Relation Deletion Could Be Inserted

**Option A -- Model-level**: Add a `supersedes_relation` field to `RawRelation` (like `RawObservation.supersedes`), and a `mark_relation_superseded()` function in `store.py` mirroring the observation pattern.

**Option B -- Graph-level**: Add a `remove_relation(graph, from_entity, to_entity, type)` function to `graph.py`:
```python
def remove_relation(graph: GraphData, from_entity: str, to_entity: str, rel_type: str) -> GraphData:
    graph.relations = [
        r for r in graph.relations
        if not (r.from_entity == from_entity and r.to_entity == to_entity and r.type == rel_type)
    ]
    return graph
```

**Option C -- LTD threshold**: In `scoring.py`'s `_apply_ltd()`, when a relation's strength decays below a threshold (e.g., 0.1), auto-remove it from both graph and MD files.

**Option D -- Dream mode step**: Add a relation pruning step that removes relations with `strength < threshold` and `days_since_reinforced > N`.

The most natural insertion point is **Option C** combined with **Option B**, since LTD already weakens relations over time but never actually removes them.

---

## 5. Fact Supersession vs. Relation Supersession

### What Exists for Facts

| Mechanism | Where | How |
|-----------|-------|-----|
| `RawObservation.supersedes` | models.py:39 | LLM sets a text describing the old fact |
| `mark_observation_superseded()` | store.py:487 | Finds matching fact by category + substring, marks `[superseded]` |
| Superseded facts preserved | store.py:200-201 | Filtered out of consolidation, kept in file |
| Dedup check skips superseded | store.py:479 | `_is_duplicate_observation()` ignores superseded lines |
| Strikethrough formatting | store.py:403-404 | Content wrapped in `~~...~~` for visual indication |

### What's Missing for Relations

| Mechanism | Status |
|-----------|--------|
| `RawRelation.supersedes` | Does not exist |
| `mark_relation_superseded()` | Does not exist |
| Relation removal from graph | Only orphan cleanup |
| Relation removal from MD | Not implemented |
| Relation strength decay to removal | LTD decays strength but never removes |
| Contradictory relation detection | Not implemented (e.g., `improves` vs `worsens` for same pair) |

---

## 6. Guard Rails

### 6.1 The 150-Character Cap

**Where**: `consolidate_entity_facts()` at store.py:221-222.
```python
if len(content) > 150:
    content = content[:147] + "..."
```

**Scope**: Only applied to LLM-consolidated facts. Normal extraction output has **no length limit**. An LLM could produce a 500-character observation content and it would be stored verbatim.

**Missing enforcement points**:
- `_format_observation()` (store.py:394) -- no length check
- `_update_existing_entity()` (enricher.py:159-163) -- no length check
- `_create_new_entity()` (enricher.py:222-225) -- no length check

### 6.2 The 3-Tag Cap

**Where**: `consolidate_entity_facts()` at store.py:228.
```python
"tags": cf.tags[:3],
```

**Scope**: Only applied to LLM-consolidated facts. Normal extraction output has **no tag limit**. The `RawObservation` model has `tags: list[str]` with no max length constraint.

**Missing enforcement points**: Same as above -- no cap in `_format_observation()` or the enricher's observation dict construction.

---

## 7. Bugs and Edge Cases Summary

### 7.1 Confirmed Bugs

| # | Location | Severity | Description |
|---|----------|----------|-------------|
| B1 | enricher.py:199 | Medium | `_create_new_entity()` falls back to `_update_existing_entity()` without passing `config`, bypassing `max_facts` enforcement |
| B2 | enricher.py:222-225 | Low | Tags dropped from individual observations on entity creation (only stored in frontmatter) |
| B3 | store.py:386 | Low | `_parse_sections()` strips leading whitespace, destroying indentation |
| B4 | store.py:455 | Low | `content.strip("~")` is greedy -- strips all leading/trailing tildes, not just the `~~` pair |
| B5 | store.py:443-446 | Low | Valence marker detection not anchored to end of content; literal `[+]` in content text triggers false match |

### 7.2 Design Gaps

| # | Location | Description |
|---|----------|-------------|
| D1 | store.py + graph.py | No relation deletion mechanism (append-only in both stores) |
| D2 | store.py + enricher.py | 150-char and 3-tag caps only enforced during consolidation, not during normal extraction |
| D3 | enricher.py:62-95 | Relation data stored in two places (graph JSON + entity MD) with no consistency guarantee |
| D4 | store.py:482 | Substring dedup is loose -- neither `"back pain" in "lower back pain with sciatica"` nor the reverse catches all near-duplicates |
| D5 | store.py:497-503 | Supersession substring match is loose -- short `supersedes` strings can match unintended facts |
| D6 | store.py:499-505 | Multiple facts can be superseded by a single supersedes_text (no "first match" limit) |
| D7 | enricher.py:131-166 | File read up to 3 times in `_update_existing_entity()` (supersession, pre-consolidation check, update_entity) |

### 7.3 Robustness Observations

- **`create_entity()` overwrites silently** (store.py:117): No existence check. If called twice with the same slug, the second call overwrites the first. The enricher guards against this at line 197, but other callers (e.g., dream mode, manual scripts) could trigger data loss.
- **`_find_entity_slug()` is weaker than full resolution** (enricher.py:250-264): Uses only exact slug, title, and alias matching -- no FAISS. Relation targets that are close but not exact matches will fail to resolve, creating unnecessary stub entities.
- **Frontmatter `score` round-trip**: Scores are written to MD files via `write_entity()` but are always recalculated from graph state. The MD score is informational only and can be stale.
- **`write_entity()` drops unknown sections** (store.py:46): Only `Facts`, `Relations`, `History` are preserved. Any custom sections (e.g., `## Notes`) added manually will be lost on next write.

---

## 8. Function Reference Table

### enricher.py

| Function | Line | Signature | Purpose |
|----------|------|-----------|---------|
| `enrich_memory` | 23 | `(resolved: ResolvedExtraction, config: Config, today: str \| None) -> EnrichmentReport` | Main entry: apply extraction to MD files + graph |
| `_update_existing_entity` | 107 | `(entity_id, raw_entity, graph, memory_path, today, report, config)` | Update existing entity with new observations |
| `_create_new_entity` | 187 | `(slug, raw_entity, graph, memory_path, config, today, report)` | Create new entity MD + graph entry |
| `_find_entity_slug` | 250 | `(name: str, graph) -> str \| None` | Resolve name to entity slug via exact/alias match |

### store.py

| Function | Line | Signature | Purpose |
|----------|------|-----------|---------|
| `init_memory_structure` | 16 | `(memory_path: Path) -> None` | Create folder structure |
| `read_entity` | 23 | `(filepath: Path) -> (EntityFrontmatter, dict)` | Read MD into structured data |
| `write_entity` | 35 | `(filepath, frontmatter, sections) -> None` | Write structured data to MD |
| `update_entity` | 57 | `(filepath, new_observations?, new_relations?, frequency_increment?, last_mentioned?, max_facts?) -> EntityFrontmatter` | Append observations/relations, bump frequency |
| `create_entity` | 108 | `(memory_path, folder, slug, frontmatter, observations?, relations?) -> Path` | Create new entity MD file |
| `create_stub_entity` | 136 | `(memory_path, folder, slug, title, entity_type, today) -> Path` | Create minimal forward-reference entity |
| `list_entities` | 167 | `(base_path: Path) -> list[dict]` | List all entity files with frontmatter |
| `consolidate_entity_facts` | 184 | `(filepath, config, max_facts?) -> dict` | LLM-based fact consolidation with guard rails |
| `save_chat` | 256 | `(messages: list[dict], memory_path) -> Path` | Save chat with `processed: false` |
| `list_unprocessed_chats` | 289 | `(memory_path) -> list[Path]` | Find unprocessed chat files |
| `mark_chat_processed` | 304 | `(filepath, entities_updated, entities_created) -> None` | Set chat as processed |
| `mark_chat_fallback` | 322 | `(filepath, fallback, error?) -> None` | Mark chat processed via fallback |
| `increment_extraction_retries` | 346 | `(filepath) -> int` | Bump retry counter in chat frontmatter |
| `get_chat_content` | 359 | `(filepath) -> str` | Read chat body without frontmatter |
| `_parse_frontmatter` | 368 | `(text) -> (dict, str)` | Delegate to shared YAML parser |
| `_parse_sections` | 373 | `(body) -> dict[str, list[str]]` | Parse `## Section` blocks from body |
| `_format_observation` | 394 | `(obs: dict) -> str` | Dict to markdown fact line |
| `_parse_observation` | 424 | `(line: str) -> dict \| None` | Markdown fact line to dict |
| `_is_duplicate_observation` | 464 | `(new_line, existing_lines) -> bool` | Check category + substring content match |
| `mark_observation_superseded` | 487 | `(existing_facts, category, supersedes_text) -> list[str]` | Find and mark matching fact as superseded |
