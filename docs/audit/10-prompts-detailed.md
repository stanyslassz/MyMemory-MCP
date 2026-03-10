# Prompt Files Deep Audit

Audit date: 2026-03-10
Scope: All 13 files in `prompts/` directory

---

## Table of Contents

1. [extract_facts.md](#1-extract_factsmd)
2. [arbitrate_entity.md](#2-arbitrate_entitymd)
3. [consolidate.md](#3-consolidatemd)
4. [consolidate_facts.md](#4-consolidate_factsmd)
5. [context_template.md](#5-context_templatemd)
6. [context_section.md](#6-context_sectionmd)
7. [context_instructions.md](#7-context_instructionsmd)
8. [generate_context.md](#8-generate_contextmd)
9. [summarize_entity.md](#9-summarize_entitymd)
10. [discover_relations.md](#10-discover_relationsmd)
11. [dream_plan.md](#11-dream_planmd)
12. [dream_validate.md](#12-dream_validatemd)
13. [extract_relations.md](#13-extract_relationsmd)
14. [Comparison Table](#comparison-table)
15. [Cross-Cutting Issues](#cross-cutting-issues)

---

## 1. extract_facts.md

### Role in Pipeline
Step 1 — the most critical prompt in the entire system. Every piece of knowledge enters via this prompt. Called by `call_extraction()` in `llm.py` which uses `_call_with_stall_detection()` (streaming with watchdog).

### Caller
`src/core/llm.py:call_extraction()` -> `_call_with_stall_detection()` with `RawExtraction` response model.

### Variable Injections

| Variable | Source | Notes |
|----------|--------|-------|
| `{chat_content}` | Passed from `call_extraction(chat_content, config)` | Raw chat transcript |
| `{user_language}` | Auto-injected by `load_prompt()` from `config.user_language` | e.g. "fr" |
| `{categories_observations}` | Auto-injected from `config.categories.observations` | 17 literal values |
| `{categories_entity_types}` | Auto-injected from `config.categories.entity_types` | 9 literal values |
| `{categories_relation_types}` | Auto-injected from `config.categories.relation_types` | 13 literal values |
| `{json_schema}` | Injected in `call_extraction()` via `RawExtraction.model_json_schema()` | **Note: the prompt does NOT actually contain `{json_schema}` — the schema variable is passed to `load_prompt()` but never referenced in the template. The prompt uses an inline example instead.** |

### Expected Output Format
JSON matching `RawExtraction` Pydantic model:
```json
{
  "entities": [{"name": str, "type": EntityType, "observations": [{"category": ObservationCategory, "content": str, "importance": float, "tags": [str], "date": str, "valence": str, "supersedes": str}]}],
  "relations": [{"from_name": str, "to_name": str, "type": RelationType, "context": str}],
  "summary": str
}
```

### Line-by-Line Analysis

**Lines 1-4 (System section):** Clean role definition. "Do NOT extract small talk" is a critical guardrail.

**Lines 6-38 (Rules):** Dense rule block — 15 distinct rules in a single enumeration. This is the core complexity problem:
- Rules 1-5: Basic constraints (stay faithful, importance scale, allowed categories)
- Rule 6: Language preservation
- Rules 7-8: Date and valence extraction (two separate cognitive tasks)
- Rule 9: Supersedes logic — the most complex rule (requires LLM to detect contradictions and reference old facts)
- Rules 10-11: AI Personality extraction — entity routing for `ai_self` type, with a cap of 3 observations
- Rule 12: Entity type routing — "don't create person for user" — complex negative constraint
- Rule 13: Medical position guardrail — negative example ("NOT separate project entities")
- Rule 14: Length constraint (120 chars)
- Rule 15: JSON-only output

**Lines 40-77 (User section):** Chat injection + JSON example. The example is in French, hard-coded (Marie, Airbus). The final instruction "Now extract..." is a strong closing anchor.

### Complexity Assessment for Qwen3-30B: HIGH

This prompt demands 6+ simultaneous cognitive tasks:
1. Entity extraction (NER)
2. Observation categorization (17 categories)
3. Importance scoring (continuous float)
4. Valence detection (sentiment analysis)
5. Date extraction (temporal reasoning)
6. Supersedes detection (contradiction detection against unseen prior state)
7. Entity type routing (complex conditional: user-self vs person vs health vs work)
8. Relation extraction (typed, directed)
9. AI personality extraction (meta-cognitive, capped)

### Failure Modes with Small Models

1. **Category hallucination**: Small models invent categories not in the allowed list. Mitigated post-hoc by `sanitize_extraction()` which fuzzy-maps invalid types, but this is a lossy fallback.
2. **Supersedes field misuse**: Models either ignore it entirely or hallucinate non-existent prior facts. This field requires knowledge the model doesn't have (what was previously stored).
3. **Importance score clustering**: Small models tend to cluster all importance values around 0.5-0.7, losing the discriminative power of the 0.1-0.9 range.
4. **Entity type confusion**: The "don't create person for user" rule is a complex negative constraint. Small models frequently create `person` entities for the user anyway, or route health facts to `interest`.
5. **Observation overflow**: No hard cap on observations per entity in the prompt. Small models may emit 20+ observations for a single entity, many near-duplicates.
6. **JSON structure errors**: Deep nested JSON (entity -> observations array -> each with 7 fields) is error-prone. The `json_repair` context manager mitigates but can't fix structural omissions.
7. **French content leaking into keys**: Small models sometimes emit French category names ("fait" instead of "fact").
8. **Relation type invention**: Models propose `prescrit_par` or `travaille_avec` instead of allowed types. `sanitize_extraction()` handles this via fuzzy mapping.

### Improvement Suggestions

1. **Split into two passes**: Pass 1 extracts entities + observations only. Pass 2 (given entities) extracts relations. This halves the cognitive load per call. The existing `extract_relations.md` placeholder was designed for this.
2. **Remove the supersedes field from the prompt**: It requires knowledge the model doesn't have. Contradiction detection should be a post-extraction step comparing against existing entity facts.
3. **Add explicit observation count limit**: "Extract at most 5 observations per entity" would prevent observation flooding.
4. **Provide the JSON schema explicitly**: The `{json_schema}` variable is computed in `call_extraction()` but never referenced in the prompt template. Either add `{json_schema}` to the prompt or remove the unused variable.
5. **Simplify the example**: The example embeds both entities and relations. Two separate smaller examples would be clearer.
6. **Consider structured extraction via function calling**: Instead of prompt-based JSON generation, use tool/function calling mode where available. This is more reliable for complex schemas.

### Could Be Split?
Yes — into entity extraction + relation extraction (two calls). The `extract_relations.md` placeholder already exists for this.

### Could Be Deterministic?
No. Entity extraction from natural language fundamentally requires LLM understanding. However, date extraction and valence detection could be moved to deterministic post-processing (regex for dates, sentiment lexicon for valence).

---

## 2. arbitrate_entity.md

### Role in Pipeline
Step 3 — resolves ambiguous entity matches. Only called when FAISS returns multiple plausible candidates for a new mention. Called by `call_arbitration()` in `llm.py` via `_call_structured()` (non-streaming).

### Caller
`src/core/llm.py:call_arbitration()` -> `_call_structured()` with `EntityResolution` response model.

### Variable Injections

| Variable | Source | Notes |
|----------|--------|-------|
| `{entity_name}` | From `call_arbitration(entity_name, ...)` | The new mention to resolve |
| `{entity_context}` | From `call_arbitration(..., entity_context, ...)` | Context of appearance (first observation) |
| `{candidates}` | Built in `call_arbitration()` as formatted string | List of existing entities with id, title, type, aliases |
| `{json_schema}` | `EntityResolution.model_json_schema()` | Schema for `{"action": "existing"|"new", ...}` |
| `{categories_entity_types}` | Auto-injected | Allowed entity types |
| `{user_language}` | Auto-injected | Not actually used in template text |
| `{categories_observations}` | Auto-injected | Not used in template |
| `{categories_relation_types}` | Auto-injected | Not used in template |

### Expected Output Format
```json
{"action": "existing", "existing_id": "slug-of-entity"}
// or
{"action": "new", "new_type": "health"}
```

### Line-by-Line Analysis

**Lines 1-4 (System):** Clear role statement. "Respond ONLY with valid JSON" is a good anchor.

**Lines 6-21 (User):** Minimal structure. Presents the mention name, context, candidates list, then asks the binary question. The JSON schema is injected at the end.

### Complexity Assessment for Qwen3-30B: LOW

This is the simplest LLM prompt in the system. Binary choice (existing vs new) with clear candidates. The context is minimal and the output schema is tiny (2-3 fields).

### Failure Modes with Small Models

1. **Always choosing "new"**: Conservative models default to "new" when unsure, defeating the purpose of resolution.
2. **Inventing non-existent `existing_id`**: Model hallucinates an ID not in the candidates list. The resolver should validate this but the prompt doesn't explicitly warn against it.
3. **Ignoring aliases**: The prompt presents aliases in the candidates but doesn't emphasize they are alternative names.

### Improvement Suggestions

1. **Add explicit instruction**: "The existing_id MUST be one of the IDs listed above" to prevent hallucinated IDs.
2. **Remove unused auto-injected variables**: `{user_language}`, `{categories_observations}`, `{categories_relation_types}` are injected but never referenced. Harmless but wastes tokens if the `load_prompt()` function doesn't strip unreferenced variables.
3. **Add a confidence threshold instruction**: "Choose 'existing' only if you are confident the mention refers to that entity. When in doubt, choose 'new'."

### Could Be Split?
No. Already atomic.

### Could Be Deterministic?
Partially. Many ambiguous cases could be resolved with better FAISS thresholds, alias normalization, or string similarity scoring. The LLM is needed only for genuinely ambiguous semantic cases (e.g., "Marie" could be a person or a place).

---

## 3. consolidate.md

### Role in Pipeline
**UNUSED.** This prompt is not loaded by any code in the codebase. No call to `load_prompt("consolidate", ...)` exists anywhere. The prompt was presumably intended for entity-level duplicate detection (as opposed to fact-level consolidation in `consolidate_facts.md`).

### Caller
None. Orphan file.

### Variable Injections

| Variable | Source | Notes |
|----------|--------|-------|
| `{entities_list}` | N/A | Would contain entity names/types |
| `{json_schema}` | N/A | Would contain duplicate detection schema |
| `{user_language}` | Auto-injected | Not referenced in template |
| `{categories_*}` | Auto-injected | Not referenced in template |

### Expected Output Format
Presumably JSON with pairs of duplicate entity IDs, but no Pydantic model exists for this.

### Line-by-Line Analysis

**Lines 1-7 (System):** Defines duplicate detection role. "Two entities are duplicates if they refer to the same concept, person, or thing under different names." This is a reasonable definition.

**Lines 9-17 (User):** Minimal — just injects entity list and asks for JSON response.

### Complexity Assessment: MEDIUM (if it were used)

Duplicate detection across a list of entities requires comparing all pairs, which is O(n^2) cognitive load. For large entity lists, this could overwhelm small models.

### Improvement Suggestions

1. **Delete or implement**: Either wire this into the CLI `consolidate` command or delete it. Currently it's dead code.
2. **The dream mode step 4** (entity merging) uses deterministic slug/alias overlap instead of LLM — this prompt was likely abandoned in favor of that approach.

### Could Be Deterministic?
Yes — and it already is. Dream mode step 4 uses slug/alias overlap detection without LLM. This prompt is obsolete.

---

## 4. consolidate_facts.md

### Role in Pipeline
Fact consolidation — merges redundant observations within a single entity. Called during dream mode step 3 and auto-consolidation in `memory run`. Used when an entity exceeds `max_facts` threshold.

### Caller
`src/core/llm.py:call_fact_consolidation()` -> `_call_structured()` with `FactConsolidation` response model.

### Variable Injections

| Variable | Source | Notes |
|----------|--------|-------|
| `{entity_title}` | From `call_fact_consolidation(entity_title, ...)` | Entity name |
| `{entity_type}` | From `call_fact_consolidation(..., entity_type, ...)` | e.g. "health", "ai_self" |
| `{facts_text}` | Indexed list of observations | Format: `0: [category] (date) content [valence] #tags` |
| `{max_facts}` | From `call_fact_consolidation(..., max_facts=50)` | Target count after consolidation |
| `{user_language}` | Auto-injected | Referenced in rules |
| `{categories_*}` | Auto-injected | Not referenced |

### Expected Output Format
```json
{
  "consolidated": [
    {
      "category": "ai_style",
      "content": "Merged observation text",
      "date": "2026-03",
      "valence": "",
      "tags": ["tag1"],
      "replaces_indices": [0, 2, 5]
    }
  ]
}
```

### Line-by-Line Analysis

**Lines 1-5 (System):** Clean role definition. "merge semantically redundant ones" is the core task.

**Lines 7-28 (Rules):** 14 rules — this is a dense prompt:
- Rule 1: Merge semantically redundant → one line
- Rule 2: MAX 120 characters — hard constraint
- Rule 3: "Do NOT concatenate with parentheses" — negative constraint from observed failure mode
- Rule 4: Preserve most recent date
- Rule 5: Preserve strongest valence (with specific ordering)
- Rule 6: Merge tags (max 3)
- Rule 7: No invention
- Rule 8: Keep distinct facts unchanged
- Rule 9: `replaces_indices` tracking — this is the most cognitively demanding requirement
- Rule 10: Solo facts must still appear with their index
- Rule 11: Hard cap at `{max_facts}` — forces ruthless prioritization
- Rule 12: Prioritization heuristic (importance > recency > uniqueness)
- Rule 13: Special rule for `ai_self` entities
- Rule 14: Language constraint

**Lines 30-54 (User section):** Entity info, indexed facts, response format with example.

### Complexity Assessment for Qwen3-30B: HIGH

This prompt requires:
1. Semantic similarity comparison across all fact pairs
2. Merging text while respecting length constraints
3. Tracking which original indices map to which output (bookkeeping)
4. Prioritization decisions when exceeding max_facts
5. Type-conditional behavior (ai_self special case)

### Failure Modes with Small Models

1. **Index tracking errors**: `replaces_indices` is frequently wrong — models skip indices, duplicate them across multiple consolidated facts, or leave gaps. This causes data loss when the caller uses indices to determine which originals to remove.
2. **Exceeding max_facts**: Models often produce more than `{max_facts}` entries, especially when most facts are distinct. The prompt says "ruthlessly prioritize" but small models are reluctant to discard.
3. **Concatenation instead of merging**: Despite rule 3, models concatenate with commas or semicolons, producing 200+ character lines.
4. **Category drift**: When merging two facts with different categories, the model may pick an incorrect category or invent one.
5. **ai_self special case ignored**: Small models don't reliably implement type-conditional behavior.
6. **Empty output**: Some models return `{"consolidated": []}` when overwhelmed.

### Improvement Suggestions

1. **Pre-cluster facts before sending to LLM**: Group facts by category + content similarity deterministically, then only ask the LLM to merge within each cluster. This reduces the comparison space from O(n^2) to O(k) per cluster.
2. **Remove index tracking from LLM**: Instead of asking the LLM to track `replaces_indices`, have the LLM output merged content only, and use post-processing (string similarity) to determine which originals were merged. Less error-prone.
3. **Hard-cap the number of input facts**: If an entity has 50 facts, don't send all 50. Pre-filter to the 30 most important/recent and discard the rest deterministically.
4. **Split ai_self handling**: Use a separate prompt or rule set for ai_self entities rather than embedding conditional logic in the main prompt.

### Could Be Split?
Yes — into (a) semantic clustering (deterministic via embeddings) + (b) within-cluster merging (LLM). This would dramatically reduce the cognitive load.

### Could Be Deterministic?
Partially. A deterministic pipeline could:
- Deduplicate exact/near-exact matches via string similarity
- Prune oldest low-importance facts
- Group by category
Only the semantic merging of similar-but-not-identical facts truly requires LLM.

---

## 5. context_template.md

### Role in Pipeline
Template for the deterministic `_context.md` generation. Not an LLM prompt — it's a string template with variable substitution. Used by `build_context()` and `build_context_with_llm()` in `context.py`.

### Caller
`src/memory/context.py:build_context()` — reads raw template, replaces variables via `str.replace()`. NOT processed through `load_prompt()`.

### Variable Injections

| Variable | Source | Notes |
|----------|--------|-------|
| `{date}` | `date.today().isoformat()` | ISO date string |
| `{user_language_name}` | `config.user_language_name` | e.g. "French" |
| `{ai_personality}` | Built from ai_self entities dossiers | Multi-line markdown |
| `{sections}` | Assembled from identity/work/personal/top/vigilance/history | Large block |
| `{available_entities}` | Pipe-separated entity list | Compact reference |
| `{custom_instructions}` | Content of `context_instructions.md` | User-editable |

### Line-by-Line Analysis

**Lines 1-8:** Header with date, role description, language instruction. Clean and functional.

**Lines 10-14:** AI personality section with `{ai_personality}` injection.

**Line 16:** `{sections}` — the bulk of the context. All entity dossiers go here.

**Lines 18-31:** "Available in memory" section listing entities not detailed above, plus RAG tool reference and custom instructions.

### Complexity Assessment: N/A (template, no LLM)

### Issues Found

1. **No `{user_language}` variable**: The template uses `{user_language_name}` (e.g. "French") but the `load_prompt()` auto-injects `{user_language}` (e.g. "fr"). Since this template is NOT processed through `load_prompt()`, this is fine — but it means `context.py` must handle the replacement directly, which it does.
2. **Hardcoded English section headers**: "Your personality & interaction style", "Available in memory", "Extended memory access" are all in English. The `{user_language_name}` instruction tells the receiving LLM to respond in the user's language, but the context structure itself is English. This is by design per CLAUDE.md ("context template stays in English").
3. **RAG tool reference**: Line 27 mentions `search_rag` tool. This couples the template to the MCP server's tool naming. If the tool name changes, this instruction becomes stale.

### Improvement Suggestions

1. **Make the `search_rag` tool name a variable**: `{rag_tool_name}` injected from config, so the template doesn't hardcode MCP tool names.
2. **Add section for entity count/stats**: A brief "Memory contains X entities, Y relations" line would help the consuming LLM understand memory scope.

### Could Be Deterministic?
Already is deterministic. This is a template, not an LLM prompt.

---

## 6. context_section.md

### Role in Pipeline
Per-section LLM cleanup — used when `context_llm_sections: true` is enabled. Each context section (identity, work, personal, top_of_mind) is individually cleaned/polished by the LLM. Called from `build_context_with_llm()`.

### Caller
`src/core/llm.py:call_context_section()` -> raw `litellm.completion()` (no Instructor, free-text output). Called by `src/memory/context.py:build_context_with_llm()._llm_section()`.

### Variable Injections

| Variable | Source | Notes |
|----------|--------|-------|
| `{section_name}` | Hardcoded per section call (e.g. "AI Personality & Interaction Style") | Section identifier |
| `{entities_dossier}` | Built by `_enrich_entity()` for each entity in section | Raw markdown dossier |
| `{rag_context}` | From `_rag_prefetch()` — FAISS search results for related context | "No additional context available." if empty |
| `{budget_tokens}` | Calculated from `config.context_budget` percentages | Integer token budget |
| `{user_language}` | Auto-injected | Referenced in rules |
| `{categories_*}` | Auto-injected | Not referenced |

### Expected Output Format
Free-text markdown — no JSON, no schema. The LLM outputs cleaned markdown sections directly.

### Line-by-Line Analysis

**Lines 1-6 (System):** Role as "memory context writer." The explicit merge instruction for duplicate entities is good — it handles a real problem (e.g., "Hernie discale L5-S1" and "dos (sciatique)" appearing as separate entries).

**Lines 7-18 (Rules):** 9 rules:
- Merge duplicates (with French medical example — leaks domain knowledge into prompt)
- Shorten facts > 120 chars
- Remove contradictory/redundant facts
- Correct type/tag errors (with examples — "medical position is not a project")
- Respect token budget
- Keep structured markdown format
- Never invent
- Keep in user_language
- Output ONLY markdown

**Lines 20-35 (Output format):** Explicit markdown structure with entity header, tags, facts grouped by category, relations. This is well-structured.

**Lines 37-49 (User section):** Section name, raw dossier, RAG context, final instruction.

### Complexity Assessment for Qwen3-30B: MEDIUM-HIGH

The prompt requires:
1. Entity deduplication (comparing entities within section)
2. Content editing (shortening, cleaning)
3. Format compliance (specific markdown structure)
4. Token budgeting (staying within budget)
5. Type correction (domain knowledge about what types map where)

### Failure Modes with Small Models

1. **Entity merging gone wrong**: Models merge entities that shouldn't be merged (false positive dedup).
2. **Token budget ignored**: Small models can't count tokens. They either produce too much or aggressively truncate.
3. **Format deviation**: Models often add explanatory text before/after the markdown, violating the "Output ONLY" instruction.
4. **Information invention**: Despite the "NEVER invent" rule, models synthesize connections not in the dossier, especially when RAG context provides tangentially related facts.
5. **Type overcorrection**: Models may change entity types that were actually correct, especially for edge cases.

### Improvement Suggestions

1. **Remove the entity merging responsibility**: This should be handled deterministically before the prompt is called. The dossier should already be deduplicated. Asking the LLM to merge entities is error-prone and the exact same operation is done deterministically in dream mode step 4.
2. **Remove the type correction responsibility**: Same reasoning — this belongs in a deterministic validation step.
3. **Reduce to "format and shorten only"**: The LLM should ONLY clean up formatting and shorten long facts. All structural decisions (merging, type correction) should be deterministic.
4. **Pre-compute token count of input**: If the input is already within budget, skip the LLM call entirely.

### Could Be Split?
Yes — dedup, type correction, and formatting are three distinct tasks that could be separate steps (first two deterministic, last one LLM or deterministic).

### Could Be Deterministic?
Mostly yes. The deterministic `build_context()` already produces good output. The LLM adds marginal polish. The main value-add (entity merging) should be deterministic anyway. This prompt could be eliminated if the deterministic pipeline had better dedup.

---

## 7. context_instructions.md

### Role in Pipeline
User-editable custom instructions injected into every `_context.md`. Not an LLM prompt — it's static content.

### Caller
`src/memory/context.py:build_context()` and `build_context_with_llm()` — read as raw text and injected into `{custom_instructions}` slot of `context_template.md`.

### Content
```
## Specific instructions

This section is for user-defined persistent instructions.
Edit this file to add rules the AI should always follow.
```

### Analysis
This is a placeholder file. Users are expected to edit it with their own rules. The default content is just a hint.

### Issues
1. **The default content gets injected into _context.md**: If the user doesn't edit it, every context file contains the placeholder text "This section is for user-defined persistent instructions. Edit this file to add rules..." This wastes tokens and looks unprofessional.

### Improvement Suggestions
1. **Make injection conditional**: In `context.py`, check if the content equals the default placeholder and skip injection if so. Or use a comment marker (e.g., `<!-- placeholder -->`) that the code detects.
2. **Move to a `.example` file**: Ship `context_instructions.md.example` and only inject if `context_instructions.md` (without `.example`) exists.

### Could Be Deterministic?
Already is — it's static content.

---

## 8. generate_context.md

### Role in Pipeline
Legacy narrative context generation. Used ONLY when `context_narrative: true` is set in config. Generates the entire `_context.md` as prose via a single LLM call. Replaced by the per-section approach (`context_section.md`).

### Caller
`src/core/llm.py:call_context_generation()` -> raw `litellm.completion()` (free-text output). Called by `src/memory/context.py:generate_context()`, which is called by `build_context_input()` chain. Only active when `context_narrative: true`.

### Variable Injections

| Variable | Source | Notes |
|----------|--------|-------|
| `{context_max_tokens}` | `config.context_max_tokens` | Total token budget |
| `{enriched_data}` | Built by `build_context_input()` — top 15 entities' dossiers | Large text block |
| `{context_budget}` | Formatted as `- section: X%` lines | Per-section percentage |
| `{date}` | `date.today().isoformat()` | For the header |
| `{user_language}` | Auto-injected | Referenced in rules |
| `{categories_*}` | Auto-injected | Not referenced |

### Expected Output Format
Free-text markdown. The LLM must produce a full `_context.md` starting with `# Contexte -- {date}`.

### Line-by-Line Analysis

**Lines 1-4 (System):** "You write a condensed memory context for an AI assistant." Clear role.

**Lines 6-17 (Rules):** 8 rules:
- Max token limit
- Follow section budgets
- Permanent entities MUST appear (strong constraint)
- Prioritize high scores
- Use relations for connections (with example)
- Fluid prose, not bullets (except Vigilances)
- Vigilances and Instructions IA sections
- Write ALL in user_language, headers too

**Lines 19-29 (User section):** Injects enriched dossier and budget, with specific output instruction.

### Complexity Assessment for Qwen3-30B: VERY HIGH

This prompt asks the LLM to:
1. Consume a large dossier (potentially thousands of tokens)
2. Rewrite it as coherent prose
3. Respect per-section token budgets
4. Ensure permanent entities always appear
5. Create meaningful narrative connections
6. Write headers in user_language (not English)
7. Stay within total token budget

### Failure Modes with Small Models

1. **Token budget completely ignored**: Small models cannot count tokens. They either produce 200 tokens or 5000.
2. **Section budget imbalance**: Models dump everything into the first section and truncate the rest.
3. **Permanent entity omission**: Models miss the MUST constraint and drop low-interest permanent entities.
4. **Prose quality**: Small models produce stilted, repetitive prose rather than fluid narrative.
5. **Language inconsistency**: Headers in French, body in English, or vice versa.
6. **Hallucinated connections**: "Back pain improving due to swimming" when no such relation exists.

### Improvement Suggestions

1. **Deprecate officially**: This prompt is documented as "legacy, replaced by per-section" in CLAUDE.md. Add a deprecation warning in the code.
2. **If keeping**: Split into per-section calls (already done in `context_section.md`). The monolithic approach is fundamentally unsuitable for small models.

### Could Be Split?
Already has been — `context_section.md` is the split replacement.

### Could Be Deterministic?
The deterministic `build_context()` already replaces this for most users. This prompt is only needed if prose narrative style is desired over structured dossiers.

---

## 9. summarize_entity.md

### Role in Pipeline
Entity summary generation — produces 1-3 sentence summaries for entities that don't have one. Called during dream mode step 8.

### Caller
`src/core/llm.py:call_entity_summary()` -> raw `litellm.completion()` (free-text output). Called by `src/pipeline/dream.py:_step_generate_summaries()`.

### Variable Injections

| Variable | Source | Notes |
|----------|--------|-------|
| `{entity_title}` | From entity graph data | Entity name |
| `{entity_type}` | From entity graph data | e.g. "health" |
| `{entity_facts}` | Formatted as `- fact1\n- fact2` or "None" | Live facts (superseded excluded) |
| `{entity_relations}` | Formatted as `- rel_type Target` or "None" | Graph relations |
| `{entity_tags}` | Comma-separated or "None" | Entity tags |
| `{user_language}` | Auto-injected | Referenced in rules |
| `{categories_*}` | Auto-injected | Not referenced |

### Expected Output Format
Free text — 1-3 sentences, under 100 words. No JSON.

### Line-by-Line Analysis

**Lines 1-5 (System):** Clean summarizer role with "Be precise and factual."

**Lines 7-11 (Rules):** 5 simple rules:
- Write in user_language
- Focus on important/recent
- Mention key relations
- Under 100 words
- Third person for people, descriptive for concepts

**Lines 13-26 (User section):** Entity info injection and clear instruction.

### Complexity Assessment for Qwen3-30B: LOW

This is a straightforward summarization task. The input is structured, the output is short, and the constraints are simple. Well-suited for small models.

### Failure Modes with Small Models

1. **Verbosity**: Models produce 5+ sentences instead of 1-3. Minor issue.
2. **Echoing instead of summarizing**: Models list facts instead of synthesizing.
3. **Language errors**: Minor grammatical errors in generated user_language.
4. **Including "None" facts**: If relations are "None", model may say "has no relations" — factual but useless in a summary.

### Improvement Suggestions

1. **Skip "None" sections in input**: Don't send `### Relations\nNone` — just omit the section entirely. This prevents the model from commenting on missing data.
2. **Add few-shot examples**: One good summary example would significantly improve consistency.

### Could Be Split?
No. Already atomic.

### Could Be Deterministic?
Partially. A template-based summary could work for simple entities: "{title} is a {type}. Key facts: {top_3_facts}." But for entities with complex relations, LLM synthesis adds real value.

---

## 10. discover_relations.md

### Role in Pipeline
Dream mode step 5 — evaluates whether a meaningful relation exists between two entities found via FAISS similarity. Called for each candidate pair.

### Caller
`src/core/llm.py:call_relation_discovery()` -> `_call_structured()` with `RelationProposal` response model. Called by `src/pipeline/dream.py:_step_discover_relations()`.

### Variable Injections

| Variable | Source | Notes |
|----------|--------|-------|
| `{entity_a_title}` | From graph entity data | Entity A name |
| `{entity_a_type}` | From graph entity data | Entity A type |
| `{entity_a_dossier}` | Built by `_build_dossier()` in dream.py | Title, type, tags, facts (max 10), summary |
| `{entity_b_title}` | Same for entity B | Entity B name |
| `{entity_b_type}` | Same | Entity B type |
| `{entity_b_dossier}` | Same | Entity B dossier |
| `{categories_relation_types}` | Auto-injected | 13 allowed relation types |
| `{user_language}` | Auto-injected | Referenced for context field |
| `{categories_observations}` | Auto-injected | Not referenced |
| `{categories_entity_types}` | Auto-injected | Not referenced |

### Expected Output Format
```json
{
  "action": "relate",         // or "none"
  "relation_type": "affects", // one of 13 types, or null
  "context": "brief explanation"
}
```

### Line-by-Line Analysis

**Lines 1-5 (System):** Clean role. "Only propose a relation if there is clear evidence" is a critical conservatism constraint.

**Lines 6-13 (Rules):** 6 rules — well-scoped:
- Evidence-based only
- No invention
- "none" if no relation
- Most specific type
- Context in user_language
- JSON only

**Lines 15-40 (User section):** Two entity dossiers, clear task statement, response format with inline JSON example.

### Complexity Assessment for Qwen3-30B: MEDIUM

The model must:
1. Read two entity dossiers
2. Determine if a meaningful connection exists
3. Choose the correct relation type and direction
4. Write a brief context explanation

This is a well-scoped binary decision with optional typing.

### Failure Modes with Small Models

1. **Over-connecting**: Models find "relates" for everything. FAISS already filtered for similarity, so the model is biased toward "relate" by the selection of candidates.
2. **Wrong direction**: Model proposes A->B when B->A is more accurate.
3. **Invalid relation type**: Model invents types not in the allowed list. The `_step_discover_relations()` code validates against `_VALID_RELATION_TYPES` and skips invalid types — good defensive coding.
4. **Defaulting to `linked_to`**: When unsure of the specific type, models fall back to the most generic relation. This creates a graph full of meaningless `linked_to` edges.
5. **`action` field as non-string**: Some models emit `"action": true` instead of `"action": "relate"`.

### Improvement Suggestions

1. **Add explicit `linked_to` deprioritization**: "Prefer specific relation types. Use `linked_to` ONLY when no other type applies."
2. **Add direction examples**: "If A worsens B, use from=A, to=B. The arrow points from cause to effect."
3. **Add a confidence threshold**: "If the relation is weak or speculative, respond with 'none'."
4. **Batch multiple pairs**: Instead of one LLM call per pair, batch 3-5 pairs in a single prompt. This would reduce latency for dream mode step 5, which can have dozens of candidates.

### Could Be Split?
No. Already atomic.

### Could Be Deterministic?
Partially. Tag overlap, shared relations, and co-occurrence patterns could determine many relations deterministically. The batch relation discovery in the main pipeline already does this (FAISS + tag overlap, zero LLM). The LLM adds value only for semantic relations not captured by surface features.

---

## 11. dream_plan.md

### Role in Pipeline
Dream mode coordinator — analyzes memory state and decides which of the 10 dream steps to run. Called once at the start of `run_dream()`.

### Caller
`src/core/llm.py:call_dream_plan()` -> `_call_structured()` with `DreamPlan` response model. Called by `src/pipeline/dream.py:run_dream()`.

### Variable Injections

| Variable | Source | Notes |
|----------|--------|-------|
| `{memory_stats}` | From `_collect_dream_stats()` — formatted key-value pairs | Entity count, relation count, candidates per step |
| `{json_schema}` | `DreamPlan.model_json_schema()` | Schema for `{"steps": [int], "reasoning": str}` |
| `{unextracted_docs}` | Hardcoded to `"(see stats)"` in `call_dream_plan()` | **Bug: always "(see stats)" instead of actual count** |
| `{consolidation_candidates}` | Hardcoded to `"(see stats)"` | Same issue |
| `{merge_candidates}` | Hardcoded to `"(see stats)"` | Same issue |
| `{relation_candidates}` | Hardcoded to `"(see stats)"` | Same issue — though the prompt also says "potential pairs" |
| `{prune_candidates}` | Hardcoded to `"(see stats)"` | Same issue |
| `{summary_candidates}` | Hardcoded to `"(see stats)"` | Same issue |
| `{user_language}` | Auto-injected | Referenced ("Respond in {user_language}") |
| `{categories_*}` | Auto-injected | Not referenced |

### Expected Output Format
```json
{
  "steps": [1, 3, 5, 8, 9, 10],
  "reasoning": "Consolidation needed for 5 entities, relations discovery has candidates..."
}
```

### Line-by-Line Analysis

**Lines 1-2:** Direct instruction — no SYSTEM/USER separation. Uses a conversational tone.

**Lines 4-5:** Memory stats injection.

**Lines 7-15:** Step descriptions with candidate counts. The counts are all `"(see stats)"` because `call_dream_plan()` hardcodes them:
```python
prompt = load_prompt("dream_plan", config,
    memory_stats=memory_stats,
    unextracted_docs="(see stats)",
    consolidation_candidates="(see stats)",
    ...
)
```
This is clearly a workaround — the candidate counts are already in `{memory_stats}` so the per-step placeholders are redundant.

**Lines 17-22:** Rules with strong constraints (include load, include rescore+rebuild if any step runs, skip 0-candidate steps).

**Lines 24-26:** JSON schema and language instruction.

### Complexity Assessment for Qwen3-30B: LOW-MEDIUM

The model reads statistics and makes a planning decision. The output schema is simple (list of ints + string). The main challenge is correctly applying the rules (always include 1, include 8+9 if any other runs, skip 0-candidate).

### Failure Modes with Small Models

1. **Including all steps regardless**: Models play it safe by running everything.
2. **Omitting steps 8/9 (rescore/rebuild)**: Models forget the "always include after changes" rule.
3. **Invalid step numbers**: Models emit step 0, 11, or negative numbers.
4. **Reasoning in wrong language**: "Respond in {user_language}" but the step descriptions are in English, causing language mixing.
5. **"(see stats)" confusion**: The placeholder `"(see stats)"` is vague. The model must cross-reference against `{memory_stats}` to find actual counts.

### Improvement Suggestions

1. **Fix the placeholder bug**: Either inject actual counts into the step descriptions or remove the placeholder variables entirely. Currently the prompt says "extract entities from (see stats) unprocessed RAG documents" which is confusing.
2. **Renumber steps to match code**: The prompt lists steps 1-9 but the code uses 1-10. The mapping: prompt step 1 (Load) = code step 1, prompt step 9 (Rebuild) = code step 10. Step 6 (transitive relations) is missing from the prompt entirely.
3. **Make deterministic**: This could be a simple rule engine: if consolidation_candidates > 0, include step 3; if merge_candidates > 0, include step 4; etc. No LLM needed.

### Critical Bug: Step Numbering Mismatch
The prompt lists 9 steps (1-9) but the code has 10 steps (1-10). Step 6 (transitive relations) is completely absent from the prompt. When the LLM returns `steps: [1, 3, 8, 9]`, the code interprets these as code step numbers. Since the prompt's step 8 (Rescore) is the code's step 9, and the prompt's step 9 (Rebuild) is the code's step 10, the LLM's plan will be misaligned with execution.

Wait — actually examining more carefully: the prompt steps are:
1. Load (code step 1)
2. Extract docs (code step 2)
3. Consolidate (code step 3)
4. Merge (code step 4)
5. Relations (code step 5)
6. Prune (code step 7 — skipping code step 6 transitive)
7. Summaries (code step 8)
8. Rescore (code step 9)
9. Rebuild (code step 10)

The code loop iterates `for s in range(1, 11)` and checks `if s not in steps_to_run`. So if the LLM returns `steps: [1, 6]` meaning "Load + Prune", the code will run code step 6 (transitive relations, NOT prune). **This is a confirmed step numbering bug.**

### Could Be Deterministic?
Yes, completely. A 10-line function checking candidate counts against zero would produce better results than LLM planning. The LLM adds no semantic value here — it's pure conditional logic.

---

## 12. dream_validate.md

### Role in Pipeline
Post-step validation — checks if the results of a critical dream step (consolidation, merging, relation discovery) look reasonable. Called after steps 3, 4, and 5 if they made changes.

### Caller
`src/core/llm.py:call_dream_validate()` -> `_call_structured()` with `DreamValidation` response model. Called by `src/pipeline/dream.py:_validate_step()`.

### Variable Injections

| Variable | Source | Notes |
|----------|--------|-------|
| `{step_name}` | Hardcoded per step ("Fact Consolidation", "Entity Merging", "Relation Discovery") | Step identifier |
| `{changes_summary}` | Summary string (e.g., "5 consolidated", "3 merged") | **Very terse — often just a count** |
| `{json_schema}` | `DreamValidation.model_json_schema()` | Schema for `{"approved": bool, "issues": [str]}` |
| `{user_language}` | Auto-injected | Referenced ("Respond in {user_language}") |
| `{categories_*}` | Auto-injected | Not referenced |

### Expected Output Format
```json
{
  "approved": true,
  "issues": []
}
```

### Line-by-Line Analysis

**Lines 1-2:** "You are reviewing the results of a memory consolidation step." Direct and clear.

**Lines 4-5:** Step name and changes injection.

**Lines 7-12:** Rules:
- Approve if reasonable
- Flag only clearly wrong results
- Be lenient (important — prevents over-rejection)
- Valid JSON

**Line 14:** Language instruction.

### Complexity Assessment for Qwen3-30B: LOW

Binary decision (approve/reject) with optional issue list. The model barely has enough information to make a meaningful assessment.

### Failure Modes with Small Models

1. **Always approving**: Given "Be lenient" and a terse summary like "5 consolidated", models will almost always approve. The validation becomes a rubber stamp.
2. **Approving with invented issues**: Models sometimes return `approved: true` but list issues anyway, creating confusing reports.
3. **No context for real validation**: The `{changes_summary}` is just "5 consolidated" — the model doesn't see what was actually consolidated. It can't validate without seeing the before/after.

### Improvement Suggestions

1. **Provide actual change details**: Instead of "5 consolidated", pass the actual before/after for at least a sample of changes. Without this, validation is meaningless.
2. **Consider removing entirely**: If the model can't see what changed, this step adds latency without value. The current implementation is a cargo-cult validation pattern.
3. **If keeping, make specific**: Instead of generic "check if reasonable", provide specific checks per step type (e.g., for merging: "Were entities of different types merged? Were entities with no name overlap merged?").

### Could Be Deterministic?
Yes. Specific validation rules could be coded:
- Fact consolidation: check that total fact count decreased, no categories were lost
- Entity merging: verify types match, alias overlap exists
- Relation discovery: verify relation types are valid, no self-referencing
All of these are already partially done in the code.

---

## 13. extract_relations.md

### Role in Pipeline
**PLACEHOLDER.** Single line: "# Reserved for v2 -- relation extraction is currently part of extract_facts.md"

### Caller
None. Not loaded anywhere.

### Analysis
This file documents the intention to split relation extraction from entity extraction into a separate step. As noted in the `extract_facts.md` analysis, this split would reduce cognitive load and improve small model performance.

### Improvement Suggestions
1. **Implement the split**: Create the actual prompt for standalone relation extraction, taking extracted entities as input and producing relations as output.
2. **Or delete**: If the split isn't planned, remove the placeholder to reduce confusion.

---

## Comparison Table

| # | Prompt | Complexity | LLM Required? | Could Be Deterministic? | Key Issue |
|---|--------|-----------|---------------|------------------------|-----------|
| 1 | `extract_facts.md` | HIGH | Yes | No (core NLU task) | 6+ simultaneous tasks; `{json_schema}` variable unused; `supersedes` field unreliable |
| 2 | `arbitrate_entity.md` | LOW | Partially | Mostly (better FAISS/alias matching) | No guard against hallucinated entity IDs |
| 3 | `consolidate.md` | N/A | N/A | Already is (dream step 4) | **DEAD CODE** — unused, no caller |
| 4 | `consolidate_facts.md` | HIGH | Yes (semantic merge) | Partially (dedup + pruning can be deterministic) | Index tracking error-prone; overwhelms small models on 50+ facts |
| 5 | `context_template.md` | N/A (template) | No | Already deterministic | Hardcodes `search_rag` tool name |
| 6 | `context_section.md` | MEDIUM-HIGH | Optional | Mostly (dedup+type fix should be pre-processed) | LLM asked to merge entities (should be deterministic) |
| 7 | `context_instructions.md` | N/A (static) | No | Already static | Default placeholder wastes tokens in every context |
| 8 | `generate_context.md` | VERY HIGH | Yes (if narrative) | Yes (deterministic already default) | **LEGACY** — replaced by per-section approach |
| 9 | `summarize_entity.md` | LOW | Yes (synthesis) | Partially (template OK for simple entities) | Sends "None" sections unnecessarily |
| 10 | `discover_relations.md` | MEDIUM | Yes (semantic judgment) | Partially (tag/co-occurrence can help) | Over-connecting bias; `linked_to` overuse |
| 11 | `dream_plan.md` | LOW-MEDIUM | No | **Yes, completely** | **BUG: step numbering mismatch with code (missing step 6 transitive)** |
| 12 | `dream_validate.md` | LOW | No | **Yes, completely** | Rubber-stamp validation — model can't see actual changes |
| 13 | `extract_relations.md` | N/A | N/A | N/A | **PLACEHOLDER** — single comment line |

---

## Cross-Cutting Issues

### 1. Dead/Orphan Prompts (2 files)
- **`consolidate.md`**: No caller in any source file. Entity duplicate detection is done deterministically in dream mode step 4. Should be deleted or documented as deprecated.
- **`extract_relations.md`**: Placeholder with a single comment. Either implement the v2 split or delete.

### 2. Legacy Prompt (1 file)
- **`generate_context.md`**: Only active with `context_narrative: true`. Documented as "legacy, replaced by per-section" in CLAUDE.md. Should add a deprecation warning in code and consider removal in next major version.

### 3. Auto-Injected but Unused Variables
`load_prompt()` always injects `{user_language}`, `{categories_observations}`, `{categories_entity_types}`, `{categories_relation_types}`. Several prompts reference none or only some of these:

| Prompt | `user_language` | `categories_obs` | `categories_entity` | `categories_rel` |
|--------|:-:|:-:|:-:|:-:|
| `arbitrate_entity.md` | No | No | Yes | No |
| `consolidate_facts.md` | Yes | No | No | No |
| `context_section.md` | Yes | No | No | No |
| `discover_relations.md` | Yes | No | No | Yes |
| `dream_plan.md` | Yes | No | No | No |
| `dream_validate.md` | Yes | No | No | No |
| `summarize_entity.md` | Yes | No | No | No |

The unreferenced variables are harmless (they remain as literal `{categories_observations}` in the final prompt text since `str.replace()` only replaces if found). However, if any model encounters these literal `{braces}` in the prompt, it could confuse JSON-mode parsing.

**Wait — actually this is not true.** Since `load_prompt()` does `content.replace("{categories_observations}", ...)`, ALL four variables ARE replaced in EVERY prompt, even if the prompt doesn't reference them. The variables simply don't appear in the prompt text, so the replace is a no-op. This is correct behavior — no unreplaced braces remain. The only waste is in the Python string scan, which is negligible.

### 4. Step Numbering Bug in dream_plan.md (Critical)
The prompt lists 9 steps numbered 1-9, but the code has 10 steps numbered 1-10. Step 6 (transitive relations) is absent from the prompt. This means:
- Prompt step 6 (Prune) maps to code step 7
- Prompt step 7 (Summaries) maps to code step 8
- Prompt step 8 (Rescore) maps to code step 9
- Prompt step 9 (Rebuild) maps to code step 10

When the LLM returns step numbers, the code uses them directly as indices into its 1-10 step loop. **The LLM will never request transitive relation inference (code step 6) and will misroute steps 6-9.**

### 5. Prompt Complexity Distribution

```
LOW:       arbitrate_entity, summarize_entity, dream_validate
MEDIUM:    discover_relations, dream_plan
HIGH:      extract_facts, consolidate_facts, context_section
VERY HIGH: generate_context (legacy)
```

The two HIGH-complexity prompts (`extract_facts` and `consolidate_facts`) are the most frequently called and most critical to system quality. They would benefit most from splitting into simpler steps.

### 6. Common Failure Pattern: Token/Length Budgeting
Three prompts ask the LLM to respect token budgets (`consolidate_facts.md` max_facts, `context_section.md` budget_tokens, `generate_context.md` context_max_tokens). Small models universally fail at this. **All token budgeting should be enforced in post-processing code, not delegated to the LLM.** The code should truncate output rather than hoping the model respects limits.

### 7. French-Specific Examples Baked Into Prompts
Several prompts contain French-specific examples:
- `context_section.md`: "Hernie discale L5-S1", "dos (sciatique)", "ski station"
- `extract_facts.md`: "Marie", "Airbus", French observation examples

These examples bias models toward French medical/health domains. For English-language deployments, the examples would be confusing. Consider making examples configurable or at least neutral.

### 8. Recommendations Priority List

**P0 — Bugs:**
1. Fix `dream_plan.md` step numbering to include transitive relations as step 6 and renumber to 1-10.

**P1 — High Impact:**
2. Replace `dream_plan.md` with deterministic logic (no LLM needed).
3. Replace `dream_validate.md` with deterministic validation rules (no LLM needed, current impl is rubber-stamp).
4. Delete `consolidate.md` (dead code).
5. Delete or implement `extract_relations.md` (placeholder).

**P2 — Quality:**
6. Remove `supersedes` field from `extract_facts.md` (unreliable, requires impossible context).
7. Pre-cluster facts before calling `consolidate_facts.md` to reduce cognitive load.
8. Remove entity merging responsibility from `context_section.md` (should be deterministic pre-processing).
9. Make `context_instructions.md` conditional injection (skip if placeholder content).

**P3 — Nice to Have:**
10. Add explicit observation count cap to `extract_facts.md`.
11. Add few-shot example to `summarize_entity.md`.
12. Batch relation pairs in `discover_relations.md` calls.
13. Deprecate `generate_context.md` formally in code.
14. Add `{json_schema}` reference to `extract_facts.md` (variable computed but not used in template).
