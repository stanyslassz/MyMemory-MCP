# memory-ai Pipeline Reference

> Consult this file before modifying any pipeline function.
> Updated: 2026-03-06

---

## CLI Commands

### run() — cli.py:38
- **Called by**: User via `uv run memory run`
- **Calls**: extract_from_chat → resolve_all → arbitrate_entity → enrich_memory → build_deterministic_context → write_context → incremental_update
- **Input**: Config (from ctx)
- **Output**: Processed chats, updated graph, context, FAISS
- **Side effects**: Writes entity MDs, _graph.json, _context.md, _index.md, FAISS files
- **On error**: Per-chat: extraction failure → fallback to doc_ingest. Enrichment failure → skip chat. Context/FAISS failure → warning only.

### rebuild_all() — cli.py:209
- **Called by**: User via `uv run memory rebuild-all`
- **Calls**: rebuild_from_md → recalculate_all_scores → save_graph → write_index → build_deterministic_context → write_context → build_index
- **Input**: Config
- **Side effects**: Rewrites _graph.json, _index.md, _context.md, FAISS files

### serve() — cli.py:410
- **Calls**: run_server (mcp/server.py)

### inbox() — cli.py:304
- **Calls**: process_inbox (pipeline/inbox.py)

### replay() — cli.py:427
- **Calls**: extract_from_chat → resolve_all → enrich_memory (retry path)

### consolidate() — cli.py:499
- **Calls**: load_graph → duplicate detection (deterministic)

---

## Pipeline Functions

### extract_from_chat() — pipeline/extractor.py
- **Called by**: cli.py:run(), cli.py:replay()
- **Calls**: llm.call_extraction()
- **Input**: chat_content (str), config (Config)
- **Output**: RawExtraction (entities, relations, summary)
- **Side effects**: None
- **On error**: StallError or Exception → caught by cli.py, fallback to doc_ingest

### call_extraction() — core/llm.py:174
- **Called by**: extract_from_chat()
- **Calls**: load_prompt("extract_facts"), _call_with_stall_detection()
- **Input**: chat_content (str), config
- **Output**: RawExtraction
- **Key detail**: Uses streaming + stall detection. Timeout = config.llm_extraction.timeout

### _call_with_stall_detection() — core/llm.py:98
- **Called by**: call_extraction()
- **Calls**: _get_client(), Instructor create_partial (streaming)
- **Input**: step_config, prompt, response_model, stall_timeout
- **Output**: T (Pydantic model from streaming)
- **Key detail**: Worker thread + watchdog. Stall = no chunk for stall_timeout seconds. Connection timeout = step_config.timeout × 3.

### _call_structured() — core/llm.py:70
- **Called by**: call_arbitration()
- **Input**: step_config, prompt, response_model
- **Output**: T (Pydantic model)
- **Key detail**: No streaming, no stall detection. Simple Instructor call.

### resolve_all() — pipeline/resolver.py
- **Called by**: cli.py:run(), cli.py:replay()
- **Calls**: graph lookup (deterministic), optional FAISS search
- **Input**: RawExtraction, GraphData, faiss_search_fn
- **Output**: ResolvedExtraction
- **Side effects**: None

### arbitrate_entity() — pipeline/arbitrator.py
- **Called by**: cli.py:run() (for ambiguous entities only)
- **Calls**: llm.call_arbitration()
- **Input**: entity_name, context, candidates, graph, config
- **Output**: EntityResolution

### enrich_memory() — pipeline/enricher.py:23
- **Called by**: cli.py:run(), cli.py:replay()
- **Calls**: load_graph, _update_existing_entity / _create_new_entity, add_relation, recalculate_all_scores, save_graph, write_index
- **Input**: ResolvedExtraction, Config
- **Output**: EnrichmentReport
- **Side effects**: Creates/updates entity MD files, writes _graph.json, writes _index.md

### _create_new_entity() — pipeline/enricher.py:154
- **Called by**: enrich_memory()
- **Calls**: create_entity (store.py), add_entity (graph.py)
- **Side effects**: Creates memory/{folder}/{slug}.md, adds to graph
- **BUG**: EntityFrontmatter at line 175 missing mention_dates=[today]

### _update_existing_entity() — pipeline/enricher.py:107
- **Called by**: enrich_memory()
- **Calls**: update_entity (store.py), add_mention (mentions.py)
- **Side effects**: Updates entity MD file, updates graph metadata

### build_deterministic_context() — memory/context.py:89
- **Called by**: cli.py:run(), cli.py:rebuild_all()
- **Calls**: get_top_entities(), read_entity()
- **Input**: GraphData, memory_path, Config
- **Output**: str (markdown)
- **Key detail**: Filters by min_score_for_context (default 0.3). Returns near-empty if all scores below threshold.

### get_top_entities() — memory/scoring.py:191
- **Called by**: build_deterministic_context(), build_context_input()
- **Input**: graph, n, include_permanent, min_score
- **Output**: list[(entity_id, GraphEntity)]
- **Key detail**: Permanent entities always included. Others filtered by min_score.

### recalculate_all_scores() — memory/scoring.py:173
- **Called by**: enrich_memory(), rebuild_all()
- **Calls**: spreading_activation(), calculate_score()
- **Side effects**: Mutates entity.score in graph

### calculate_actr_base() — memory/scoring.py:18
- **Key detail**: Returns -5.0 if mention_dates is empty → sigmoid(-5) ≈ 0.008

### rebuild_from_md() — memory/graph.py:186
- **Called by**: rebuild_all(), load_graph (fallback)
- **Side effects**: Creates new GraphData from MD files
- **BUG**: Does NOT read mention_dates, monthly_buckets, created from frontmatter (lines 204-215)

### build_index() — pipeline/indexer.py
- **Called by**: cli.py:rebuild_all(), cli.py:rebuild_faiss()
- **Side effects**: Writes FAISS index + pickle mapping + manifest JSON

### incremental_update() — pipeline/indexer.py
- **Called by**: cli.py:run()
- **Side effects**: Updates FAISS index for changed files

### ingest_document() — pipeline/doc_ingest.py
- **Called by**: _fallback_to_doc_ingest (cli.py), process_inbox (inbox.py)
- **Side effects**: Adds chunks to FAISS (RAG only, no graph entities)

### classify() — pipeline/router.py
- **Called by**: process_inbox()
- **Output**: RouteDecision (conversation/document/uncertain)
- **Key detail**: Pure heuristics, zero LLM

---

## MCP Tools

### get_context() — mcp/server.py:27
- Reads _context.md file and returns content as-is
- Fallback: _index.md if _context.md absent

### save_chat() — mcp/server.py:47
- Saves messages as chat MD file with frontmatter (processed: false)

### search_rag() — mcp/server.py:63
- FAISS search + graph re-ranking (60% FAISS / 40% graph score)
- L2→L1 re-emergence: updates mention_dates on queried entities
