You are a memory consolidation planner. Analyze the memory state below and decide which dream steps to run.

## Memory State
{memory_stats}

## Available Steps
1. Load — always runs first
2. Extract docs — extract entities from {unextracted_docs} unprocessed RAG documents
3. Consolidate — merge redundant facts in entities with 8+ observations ({consolidation_candidates} candidates)
4. Merge — merge duplicate entities detected by slug/alias overlap ({merge_candidates} candidates)
5. Relations — discover new relations via FAISS similarity ({relation_candidates} potential pairs)
6. Prune — archive dead entities: score < 0.1, freq <= 1, age > 90d, no relations ({prune_candidates} candidates)
7. Summaries — generate summaries for entities without one ({summary_candidates} candidates)
8. Rescore — recalculate ACT-R scores (always recommended after changes)
9. Rebuild — rebuild _context.md and FAISS index (always recommended last)

## Rules
- Step 1 (Load) is always included
- Steps 8 and 9 should be included if ANY other step runs
- Skip steps with 0 candidates unless there's a good reason
- Prefer smaller focused runs over running everything
- Output valid JSON matching this schema:

{json_schema}

Respond in {user_language}.