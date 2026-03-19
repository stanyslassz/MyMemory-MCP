"""Dream mode: brain-like memory reorganization during idle time.

10-step pipeline: load -> extract docs -> consolidate facts -> merge entities
-> discover relations -> transitive relations -> prune dead -> generate summaries
-> rescore -> rebuild.

No new information enters -- only reorganization of existing knowledge.
LLM coordinator plans which steps to run and validates critical results.
"""

from src.pipeline.dream.coordinator import (
    DreamReport,
    decide_dream_steps,
    run_dream,
    validate_dream_step,
    _generate_dream_report,
    _validate_step,
    _count_live_facts,
    _save_checkpoint,
    _load_checkpoint,
    _clear_checkpoint,
    _collect_dream_stats,
)
from src.pipeline.dream.consolidator import (
    _step_consolidate_facts,
    _step_generate_summaries,
    _step_extract_documents,
)
from src.pipeline.dream.merger import (
    _step_merge_entities,
    _do_merge,
    _find_faiss_dedup_candidates,
)
from src.pipeline.dream.discovery import (
    _step_discover_relations,
    _build_dossier,
    _step_transitive_relations,
    _TRANSITIVE_RULES,
)
from src.pipeline.dream.maintenance import (
    _step_prune_dead,
    _step_load,
    _step_rebuild,
)

__all__ = [
    "run_dream",
    "DreamReport",
    "decide_dream_steps",
    "validate_dream_step",
    "_generate_dream_report",
    "_validate_step",
    "_count_live_facts",
    "_save_checkpoint",
    "_load_checkpoint",
    "_clear_checkpoint",
    "_collect_dream_stats",
    "_step_consolidate_facts",
    "_step_generate_summaries",
    "_step_extract_documents",
    "_step_merge_entities",
    "_do_merge",
    "_find_faiss_dedup_candidates",
    "_step_discover_relations",
    "_build_dossier",
    "_step_transitive_relations",
    "_TRANSITIVE_RULES",
    "_step_prune_dead",
    "_step_load",
    "_step_rebuild",
]
