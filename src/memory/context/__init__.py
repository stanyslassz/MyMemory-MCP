"""Context generation package: builds enriched dossier and generates _context.md and _index.md."""

from src.memory.context.builder import (
    build_context,
    build_context_for_config,
    build_natural_context,
    build_context_with_llm,
    write_context,
    generate_index,
    write_index,
)
from src.memory.context.formatter import (
    _enrich_entity,
    _enrich_entity_natural,
    _extract_vigilances,
    _collect_section,
    _select_entities_for_natural,
    _classify_temporal,
    _build_natural_bullet,
    _build_section_llm,
    _read_entity_facts,
    _is_fact_expired,
    _filter_expired_facts,
)
from src.memory.context.utilities import (
    _content_similarity,
    _deduplicate_facts_for_context,
    _sort_facts_by_date,
)

__all__ = [
    "build_context",
    "build_context_for_config",
    "build_natural_context",
    "build_context_with_llm",
    "write_context",
    "generate_index",
    "write_index",
]
