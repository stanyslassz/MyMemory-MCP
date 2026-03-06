"""Step 1: Extract structured information from chat content."""

from __future__ import annotations

import logging

from src.core.config import Config
from src.core.llm import call_extraction
from src.core.models import RawExtraction, RawEntity
from src.pipeline.resolver import slugify

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: words * 1.3."""
    return int(len(text.split()) * 1.3)


def _merge_extractions(extractions: list[RawExtraction]) -> RawExtraction:
    """Merge multiple segment extractions into one, deduplicating."""
    merged_entities: dict[str, RawEntity] = {}

    for ext in extractions:
        for entity in ext.entities:
            slug = slugify(entity.name)
            if slug in merged_entities:
                existing = merged_entities[slug]
                seen_content = {o.content.lower() for o in existing.observations}
                for obs in entity.observations:
                    if obs.content.lower() not in seen_content:
                        existing.observations.append(obs)
                        seen_content.add(obs.content.lower())
            else:
                merged_entities[slug] = entity.model_copy(deep=True)

    seen_rels: set[tuple[str, str, str]] = set()
    merged_relations = []
    for ext in extractions:
        for rel in ext.relations:
            key = (slugify(rel.from_name), slugify(rel.to_name), rel.type)
            if key not in seen_rels:
                merged_relations.append(rel)
                seen_rels.add(key)

    summary = " ".join(ext.summary for ext in extractions if ext.summary)

    return RawExtraction(
        entities=list(merged_entities.values()),
        relations=merged_relations,
        summary=summary,
    )


def _split_text(text: str, segment_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text into overlapping segments by word count."""
    words = text.split()
    words_per_segment = int(segment_tokens / 1.3)
    words_overlap = int(overlap_tokens / 1.3)

    if len(words) <= words_per_segment:
        return [text]

    segments = []
    start = 0
    while start < len(words):
        end = start + words_per_segment
        segment = " ".join(words[start:end])
        segments.append(segment)
        start = end - words_overlap
        if start >= len(words):
            break

    return segments


def extract_from_chat(chat_content: str, config: Config) -> RawExtraction:
    """Extract structured information from a chat conversation.

    If content exceeds 70% of the model's context_window, splits into
    overlapping segments and merges the results.
    """
    if not chat_content.strip():
        return RawExtraction(entities=[], relations=[], summary="")

    content_tokens = _estimate_tokens(chat_content)
    prompt_overhead = 500
    context_window = config.llm_extraction.context_window
    threshold = int(context_window * 0.7)

    if content_tokens + prompt_overhead < threshold:
        return call_extraction(chat_content, config)

    # Split and extract per segment
    segment_tokens = int(context_window * 0.5)
    segments = _split_text(chat_content, segment_tokens, overlap_tokens=200)

    logger.info(
        "Content too large (%d tokens, window=%d). Splitting into %d segments.",
        content_tokens, context_window, len(segments),
    )

    extractions = []
    for i, segment in enumerate(segments):
        logger.info("Extracting segment %d/%d (%d tokens)", i + 1, len(segments), _estimate_tokens(segment))
        ext = call_extraction(segment, config)
        extractions.append(ext)

    return _merge_extractions(extractions)
