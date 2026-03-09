"""Step 1: Extract structured information from chat content."""

from __future__ import annotations

import logging

from typing import Union, get_args

from src.core.config import Config
from src.core.llm import call_extraction
from src.core.models import (
    RawExtraction, RawEntity, RawObservation, RawRelation,
    RelationType, EntityType, ObservationCategory,
)
from src.core.utils import estimate_tokens as _estimate_tokens
from src.core.utils import slugify

logger = logging.getLogger(__name__)

# ── Valid sets (derived from Literal types) ──────────────────
_VALID_RELATION_TYPES: set[str] = set(get_args(RelationType))
_VALID_ENTITY_TYPES: set[str] = set(get_args(EntityType))
_VALID_OBSERVATION_CATEGORIES: set[str] = set(get_args(ObservationCategory))

# ── Fuzzy mapping for common LLM inventions ──────────────────
_RELATION_FALLBACK: dict[str, str] = {
    "prescrit_par": "linked_to",
    "prescribed_by": "linked_to",
    "travaille_a": "works_at",
    "travaille_à": "works_at",
    "ami_de": "friend_of",
    "parent_de": "parent_of",
    "vit_avec": "lives_with",
    "utilise": "uses",
    "fait_partie_de": "part_of",
    "cause": "affects",
    "ameliore": "improves",
    "aggrave": "worsens",
}


def sanitize_extraction(raw: Union[RawExtraction, dict]) -> RawExtraction:
    """Clean up LLM output: fix invalid types, null fields, empty refs.

    Accepts either a RawExtraction or a raw dict (for cases where Pydantic
    validation would fail on the raw LLM output). Returns a valid RawExtraction.
    """
    if isinstance(raw, RawExtraction):
        data = raw.model_dump()
    else:
        data = raw

    # Fix summary
    if data.get("summary") is None:
        logger.warning("Sanitized summary None → ''")
        data["summary"] = ""

    # Fix entities
    clean_entities = []
    for ent in data.get("entities", []):
        name = (ent.get("name") or "").strip()
        if not name:
            logger.warning("Dropped entity with empty name")
            continue

        # Fix entity type
        etype = ent.get("type", "")
        if etype not in _VALID_ENTITY_TYPES:
            logger.warning("Sanitized entity type '%s' → 'interest' for '%s'", etype, name)
            ent["type"] = "interest"

        # Fix observations
        clean_obs = []
        for obs in ent.get("observations", []):
            content = (obs.get("content") or "").strip()
            if not content:
                logger.warning("Dropped observation with empty content in '%s'", name)
                continue

            cat = obs.get("category", "")
            if cat not in _VALID_OBSERVATION_CATEGORIES:
                logger.warning("Sanitized observation category '%s' → 'fact' in '%s'", cat, name)
                obs["category"] = "fact"

            # Clamp importance
            imp = obs.get("importance", 0.5)
            if imp is None:
                imp = 0.5
            obs["importance"] = max(0.0, min(1.0, float(imp)))

            # Coerce None fields
            if obs.get("valence") is None:
                obs["valence"] = ""
            if obs.get("date") is None:
                obs["date"] = ""
            if obs.get("supersedes") is None:
                obs["supersedes"] = ""
            if obs.get("tags") is None:
                obs["tags"] = []

            clean_obs.append(obs)

        ent["observations"] = clean_obs
        clean_entities.append(ent)

    data["entities"] = clean_entities

    # Fix relations
    clean_relations = []
    for rel in data.get("relations", []):
        from_name = (rel.get("from_name") or "").strip()
        to_name = (rel.get("to_name") or "").strip()
        if not from_name or not to_name:
            logger.warning("Dropped relation with empty ref: '%s' → '%s'", from_name, to_name)
            continue

        rtype = rel.get("type", "")
        if rtype not in _VALID_RELATION_TYPES:
            mapped = _RELATION_FALLBACK.get(rtype, "linked_to")
            logger.warning("Sanitized relation type '%s' → '%s'", rtype, mapped)
            rel["type"] = mapped

        # Coerce None context
        if rel.get("context") is None:
            rel["context"] = ""

        clean_relations.append(rel)

    data["relations"] = clean_relations

    return RawExtraction(**data)


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
