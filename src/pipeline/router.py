"""Content router: deterministic classification of inbox items as conversation or document."""

from __future__ import annotations

import json
import re

from src.core.models import RouteDecision


# ── Heuristic signals ────────────────────────────────────────

# Conversation indicators
_SPEAKER_PATTERN = re.compile(
    r"^(User|Assistant|Human|AI|System|Utilisateur|Humain)\s*:", re.MULTILINE
)
_TIMESTAMP_SPEAKER = re.compile(
    r"^\[?\d{1,4}[-/]\d{1,2}[-/]\d{1,4}[\s,T]?\d{0,2}:?\d{0,2}\]?\s*\w+\s*:", re.MULTILINE
)
_ROLE_CONTENT_JSON = re.compile(r'"role"\s*:\s*"(user|assistant|system|human)"', re.IGNORECASE)

# Document indicators
_HEADING_PATTERN = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_LONG_PARAGRAPH_THRESHOLD = 200  # chars without line breaks
_BULLET_PATTERN = re.compile(r"^[\s]*[-*•]\s+\S", re.MULTILINE)
_NUMBERED_LIST = re.compile(r"^[\s]*\d+[.)]\s+\S", re.MULTILINE)


def _count_speaker_turns(text: str) -> int:
    return len(_SPEAKER_PATTERN.findall(text))


def _count_timestamp_speakers(text: str) -> int:
    return len(_TIMESTAMP_SPEAKER.findall(text))


def _count_role_content_markers(text: str) -> int:
    return len(_ROLE_CONTENT_JSON.findall(text))


def _is_json_chat_array(text: str) -> bool:
    """Check if text is a JSON array with role/content dicts."""
    stripped = text.strip()
    if not stripped.startswith("["):
        return False
    try:
        data = json.loads(stripped)
        if isinstance(data, list) and len(data) > 0:
            return all(
                isinstance(item, dict) and "role" in item and "content" in item
                for item in data[:5]  # check first 5 elements
            )
    except (json.JSONDecodeError, TypeError):
        pass
    return False


def _count_headings(text: str) -> int:
    return len(_HEADING_PATTERN.findall(text))


def _count_long_paragraphs(text: str) -> int:
    """Count paragraphs exceeding threshold length."""
    paragraphs = re.split(r"\n\s*\n", text)
    return sum(1 for p in paragraphs if len(p.strip()) > _LONG_PARAGRAPH_THRESHOLD)


def _count_bullet_items(text: str) -> int:
    return len(_BULLET_PATTERN.findall(text)) + len(_NUMBERED_LIST.findall(text))


def _dialogue_turn_density(text: str) -> float:
    """Ratio of speaker-turn lines to total non-empty lines."""
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return 0.0
    turns = _count_speaker_turns(text) + _count_timestamp_speakers(text)
    return turns / len(lines)


# ── Main router ──────────────────────────────────────────────


def classify(text: str, source_filename: str | None = None) -> RouteDecision:
    """Classify text as conversation, document, or uncertain.

    Uses deterministic heuristics only (zero LLM calls).
    """
    if not text or not text.strip():
        return RouteDecision(route="uncertain", confidence=0.0, reasons=["empty content"])

    reasons: list[str] = []
    conv_score = 0.0
    doc_score = 0.0

    # JSON chat array is a strong conversation signal
    if _is_json_chat_array(text):
        return RouteDecision(
            route="conversation",
            confidence=0.98,
            reasons=["JSON array with role/content structure"],
        )

    # Speaker turn markers
    speaker_turns = _count_speaker_turns(text)
    ts_speakers = _count_timestamp_speakers(text)
    role_markers = _count_role_content_markers(text)
    total_turns = speaker_turns + ts_speakers

    if total_turns >= 4:
        conv_score += 0.5
        reasons.append(f"{total_turns} speaker-turn markers")
    elif total_turns >= 2:
        conv_score += 0.25
        reasons.append(f"{total_turns} speaker-turn markers (moderate)")

    if role_markers >= 2:
        conv_score += 0.3
        reasons.append(f"{role_markers} role/content JSON markers")

    # Dialogue density
    density = _dialogue_turn_density(text)
    if density > 0.3:
        conv_score += 0.2
        reasons.append(f"dialogue density {density:.2f}")

    # Document signals
    headings = _count_headings(text)
    long_paras = _count_long_paragraphs(text)
    bullets = _count_bullet_items(text)

    if headings >= 2:
        doc_score += 0.35
        reasons.append(f"{headings} markdown headings")
    elif headings == 1:
        doc_score += 0.15
        reasons.append("1 markdown heading")

    if long_paras >= 2:
        doc_score += 0.3
        reasons.append(f"{long_paras} long paragraphs")
    elif long_paras == 1:
        doc_score += 0.15
        reasons.append("1 long paragraph")

    if bullets >= 5:
        doc_score += 0.2
        reasons.append(f"{bullets} bullet/list items")
    elif bullets >= 2:
        doc_score += 0.1
        reasons.append(f"{bullets} bullet/list items")

    # Single-author note style: no speaker markers + some structure
    if total_turns == 0 and (headings > 0 or long_paras > 0):
        doc_score += 0.15
        reasons.append("single-author style (no speaker markers)")

    # Decision
    margin = abs(conv_score - doc_score)

    if conv_score > doc_score and conv_score >= 0.4:
        confidence = min(0.95, 0.5 + margin)
        return RouteDecision(route="conversation", confidence=round(confidence, 2), reasons=reasons)

    if doc_score > conv_score and doc_score >= 0.3:
        confidence = min(0.95, 0.5 + margin)
        return RouteDecision(route="document", confidence=round(confidence, 2), reasons=reasons)

    # Uncertain — default to document for safety (immediate retrieval)
    if doc_score > 0 or conv_score > 0:
        reasons.append("low signal margin, defaulting uncertain")
    else:
        reasons.append("no strong signals detected")

    return RouteDecision(
        route="uncertain",
        confidence=round(max(conv_score, doc_score, 0.1), 2),
        reasons=reasons,
    )
