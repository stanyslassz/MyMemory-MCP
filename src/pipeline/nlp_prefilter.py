"""Optional spaCy NLP pre-filter for date extraction, NER, and dedup."""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta

logger = logging.getLogger(__name__)

_nlp = None


def is_available() -> bool:
    """Check if spaCy is installed and the model is available."""
    try:
        import spacy
        spacy.load("fr_core_news_sm")
        return True
    except (ImportError, OSError):
        return False


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("fr_core_news_sm")
    return _nlp


def extract_dates(text: str, reference_date: str | None = None) -> list[str]:
    """Extract dates from French text using spaCy NER. Returns ISO date strings."""
    if not is_available():
        return []
    nlp = _get_nlp()
    doc = nlp(text)
    dates = []
    for ent in doc.ents:
        if ent.label_ == "DATE" or ent.label_ == "TIME":
            # Try to parse with regex patterns for common French date formats
            parsed = _try_parse_french_date(ent.text, reference_date)
            if parsed:
                dates.append(parsed)
    return dates


def extract_entities(text: str) -> list[dict]:
    """Extract named entities (PER, ORG, LOC). Returns list of {text, type, start, end}."""
    if not is_available():
        return []
    nlp = _get_nlp()
    doc = nlp(text)
    return [
        {"text": ent.text, "type": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
        if ent.label_ in ("PER", "ORG", "LOC")
    ]


def compute_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity between two texts using spaCy vectors."""
    if not is_available():
        return 0.0
    nlp = _get_nlp()
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    if doc1.vector_norm == 0 or doc2.vector_norm == 0:
        return 0.0
    return doc1.similarity(doc2)


def _try_parse_french_date(text: str, reference_date: str | None = None) -> str | None:
    """Try to parse a French date expression to ISO format."""
    text_lower = text.lower().strip()
    ref = date.fromisoformat(reference_date) if reference_date else date.today()

    # "aujourd'hui"
    if "aujourd" in text_lower:
        return ref.isoformat()

    # "hier"
    if text_lower == "hier":
        return (ref - timedelta(days=1)).isoformat()

    # "il y a N jours"
    m = re.search(r"il y a (\d+) jours?", text_lower)
    if m:
        return (ref - timedelta(days=int(m.group(1)))).isoformat()

    # "il y a N semaines"
    m = re.search(r"il y a (\d+) semaines?", text_lower)
    if m:
        return (ref - timedelta(weeks=int(m.group(1)))).isoformat()

    # Try direct ISO parse
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text_lower)
    if m:
        return m.group(1)

    return None
