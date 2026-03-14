"""NLP utilities: dedup, NER relation extraction, extractive summary.

All functions degrade gracefully when optional dependencies
(rapidfuzz, spaCy, scikit-learn) are not installed.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)


# ── 3a. Fact deduplication (rapidfuzz with difflib fallback) ──────


def dedup_facts_deterministic(facts: list[str], threshold: float = 85.0) -> list[str]:
    """Remove near-duplicate facts using fuzzy string matching.

    Uses rapidfuzz.fuzz.ratio when available, falls back to
    difflib.SequenceMatcher otherwise.
    """
    similarity_fn = _get_similarity_fn()
    kept: list[str] = []
    for fact in facts:
        is_dup = False
        for existing in kept:
            if similarity_fn(fact.lower(), existing.lower()) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(fact)
    return kept


def _get_similarity_fn():
    """Return a similarity function (str, str) -> float in 0..100 range."""
    try:
        from rapidfuzz.fuzz import ratio
        return ratio
    except ImportError:
        logger.debug("rapidfuzz not installed, falling back to difflib")
        from difflib import SequenceMatcher

        def _difflib_ratio(a: str, b: str) -> float:
            return SequenceMatcher(None, a, b).ratio() * 100.0

        return _difflib_ratio


# ── 3b. NER-based relation extraction (spaCy) ────────────────────


# NER label → entity type mapping for relation inference
_NER_RELATION_MAP: dict[tuple[str, str], str] = {
    ("PER", "ORG"): "works_at",
    ("PER", "LOC"): "linked_to",
    ("PER", "GPE"): "linked_to",
    ("ORG", "LOC"): "linked_to",
    ("ORG", "GPE"): "linked_to",
}


def extract_relations_nlp(
    text: str,
    language: str = "fr",
) -> list[dict[str, str]]:
    """Extract typed relations from text using spaCy NER.

    Returns list of dicts: {"from": name, "to": name, "type": relation_type}.
    Returns empty list if spaCy is not installed or model unavailable.
    """
    nlp = _load_spacy(language)
    if nlp is None:
        return []

    doc = nlp(text)
    entities_by_label: dict[str, list[str]] = {}
    for ent in doc.ents:
        label = ent.label_
        # Normalize to coarse labels
        coarse = _coarse_ner_label(label)
        if coarse:
            entities_by_label.setdefault(coarse, []).append(ent.text)

    relations: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for (label_a, label_b), rel_type in _NER_RELATION_MAP.items():
        for name_a in entities_by_label.get(label_a, []):
            for name_b in entities_by_label.get(label_b, []):
                if name_a == name_b:
                    continue
                key = (name_a, name_b, rel_type)
                if key not in seen:
                    seen.add(key)
                    relations.append({
                        "from": name_a,
                        "to": name_b,
                        "type": rel_type,
                    })

    return relations


def _coarse_ner_label(label: str) -> str | None:
    """Map fine-grained NER labels to coarse categories."""
    mapping = {
        "PER": "PER", "PERSON": "PER",
        "ORG": "ORG", "ORGANIZATION": "ORG",
        "LOC": "LOC", "LOCATION": "LOC",
        "GPE": "GPE",
    }
    return mapping.get(label)


def _load_spacy(language: str):
    """Load spaCy model. Returns None if spaCy not installed."""
    try:
        from src.pipeline.indexer import _get_spacy_nlp
        return _get_spacy_nlp(language)
    except ImportError:
        logger.debug("spaCy not available for NER relation extraction")
        return None


# ── 3c. Extractive summary (TF-IDF + cosine) ─────────────────────


def extractive_summary(facts: list[str], n_sentences: int = 3) -> list[str]:
    """Select the N most representative facts using TF-IDF + cosine similarity.

    Falls back to returning the first N facts if scikit-learn is not installed.
    """
    if len(facts) <= n_sentences:
        return list(facts)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        logger.debug("scikit-learn not installed, using simple fallback for extractive summary")
        return facts[:n_sentences]

    # Strip markdown formatting for better TF-IDF
    clean = [_strip_fact_markup(f) for f in facts]

    # Handle edge case: all facts identical after cleanup
    if len(set(clean)) <= 1:
        return facts[:n_sentences]

    try:
        vectorizer = TfidfVectorizer(stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(clean)
    except ValueError:
        return facts[:n_sentences]

    # Score each fact by its average cosine similarity to all others
    sim_matrix = cosine_similarity(tfidf_matrix)
    scores = sim_matrix.mean(axis=1)

    # Select top-N indices
    top_indices = np.argsort(scores)[-n_sentences:][::-1]
    # Return in original order
    top_indices_sorted = sorted(top_indices)

    return [facts[i] for i in top_indices_sorted]


def _strip_fact_markup(line: str) -> str:
    """Remove markdown observation markup for cleaner TF-IDF input."""
    # Remove leading "- [category] (date)" prefix
    m = re.match(r"^- \[\w+\]\s*(?:\([^)]*\)\s*)?(.+)", line)
    text = m.group(1) if m else line
    # Remove valence markers and tags
    text = re.sub(r"\[[\+\-\~]\]", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"\[superseded\]", "", text)
    return text.strip()
