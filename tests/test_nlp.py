"""Tests for src/pipeline/nlp.py — dedup, NER relations, extractive summary."""

from src.pipeline.nlp import (
    dedup_facts_deterministic,
    extract_relations_nlp,
    extractive_summary,
)


# ── dedup_facts_deterministic ─────────────────────────────────


def test_dedup_removes_obvious_duplicates():
    facts = [
        "- [fact] Chronic back pain since 2020",
        "- [fact] chronic back pain since 2020",
        "- [diagnosis] Sciatica diagnosed in 2021",
    ]
    result = dedup_facts_deterministic(facts, threshold=85.0)
    assert len(result) == 2
    assert result[0] == facts[0]
    assert result[1] == facts[2]


def test_dedup_keeps_distinct_facts():
    facts = [
        "- [fact] Enjoys swimming",
        "- [fact] Works at Acme Corp",
        "- [treatment] Takes ibuprofen",
    ]
    result = dedup_facts_deterministic(facts, threshold=85.0)
    assert len(result) == 3


def test_dedup_empty_list():
    assert dedup_facts_deterministic([], threshold=85.0) == []


def test_dedup_single_item():
    facts = ["- [fact] One fact"]
    assert dedup_facts_deterministic(facts) == facts


# ── extract_relations_nlp ─────────────────────────────────────


def test_extract_relations_returns_list():
    """Even without spaCy, should return an empty list (graceful fallback)."""
    result = extract_relations_nlp("Alice works at Google in Paris.", language="en")
    # If spaCy is installed, we expect relations; if not, empty list
    assert isinstance(result, list)


def test_extract_relations_typed():
    """If spaCy available, relations should have from/to/type keys."""
    result = extract_relations_nlp("Jean travaille chez Airbus à Toulouse.", language="fr")
    for rel in result:
        assert "from" in rel
        assert "to" in rel
        assert "type" in rel


# ── extractive_summary ────────────────────────────────────────


def test_extractive_summary_returns_top_n():
    facts = [
        "- [fact] Patient has chronic back pain",
        "- [treatment] Started physiotherapy sessions",
        "- [diagnosis] Sciatica confirmed by MRI",
        "- [fact] Regular swimming helps with pain",
        "- [preference] Prefers morning appointments",
        "- [fact] Pain worsens in cold weather",
    ]
    result = extractive_summary(facts, n_sentences=3)
    assert len(result) == 3
    # All returned facts should be from the original list
    for f in result:
        assert f in facts


def test_extractive_summary_fewer_than_n():
    facts = ["- [fact] Only one fact"]
    result = extractive_summary(facts, n_sentences=3)
    assert result == facts


def test_extractive_summary_empty():
    assert extractive_summary([], n_sentences=3) == []


def test_extractive_summary_preserves_original_order():
    facts = [
        "- [fact] A first fact about health",
        "- [fact] A second fact about work",
        "- [fact] A third fact about hobbies",
        "- [fact] A fourth fact about travel",
    ]
    result = extractive_summary(facts, n_sentences=2)
    # Check that returned facts maintain their relative order
    indices = [facts.index(f) for f in result]
    assert indices == sorted(indices)


# ── Graceful fallback tests ───────────────────────────────────


def test_dedup_works_without_rapidfuzz(monkeypatch):
    """Ensure dedup works even if rapidfuzz import fails."""
    import importlib
    import src.pipeline.nlp as nlp_mod

    # Force re-evaluation of similarity function by clearing any cache
    original_fn = nlp_mod._get_similarity_fn

    def _mock_get_similarity_fn():
        from difflib import SequenceMatcher
        def _ratio(a, b):
            return SequenceMatcher(None, a, b).ratio() * 100.0
        return _ratio

    monkeypatch.setattr(nlp_mod, "_get_similarity_fn", _mock_get_similarity_fn)
    facts = [
        "- [fact] Chronic back pain since 2020",
        "- [fact] chronic back pain since 2020",
    ]
    result = nlp_mod.dedup_facts_deterministic(facts, threshold=85.0)
    assert len(result) == 1
