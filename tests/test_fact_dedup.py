"""Tests for deterministic fact dedup (Task 27)."""

from src.memory.store import _dedup_facts_deterministic


def test_dedup_removes_near_duplicates():
    """Near-duplicate facts should be removed."""
    facts = [
        "- [fact] Travaille chez Airbus comme développeur",
        "- [fact] Travaille chez Airbus comme developpeur",  # accents differ
        "- [fact] Habite à Toulouse",
    ]
    result = _dedup_facts_deterministic(facts)
    assert len(result) == 2
    assert facts[0] in result  # First kept
    assert facts[2] in result  # Distinct kept


def test_dedup_keeps_distinct_facts():
    """Clearly different facts should all be kept."""
    facts = [
        "- [fact] Travaille chez Airbus",
        "- [preference] Aime le yoga",
        "- [diagnosis] Sciatique chronique",
    ]
    result = _dedup_facts_deterministic(facts)
    assert len(result) == 3


def test_dedup_empty_list():
    """Empty input returns empty output."""
    assert _dedup_facts_deterministic([]) == []


def test_dedup_single_fact():
    """Single fact is always kept."""
    facts = ["- [fact] Some fact"]
    assert _dedup_facts_deterministic(facts) == facts


def test_dedup_exact_duplicates():
    """Exact duplicates (ratio=1.0) should be removed."""
    facts = [
        "- [fact] Travaille chez Airbus",
        "- [fact] Travaille chez Airbus",
    ]
    result = _dedup_facts_deterministic(facts)
    assert len(result) == 1


def test_dedup_custom_threshold():
    """Lower threshold removes more, higher threshold keeps more."""
    facts = [
        "- [fact] Aime les randonnees en montagne",
        "- [fact] Aime les promenades en foret",
    ]
    # With high threshold (0.95), these are different enough to keep both
    result_high = _dedup_facts_deterministic(facts, threshold=0.95)
    assert len(result_high) == 2

    # With very low threshold (0.50), they look similar enough to drop
    result_low = _dedup_facts_deterministic(facts, threshold=0.50)
    assert len(result_low) == 1
