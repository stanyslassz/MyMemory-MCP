"""Tests for deterministic dream coordinator and validation (Tasks 24 & 25)."""

from src.pipeline.dream import decide_dream_steps, validate_dream_step


# ── Task 24: decide_dream_steps ─────────────────────────────

def test_decide_dream_steps_load_only():
    """With no candidates, only step 1 is selected."""
    stats = {
        "unextracted_docs": 0,
        "consolidation_candidates": 0,
        "merge_candidates": 0,
        "relation_candidates": 0,
        "transitive_candidates": 0,
        "prune_candidates": 0,
        "summary_candidates": 0,
    }
    assert decide_dream_steps(stats) == [1]


def test_decide_dream_steps_all_candidates():
    """With all candidates above thresholds, all steps selected."""
    stats = {
        "unextracted_docs": 1,
        "consolidation_candidates": 3,
        "merge_candidates": 2,
        "relation_candidates": 5,
        "transitive_candidates": 3,
        "prune_candidates": 1,
        "summary_candidates": 3,
    }
    result = decide_dream_steps(stats)
    assert result == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_decide_dream_steps_rescore_rebuild_when_work_done():
    """Steps 9 and 10 included when any work step is present."""
    stats = {
        "unextracted_docs": 0,
        "consolidation_candidates": 0,
        "merge_candidates": 0,
        "relation_candidates": 0,
        "transitive_candidates": 0,
        "prune_candidates": 1,  # Only prune
        "summary_candidates": 0,
    }
    result = decide_dream_steps(stats)
    assert 9 in result
    assert 10 in result
    assert 7 in result


def test_decide_dream_steps_no_rescore_without_work():
    """Steps 9 and 10 NOT included when only step 1 runs."""
    stats = {
        "unextracted_docs": 0,
        "consolidation_candidates": 2,  # Below threshold of 3
        "merge_candidates": 1,  # Below threshold of 2
        "relation_candidates": 4,  # Below threshold of 5
        "transitive_candidates": 2,  # Below threshold of 3
        "prune_candidates": 0,
        "summary_candidates": 2,  # Below threshold of 3
    }
    result = decide_dream_steps(stats)
    assert result == [1]


def test_decide_dream_steps_thresholds():
    """Test exact threshold boundaries."""
    # consolidation: 3 triggers, 2 does not
    assert 3 not in decide_dream_steps({"consolidation_candidates": 2})
    assert 3 in decide_dream_steps({"consolidation_candidates": 3})

    # merge: 2 triggers, 1 does not
    assert 4 not in decide_dream_steps({"merge_candidates": 1})
    assert 4 in decide_dream_steps({"merge_candidates": 2})

    # relations: 5 triggers, 4 does not
    assert 5 not in decide_dream_steps({"relation_candidates": 4})
    assert 5 in decide_dream_steps({"relation_candidates": 5})

    # transitive: 3 triggers, 2 does not
    assert 6 not in decide_dream_steps({"transitive_candidates": 2})
    assert 6 in decide_dream_steps({"transitive_candidates": 3})

    # prune: 1 triggers, 0 does not
    assert 7 not in decide_dream_steps({"prune_candidates": 0})
    assert 7 in decide_dream_steps({"prune_candidates": 1})

    # summary: 3 triggers, 2 does not
    assert 8 not in decide_dream_steps({"summary_candidates": 2})
    assert 8 in decide_dream_steps({"summary_candidates": 3})


def test_decide_dream_steps_empty_stats():
    """Empty dict yields only step 1."""
    assert decide_dream_steps({}) == [1]


def test_decide_dream_steps_returns_sorted():
    """Steps are always sorted."""
    stats = {"summary_candidates": 10, "unextracted_docs": 1}
    result = decide_dream_steps(stats)
    assert result == sorted(result)


# ── Task 25: validate_dream_step ─────────────────────────────

def test_validate_consolidation_ok():
    """Consolidation that reduced facts should pass."""
    before = {"total_facts": 20}
    after = {"total_facts": 12}
    approved, issues = validate_dream_step(3, before, after)
    assert approved
    assert issues == []


def test_validate_consolidation_increased_facts():
    """Consolidation that increased facts should fail."""
    before = {"total_facts": 10}
    after = {"total_facts": 15}
    approved, issues = validate_dream_step(3, before, after)
    assert not approved
    assert "increased fact count" in issues[0]


def test_validate_merge_ok():
    """Merge that reduced entities should pass."""
    before = {"total_entities": 50}
    after = {"total_entities": 48}
    approved, issues = validate_dream_step(4, before, after)
    assert approved
    assert issues == []


def test_validate_merge_increased_entities():
    """Merge that increased entities should fail."""
    before = {"total_entities": 50}
    after = {"total_entities": 52}
    approved, issues = validate_dream_step(4, before, after)
    assert not approved
    assert "increased entity count" in issues[0]


def test_validate_relation_discovery_ok():
    """Normal relation discovery should pass."""
    before = {"total_relations": 20}
    after = {"total_relations": 25}
    approved, issues = validate_dream_step(5, before, after)
    assert approved


def test_validate_relation_discovery_too_many():
    """Adding > 50 relations in one step is suspicious."""
    before = {"total_relations": 20}
    after = {"total_relations": 80}
    approved, issues = validate_dream_step(5, before, after)
    assert not approved
    assert "60" in issues[0]


def test_validate_unhandled_step():
    """Steps without specific rules should always pass."""
    approved, issues = validate_dream_step(7, {}, {})
    assert approved
    assert issues == []


def test_validate_missing_keys():
    """Missing keys default to 0, should pass for most cases."""
    approved, issues = validate_dream_step(3, {}, {})
    assert approved  # 0 > 0 is False, so no issue

    approved, issues = validate_dream_step(4, {}, {})
    assert approved

    approved, issues = validate_dream_step(5, {}, {})
    assert approved
