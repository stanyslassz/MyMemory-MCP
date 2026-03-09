"""Tests for dream agent LLM coordinator models."""

from src.core.models import DreamPlan, DreamValidation


def test_dream_plan_model():
    """DreamPlan should accept valid step lists."""
    plan = DreamPlan(steps=[1, 3, 8, 9], reasoning="Consolidation needed")
    assert 1 in plan.steps
    assert plan.reasoning == "Consolidation needed"


def test_dream_validation_model():
    """DreamValidation should accept approval with optional issues."""
    v = DreamValidation(approved=True, issues=[])
    assert v.approved

    v2 = DreamValidation(approved=False, issues=["Bad merge: X into Y"])
    assert not v2.approved
    assert len(v2.issues) == 1
