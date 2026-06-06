"""Unit tests for the content-ops review vocabulary (slice 1).

Pure value types + helpers; no DB, no async, no Atlas imports.
"""

from __future__ import annotations

from datetime import date

import pytest

from extracted_content_pipeline.review_contract import (
    REQUIRED_REVIEW_BY_TIER,
    ExceptionRecord,
    FailureCategory,
    GateStage,
    ReviewDecision,
    RiskTier,
    recurring_failure_categories,
    required_stages_for,
)


def test_enums_are_str_valued() -> None:
    # StrEnum members compare equal to their string value (host-extensible
    # string statuses can interoperate without casting).
    assert RiskTier.LOW == "low"
    assert ReviewDecision.APPROVED_WITH_EXCEPTION == "approved_with_exception"
    assert FailureCategory.UNCLEAR_PROMISE == "unclear_promise"
    assert GateStage.CLAIMS_COMPLIANCE == "claims_compliance"


def test_failure_taxonomy_is_the_documented_fifteen() -> None:
    assert len(list(FailureCategory)) == 15


def test_gate_stack_is_the_four_part_split() -> None:
    assert [s.value for s in GateStage] == [
        "schema",
        "claims_compliance",
        "model_assisted",
        "human_editor",
    ]


def test_review_decision_includes_exception_and_escalation() -> None:
    values = {d.value for d in ReviewDecision}
    assert {"approved_with_exception", "escalated", "blocked"} <= values


@pytest.mark.parametrize("tier", list(RiskTier))
def test_every_tier_routes_through_schema_and_human_editor(tier: RiskTier) -> None:
    stages = required_stages_for(tier)
    # Every tier must start at schema and end at an accountable human decision.
    assert stages[0] is GateStage.SCHEMA
    assert stages[-1] is GateStage.HUMAN_EDITOR
    # No duplicate stages, and the mapping is complete.
    assert len(stages) == len(set(stages))
    assert tier in REQUIRED_REVIEW_BY_TIER


def test_low_tier_skips_claims_compliance_but_others_require_it() -> None:
    assert GateStage.CLAIMS_COMPLIANCE not in required_stages_for(RiskTier.LOW)
    for tier in (RiskTier.MEDIUM, RiskTier.HIGH, RiskTier.CRITICAL):
        assert GateStage.CLAIMS_COMPLIANCE in required_stages_for(tier)


def test_exception_record_without_expiration_never_lapses() -> None:
    rec = ExceptionRecord(
        rule="VOICE-02",
        reason="approved launch language for this campaign",
        owner="editor@example.com",
    )
    assert rec.is_active(date(2999, 1, 1)) is True
    assert rec.should_update_rule is False


def test_exception_record_expiration_is_inclusive() -> None:
    rec = ExceptionRecord(
        rule="CLAIM-03",
        reason="temporary Q4 pricing waiver",
        owner="legal@example.com",
        expiration=date(2026, 1, 15),
    )
    assert rec.is_active(date(2026, 1, 15)) is True   # inclusive of the day
    assert rec.is_active(date(2026, 1, 14)) is True
    assert rec.is_active(date(2026, 1, 16)) is False


def test_exception_record_is_frozen() -> None:
    rec = ExceptionRecord(rule="R", reason="x", owner="o")
    with pytest.raises(Exception):
        rec.rule = "other"  # type: ignore[misc]


def test_recurring_failures_flags_at_threshold() -> None:
    cats = [
        FailureCategory.UNCLEAR_PROMISE,
        FailureCategory.UNCLEAR_PROMISE,
        FailureCategory.UNCLEAR_PROMISE,
        FailureCategory.CTA_FRICTION,
        FailureCategory.CTA_FRICTION,
        FailureCategory.WEAK_HOOK,
    ]
    # Default threshold of 3: only unclear_promise qualifies.
    assert recurring_failure_categories(cats) == frozenset(
        {FailureCategory.UNCLEAR_PROMISE}
    )


def test_recurring_failures_respects_custom_threshold() -> None:
    cats = [FailureCategory.CTA_FRICTION, FailureCategory.CTA_FRICTION]
    assert recurring_failure_categories(cats, threshold=2) == frozenset(
        {FailureCategory.CTA_FRICTION}
    )


def test_recurring_failures_empty_input_is_empty() -> None:
    assert recurring_failure_categories([]) == frozenset()


def test_recurring_failures_rejects_nonpositive_threshold() -> None:
    with pytest.raises(ValueError):
        recurring_failure_categories([FailureCategory.TIMING], threshold=0)
