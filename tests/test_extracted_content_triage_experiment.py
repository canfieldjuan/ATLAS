"""Unit tests for slice 2: stage-0 triage + stage-7 experiment contract.

Pure value types + completeness helpers; no DB, no async, no Atlas imports.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from extracted_content_pipeline.review_contract import (
    ExperimentContract,
    RiskTier,
    TriageDecision,
    TriageRequest,
)


def _complete_triage() -> TriageRequest:
    return TriageRequest(
        audience_segment="enterprise admins",
        lifecycle_stage="onboarding",
        business_goal="activation",
        expected_behavior_change="complete first integration",
        channel="lifecycle email",
        why_now="new onboarding flow shipped",
        risk_tier=RiskTier.MEDIUM,
    )


def _complete_experiment() -> ExperimentContract:
    return ExperimentContract(
        hypothesis="a shorter subject lifts opens",
        primary_metric="open_rate",
        secondary_metric="unsubscribe_rate",
        audience="trial users day 3",
        comparison="current subject line",
        success_definition="+3pp open rate",
        inconclusive_definition="<1pp difference",
        decision_if_works="roll out to all trials",
        decision_if_not="revert",
        attribution_window_days=7,
        min_sample_size=500,
    )


def test_triage_decision_values() -> None:
    assert {d.value for d in TriageDecision} == {
        "create",
        "clone_winner",
        "defer",
        "reject",
    }
    assert TriageDecision.CLONE_WINNER == "clone_winner"


def test_triage_complete_request_has_no_missing_fields() -> None:
    req = _complete_triage()
    assert req.missing_fields() == ()
    assert req.is_complete() is True


def test_triage_empty_request_lists_required_fields_in_order() -> None:
    req = TriageRequest()
    assert req.missing_fields() == (
        "audience_segment",
        "lifecycle_stage",
        "business_goal",
        "expected_behavior_change",
        "channel",
        "why_now",
        "risk_tier",
    )
    assert req.is_complete() is False


def test_triage_optional_context_does_not_gate_completeness() -> None:
    # opportunity_size / reuse_potential blank but everything required present.
    req = _complete_triage()
    assert req.opportunity_size == ""
    assert req.reuse_potential == ""
    assert req.is_complete() is True


def test_triage_whitespace_counts_as_missing() -> None:
    req = _complete_triage()
    holey = TriageRequest(
        audience_segment="   ",
        lifecycle_stage=req.lifecycle_stage,
        business_goal=req.business_goal,
        expected_behavior_change=req.expected_behavior_change,
        channel=req.channel,
        why_now=req.why_now,
        risk_tier=req.risk_tier,
    )
    assert "audience_segment" in holey.missing_fields()


def test_triage_missing_risk_tier_is_flagged() -> None:
    req = TriageRequest(
        audience_segment="a",
        lifecycle_stage="b",
        business_goal="c",
        expected_behavior_change="d",
        channel="e",
        why_now="f",
        risk_tier=None,
    )
    assert req.missing_fields() == ("risk_tier",)


def test_triage_request_is_frozen() -> None:
    req = _complete_triage()
    with pytest.raises(FrozenInstanceError):
        req.channel = "other"  # type: ignore[misc]


def test_triage_none_text_is_missing_not_error() -> None:
    # A None field (e.g. from JSON null) must report missing, not raise.
    req = TriageRequest(
        audience_segment=None,  # type: ignore[arg-type]
        lifecycle_stage="b",
        business_goal="c",
        expected_behavior_change="d",
        channel="e",
        why_now="f",
        risk_tier=RiskTier.LOW,
    )
    assert req.missing_fields() == ("audience_segment",)


def test_experiment_complete_contract_has_no_missing_fields() -> None:
    contract = _complete_experiment()
    assert contract.missing_fields() == ()
    assert contract.is_complete() is True


def test_experiment_empty_contract_lists_all_required() -> None:
    contract = ExperimentContract()
    missing = contract.missing_fields()
    assert missing == (
        "hypothesis",
        "primary_metric",
        "secondary_metric",
        "audience",
        "comparison",
        "success_definition",
        "inconclusive_definition",
        "decision_if_works",
        "decision_if_not",
        "attribution_window_days",
        "min_sample_size",
    )
    assert contract.is_complete() is False


def test_experiment_secondary_metric_is_required() -> None:
    # The canonical doc lists secondary_metric among the required stage-7 fields.
    contract = _complete_experiment()
    without = ExperimentContract(
        hypothesis=contract.hypothesis,
        primary_metric=contract.primary_metric,
        secondary_metric="",
        audience=contract.audience,
        comparison=contract.comparison,
        success_definition=contract.success_definition,
        inconclusive_definition=contract.inconclusive_definition,
        decision_if_works=contract.decision_if_works,
        decision_if_not=contract.decision_if_not,
        attribution_window_days=contract.attribution_window_days,
        min_sample_size=contract.min_sample_size,
    )
    assert without.missing_fields() == ("secondary_metric",)
    assert without.is_complete() is False


@pytest.mark.parametrize("window", [0, -1])
def test_experiment_nonpositive_window_is_missing(window: int) -> None:
    contract = ExperimentContract(
        hypothesis="h",
        primary_metric="m",
        secondary_metric="m2",
        audience="a",
        comparison="c",
        success_definition="s",
        inconclusive_definition="i",
        decision_if_works="w",
        decision_if_not="n",
        attribution_window_days=window,
        min_sample_size=100,
    )
    assert "attribution_window_days" in contract.missing_fields()
    assert "min_sample_size" not in contract.missing_fields()


def test_experiment_nonpositive_sample_size_is_missing() -> None:
    contract = _complete_experiment()
    bad = ExperimentContract(
        hypothesis=contract.hypothesis,
        primary_metric=contract.primary_metric,
        secondary_metric=contract.secondary_metric,
        audience=contract.audience,
        comparison=contract.comparison,
        success_definition=contract.success_definition,
        inconclusive_definition=contract.inconclusive_definition,
        decision_if_works=contract.decision_if_works,
        decision_if_not=contract.decision_if_not,
        attribution_window_days=contract.attribution_window_days,
        min_sample_size=0,
    )
    assert bad.missing_fields() == ("min_sample_size",)


def test_experiment_none_text_is_missing_not_error() -> None:
    contract = _complete_experiment()
    holey = ExperimentContract(
        hypothesis=None,  # type: ignore[arg-type]
        primary_metric=contract.primary_metric,
        secondary_metric=contract.secondary_metric,
        audience=contract.audience,
        comparison=contract.comparison,
        success_definition=contract.success_definition,
        inconclusive_definition=contract.inconclusive_definition,
        decision_if_works=contract.decision_if_works,
        decision_if_not=contract.decision_if_not,
        attribution_window_days=contract.attribution_window_days,
        min_sample_size=contract.min_sample_size,
    )
    assert holey.missing_fields() == ("hypothesis",)


def test_experiment_none_numeric_is_missing_not_error() -> None:
    # None / non-int numeric fields report missing, not raise TypeError.
    contract = _complete_experiment()
    holey = ExperimentContract(
        hypothesis=contract.hypothesis,
        primary_metric=contract.primary_metric,
        secondary_metric=contract.secondary_metric,
        audience=contract.audience,
        comparison=contract.comparison,
        success_definition=contract.success_definition,
        inconclusive_definition=contract.inconclusive_definition,
        decision_if_works=contract.decision_if_works,
        decision_if_not=contract.decision_if_not,
        attribution_window_days=None,  # type: ignore[arg-type]
        min_sample_size=None,  # type: ignore[arg-type]
    )
    missing = holey.missing_fields()
    assert "attribution_window_days" in missing
    assert "min_sample_size" in missing


def test_experiment_bool_sample_size_is_missing() -> None:
    # bool is an int subclass but is non-conforming for a count.
    contract = _complete_experiment()
    holey = ExperimentContract(
        hypothesis=contract.hypothesis,
        primary_metric=contract.primary_metric,
        secondary_metric=contract.secondary_metric,
        audience=contract.audience,
        comparison=contract.comparison,
        success_definition=contract.success_definition,
        inconclusive_definition=contract.inconclusive_definition,
        decision_if_works=contract.decision_if_works,
        decision_if_not=contract.decision_if_not,
        attribution_window_days=contract.attribution_window_days,
        min_sample_size=True,  # type: ignore[arg-type]
    )
    assert "min_sample_size" in holey.missing_fields()


def test_experiment_contract_is_frozen() -> None:
    contract = _complete_experiment()
    with pytest.raises(FrozenInstanceError):
        contract.primary_metric = "other"  # type: ignore[misc]
