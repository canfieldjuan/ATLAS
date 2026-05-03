"""Smoke tests for the public types promoted by PR-C1c.

Locks the public-contract shape for:

  - `ConclusionResult` and `SuppressionResult` (newly promoted public
    supporting types added in PR-C1c; consumed by the slim
    `EvidenceEngine` consolidation in PR-C1d).
  - The four temporal sub-types now exposed alongside `TemporalEvidence`
    (`VendorVelocity`, `LongTermTrend`, `CategoryPercentile`,
    `AnomalyScore`).

The existing 18 temporal tests in
`test_extracted_reasoning_core_temporal.py` already exercise these
shapes through `temporal.py`; this file just nails the public-import
path and the new evidence-result shapes that don't go through temporal.
"""

from __future__ import annotations

from extracted_reasoning_core.types import (
    AnomalyScore,
    CategoryPercentile,
    ConclusionResult,
    LongTermTrend,
    SuppressionResult,
    TemporalEvidence,
    VendorVelocity,
)


# ------------------------------------------------------------------
# ConclusionResult / SuppressionResult (newly promoted)
# ------------------------------------------------------------------


def test_conclusion_result_required_fields_and_defaults() -> None:
    result = ConclusionResult(
        conclusion_id="urgency_threshold_breached",
        met=True,
        confidence="high",
    )
    assert result.conclusion_id == "urgency_threshold_breached"
    assert result.met is True
    assert result.confidence == "high"
    assert result.fallback_label is None
    assert result.fallback_action is None


def test_conclusion_result_optional_fallback_fields() -> None:
    result = ConclusionResult(
        conclusion_id="insufficient_data",
        met=True,
        confidence="insufficient",
        fallback_label="Not enough recent reviews",
        fallback_action="suppress_section",
    )
    assert result.fallback_label == "Not enough recent reviews"
    assert result.fallback_action == "suppress_section"


def test_conclusion_result_is_frozen() -> None:
    result = ConclusionResult(
        conclusion_id="x", met=False, confidence="low",
    )
    try:
        result.met = True  # type: ignore[misc]
    except Exception as exc:
        assert exc.__class__.__name__ in {"FrozenInstanceError", "AttributeError"}
    else:
        raise AssertionError("frozen dataclass should reject reassignment")


def test_suppression_result_defaults() -> None:
    result = SuppressionResult()
    assert result.suppress is False
    assert result.degrade is False
    assert result.disclaimer is None
    assert result.fallback_label is None


def test_suppression_result_explicit_values() -> None:
    result = SuppressionResult(
        suppress=True,
        degrade=False,
        disclaimer="Insufficient data for confidence above 80%",
        fallback_label="Limited insight",
    )
    assert result.suppress is True
    assert result.disclaimer == "Insufficient data for confidence above 80%"


def test_suppression_result_is_frozen() -> None:
    result = SuppressionResult()
    try:
        result.suppress = True  # type: ignore[misc]
    except Exception as exc:
        assert exc.__class__.__name__ in {"FrozenInstanceError", "AttributeError"}
    else:
        raise AssertionError("frozen dataclass should reject reassignment")


# ------------------------------------------------------------------
# Temporal sub-types: public-import path
# ------------------------------------------------------------------


def test_temporal_subtypes_importable_from_types_module() -> None:
    # Pure existence check; the temporal_test file already exercises
    # behavior. This locks the public-import path that PR-C1d / api.py
    # wiring will consume.
    v = VendorVelocity(
        vendor_name="acme",
        metric="avg_urgency",
        current_value=7.0,
        previous_value=5.0,
        velocity=0.2,
        days_between=10,
    )
    assert v.vendor_name == "acme"

    t = LongTermTrend(metric="avg_urgency", slope_30d=0.05, data_points=14)
    assert t.metric == "avg_urgency"

    p = CategoryPercentile(
        product_category="crm",
        metric="avg_urgency",
        p25=4.0, p50=5.0, p75=6.0,
        sample_count=12,
    )
    assert p.p50 == 5.0

    a = AnomalyScore(
        vendor_name="acme",
        metric="avg_urgency",
        value=8.0,
        z_score=2.5,
        p_value=0.01,
        is_anomaly=True,
    )
    assert a.is_anomaly is True


def test_temporal_evidence_rich_shape() -> None:
    # Rich shape requires vendor_name + snapshot_days; collections
    # default to empty lists. Replaces PR #79's coarse Mapping[str, Any]
    # placeholder per the PR #82 audit's contract amendment.
    te = TemporalEvidence(vendor_name="acme", snapshot_days=30)
    assert te.vendor_name == "acme"
    assert te.snapshot_days == 30
    assert te.velocities == []
    assert te.trends == []
    assert te.anomalies == []
    assert te.category_baselines == []
    assert te.insufficient_data is False


def test_temporal_evidence_carries_subtype_lists() -> None:
    v = VendorVelocity(
        vendor_name="acme",
        metric="avg_urgency",
        current_value=7.0,
        previous_value=5.0,
        velocity=0.2,
        days_between=10,
    )
    te = TemporalEvidence(
        vendor_name="acme",
        snapshot_days=30,
        velocities=[v],
    )
    assert te.velocities == [v]
    assert isinstance(te.velocities[0], VendorVelocity)
