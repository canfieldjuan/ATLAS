"""Phase 10 Patch 1: pin the ProductClaim contract.

The shared envelope every UI card and report section consumes lives or
dies by its gate semantics. If render_allowed and report_allowed do
not behave the way the contract spec says, every downstream patch
(Vendor Workspace, Evidence UI, Opportunities, Challenger, Reports)
inherits the wrong behavior.

These tests pin:

  - Every (posture, confidence) combination produces the right gate
    pair, so a future change can't accidentally loosen report_allowed.
  - Hard blocks (unverified, insufficient, below-threshold counts)
    reject from both render AND report, never just one.
  - Rate-claim denominator gating fires only when is_rate_claim=True.
  - claim_id is deterministic across re-runs of the same inputs and
    differs across different inputs.
  - build_product_claim() never lets a caller hand-set render_allowed
    / report_allowed past what decide_render_gates() would have
    chosen.
"""

from __future__ import annotations

from datetime import date

import pytest

from atlas_brain.services.b2b.product_claim import (
    ClaimScope,
    ConfidenceLabel,
    EvidencePosture,
    ProductClaim,
    SuppressionReason,
    build_product_claim,
    compute_claim_id,
    decide_render_gates,
)


# ----------------------------------------------------------------------------
# Hard-block tests: rejected from BOTH surfaces.
# ----------------------------------------------------------------------------


def test_unverified_posture_blocks_both_surfaces():
    render, report, reason = decide_render_gates(
        evidence_posture=EvidencePosture.UNVERIFIED,
        confidence=ConfidenceLabel.HIGH,  # high confidence cannot rescue
        supporting_count=100,             # large counts cannot rescue
        direct_evidence_count=50,
        contradiction_count=0,
        denominator=1000,
        sample_size=2000,
    )
    assert render is False
    assert report is False
    assert reason == SuppressionReason.UNVERIFIED_EVIDENCE


def test_insufficient_posture_blocks_both_surfaces():
    render, report, reason = decide_render_gates(
        evidence_posture=EvidencePosture.INSUFFICIENT,
        confidence=ConfidenceLabel.HIGH,
        supporting_count=100,
        direct_evidence_count=50,
        contradiction_count=0,
        denominator=None,
        sample_size=None,
    )
    assert render is False
    assert report is False
    assert reason == SuppressionReason.INSUFFICIENT_SUPPORTING_COUNT


def test_supporting_count_below_threshold_blocks_both():
    render, report, reason = decide_render_gates(
        evidence_posture=EvidencePosture.USABLE,
        confidence=ConfidenceLabel.HIGH,
        supporting_count=2,  # default threshold is 3
        direct_evidence_count=2,
        contradiction_count=0,
        denominator=None,
        sample_size=None,
    )
    assert render is False
    assert report is False
    assert reason == SuppressionReason.INSUFFICIENT_SUPPORTING_COUNT


def test_direct_evidence_zero_blocks_both():
    render, report, reason = decide_render_gates(
        evidence_posture=EvidencePosture.USABLE,
        confidence=ConfidenceLabel.HIGH,
        supporting_count=10,
        direct_evidence_count=0,  # no quote-grade evidence at all
        contradiction_count=0,
        denominator=None,
        sample_size=None,
    )
    assert render is False
    assert report is False
    assert reason == SuppressionReason.WEAK_EVIDENCE_ONLY


# ----------------------------------------------------------------------------
# Soft-block tests: render allowed (UI shows monitor-only), report blocked.
# ----------------------------------------------------------------------------


def test_weak_posture_renders_but_does_not_publish():
    render, report, reason = decide_render_gates(
        evidence_posture=EvidencePosture.WEAK,
        confidence=ConfidenceLabel.HIGH,
        supporting_count=10,
        direct_evidence_count=5,
        contradiction_count=0,
        denominator=None,
        sample_size=None,
    )
    assert render is True
    assert report is False
    assert reason == SuppressionReason.WEAK_EVIDENCE_ONLY


def test_contradictory_posture_renders_but_does_not_publish():
    render, report, reason = decide_render_gates(
        evidence_posture=EvidencePosture.CONTRADICTORY,
        confidence=ConfidenceLabel.HIGH,
        supporting_count=10,
        direct_evidence_count=5,
        contradiction_count=4,
        denominator=None,
        sample_size=None,
    )
    assert render is True
    assert report is False
    assert reason == SuppressionReason.CONTRADICTORY_EVIDENCE


def test_low_confidence_renders_but_does_not_publish():
    render, report, reason = decide_render_gates(
        evidence_posture=EvidencePosture.USABLE,
        confidence=ConfidenceLabel.LOW,
        supporting_count=10,
        direct_evidence_count=5,
        contradiction_count=0,
        denominator=None,
        sample_size=None,
    )
    assert render is True
    assert report is False
    assert reason == SuppressionReason.LOW_CONFIDENCE


# ----------------------------------------------------------------------------
# Pass-through: usable + high/medium confidence + counts above threshold.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [ConfidenceLabel.HIGH, ConfidenceLabel.MEDIUM])
def test_usable_high_or_medium_confidence_passes_both_gates(confidence):
    render, report, reason = decide_render_gates(
        evidence_posture=EvidencePosture.USABLE,
        confidence=confidence,
        supporting_count=10,
        direct_evidence_count=5,
        contradiction_count=0,
        denominator=None,
        sample_size=None,
    )
    assert render is True
    assert report is True
    assert reason is None


# ----------------------------------------------------------------------------
# Rate-claim denominator gate.
# ----------------------------------------------------------------------------


def test_rate_claim_with_unknown_denominator_blocks_both():
    render, report, reason = decide_render_gates(
        evidence_posture=EvidencePosture.USABLE,
        confidence=ConfidenceLabel.HIGH,
        supporting_count=10,
        direct_evidence_count=5,
        contradiction_count=0,
        denominator=None,
        sample_size=100,
        is_rate_claim=True,
    )
    assert render is False
    assert report is False
    assert reason == SuppressionReason.DENOMINATOR_UNKNOWN


def test_rate_claim_with_below_threshold_denominator_blocks_both():
    render, report, reason = decide_render_gates(
        evidence_posture=EvidencePosture.USABLE,
        confidence=ConfidenceLabel.HIGH,
        supporting_count=10,
        direct_evidence_count=5,
        contradiction_count=0,
        denominator=5,  # below default threshold of 10
        sample_size=100,
        is_rate_claim=True,
    )
    assert render is False
    assert report is False
    assert reason == SuppressionReason.SAMPLE_SIZE_BELOW_THRESHOLD


def test_non_rate_claim_does_not_require_denominator():
    render, report, _ = decide_render_gates(
        evidence_posture=EvidencePosture.USABLE,
        confidence=ConfidenceLabel.HIGH,
        supporting_count=10,
        direct_evidence_count=5,
        contradiction_count=0,
        denominator=None,
        sample_size=None,
        is_rate_claim=False,
    )
    assert render is True
    assert report is True


# ----------------------------------------------------------------------------
# Invariants: report_allowed implies render_allowed (one-way, never both off).
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "posture",
    list(EvidencePosture),
)
@pytest.mark.parametrize(
    "confidence",
    list(ConfidenceLabel),
)
def test_report_allowed_implies_render_allowed_across_all_combos(posture, confidence):
    render, report, _ = decide_render_gates(
        evidence_posture=posture,
        confidence=confidence,
        supporting_count=10,
        direct_evidence_count=5,
        contradiction_count=0,
        denominator=None,
        sample_size=None,
    )
    if report:
        assert render, (
            f"report_allowed=True without render_allowed for "
            f"posture={posture.value} confidence={confidence.value} -- "
            "the report gate must be strictly tighter than the render gate"
        )


@pytest.mark.parametrize(
    "posture",
    list(EvidencePosture),
)
@pytest.mark.parametrize(
    "confidence",
    list(ConfidenceLabel),
)
def test_only_usable_high_or_medium_confidence_publishes(posture, confidence):
    """The report gate must publish if and only if posture is usable
    AND confidence is high/medium AND counts are above thresholds.
    Pin the truth table so a future change can't drift."""
    render, report, _ = decide_render_gates(
        evidence_posture=posture,
        confidence=confidence,
        supporting_count=10,
        direct_evidence_count=5,
        contradiction_count=0,
        denominator=None,
        sample_size=None,
    )
    expected_report = (
        posture == EvidencePosture.USABLE
        and confidence in {ConfidenceLabel.HIGH, ConfidenceLabel.MEDIUM}
    )
    assert report is expected_report


# ----------------------------------------------------------------------------
# claim_id determinism + variance.
# ----------------------------------------------------------------------------


def test_claim_id_deterministic_across_runs():
    args = dict(
        claim_scope=ClaimScope.VENDOR,
        claim_type="weakness_theme",
        target_entity="Asana",
        secondary_target=None,
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
    )
    a = compute_claim_id(**args)
    b = compute_claim_id(**args)
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_claim_id_varies_with_target_and_date():
    base = dict(
        claim_scope=ClaimScope.VENDOR,
        claim_type="weakness_theme",
        target_entity="Asana",
        secondary_target=None,
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
    )
    by_target = compute_claim_id(**{**base, "target_entity": "Pipedrive"})
    by_date = compute_claim_id(**{**base, "as_of_date": date(2026, 4, 27)})
    by_window = compute_claim_id(**{**base, "analysis_window_days": 30})
    by_secondary = compute_claim_id(**{**base, "secondary_target": "HubSpot"})
    base_id = compute_claim_id(**base)
    assert len({base_id, by_target, by_date, by_window, by_secondary}) == 5


# ----------------------------------------------------------------------------
# build_product_claim() integration.
# ----------------------------------------------------------------------------


def test_build_product_claim_derives_gates_consistently():
    """The single supported entry point. Callers cannot bypass the
    gate logic; if they pass weak posture, report_allowed is False
    even if they intended otherwise."""
    claim = build_product_claim(
        claim_scope=ClaimScope.VENDOR,
        claim_type="weakness_theme",
        claim_text="Pricing renewal pressure",
        target_entity="Asana",
        secondary_target=None,
        supporting_count=20,
        direct_evidence_count=8,
        witness_count=12,
        contradiction_count=0,
        denominator=None,
        sample_size=None,
        confidence=ConfidenceLabel.HIGH,
        evidence_posture=EvidencePosture.WEAK,  # weak -> publishable=False
        evidence_links=("witness:1", "witness:2"),
        contradicting_links=(),
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
    )
    assert isinstance(claim, ProductClaim)
    assert claim.render_allowed is True
    assert claim.report_allowed is False
    assert claim.suppression_reason == SuppressionReason.WEAK_EVIDENCE_ONLY
    assert claim.claim_id  # populated by helper
    assert claim.evidence_links == ("witness:1", "witness:2")


def test_build_product_claim_publishable_path():
    claim = build_product_claim(
        claim_scope=ClaimScope.VENDOR,
        claim_type="weakness_theme",
        claim_text="Pricing renewal pressure",
        target_entity="Asana",
        secondary_target=None,
        supporting_count=20,
        direct_evidence_count=8,
        witness_count=12,
        contradiction_count=0,
        denominator=None,
        sample_size=None,
        confidence=ConfidenceLabel.HIGH,
        evidence_posture=EvidencePosture.USABLE,
        evidence_links=("witness:1",),
        contradicting_links=(),
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
    )
    assert claim.render_allowed is True
    assert claim.report_allowed is True
    assert claim.suppression_reason is None


def test_product_claim_dataclass_is_frozen():
    """Mutability would let a downstream consumer flip report_allowed
    after construction. The contract requires the gate to be set once
    by build_product_claim(); freezing prevents drift."""
    claim = build_product_claim(
        claim_scope=ClaimScope.VENDOR,
        claim_type="weakness_theme",
        claim_text="x",
        target_entity="V",
        secondary_target=None,
        supporting_count=20,
        direct_evidence_count=8,
        witness_count=12,
        contradiction_count=0,
        denominator=None,
        sample_size=None,
        confidence=ConfidenceLabel.HIGH,
        evidence_posture=EvidencePosture.WEAK,
        evidence_links=(),
        contradicting_links=(),
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
    )
    with pytest.raises((AttributeError, Exception)):
        claim.report_allowed = True  # type: ignore[misc]
