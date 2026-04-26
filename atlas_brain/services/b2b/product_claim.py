"""ProductClaim contract — the shared envelope every UI card and every
report section consumes.

Phase 10 Patch 1. The operating rule is "UI first, reports inherit":
the dashboard / UI is the truth layer, and reports are downstream
renderings of the same validated objects. This module defines the
envelope and the deterministic gate helpers (render_allowed,
report_allowed, suppression_reason) so a report can never accidentally
publish a claim the UI already suppressed.

Pure deterministic, sync, pool-free. The DB layer (storage of
ProductClaim rows, aggregation from b2b_evidence_claims into VENDOR /
ACCOUNT / COMPETITOR_PAIR / ALERT scope rows) lands in Patch 2 onward
when the first consumer needs it. This module is the contract; it has
no opinion on persistence.

See docs/progress/product_claim_contract_plan_2026-04-26.md for the
design doc this contract pins.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class ClaimScope(StrEnum):
    """Discriminator for ProductClaim. Determines what target_entity
    means and which UI surface consumes the row."""

    WITNESS = "witness"
    VENDOR = "vendor"
    ACCOUNT = "account"
    COMPETITOR_PAIR = "competitor_pair"
    ALERT = "alert"


class EvidencePosture(StrEnum):
    """Quality of the evidence backing a claim. Drives both the render
    gate and the report gate (the report gate is strictly tighter)."""

    USABLE = "usable"
    """Grounded, sufficient, no contradictions. Renderable + publishable."""

    WEAK = "weak"
    """Grounded but thin (low supporting_count or low confidence).
    Renderable as 'monitor only'; not publishable."""

    CONTRADICTORY = "contradictory"
    """Grounded but contradicted by other evidence in the same window.
    Renderable WITH the contradiction surfaced; not publishable."""

    UNVERIFIED = "unverified"
    """Cannot ground (e.g. v3-backed witness, synthesized span). Audit
    only; blocked from both render and report."""

    INSUFFICIENT = "insufficient"
    """Below minimum supporting_count or denominator threshold. Audit
    only; blocked from both render and report."""


class ConfidenceLabel(StrEnum):
    """Coarse confidence bucket. Used by the report gate -- a claim
    with confidence=low never reaches a customer-facing report even
    if the posture is usable."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SuppressionReason(StrEnum):
    """Closed set of reasons render_allowed or report_allowed went
    False. The audit rolls these up so the operator can see WHY claims
    are dropping out, not just that they are."""

    INSUFFICIENT_SUPPORTING_COUNT = "insufficient_supporting_count"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    UNVERIFIED_EVIDENCE = "unverified_evidence"
    DENOMINATOR_UNKNOWN = "denominator_unknown"
    SAMPLE_SIZE_BELOW_THRESHOLD = "sample_size_below_threshold"
    WEAK_EVIDENCE_ONLY = "weak_evidence_only"
    PASSING_MENTION_ONLY = "passing_mention_only"
    LOW_CONFIDENCE = "low_confidence"
    CONSUMER_FILTER_APPLIED = "consumer_filter_applied"


# Default thresholds. Overridable per claim_type once the canary surfaces
# concrete tuning needs (Patch 2 onward).
_MIN_SUPPORTING_COUNT_DEFAULT = 3
_MIN_DIRECT_EVIDENCE_DEFAULT = 1
_MIN_DENOMINATOR_FOR_RATE_CLAIM = 10


@dataclass(frozen=True)
class ProductClaim:
    """Shared envelope every UI card and report section consumes.

    Construction note: the gate fields (render_allowed, report_allowed,
    suppression_reason) MUST be derived from the underlying counts +
    posture via decide_render_gates(). Setting them directly is a
    contract violation -- a consumer could otherwise loosen
    report_allowed past what the UI surface allowed.
    """

    claim_id: str
    claim_scope: ClaimScope
    claim_type: str
    claim_text: str
    target_entity: str
    secondary_target: str | None

    # Numerator / denominator / context.
    supporting_count: int
    direct_evidence_count: int
    witness_count: int
    contradiction_count: int
    denominator: int | None
    sample_size: int | None

    # Quality flags.
    confidence: ConfidenceLabel
    evidence_posture: EvidencePosture

    # Render gates -- derived; do not set independently.
    render_allowed: bool
    report_allowed: bool
    suppression_reason: SuppressionReason | None

    # Provenance.
    evidence_links: tuple[str, ...]
    contradicting_links: tuple[str, ...]

    as_of_date: date
    analysis_window_days: int
    schema_version: str = "v1"


def compute_claim_id(
    *,
    claim_scope: ClaimScope,
    claim_type: str,
    target_entity: str,
    secondary_target: str | None,
    as_of_date: date,
    analysis_window_days: int,
) -> str:
    """Deterministic id so the same logical claim across re-runs maps
    to the same id, which lets the React side cache and diff cleanly.

    Stable across runs that share the inputs; varies on different
    scopes / types / targets / windows / dates."""
    payload = (
        f"{claim_scope.value}::{claim_type}::{target_entity}::"
        f"{secondary_target or ''}::{as_of_date.isoformat()}::"
        f"{int(analysis_window_days)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def decide_render_gates(
    *,
    evidence_posture: EvidencePosture,
    confidence: ConfidenceLabel,
    supporting_count: int,
    direct_evidence_count: int,
    contradiction_count: int,
    denominator: int | None,
    sample_size: int | None,
    is_rate_claim: bool = False,
    min_supporting_count: int = _MIN_SUPPORTING_COUNT_DEFAULT,
    min_direct_evidence: int = _MIN_DIRECT_EVIDENCE_DEFAULT,
    min_denominator: int = _MIN_DENOMINATOR_FOR_RATE_CLAIM,
) -> tuple[bool, bool, SuppressionReason | None]:
    """Derive (render_allowed, report_allowed, suppression_reason) from
    the underlying counts + posture. Pure deterministic, no I/O.

    The two gates are NOT set independently per consumer: the report
    gate is strictly tighter than the render gate, so a report can
    never accidentally publish a claim the UI suppressed.

    Render gate (UI detail view OK):
      not unverified / insufficient
      AND supporting_count >= min_supporting_count
      AND direct_evidence_count >= min_direct_evidence
      AND (rate claims) denominator >= min_denominator

    Report gate (customer-facing publish OK):
      render_allowed
      AND posture == usable (not weak, not contradictory)
      AND confidence in {high, medium}
    """
    # Hard blocks first -- these reject from both surfaces.
    if evidence_posture == EvidencePosture.UNVERIFIED:
        return (False, False, SuppressionReason.UNVERIFIED_EVIDENCE)

    if evidence_posture == EvidencePosture.INSUFFICIENT:
        return (False, False, SuppressionReason.INSUFFICIENT_SUPPORTING_COUNT)

    if supporting_count < min_supporting_count:
        return (
            False,
            False,
            SuppressionReason.INSUFFICIENT_SUPPORTING_COUNT,
        )

    if direct_evidence_count < min_direct_evidence:
        return (
            False,
            False,
            SuppressionReason.WEAK_EVIDENCE_ONLY,
        )

    if is_rate_claim:
        if denominator is None:
            return (False, False, SuppressionReason.DENOMINATOR_UNKNOWN)
        if denominator < min_denominator:
            return (
                False,
                False,
                SuppressionReason.SAMPLE_SIZE_BELOW_THRESHOLD,
            )

    # Past the hard gate. Render is allowed; decide whether report is.
    render_allowed = True

    # Contradictory and weak postures are renderable (with appropriate
    # UI labels) but never publishable to a report.
    if evidence_posture == EvidencePosture.CONTRADICTORY:
        return (render_allowed, False, SuppressionReason.CONTRADICTORY_EVIDENCE)

    if evidence_posture == EvidencePosture.WEAK:
        return (render_allowed, False, SuppressionReason.WEAK_EVIDENCE_ONLY)

    # Posture is USABLE at this point. Confidence decides the report gate.
    if confidence == ConfidenceLabel.LOW:
        return (render_allowed, False, SuppressionReason.LOW_CONFIDENCE)

    # Usable + high/medium confidence + counts above thresholds.
    return (True, True, None)


def build_product_claim(
    *,
    claim_scope: ClaimScope,
    claim_type: str,
    claim_text: str,
    target_entity: str,
    secondary_target: str | None,
    supporting_count: int,
    direct_evidence_count: int,
    witness_count: int,
    contradiction_count: int,
    denominator: int | None,
    sample_size: int | None,
    confidence: ConfidenceLabel,
    evidence_posture: EvidencePosture,
    evidence_links: tuple[str, ...],
    contradicting_links: tuple[str, ...],
    as_of_date: date,
    analysis_window_days: int,
    is_rate_claim: bool = False,
    schema_version: str = "v1",
) -> ProductClaim:
    """Construct a ProductClaim with derived gate fields. The single
    supported entry point so callers cannot accidentally bypass the
    gate logic.

    is_rate_claim signals that the claim is a rate (e.g. churn rate,
    recommend ratio) and therefore must carry a denominator above the
    minimum threshold to be renderable. Theme / count-based claims
    pass False.
    """
    render_allowed, report_allowed, suppression_reason = decide_render_gates(
        evidence_posture=evidence_posture,
        confidence=confidence,
        supporting_count=supporting_count,
        direct_evidence_count=direct_evidence_count,
        contradiction_count=contradiction_count,
        denominator=denominator,
        sample_size=sample_size,
        is_rate_claim=is_rate_claim,
    )
    claim_id = compute_claim_id(
        claim_scope=claim_scope,
        claim_type=claim_type,
        target_entity=target_entity,
        secondary_target=secondary_target,
        as_of_date=as_of_date,
        analysis_window_days=analysis_window_days,
    )
    return ProductClaim(
        claim_id=claim_id,
        claim_scope=claim_scope,
        claim_type=claim_type,
        claim_text=claim_text,
        target_entity=target_entity,
        secondary_target=secondary_target,
        supporting_count=supporting_count,
        direct_evidence_count=direct_evidence_count,
        witness_count=witness_count,
        contradiction_count=contradiction_count,
        denominator=denominator,
        sample_size=sample_size,
        confidence=confidence,
        evidence_posture=evidence_posture,
        render_allowed=render_allowed,
        report_allowed=report_allowed,
        suppression_reason=suppression_reason,
        evidence_links=evidence_links,
        contradicting_links=contradicting_links,
        as_of_date=as_of_date,
        analysis_window_days=analysis_window_days,
        schema_version=schema_version,
    )


__all__ = [
    "ClaimScope",
    "EvidencePosture",
    "ConfidenceLabel",
    "SuppressionReason",
    "ProductClaim",
    "compute_claim_id",
    "decide_render_gates",
    "build_product_claim",
]
