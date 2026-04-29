"""ProductClaim contract — the shared envelope every UI card and every
report section consumes.

Phase 10 Patch 1 (post-audit hardening). Operating rule: UI first,
reports inherit. The dashboard / UI is the truth layer; reports are
downstream renderings of the same validated objects.

Contract design (audit-driven):

  - The caller provides OBJECTIVE FACTS (counts, witness_count,
    contradiction_count, denominator, evidence_links, claim_key).
    The builder DERIVES posture, confidence, and the render / report
    gates from those facts via a single ClaimGatePolicy. The caller
    does not make subjective quality calls anywhere; that is the
    only way a report can be guaranteed inheriting the UI's gating.

  - claim_id includes a required claim_key dimension so two claims
    of the same (scope, type, target) can coexist (e.g. weakness on
    pricing AND weakness on support for the same vendor).

  - Per-(claim_scope, claim_type) policies live in a registry so a
    claim type that needs stricter thresholds can register its own
    policy without changing global defaults. Production aggregators
    should require an explicit registry entry; the permissive default
    remains available only for tests, audit scenarios, and low-risk
    internal construction.

  - ProductClaim's __post_init__ re-derives posture, confidence, and
    gates from the stored fields and asserts equality. That blocks
    direct dataclass construction with hand-set gate fields and
    keeps build_product_claim() the only safe construction path.

Pure deterministic, sync, pool-free. Persistence and aggregation
land in Patch 2 onward.

See docs/progress/product_claim_contract_plan_2026-04-26.md.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum


class ClaimScope(StrEnum):
    WITNESS = "witness"
    VENDOR = "vendor"
    ACCOUNT = "account"
    COMPETITOR_PAIR = "competitor_pair"
    ALERT = "alert"


class EvidencePosture(StrEnum):
    """Quality of the evidence backing a claim. Derived, not caller-set."""

    USABLE = "usable"
    WEAK = "weak"
    CONTRADICTORY = "contradictory"
    UNVERIFIED = "unverified"
    INSUFFICIENT = "insufficient"


class ConfidenceLabel(StrEnum):
    """Coarse confidence bucket. Derived from supporting_count + witness_count
    via the active ClaimGatePolicy."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SuppressionReason(StrEnum):
    INSUFFICIENT_SUPPORTING_COUNT = "insufficient_supporting_count"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    UNVERIFIED_EVIDENCE = "unverified_evidence"
    DENOMINATOR_UNKNOWN = "denominator_unknown"
    SAMPLE_SIZE_BELOW_THRESHOLD = "sample_size_below_threshold"
    WEAK_EVIDENCE_ONLY = "weak_evidence_only"
    PASSING_MENTION_ONLY = "passing_mention_only"
    SUBJECT_NOT_SUBJECT_VENDOR = "subject_not_subject_vendor"
    POLARITY_NOT_RENDERABLE = "polarity_not_renderable"
    ROLE_NOT_RENDERABLE = "role_not_renderable"
    LOW_CONFIDENCE = "low_confidence"
    CONSUMER_FILTER_APPLIED = "consumer_filter_applied"


# ----------------------------------------------------------------------------
# Per-(scope, claim_type) policy.
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimGatePolicy:
    """Per-(claim_scope, claim_type) tuning of derivation + gate thresholds.

    Defaults model a permissive baseline; specific claim types can
    register stricter policies (e.g. churn-rate claims demand a
    larger denominator). All thresholds are inputs to derive_*()
    helpers and decide_render_gates(), never to caller decisions.
    """

    # Derivation -> EvidencePosture
    min_supporting_count: int = 3
    min_direct_evidence: int = 1
    contradiction_ratio_threshold: float = 0.4

    # Derivation -> ConfidenceLabel
    high_confidence_min_supporting: int = 10
    high_confidence_min_witnesses: int = 5
    medium_confidence_min_supporting: int = 3
    medium_confidence_min_witnesses: int = 2

    # Rate-claim denominator gate (applied only when is_rate_claim=True)
    is_rate_claim: bool = False
    min_denominator_for_rate: int = 10

    # Direct-evidence source. False keeps the v1 row-level grounding
    # approximation. True tells aggregators to derive direct evidence
    # from validated b2b_evidence_claims lineage when available.
    use_claim_lineage_for_direct_evidence: bool = False


_DEFAULT_POLICY = ClaimGatePolicy()


# Registry: per-(scope, claim_type) policy. Empty by default; consumers
# register their own as they migrate (Patch 2+). Lookup falls back to
# _DEFAULT_POLICY for unregistered (scope, claim_type) pairs.
_POLICY_REGISTRY: dict[tuple[ClaimScope, str], ClaimGatePolicy] = {}


class MissingClaimGatePolicyError(ValueError):
    """Raised when production code tries to build an unregistered claim type."""


def register_policy(
    claim_scope: ClaimScope, claim_type: str, policy: ClaimGatePolicy
) -> None:
    """Register a per-(scope, claim_type) policy. Subsequent
    build_product_claim() calls without an explicit policy use it."""
    _POLICY_REGISTRY[(claim_scope, claim_type)] = policy


def get_policy(claim_scope: ClaimScope, claim_type: str) -> ClaimGatePolicy:
    """Resolve the active policy for a (scope, claim_type). Falls back
    to the permissive default when none is registered."""
    return _POLICY_REGISTRY.get((claim_scope, claim_type), _DEFAULT_POLICY)


def get_registered_policy(
    claim_scope: ClaimScope,
    claim_type: str,
) -> ClaimGatePolicy:
    """Resolve a registered policy, failing closed when none exists.

    Use this from production aggregators before publishing a new
    customer-facing ProductClaim type. get_policy() intentionally
    keeps the historical default fallback for tests and transitional
    internal callers.
    """
    try:
        return _POLICY_REGISTRY[(claim_scope, claim_type)]
    except KeyError as exc:
        raise MissingClaimGatePolicyError(
            f"ProductClaim policy is not registered for {claim_scope.value}:{claim_type}"
        ) from exc


def reset_policy_registry() -> None:
    """Test helper. Clear all registered per-claim policies."""
    _POLICY_REGISTRY.clear()


# ----------------------------------------------------------------------------
# Pure derivation helpers.
# ----------------------------------------------------------------------------


def derive_evidence_posture(
    *,
    supporting_count: int,
    direct_evidence_count: int,
    contradiction_count: int,
    has_grounded_evidence: bool = True,
    policy: ClaimGatePolicy = _DEFAULT_POLICY,
) -> EvidencePosture:
    """Classify evidence quality from objective inputs. Pure, no I/O.

    has_grounded_evidence is the witness-scope adapter's signal for
    cannot_validate / synthesized evidence (the caller sets False when
    the underlying b2b_evidence_claims row was status='cannot_validate'
    or no phrase metadata exists).
    """
    if not has_grounded_evidence:
        return EvidencePosture.UNVERIFIED
    if supporting_count <= 0:
        return EvidencePosture.INSUFFICIENT
    if direct_evidence_count <= 0:
        return EvidencePosture.WEAK
    contradiction_ratio = contradiction_count / max(supporting_count, 1)
    if contradiction_ratio >= policy.contradiction_ratio_threshold:
        return EvidencePosture.CONTRADICTORY
    if supporting_count < policy.min_supporting_count:
        return EvidencePosture.INSUFFICIENT
    return EvidencePosture.USABLE


def derive_confidence(
    *,
    supporting_count: int,
    witness_count: int,
    policy: ClaimGatePolicy = _DEFAULT_POLICY,
) -> ConfidenceLabel:
    """Derive confidence bucket from coverage inputs. Pure, no I/O."""
    if (
        supporting_count >= policy.high_confidence_min_supporting
        and witness_count >= policy.high_confidence_min_witnesses
    ):
        return ConfidenceLabel.HIGH
    if (
        supporting_count >= policy.medium_confidence_min_supporting
        and witness_count >= policy.medium_confidence_min_witnesses
    ):
        return ConfidenceLabel.MEDIUM
    return ConfidenceLabel.LOW


def decide_render_gates(
    *,
    evidence_posture: EvidencePosture,
    confidence: ConfidenceLabel,
    supporting_count: int,
    direct_evidence_count: int,
    contradiction_count: int,
    denominator: int | None,
    sample_size: int | None,
    policy: ClaimGatePolicy = _DEFAULT_POLICY,
) -> tuple[bool, bool, SuppressionReason | None]:
    """Pure deterministic gate decision.

    Render gate (UI detail view OK):
      not unverified / insufficient AND
      counts above policy thresholds AND
      (rate claims) denominator above threshold

    Report gate (customer-facing publish OK):
      render_allowed AND posture == usable AND
      confidence in {high, medium}

    The two gates are NOT set independently per consumer.
    """
    if evidence_posture == EvidencePosture.UNVERIFIED:
        return (False, False, SuppressionReason.UNVERIFIED_EVIDENCE)
    if evidence_posture == EvidencePosture.INSUFFICIENT:
        return (False, False, SuppressionReason.INSUFFICIENT_SUPPORTING_COUNT)
    if supporting_count < policy.min_supporting_count:
        return (False, False, SuppressionReason.INSUFFICIENT_SUPPORTING_COUNT)
    if direct_evidence_count < policy.min_direct_evidence:
        return (False, False, SuppressionReason.WEAK_EVIDENCE_ONLY)
    if policy.is_rate_claim:
        if denominator is None:
            return (False, False, SuppressionReason.DENOMINATOR_UNKNOWN)
        if denominator < policy.min_denominator_for_rate:
            return (False, False, SuppressionReason.SAMPLE_SIZE_BELOW_THRESHOLD)

    render_allowed = True
    if evidence_posture == EvidencePosture.CONTRADICTORY:
        return (render_allowed, False, SuppressionReason.CONTRADICTORY_EVIDENCE)
    if evidence_posture == EvidencePosture.WEAK:
        return (render_allowed, False, SuppressionReason.WEAK_EVIDENCE_ONLY)
    if confidence == ConfidenceLabel.LOW:
        return (render_allowed, False, SuppressionReason.LOW_CONFIDENCE)
    return (True, True, None)


# ----------------------------------------------------------------------------
# Identity.
# ----------------------------------------------------------------------------


def compute_claim_id(
    *,
    claim_scope: ClaimScope,
    claim_type: str,
    claim_key: str,
    target_entity: str,
    secondary_target: str | None,
    as_of_date: date,
    analysis_window_days: int,
) -> str:
    """Deterministic id. claim_key is the disambiguating dimension
    within a (scope, type, target) tuple -- e.g. pain_category for
    VENDOR.weakness_theme, the vendor being evaluated for
    ACCOUNT.active_evaluation, or the witness_id for WITNESS.* claims.

    Without claim_key, multiple legitimate claims of the same
    (scope, type, target) collide. Required for that reason."""
    if not claim_key or not str(claim_key).strip():
        raise ValueError("claim_key is required and must be non-empty")
    payload = (
        f"{claim_scope.value}::{claim_type}::{claim_key.strip()}::"
        f"{target_entity}::{secondary_target or ''}::"
        f"{as_of_date.isoformat()}::{int(analysis_window_days)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# The envelope.
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ProductClaim:
    """Shared envelope every UI card and report section consumes.

    Construction must go through build_product_claim(); direct
    construction is rejected by __post_init__ unless the caller
    supplied gate fields that EXACTLY match what build_product_claim
    would have derived. This keeps the gate logic the only source of
    truth.
    """

    claim_id: str
    claim_key: str
    claim_scope: ClaimScope
    claim_type: str
    claim_text: str
    target_entity: str
    secondary_target: str | None

    # Numerator / denominator / context (objective facts).
    supporting_count: int
    direct_evidence_count: int
    witness_count: int
    contradiction_count: int
    denominator: int | None
    sample_size: int | None
    has_grounded_evidence: bool

    # Derived quality.
    confidence: ConfidenceLabel
    evidence_posture: EvidencePosture

    # Derived gates.
    render_allowed: bool
    report_allowed: bool
    suppression_reason: SuppressionReason | None

    # Provenance.
    evidence_links: tuple[str, ...]
    contradicting_links: tuple[str, ...]

    as_of_date: date
    analysis_window_days: int

    # Policy used at construction time. Stored on the row so __post_init__
    # can re-derive posture / confidence / gates with the same thresholds.
    policy: ClaimGatePolicy = field(default_factory=lambda: _DEFAULT_POLICY)
    schema_version: str = "v1"

    def __post_init__(self) -> None:
        # Re-derive everything and assert equality. Catches direct
        # constructor abuse: a caller cannot hand-set posture, gate, or
        # claim_id fields past what the inputs would have produced.
        expected_id = compute_claim_id(
            claim_scope=self.claim_scope,
            claim_type=self.claim_type,
            claim_key=self.claim_key,
            target_entity=self.target_entity,
            secondary_target=self.secondary_target,
            as_of_date=self.as_of_date,
            analysis_window_days=self.analysis_window_days,
        )
        if self.claim_id != expected_id:
            raise ValueError(
                f"claim_id={self.claim_id!r} inconsistent with derived="
                f"{expected_id!r}. Use build_product_claim()."
            )
        expected_posture = derive_evidence_posture(
            supporting_count=self.supporting_count,
            direct_evidence_count=self.direct_evidence_count,
            contradiction_count=self.contradiction_count,
            has_grounded_evidence=self.has_grounded_evidence,
            policy=self.policy,
        )
        if self.evidence_posture != expected_posture:
            raise ValueError(
                f"evidence_posture={self.evidence_posture} inconsistent with "
                f"derived={expected_posture}. Use build_product_claim()."
            )
        expected_confidence = derive_confidence(
            supporting_count=self.supporting_count,
            witness_count=self.witness_count,
            policy=self.policy,
        )
        if self.confidence != expected_confidence:
            raise ValueError(
                f"confidence={self.confidence} inconsistent with "
                f"derived={expected_confidence}. Use build_product_claim()."
            )
        expected_render, expected_report, expected_reason = decide_render_gates(
            evidence_posture=self.evidence_posture,
            confidence=self.confidence,
            supporting_count=self.supporting_count,
            direct_evidence_count=self.direct_evidence_count,
            contradiction_count=self.contradiction_count,
            denominator=self.denominator,
            sample_size=self.sample_size,
            policy=self.policy,
        )
        if (self.render_allowed, self.report_allowed, self.suppression_reason) != (
            expected_render,
            expected_report,
            expected_reason,
        ):
            raise ValueError(
                f"gate fields inconsistent with derivation. "
                f"got render={self.render_allowed}, report={self.report_allowed}, "
                f"reason={self.suppression_reason}. expected "
                f"render={expected_render}, report={expected_report}, "
                f"reason={expected_reason}. Use build_product_claim()."
            )


def build_product_claim(
    *,
    claim_scope: ClaimScope,
    claim_type: str,
    claim_key: str,
    claim_text: str,
    target_entity: str,
    secondary_target: str | None,
    supporting_count: int,
    direct_evidence_count: int,
    witness_count: int,
    contradiction_count: int,
    as_of_date: date,
    analysis_window_days: int,
    denominator: int | None = None,
    sample_size: int | None = None,
    has_grounded_evidence: bool = True,
    evidence_links: tuple[str, ...] = (),
    contradicting_links: tuple[str, ...] = (),
    policy: ClaimGatePolicy | None = None,
    require_registered_policy: bool = False,
    schema_version: str = "v1",
) -> ProductClaim:
    """Single supported entry point. Caller passes objective facts;
    builder derives posture, confidence, and gates from them via the
    active ClaimGatePolicy.

    policy resolution order:
      1. Explicit policy argument (test scenarios, audit overrides).
      2. Registered policy for (claim_scope, claim_type).
      3. _DEFAULT_POLICY (permissive baseline), unless
         require_registered_policy=True.
    """
    resolved_policy = (
        policy
        or (
            get_registered_policy(claim_scope, claim_type)
            if require_registered_policy
            else get_policy(claim_scope, claim_type)
        )
    )
    posture = derive_evidence_posture(
        supporting_count=supporting_count,
        direct_evidence_count=direct_evidence_count,
        contradiction_count=contradiction_count,
        has_grounded_evidence=has_grounded_evidence,
        policy=resolved_policy,
    )
    confidence = derive_confidence(
        supporting_count=supporting_count,
        witness_count=witness_count,
        policy=resolved_policy,
    )
    render_allowed, report_allowed, suppression_reason = decide_render_gates(
        evidence_posture=posture,
        confidence=confidence,
        supporting_count=supporting_count,
        direct_evidence_count=direct_evidence_count,
        contradiction_count=contradiction_count,
        denominator=denominator,
        sample_size=sample_size,
        policy=resolved_policy,
    )
    claim_id = compute_claim_id(
        claim_scope=claim_scope,
        claim_type=claim_type,
        claim_key=claim_key,
        target_entity=target_entity,
        secondary_target=secondary_target,
        as_of_date=as_of_date,
        analysis_window_days=analysis_window_days,
    )
    return ProductClaim(
        claim_id=claim_id,
        claim_key=claim_key,
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
        has_grounded_evidence=has_grounded_evidence,
        confidence=confidence,
        evidence_posture=posture,
        render_allowed=render_allowed,
        report_allowed=report_allowed,
        suppression_reason=suppression_reason,
        evidence_links=evidence_links,
        contradicting_links=contradicting_links,
        as_of_date=as_of_date,
        analysis_window_days=analysis_window_days,
        policy=resolved_policy,
        schema_version=schema_version,
    )


__all__ = [
    "ClaimScope",
    "EvidencePosture",
    "ConfidenceLabel",
    "SuppressionReason",
    "ClaimGatePolicy",
    "ProductClaim",
    "MissingClaimGatePolicyError",
    "register_policy",
    "get_policy",
    "get_registered_policy",
    "reset_policy_registry",
    "compute_claim_id",
    "derive_evidence_posture",
    "derive_confidence",
    "decide_render_gates",
    "build_product_claim",
]
