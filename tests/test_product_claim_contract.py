"""Phase 10 Patch 1: pin the ProductClaim contract.

The shared envelope every UI card and report section consumes lives or
dies by its gate semantics. If render_allowed and report_allowed do
not behave the way the contract spec says, every downstream patch
inherits the wrong behavior.

These tests pin (post-audit hardening):

  - Every (posture, confidence) combination produces the right gate
    pair, so a future change cannot loosen report_allowed.
  - Hard blocks (unverified, insufficient, below-threshold counts)
    reject from BOTH surfaces.
  - claim_key is required and disambiguates within (scope, type,
    target, secondary, date, window).
  - Posture is DERIVED from objective inputs; a caller cannot pass
    contradiction_count=N and posture=USABLE to bypass the
    contradictory-evidence gate.
  - Confidence is DERIVED from supporting_count + witness_count.
  - __post_init__ rejects direct dataclass construction with hand-set
    gate / posture / confidence fields that do not match the inputs.
  - Per-(scope, claim_type) policies override defaults via the
    registry and via explicit policy argument.
"""

from __future__ import annotations

from datetime import date
from itertools import product

import pytest

from atlas_brain.services.b2b.product_claim import (
    ClaimGatePolicy,
    ClaimScope,
    ConfidenceLabel,
    EvidencePosture,
    ProductClaim,
    SuppressionReason,
    build_product_claim,
    compute_claim_id,
    decide_render_gates,
    derive_confidence,
    derive_evidence_posture,
    get_policy,
    register_policy,
    reset_policy_registry,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    """Ensure each test starts with an empty policy registry."""
    reset_policy_registry()
    yield
    reset_policy_registry()


# ----------------------------------------------------------------------------
# Posture derivation (audit High #2: contradictions force CONTRADICTORY).
# ----------------------------------------------------------------------------


def test_unverified_when_evidence_not_grounded():
    posture = derive_evidence_posture(
        supporting_count=20,
        direct_evidence_count=10,
        contradiction_count=0,
        has_grounded_evidence=False,
    )
    assert posture == EvidencePosture.UNVERIFIED


def test_insufficient_when_no_supporting_count():
    posture = derive_evidence_posture(
        supporting_count=0,
        direct_evidence_count=0,
        contradiction_count=0,
        has_grounded_evidence=True,
    )
    assert posture == EvidencePosture.INSUFFICIENT


def test_weak_when_no_direct_evidence():
    posture = derive_evidence_posture(
        supporting_count=10,
        direct_evidence_count=0,
        contradiction_count=0,
        has_grounded_evidence=True,
    )
    assert posture == EvidencePosture.WEAK


def test_contradictory_when_contradiction_ratio_above_threshold():
    """Audit High #2: passing high contradiction_count must force
    CONTRADICTORY posture; the caller cannot label it USABLE to
    bypass the gate."""
    posture = derive_evidence_posture(
        supporting_count=10,
        direct_evidence_count=5,
        contradiction_count=4,  # 4/10 = 0.4, at default threshold
        has_grounded_evidence=True,
    )
    assert posture == EvidencePosture.CONTRADICTORY


def test_contradictory_below_threshold_stays_usable():
    posture = derive_evidence_posture(
        supporting_count=10,
        direct_evidence_count=5,
        contradiction_count=3,  # 3/10 = 0.3, below 0.4 threshold
        has_grounded_evidence=True,
    )
    assert posture == EvidencePosture.USABLE


def test_insufficient_when_supporting_below_min():
    """Default min_supporting_count=3. Below that returns INSUFFICIENT
    even with one direct evidence piece (indicates a single-quote
    claim that should not graduate to USABLE)."""
    posture = derive_evidence_posture(
        supporting_count=2,
        direct_evidence_count=1,
        contradiction_count=0,
        has_grounded_evidence=True,
    )
    assert posture == EvidencePosture.INSUFFICIENT


def test_usable_when_all_thresholds_met():
    posture = derive_evidence_posture(
        supporting_count=10,
        direct_evidence_count=5,
        contradiction_count=0,
        has_grounded_evidence=True,
    )
    assert posture == EvidencePosture.USABLE


# ----------------------------------------------------------------------------
# Confidence derivation.
# ----------------------------------------------------------------------------


def test_confidence_high_at_default_thresholds():
    assert (
        derive_confidence(supporting_count=10, witness_count=5)
        == ConfidenceLabel.HIGH
    )


def test_confidence_medium_at_default_thresholds():
    assert (
        derive_confidence(supporting_count=5, witness_count=3)
        == ConfidenceLabel.MEDIUM
    )


def test_confidence_low_below_medium_threshold():
    assert (
        derive_confidence(supporting_count=2, witness_count=1)
        == ConfidenceLabel.LOW
    )


def test_confidence_high_requires_both_thresholds():
    """Supporting count above HIGH threshold but witness count low
    must downgrade to MEDIUM, not stay at HIGH. Otherwise a single
    review with many phrases would grade HIGH despite no diversity."""
    assert (
        derive_confidence(supporting_count=20, witness_count=2)
        == ConfidenceLabel.MEDIUM
    )


# ----------------------------------------------------------------------------
# Render / report gate semantics.
# ----------------------------------------------------------------------------


def test_unverified_posture_blocks_both_surfaces():
    render, report, reason = decide_render_gates(
        evidence_posture=EvidencePosture.UNVERIFIED,
        confidence=ConfidenceLabel.HIGH,
        supporting_count=100,
        direct_evidence_count=50,
        contradiction_count=0,
        denominator=1000,
        sample_size=2000,
    )
    assert (render, report) == (False, False)
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
    assert (render, report) == (False, False)
    assert reason == SuppressionReason.INSUFFICIENT_SUPPORTING_COUNT


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
    assert (render, report) == (True, False)
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
    assert (render, report) == (True, False)
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
    assert (render, report) == (True, False)
    assert reason == SuppressionReason.LOW_CONFIDENCE


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
    assert (render, report) == (True, True)
    assert reason is None


# ----------------------------------------------------------------------------
# Rate-claim policy.
# ----------------------------------------------------------------------------


def test_rate_claim_policy_blocks_when_denominator_unknown():
    policy = ClaimGatePolicy(is_rate_claim=True)
    render, report, reason = decide_render_gates(
        evidence_posture=EvidencePosture.USABLE,
        confidence=ConfidenceLabel.HIGH,
        supporting_count=10,
        direct_evidence_count=5,
        contradiction_count=0,
        denominator=None,
        sample_size=100,
        policy=policy,
    )
    assert (render, report) == (False, False)
    assert reason == SuppressionReason.DENOMINATOR_UNKNOWN


def test_rate_claim_policy_blocks_below_threshold_denominator():
    policy = ClaimGatePolicy(is_rate_claim=True)
    render, report, reason = decide_render_gates(
        evidence_posture=EvidencePosture.USABLE,
        confidence=ConfidenceLabel.HIGH,
        supporting_count=10,
        direct_evidence_count=5,
        contradiction_count=0,
        denominator=5,
        sample_size=100,
        policy=policy,
    )
    assert (render, report) == (False, False)
    assert reason == SuppressionReason.SAMPLE_SIZE_BELOW_THRESHOLD


def test_non_rate_default_policy_does_not_require_denominator():
    render, report, _ = decide_render_gates(
        evidence_posture=EvidencePosture.USABLE,
        confidence=ConfidenceLabel.HIGH,
        supporting_count=10,
        direct_evidence_count=5,
        contradiction_count=0,
        denominator=None,
        sample_size=None,
    )
    assert (render, report) == (True, True)


def test_claim_lineage_policy_flag_is_stored_not_gate_logic():
    """The lineage flag is consumed by aggregators before they call the
    ProductClaim builder. It must not loosen or tighten render gates by
    itself; only the objective counts passed to the builder do that."""
    policy = ClaimGatePolicy(
        is_rate_claim=True,
        use_claim_lineage_for_direct_evidence=True,
    )
    render, report, reason = decide_render_gates(
        evidence_posture=EvidencePosture.USABLE,
        confidence=ConfidenceLabel.HIGH,
        supporting_count=10,
        direct_evidence_count=5,
        contradiction_count=0,
        denominator=100,
        sample_size=100,
        policy=policy,
    )
    assert policy.use_claim_lineage_for_direct_evidence is True
    assert (render, report, reason) == (True, True, None)


# ----------------------------------------------------------------------------
# Truth-table invariant: report ⊆ render across all combos.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("posture,confidence", list(product(EvidencePosture, ConfidenceLabel)))
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
            f"posture={posture.value} confidence={confidence.value}"
        )


@pytest.mark.parametrize("posture,confidence", list(product(EvidencePosture, ConfidenceLabel)))
def test_only_usable_high_or_medium_publishes(posture, confidence):
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
# claim_id (audit High #1: claim_key must disambiguate within scope).
# ----------------------------------------------------------------------------


def test_claim_id_requires_claim_key():
    with pytest.raises(ValueError, match="claim_key"):
        compute_claim_id(
            claim_scope=ClaimScope.VENDOR,
            claim_type="weakness_theme",
            claim_key="",
            target_entity="Asana",
            secondary_target=None,
            as_of_date=date(2026, 4, 26),
            analysis_window_days=90,
        )


def test_claim_id_deterministic_across_runs():
    args = dict(
        claim_scope=ClaimScope.VENDOR,
        claim_type="weakness_theme",
        claim_key="pricing",
        target_entity="Asana",
        secondary_target=None,
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
    )
    a = compute_claim_id(**args)
    b = compute_claim_id(**args)
    assert a == b
    assert len(a) == 64


def test_claim_id_disambiguates_via_claim_key():
    """Audit High #1: VENDOR.weakness_theme on Asana with different
    pain categories must produce different claim_ids. Without
    claim_key disambiguation, all weakness themes for Asana would
    hash to the same id."""
    base = dict(
        claim_scope=ClaimScope.VENDOR,
        claim_type="weakness_theme",
        target_entity="Asana",
        secondary_target=None,
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
    )
    pricing_id = compute_claim_id(**base, claim_key="pricing")
    support_id = compute_claim_id(**base, claim_key="support")
    ux_id = compute_claim_id(**base, claim_key="ux")
    assert len({pricing_id, support_id, ux_id}) == 3


def test_claim_id_disambiguates_account_evaluation_per_vendor():
    """ACCOUNT.active_evaluation on Globex evaluating Asana vs
    Pipedrive must produce different ids. claim_key carries the
    evaluated vendor."""
    base = dict(
        claim_scope=ClaimScope.ACCOUNT,
        claim_type="active_evaluation",
        target_entity="Globex Corp",
        secondary_target=None,
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
    )
    asana_id = compute_claim_id(**base, claim_key="Asana")
    pipedrive_id = compute_claim_id(**base, claim_key="Pipedrive")
    assert asana_id != pipedrive_id


def test_claim_id_varies_with_target_secondary_date_window():
    base = dict(
        claim_scope=ClaimScope.VENDOR,
        claim_type="weakness_theme",
        claim_key="pricing",
        target_entity="Asana",
        secondary_target=None,
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
    )
    base_id = compute_claim_id(**base)
    by_target = compute_claim_id(**{**base, "target_entity": "Pipedrive"})
    by_date = compute_claim_id(**{**base, "as_of_date": date(2026, 4, 27)})
    by_window = compute_claim_id(**{**base, "analysis_window_days": 30})
    by_secondary = compute_claim_id(**{**base, "secondary_target": "HubSpot"})
    assert len({base_id, by_target, by_date, by_window, by_secondary}) == 5


# ----------------------------------------------------------------------------
# build_product_claim integration (audit High #2 + Medium #1).
# ----------------------------------------------------------------------------


def _claim_args(**overrides):
    args = dict(
        claim_scope=ClaimScope.VENDOR,
        claim_type="weakness_theme",
        claim_key="pricing",
        claim_text="Pricing renewal pressure",
        target_entity="Asana",
        secondary_target=None,
        supporting_count=20,
        direct_evidence_count=8,
        witness_count=12,
        contradiction_count=0,
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
    )
    args.update(overrides)
    return args


def test_build_product_claim_publishable_path():
    claim = build_product_claim(**_claim_args())
    assert claim.evidence_posture == EvidencePosture.USABLE
    assert claim.confidence == ConfidenceLabel.HIGH
    assert claim.render_allowed is True
    assert claim.report_allowed is True
    assert claim.suppression_reason is None
    assert claim.claim_id  # populated


def test_build_product_claim_contradictions_force_unpublishable():
    """Audit High #2 closure: passing high contradictions through
    build_product_claim() must downgrade posture to CONTRADICTORY
    and block report_allowed -- the caller has no way to label it
    USABLE."""
    claim = build_product_claim(**_claim_args(contradiction_count=10))
    assert claim.evidence_posture == EvidencePosture.CONTRADICTORY
    assert claim.render_allowed is True
    assert claim.report_allowed is False
    assert claim.suppression_reason == SuppressionReason.CONTRADICTORY_EVIDENCE


def test_build_product_claim_unverified_via_grounding_flag():
    claim = build_product_claim(**_claim_args(has_grounded_evidence=False))
    assert claim.evidence_posture == EvidencePosture.UNVERIFIED
    assert claim.render_allowed is False
    assert claim.report_allowed is False
    assert claim.suppression_reason == SuppressionReason.UNVERIFIED_EVIDENCE


def test_build_product_claim_low_confidence_blocks_publish():
    claim = build_product_claim(**_claim_args(supporting_count=2, witness_count=1, direct_evidence_count=1))
    # supporting_count=2 is below default min_supporting_count=3 -> INSUFFICIENT
    assert claim.evidence_posture == EvidencePosture.INSUFFICIENT
    assert claim.render_allowed is False
    assert claim.report_allowed is False


def test_product_claim_dataclass_is_frozen():
    claim = build_product_claim(**_claim_args())
    with pytest.raises((AttributeError, Exception)):
        claim.report_allowed = False  # type: ignore[misc]


# ----------------------------------------------------------------------------
# __post_init__ rejects direct construction with hand-set fields
# (audit Medium #1: single entry point enforcement).
# ----------------------------------------------------------------------------


def _direct_construct_kwargs(**overrides):
    """Default kwargs that match a USABLE / HIGH path.

    The default claim_id is computed from the identity fields so that
    __post_init__'s claim_id check passes; tests that want to exercise
    a posture / confidence / gate mismatch can override the OFFENDING
    field while the identity stays consistent. Tests that want to
    exercise the claim_id check explicitly override claim_id directly.
    """
    base_identity = dict(
        claim_scope=ClaimScope.VENDOR,
        claim_type="weakness_theme",
        claim_key="pricing",
        target_entity="V",
        secondary_target=None,
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
    )
    # Honor identity-field overrides when computing the default id so
    # tests that change e.g. target_entity get a matching claim_id by
    # default and only need to override claim_id when testing identity.
    identity_for_id = {**base_identity}
    for k in list(base_identity.keys()):
        if k in overrides:
            identity_for_id[k] = overrides[k]
    default_id = compute_claim_id(**identity_for_id)
    base = dict(
        claim_id=default_id,
        **base_identity,
        claim_text="x",
        supporting_count=20,
        direct_evidence_count=8,
        witness_count=12,
        contradiction_count=0,
        denominator=None,
        sample_size=None,
        has_grounded_evidence=True,
        confidence=ConfidenceLabel.HIGH,
        evidence_posture=EvidencePosture.USABLE,
        render_allowed=True,
        report_allowed=True,
        suppression_reason=None,
        evidence_links=(),
        contradicting_links=(),
    )
    base.update(overrides)
    return base


def test_post_init_rejects_hand_set_report_allowed_when_posture_weak():
    """Caller fakes USABLE posture + report_allowed=True despite the
    inputs: rejected by __post_init__."""
    with pytest.raises(ValueError, match="evidence_posture"):
        ProductClaim(**_direct_construct_kwargs(direct_evidence_count=0))


def test_post_init_rejects_inconsistent_confidence():
    """supporting_count=2 + witness_count=1 derives LOW confidence;
    setting HIGH directly is rejected."""
    with pytest.raises(ValueError, match="confidence|evidence_posture"):
        ProductClaim(
            **_direct_construct_kwargs(supporting_count=2, witness_count=1)
        )


def test_post_init_rejects_inconsistent_gate_fields():
    """Posture is correctly USABLE for inputs but caller tried to
    set report_allowed=False with no suppression_reason. Rejected."""
    with pytest.raises(ValueError, match="gate fields"):
        ProductClaim(
            **_direct_construct_kwargs(report_allowed=False, suppression_reason=None)
        )


def test_post_init_rejects_caller_setting_contradictory_to_render_only():
    """Caller passes contradiction_count=5 with USABLE posture and
    publishable gates -- the audit High #2 case. __post_init__ catches
    it because derive_evidence_posture would have classified it
    CONTRADICTORY."""
    with pytest.raises(ValueError, match="evidence_posture"):
        ProductClaim(
            **_direct_construct_kwargs(
                contradiction_count=10,
                # claim USABLE + publishable, which is now wrong
            )
        )


def test_post_init_rejects_wrong_claim_id():
    """Audit follow-up: __post_init__ must re-compute claim_id from
    (scope, type, claim_key, target, secondary, date, window) and
    reject mismatches. Otherwise direct construction can still create
    a valid-looking claim with an arbitrary claim_id, which would
    silently break dedup / caching at the persistence layer."""
    with pytest.raises(ValueError, match="claim_id"):
        ProductClaim(**_direct_construct_kwargs(claim_id="0" * 64))


def test_post_init_rejects_claim_id_from_different_inputs():
    """Same kwargs except claim_id derived from a different
    target_entity. __post_init__ recomputes from the stored
    target_entity and rejects the stale id."""
    other_id = compute_claim_id(
        claim_scope=ClaimScope.VENDOR,
        claim_type="weakness_theme",
        claim_key="pricing",
        target_entity="Different Vendor",  # not what's on the row
        secondary_target=None,
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
    )
    with pytest.raises(ValueError, match="claim_id"):
        ProductClaim(**_direct_construct_kwargs(claim_id=other_id))


def test_post_init_passes_when_constructed_via_builder():
    """Building through build_product_claim() always satisfies
    __post_init__ -- the gate fields are derived from the same logic."""
    claim = build_product_claim(**_claim_args())
    # If we round-trip via the dataclass constructor with the SAME
    # values, __post_init__ accepts.
    same = ProductClaim(
        claim_id=claim.claim_id,
        claim_key=claim.claim_key,
        claim_scope=claim.claim_scope,
        claim_type=claim.claim_type,
        claim_text=claim.claim_text,
        target_entity=claim.target_entity,
        secondary_target=claim.secondary_target,
        supporting_count=claim.supporting_count,
        direct_evidence_count=claim.direct_evidence_count,
        witness_count=claim.witness_count,
        contradiction_count=claim.contradiction_count,
        denominator=claim.denominator,
        sample_size=claim.sample_size,
        has_grounded_evidence=claim.has_grounded_evidence,
        confidence=claim.confidence,
        evidence_posture=claim.evidence_posture,
        render_allowed=claim.render_allowed,
        report_allowed=claim.report_allowed,
        suppression_reason=claim.suppression_reason,
        evidence_links=claim.evidence_links,
        contradicting_links=claim.contradicting_links,
        as_of_date=claim.as_of_date,
        analysis_window_days=claim.analysis_window_days,
        policy=claim.policy,
        schema_version=claim.schema_version,
    )
    assert same == claim


# ----------------------------------------------------------------------------
# ClaimGatePolicy registry (audit Medium #2: per-claim tuning wired).
# ----------------------------------------------------------------------------


def test_default_policy_used_when_none_registered():
    p = get_policy(ClaimScope.VENDOR, "weakness_theme")
    assert p.min_supporting_count == 3
    assert p.is_rate_claim is False


def test_registered_policy_applied_by_builder():
    """Audit Medium #2 closure: registering a stricter policy for a
    (scope, claim_type) flows through build_product_claim()
    automatically."""
    strict = ClaimGatePolicy(min_supporting_count=20, min_direct_evidence=10)
    register_policy(ClaimScope.VENDOR, "churn_pressure", strict)

    # supporting_count=15 passes default but fails the strict policy.
    claim = build_product_claim(
        **_claim_args(claim_type="churn_pressure", supporting_count=15)
    )
    assert claim.evidence_posture == EvidencePosture.INSUFFICIENT
    assert claim.suppression_reason == SuppressionReason.INSUFFICIENT_SUPPORTING_COUNT
    assert claim.render_allowed is False


def test_explicit_policy_argument_overrides_registry():
    """Caller can pass a policy explicitly (for tests / audit overrides)
    and it takes precedence over the registry."""
    strict = ClaimGatePolicy(min_supporting_count=50)
    register_policy(ClaimScope.VENDOR, "weakness_theme", strict)
    looser = ClaimGatePolicy(min_supporting_count=2)
    claim = build_product_claim(**_claim_args(supporting_count=5), policy=looser)
    assert claim.evidence_posture == EvidencePosture.USABLE
    assert claim.render_allowed is True


def test_rate_claim_policy_via_registry():
    """Register a rate-claim policy and confirm denominator gate
    fires through build_product_claim()."""
    rate_policy = ClaimGatePolicy(is_rate_claim=True, min_denominator_for_rate=100)
    register_policy(ClaimScope.VENDOR, "decision_maker_churn_rate", rate_policy)

    no_denom = build_product_claim(
        **_claim_args(claim_type="decision_maker_churn_rate")
    )
    assert no_denom.suppression_reason == SuppressionReason.DENOMINATOR_UNKNOWN
    assert no_denom.render_allowed is False

    below = build_product_claim(
        **_claim_args(claim_type="decision_maker_churn_rate", denominator=50)
    )
    assert below.suppression_reason == SuppressionReason.SAMPLE_SIZE_BELOW_THRESHOLD

    above = build_product_claim(
        **_claim_args(claim_type="decision_maker_churn_rate", denominator=200)
    )
    assert above.render_allowed is True
    assert above.report_allowed is True
