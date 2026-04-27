"""Shared witness-scope UI/report render gates.

These gates adapt persisted b2b_vendor_witnesses rows into the ProductClaim
render contract. They are intentionally claim-type agnostic: a row can be
visible for audit while still being unsafe to render as customer-facing
evidence.
"""

from .product_claim import (
    ClaimGatePolicy,
    ConfidenceLabel,
    EvidencePosture,
    SuppressionReason,
    decide_render_gates,
)

_WITNESS_RENDER_POLICY = ClaimGatePolicy(
    min_supporting_count=1,
    min_direct_evidence=1,
    high_confidence_min_supporting=1,
    high_confidence_min_witnesses=1,
    medium_confidence_min_supporting=1,
    medium_confidence_min_witnesses=1,
)


def _witness_confidence(row: dict) -> ConfidenceLabel:
    value = str(row.get("pain_confidence") or "").strip().lower()
    if value == "strong":
        return ConfidenceLabel.HIGH
    if value == "weak":
        return ConfidenceLabel.MEDIUM
    return ConfidenceLabel.LOW


def _witness_gate_payload(
    *,
    evidence_posture: EvidencePosture,
    confidence: ConfidenceLabel,
    suppression_reason: SuppressionReason | None = None,
) -> dict:
    if suppression_reason is None:
        render_allowed, report_allowed, derived_reason = decide_render_gates(
            evidence_posture=evidence_posture,
            confidence=confidence,
            supporting_count=1,
            direct_evidence_count=1,
            contradiction_count=0,
            denominator=None,
            sample_size=None,
            policy=_WITNESS_RENDER_POLICY,
        )
        suppression_reason = derived_reason
    else:
        render_allowed = False
        report_allowed = False

    return {
        "evidence_posture": evidence_posture.value,
        "confidence": confidence.value,
        "render_allowed": render_allowed,
        "report_allowed": report_allowed,
        "suppression_reason": (
            suppression_reason.value if suppression_reason else None
        ),
    }


def apply_witness_render_gate(row: dict) -> dict:
    """Attach ProductClaim-style render gates to a witness row."""
    confidence = _witness_confidence(row)
    grounding_status = str(row.get("grounding_status") or "pending").strip()
    row["quote_grade"] = grounding_status == "grounded"

    if not row["quote_grade"]:
        row.update(
            _witness_gate_payload(
                evidence_posture=EvidencePosture.UNVERIFIED,
                confidence=confidence,
            )
        )
        return row

    required_tags = ("phrase_subject", "phrase_polarity", "phrase_role")
    if any(row.get(field) is None for field in required_tags):
        row.update(
            _witness_gate_payload(
                evidence_posture=EvidencePosture.UNVERIFIED,
                confidence=confidence,
                suppression_reason=SuppressionReason.UNVERIFIED_EVIDENCE,
            )
        )
        return row

    subject = str(row.get("phrase_subject") or "").strip().lower()
    if subject != "subject_vendor":
        row.update(
            _witness_gate_payload(
                evidence_posture=EvidencePosture.INSUFFICIENT,
                confidence=confidence,
                suppression_reason=SuppressionReason.SUBJECT_NOT_SUBJECT_VENDOR,
            )
        )
        return row

    role = str(row.get("phrase_role") or "").strip().lower()
    if role == "passing_mention":
        row.update(
            _witness_gate_payload(
                evidence_posture=EvidencePosture.WEAK,
                confidence=confidence,
                suppression_reason=SuppressionReason.PASSING_MENTION_ONLY,
            )
        )
        return row

    if role not in {"primary_driver", "supporting_context"}:
        row.update(
            _witness_gate_payload(
                evidence_posture=EvidencePosture.INSUFFICIENT,
                confidence=confidence,
                suppression_reason=SuppressionReason.ROLE_NOT_RENDERABLE,
            )
        )
        return row

    polarity = str(row.get("phrase_polarity") or "").strip().lower()
    witness_type = str(row.get("witness_type") or "").strip().lower()
    positive_allowed = witness_type in {"strength", "counterevidence"}
    if polarity not in {"negative", "mixed"} and not (
        polarity == "positive" and positive_allowed
    ):
        row.update(
            _witness_gate_payload(
                evidence_posture=EvidencePosture.INSUFFICIENT,
                confidence=confidence,
                suppression_reason=SuppressionReason.POLARITY_NOT_RENDERABLE,
            )
        )
        return row

    if str(row.get("pain_confidence") or "").strip().lower() == "none":
        row.update(
            _witness_gate_payload(
                evidence_posture=EvidencePosture.WEAK,
                confidence=confidence,
                suppression_reason=SuppressionReason.LOW_CONFIDENCE,
            )
        )
        return row

    row.update(
        _witness_gate_payload(
            evidence_posture=EvidencePosture.USABLE,
            confidence=confidence,
        )
    )
    return row


__all__ = ["apply_witness_render_gate"]
