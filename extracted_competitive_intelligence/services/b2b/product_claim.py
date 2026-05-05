"""Competitive Intelligence ProductClaim compatibility surface.

The ProductClaim contract is owned by ``extracted_quality_gate``. Competitive
Intelligence re-exports the public contract here so existing package imports
keep working without importing Atlas internals.
"""

from __future__ import annotations

from extracted_quality_gate.product_claim import (
    ClaimGatePolicy,
    ClaimScope,
    ConfidenceLabel,
    EvidencePosture,
    MissingClaimGatePolicyError,
    ProductClaim,
    SuppressionReason,
    build_product_claim,
    compute_claim_id,
    decide_render_gates,
    derive_confidence,
    derive_evidence_posture,
    get_policy,
    get_registered_policy,
    register_policy,
    reset_policy_registry,
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
