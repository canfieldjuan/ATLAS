"""Standalone quality-gate core.

Import product-facing entry points from `extracted_quality_gate.api`.
"""

from __future__ import annotations

from .api import (
    ClaimGatePolicy,
    ClaimScope,
    ConfidenceLabel,
    EvidencePosture,
    GateDecision,
    GateFinding,
    GateSeverity,
    MissingClaimGatePolicyError,
    ProductClaim,
    QualityInput,
    QualityPolicy,
    QualityReport,
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
    "ClaimGatePolicy",
    "ClaimScope",
    "ConfidenceLabel",
    "EvidencePosture",
    "GateDecision",
    "GateFinding",
    "GateSeverity",
    "MissingClaimGatePolicyError",
    "ProductClaim",
    "QualityInput",
    "QualityPolicy",
    "QualityReport",
    "SuppressionReason",
    "build_product_claim",
    "compute_claim_id",
    "decide_render_gates",
    "derive_confidence",
    "derive_evidence_posture",
    "get_policy",
    "get_registered_policy",
    "register_policy",
    "reset_policy_registry",
]
