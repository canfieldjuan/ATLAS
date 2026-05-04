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
from .blog_pack import evaluate_blog_post
from .campaign_pack import evaluate_campaign
from .safety_gate import assess_risk, check_content
from .types import (
    ContentFlag,
    ContentScanResult,
    RiskAssessment,
    RiskLevel,
)


__all__ = [
    "ClaimGatePolicy",
    "ClaimScope",
    "ConfidenceLabel",
    "ContentFlag",
    "ContentScanResult",
    "EvidencePosture",
    "GateDecision",
    "GateFinding",
    "GateSeverity",
    "MissingClaimGatePolicyError",
    "ProductClaim",
    "QualityInput",
    "QualityPolicy",
    "QualityReport",
    "RiskAssessment",
    "RiskLevel",
    "SuppressionReason",
    "assess_risk",
    "build_product_claim",
    "check_content",
    "evaluate_blog_post",
    "evaluate_campaign",
    "compute_claim_id",
    "decide_render_gates",
    "derive_confidence",
    "derive_evidence_posture",
    "get_policy",
    "get_registered_policy",
    "register_policy",
    "reset_policy_registry",
]
