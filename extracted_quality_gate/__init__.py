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
from .evidence_pack import (
    audit_witness_evidence_coverage,
    evaluate_evidence_coverage,
)
from .safety_gate import assess_risk, check_content
from .source_quality_pack import (
    apply_witness_render_gate,
    build_non_empty_text_check,
    compute_coverage_ratio,
    evaluate_source_quality,
    row_count,
)
from .types import (
    ContentFlag,
    ContentScanResult,
    RiskAssessment,
    RiskLevel,
)
from .witness_pack import (
    campaign_proof_terms_from_audit,
    evaluate_specificity_support,
    evaluate_witness_specificity,
    merge_specificity_contexts,
    specificity_audit_snapshot,
    specificity_signal_terms,
    surface_specificity_context,
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
    "apply_witness_render_gate",
    "assess_risk",
    "audit_witness_evidence_coverage",
    "build_non_empty_text_check",
    "build_product_claim",
    "campaign_proof_terms_from_audit",
    "check_content",
    "compute_claim_id",
    "compute_coverage_ratio",
    "decide_render_gates",
    "derive_confidence",
    "derive_evidence_posture",
    "evaluate_blog_post",
    "evaluate_campaign",
    "evaluate_evidence_coverage",
    "evaluate_source_quality",
    "evaluate_specificity_support",
    "evaluate_witness_specificity",
    "get_policy",
    "get_registered_policy",
    "merge_specificity_contexts",
    "register_policy",
    "reset_policy_registry",
    "row_count",
    "specificity_audit_snapshot",
    "specificity_signal_terms",
    "surface_specificity_context",
]
