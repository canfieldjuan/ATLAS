from __future__ import annotations

from extracted_reasoning_core.api import validate_reasoning_output
from extracted_reasoning_core.types import OutputPolicy, ReasoningResult, ValidationReport


def _result(
    *,
    claims=(
        {"claim": "Renewal pricing drives churn.", "confidence": 0.85, "source_ids": ["r1"], "type": "driver"},
        {"claim": "Onboarding friction is secondary.", "confidence": 0.55, "source_ids": ["r2"], "type": "driver"},
    ),
    confidence: float = 0.78,
    summary: str = "Pricing pressure dominates.",
) -> ReasoningResult:
    return ReasoningResult(
        summary=summary,
        claims=claims,
        confidence=confidence,
        tier="L2",
        state={},
    )


def test_validate_reasoning_output_default_policy_passes_well_formed_result() -> None:
    report = validate_reasoning_output(_result())
    assert isinstance(report, ValidationReport)
    assert report.passed is True
    assert report.blockers == ()
    assert report.repaired_fields == {}


def test_validate_reasoning_output_no_claims_blocks() -> None:
    report = validate_reasoning_output(_result(claims=()))
    assert report.passed is False
    assert "no_claims" in report.blockers


def test_validate_reasoning_output_min_confidence_blocks_when_below_threshold() -> None:
    report = validate_reasoning_output(_result(confidence=0.4), policy=OutputPolicy(min_confidence=0.6))
    assert report.passed is False
    assert "confidence_below_min" in report.blockers


def test_validate_reasoning_output_require_citations_blocks_per_uncited_claim() -> None:
    claims = (
        {"claim": "cited", "confidence": 0.8, "source_ids": ["r1"], "type": "driver"},
        {"claim": "uncited", "confidence": 0.7, "source_ids": [], "type": "driver"},
        {"claim": "missing key", "confidence": 0.7, "type": "driver"},
    )
    report = validate_reasoning_output(_result(claims=claims), policy=OutputPolicy(require_citations=True))
    assert report.passed is False
    assert "claim_missing_citations:1" in report.blockers
    assert "claim_missing_citations:2" in report.blockers
    assert "claim_missing_citations:0" not in report.blockers


def test_validate_reasoning_output_required_claim_types_blocks_missing_types() -> None:
    policy = OutputPolicy(required_claim_types=("driver", "competitive"))
    report = validate_reasoning_output(_result(), policy=policy)
    assert report.passed is False
    assert "missing_required_claim_type:competitive" in report.blockers
    assert "missing_required_claim_type:driver" not in report.blockers


def test_validate_reasoning_output_blocked_phrasing_blocks_case_insensitive() -> None:
    claims = (
        {"claim": "We GUARANTEE results within 30 days.", "confidence": 0.8, "source_ids": ["r1"], "type": "driver"},
    )
    policy = OutputPolicy(blocked_phrasing=("guarantee", "promise"))
    report = validate_reasoning_output(_result(claims=claims), policy=policy)
    assert report.passed is False
    assert "blocked_phrasing:guarantee" in report.blockers
    assert "blocked_phrasing:promise" not in report.blockers
