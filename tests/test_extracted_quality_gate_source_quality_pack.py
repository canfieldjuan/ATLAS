"""Tests for extracted_quality_gate.source_quality_pack.

Pure unit tests against the lifted helpers + the
``evaluate_source_quality`` pack-contract entry point.
"""

from __future__ import annotations

import pytest

from extracted_quality_gate.product_claim import (
    ClaimGatePolicy,
    ConfidenceLabel,
    EvidencePosture,
    SuppressionReason,
)
from extracted_quality_gate.source_quality_pack import (
    apply_witness_render_gate,
    build_non_empty_text_check,
    compute_coverage_ratio,
    evaluate_source_quality,
    row_count,
)
from extracted_quality_gate.types import (
    GateDecision,
    GateSeverity,
    QualityInput,
    QualityPolicy,
    QualityReport,
)


# ---- compute_coverage_ratio ----


def test_coverage_ratio_basic():
    assert compute_coverage_ratio(1, 3) == round(1 / 3, 3)


def test_coverage_ratio_full():
    assert compute_coverage_ratio(5, 5) == 1.0


def test_coverage_ratio_returns_none_for_zero_denominator():
    assert compute_coverage_ratio(1, 0) is None


def test_coverage_ratio_returns_none_for_none_inputs():
    assert compute_coverage_ratio(None, 3) is None
    assert compute_coverage_ratio(1, None) is None


def test_coverage_ratio_supports_custom_precision():
    assert compute_coverage_ratio(1, 3, precision=5) == round(1 / 3, 5)


def test_coverage_ratio_accepts_floats():
    assert compute_coverage_ratio(1.5, 3.0) == round(0.5, 3)


# ---- row_count ----


def test_row_count_returns_value_when_key_present():
    assert row_count({"x": 5}, "x") == 5


def test_row_count_returns_zero_for_missing_key():
    assert row_count({}, "x") == 0


def test_row_count_uses_fallback_key():
    assert row_count({"y": 7}, "x", fallback_key="y") == 7


def test_row_count_prefers_primary_over_fallback():
    assert row_count({"x": 5, "y": 7}, "x", fallback_key="y") == 5


def test_row_count_handles_none_value():
    assert row_count({"x": None}, "x") == 0


# ---- build_non_empty_text_check ----


def test_non_empty_text_check_wraps_expression():
    sql = build_non_empty_text_check("reviewer_title")
    assert "NULLIF" in sql
    assert "TRIM" in sql
    assert "COALESCE" in sql
    assert "reviewer_title" in sql
    assert "IS NOT NULL" in sql


def test_non_empty_text_check_supports_jsonb_paths():
    sql = build_non_empty_text_check("enrichment->'reviewer_context'->>'industry'")
    assert "enrichment->'reviewer_context'->>'industry'" in sql


# ---- apply_witness_render_gate ----


def _strong_grounded_row(**overrides):
    row = {
        "pain_confidence": "strong",
        "grounding_status": "grounded",
        "phrase_subject": "subject_vendor",
        "phrase_polarity": "negative",
        "phrase_role": "primary_driver",
        "witness_type": "weakness",
    }
    row.update(overrides)
    return row


def test_render_gate_marks_ungrounded_as_unverified():
    row = {"pain_confidence": "strong", "grounding_status": "pending"}
    out = apply_witness_render_gate(row)
    assert out is row  # in-place mutation
    assert out["quote_grade"] is False
    assert out["evidence_posture"] == EvidencePosture.UNVERIFIED.value
    assert out["render_allowed"] is False
    assert out["confidence"] == ConfidenceLabel.HIGH.value


def test_render_gate_grounded_subject_vendor_passes():
    row = _strong_grounded_row()
    out = apply_witness_render_gate(row)
    assert out["quote_grade"] is True
    assert out["evidence_posture"] == EvidencePosture.USABLE.value
    assert out["render_allowed"] is True


def test_render_gate_blocks_passing_mention():
    row = _strong_grounded_row(phrase_role="passing_mention")
    out = apply_witness_render_gate(row)
    assert out["render_allowed"] is False
    assert out["evidence_posture"] == EvidencePosture.WEAK.value
    assert out["suppression_reason"] == SuppressionReason.PASSING_MENTION_ONLY.value


def test_render_gate_blocks_subject_not_vendor():
    row = _strong_grounded_row(phrase_subject="other_vendor")
    out = apply_witness_render_gate(row)
    assert out["render_allowed"] is False
    assert out["suppression_reason"] == SuppressionReason.SUBJECT_NOT_SUBJECT_VENDOR.value


def test_render_gate_blocks_unverified_when_required_tag_missing():
    row = _strong_grounded_row()
    row["phrase_polarity"] = None
    out = apply_witness_render_gate(row)
    assert out["render_allowed"] is False
    assert out["suppression_reason"] == SuppressionReason.UNVERIFIED_EVIDENCE.value


def test_render_gate_allows_positive_polarity_for_strength_witness_type():
    row = _strong_grounded_row(phrase_polarity="positive", witness_type="strength")
    out = apply_witness_render_gate(row)
    assert out["render_allowed"] is True


def test_render_gate_blocks_positive_polarity_for_weakness_witness_type():
    row = _strong_grounded_row(phrase_polarity="positive", witness_type="weakness")
    out = apply_witness_render_gate(row)
    assert out["render_allowed"] is False
    assert out["suppression_reason"] == SuppressionReason.POLARITY_NOT_RENDERABLE.value


def test_render_gate_blocks_low_pain_confidence():
    row = _strong_grounded_row(pain_confidence="none")
    out = apply_witness_render_gate(row)
    assert out["render_allowed"] is False
    assert out["suppression_reason"] == SuppressionReason.LOW_CONFIDENCE.value


def test_render_gate_confidence_mapping():
    strong = apply_witness_render_gate(_strong_grounded_row(pain_confidence="strong"))
    assert strong["confidence"] == ConfidenceLabel.HIGH.value
    weak = apply_witness_render_gate(_strong_grounded_row(pain_confidence="weak"))
    assert weak["confidence"] == ConfidenceLabel.MEDIUM.value
    unknown = _strong_grounded_row(pain_confidence="unknown")
    out = apply_witness_render_gate(unknown)
    assert out["confidence"] == ConfidenceLabel.LOW.value


def test_render_gate_accepts_explicit_policy():
    # Tighten the policy beyond the legacy default; legacy defaults
    # were min=1, so using min=2 should change render eligibility.
    tight_policy = ClaimGatePolicy(
        min_supporting_count=2,
        min_direct_evidence=2,
        high_confidence_min_supporting=2,
        high_confidence_min_witnesses=2,
        medium_confidence_min_supporting=2,
        medium_confidence_min_witnesses=2,
    )
    row = _strong_grounded_row()
    out = apply_witness_render_gate(row, policy=tight_policy)
    assert out["render_allowed"] is False


# ---- evaluate_source_quality (pack contract) ----


def _build_input(witnesses) -> QualityInput:
    return QualityInput(
        artifact_type="witness_collection",
        artifact_id="test",
        content=None,
        context={"witnesses": witnesses},
    )


def test_pack_contract_returns_quality_report():
    report = evaluate_source_quality(_build_input([]))
    assert isinstance(report, QualityReport)


def test_pack_contract_passes_for_empty_input():
    report = evaluate_source_quality(_build_input([]))
    assert report.decision == GateDecision.PASS
    assert report.findings == ()
    assert report.metadata["total_witnesses"] == 0


def test_pack_contract_passes_when_all_witnesses_render():
    rows = [_strong_grounded_row() for _ in range(3)]
    report = evaluate_source_quality(_build_input(rows))
    assert report.decision == GateDecision.PASS
    assert report.metadata["rendered_witnesses"] == 3
    assert report.metadata["suppressed_witnesses"] == 0


def test_pack_contract_warns_on_partial_suppression():
    rows = [
        _strong_grounded_row(),
        _strong_grounded_row(phrase_role="passing_mention", witness_id="bad-1"),
    ]
    report = evaluate_source_quality(_build_input(rows))
    assert report.decision == GateDecision.WARN
    suppressed = [f for f in report.findings if f.code == "witness_suppressed"]
    assert len(suppressed) == 1
    assert "bad-1" in suppressed[0].message
    assert suppressed[0].metadata["suppression_reason"] == SuppressionReason.PASSING_MENTION_ONLY.value


def test_pack_contract_blocks_when_no_witnesses_render():
    rows = [
        _strong_grounded_row(phrase_role="passing_mention", witness_id="w1"),
        _strong_grounded_row(phrase_subject="other_vendor", witness_id="w2"),
    ]
    report = evaluate_source_quality(_build_input(rows))
    assert report.decision == GateDecision.BLOCK
    blockers = [f for f in report.findings if f.severity == GateSeverity.BLOCKER]
    assert any(f.code == "no_renderable_witnesses" for f in blockers)
    assert report.metadata["rendered_witnesses"] == 0
    assert report.metadata["suppressed_witnesses"] == 2


def test_pack_contract_does_not_mutate_input_rows():
    rows = [_strong_grounded_row()]
    snapshot = dict(rows[0])
    evaluate_source_quality(_build_input(rows))
    assert rows[0] == snapshot


def test_pack_contract_skips_non_dict_entries():
    rows = [_strong_grounded_row(), "not a dict", None, 42]
    report = evaluate_source_quality(_build_input(rows))
    assert report.metadata["total_witnesses"] == 1


def test_pack_contract_metadata_carries_processed_rows():
    rows = [_strong_grounded_row()]
    report = evaluate_source_quality(_build_input(rows))
    assert "rows" in report.metadata
    processed = report.metadata["rows"]
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert processed[0]["render_allowed"] is True


def test_pack_contract_policy_overrides_render_thresholds():
    # Tightened policy that no row can satisfy under the default
    # supporting/direct counts of 1 used internally.
    policy = QualityPolicy(
        name="source_quality",
        thresholds={
            "min_supporting_count": 5,
            "min_direct_evidence": 5,
            "high_confidence_min_supporting": 5,
            "high_confidence_min_witnesses": 5,
            "medium_confidence_min_supporting": 5,
            "medium_confidence_min_witnesses": 5,
        },
    )
    rows = [_strong_grounded_row()]
    report = evaluate_source_quality(_build_input(rows), policy=policy)
    assert report.decision == GateDecision.BLOCK


def test_quality_report_is_frozen():
    report = evaluate_source_quality(_build_input([]))
    with pytest.raises(Exception):
        report.passed = False  # type: ignore[misc]


# ---- Atlas re-export sanity ----


def test_atlas_re_export_paths_match():
    from atlas_brain.services.b2b.witness_render_gate import (
        apply_witness_render_gate as atlas_apply,
        evaluate_source_quality as atlas_evaluate,
    )
    assert atlas_apply is apply_witness_render_gate
    assert atlas_evaluate is evaluate_source_quality

    # source_impact.py keeps its underscore aliases for diff stability
    from atlas_brain.services.b2b.source_impact import (
        _build_non_empty_text_check,
        _compute_coverage_ratio,
        _row_count,
    )
    assert _build_non_empty_text_check is build_non_empty_text_check
    assert _compute_coverage_ratio is compute_coverage_ratio
    assert _row_count is row_count
