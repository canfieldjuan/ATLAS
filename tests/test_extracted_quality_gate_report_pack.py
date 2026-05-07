from __future__ import annotations

from extracted_quality_gate.report_pack import evaluate_report
from extracted_quality_gate.types import (
    GateDecision,
    GateSeverity,
    QualityInput,
    QualityPolicy,
)


def _section(*, title: str = "Findings", body: str = "Renewal pricing dominates.", evidence_ids: tuple[str, ...] = ("r1",)) -> dict:
    return {
        "id": "findings",
        "title": title,
        "body_markdown": body,
        "claim_ids": ("c1",),
        "evidence_ids": evidence_ids,
    }


def _input(**overrides) -> QualityInput:
    context = {
        "title": "Acme: Q3 Pressure Report",
        "summary": "Pricing renewal pressure dominates the displacement signal.",
        "sections": (_section(),),
        "reference_ids": ("r1", "r2"),
        "metadata": {"confidence": 0.84},
    }
    context.update(overrides)
    return QualityInput(artifact_type="report", context=context)


def test_evaluate_report_happy_path_passes() -> None:
    report = evaluate_report(_input())
    assert report.passed is True
    assert report.decision == GateDecision.PASS
    assert report.blockers == ()
    assert report.metadata["section_count"] == 1


def test_evaluate_report_no_title_blocks() -> None:
    report = evaluate_report(_input(title=""))
    assert report.passed is False
    assert report.decision == GateDecision.BLOCK
    codes = {f.code for f in report.blockers}
    assert "no_title" in codes


def test_evaluate_report_no_summary_blocks() -> None:
    report = evaluate_report(_input(summary=""))
    assert report.passed is False
    codes = {f.code for f in report.blockers}
    assert "no_summary" in codes


def test_evaluate_report_no_sections_blocks() -> None:
    report = evaluate_report(_input(sections=()))
    assert report.passed is False
    codes = {f.code for f in report.blockers}
    assert "no_sections" in codes


def test_evaluate_report_section_missing_body_blocks_per_section() -> None:
    sections = (
        _section(),
        _section(title="Drivers", body=""),  # broken
    )
    report = evaluate_report(_input(sections=sections))
    assert report.passed is False
    blocker_msgs = {f.message for f in report.blockers}
    assert "section_missing_body:1" in blocker_msgs
    assert "section_missing_body:0" not in blocker_msgs


def test_evaluate_report_section_missing_title_blocks_per_section() -> None:
    sections = (_section(title=""),)
    report = evaluate_report(_input(sections=sections))
    assert report.passed is False
    assert any(f.code == "section_missing_title" for f in report.blockers)


def test_evaluate_report_no_references_blocks_when_neither_top_level_nor_section_evidence() -> None:
    sections = (_section(evidence_ids=()),)
    report = evaluate_report(_input(sections=sections, reference_ids=()))
    assert report.passed is False
    assert any(f.code == "no_references" for f in report.blockers)


def test_evaluate_report_no_references_passes_when_section_evidence_present() -> None:
    """If a section carries evidence_ids, top-level reference_ids may be empty."""
    sections = (_section(evidence_ids=("r1",)),)
    report = evaluate_report(_input(sections=sections, reference_ids=()))
    codes = {f.code for f in report.blockers}
    assert "no_references" not in codes


def test_evaluate_report_blocked_phrasing_uses_word_boundaries_not_substrings() -> None:
    """Mirrors validate_reasoning_output regression: 'promise' must not match 'compromise'."""
    sections = (_section(body="We cannot compromise on quality."),)
    policy = QualityPolicy(
        name="report_policy",
        metadata={"blocked_phrasing": ("promise",)},
    )
    report = evaluate_report(_input(sections=sections), policy=policy)
    blocker_codes = {f.code for f in report.blockers}
    assert "blocked_phrasing" not in blocker_codes


def test_evaluate_report_blocked_phrasing_blocks_with_word_boundary_match() -> None:
    sections = (_section(body="We GUARANTEE results within 30 days."),)
    policy = QualityPolicy(
        name="report_policy",
        metadata={"blocked_phrasing": ("guarantee",)},
    )
    report = evaluate_report(_input(sections=sections), policy=policy)
    blocker_msgs = {f.message for f in report.blockers}
    assert "blocked_phrasing:guarantee" in blocker_msgs


def test_evaluate_report_min_confidence_warns_when_below_threshold() -> None:
    policy = QualityPolicy(
        name="report_policy",
        thresholds={"min_confidence": 0.7},
    )
    report = evaluate_report(_input(metadata={"confidence": 0.5}), policy=policy)
    warning_codes = {f.code for f in report.warnings}
    assert "confidence_below_min" in warning_codes


def test_evaluate_report_min_confidence_warns_when_missing() -> None:
    policy = QualityPolicy(
        name="report_policy",
        thresholds={"min_confidence": 0.7},
    )
    report = evaluate_report(_input(metadata={}), policy=policy)
    warning_codes = {f.code for f in report.warnings}
    assert "missing_confidence" in warning_codes


def test_evaluate_report_blocked_phrasing_auto_wraps_bare_string_policy() -> None:
    """A host passing blocked_phrasing as a bare string (not a sequence) auto-wraps; no silent no-op."""
    sections = (_section(body="We GUARANTEE results within 30 days."),)
    # Bare string instead of tuple/list — historical mistake, should still gate.
    policy = QualityPolicy(
        name="report_policy",
        metadata={"blocked_phrasing": "guarantee"},
    )
    report = evaluate_report(_input(sections=sections), policy=policy)
    blocker_msgs = {f.message for f in report.blockers}
    assert "blocked_phrasing:guarantee" in blocker_msgs


def test_evaluate_report_metadata_carries_score_and_threshold() -> None:
    report = evaluate_report(_input())
    assert "score" in report.metadata
    assert "threshold" in report.metadata
    assert report.metadata["status"] == "pass"
