from __future__ import annotations

from extracted_quality_gate.sales_brief_pack import evaluate_sales_brief
from extracted_quality_gate.types import (
    GateDecision,
    QualityInput,
    QualityPolicy,
)


def _section(
    *,
    title: str = "Account Context",
    body: str = "Mid-market SaaS, Series C, 350 seats.",
    evidence_ids: tuple[str, ...] = ("r1",),
) -> dict:
    return {
        "id": "account_context",
        "title": title,
        "body_markdown": body,
        "claim_ids": ("c1",),
        "evidence_ids": evidence_ids,
    }


def _input(**overrides) -> QualityInput:
    context = {
        "title": "Pre-call brief: Acme renewal",
        "headline": "Renewal Q3 -- 90-day pressure window opens this week",
        "sections": (_section(),),
        "reference_ids": ("r1", "r2"),
        "metadata": {"confidence": 0.82},
    }
    context.update(overrides)
    return QualityInput(artifact_type="sales_brief", context=context)


def test_evaluate_sales_brief_happy_path_passes() -> None:
    report = evaluate_sales_brief(_input())
    assert report.passed is True
    assert report.decision == GateDecision.PASS
    assert report.blockers == ()
    assert report.metadata["section_count"] == 1


def test_evaluate_sales_brief_no_title_blocks() -> None:
    report = evaluate_sales_brief(_input(title=""))
    assert report.passed is False
    codes = {f.code for f in report.blockers}
    assert "no_title" in codes


def test_evaluate_sales_brief_no_headline_blocks() -> None:
    report = evaluate_sales_brief(_input(headline=""))
    assert report.passed is False
    codes = {f.code for f in report.blockers}
    assert "no_headline" in codes


def test_evaluate_sales_brief_no_sections_blocks() -> None:
    report = evaluate_sales_brief(_input(sections=()))
    assert report.passed is False
    codes = {f.code for f in report.blockers}
    assert "no_sections" in codes


def test_evaluate_sales_brief_section_missing_body_blocks_per_section() -> None:
    sections = (_section(), _section(title="Signals", body=""))
    report = evaluate_sales_brief(_input(sections=sections))
    blocker_msgs = {f.message for f in report.blockers}
    assert "section_missing_body:1" in blocker_msgs
    assert "section_missing_body:0" not in blocker_msgs


def test_evaluate_sales_brief_section_missing_title_blocks_per_section() -> None:
    sections = (_section(title=""),)
    report = evaluate_sales_brief(_input(sections=sections))
    assert any(f.code == "section_missing_title" for f in report.blockers)


def test_evaluate_sales_brief_no_references_blocks_when_neither_top_nor_section_evidence() -> None:
    """A sales brief with no source ids is useless to the rep."""
    sections = (_section(evidence_ids=()),)
    report = evaluate_sales_brief(_input(sections=sections, reference_ids=()))
    assert any(f.code == "no_references" for f in report.blockers)


def test_evaluate_sales_brief_no_references_passes_when_section_evidence_present() -> None:
    """Section evidence alone is sufficient -- top-level reference_ids may be empty."""
    sections = (_section(evidence_ids=("r1",)),)
    report = evaluate_sales_brief(_input(sections=sections, reference_ids=()))
    codes = {f.code for f in report.blockers}
    assert "no_references" not in codes


def test_evaluate_sales_brief_headline_too_long_warns_only() -> None:
    """Default ceiling is 280 chars; over that warns but doesn't block."""
    long_headline = "x" * 300
    report = evaluate_sales_brief(_input(headline=long_headline))
    blocker_codes = {f.code for f in report.blockers}
    warning_codes = {f.code for f in report.warnings}
    assert "headline_too_long" not in blocker_codes
    assert "headline_too_long" in warning_codes


def test_evaluate_sales_brief_headline_within_ceiling_no_warning() -> None:
    report = evaluate_sales_brief(_input(headline="Short and punchy."))
    warning_codes = {f.code for f in report.warnings}
    assert "headline_too_long" not in warning_codes


def test_evaluate_sales_brief_headline_ceiling_is_configurable() -> None:
    """Hosts can tighten or loosen the ceiling via policy.thresholds."""
    policy = QualityPolicy(name="brief_policy", thresholds={"max_headline_chars": 50})
    report = evaluate_sales_brief(
        _input(headline="x" * 80),
        policy=policy,
    )
    warning_codes = {f.code for f in report.warnings}
    assert "headline_too_long" in warning_codes


def test_evaluate_sales_brief_blocked_phrasing_uses_word_boundaries_not_substrings() -> None:
    """Mirrors validate_reasoning_output regression: 'promise' must not match 'compromise'."""
    sections = (_section(body="We cannot compromise on quality."),)
    policy = QualityPolicy(
        name="brief_policy",
        metadata={"blocked_phrasing": ("promise",)},
    )
    report = evaluate_sales_brief(_input(sections=sections), policy=policy)
    blocker_codes = {f.code for f in report.blockers}
    assert "blocked_phrasing" not in blocker_codes


def test_evaluate_sales_brief_blocked_phrasing_blocks_with_word_boundary_match() -> None:
    sections = (_section(body="We GUARANTEE results within 30 days."),)
    policy = QualityPolicy(
        name="brief_policy",
        metadata={"blocked_phrasing": ("guarantee",)},
    )
    report = evaluate_sales_brief(_input(sections=sections), policy=policy)
    blocker_msgs = {f.message for f in report.blockers}
    assert "blocked_phrasing:guarantee" in blocker_msgs


def test_evaluate_sales_brief_blocked_phrasing_scans_headline_and_title() -> None:
    """Blocked-phrase scan covers title + headline, not just sections."""
    policy = QualityPolicy(
        name="brief_policy",
        metadata={"blocked_phrasing": ("guarantee",)},
    )
    report = evaluate_sales_brief(
        _input(headline="We guarantee renewal -- pressure window opens"),
        policy=policy,
    )
    blocker_msgs = {f.message for f in report.blockers}
    assert "blocked_phrasing:guarantee" in blocker_msgs


def test_evaluate_sales_brief_blocked_phrasing_auto_wraps_bare_string_policy() -> None:
    """Bare-string policy auto-wraps -- a host typo shouldn't silently disable the gate."""
    sections = (_section(body="We GUARANTEE results."),)
    policy = QualityPolicy(
        name="brief_policy",
        metadata={"blocked_phrasing": "guarantee"},
    )
    report = evaluate_sales_brief(_input(sections=sections), policy=policy)
    blocker_msgs = {f.message for f in report.blockers}
    assert "blocked_phrasing:guarantee" in blocker_msgs


def test_evaluate_sales_brief_min_confidence_warns_when_below_threshold() -> None:
    policy = QualityPolicy(name="brief_policy", thresholds={"min_confidence": 0.7})
    report = evaluate_sales_brief(_input(metadata={"confidence": 0.5}), policy=policy)
    warning_codes = {f.code for f in report.warnings}
    assert "confidence_below_min" in warning_codes


def test_evaluate_sales_brief_min_confidence_warns_when_missing() -> None:
    policy = QualityPolicy(name="brief_policy", thresholds={"min_confidence": 0.7})
    report = evaluate_sales_brief(_input(metadata={}), policy=policy)
    warning_codes = {f.code for f in report.warnings}
    assert "missing_confidence" in warning_codes


def test_evaluate_sales_brief_metadata_carries_score_and_threshold() -> None:
    report = evaluate_sales_brief(_input())
    assert "score" in report.metadata
    assert "threshold" in report.metadata
    assert report.metadata["status"] == "pass"
