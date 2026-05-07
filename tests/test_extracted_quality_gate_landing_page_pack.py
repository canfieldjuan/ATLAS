from __future__ import annotations

from extracted_quality_gate.landing_page_pack import evaluate_landing_page
from extracted_quality_gate.types import (
    GateDecision,
    QualityInput,
    QualityPolicy,
)


def _section(*, title: str = "Problem", body: str = "Renewal pricing is the #1 churn driver."):
    return {
        "id": "problem",
        "title": title,
        "body_markdown": body,
        "metadata": {"order": 1},
    }


def _input(**overrides) -> QualityInput:
    context = {
        "title": "Acme Q3: Stop Renewal Surprises",
        "slug": "acme-q3-launch",
        "hero": {
            "headline": "Stop renewal surprises",
            "subheadline": "Acme catches pricing pressure 90 days early",
            "cta_label": "Book a 15-min demo",
            "cta_url": "/demo",
        },
        "sections": (_section(),),
        "cta": {"label": "Book a 15-min demo", "url": "/demo"},
        "meta": {
            "title_tag": "Stop Renewal Surprises | Acme",
            "description": "Acme catches renewal pressure 90 days early so you can prevent unplanned churn at scale.",
        },
    }
    context.update(overrides)
    return QualityInput(artifact_type="landing_page", context=context)


def test_evaluate_landing_page_happy_path_passes() -> None:
    report = evaluate_landing_page(_input())
    assert report.passed is True
    assert report.decision == GateDecision.PASS
    assert report.blockers == ()


def test_evaluate_landing_page_no_title_blocks() -> None:
    report = evaluate_landing_page(_input(title=""))
    codes = {f.code for f in report.blockers}
    assert "no_title" in codes


def test_evaluate_landing_page_no_slug_blocks() -> None:
    report = evaluate_landing_page(_input(slug=""))
    codes = {f.code for f in report.blockers}
    assert "no_slug" in codes


def test_evaluate_landing_page_no_hero_headline_blocks() -> None:
    report = evaluate_landing_page(_input(hero={"subheadline": "x", "cta_label": "y", "cta_url": "/z"}))
    codes = {f.code for f in report.blockers}
    assert "no_hero_headline" in codes


def test_evaluate_landing_page_no_hero_subheadline_warns_only() -> None:
    """Subheadline absence is a warning, not a blocker (some heroes are headline-only)."""
    report = evaluate_landing_page(
        _input(hero={"headline": "h", "cta_label": "y", "cta_url": "/z"})
    )
    blocker_codes = {f.code for f in report.blockers}
    warning_codes = {f.code for f in report.warnings}
    assert "no_hero_subheadline" not in blocker_codes
    assert "no_hero_subheadline" in warning_codes


def test_evaluate_landing_page_no_cta_blocks() -> None:
    report = evaluate_landing_page(_input(cta={}))
    codes = {f.code for f in report.blockers}
    assert "no_cta" in codes


def test_evaluate_landing_page_partial_cta_still_blocks() -> None:
    """CTA needs both label and url; either alone is incomplete."""
    report = evaluate_landing_page(_input(cta={"label": "Demo"}))
    assert any(f.code == "no_cta" for f in report.blockers)


def test_evaluate_landing_page_no_sections_blocks() -> None:
    report = evaluate_landing_page(_input(sections=()))
    codes = {f.code for f in report.blockers}
    assert "no_sections" in codes


def test_evaluate_landing_page_section_missing_title_blocks_per_section() -> None:
    sections = (_section(), _section(title=""))
    report = evaluate_landing_page(_input(sections=sections))
    blocker_msgs = {f.message for f in report.blockers}
    assert "section_missing_title:1" in blocker_msgs
    assert "section_missing_title:0" not in blocker_msgs


def test_evaluate_landing_page_section_missing_body_blocks_per_section() -> None:
    sections = (_section(), _section(body=""))
    report = evaluate_landing_page(_input(sections=sections))
    blocker_msgs = {f.message for f in report.blockers}
    assert "section_missing_body:1" in blocker_msgs


def test_evaluate_landing_page_meta_description_too_short_warns() -> None:
    """Below the configured min for SEO."""
    policy = QualityPolicy(name="lp_policy", thresholds={"min_meta_description_chars": 200})
    report = evaluate_landing_page(_input(), policy=policy)
    codes = {f.code for f in report.warnings}
    assert "meta_description_too_short" in codes


def test_evaluate_landing_page_meta_description_missing_warns() -> None:
    policy = QualityPolicy(name="lp_policy", thresholds={"min_meta_description_chars": 100})
    report = evaluate_landing_page(_input(meta={}), policy=policy)
    codes = {f.code for f in report.warnings}
    assert "missing_meta_description" in codes


def test_evaluate_landing_page_blocked_phrasing_word_boundary_does_not_match_substring() -> None:
    sections = (_section(body="We cannot compromise on quality."),)
    policy = QualityPolicy(name="lp_policy", metadata={"blocked_phrasing": ("promise",)})
    report = evaluate_landing_page(_input(sections=sections), policy=policy)
    codes = {f.code for f in report.blockers}
    assert "blocked_phrasing" not in codes


def test_evaluate_landing_page_blocked_phrasing_blocks_with_word_boundary() -> None:
    sections = (_section(body="We GUARANTEE results within 30 days."),)
    policy = QualityPolicy(name="lp_policy", metadata={"blocked_phrasing": ("guarantee",)})
    report = evaluate_landing_page(_input(sections=sections), policy=policy)
    blocker_msgs = {f.message for f in report.blockers}
    assert "blocked_phrasing:guarantee" in blocker_msgs


def test_evaluate_landing_page_blocked_phrasing_auto_wraps_bare_string_policy() -> None:
    sections = (_section(body="We GUARANTEE results."),)
    policy = QualityPolicy(name="lp_policy", metadata={"blocked_phrasing": "guarantee"})
    report = evaluate_landing_page(_input(sections=sections), policy=policy)
    blocker_msgs = {f.message for f in report.blockers}
    assert "blocked_phrasing:guarantee" in blocker_msgs


def test_evaluate_landing_page_blocked_phrasing_scans_hero_and_cta() -> None:
    """Blocked-phrase scan covers hero (headline + subheadline) and CTA label, not just sections."""
    policy = QualityPolicy(name="lp_policy", metadata={"blocked_phrasing": ("guarantee",)})
    report = evaluate_landing_page(
        _input(hero={"headline": "We guarantee results", "subheadline": "x", "cta_label": "y", "cta_url": "/z"}),
        policy=policy,
    )
    blocker_msgs = {f.message for f in report.blockers}
    assert "blocked_phrasing:guarantee" in blocker_msgs


def test_evaluate_landing_page_metadata_carries_score_and_section_count() -> None:
    report = evaluate_landing_page(_input())
    assert "score" in report.metadata
    assert report.metadata["section_count"] == 1
    assert report.metadata["status"] == "pass"
