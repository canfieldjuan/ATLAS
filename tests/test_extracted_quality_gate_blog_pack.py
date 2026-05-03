"""Tests for extracted_quality_gate.blog_pack.evaluate_blog_post.

The function under test is pure: no DB, no clock, no network. So
these tests are pure unit tests -- no fixtures, no mocks, no async.
"""

from __future__ import annotations

import pytest

from extracted_quality_gate.blog_pack import evaluate_blog_post
from extracted_quality_gate.types import (
    GateDecision,
    GateSeverity,
    QualityInput,
    QualityPolicy,
    QualityReport,
)


def _make_input(content: str, **context_overrides) -> QualityInput:
    """Build a `QualityInput` with sensible defaults for blog testing."""
    base_context: dict = {
        "topic_type": "vendor_alternative",
        "slug": "test-post",
        "suggested_title": "Test Post",
        "data_context": {},
        "charts": (),
        "source_quotes": (),
        "required_vendors": (),
        "grounded_vendors": frozenset(),
    }
    base_context.update(context_overrides)
    return QualityInput(
        artifact_type="blog_post",
        artifact_id="test-post",
        content=content,
        context=base_context,
    )


# Build a long, well-formed body that should pass everything by default.
# 1500 words is the default min; 2200 is the SEO target.
_LONG_GOOD_BODY = (
    "This is a self-selected analysis of vendor X. " * 300
    + '\n> "It works really well for our team," said one user.\n'
    + '\n> "We saw real improvements after switching," noted another.\n'
)


# ---- Decision shape ----


def test_evaluate_returns_quality_report():
    report = evaluate_blog_post(_make_input(_LONG_GOOD_BODY))
    assert isinstance(report, QualityReport)


def test_clean_long_body_passes():
    report = evaluate_blog_post(_make_input(_LONG_GOOD_BODY))
    assert report.passed is True
    assert report.decision == GateDecision.PASS
    assert report.findings == ()


def test_metadata_mirrors_legacy_dict_shape():
    # The wrapper consumes these keys directly, so they must stay stable.
    report = evaluate_blog_post(_make_input(_LONG_GOOD_BODY))
    md = report.metadata
    assert "score" in md
    assert "threshold" in md
    assert "status" in md
    assert "blocking_issues" in md
    assert "warnings" in md
    assert "word_count" in md
    assert "min_words_required" in md
    assert "target_words" in md
    assert "quote_count" in md


# ---- Word count ----


def test_short_body_blocks():
    report = evaluate_blog_post(_make_input("Too short."))
    assert report.passed is False
    assert report.decision == GateDecision.BLOCK
    codes = {f.code for f in report.findings}
    assert "content_too_short" in codes


def test_word_count_below_seo_target_warns():
    body = " ".join(["word"] * 1700)  # above min_words 1500, below target 2200
    report = evaluate_blog_post(
        _make_input(body)
    )
    codes = {f.code for f in report.findings if f.severity == GateSeverity.WARNING}
    assert "content_below_seo_target" in codes


def test_custom_min_words_via_policy():
    body = " ".join(["word"] * 100)
    policy = QualityPolicy(
        name="blog_post",
        thresholds={"min_words": 50, "target_words": 80, "pass_score": 70},
    )
    report = evaluate_blog_post(_make_input(body), policy=policy)
    # 100 words is above policy.target_words (80), so word-count rules
    # do not contribute findings even though the default would block.
    codes = {f.code for f in report.findings}
    assert "content_too_short" not in codes
    assert "content_below_seo_target" not in codes


# ---- Chart placeholders ----


def test_missing_chart_placeholder_blocks():
    charts = ({"chart_id": "vendor-share-chart", "data_labels_lower": ()},)
    report = evaluate_blog_post(_make_input(_LONG_GOOD_BODY, charts=charts))
    codes = {f.code for f in report.findings}
    assert "missing_chart_placeholder" in codes


def test_chart_placeholder_present_passes():
    body = _LONG_GOOD_BODY + "\n\n{{chart:vendor-share-chart}}\n"
    charts = ({"chart_id": "vendor-share-chart", "data_labels_lower": ()},)
    report = evaluate_blog_post(_make_input(body, charts=charts))
    codes = {f.code for f in report.findings}
    assert "missing_chart_placeholder" not in codes


def test_duplicate_chart_placeholder_blocks():
    body = _LONG_GOOD_BODY + "\n\n{{chart:c1}}\n\n{{chart:c1}}\n"
    charts = ({"chart_id": "c1", "data_labels_lower": ()},)
    report = evaluate_blog_post(_make_input(body, charts=charts))
    codes = {f.code for f in report.findings}
    assert "duplicate_chart_placeholder" in codes


def test_unknown_chart_placeholder_blocks():
    body = _LONG_GOOD_BODY + "\n\n{{chart:unknown-id}}\n"
    report = evaluate_blog_post(_make_input(body, charts=()))
    codes = {f.code for f in report.findings}
    assert "unknown_chart_placeholders" in codes


# ---- Unresolved tokens ----


def test_unresolved_token_blocks():
    body = _LONG_GOOD_BODY + "\n\nSee {{vendor_name}} for details.\n"
    report = evaluate_blog_post(_make_input(body))
    codes = {f.code for f in report.findings}
    assert "unresolved_placeholders" in codes


def test_chart_token_does_not_count_as_unresolved():
    body = _LONG_GOOD_BODY + "\n{{chart:c1}}\n"
    charts = ({"chart_id": "c1", "data_labels_lower": ()},)
    report = evaluate_blog_post(_make_input(body, charts=charts))
    codes = {f.code for f in report.findings}
    assert "unresolved_placeholders" not in codes


# ---- Quotes ----


def test_too_few_sourced_quotes_blocks():
    # source_quotes present, body has zero quotes
    body = " ".join(["word"] * 1600) + " self-selected"
    report = evaluate_blog_post(
        _make_input(body, source_quotes=("real quote",))
    )
    codes = {f.code for f in report.findings}
    assert "too_few_sourced_quotes" in codes


def test_no_quotes_warns_when_no_source_quotes():
    body = " ".join(["self-selected word"] * 800)
    report = evaluate_blog_post(_make_input(body))
    codes = {f.code for f in report.findings if f.severity == GateSeverity.WARNING}
    assert "no_quotes_present" in codes


# ---- Review period ----


def test_missing_review_period_warns():
    report = evaluate_blog_post(
        _make_input(_LONG_GOOD_BODY, data_context={"review_period": "2024-Q4"})
    )
    codes = {f.code for f in report.findings if f.severity == GateSeverity.WARNING}
    assert "review_period_not_explicitly_mentioned" in codes


def test_present_review_period_does_not_warn():
    body = _LONG_GOOD_BODY + " The 2024-Q4 review period covered..."
    report = evaluate_blog_post(
        _make_input(body, data_context={"review_period": "2024-Q4"})
    )
    codes = {f.code for f in report.findings if f.severity == GateSeverity.WARNING}
    assert "review_period_not_explicitly_mentioned" not in codes


# ---- Methodology disclaimer ----


def test_missing_self_selected_disclaimer_warns():
    body = " ".join(["word"] * 1600)  # no "self-selected"
    body += '\n> "It works really well for our team."\n'
    body += '\n> "We saw real improvements."\n'
    report = evaluate_blog_post(_make_input(body))
    codes = {f.code for f in report.findings if f.severity == GateSeverity.WARNING}
    assert "methodology_disclaimer_missing_self_selected" in codes


# ---- Required vendors ----


def test_missing_required_vendor_blocks():
    body = _LONG_GOOD_BODY  # mentions only "vendor X"
    report = evaluate_blog_post(
        _make_input(body, required_vendors=("Acme",))
    )
    codes = {f.code for f in report.findings}
    assert "missing_vendor_mentions" in codes


def test_required_vendor_present_passes():
    body = _LONG_GOOD_BODY + "\nAcme is a vendor we discuss.\n"
    report = evaluate_blog_post(
        _make_input(body, required_vendors=("Acme",))
    )
    codes = {f.code for f in report.findings}
    assert "missing_vendor_mentions" not in codes


# ---- Placeholder href ----


def test_placeholder_href_hash_blocks():
    body = _LONG_GOOD_BODY + '\n<a href="#">Link</a>\n'
    report = evaluate_blog_post(_make_input(body))
    codes = {f.code for f in report.findings}
    assert "placeholder_links_href_hash" in codes


# ---- Internal links ----


def test_unknown_internal_link_blocks():
    body = _LONG_GOOD_BODY + "\nSee /blog/ghost-post for context.\n"
    report = evaluate_blog_post(
        _make_input(
            body,
            slug="this-post",
            data_context={"_valid_internal_slugs": ["real-post"]},
        )
    )
    codes = {f.code for f in report.findings}
    assert "nonexistent_internal_links" in codes


def test_self_link_does_not_block():
    body = _LONG_GOOD_BODY + "\nSee /blog/this-post for context.\n"
    report = evaluate_blog_post(
        _make_input(
            body,
            slug="this-post",
            data_context={"_valid_internal_slugs": []},
        )
    )
    codes = {f.code for f in report.findings}
    assert "nonexistent_internal_links" not in codes


# ---- Title/vendor mismatch ----


def test_title_missing_vendor_warns():
    report = evaluate_blog_post(
        _make_input(
            _LONG_GOOD_BODY,
            suggested_title="Generic post",
            data_context={"vendor": "Acme"},
        )
    )
    codes = {f.code for f in report.findings if f.severity == GateSeverity.WARNING}
    assert "title_missing_expected_vendor" in codes


# ---- Category outcome ----


def test_unsupported_category_winner_blocks():
    body = _LONG_GOOD_BODY + " The category winner is unclear."
    report = evaluate_blog_post(
        _make_input(body, data_context={})  # no category_winner
    )
    codes = {f.code for f in report.findings}
    assert "unsupported_category_outcome_assertion" in codes


def test_category_winner_with_data_does_not_block():
    body = _LONG_GOOD_BODY + " The category winner is Acme."
    report = evaluate_blog_post(
        _make_input(body, data_context={"category_winner": "Acme"})
    )
    codes = {f.code for f in report.findings}
    assert "unsupported_category_outcome_assertion" not in codes


# ---- Unsupported data claims ----


def test_ungrounded_vendor_in_claim_warns():
    body = (
        _LONG_GOOD_BODY
        + " According to data shows, GhostVendor leads the market."
    )
    report = evaluate_blog_post(
        _make_input(
            body,
            grounded_vendors=frozenset({"acme"}),
            data_context={"_known_vendors": ["GhostVendor", "Acme"]},
        )
    )
    codes = {f.code for f in report.findings}
    assert "unsupported_data_claim" in codes


# ---- Migration direction drift ----


def test_migration_guide_outbound_drift_warns():
    body = (
        " ".join(["self-selected word"] * 800)
        + " switching from Acme to other vendors. leaving Acme behind."
        + '\n> "It works really well for our team."\n'
        + '\n> "We saw real improvements."\n'
    )
    report = evaluate_blog_post(
        _make_input(
            body,
            topic_type="migration_guide",
            data_context={"vendor": "Acme"},
        )
    )
    codes = {f.code for f in report.findings if f.severity == GateSeverity.WARNING}
    assert "migration_direction_drift" in codes


# ---- Numeric consistency ----


def test_subcounts_exceeding_headline_warns():
    body = (
        _LONG_GOOD_BODY
        + " We saw 100 churn signals in total. "
        + " 60 churn signals from segment A and 70 churn signals from segment B."
    )
    report = evaluate_blog_post(_make_input(body))
    codes = {f.code for f in report.findings if f.severity == GateSeverity.WARNING}
    assert "numeric_inconsistency" in codes


# ---- Decision aggregation ----


def test_only_warnings_returns_warn():
    # Long body, no blockers, but missing methodology disclaimer
    body = " ".join(["word"] * 2300)  # above target_words
    body += '\n> "It works really well for our team."\n'
    body += '\n> "We saw real improvements."\n'
    report = evaluate_blog_post(_make_input(body))
    blockers = [f for f in report.findings if f.severity == GateSeverity.BLOCKER]
    warnings = [f for f in report.findings if f.severity == GateSeverity.WARNING]
    assert blockers == []
    assert len(warnings) >= 1
    assert report.decision == GateDecision.WARN


def test_quality_report_is_frozen():
    report = evaluate_blog_post(_make_input(_LONG_GOOD_BODY))
    with pytest.raises(Exception):
        report.passed = False  # type: ignore[misc]
