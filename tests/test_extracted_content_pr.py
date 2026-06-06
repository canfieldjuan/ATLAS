"""Unit tests for slice 4: the Content-PR review contract.

Pure value types + verdict logic; no DB, no async, no Atlas imports, no LLM.
Composes slice 1 (ReviewDecision) and slice 3 (claims map).
"""

from __future__ import annotations

from datetime import date

import pytest

from extracted_content_pipeline.claims_map import ExtractedClaim, RegistryClaim, map_claim
from extracted_content_pipeline.content_pr import (
    CommentCategory,
    ContentPR,
    CoverageRow,
    CoverageStatus,
    ReviewComment,
    RulePacketVersions,
    blocking_comments,
    failing_required_rows,
    review_verdict,
    unresolved_required_rows,
    verdict_reasons,
)
from extracted_content_pipeline.review_contract import ReviewDecision

_PINNED = RulePacketVersions(
    brief="b1",
    brand_voice="v1",
    claim_registry="c1",
    compliance="x1",
    channel_schema="s1",
)

_REGISTRY = {
    "feature.sso": RegistryClaim(id="feature.sso", approved_wording="SSO is included"),
}


def _resolved_pass(rule_id: str) -> CoverageRow:
    return CoverageRow(
        rule_id=rule_id,
        requirement="...",
        status=CoverageStatus.PASS,
        evidence="quote from draft",
    )


# -- RulePacketVersions ------------------------------------------------------


def test_rule_packet_pinned_only_when_all_stamps_present() -> None:
    assert _PINNED.is_pinned() is True
    assert _PINNED.missing() == ()
    bare = RulePacketVersions(brief="b1")
    assert bare.is_pinned() is False
    assert "compliance" in bare.missing()


# -- CoverageRow.is_resolved -------------------------------------------------


def test_unresolved_status_is_not_resolved() -> None:
    assert CoverageRow(rule_id="R1").is_resolved() is False  # default UNRESOLVED


def test_pass_without_evidence_is_not_resolved() -> None:
    assert CoverageRow(rule_id="R1", status=CoverageStatus.PASS).is_resolved() is False


def test_pass_with_evidence_is_resolved() -> None:
    assert _resolved_pass("R1").is_resolved() is True


def test_not_applicable_needs_no_evidence() -> None:
    row = CoverageRow(rule_id="R1", status=CoverageStatus.NOT_APPLICABLE)
    assert row.is_resolved() is True


def test_plain_string_not_applicable_is_resolved() -> None:
    # Decoded JSON carries plain strings; value-equality must treat them the same.
    row = CoverageRow(rule_id="R1", status="not_applicable")  # type: ignore[arg-type]
    assert row.is_resolved() is True


def test_optional_unresolved_row_does_not_count_as_required_gap() -> None:
    rows = [CoverageRow(rule_id="R1", required=False)]
    assert unresolved_required_rows(rows) == ()


# -- ReviewComment -----------------------------------------------------------


def test_nit_comment_cannot_be_blocking() -> None:
    with pytest.raises(ValueError):
        ReviewComment(category=CommentCategory.NIT, message="punchier?", blocking=True)


def test_nit_comment_non_blocking_is_fine() -> None:
    c = ReviewComment(category=CommentCategory.NIT, message="punchier?")
    assert c.blocking is False


def test_plain_string_nit_cannot_be_blocking() -> None:
    # Even when category arrives as the plain string "nit".
    with pytest.raises(ValueError):
        ReviewComment(category="nit", blocking=True)  # type: ignore[arg-type]


def test_blocking_comments_filters() -> None:
    comments = [
        ReviewComment(category=CommentCategory.BRAND_RULE, message="off voice", blocking=True),
        ReviewComment(category=CommentCategory.NIT, message="nit"),
    ]
    assert len(blocking_comments(comments)) == 1


# -- review_verdict (the integration) ---------------------------------------


def _approvable_pr() -> ContentPR:
    return ContentPR(
        asset_id="email-1",
        rule_packet=_PINNED,
        coverage=(_resolved_pass("VOICE-01"), _resolved_pass("CLAIM-01")),
        claims=(
            map_claim(
                ExtractedClaim(text="SSO is included", registry_id="feature.sso"),
                _REGISTRY,
                as_of=date(2026, 6, 6),
            ),
        ),
        comments=(ReviewComment(category=CommentCategory.NIT, message="tighten intro"),),
    )


def test_empty_coverage_matrix_blocks_missing_coverage() -> None:
    # Pinned packet but no required rows asserts nothing -> must not approve.
    pr = ContentPR(rule_packet=_PINNED, coverage=())
    assert review_verdict(pr) is ReviewDecision.BLOCKED
    assert any("missing coverage matrix" in r for r in verdict_reasons(pr))


def test_only_optional_rows_also_blocks_missing_coverage() -> None:
    pr = ContentPR(
        rule_packet=_PINNED,
        coverage=(CoverageRow(rule_id="opt", required=False, status=CoverageStatus.PASS, evidence="e"),),
    )
    assert review_verdict(pr) is ReviewDecision.BLOCKED


def test_clean_pr_is_approved() -> None:
    pr = _approvable_pr()
    assert review_verdict(pr) is ReviewDecision.APPROVED
    assert verdict_reasons(pr) == ()


def test_unpinned_rule_packet_blocks() -> None:
    pr = ContentPR(rule_packet=RulePacketVersions(brief="only"), coverage=(_resolved_pass("R1"),))
    assert review_verdict(pr) is ReviewDecision.BLOCKED
    assert any("rule packet not pinned" in r for r in verdict_reasons(pr))


def test_unresolved_required_row_blocks_no_silent_pass() -> None:
    pr = ContentPR(
        rule_packet=_PINNED,
        coverage=(_resolved_pass("R1"), CoverageRow(rule_id="R2")),  # R2 UNRESOLVED
    )
    assert review_verdict(pr) is ReviewDecision.BLOCKED
    assert any("unresolved required coverage" in r for r in verdict_reasons(pr))


def test_failed_required_row_requires_revision() -> None:
    pr = ContentPR(
        rule_packet=_PINNED,
        coverage=(
            _resolved_pass("R1"),
            CoverageRow(rule_id="R2", status=CoverageStatus.FAIL, evidence="missing CTA"),
        ),
    )
    assert review_verdict(pr) is ReviewDecision.REVISION_REQUIRED
    assert failing_required_rows(pr.coverage)[0].rule_id == "R2"


def test_plain_string_fail_status_blocks_approval() -> None:
    # The P1 regression: a required row with status="fail" (plain string) must
    # NOT slip through as APPROVED.
    pr = ContentPR(
        rule_packet=_PINNED,
        coverage=(
            _resolved_pass("R1"),
            CoverageRow(rule_id="R2", status="fail", evidence="missing CTA"),  # type: ignore[arg-type]
        ),
    )
    assert review_verdict(pr) is ReviewDecision.REVISION_REQUIRED
    assert any("failed required coverage" in r for r in verdict_reasons(pr))


def test_verdict_reasons_tolerates_non_str_rule_id() -> None:
    pr = ContentPR(
        rule_packet=_PINNED,
        coverage=(CoverageRow(rule_id=None),),  # type: ignore[arg-type]  # unresolved
    )
    # Must not raise on join of a non-str rule_id.
    assert any("unresolved required coverage" in r for r in verdict_reasons(pr))


def test_blocking_claim_requires_revision() -> None:
    bad_claim = map_claim(
        ExtractedClaim(text="SSO costs extra", registry_id="feature.sso"),
        _REGISTRY,
        as_of=date(2026, 6, 6),
    )  # MISMATCH -> blocking
    pr = ContentPR(rule_packet=_PINNED, coverage=(_resolved_pass("R1"),), claims=(bad_claim,))
    assert review_verdict(pr) is ReviewDecision.REVISION_REQUIRED
    assert any("blocking claim" in r for r in verdict_reasons(pr))


def test_blocking_comment_requires_revision() -> None:
    pr = ContentPR(
        rule_packet=_PINNED,
        coverage=(_resolved_pass("R1"),),
        comments=(ReviewComment(category=CommentCategory.COMPLIANCE, message="no disclaimer", blocking=True),),
    )
    assert review_verdict(pr) is ReviewDecision.REVISION_REQUIRED


def test_incomplete_review_blocks_before_content_failures() -> None:
    # An unresolved required row AND a failing row -> BLOCKED wins (the review
    # isn't even complete enough to demand a specific revision).
    pr = ContentPR(
        rule_packet=_PINNED,
        coverage=(
            CoverageRow(rule_id="R1"),  # unresolved
            CoverageRow(rule_id="R2", status=CoverageStatus.FAIL, evidence="x"),
        ),
    )
    assert review_verdict(pr) is ReviewDecision.BLOCKED


def test_verdict_never_auto_produces_human_only_states() -> None:
    # Sweep a few PRs; auto verdict is only ever one of these three.
    allowed = {
        ReviewDecision.APPROVED,
        ReviewDecision.REVISION_REQUIRED,
        ReviewDecision.BLOCKED,
    }
    assert review_verdict(_approvable_pr()) in allowed
    assert review_verdict(ContentPR()) in allowed  # empty/unpinned


def test_content_pr_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    pr = _approvable_pr()
    with pytest.raises(FrozenInstanceError):
        pr.asset_id = "other"  # type: ignore[misc]
