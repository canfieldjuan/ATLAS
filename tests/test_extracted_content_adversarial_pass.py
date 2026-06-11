"""Unit tests for slice 5b: the adversarial review pass.

Pure value types + deterministic merge helpers; no DB, no async, no Atlas
imports, no LLM. Composes slice 4 (ReviewComment / CommentCategory).
"""

from __future__ import annotations

from extracted_content_pipeline.adversarial_pass import (
    AdversarialFinding,
    AdversarialFindingCategory,
    AdversarialPass,
    comment_from_finding,
    corroborated_categories,
    disagreement_categories,
    merge_findings,
)
from extracted_content_pipeline.content_pr import CommentCategory


def _finding(
    category: AdversarialFindingCategory,
    *,
    message: str = "the strongest reason this should not ship",
    evidence: str = "quoted draft text",
    location: str = "para 2",
) -> AdversarialFinding:
    return AdversarialFinding(
        category=category, message=message, evidence=evidence, location=location
    )


# -- AdversarialFinding -------------------------------------------------------


def test_finding_is_substantiated_requires_message_and_evidence() -> None:
    assert _finding(AdversarialFindingCategory.OVERCLAIM).is_substantiated() is True
    assert (
        _finding(AdversarialFindingCategory.OVERCLAIM, message="  ").is_substantiated()
        is False
    )
    assert (
        _finding(AdversarialFindingCategory.OVERCLAIM, evidence="").is_substantiated()
        is False
    )


def test_finding_is_substantiated_tolerates_decoded_none() -> None:
    # message/evidence may arrive as JSON null; missing, not a raise.
    decoded = AdversarialFinding(
        category=AdversarialFindingCategory.AMBIGUITY,
        message=None,  # type: ignore[arg-type]
        evidence=None,  # type: ignore[arg-type]
    )
    assert decoded.is_substantiated() is False


# -- AdversarialPass ----------------------------------------------------------


def _pass_a() -> AdversarialPass:
    return AdversarialPass(
        pass_id="a",
        source="adversarial-prompt@v1 / model-a",
        findings=(
            _finding(AdversarialFindingCategory.OVERCLAIM),
            _finding(AdversarialFindingCategory.MISSING_PROOF),
            _finding(AdversarialFindingCategory.AMBIGUITY, evidence=""),  # noise
        ),
    )


def _pass_b() -> AdversarialPass:
    return AdversarialPass(
        pass_id="b",
        source="adversarial-prompt@v2 / model-b",
        findings=(
            _finding(
                AdversarialFindingCategory.OVERCLAIM,
                message="differently worded overclaim objection",
            ),
            _finding(AdversarialFindingCategory.VOICE_SLIP),
        ),
    )


def test_pass_categories_are_distinct() -> None:
    assert _pass_a().categories() == frozenset(
        {
            AdversarialFindingCategory.OVERCLAIM,
            AdversarialFindingCategory.MISSING_PROOF,
            AdversarialFindingCategory.AMBIGUITY,
        }
    )


def test_pass_substantiated_filters_noise_in_order() -> None:
    kept = _pass_a().substantiated()
    assert tuple(f.category for f in kept) == (
        AdversarialFindingCategory.OVERCLAIM,
        AdversarialFindingCategory.MISSING_PROOF,
    )


def test_empty_pass_has_no_categories_or_findings() -> None:
    empty = AdversarialPass(pass_id="e")
    assert empty.categories() == frozenset()
    assert empty.substantiated() == ()


# -- corroboration / disagreement ---------------------------------------------


def test_corroborated_categories_is_the_intersection() -> None:
    assert corroborated_categories(_pass_a(), _pass_b()) == frozenset(
        {AdversarialFindingCategory.OVERCLAIM}
    )


def test_disagreement_categories_is_the_symmetric_difference() -> None:
    assert disagreement_categories(_pass_a(), _pass_b()) == frozenset(
        {
            AdversarialFindingCategory.MISSING_PROOF,
            AdversarialFindingCategory.AMBIGUITY,
            AdversarialFindingCategory.VOICE_SLIP,
        }
    )


def test_identical_passes_have_no_disagreement() -> None:
    assert disagreement_categories(_pass_a(), _pass_a()) == frozenset()


def test_corroboration_matches_decoded_string_category() -> None:
    # One pass decoded its category as a plain string; still corroborates.
    decoded = AdversarialPass(
        pass_id="d",
        findings=(
            AdversarialFinding(
                category="overclaim",  # type: ignore[arg-type]
                message="m",
                evidence="e",
            ),
        ),
    )
    assert corroborated_categories(decoded, _pass_b()) == frozenset(
        {AdversarialFindingCategory.OVERCLAIM}
    )


# -- merge_findings -----------------------------------------------------------


def test_merge_preserves_first_then_second_order() -> None:
    merged = merge_findings(_pass_a(), _pass_b())
    assert tuple(f.category for f in merged) == (
        AdversarialFindingCategory.OVERCLAIM,
        AdversarialFindingCategory.MISSING_PROOF,
        AdversarialFindingCategory.AMBIGUITY,
        AdversarialFindingCategory.OVERCLAIM,  # differently worded -> kept
        AdversarialFindingCategory.VOICE_SLIP,
    )


def test_merge_drops_only_exact_duplicates() -> None:
    shared = _finding(AdversarialFindingCategory.GENERIC_STRETCH)
    first = AdversarialPass(pass_id="a", findings=(shared,))
    second = AdversarialPass(
        pass_id="b",
        findings=(
            shared,  # exact duplicate -> merged away
            _finding(
                AdversarialFindingCategory.GENERIC_STRETCH,
                message="same category, different objection",
            ),
        ),
    )
    merged = merge_findings(first, second)
    assert len(merged) == 2
    assert merged[0] == shared


def test_merge_of_empty_passes_is_empty() -> None:
    assert merge_findings(AdversarialPass(pass_id="a"), AdversarialPass(pass_id="b")) == ()


# -- comment_from_finding (the "not a judge" seam) ------------------------------


def test_comment_is_never_blocking() -> None:
    for category in AdversarialFindingCategory:
        comment = comment_from_finding(_finding(category))
        assert comment.blocking is False


def test_voice_slip_maps_to_brand_rule_lane() -> None:
    comment = comment_from_finding(_finding(AdversarialFindingCategory.VOICE_SLIP))
    assert comment.category == CommentCategory.BRAND_RULE


def test_other_findings_map_to_editorial_judgment_lane() -> None:
    comment = comment_from_finding(_finding(AdversarialFindingCategory.OVERCLAIM))
    assert comment.category == CommentCategory.EDITORIAL_JUDGMENT


def test_comment_carries_category_prefix_message_and_evidence() -> None:
    finding = _finding(
        AdversarialFindingCategory.MISSING_PROOF,
        message="cites no source for the 40% number",
        evidence="cuts support tickets by 40%",
    )
    comment = comment_from_finding(finding)
    assert comment.message == "[adversarial:missing_proof] cites no source for the 40% number"
    assert comment.evidence == "cuts support tickets by 40%"


def test_comment_from_unsubstantiated_finding_still_never_blocks() -> None:
    # Even an empty-message finding converts safely (prefix only, no raise).
    comment = comment_from_finding(
        AdversarialFinding(category=AdversarialFindingCategory.AMBIGUITY)
    )
    assert comment.message == "[adversarial:ambiguity]"
    assert comment.blocking is False
