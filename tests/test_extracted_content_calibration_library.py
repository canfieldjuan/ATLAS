"""Unit tests for slice 5a: the review calibration library.

Pure value types + query helpers; no DB, no async, no Atlas imports, no LLM.
Composes slice 1 (ReviewDecision, FailureCategory, ExceptionRecord).
"""

from __future__ import annotations

from datetime import date

from extracted_content_pipeline.calibration_library import (
    CalibrationExample,
    CalibrationLabel,
    CalibrationLibrary,
    example_from_exception,
    is_negative_label,
    is_positive_label,
)
from extracted_content_pipeline.review_contract import (
    ExceptionRecord,
    FailureCategory,
    ReviewDecision,
)


def _ex(
    example_id: str,
    label: CalibrationLabel,
    *,
    excerpt: str = "some worked copy",
    reasoning: str = "why it earned the label",
    verdict: ReviewDecision | None = None,
    failure_category: FailureCategory | None = None,
) -> CalibrationExample:
    return CalibrationExample(
        example_id=example_id,
        excerpt=excerpt,
        label=label,
        reasoning=reasoning,
        verdict=verdict,
        failure_category=failure_category,
    )


# -- label polarity ----------------------------------------------------------


def test_positive_labels_are_the_do_this_anchors() -> None:
    assert is_positive_label(CalibrationLabel.APPROVED) is True
    assert is_positive_label(CalibrationLabel.GOOD_VOICE) is True
    assert is_positive_label(CalibrationLabel.STRONG_PERSUASION) is True
    assert is_positive_label(CalibrationLabel.OVERCLAIM) is False


def test_negative_labels_are_the_not_this_anchors() -> None:
    for label in (
        CalibrationLabel.REJECTED,
        CalibrationLabel.KNOWN_DEFECT,
        CalibrationLabel.VOICE_DRIFT,
        CalibrationLabel.OVERCLAIM,
        CalibrationLabel.WEAK_PERSUASION,
    ):
        assert is_negative_label(label) is True
    assert is_negative_label(CalibrationLabel.APPROVED) is False


def test_borderline_is_neither_positive_nor_negative() -> None:
    assert is_positive_label(CalibrationLabel.BORDERLINE) is False
    assert is_negative_label(CalibrationLabel.BORDERLINE) is False


def test_label_polarity_classifies_decoded_string() -> None:
    # A label arriving as a plain string (decoded JSON) classifies by value.
    assert is_positive_label("approved") is True
    assert is_negative_label("overclaim") is True
    assert is_positive_label("overclaim") is False


# -- CalibrationExample ------------------------------------------------------


def test_example_polarity_delegates_to_label() -> None:
    assert _ex("e1", CalibrationLabel.GOOD_VOICE).is_positive() is True
    assert _ex("e2", CalibrationLabel.VOICE_DRIFT).is_negative() is True


def test_example_is_teachable_requires_excerpt_and_reasoning() -> None:
    assert _ex("e1", CalibrationLabel.APPROVED).is_teachable() is True
    assert _ex("e2", CalibrationLabel.APPROVED, excerpt="   ").is_teachable() is False
    assert _ex("e3", CalibrationLabel.APPROVED, reasoning="").is_teachable() is False


def test_example_is_teachable_tolerates_decoded_none() -> None:
    # excerpt/reasoning may arrive as JSON null; that counts as missing, not a raise.
    decoded = CalibrationExample(
        example_id="e1",
        excerpt=None,  # type: ignore[arg-type]
        label=CalibrationLabel.OVERCLAIM,
        reasoning=None,  # type: ignore[arg-type]
    )
    assert decoded.is_teachable() is False


# -- CalibrationLibrary queries ----------------------------------------------


def _library() -> CalibrationLibrary:
    return CalibrationLibrary(
        examples=(
            _ex("p1", CalibrationLabel.GOOD_VOICE, verdict=ReviewDecision.APPROVED),
            _ex(
                "n1",
                CalibrationLabel.OVERCLAIM,
                verdict=ReviewDecision.REVISION_REQUIRED,
                failure_category=FailureCategory.CLAIM_CREDIBILITY_GAP,
            ),
            _ex(
                "n2",
                CalibrationLabel.WEAK_PERSUASION,
                failure_category=FailureCategory.WEAK_HOOK,
            ),
            _ex("b1", CalibrationLabel.BORDERLINE),
        )
    )


def test_by_label_returns_only_matching_in_order() -> None:
    lib = _library()
    assert tuple(e.example_id for e in lib.by_label(CalibrationLabel.OVERCLAIM)) == ("n1",)
    assert lib.by_label(CalibrationLabel.REJECTED) == ()


def test_by_label_matches_decoded_string_label() -> None:
    # A stored label decoded as a plain string still matches the enum query.
    lib = CalibrationLibrary(
        examples=(
            CalibrationExample(
                example_id="d1",
                excerpt="copy",
                label="overclaim",  # type: ignore[arg-type]
                reasoning="r",
            ),
        )
    )
    assert tuple(e.example_id for e in lib.by_label(CalibrationLabel.OVERCLAIM)) == ("d1",)


def test_positives_and_negatives_split_the_set() -> None:
    lib = _library()
    assert tuple(e.example_id for e in lib.positives()) == ("p1",)
    assert tuple(e.example_id for e in lib.negatives()) == ("n1", "n2")


def test_by_failure_category_filters_and_ignores_uncategorized() -> None:
    lib = _library()
    assert tuple(e.example_id for e in lib.by_failure_category(FailureCategory.WEAK_HOOK)) == (
        "n2",
    )
    # The positive and borderline examples have no category -> never swept in.
    assert lib.by_failure_category(FailureCategory.TIMING) == ()


def test_by_failure_category_none_returns_empty() -> None:
    # Detection branch: a None query must not match the uncategorized examples.
    assert _library().by_failure_category(None) == ()  # type: ignore[arg-type]


def test_by_verdict_filters_and_none_returns_empty() -> None:
    lib = _library()
    assert tuple(e.example_id for e in lib.by_verdict(ReviewDecision.APPROVED)) == ("p1",)
    assert lib.by_verdict(None) == ()  # type: ignore[arg-type]


def test_teachable_drops_decoration() -> None:
    lib = CalibrationLibrary(
        examples=(
            _ex("ok", CalibrationLabel.APPROVED),
            _ex("blank", CalibrationLabel.APPROVED, reasoning=""),
        )
    )
    assert tuple(e.example_id for e in lib.teachable()) == ("ok",)


def test_labels_covered_reports_distinct_labels() -> None:
    assert _library().labels_covered() == frozenset(
        {
            CalibrationLabel.GOOD_VOICE,
            CalibrationLabel.OVERCLAIM,
            CalibrationLabel.WEAK_PERSUASION,
            CalibrationLabel.BORDERLINE,
        }
    )


def test_missing_labels_reports_gaps_in_order_deduped() -> None:
    lib = _library()
    required = (
        CalibrationLabel.GOOD_VOICE,  # covered
        CalibrationLabel.VOICE_DRIFT,  # gap
        CalibrationLabel.REJECTED,  # gap
        CalibrationLabel.VOICE_DRIFT,  # duplicate gap -> deduped
    )
    assert lib.missing_labels(required) == (
        CalibrationLabel.VOICE_DRIFT,
        CalibrationLabel.REJECTED,
    )


def test_empty_library_is_all_gaps() -> None:
    lib = CalibrationLibrary()
    assert lib.positives() == ()
    assert lib.labels_covered() == frozenset()
    assert lib.missing_labels((CalibrationLabel.APPROVED,)) == (CalibrationLabel.APPROVED,)


# -- overrides feed the set (example_from_exception) --------------------------


def test_example_from_exception_defaults_to_borderline_override_anchor() -> None:
    record = ExceptionRecord(
        rule="brand_voice.no_superlatives",
        reason="founder quote keeps the superlative; on-brand exception",
        owner="editor@example.com",
        expiration=date(2026, 12, 31),
    )
    example = example_from_exception(record, excerpt="the best onboarding, period.")
    assert example.example_id == "override:brand_voice.no_superlatives"
    assert example.label == CalibrationLabel.BORDERLINE
    assert example.verdict == ReviewDecision.APPROVED_WITH_EXCEPTION
    assert example.reasoning == record.reason
    assert example.source == "override"
    assert example.is_teachable() is True


def test_example_from_exception_honors_explicit_label_and_category() -> None:
    record = ExceptionRecord(rule="claim.tos_uptime", reason="legacy SLA still cited", owner="x")
    example = example_from_exception(
        record,
        excerpt="99.99% uptime guaranteed",
        label=CalibrationLabel.OVERCLAIM,
        failure_category=FailureCategory.CLAIM_CREDIBILITY_GAP,
    )
    assert example.label == CalibrationLabel.OVERCLAIM
    assert example.failure_category == FailureCategory.CLAIM_CREDIBILITY_GAP
    assert example.is_negative() is True
    # An override-fed negative example is queryable like any curated one.
    lib = CalibrationLibrary(examples=(example,))
    assert lib.by_failure_category(FailureCategory.CLAIM_CREDIBILITY_GAP) == (example,)
