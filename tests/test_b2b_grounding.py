"""Tests for the normalized grounding helper (Phase 1b).

Covers:
- `_normalize_for_grounding`: whitespace, unicode punctuation, zero-width
  characters, markdown emphasis, backslash escapes, None/non-string inputs.
- `check_phrase_grounded`: exact and noisy substring matches against a
  combined summary+review source blob.
"""

from __future__ import annotations

from atlas_brain.autonomous.tasks._b2b_grounding import (
    _normalize_for_grounding,
    check_phrase_grounded,
)


def test_normalize_passthrough_clean_text():
    assert _normalize_for_grounding("hello world") == "hello world"


def test_normalize_collapses_whitespace_runs():
    assert _normalize_for_grounding("a\n\n  b\t\tc") == "a b c"


def test_normalize_strips_leading_trailing_whitespace():
    assert _normalize_for_grounding("  hello  ") == "hello"


def test_normalize_casefolds():
    assert _normalize_for_grounding("Hello WORLD") == "hello world"


def test_normalize_curly_single_quotes_to_ascii():
    # \u2019 = right single quote (typographic apostrophe)
    assert _normalize_for_grounding("it\u2019s great") == "it's great"
    # \u2018 = left single quote
    assert _normalize_for_grounding("it\u2018s great") == "it's great"


def test_normalize_curly_double_quotes_to_ascii():
    # \u201C / \u201D = left / right double quotes
    assert _normalize_for_grounding("\u201Chello\u201D") == '"hello"'


def test_normalize_em_dash_to_double_hyphen():
    # \u2014 = em dash
    assert _normalize_for_grounding("a \u2014 b") == "a -- b"


def test_normalize_en_dash_to_single_hyphen():
    # \u2013 = en dash
    assert _normalize_for_grounding("pre\u2013post") == "pre-post"


def test_normalize_ellipsis_to_three_dots():
    # \u2026 = horizontal ellipsis
    assert _normalize_for_grounding("wait\u2026") == "wait..."


def test_normalize_non_breaking_space_to_space():
    # \u00A0 = non-breaking space
    assert _normalize_for_grounding("hello\u00A0world") == "hello world"


def test_normalize_strips_zero_width_chars():
    # \u200B = ZWSP; \u200C = ZWNJ; \u200D = ZWJ; \uFEFF = BOM
    assert _normalize_for_grounding("he\u200Bllo") == "hello"
    assert _normalize_for_grounding("he\u200Cllo") == "hello"
    assert _normalize_for_grounding("he\u200Dllo") == "hello"
    assert _normalize_for_grounding("\uFEFFhello") == "hello"


def test_normalize_markdown_bold_asterisks():
    assert _normalize_for_grounding("this is **bold** text") == "this is bold text"


def test_normalize_markdown_bold_underscores():
    assert _normalize_for_grounding("this is __bold__ text") == "this is bold text"


def test_normalize_markdown_italic_asterisks():
    assert _normalize_for_grounding("this is *italic* text") == "this is italic text"


def test_normalize_markdown_italic_underscores():
    assert _normalize_for_grounding("this is _italic_ text") == "this is italic text"


def test_normalize_strips_backslash_escapes_before_punct():
    assert _normalize_for_grounding(r"it\'s a test") == "it's a test"
    assert _normalize_for_grounding(r"no \. way") == "no . way"


def test_normalize_none_is_empty_string():
    assert _normalize_for_grounding(None) == ""


def test_normalize_non_string_is_empty_string():
    assert _normalize_for_grounding(42) == ""
    assert _normalize_for_grounding([]) == ""


def test_normalize_empty_string():
    assert _normalize_for_grounding("") == ""


def test_check_exact_match_in_review():
    assert check_phrase_grounded("hello", review_text="hello world") is True


def test_check_case_insensitive_match():
    assert check_phrase_grounded("HELLO", review_text="hello world") is True


def test_check_whitespace_normalization_bridges_mismatch():
    assert check_phrase_grounded("hello world", review_text="hello   world") is True
    assert check_phrase_grounded("hello world", review_text="hello\nworld") is True


def test_check_matches_phrase_with_curly_quotes_against_ascii_source():
    messy_phrase = "it\u2019s \u201Ctoo expensive\u201D"
    raw_review = 'It\'s "too expensive" for what we use.'
    assert check_phrase_grounded(messy_phrase, review_text=raw_review) is True


def test_check_matches_ascii_phrase_against_curly_quote_source():
    phrase = "it's good"
    messy_review = "it\u2019s good"
    assert check_phrase_grounded(phrase, review_text=messy_review) is True


def test_check_matches_through_markdown_wrapper():
    assert check_phrase_grounded("hello world", review_text="**hello** world") is True


def test_check_found_in_summary_alone():
    assert check_phrase_grounded(
        "hello",
        summary="hello there",
        review_text="different text entirely",
    ) is True


def test_check_rejects_paraphrase():
    paraphrase = "they charge way too much money"
    raw_review = "It's too expensive for what we use."
    assert check_phrase_grounded(paraphrase, review_text=raw_review) is False


def test_check_rejects_phrase_not_in_source():
    assert check_phrase_grounded(
        "missing phrase",
        review_text="something else entirely",
    ) is False


def test_check_empty_phrase_returns_false():
    assert check_phrase_grounded("", review_text="anything") is False
    assert check_phrase_grounded(None, review_text="anything") is False


def test_check_empty_source_returns_false():
    assert check_phrase_grounded("hello", review_text="") is False
    assert check_phrase_grounded("hello", summary=None, review_text=None) is False


def test_check_rejects_phrase_only_present_across_summary_review_boundary():
    # Quote-grade verbatim cannot accept a phrase that exists only because
    # we glued summary + review together. Each candidate source must contain
    # the phrase on its own.
    assert check_phrase_grounded(
        "title. body starts",
        summary="title.",
        review_text="body starts with this sentence",
    ) is False


def test_check_accepts_phrase_in_summary_alone():
    # Summary-only grounding is still a real quote (just from the title).
    assert check_phrase_grounded(
        "great product overall",
        summary="Great product overall, would buy again.",
        review_text="The reviewer wrote at length about other features.",
    ) is True


def test_check_accepts_phrase_in_review_alone():
    assert check_phrase_grounded(
        "really frustrating experience",
        summary="Short title.",
        review_text="It was a really frustrating experience from day one.",
    ) is True


def test_check_realistic_messy_review():
    review = (
        "I used Monday few years ago, it was helpful but a little to "
        "expensive.\nFew days ago I visited monday.com to take a look as "
        "I am looking for similar app.\tI did not subscribe or give my "
        "email but still I\u2019m getting sales emails that I can\u2019t "
        "unsubscribe from."
    )
    # Phrase extracted with typographic apostrophe, source has same
    assert check_phrase_grounded(
        "getting sales emails that I can\u2019t unsubscribe from",
        review_text=review,
    ) is True
    # Extracted with ASCII apostrophe, source has curly
    assert check_phrase_grounded(
        "getting sales emails that I can't unsubscribe from",
        review_text=review,
    ) is True
