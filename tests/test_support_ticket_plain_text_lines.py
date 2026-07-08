"""S6B: line-preserving HTML hygiene for support-ticket text.

Covers the round-trip contract: block tags emit line boundaries, excluded
bodies (script/style/blockquote) stay excluded, malformed skip tags cannot
erase later customer text, and the compact API keeps its exact prior shape.
"""

from __future__ import annotations

import re

import pytest

from extracted_content_pipeline.support_ticket_clustering import (
    _BLOCK_TAG_NAMES,
    _HTML_TAG_NAMES_RE,
    support_ticket_plain_text,
    support_ticket_plain_text_lines,
)


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        (
            "<p>How do I export?</p><p>Thanks,</p><div>Jane Smith</div>",
            "How do I export?\nThanks,\nJane Smith",
        ),
        ("Line one<br>Line two<br/>Line three", "Line one\nLine two\nLine three"),
        ("<ul><li>First</li><li>Second</li></ul>", "First\nSecond"),
        (
            "<h1>Title</h1><table><tr><td>A</td><td>B</td></tr>"
            "<tr><td>C</td></tr></table>",
            "Title\nA B\nC",
        ),
        ("<section><h2>Head</h2><p>Body</p></section>", "Head\nBody"),
    ],
)
def test_block_tags_emit_line_boundaries(html: str, expected: str) -> None:
    assert support_ticket_plain_text_lines(html) == expected


def test_inline_tags_stay_inline() -> None:
    assert support_ticket_plain_text_lines(
        "Use <b>billing portal</b> now"
    ) == "Use billing portal now"


def test_plain_text_keeps_its_own_newlines() -> None:
    assert support_ticket_plain_text_lines("Line one\n\nLine two") == (
        "Line one\nLine two"
    )


def test_escaped_html_reaches_line_extraction() -> None:
    assert support_ticket_plain_text_lines(
        "&lt;p&gt;First&lt;/p&gt;&lt;p&gt;Second&lt;/p&gt;"
    ) == "First\nSecond"


def test_blockquote_bodies_are_excluded_with_lines() -> None:
    html = (
        "<p>See below</p>"
        "<blockquote><p>On Mon Bob wrote: reset your password</p></blockquote>"
        "<p>My new question</p>"
    )
    assert support_ticket_plain_text_lines(html) == "See below\nMy new question"


def test_blockquote_bodies_are_excluded_from_compact_text() -> None:
    assert support_ticket_plain_text(
        "<p>See below</p><blockquote>quoted stuff</blockquote>"
        "<p>My new question</p>"
    ) == "See below My new question"


def test_nested_blockquotes_stay_excluded() -> None:
    assert support_ticket_plain_text_lines(
        "<blockquote>a<blockquote>b</blockquote>c</blockquote><p>new</p>"
    ) == "new"


@pytest.mark.parametrize("tag", ["script", "style"])
def test_script_style_bodies_stay_excluded(tag: str) -> None:
    assert support_ticket_plain_text(
        f"<{tag}>machinery</{tag}><p>Real text</p>"
    ) == "Real text"


def test_unclosed_script_does_not_swallow_the_ticket() -> None:
    # Pre-S6B this returned "" -- the whole ticket body was lost to the
    # parser's CDATA mode. Recovery keeps the customer text.
    result = support_ticket_plain_text(
        "<script>var x=1;<p>My invoice total is wrong</p>"
    )
    assert "My invoice total is wrong" in result


def test_self_closing_script_does_not_suppress_later_text() -> None:
    assert support_ticket_plain_text(
        "<script/><p>Real customer text</p>"
    ) == "Real customer text"


def test_unclosed_blockquote_stays_excluded() -> None:
    # A quote running to end-of-message is a quote, not data loss.
    assert support_ticket_plain_text_lines(
        "<p>intro</p><blockquote>quote to eof"
    ) == "intro"


def test_all_quote_body_falls_back_instead_of_losing_the_row() -> None:
    assert support_ticket_plain_text(
        "<blockquote><p>How do I export invoices?</p></blockquote>"
    ) == "How do I export invoices?"


def test_all_quote_body_is_empty_for_line_hygiene() -> None:
    assert support_ticket_plain_text_lines(
        "<blockquote><p>How do I export invoices?</p></blockquote>"
    ) == ""


def test_compact_api_shape_is_unchanged_for_inline_html() -> None:
    assert support_ticket_plain_text(
        'Please <a href="https://ex.test/x">click here</a> to reset and '
        "<b>confirm</b>"
    ) == "Please click here to reset and confirm"


def test_compact_api_shape_is_unchanged_for_plain_text() -> None:
    assert support_ticket_plain_text("If a<b and c>d then fail") == (
        "If a<b and c>d then fail"
    )
    assert support_ticket_plain_text("hello   world\nagain") == "hello world again"


def test_customer_wording_about_auto_reply_is_untouched() -> None:
    assert support_ticket_plain_text_lines(
        "<p>How do I set an out of office auto-reply?</p>"
    ) == "How do I set an out of office auto-reply?"


def test_every_block_tag_is_in_the_detector_families() -> None:
    # Drift guard: line extraction and HTML detection share tag families.
    for tag in sorted(_BLOCK_TAG_NAMES):
        assert re.fullmatch(_HTML_TAG_NAMES_RE, tag, re.IGNORECASE), tag
