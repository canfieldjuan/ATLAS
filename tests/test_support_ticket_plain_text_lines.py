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
    _HTML_EXCLUDED_TAG_RE,
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
    # Drift guard: line extraction and HTML detection share tag families --
    # either the bare signal families or the excluded-tag family.
    for tag in sorted(_BLOCK_TAG_NAMES):
        bare = re.fullmatch(_HTML_TAG_NAMES_RE, tag, re.IGNORECASE)
        excluded = _HTML_EXCLUDED_TAG_RE.fullmatch(f"<{tag}>")
        assert bare or excluded, tag


# Round-1 review refinements: script/style are detector signals, EOF
# recovery admits only customer text, excluded-scope boundaries follow the
# block rule, and the lines seam is exported.


def test_script_only_bodies_are_excluded_not_passed_through() -> None:
    assert support_ticket_plain_text_lines("<script>alert(1)</script>") == ""
    assert support_ticket_plain_text("<script>alert(1)</script>") == ""


def test_script_then_text_extracts_only_the_text() -> None:
    assert support_ticket_plain_text_lines(
        "<script>x=1;</script><p>Real question</p>"
    ) == "Real question"


def test_unclosed_script_recovery_drops_machinery() -> None:
    assert support_ticket_plain_text(
        "<p>before</p><script>var x=1;<p>after</p>"
    ) == "before after"
    assert support_ticket_plain_text_lines(
        "<p>before</p><script>var x=1;<p>after</p>"
    ) == "before\nafter"


def test_unclosed_script_with_no_markup_recovers_nothing() -> None:
    assert support_ticket_plain_text(
        "<p>before</p><script>var x=1; no more tags"
    ) == "before"


def test_inline_excluded_scope_does_not_split_the_line() -> None:
    assert support_ticket_plain_text_lines(
        "<p>foo<script>x</script>bar</p>"
    ) == "foo bar"


def test_lines_seam_is_exported() -> None:
    import extracted_content_pipeline.support_ticket_clustering as module

    assert "support_ticket_plain_text_lines" in module.__all__


# Round-2 refinement: EOF recovery ignores markup inside script string
# literals and comments -- code templates are code, not lost ticket text.


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ('<script>var t="<p>template</p>";<p>Real after</p>', "Real after"),
        ("<script>let t=`<div>tpl</div>`;<p>Real</p>", "Real"),
        ("<script>// <p>note</p>\n<p>Real</p>", "Real"),
        ("<script>/* <p>x</p> */<p>Real</p>", "Real"),
        ('<script>var s="a\\"<p>t</p>";<p>Real</p>', "Real"),
    ],
)
def test_unclosed_script_recovery_skips_code_literals(
    html: str, expected: str,
) -> None:
    assert support_ticket_plain_text(html) == expected


@pytest.mark.parametrize(
    "html",
    [
        '<p>before</p><script>var t="<p>only template</p>";',
        '<p>before</p><script>var t="<p>unterminated tpl</p>',
    ],
)
def test_unclosed_script_with_only_code_templates_recovers_nothing(
    html: str,
) -> None:
    assert support_ticket_plain_text(html) == "before"


# Round-3 refinements: regex-literal masking, no invented recovery boundary,
# and lone script/style mentions in prose stay customer wording.


def test_unclosed_script_regex_literal_is_masked() -> None:
    assert support_ticket_plain_text(
        "<script>var r=/<p>template<\\/p>/;<p>Real</p>"
    ) == "Real"


def test_division_is_not_a_regex_literal() -> None:
    assert support_ticket_plain_text(
        "<script>var a=b / 2; c=d/e;<p>Real</p>"
    ) == "Real"


def test_recovered_inline_markup_keeps_inline_boundary() -> None:
    assert support_ticket_plain_text_lines(
        "<p>foo<script>x;<span>bar</span></p>"
    ) == "foo bar"


@pytest.mark.parametrize(
    "text",
    [
        "How do I add <script> to the page?",
        "What does <style> do?",
    ],
)
def test_lone_script_style_mentions_stay_customer_wording(text: str) -> None:
    assert support_ticket_plain_text(text) == text


def test_paired_script_only_body_still_excluded() -> None:
    assert support_ticket_plain_text("<script>alert(1)</script>") == ""


# Round-4 refinements: division after value literals, custom-element
# recovery, and CDATA recovery for scripts nested in excluded scopes.


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ('<script>var a="x" / 2;<p>Real</p>', "Real"),
        ("<script>var a={n:1} / 2;<p>Real</p>", "Real"),
    ],
)
def test_division_after_value_literals_is_not_a_regex(
    html: str, expected: str,
) -> None:
    assert support_ticket_plain_text(html) == expected


def test_recovery_finds_custom_element_markup() -> None:
    assert support_ticket_plain_text(
        "<script>x;<x-ticket>Real</x-ticket>"
    ) == "Real"


def test_malformed_script_inside_blockquote_recovers_tail() -> None:
    assert support_ticket_plain_text_lines(
        "<blockquote><script>x</blockquote><p>Real</p>"
    ) == "Real"


def test_quoted_text_before_nested_malformed_script_stays_excluded() -> None:
    assert support_ticket_plain_text_lines(
        "<blockquote>quoted<script>x</blockquote><p>Real</p>"
    ) == "Real"


# Round-5 refinements: postfix division, char classes, single-line regexes,
# CDATA-swallowed close-tag unwinding, and lone blockquote mentions.


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("<script>a++ / 2;<p>Real</p>", "Real"),
        ("<script>b-- / 2;<p>Real</p>", "Real"),
    ],
)
def test_postfix_division_is_not_a_regex(html: str, expected: str) -> None:
    assert support_ticket_plain_text(html) == expected


def test_regex_character_class_slash_does_not_close_the_regex() -> None:
    assert support_ticket_plain_text(
        "<script>var r=/[/]<p>x<\\/p>/;<p>Real</p>"
    ) == "Real"


def test_regex_candidates_cannot_span_lines() -> None:
    assert support_ticket_plain_text(
        "<script>a = b / 2\n<p>Real</p>"
    ) == "Real"


def test_cdata_swallowed_close_tags_are_unwound() -> None:
    assert support_ticket_plain_text_lines(
        "<blockquote><script>x</blockquote></script><p>Real</p>"
    ) == "Real"


def test_lone_blockquote_mention_stays_customer_wording() -> None:
    assert support_ticket_plain_text(
        "How do I add <blockquote> to my template?"
    ) == "How do I add <blockquote> to my template?"


def test_end_tag_slash_in_buffer_is_not_a_regex() -> None:
    assert support_ticket_plain_text_lines(
        "<blockquote><script>x</blockquote><p>Real</p>"
    ) == "Real"


# Round-6 refinements: position-based detection for excluded tags (start of
# body = markup intent, mid-prose = customer wording), recovery respects
# excluded bodies, and keyword-position regex literals are masked.


def test_body_starting_with_unclosed_script_is_excluded() -> None:
    assert support_ticket_plain_text_lines("<script>alert(1)") == ""


def test_body_starting_with_unclosed_blockquote_is_all_quote() -> None:
    assert support_ticket_plain_text_lines("<blockquote>quoted prior reply") == ""
    # The compact API keeps the row via the all-quote fallback.
    assert support_ticket_plain_text(
        "<blockquote>quoted prior reply"
    ) == "quoted prior reply"


def test_paired_blockquote_mention_mid_prose_is_preserved() -> None:
    text = "How do I write <blockquote>hello</blockquote> in the editor?"
    assert support_ticket_plain_text(text) == text


def test_recovery_does_not_start_inside_excluded_bodies() -> None:
    assert support_ticket_plain_text(
        "<p>x</p><script>y;<blockquote>quoted junk</blockquote><p>Real</p>"
    ) == "x Real"


@pytest.mark.parametrize(
    "html",
    [
        "<script>return /<p>tpl<\\/p>/;<p>Real</p>",
        "<script>let r = new RegExp(x); return /<p>t<\\/p>/;<p>Real</p>",
    ],
)
def test_keyword_position_regex_literals_are_masked(html: str) -> None:
    assert support_ticket_plain_text(html) == "Real"


# Round-7 refinements: property-access keywords, spaced less-than regexes,
# single-consumption CDATA unwind, and metadata-prefixed excluded bodies.


def test_property_access_keyword_is_division() -> None:
    assert support_ticket_plain_text(
        "<script>obj.return / 2;<p>Real</p>"
    ) == "Real"


def test_infix_keyword_opens_regex() -> None:
    assert support_ticket_plain_text(
        "<script>if (x in /<p>t<\\/p>/.exec(y)) {}<p>Real</p>"
    ) == "Real"


def test_spaced_less_than_opens_regex_but_end_tags_stay_markup() -> None:
    assert support_ticket_plain_text(
        "<script>x = y < /<p>tpl<\\/p>/;<p>Real</p>"
    ) == "Real"
    assert support_ticket_plain_text_lines(
        "<blockquote><script>x</blockquote><p>Real</p>"
    ) == "Real"


def test_cdata_unwind_consumes_each_close_tag_once() -> None:
    assert support_ticket_plain_text_lines(
        "<blockquote>outer<blockquote><script>x</blockquote></script>"
        "<p>Still quoted</p></blockquote><p>Real</p>"
    ) == "Real"


def test_metadata_prefixed_excluded_bodies_are_detected() -> None:
    assert support_ticket_plain_text_lines(
        "[External] <script>x</script>"
    ) == "[External]"
    assert support_ticket_plain_text_lines(
        "[Fwd] <blockquote>quoted reply"
    ) == "[Fwd]"


def test_metadata_prefixed_prose_mention_is_preserved() -> None:
    text = "[Urgent] How do I add <script> to the page?"
    assert support_ticket_plain_text(text) == text


# Round-8 refinements: per-scope buffer lifetime, regex-close as value,
# legacy HTML comments in scripts, recovery under open quotes, and prologs.


def test_closed_script_buffer_never_leaks_into_later_recovery() -> None:
    html = (
        "<blockquote><script>var a='<p>old junk</p>';</script>q</blockquote>"
        "<p>mid</p><script>y;<p>Real</p>"
    )
    assert support_ticket_plain_text_lines(html) == "mid\nReal"
    assert "old junk" not in support_ticket_plain_text(html)


def test_division_after_closed_regex_literal() -> None:
    assert support_ticket_plain_text(
        "<script>var r=/x/; var d = r / 2;<p>Real</p>"
    ) == "Real"


def test_legacy_html_comments_in_scripts_are_masked() -> None:
    assert support_ticket_plain_text_lines(
        "<script><!-- <p>template</p> -->\nvar x;<p>Real</p>"
    ) == "Real"


def test_no_recovery_while_an_outer_quote_stays_open() -> None:
    assert support_ticket_plain_text_lines(
        "<p>before</p><blockquote><script>x<p>quoted-tail</p>"
    ) == "before"


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("<!doctype html><script>x</script>", ""),
        ("<!-- fwd --><blockquote>quoted reply", ""),
    ],
)
def test_prologs_before_excluded_only_bodies_are_detected(
    html: str, expected: str,
) -> None:
    assert support_ticket_plain_text_lines(html) == expected
