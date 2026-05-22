"""Tests for Phase 2.3 blog quote-grade migration (Commit A1).

Coverage:
  - _quote_grade_blueprint_phrases gates on subject + polarity + verbatim
    via the enrichment contract, accepts both 'enrichment' and
    'enrichment_raw' keys, and stamps traceability fields onto output.
  - _remove_unmatched_quote_lines fails closed when source_quotes is
    empty (strips ALL blockquote lines), preserves matches when the pool
    is populated.

This commit pilots only _blueprint_pricing_reality_check; full producer
migration follows in a later commit. These tests lock in the helper
contract + validator behavior independently.
"""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import MagicMock

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)

from atlas_brain.autonomous.tasks.b2b_blog_post_generation import (  # noqa: E402
    _apply_specificity_anchor_repair,
    _blog_quote_highlights,
    _blueprint_best_fit_guide,
    _blueprint_market_landscape,
    _blueprint_pain_point_roundup,
    _blueprint_pricing_reality_check,
    _blueprint_vendor_showdown,
    _is_form_prompt,
    _is_placeholder_partner,
    _looks_like_orphan_quote_reference,
    _pick_affiliate_partner_for_vendors,
    _quote_grade_blueprint_phrases,
    _remove_unmatched_quote_lines,
    _split_and_gate_blog_quotes,
)


# ---------------------------------------------------------------------------
# _quote_grade_blueprint_phrases helper
# ---------------------------------------------------------------------------


def _v4_row(
    *,
    text: str = "it costs too much money",
    subject: str = "subject_vendor",
    polarity: str = "negative",
    role: str = "primary_driver",
    verbatim: bool = True,
    field: str = "pricing_phrases",
    enrichment_key: str = "enrichment_raw",
    vendor_name: str = "Shopify",
    review_id: str = "00000000-0000-0000-0000-000000000001",
    source: str = "g2",
    urgency: float = 8.0,
) -> dict[str, Any]:
    enrichment = {
        "enrichment_schema_version": 4,
        field: [text],
        "phrase_metadata": [
            {
                "field": field,
                "index": 0,
                "text": text,
                "subject": subject,
                "polarity": polarity,
                "role": role,
                "verbatim": verbatim,
            },
        ],
    }
    return {
        "vendor_name": vendor_name,
        "review_id": review_id,
        "source": source,
        "urgency": urgency,
        "reviewer_title": "Director of Operations",
        # Phase 2.3 wrapper marker: SQL/review rows must carry this so
        # _split_and_gate_blog_quotes routes them to the contract gate.
        # Unmarked rows are dropped at the wrapper -- closes the prior
        # 'no enrichment means vault' loophole.
        "quote_origin": "review",
        # Either key name should work
        enrichment_key: json.dumps(enrichment),
    }


def test_quote_grade_returns_empty_for_v3_legacy():
    """v3 rows without phrase_metadata produce nothing -- they cannot
    flow into the customer-facing quote pool."""
    legacy = {
        "vendor_name": "Shopify",
        "enrichment_raw": json.dumps({
            "enrichment_schema_version": 3,
            "pricing_phrases": ["too expensive", "price keeps rising"],
        }),
    }
    assert _quote_grade_blueprint_phrases([legacy]) == []


def test_quote_grade_returns_empty_for_missing_enrichment():
    row = {"vendor": "Shopify", "text": "some review text"}
    assert _quote_grade_blueprint_phrases([row]) == []


def test_quote_grade_accepts_enrichment_key():
    """Helper must accept BOTH 'enrichment' and 'enrichment_raw' keys."""
    row_a = _v4_row(enrichment_key="enrichment")
    row_b = _v4_row(enrichment_key="enrichment_raw")
    out_a = _quote_grade_blueprint_phrases([row_a])
    out_b = _quote_grade_blueprint_phrases([row_b])
    assert len(out_a) == 1
    assert len(out_b) == 1
    assert out_a[0]["phrase"] == out_b[0]["phrase"]


def test_quote_grade_accepts_dict_enrichment_directly():
    """Some producers pass parsed dicts; helper should not require a JSON string."""
    enrichment = {
        "enrichment_schema_version": 4,
        "pricing_phrases": ["it costs too much money"],
        "phrase_metadata": [{
            "field": "pricing_phrases", "index": 0, "text": "it costs too much money",
            "subject": "subject_vendor", "polarity": "negative",
            "role": "primary_driver", "verbatim": True,
        }],
    }
    row = {"vendor_name": "Shopify", "enrichment": enrichment}
    out = _quote_grade_blueprint_phrases([row])
    assert len(out) == 1
    assert out[0]["phrase"] == "it costs too much money"


def test_quote_grade_filters_competitor_subject():
    row = _v4_row(subject="competitor")
    assert _quote_grade_blueprint_phrases([row]) == []


def test_quote_grade_filters_self_subject():
    row = _v4_row(subject="self")
    assert _quote_grade_blueprint_phrases([row]) == []


def test_quote_grade_filters_positive_polarity():
    row = _v4_row(polarity="positive")
    assert _quote_grade_blueprint_phrases([row]) == []


def test_quote_grade_filters_non_verbatim():
    row = _v4_row(verbatim=False)
    assert _quote_grade_blueprint_phrases([row]) == []


def test_quote_grade_passes_when_all_gates_satisfied():
    row = _v4_row()
    out = _quote_grade_blueprint_phrases([row])
    assert len(out) == 1
    assert out[0]["phrase"] == "it costs too much money"


def test_quote_grade_stamps_traceability_fields():
    """Output dicts must carry review_id + source + field for downstream
    audit. Vendor/urgency/role passed through from the input row."""
    row = _v4_row(
        review_id="abc-123",
        source="g2",
        vendor_name="Shopify",
        urgency=7.5,
    )
    out = _quote_grade_blueprint_phrases([row])
    assert len(out) == 1
    item = out[0]
    assert item["review_id"] == "abc-123"
    assert item["source"] == "g2"
    assert item["vendor"] == "Shopify"
    assert item["urgency"] == 7.5
    assert item["field"] == "pricing_phrases"
    assert item["role"] == "Director of Operations"


def test_quote_grade_respects_field_scoping():
    """field='pricing_phrases' must only return pricing-tagged phrases."""
    enrichment = {
        "enrichment_schema_version": 4,
        "pricing_phrases": ["it costs too much"],
        "specific_complaints": ["the support is slow"],
        "phrase_metadata": [
            {"field": "pricing_phrases", "index": 0, "text": "it costs too much",
             "subject": "subject_vendor", "polarity": "negative",
             "role": "primary_driver", "verbatim": True},
            {"field": "specific_complaints", "index": 0, "text": "the support is slow",
             "subject": "subject_vendor", "polarity": "negative",
             "role": "primary_driver", "verbatim": True},
        ],
    }
    row = {"vendor_name": "Shopify", "enrichment_raw": json.dumps(enrichment)}
    pricing_only = _quote_grade_blueprint_phrases([row], field="pricing_phrases")
    assert len(pricing_only) == 1
    assert pricing_only[0]["phrase"] == "it costs too much"
    # No field filter -> both come through
    both = _quote_grade_blueprint_phrases([row])
    assert len(both) == 2


def test_quote_grade_respects_limit():
    rows = [_v4_row(text=f"phrase {i}", review_id=f"id-{i}") for i in range(5)]
    out = _quote_grade_blueprint_phrases(rows, limit=2)
    assert len(out) == 2


def test_quote_grade_handles_malformed_enrichment_string():
    """Bad JSON in enrichment_raw should be skipped, not raise."""
    row = {"vendor_name": "Shopify", "enrichment_raw": "{not json"}
    assert _quote_grade_blueprint_phrases([row]) == []


def test_quote_grade_handles_none_enrichment():
    row = {"vendor_name": "Shopify", "enrichment_raw": None}
    assert _quote_grade_blueprint_phrases([row]) == []


# ---------------------------------------------------------------------------
# _remove_unmatched_quote_lines fail-closed
# ---------------------------------------------------------------------------


def test_validator_strips_all_blockquotes_when_source_pool_empty():
    """Fail-closed: empty source_quotes means NO blockquote line passes.

    The prior behavior returned the markdown unchanged, allowing
    paraphrased LLM-generated quotes to ship when the producer hadn't
    supplied any verbatim source pool.
    """
    markdown = (
        "## Section\n"
        "\n"
        "Some prose.\n"
        "\n"
        "> a fabricated quote that has no source\n"
        "\n"
        "More prose.\n"
        "\n"
        "> another fabricated quote\n"
    )
    out, removed = _remove_unmatched_quote_lines(markdown, [])
    assert removed == 2
    assert "> a fabricated quote" not in out
    assert "> another fabricated quote" not in out
    # Non-blockquote content preserved
    assert "## Section" in out
    assert "Some prose." in out


def test_validator_preserves_matched_blockquote_when_pool_populated():
    source = ["it costs too much money for what you get"]
    markdown = (
        "## Section\n"
        "\n"
        "> it costs too much money for what you get\n"
        "\n"
        "More prose.\n"
    )
    out, removed = _remove_unmatched_quote_lines(markdown, source)
    assert removed == 0
    assert "> it costs too much money for what you get" in out


def test_validator_strips_unmatched_blockquote_when_pool_populated():
    source = ["it costs too much money for what you get"]
    markdown = (
        "## Section\n"
        "\n"
        "> a totally different fabricated quote about something else\n"
        "\n"
        "More prose.\n"
    )
    out, removed = _remove_unmatched_quote_lines(markdown, source)
    assert removed == 1
    assert "fabricated" not in out


def test_validator_handles_empty_markdown():
    out, removed = _remove_unmatched_quote_lines("", [])
    assert out == ""
    assert removed == 0


def test_validator_handles_markdown_without_blockquotes():
    markdown = "## Heading\n\nJust prose, no quotes."
    out, removed = _remove_unmatched_quote_lines(markdown, [])
    assert removed == 0
    assert out == markdown


# ---------------------------------------------------------------------------
# Block-level blockquote stripper -- formerly the orphan-attribution bug
#
# These tests originally documented a regression: the line-level stripper
# produced empty <blockquote><p>-- attribution</p></blockquote> elements in
# 44 of 80 published posts because it removed the quote line but kept the
# attribution line. The stripper is now block-level: it identifies a
# contiguous run of `>` lines, decides keep-or-strip based on whether ANY
# line in the block matches the source pool, and acts atomically on the
# whole block.
#
# The tests below assert the NEW correct behavior. The third test
# (introductory_prose) still documents a separate gap that block-level
# stripping doesn't address -- orphan introductory prose and disclaimer
# prose adjacent to stripped blocks. That gap is deferred to a future
# fix that detects and removes such prose.
# ---------------------------------------------------------------------------


def test_full_block_stripped_when_pool_empty():
    """Baseline: with empty source_quotes, the fail-closed path strips
    every blockquote block atomically.

    The stripper also sweeps the orphan introductory paragraph
    immediately preceding the block (``One reviewer noted:`` -- a
    single-line paragraph ending with ``:``) so the resulting body
    doesn't dangle the lead-in with no quote to follow."""
    markdown = (
        "## Section\n"
        "\n"
        "One reviewer noted:\n"
        "\n"
        "> \"A fabricated quote that won't match the source pool\"\n"
        "> -- Group Director, verified reviewer on TrustRadius\n"
        "\n"
        "More prose continues here.\n"
    )
    out, removed = _remove_unmatched_quote_lines(markdown, [])
    # Block (2 lines) + blank separator + intro line = 4 lines removed.
    assert removed == 4
    assert "fabricated quote" not in out
    assert "Group Director" not in out
    # Orphan intro prose is also gone.
    assert "One reviewer noted:" not in out
    # Surrounding content preserved.
    assert "## Section" in out
    assert "More prose continues here." in out


def test_full_block_stripped_when_pool_populated_no_match():
    """The fixed behavior: with a populated source pool, if NO line in
    the block has a quote body matching the pool, the WHOLE block is
    stripped -- including the attribution-only line that previously
    survived because _extract_quote_body returns empty for it.

    This is the test that was previously the bug-documenting regression
    (removed == 1, attribution remained). Now both lines removed."""
    source = ["something else entirely that won't match"]
    markdown = (
        "## Section\n"
        "\n"
        "> \"A quote the LLM made up\"\n"
        "> -- Group Director, verified reviewer on TrustRadius\n"
        "\n"
        "More prose.\n"
    )
    out, removed = _remove_unmatched_quote_lines(markdown, source)
    # Both lines of the block removed atomically.
    assert removed == 2
    assert "made up" not in out
    assert "-- Group Director" not in out
    # Surrounding prose preserved.
    assert "## Section" in out
    assert "More prose." in out


def test_block_kept_when_any_line_matches():
    """If at least one quote line in the block matches the source pool,
    the entire block is kept -- including its attribution line."""
    source = ["it costs too much money for what you get"]
    markdown = (
        "## Section\n"
        "\n"
        "> \"it costs too much money for what you get\"\n"
        "> -- Customer Reviewer on G2\n"
        "\n"
        "More prose.\n"
    )
    out, removed = _remove_unmatched_quote_lines(markdown, source)
    assert removed == 0
    assert "costs too much money" in out
    assert "Customer Reviewer on G2" in out


def test_block_with_attribution_first_then_quote():
    """Some renderers emit attribution before the quote inside a single
    block. The block-level matcher should find the quote regardless of
    position within the block and keep the entire block."""
    source = ["it costs too much money for what you get"]
    markdown = (
        "## Section\n"
        "\n"
        "> -- Customer Reviewer on G2\n"
        "> \"it costs too much money for what you get\"\n"
        "\n"
        "More prose.\n"
    )
    out, removed = _remove_unmatched_quote_lines(markdown, source)
    assert removed == 0
    assert "Customer Reviewer on G2" in out
    assert "costs too much money" in out


def test_block_with_mixed_grounded_and_ungrounded_quotes_strips_block():
    """The block is the unit of trust. If ANY quote-bearing line is
    ungrounded, the entire block is stripped -- even if other lines
    in the same block are grounded.

    This catches the case Codex flagged on PR #625: a grounded quote
    followed by an LLM-added fabricated sentence in the same `>` run
    should NOT ship either line. The contract is "no ungrounded quote
    text ever ships", and per-line stripping inside a block would
    reintroduce orphan-attribution bugs anyway.
    """
    source = ["it costs too much money for what you get"]
    markdown = (
        "## Section\n"
        "\n"
        "> \"it costs too much money for what you get\"\n"
        "> \"and the support team never responded to my emails\"\n"
        "> -- Customer Reviewer on G2\n"
        "\n"
        "More prose.\n"
    )
    # The first quote matches the source pool; the second does not.
    # Even though one line is grounded, the whole block ships nothing.
    out, removed = _remove_unmatched_quote_lines(markdown, source)
    assert removed == 3  # all three lines of the block
    assert "costs too much money" not in out
    assert "support team never responded" not in out
    assert "Customer Reviewer on G2" not in out
    # Surrounding prose preserved.
    assert "## Section" in out
    assert "More prose." in out


def test_multiple_blocks_independent_decisions():
    """Two separate blockquote blocks separated by prose. One should be
    kept (matched), one should be stripped (unmatched). The stripper
    treats them independently."""
    source = ["a real quote that matches"]
    markdown = (
        "## Section\n"
        "\n"
        "> \"a real quote that matches\"\n"
        "> -- Real Reviewer on G2\n"
        "\n"
        "Some prose between blocks.\n"
        "\n"
        "> \"a fabricated quote that does not match\"\n"
        "> -- Fabricated Person on Nowhere\n"
        "\n"
        "Final prose.\n"
    )
    out, removed = _remove_unmatched_quote_lines(markdown, source)
    assert removed == 2  # only the second block's two lines
    # First block kept entirely
    assert "real quote that matches" in out
    assert "Real Reviewer on G2" in out
    # Second block fully stripped
    assert "fabricated quote that does not match" not in out
    assert "Fabricated Person on Nowhere" not in out
    # Prose preserved
    assert "Some prose between blocks." in out
    assert "Final prose." in out


def test_orphan_intro_and_disclaimer_prose_stripped_with_block():
    """When a blockquote is stripped, the stripper also sweeps the
    orphan introductory paragraph immediately before it AND any
    acknowledged-misattribution disclaimer paragraph immediately
    after it. Both reference content that no longer exists, so leaving
    them in the body produces dangling prose.

    The disclaimer detection mirrors the seo-geo-aeo-blog-post skill's
    v1.4.0 audit patterns -- the audit catches these post-publish, and
    the generator now catches them upstream.

    Previously this test documented the gap (intro and disclaimer
    survived stripping). The orphan-prose extension to the stripper
    closes the gap; the assertions are inverted accordingly."""
    source: list[str] = []  # fail-closed: all blockquotes stripped
    markdown = (
        "## Section\n"
        "\n"
        "One reviewer on Reddit noted:\n"
        "\n"
        "> a quote that gets stripped because source pool is empty\n"
        "\n"
        "That quote is from a compensation discussion, not a CRM review, but it surfaced in the same complaint pattern analysis.\n"
    )
    out, removed = _remove_unmatched_quote_lines(markdown, source)
    # Block (1 line) + 2 blanks + intro line + disclaimer line = 5 lines.
    assert removed == 5
    assert "gets stripped" not in out
    assert "One reviewer on Reddit noted:" not in out
    assert "That quote is from a compensation discussion" not in out
    # Surrounding content (the H2) is preserved.
    assert "## Section" in out


def test_orphan_intro_alone_stripped_when_no_disclaimer():
    """Intro-only case: block has an intro paragraph before it but no
    disclaimer after. Only the intro is swept along with the block."""
    source: list[str] = []
    markdown = (
        "## Section\n"
        "\n"
        "A reviewer on G2 said:\n"
        "\n"
        "> some fabricated quote\n"
        "\n"
        "Body prose continues here.\n"
    )
    out, removed = _remove_unmatched_quote_lines(markdown, source)
    assert "some fabricated quote" not in out
    assert "A reviewer on G2 said:" not in out
    assert "Body prose continues here." in out


def test_orphan_disclaimer_alone_stripped_when_no_intro():
    """Disclaimer-only case: block has a disclaimer after but no
    introductory ':' paragraph before. Only the disclaimer is swept
    along with the block."""
    source: list[str] = []
    markdown = (
        "## Section\n"
        "\n"
        "Some normal prose without a colon ending.\n"
        "\n"
        "> a quote\n"
        "\n"
        "That quote is from a payment platform discussion, not a CRM review.\n"
    )
    out, removed = _remove_unmatched_quote_lines(markdown, source)
    # The blockquote line `> a quote` is gone (assertion was previously
    # `"a quote" not in out OR ">" not in out` which would pass even when
    # the blockquote remained as long as "a quote" appeared elsewhere --
    # tightened per Copilot review on PR #638).
    assert "> a quote" not in out
    assert "That quote is from a payment platform discussion" not in out
    # The 'normal prose' line doesn't end with ':' so it's preserved.
    assert "Some normal prose without a colon ending." in out


def test_orphan_disclaimer_offtopic_keyword_match_variants_stripped():
    """Off-topic keyword-match disclaimers (corpus-audit additions) are
    swept along with their stripped blockquote, the same as the original
    acknowledged-misattribution patterns. These wordings shipped in
    production posts (copper-the-metal, audio gear, an ISP complaint, a
    financial-planning note) before the pattern set was widened."""
    variants = [
        "While the quote references internet service providers rather than CRM software, the sentiment mirrors hidden-fee complaints.",
        "This quote does not directly reference Copper CRM, but the erosion of trust is similar.",
        "This appears to reference audio equipment rather than the software product, indicating data noise in the corpus.",
        "Two complaints came from community platforms but lacked CRM-specific context.",
        'Not every mention of "Copper" and "pricing" refers to the software product.',
    ]
    for disclaimer in variants:
        markdown = (
            "## Section\n\n"
            "> an off-topic quote that does not ground\n\n"
            f"{disclaimer}\n"
        )
        out, _removed = _remove_unmatched_quote_lines(markdown, [])
        assert "off-topic quote" not in out, disclaimer
        assert disclaimer not in out, f"disclaimer survived: {disclaimer}"
        assert "## Section" in out


def test_offtopic_pattern_does_not_sweep_nondisclaimer_following_prose():
    """False-positive guard: prose adjacent to a STRIPPED block that merely
    contains 'rather than' (without the 'reference ... CRM/software' shape)
    is NOT treated as a disclaimer. Only disclaimer-shaped wording is swept;
    legitimate analysis survives."""
    markdown = (
        "## Section\n\n"
        "> ungrounded quote\n\n"
        "Buyers value simplicity rather than feature depth, a recurring theme.\n"
    )
    out, _removed = _remove_unmatched_quote_lines(markdown, [])
    assert "ungrounded quote" not in out  # block stripped (fail-closed)
    # Not disclaimer-shaped -> preserved.
    assert "Buyers value simplicity rather than feature depth" in out


def test_orphan_quote_reference_swept_with_block():
    """A stripped blockquote followed by a paragraph that back-references the
    quote ("This quote ...", "The excerpt ...", "quoted earlier") is swept
    with the block -- the reference dangles once the quote is gone."""
    for follow in (
        "This quote shows an active evaluation in progress.",
        "The excerpt cuts off, but the signal is clear: teams compare costs.",
        "The Reddit reviewer quoted earlier described it as the worst so far.",
        # "the witness" prefix must NOT suppress a genuine orphan ref -- the
        # guard only excludes the aggregate-noun forms (Codex review on #728).
        "The witness quoted earlier described the rollout as a disaster.",
    ):
        markdown = (
            "## Section\n\n"
            "> a quote that gets stripped (empty source pool)\n\n"
            f"{follow}\n"
        )
        out, _removed = _remove_unmatched_quote_lines(markdown, [])
        assert "gets stripped" not in out
        assert follow not in out, follow
        assert "## Section" in out


def test_quote_reference_after_kept_block_survives():
    """A quote-reference-shaped follow-on after a KEPT (grounded) block must
    survive -- the quote it references is present, so the reference is
    legitimate (the #723 close-vs-zoho 176/205 / slack 67/110 case). The line
    DOES match the matcher, so the only thing protecting it is the sweep
    firing on STRIPPED blocks alone; this pins that protection."""
    follow = "This quote reflects the evaluation fatigue common in a high-churn category."
    assert _looks_like_orphan_quote_reference(follow)  # matcher would flag it...
    source = ["evaluation fatigue is common in this high-churn category"]
    markdown = (
        "## Section\n\n"
        "> evaluation fatigue is common in this high-churn category\n\n"
        f"{follow}\n"
    )
    out, removed = _remove_unmatched_quote_lines(markdown, source)
    assert removed == 0                       # block grounds -> nothing stripped
    assert "evaluation fatigue is common" in out  # the quote is kept
    assert follow in out                      # ...but the reference legitimately survives


def test_quote_reference_does_not_sweep_generic_or_witness_followon():
    """False-positive guard: a generic follow-on or an aggregate
    "The witness evidence ..." reference after a stripped block is preserved
    -- neither is an orphaned quote back-reference."""
    for follow in (
        "This pattern recurs across the dataset.",
        "The witness evidence shows workflow migration at the team level.",
    ):
        markdown = (
            "## Section\n\n"
            "> an ungrounded quote\n\n"
            f"{follow}\n"
        )
        out, _removed = _remove_unmatched_quote_lines(markdown, [])
        assert "ungrounded quote" not in out  # block still stripped
        assert follow in out, follow          # follow-on preserved


def test_orphan_question_reference_matches_question_shapes():
    """#745 hardening: back-references to a quoted *question* (G2 form
    prompts ARE questions) must be detected, not just "quote"/"excerpt"
    shapes. The detector previously missed "This open-ended question ..."
    and "This question format ..." follow-ons left behind when a
    form-prompt blockquote was stripped."""
    # Question-shaped back-references -> orphaned.
    assert _looks_like_orphan_quote_reference(
        "This open-ended question from a verified review described the rollout as confusing."
    )
    assert _looks_like_orphan_quote_reference(
        "This question format suggests a structured review prompt rather than a candid opinion."
    )
    # "This quote is a question rather than a statement ..." already matched
    # on the leading "This quote", but pin it so the merged pattern keeps it.
    assert _looks_like_orphan_quote_reference(
        "This quote is a question rather than a statement, but it signals active evaluation."
    )
    # The article x noun cross-product also matches these (same regex branch as
    # the covered shapes): "that excerpt" (newly added vs the old
    # this/the-excerpt-only pattern), and the article-led / no-hyphen
    # open-ended-question forms. Pin them so the merged pattern keeps them.
    for line in (
        "That excerpt captures the migration frustration teams describe.",
        "The open-ended question hints at an evaluation still in progress.",
        "This open ended question reads as a dangling reference once the quote is gone.",
    ):
        assert _looks_like_orphan_quote_reference(line), line
    # Existing negatives still hold: generic follow-ons and aggregate
    # witness references are NOT orphan quote back-references.
    assert not _looks_like_orphan_quote_reference("This pattern recurs across the dataset.")
    assert not _looks_like_orphan_quote_reference(
        "The witness evidence shows workflow migration at the team level."
    )
    # Rhetorical "The question is/isn't ..." author prose references NO quote
    # and must NOT be flagged. A bare-"question" pattern false-positived on
    # these against the live corpus (help-scout-vs-zendesk, real-cost-of-
    # woocommerce); the pattern is narrowed to quoted-question artifacts only.
    assert not _looks_like_orphan_quote_reference(
        "The question is not which vendor is objectively better."
    )
    assert not _looks_like_orphan_quote_reference(
        "The question isn't whether WooCommerce is expensive in absolute terms."
    )


def test_orphan_question_reference_swept_with_block():
    """A stripped blockquote followed by a question-shaped back-reference
    is swept along with the block, the same as a "This quote ..." ref. These
    are the #745 power-bi ("a question rather than a statement") and
    switch-to-woocommerce ("This open-ended question ...") danglers."""
    for follow in (
        "This open-ended question from a verified review hints at active evaluation.",
        "This question format suggests a structured review prompt, not a candid take.",
    ):
        markdown = (
            "## Section\n\n"
            "> a form-prompt quote that gets stripped (empty source pool)\n\n"
            f"{follow}\n"
        )
        out, _removed = _remove_unmatched_quote_lines(markdown, [])
        assert "gets stripped" not in out
        assert follow not in out, follow
        assert "## Section" in out


def test_question_reference_after_kept_block_survives():
    """Parity with the "This quote ..." guard: a question-shaped follow-on
    after a KEPT (grounded) block survives -- the quote it references is
    present, so the reference is legitimate. The sweep only fires on
    stripped blocks, so the kept block protects it."""
    follow = "This open-ended question mirrors the evaluation fatigue common in the category."
    assert _looks_like_orphan_quote_reference(follow)  # matcher would flag it...
    source = ["evaluation fatigue is common in this high-churn category"]
    markdown = (
        "## Section\n\n"
        "> evaluation fatigue is common in this high-churn category\n\n"
        f"{follow}\n"
    )
    out, removed = _remove_unmatched_quote_lines(markdown, source)
    assert removed == 0                            # block grounds -> nothing stripped
    assert "evaluation fatigue is common" in out   # the quote is kept
    assert follow in out                           # ...and the reference survives


def test_orphan_prose_preserved_for_kept_blocks():
    """When a block is KEPT (its quotes ground), the surrounding intro
    and prose are also preserved. The orphan-prose cleanup only fires
    on stripped blocks."""
    source = ["it costs too much money for what you get"]
    markdown = (
        "## Section\n"
        "\n"
        "A verified reviewer noted:\n"
        "\n"
        "> \"it costs too much money for what you get\"\n"
        "> -- Customer Reviewer on G2\n"
        "\n"
        "Body prose continues here.\n"
    )
    out, removed = _remove_unmatched_quote_lines(markdown, source)
    assert removed == 0
    assert "A verified reviewer noted:" in out
    assert "costs too much money" in out
    assert "Customer Reviewer on G2" in out
    assert "Body prose continues here." in out


def test_orphan_intro_not_stripped_when_too_long():
    """An 'intro paragraph' must be reasonably short. A long paragraph
    that happens to end with ':' is NOT swept along with a stripped
    block -- it's substantive content, not a lead-in."""
    source: list[str] = []
    long_intro = (
        "This is a long paragraph that contains substantive analysis "
        "and discussion of the methodology, the data sources, the "
        "limitations of self-selected reviewer feedback, and the way "
        "patterns emerge from cross-vendor analysis across the corpus:"
    )
    markdown = (
        "## Section\n"
        "\n"
        f"{long_intro}\n"
        "\n"
        "> some fabricated quote\n"
        "\n"
        "Body prose continues here.\n"
    )
    out, removed = _remove_unmatched_quote_lines(markdown, source)
    # The block (1 line) is stripped; the long intro is preserved.
    assert "some fabricated quote" not in out
    assert long_intro in out


def test_adjacent_blocks_disclaimer_shaped_second_block_not_absorbed():
    """Codex P1 on PR #638: with two adjacent blockquotes, if the
    second block starts with text matching the disclaimer regex
    (e.g. `> "That quote is from a different review..."`), the first
    stripped block's forward span must NOT widen into the second
    block. Otherwise a grounded second block can be silently deleted.

    The fix lives in ``_looks_like_orphan_disclaimer``: it now refuses
    to classify blockquote-, heading-, or list-prefixed lines as
    orphan disclaimer prose, so the forward-span widening stops at
    structural boundaries.
    """
    source = ["that quote is from a real source about pricing pressure"]
    markdown = (
        "## Section\n"
        "\n"
        "> a fabricated quote\n"
        "\n"
        "> that quote is from a real source about pricing pressure\n"
        "\n"
        "Final prose.\n"
    )
    out, removed = _remove_unmatched_quote_lines(markdown, source)
    # First block (fabricated) -- stripped.
    assert "a fabricated quote" not in out
    # Second block matches the source pool AND happens to start with
    # disclaimer-shaped text -- must be preserved, NOT absorbed into
    # the first block's strip span.
    assert "that quote is from a real source about pricing pressure" in out
    # Final prose unaffected.
    assert "Final prose." in out


# ---------------------------------------------------------------------------
# Vendor-matched affiliate selection
#
# Tests for _pick_affiliate_partner_for_vendors -- the pure matcher that
# replaces the prior category-only fallback. The matcher returns the
# first enabled partner whose product_name or alias matches a vendor
# in the post's vendor set (bidirectional whole-word match).
# ---------------------------------------------------------------------------


def _partner(
    name: str,
    product_name: str,
    *,
    aliases: list[str] | None = None,
    category: str = "CRM",
    enabled: bool = True,
    affiliate_url: str | None = None,
) -> dict[str, Any]:
    # Note: do NOT use example.com here -- _is_placeholder_partner flags
    # that domain as a placeholder signal. Use the product slug as a
    # plausible-looking real referrer URL.
    slug = product_name.lower().replace(' ', '-').replace('.', '-')
    return {
        "id": f"00000000-0000-0000-0000-{abs(hash(name)) % 10**12:012x}",
        "name": name,
        "product_name": product_name,
        "product_aliases": aliases or [],
        "affiliate_url": affiliate_url or f"https://{slug}.test-affiliates.io/ref=atlas",
        "category": category,
        "enabled": enabled,
    }


def test_vendor_matcher_returns_none_for_empty_vendor_list():
    """Category-level posts (landscapes, roundups) have no specific
    vendor. They should not get an affiliate injected."""
    partners = [_partner("HubSpot Partner", "HubSpot")]
    assert _pick_affiliate_partner_for_vendors(partners, []) is None


def test_vendor_matcher_returns_none_for_empty_partners():
    """No partners registered -> no affiliate, regardless of vendors."""
    assert _pick_affiliate_partner_for_vendors([], ["HubSpot"]) is None


def test_vendor_matcher_exact_product_name_match():
    partners = [_partner("HubSpot Partner", "HubSpot")]
    out = _pick_affiliate_partner_for_vendors(partners, ["HubSpot"])
    assert out is not None
    assert out["name"] == "HubSpot Partner"


def test_vendor_matcher_case_insensitive():
    """Vendor name 'hubspot' should match partner 'HubSpot'."""
    partners = [_partner("HubSpot Partner", "HubSpot")]
    out = _pick_affiliate_partner_for_vendors(partners, ["hubspot"])
    assert out is not None
    assert out["name"] == "HubSpot Partner"


def test_vendor_matcher_alias_match():
    """Vendor matches an alias rather than the canonical product_name."""
    partners = [_partner(
        "Monday.com",
        "Monday.com",
        aliases=["monday", "monday CRM", "monday work OS"],
    )]
    out = _pick_affiliate_partner_for_vendors(partners, ["monday CRM"])
    assert out is not None
    assert out["product_name"] == "Monday.com"


def test_vendor_matcher_punctuation_in_product_name():
    """Punctuation like the dot in 'Monday.com' should match vendor
    'Monday.com' literally without regex issues."""
    partners = [_partner("Monday.com", "Monday.com")]
    out = _pick_affiliate_partner_for_vendors(partners, ["Monday.com"])
    assert out is not None
    assert out["product_name"] == "Monday.com"


def test_vendor_matcher_bidirectional_alias_longer_than_vendor():
    """Vendor 'monday' should match alias 'monday CRM' because the vendor
    name appears as a whole word in the alias (bidirectional matching).
    The prior simple substring check would fail here."""
    partners = [_partner(
        "Monday.com",
        "Monday.com",
        aliases=["monday CRM"],
    )]
    out = _pick_affiliate_partner_for_vendors(partners, ["monday"])
    assert out is not None
    assert out["product_name"] == "Monday.com"


def test_vendor_matcher_rejects_partial_word():
    """Partner 'Up' should NOT match vendor 'PipeUp' -- word boundaries
    prevent the partial-word false positive."""
    partners = [_partner("Up Partner", "Up", category="Misc")]
    out = _pick_affiliate_partner_for_vendors(partners, ["PipeUp"])
    assert out is None


def test_vendor_matcher_no_match_when_partner_in_different_category():
    """The HubSpot-on-CRM-roundup case generalised: when the post's
    vendor set does not include HubSpot, the HubSpot partner does NOT
    get injected even though both might share the 'CRM' category."""
    partners = [_partner("HubSpot Partner", "HubSpot")]
    # Post analyzes Salesforce, Zoho, Pipedrive (no HubSpot).
    vendor_names = ["Salesforce", "Zoho CRM", "Pipedrive"]
    out = _pick_affiliate_partner_for_vendors(partners, vendor_names)
    assert out is None


def test_vendor_matcher_first_vendor_wins_priority():
    """When multiple vendors could match different partners, the FIRST
    vendor in the list wins. Mirrors how callers stack
    (vendor, vendor_a, vendor_b, from_vendor)."""
    partners = [
        _partner("HubSpot Partner", "HubSpot"),
        _partner("Pipedrive Partner", "Pipedrive"),
    ]
    # vendor (primary) = Pipedrive, vendor_a (secondary) = HubSpot.
    out = _pick_affiliate_partner_for_vendors(partners, ["Pipedrive", "HubSpot"])
    assert out is not None
    assert out["product_name"] == "Pipedrive"


def test_vendor_matcher_skips_empty_or_whitespace_vendors():
    """Empty / whitespace strings in the vendor list are ignored, not
    treated as match candidates."""
    partners = [_partner("HubSpot Partner", "HubSpot")]
    out = _pick_affiliate_partner_for_vendors(partners, ["", "  ", "HubSpot"])
    assert out is not None
    assert out["product_name"] == "HubSpot"


def test_vendor_matcher_returns_first_match_within_partner_order():
    """When one vendor matches multiple partners (rare), partners are
    tried in supplied order. Caller is responsible for ordering by
    insertion time / priority."""
    partners = [
        _partner("First Partner", "Salesforce"),
        _partner("Second Partner", "Salesforce", aliases=["sfdc"]),
    ]
    out = _pick_affiliate_partner_for_vendors(partners, ["Salesforce"])
    assert out is not None
    assert out["name"] == "First Partner"


def test_vendor_matcher_handles_none_aliases():
    """Partner row with product_aliases=None (rather than []) should not
    crash. asyncpg may return None for an empty TEXT[] column in some
    drivers."""
    partner = _partner("HubSpot Partner", "HubSpot")
    partner["product_aliases"] = None
    out = _pick_affiliate_partner_for_vendors([partner], ["HubSpot"])
    assert out is not None


# ---------------------------------------------------------------------------
# Placeholder-partner defense-in-depth
#
# Triggered by the "Atlas Live Test Partner" incident: a test row with
# affiliate_url=https://example.com/atlas-live-test-partner lived in the
# live affiliate_partners table for ~8 weeks and re-injected itself into
# every category='B2B Software' post until manual cleanup. The DB row was
# deleted, but the matcher should also refuse such rows defensively.
# ---------------------------------------------------------------------------


def test_is_placeholder_partner_flags_example_com_url():
    """The exact pattern from the production incident."""
    partner = _partner(
        "Atlas Live Test Partner",
        "Atlas B2B Software Partner",
    )
    partner["affiliate_url"] = "https://example.com/atlas-live-test-partner"
    assert _is_placeholder_partner(partner) is True


def test_is_placeholder_partner_flags_test_in_name():
    partner = _partner("Test Vendor", "Real Product")
    assert _is_placeholder_partner(partner) is True


def test_is_placeholder_partner_flags_test_in_product_name():
    partner = _partner("Real Vendor", "Test Product")
    assert _is_placeholder_partner(partner) is True


def test_is_placeholder_partner_flags_placeholder_in_name():
    partner = _partner("Placeholder Partner", "RealProduct")
    assert _is_placeholder_partner(partner) is True


def test_is_placeholder_partner_flags_test_partner_in_url():
    partner = _partner("Some Name", "SomeProduct")
    partner["affiliate_url"] = "https://realdomain.com/test-partner-ref"
    assert _is_placeholder_partner(partner) is True


def test_is_placeholder_partner_does_not_flag_real_data():
    """Real partner rows should pass through cleanly."""
    real_partners = [
        _partner("HubSpot Partner", "HubSpot"),
        _partner("Pipedrive Partner", "Pipedrive"),
        _partner("Monday.com", "Monday.com", aliases=["monday CRM"]),
        _partner("Amazon Associates", "Amazon"),
        _partner("Shopify Affiliates", "Shopify"),
        _partner("HelpDesk", "HelpDesk"),
    ]
    for p in real_partners:
        assert _is_placeholder_partner(p) is False, (
            f"Real partner {p['name']!r} should not be flagged as placeholder"
        )


def test_is_placeholder_partner_handles_attest_substring():
    """`\\btest\\b` is whole-word, so partner names like 'Attestation
    Service' should NOT flag (substring 'test' inside 'Attestation')."""
    partner = _partner("Attestation Service", "Attest")
    assert _is_placeholder_partner(partner) is False


def test_vendor_matcher_drops_placeholder_partner_even_if_vendor_matches():
    """If a placeholder row sneaks into the DB and somehow shares a vendor
    name with the post, the matcher still refuses to inject it. This is
    the integration test for the defensive filter inside
    _pick_affiliate_partner_for_vendors.
    """
    placeholder = _partner(
        "Atlas Live Test Partner",
        "Atlas B2B Software Partner",
    )
    placeholder["affiliate_url"] = "https://example.com/atlas-live-test-partner"
    # Pretend the post's vendor matches the placeholder's product_name.
    out = _pick_affiliate_partner_for_vendors(
        [placeholder],
        ["Atlas B2B Software Partner"],
    )
    assert out is None


def test_vendor_matcher_prefers_real_partner_over_placeholder():
    """When both a placeholder AND a real partner could match, the
    placeholder is filtered out and the real partner is returned."""
    placeholder = _partner("Test HubSpot", "HubSpot")
    real = _partner("HubSpot Partner", "HubSpot")
    out = _pick_affiliate_partner_for_vendors([placeholder, real], ["HubSpot"])
    assert out is not None
    assert out["name"] == "HubSpot Partner"


# ---------------------------------------------------------------------------
# _blueprint_pricing_reality_check pilot integration
# ---------------------------------------------------------------------------


def _pricing_ctx(**overrides: Any) -> dict[str, Any]:
    ctx = {
        "slug": "shopify-pricing",
        "vendor": "Shopify",
        "category": "Ecommerce",
        "pricing_complaints": 12,
        "total_reviews": 100,
        "avg_urgency": 7.2,
    }
    ctx.update(overrides)
    return ctx


def test_pricing_blueprint_uses_quote_grade_sql_rows():
    row = _v4_row(
        text="Shopify keeps raising the price without adding enough value",
        review_id="review-1",
        source="capterra",
        vendor_name="Shopify",
        urgency=8.5,
    )
    blueprint = _blueprint_pricing_reality_check(
        _pricing_ctx(),
        {"pricing_reviews": [row]},
    )
    assert len(blueprint.quotable_phrases) == 1
    quote = blueprint.quotable_phrases[0]
    assert quote["phrase"] == "Shopify keeps raising the price without adding enough value"
    assert quote["review_id"] == "review-1"
    assert quote["source"] == "capterra"
    assert quote["field"] == "pricing_phrases"


def test_pricing_blueprint_drops_v3_sql_rows_from_quote_pool():
    legacy_sql_row = {
        "vendor_name": "Shopify",
        "vendor": "Shopify",
        "urgency": 7.0,
        "review_id": "legacy-review",
        "source": "g2",
        "text": "This text should not become a quote.",
        "enrichment_raw": json.dumps({
            "enrichment_schema_version": 3,
            "pricing_phrases": ["too expensive"],
        }),
    }
    blueprint = _blueprint_pricing_reality_check(
        _pricing_ctx(),
        {"pricing_reviews": [legacy_sql_row]},
    )
    assert blueprint.quotable_phrases == []


def test_pricing_blueprint_renders_verbatim_vault_rows():
    """Phase 2.3 4i: vault rows must carry phrase_verbatim=True (stamped
    by the vault writer post-4e-A and propagated through the blog
    vault mergers). The legacy truncation path then re-emits the row
    with the marker preserved, so downstream consumers can also
    enforce the policy."""
    vault_row = {
        "vendor": "Shopify",
        "urgency": 6.0,
        "role": "Finance",
        "text": "Vault-sourced quote text passes the verbatim gate.",
        "quote_origin": "vault",
        "phrase_verbatim": True,
    }
    blueprint = _blueprint_pricing_reality_check(
        _pricing_ctx(),
        {"pricing_reviews": [vault_row]},
    )
    assert blueprint.quotable_phrases == [{
        "phrase": "Vault-sourced quote text passes the verbatim gate.",
        "vendor": "Shopify",
        "urgency": 6.0,
        "role": "Finance",
        "phrase_verbatim": True,
    }]


def test_pricing_blueprint_drops_vault_row_missing_phrase_verbatim():
    """Phase 2.3 4i regression: a vault-origin row without the marker
    must drop, NOT render. Closes the policy gap where 'safe by
    construction' isn't the same as 'enforced'."""
    vault_row = {
        "vendor": "Shopify",
        "urgency": 6.0,
        "role": "Finance",
        "text": "Unmarked vault quote must not surface.",
        "quote_origin": "vault",
        # phrase_verbatim missing
    }
    blueprint = _blueprint_pricing_reality_check(
        _pricing_ctx(),
        {"pricing_reviews": [vault_row]},
    )
    assert blueprint.quotable_phrases == []


# ---------------------------------------------------------------------------
# _split_and_gate_blog_quotes wrapper -- discriminator policy
# ---------------------------------------------------------------------------


def test_wrapper_drops_unmarked_row():
    """No quote_origin -> dropped. Closes the 'no enrichment means
    vault' loophole."""
    row = {"vendor": "X", "text": "no marker"}
    assert _split_and_gate_blog_quotes([row]) == []


def test_wrapper_drops_unknown_origin():
    row = {"vendor": "X", "text": "junk", "quote_origin": "weather_forecast"}
    assert _split_and_gate_blog_quotes([row]) == []


def test_wrapper_routes_review_origin_through_contract_gate():
    row = _v4_row()  # already stamped quote_origin="review"
    out = _split_and_gate_blog_quotes([row], field="pricing_phrases")
    assert len(out) == 1
    assert out[0]["phrase"] == "it costs too much money"
    assert out[0]["review_id"] == "00000000-0000-0000-0000-000000000001"


def test_wrapper_drops_review_origin_with_missing_enrichment():
    """A row marked 'review' but lacking enrichment data still drops --
    the contract gate must produce a verbatim phrase or nothing."""
    row = {
        "vendor": "X",
        "review_id": "no-enrichment",
        "source": "g2",
        "quote_origin": "review",
        # NO enrichment / enrichment_raw
    }
    assert _split_and_gate_blog_quotes([row]) == []


def test_wrapper_routes_vault_origin_through_legacy_path():
    row = {
        "vendor": "Shopify",
        "urgency": 5.0,
        "role": "Director",
        "text": "Vault-curated text passes through legacy truncation.",
        "quote_origin": "vault",
        "phrase_verbatim": True,
    }
    out = _split_and_gate_blog_quotes([row])
    assert len(out) == 1
    assert out[0]["phrase"] == "Vault-curated text passes through legacy truncation."
    assert out[0]["vendor"] == "Shopify"
    assert out[0]["phrase_verbatim"] is True


def test_wrapper_drops_vault_origin_without_phrase_verbatim():
    """Phase 2.3 4i: vault rows are no longer accepted by origin alone.
    They must carry phrase_verbatim=True (stamped by the blog vault
    mergers post-4i). Unmarked vault rows fail closed."""
    row = {
        "vendor": "Shopify",
        "urgency": 5.0,
        "role": "Director",
        "text": "Vault row missing the marker must not surface.",
        "quote_origin": "vault",
    }
    assert _split_and_gate_blog_quotes([row]) == []


def test_wrapper_handles_vault_phrase_field_too():
    """Vault rows from _merge_blog_quotes_with_evidence_vault carry
    'phrase' rather than 'text' -- both shapes accepted."""
    row = {
        "phrase": "Curated phrase from vault.",
        "vendor": "Shopify",
        "urgency": 4.0,
        "role": "VP",
        "quote_origin": "vault",
        "phrase_verbatim": True,
    }
    out = _split_and_gate_blog_quotes([row])
    assert len(out) == 1
    assert out[0]["phrase"] == "Curated phrase from vault."


def test_wrapper_combines_review_and_vault_rows():
    review_row = _v4_row(text="Real verbatim review quote.")
    vault_row = {
        "phrase": "Curated vault phrase.",
        "vendor": "Shopify",
        "urgency": 5.0,
        "role": "PM",
        "quote_origin": "vault",
        "phrase_verbatim": True,
    }
    out = _split_and_gate_blog_quotes([review_row, vault_row])
    assert len(out) == 2
    phrases = [q["phrase"] for q in out]
    assert "Real verbatim review quote." in phrases
    assert "Curated vault phrase." in phrases


def test_wrapper_respects_limit():
    rows = [
        _v4_row(text=f"verbatim {i}", review_id=f"id-{i}")
        for i in range(10)
    ]
    out = _split_and_gate_blog_quotes(rows, limit=3)
    assert len(out) == 3


# ---------------------------------------------------------------------------
# _blog_quote_highlights reviewer_voice section stats
# ---------------------------------------------------------------------------


def test_blog_quote_highlights_drops_unmarked_rows():
    """Reviewer_voice key_stats are customer-visible and must use the
    same explicit-origin gate as PostBlueprint.quotable_phrases."""
    rows = [
        {"vendor": "Shopify", "phrase": "Unmarked quote must not surface."},
    ]
    assert _blog_quote_highlights(rows) == []


def test_blog_quote_highlights_routes_review_rows_through_contract_gate():
    row = _v4_row(
        text="Pricing keeps rising every renewal cycle",
        review_id="highlight-review-1",
        source="g2",
        vendor_name="Shopify",
    )
    highlights = _blog_quote_highlights([row], vendors=["Shopify"])
    assert highlights == [
        {
            "vendor": "Shopify",
            "phrase": "Pricing keeps rising every renewal cycle",
            "sentiment": "",
            "role": "Director of Operations",
        }
    ]


def test_blog_quote_highlights_drops_non_verbatim_review_rows():
    row = _v4_row(
        text="Pricing keeps rising every renewal cycle",
        verbatim=False,
        vendor_name="Shopify",
    )
    assert _blog_quote_highlights([row], vendors=["Shopify"]) == []


def test_blog_quote_highlights_preserves_verbatim_vault_rows():
    """Phase 2.3 4i: vault rows must carry phrase_verbatim=True after
    the blog vault mergers stamp it. Pre-4i fixtures used origin-only
    marking; the gate now requires both."""
    row = {
        "vendor": "Shopify",
        "phrase": "Vault-curated quote text.",
        "role": "Finance",
        "quote_origin": "vault",
        "phrase_verbatim": True,
    }
    highlights = _blog_quote_highlights([row], vendors=["Shopify"])
    assert highlights == [
        {
            "vendor": "Shopify",
            "phrase": "Vault-curated quote text.",
            "sentiment": "",
            "role": "Finance",
        }
    ]


def test_blog_quote_highlights_drops_vault_row_missing_phrase_verbatim():
    """Phase 2.3 4i regression: vault-origin rows missing the marker
    must not surface in customer-visible reviewer_voice section stats."""
    row = {
        "vendor": "Shopify",
        "phrase": "Vault row without marker must not surface.",
        "role": "Finance",
        "quote_origin": "vault",
    }
    assert _blog_quote_highlights([row], vendors=["Shopify"]) == []


# ---------------------------------------------------------------------------
# _blueprint_pricing_reality_check pilot integration (loophole closure)
# ---------------------------------------------------------------------------


def test_pricing_blueprint_drops_unmarked_rows_closing_loophole():
    """A row missing quote_origin must be dropped, NOT preserved as
    'vault by absence of enrichment'. This closes the discriminator
    loophole the policy correction was designed to fix."""
    unmarked_sql_row = {
        "vendor": "Shopify",
        "urgency": 8.0,
        "review_id": "looks-like-sql",
        "source": "g2",
        "text": "Real review text but no origin marker.",
        "enrichment_raw": json.dumps({
            "enrichment_schema_version": 4,
            "pricing_phrases": ["should not surface"],
            "phrase_metadata": [{
                "field": "pricing_phrases", "index": 0,
                "text": "should not surface",
                "subject": "subject_vendor", "polarity": "negative",
                "role": "primary_driver", "verbatim": True,
            }],
        }),
    }
    blueprint = _blueprint_pricing_reality_check(
        _pricing_ctx(),
        {"pricing_reviews": [unmarked_sql_row]},
    )
    assert blueprint.quotable_phrases == []


# ---------------------------------------------------------------------------
# Phase 2.3 Commit B2 -- category-scoped blueprint integration tests
#
# Each test verifies the four traceability fields the contract gate
# stamps onto its output: review_id, source, field, and the exact
# verbatim phrase text. This locks in that the wrapper is wired at the
# PostBlueprint.quotable_phrases assignment site for every category-
# scoped producer.
# ---------------------------------------------------------------------------


def test_vendor_showdown_blueprint_routes_quotes_through_contract_gate():
    row = _v4_row(
        text="Shopify charges way too much for what you actually get",
        review_id="vs-review-1",
        source="g2",
        vendor_name="Shopify",
        urgency=8.4,
        field="pricing_phrases",
    )
    ctx = {
        "slug": "shopify-vs-bigcommerce",
        "vendor_a": "Shopify",
        "vendor_b": "BigCommerce",
        "category": "ecommerce",
        "total_reviews": 240,
        "reviews_a": 130,
        "reviews_b": 110,
        "urgency_a": 7.4,
        "urgency_b": 6.8,
        "pain_diff": 0.6,
    }
    data = {"data_context": {"category": "ecommerce"}, "quotes": [row]}
    blueprint = _blueprint_vendor_showdown(ctx, data)
    assert len(blueprint.quotable_phrases) == 1
    quote = blueprint.quotable_phrases[0]
    assert quote["phrase"] == "Shopify charges way too much for what you actually get"
    assert quote["review_id"] == "vs-review-1"
    assert quote["source"] == "g2"
    assert quote["field"] == "pricing_phrases"


def test_vendor_showdown_reviewer_voice_drops_unmarked_quote_highlights():
    ctx = {
        "slug": "shopify-vs-bigcommerce",
        "vendor_a": "Shopify",
        "vendor_b": "BigCommerce",
        "category": "ecommerce",
        "total_reviews": 240,
        "reviews_a": 130,
        "reviews_b": 110,
        "urgency_a": 7.4,
        "urgency_b": 6.8,
        "pain_diff": 0.6,
    }
    data = {
        "data_context": {"category": "ecommerce"},
        "quotes": [
            {
                "vendor": "Shopify",
                "phrase": "Unmarked section-stat text must not surface.",
            },
        ],
    }
    blueprint = _blueprint_vendor_showdown(ctx, data)
    assert blueprint.quotable_phrases == []
    assert "reviewer_voice" not in {section.id for section in blueprint.sections}


def test_vendor_showdown_reviewer_voice_uses_quote_grade_highlights():
    row = _v4_row(
        text="Shopify pricing keeps rising every renewal cycle",
        review_id="vs-review-voice-1",
        source="g2",
        vendor_name="Shopify",
        urgency=8.4,
        field="pricing_phrases",
    )
    ctx = {
        "slug": "shopify-vs-bigcommerce",
        "vendor_a": "Shopify",
        "vendor_b": "BigCommerce",
        "category": "ecommerce",
        "total_reviews": 240,
        "reviews_a": 130,
        "reviews_b": 110,
        "urgency_a": 7.4,
        "urgency_b": 6.8,
        "pain_diff": 0.6,
    }
    data = {"data_context": {"category": "ecommerce"}, "quotes": [row]}
    blueprint = _blueprint_vendor_showdown(ctx, data)
    reviewer_voice = next(section for section in blueprint.sections if section.id == "reviewer_voice")
    highlights = reviewer_voice.key_stats["quote_highlights"]
    assert highlights == [
        {
            "vendor": "Shopify",
            "phrase": "Shopify pricing keeps rising every renewal cycle",
            "sentiment": "",
            "role": "Director of Operations",
        }
    ]


def test_market_landscape_blueprint_routes_quotes_through_contract_gate():
    row = _v4_row(
        text="HubSpot's pricing tiers force you into upgrades you don't need",
        review_id="ml-review-1",
        source="capterra",
        vendor_name="HubSpot",
        urgency=7.9,
        field="pricing_phrases",
    )
    ctx = {
        "slug": "crm-landscape-2026",
        "category": "crm",
        "vendor_count": 8,
        "total_reviews": 1200,
        "avg_urgency": 6.5,
    }
    data = {"data_context": {"category": "crm"}, "quotes": [row]}
    blueprint = _blueprint_market_landscape(ctx, data)
    assert len(blueprint.quotable_phrases) == 1
    quote = blueprint.quotable_phrases[0]
    assert quote["phrase"] == "HubSpot's pricing tiers force you into upgrades you don't need"
    assert quote["review_id"] == "ml-review-1"
    assert quote["source"] == "capterra"
    assert quote["field"] == "pricing_phrases"


def test_market_landscape_headline_vendor_count_reflects_rendered_chart():
    """D8: the headline "{N} major vendors" must reflect the vendors actually
    rendered in the urgency chart (one bar per vendor with signals), NOT the
    independent category-wide ctx["vendor_count"]. The published crm-landscape
    claimed "8 vendors" while the chart showed 7. Here ctx says 8 but only 7
    vendor_signals carry signals -> the hook section must read 7. Reverting to
    ctx["vendor_count"] fails this."""
    data = {
        "data_context": {"category": "CRM"},
        "quotes": [],
        "vendor_signals": [
            {"vendor": f"Vendor {i}",
             "signals": [{"pain_category": "pricing", "avg_urgency": 5.0}]}
            for i in range(7)
        ],
    }
    ctx = {
        "slug": "crm-landscape-2026",
        "category": "CRM",
        "vendor_count": 8,  # independent category-wide count (the overclaiming source)
        "total_reviews": 1200,
        "avg_urgency": 6.5,
    }
    blueprint = _blueprint_market_landscape(ctx, data)
    hook = next(s for s in blueprint.sections if s.id == "hook")
    assert hook.key_stats["vendor_count"] == 7   # rendered, not ctx's 8
    assert "7 major vendors" in hook.data_summary
    assert "8 major vendors" not in hook.data_summary


def test_market_landscape_exposes_capped_profile_count():
    """D8b: the landscape profiles at most the leading vendors
    (vendor_profiles[:5]), not every charted vendor, so the hook surfaces a
    profile_count capped at 5. With 7 profiles available (all populated),
    profile_count is 5 (the cap). Reverting (drop the key, or use the uncapped
    len) fails this. (The empty-profile predicate is pinned separately by
    test_profile_count_counts_only_rendered_sections.)"""
    data = {
        "data_context": {"category": "CRM"},
        "quotes": [],
        "vendor_signals": [],
        "vendor_profiles": [
            {"vendor": f"V{i}", "profile": {"strengths": ["s"], "weaknesses": ["w"]}}
            for i in range(7)
        ],
    }
    ctx = {
        "slug": "crm-landscape-2026",
        "category": "CRM",
        "vendor_count": 7,
        "total_reviews": 1000,
        "avg_urgency": 5.0,
    }
    blueprint = _blueprint_market_landscape(ctx, data)
    hook = next(s for s in blueprint.sections if s.id == "hook")
    assert hook.key_stats["profile_count"] == 5   # capped, not the 7 available


def test_profile_count_counts_only_rendered_sections():
    """D8b / Codex P2 on #780: profile_count must count vendors that ACTUALLY
    emit a profile section (strengths OR weaknesses non-empty, the render-loop
    predicate), not bare len(vendor_profiles[:5]) -- otherwise an empty profile
    in the top 5 overstates coverage. Here B has an empty profile, so 5 entries
    -> 4 rendered. The count is also wired deterministically into data_summary."""
    data = {
        "data_context": {"category": "CRM"},
        "quotes": [],
        "vendor_signals": [],
        "vendor_profiles": [
            {"vendor": "A", "profile": {"strengths": ["s"], "weaknesses": ["w"]}},
            {"vendor": "B", "profile": {"strengths": [], "weaknesses": []}},  # emits no section
            {"vendor": "C", "profile": {"strengths": ["s"]}},
            {"vendor": "D", "profile": {"weaknesses": ["w"]}},
            {"vendor": "E", "profile": {"strengths": ["s"], "weaknesses": ["w"]}},
        ],
    }
    ctx = {
        "slug": "crm-landscape-2026",
        "category": "CRM",
        "vendor_count": 5,
        "total_reviews": 500,
        "avg_urgency": 5.0,
    }
    blueprint = _blueprint_market_landscape(ctx, data)
    hook = next(s for s in blueprint.sections if s.id == "hook")
    assert hook.key_stats["profile_count"] == 4   # B's empty profile is not counted
    assert "4 leading vendors" in hook.data_summary   # deterministically used, not just surfaced


def test_pain_point_roundup_blueprint_routes_quotes_through_contract_gate():
    row = _v4_row(
        text="Salesforce reporting takes forever to load and constantly times out",
        review_id="pp-review-1",
        source="trustradius",
        vendor_name="Salesforce",
        urgency=8.1,
        field="specific_complaints",
    )
    ctx = {
        "slug": "crm-complaints-2026",
        "category": "crm",
        "vendor_count": 6,
        "total_complaints": 480,
    }
    data = {"vendor_pains": [], "quotes": [row]}
    blueprint = _blueprint_pain_point_roundup(ctx, data)
    assert len(blueprint.quotable_phrases) == 1
    quote = blueprint.quotable_phrases[0]
    assert quote["phrase"] == "Salesforce reporting takes forever to load and constantly times out"
    assert quote["review_id"] == "pp-review-1"
    assert quote["source"] == "trustradius"
    assert quote["field"] == "specific_complaints"


def test_best_fit_guide_blueprint_routes_quotes_through_contract_gate():
    row = _v4_row(
        text="Asana feels overwhelming for a small team and most features go unused",
        review_id="bf-review-1",
        source="g2",
        vendor_name="Asana",
        urgency=6.7,
        field="specific_complaints",
    )
    ctx = {
        "slug": "best-pm-tools-smb",
        "category": "project management",
        "vendor_count": 5,
        "total_reviews": 800,
    }
    data = {"vendor_profiles": [], "vendor_signals": [], "quotes": [row]}
    blueprint = _blueprint_best_fit_guide(ctx, data)
    assert len(blueprint.quotable_phrases) == 1
    quote = blueprint.quotable_phrases[0]
    assert quote["phrase"] == "Asana feels overwhelming for a small team and most features go unused"
    assert quote["review_id"] == "bf-review-1"
    assert quote["source"] == "g2"
    assert quote["field"] == "specific_complaints"


# ---------------------------------------------------------------------------
# Evidence-anchor auto-injection disable
#
# Auto-injection of "Evidence anchor: ..." prose was disabled (see
# _apply_specificity_anchor_repair). The function is now cleanup-only:
# it removes any existing "Evidence anchor:" line from the body so a
# refreshed post doesn't carry over the legacy jargon, but it does NOT
# inject a new line on specificity failures. The specificity issues
# stay in the report so the post is held for human review.
#
# These tests cover the disable directly, independent of the broader
# deterministic-repairs flow that's tested in test_b2b_blog_post_generation.
# ---------------------------------------------------------------------------


def test_specificity_anchor_repair_does_not_inject_when_specificity_fails():
    """Even with ``witness_specificity`` entries in the report's
    ``blocking_issues``, the disabled repair does NOT inject an
    "Evidence anchor:" prose line into the body."""
    from atlas_brain.autonomous.tasks.b2b_blog_post_generation import PostBlueprint
    blueprint = PostBlueprint(
        topic_type="vendor_deep_dive",
        slug="x-deep-dive-2026-04",
        suggested_title="X Deep Dive",
        tags=["test"],
        data_context={"vendor": "X", "review_period": "2025-06 to 2026-03"},
        sections=[],
        charts=[],
    )
    content = {"content": "# X\n\nGeneric prose with no specific anchor.\n"}
    report = {
        "blocking_issues": ["witness_specificity:missing_timing_or_numeric"],
        "warnings": [],
        "fixes_applied": [],
    }
    repaired, repaired_report, did_repair = _apply_specificity_anchor_repair(
        blueprint, content, report,
    )
    assert "Evidence anchor:" not in repaired["content"]
    assert did_repair is False
    # The specificity issue remains in the report so the gate fails the post.
    assert any(
        issue.startswith("witness_specificity:")
        for issue in repaired_report["blocking_issues"]
    )


def test_specificity_anchor_repair_removes_existing_legacy_anchor():
    """When a post body has a legacy 'Evidence anchor:' line from a prior
    generation, the cleanup branch strips it so subsequent passes don't
    ship stale jargon."""
    from atlas_brain.autonomous.tasks.b2b_blog_post_generation import PostBlueprint
    blueprint = PostBlueprint(
        topic_type="vendor_deep_dive",
        slug="x-deep-dive-2026-04",
        suggested_title="X Deep Dive",
        tags=["test"],
        data_context={"vendor": "X", "review_period": "2025-06 to 2026-03"},
        sections=[],
        charts=[],
    )
    content = {
        "content": (
            "# X Deep Dive\n\n"
            "Evidence anchor: month-end is the live timing trigger, "
            "$200k is the concrete spend anchor.\n\n"
            "More prose continues.\n"
        ),
    }
    report = {
        "blocking_issues": [],
        "warnings": [],
        "fixes_applied": [],
    }
    repaired, repaired_report, did_repair = _apply_specificity_anchor_repair(
        blueprint, content, report,
    )
    assert "Evidence anchor:" not in repaired["content"]
    assert did_repair is True
    assert "removed_disabled_witness_anchor_note" in repaired_report["fixes_applied"]


def test_specificity_anchor_repair_no_op_when_no_anchor_and_no_issues():
    """No existing anchor + no specificity issues = nothing to do."""
    from atlas_brain.autonomous.tasks.b2b_blog_post_generation import PostBlueprint
    blueprint = PostBlueprint(
        topic_type="vendor_deep_dive",
        slug="x-deep-dive-2026-04",
        suggested_title="X Deep Dive",
        tags=["test"],
        data_context={"vendor": "X", "review_period": "2025-06 to 2026-03"},
        sections=[],
        charts=[],
    )
    content = {"content": "# X Deep Dive\n\nClean prose, no anchor line.\n"}
    report = {"blocking_issues": [], "warnings": [], "fixes_applied": []}
    repaired, repaired_report, did_repair = _apply_specificity_anchor_repair(
        blueprint, content, report,
    )
    assert did_repair is False
    assert repaired["content"] == content["content"]


def test_form_prompt_detection():
    # G2 review-form prompts are not genuine quotes.
    assert _is_form_prompt("What do you like best about Pipedrive?")
    assert _is_form_prompt("What do you dislike about Zoho CRM")
    assert _is_form_prompt("Recommendations to others considering HubSpot")
    # Real reviewer phrases survive.
    assert not _is_form_prompt("Pricing is too high for what you get")
    assert not _is_form_prompt("Support never responded for weeks")


def test_split_and_gate_drops_form_prompt_quotes():
    rows = [
        {"quote_origin": "vault", "phrase_verbatim": True,
         "phrase": "What do you like best about Zoho CRM"},
        {"quote_origin": "vault", "phrase_verbatim": True,
         "phrase": "Support never responded for weeks"},
    ]
    kept = [q["phrase"] for q in _split_and_gate_blog_quotes(rows, limit=10)]
    assert "Support never responded for weeks" in kept
    assert not any(_is_form_prompt(p) for p in kept)


def test_form_prompts_do_not_consume_limit_slots():
    """Codex P2 on #752: the form-prompt filter must run BEFORE the limit
    is enforced. If the first `limit` review candidates are form-prompt
    boilerplate, dropping them must not shrink the result below `limit`
    when enough genuine quotes exist further down the pool.

    Regression: _split_and_gate_blog_quotes used to pass `limit` into
    _quote_grade_blueprint_phrases, which early-returns after collecting
    `limit` candidates -- so leading form prompts ate the slots and the
    post-collection filter left fewer than `limit` quotes (here: zero).
    """
    rows = [
        _v4_row(text="What do you like best about Pipedrive?", review_id="fp-1"),
        _v4_row(text="What do you dislike about Pipedrive", review_id="fp-2"),
        _v4_row(text="Pricing climbs at every renewal", review_id="real-1"),
        _v4_row(text="Support never responded for weeks", review_id="real-2"),
        _v4_row(text="Onboarding dragged on for months", review_id="real-3"),
    ]
    out = _split_and_gate_blog_quotes(rows, limit=2)
    phrases = [q["phrase"] for q in out]
    # Limit constrains ACCEPTED quotes, not pre-filter candidates.
    assert len(out) == 2
    assert not any(_is_form_prompt(p) for p in phrases)
    # The genuine quotes backfill in pool order once the form prompts go.
    assert phrases == [
        "Pricing climbs at every renewal",
        "Support never responded for weeks",
    ]
