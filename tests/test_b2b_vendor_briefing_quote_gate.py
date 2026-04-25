"""Tests for Phase 2.3 4c-A vendor_briefing template verbatim gate.

Coverage:
  - _selected_reasoning_anchors drops witness rows that are missing
    phrase_verbatim or have it set to anything other than True. The
    Proof Anchors render block must not surface italic / smart-quote
    excerpt_text for non-verbatim rows (Sites 2 and 3).
  - Section 6 "What Customers Are Saying" gates briefing["evidence"]
    on phrase_verbatim is True; string evidence and unmarked dicts are
    dropped (Site 1). Producers will be migrated to stamp the marker
    in 4c-B; vault-sourced rows stay unmarked until a separate
    follow-up.

Policy: customer-facing blockquote / italic excerpt text requires an
explicit phrase_verbatim is True marker. Anything else fails closed.
"""

from __future__ import annotations

from typing import Any

from atlas_brain.templates.email.vendor_briefing import (  # noqa: E402
    _is_verbatim_witness,
    _render_reasoning_anchor_section,
    _selected_reasoning_anchors,
    render_vendor_briefing_html,
)


# ---------------------------------------------------------------------------
# _is_verbatim_witness predicate
# ---------------------------------------------------------------------------


def test_is_verbatim_witness_requires_explicit_true():
    assert _is_verbatim_witness({"phrase_verbatim": True}) is True


def test_is_verbatim_witness_rejects_missing_marker():
    assert _is_verbatim_witness({"excerpt_text": "anything"}) is False


def test_is_verbatim_witness_rejects_false_marker():
    assert _is_verbatim_witness({"phrase_verbatim": False}) is False


def test_is_verbatim_witness_rejects_truthy_non_true_values():
    # Strict "is True" -- a string, integer, or dict that happens to be
    # truthy must NOT pass the gate.
    assert _is_verbatim_witness({"phrase_verbatim": "true"}) is False
    assert _is_verbatim_witness({"phrase_verbatim": 1}) is False
    assert _is_verbatim_witness({"phrase_verbatim": {"value": True}}) is False


# ---------------------------------------------------------------------------
# Sites 2 & 3 -- Proof Anchors (anchor_examples + witness_highlights)
# ---------------------------------------------------------------------------


def _verbatim_witness(**overrides: Any) -> dict[str, Any]:
    base = {
        "witness_id": "witness:r1:0",
        "excerpt_text": "Pricing jumped 40% at the Q2 renewal.",
        "reviewer_company": "Acme Corp",
        "time_anchor": "Q2 renewal",
        "phrase_verbatim": True,
    }
    base.update(overrides)
    return base


def test_anchor_section_drops_unmarked_anchor_row():
    """A row missing phrase_verbatim must NOT render the italic
    excerpt block, even though excerpt_text is non-empty."""
    briefing = {
        "reasoning_anchor_examples": {
            "outlier_or_named_account": [
                {
                    "witness_id": "witness:unmarked:0",
                    "excerpt_text": "We were quoted $200k/year at renewal.",
                    "reviewer_company": "Northwind",
                    "time_anchor": "Q2 renewal",
                    # phrase_verbatim missing
                },
            ],
        },
    }
    assert _render_reasoning_anchor_section(briefing) == ""


def test_anchor_section_drops_explicit_false_anchor_row():
    briefing = {
        "reasoning_anchor_examples": {
            "common_pattern": [
                {
                    "witness_id": "witness:false:0",
                    "excerpt_text": "Customers complain about pricing tiers.",
                    "reviewer_company": "Northwind",
                    "phrase_verbatim": False,
                },
            ],
        },
    }
    assert _render_reasoning_anchor_section(briefing) == ""


def test_anchor_section_renders_only_verbatim_rows_when_mixed():
    """Verbatim row and non-verbatim row in the same payload -- only
    the verbatim row survives. The non-verbatim excerpt MUST NOT
    appear in the rendered HTML."""
    briefing = {
        "reasoning_anchor_examples": {
            "outlier_or_named_account": [
                _verbatim_witness(
                    excerpt_text="Hack Club hit a $200k Q2 renewal jump.",
                    reviewer_company="Hack Club",
                ),
            ],
            "common_pattern": [
                {
                    "witness_id": "witness:unmarked:0",
                    "excerpt_text": "Pricing pressure shows up in renewal reviews.",
                    "reviewer_company": "Northwind",
                    # phrase_verbatim missing
                },
            ],
        },
    }
    html = _render_reasoning_anchor_section(briefing)
    assert "Hack Club hit a $200k Q2 renewal jump." in html
    assert "Pricing pressure shows up in renewal reviews." not in html
    assert "Northwind" not in html


def test_anchor_section_drops_unmarked_witness_highlight_fallback():
    """Site 3 fallback path: witness_highlights without verbatim
    marker must NOT render."""
    briefing = {
        "reasoning_witness_highlights": [
            {
                "witness_id": "witness:fallback:0",
                "excerpt_text": "Team is evaluating alternatives before April renewal.",
                "reviewer_company": "Acme Corp",
                # phrase_verbatim missing
            },
        ],
    }
    assert _render_reasoning_anchor_section(briefing) == ""


def test_anchor_section_renders_verbatim_witness_highlight_fallback():
    briefing = {
        "reasoning_witness_highlights": [
            _verbatim_witness(
                excerpt_text="Team is evaluating alternatives before April renewal.",
                reviewer_company="Acme Corp",
                time_anchor="April renewal",
            ),
        ],
    }
    html = _render_reasoning_anchor_section(briefing)
    assert "Proof Anchors" in html
    assert "Team is evaluating alternatives before April renewal." in html
    assert "Acme Corp" in html


def test_selected_anchors_returns_empty_when_no_verbatim_rows():
    briefing = {
        "reasoning_anchor_examples": {
            "outlier_or_named_account": [
                {"witness_id": "w1", "excerpt_text": "no marker"},
            ],
        },
        "reasoning_witness_highlights": [
            {"witness_id": "w2", "excerpt_text": "still no marker"},
        ],
    }
    assert _selected_reasoning_anchors(briefing) == []


# ---------------------------------------------------------------------------
# Site 1 -- Section 6 "What Customers Are Saying" (briefing["evidence"])
# ---------------------------------------------------------------------------


def _briefing_skeleton(**overrides: Any) -> dict[str, Any]:
    base = {
        "vendor_name": "Acme",
        "category": "CRM",
        "churn_pressure_score": 55,
        "trend": "stable",
        "churn_signal_density": 8.0,
        "avg_urgency": 6.0,
        "review_count": 30,
        "dm_churn_rate": 18.0,
    }
    base.update(overrides)
    return base


def test_evidence_section_drops_string_evidence():
    """String evidence rows have no place to carry a marker -- must
    not render with quote styling."""
    briefing = _briefing_skeleton(
        evidence=[
            "Pricing keeps climbing every quarter.",
            "Support response time is unacceptable.",
        ],
    )
    html = render_vendor_briefing_html(briefing)
    assert "What Customers Are Saying" not in html
    assert "Pricing keeps climbing" not in html
    assert "Support response time" not in html


def test_evidence_section_drops_unmarked_dict_evidence():
    """Dict evidence rows without phrase_verbatim must not render,
    even though they carry a 'quote' / 'text' field."""
    briefing = _briefing_skeleton(
        evidence=[
            {"quote": "Pricing keeps climbing every quarter."},
            {"text": "Support response time is unacceptable."},
        ],
    )
    html = render_vendor_briefing_html(briefing)
    assert "What Customers Are Saying" not in html
    assert "Pricing keeps climbing" not in html
    assert "Support response time" not in html


def test_evidence_section_drops_explicit_false_marker():
    briefing = _briefing_skeleton(
        evidence=[
            {
                "quote": "Pricing keeps climbing every quarter.",
                "phrase_verbatim": False,
            },
        ],
    )
    html = render_vendor_briefing_html(briefing)
    assert "What Customers Are Saying" not in html
    assert "Pricing keeps climbing" not in html


def test_evidence_section_renders_marked_dict_evidence():
    """The path that 4c-B will restore: dict evidence with
    phrase_verbatim is True renders as a blockquote with smart quotes
    and red-bar styling."""
    briefing = _briefing_skeleton(
        evidence=[
            {
                "quote": "We were quoted $200k/year at the Q2 renewal.",
                "phrase_verbatim": True,
            },
        ],
    )
    html = render_vendor_briefing_html(briefing)
    assert "What Customers Are Saying" in html
    assert "We were quoted $200k/year at the Q2 renewal." in html
    # Smart quotes are how the template wraps the excerpt.
    assert "&ldquo;" in html and "&rdquo;" in html


def test_evidence_section_renders_only_marked_when_mixed():
    """Mixed list: only the verbatim dict survives the gate."""
    briefing = _briefing_skeleton(
        evidence=[
            "Plain string evidence -- should not render.",
            {"text": "Unmarked dict evidence -- should not render."},
            {
                "quote": "Marked verbatim quote that should render.",
                "phrase_verbatim": True,
            },
        ],
    )
    html = render_vendor_briefing_html(briefing)
    assert "Marked verbatim quote that should render." in html
    assert "Plain string evidence" not in html
    assert "Unmarked dict evidence" not in html


def test_evidence_section_takes_first_three_verbatim_rows_after_filtering():
    """The render cap must apply after the verbatim gate. Otherwise
    three non-verbatim rows at the front of the list can hide a valid
    quote that appears later."""
    briefing = _briefing_skeleton(
        evidence=[
            "string evidence -- should not render",
            {"quote": "unmarked dict -- should not render"},
            {"quote": "false marker -- should not render", "phrase_verbatim": False},
            {"quote": "first marked quote", "phrase_verbatim": True},
            {"quote": "second marked quote", "phrase_verbatim": True},
            {"quote": "third marked quote", "phrase_verbatim": True},
            {"quote": "fourth marked quote beyond cap", "phrase_verbatim": True},
        ],
    )
    html = render_vendor_briefing_html(briefing)
    assert "first marked quote" in html
    assert "second marked quote" in html
    assert "third marked quote" in html
    assert "fourth marked quote beyond cap" not in html
    assert "string evidence" not in html
    assert "unmarked dict" not in html
    assert "false marker" not in html


def test_evidence_section_omits_block_entirely_when_no_verbatim_rows():
    """If every evidence row fails the gate, the whole 'What
    Customers Are Saying' section must not appear in the email."""
    briefing = _briefing_skeleton(
        evidence=[
            "string a",
            "string b",
            {"quote": "no marker"},
        ],
    )
    html = render_vendor_briefing_html(briefing)
    assert "What Customers Are Saying" not in html
