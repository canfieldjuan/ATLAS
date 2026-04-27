"""Tests for Phase 2.3 4c-A vendor_briefing template verbatim gate.

Coverage:
  - _selected_reasoning_anchors drops witness rows that are missing
    phrase_verbatim or have it set to anything other than True. The
    Proof Anchors render block must not surface italic / smart-quote
    excerpt_text for non-verbatim rows (Sites 2 and 3).
  - Section 6 "What Customers Are Saying" gates briefing["evidence"]
    on phrase_verbatim is True; string evidence and unmarked dicts are
    dropped (Site 1). SQL and vault producers stamp the marker before
    customer-facing quote rendering.

Policy: customer-facing blockquote / italic excerpt text requires an
explicit phrase_verbatim is True marker. Anything else fails closed.
"""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)

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


# ---------------------------------------------------------------------------
# Phase 2.3 4c-B -- _fetch_high_urgency_quotes producer migration
#
# These tests cover the SQL-evidence producer that feeds Site 1 in the
# email template. After 4c-B, the producer routes enrichment through
# the contract (quote_grade_phrases) and stamps phrase_verbatim=True
# plus traceability fields onto every output row. v3 rows produce
# nothing (no verbatim guarantee).
# ---------------------------------------------------------------------------


def _v4_enrichment(
    *,
    text: str = "Pricing jumped 40% at the Q2 renewal.",
    field: str = "pricing_phrases",
    subject: str = "subject_vendor",
    polarity: str = "negative",
    role: str = "primary_driver",
    verbatim: bool = True,
) -> dict[str, Any]:
    return {
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
            }
        ],
    }


def _read_evidence_row(
    *,
    enrichment: dict[str, Any] | str | None,
    review_id: str = "rev-1",
    source: str = "g2",
    reviewer_company: str = "Acme Corp",
    reviewer_title: str = "Director of Operations",
    industry: str = "SaaS",
    urgency: float = 8.4,
) -> dict[str, Any]:
    """Mirror the row shape returned by read_vendor_quote_evidence."""
    return {
        "review_id": review_id,
        "vendor_name": "Acme",
        "source": source,
        "reviewer_company": reviewer_company,
        "reviewer_title": reviewer_title,
        "role_level": reviewer_title,
        "pain_category": "pricing",
        "urgency": urgency,
        "review_text": "...",
        "rating": 2.0,
        "quotable_phrases": ["legacy quotable string -- ignored after 4c-B"],
        "enrichment_raw": enrichment,
        "industry": industry,
    }


async def test_fetch_high_urgency_quotes_extracts_quote_grade_phrase(monkeypatch):
    """v4 row with verbatim subject_vendor negative phrase produces a
    quote-grade output row stamped with phrase_verbatim=True, the
    phrase field, review_id, source, and quote_origin='review'."""
    from atlas_brain.autonomous.tasks import b2b_vendor_briefing as briefing_mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    enrichment = _v4_enrichment(
        text="Renewal jumped to $200k/year with no warning.",
        field="pricing_phrases",
    )
    sql_rows = [
        _read_evidence_row(
            enrichment=enrichment,
            review_id="rev-200k",
            source="capterra",
            reviewer_company="Hack Club",
        ),
    ]
    monkeypatch.setattr(
        shared_mod, "read_vendor_quote_evidence",
        AsyncMock(return_value=sql_rows),
    )

    out = await briefing_mod._fetch_high_urgency_quotes(
        pool=MagicMock(), vendor_name="Acme", limit=5,
    )
    assert len(out) == 1
    row = out[0]
    assert row["text"] == "Renewal jumped to $200k/year with no warning."
    assert row["quote"] == "Renewal jumped to $200k/year with no warning."
    assert row["phrase_verbatim"] is True
    assert row["quote_origin"] == "review"
    assert row["review_id"] == "rev-200k"
    assert row["source"] == "capterra"
    assert row["field"] == "pricing_phrases"
    assert row["company"] == "Hack Club"


async def test_fetch_high_urgency_quotes_drops_v3_rows(monkeypatch):
    """v3 enrichments have no verbatim guarantee -- produce nothing."""
    from atlas_brain.autonomous.tasks import b2b_vendor_briefing as briefing_mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    legacy_enrichment = {
        "enrichment_schema_version": 3,
        "pricing_phrases": ["renewal price way too high"],
        "quotable_phrases": ["renewal price way too high"],
    }
    sql_rows = [_read_evidence_row(enrichment=legacy_enrichment)]
    monkeypatch.setattr(
        shared_mod, "read_vendor_quote_evidence",
        AsyncMock(return_value=sql_rows),
    )

    out = await briefing_mod._fetch_high_urgency_quotes(
        pool=MagicMock(), vendor_name="Acme", limit=5,
    )
    assert out == []


async def test_fetch_high_urgency_quotes_drops_non_verbatim_v4(monkeypatch):
    """v4 row with verbatim=False fails the contract gate."""
    from atlas_brain.autonomous.tasks import b2b_vendor_briefing as briefing_mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    enrichment = _v4_enrichment(verbatim=False)
    sql_rows = [_read_evidence_row(enrichment=enrichment)]
    monkeypatch.setattr(
        shared_mod, "read_vendor_quote_evidence",
        AsyncMock(return_value=sql_rows),
    )

    out = await briefing_mod._fetch_high_urgency_quotes(
        pool=MagicMock(), vendor_name="Acme", limit=5,
    )
    assert out == []


async def test_fetch_high_urgency_quotes_drops_non_subject_vendor(monkeypatch):
    """A phrase about a competitor is not quote-grade for THIS vendor's
    briefing -- the contract drops subject!=subject_vendor rows."""
    from atlas_brain.autonomous.tasks import b2b_vendor_briefing as briefing_mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    enrichment = _v4_enrichment(subject="competitor")
    sql_rows = [_read_evidence_row(enrichment=enrichment)]
    monkeypatch.setattr(
        shared_mod, "read_vendor_quote_evidence",
        AsyncMock(return_value=sql_rows),
    )

    out = await briefing_mod._fetch_high_urgency_quotes(
        pool=MagicMock(), vendor_name="Acme", limit=5,
    )
    assert out == []


async def test_fetch_high_urgency_quotes_drops_positive_polarity(monkeypatch):
    from atlas_brain.autonomous.tasks import b2b_vendor_briefing as briefing_mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    enrichment = _v4_enrichment(polarity="positive")
    sql_rows = [_read_evidence_row(enrichment=enrichment)]
    monkeypatch.setattr(
        shared_mod, "read_vendor_quote_evidence",
        AsyncMock(return_value=sql_rows),
    )

    out = await briefing_mod._fetch_high_urgency_quotes(
        pool=MagicMock(), vendor_name="Acme", limit=5,
    )
    assert out == []


async def test_fetch_high_urgency_quotes_accepts_json_string_enrichment(monkeypatch):
    """asyncpg normally returns JSONB as a Python dict, but defensive
    handling for any raw-string code path: a JSON string also works."""
    from atlas_brain.autonomous.tasks import b2b_vendor_briefing as briefing_mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    enrichment = _v4_enrichment(text="JSON string path verbatim quote.")
    sql_rows = [_read_evidence_row(enrichment=json.dumps(enrichment))]
    monkeypatch.setattr(
        shared_mod, "read_vendor_quote_evidence",
        AsyncMock(return_value=sql_rows),
    )

    out = await briefing_mod._fetch_high_urgency_quotes(
        pool=MagicMock(), vendor_name="Acme", limit=5,
    )
    assert len(out) == 1
    assert out[0]["text"] == "JSON string path verbatim quote."
    assert out[0]["phrase_verbatim"] is True


async def test_fetch_high_urgency_quotes_skips_malformed_enrichment(monkeypatch):
    """Bad JSON or unexpected types are skipped, not raised."""
    from atlas_brain.autonomous.tasks import b2b_vendor_briefing as briefing_mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    sql_rows = [
        _read_evidence_row(enrichment="{not valid json"),
        _read_evidence_row(enrichment=None),
        _read_evidence_row(enrichment=12345),
    ]
    monkeypatch.setattr(
        shared_mod, "read_vendor_quote_evidence",
        AsyncMock(return_value=sql_rows),
    )

    out = await briefing_mod._fetch_high_urgency_quotes(
        pool=MagicMock(), vendor_name="Acme", limit=5,
    )
    assert out == []


async def test_fetch_high_urgency_quotes_respects_limit_at_phrase_level(monkeypatch):
    """When several rows yield more than `limit` quote-grade phrases,
    output stops at `limit`. Different from the legacy producer, which
    counted SQL rows -- 4c-B counts emitted phrases."""
    from atlas_brain.autonomous.tasks import b2b_vendor_briefing as briefing_mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    sql_rows = [
        _read_evidence_row(
            enrichment=_v4_enrichment(text=f"verbatim phrase {i}"),
            review_id=f"rev-{i}",
        )
        for i in range(10)
    ]
    monkeypatch.setattr(
        shared_mod, "read_vendor_quote_evidence",
        AsyncMock(return_value=sql_rows),
    )

    out = await briefing_mod._fetch_high_urgency_quotes(
        pool=MagicMock(), vendor_name="Acme", limit=3,
    )
    assert len(out) == 3
    assert out[0]["text"] == "verbatim phrase 0"
    assert out[2]["text"] == "verbatim phrase 2"


async def test_fetch_high_urgency_quotes_output_renders_through_template(monkeypatch):
    """End-to-end: producer output, mounted on briefing['evidence'],
    survives the 4c-A template gate and renders as a blockquote with
    smart-quote styling. Closes the loop on Site 1 restoration."""
    from atlas_brain.autonomous.tasks import b2b_vendor_briefing as briefing_mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    enrichment = _v4_enrichment(
        text="Quoted $200k/year at our Q2 renewal -- no warning.",
    )
    sql_rows = [_read_evidence_row(enrichment=enrichment)]
    monkeypatch.setattr(
        shared_mod, "read_vendor_quote_evidence",
        AsyncMock(return_value=sql_rows),
    )

    quotes = await briefing_mod._fetch_high_urgency_quotes(
        pool=MagicMock(), vendor_name="Acme", limit=5,
    )
    assert len(quotes) == 1

    briefing = _briefing_skeleton(evidence=quotes)
    html = render_vendor_briefing_html(briefing)
    assert "What Customers Are Saying" in html
    assert "Quoted $200k/year at our Q2 renewal -- no warning." in html
    assert "&ldquo;" in html and "&rdquo;" in html


async def test_evidence_vault_quotes_require_verbatim_marker():
    """Vault rows without phrase_verbatim stay closed at the producer edge."""
    from atlas_brain.autonomous.tasks import b2b_vendor_briefing as briefing_mod

    vault = {
        "weakness_evidence": [
            {
                "key": "pricing",
                "best_quote": "Vault quote text -- still unmarked.",
                "quote_source": {
                    "company": "Hack Club",
                    "reviewer_title": "VP Eng",
                    "source": "g2",
                    "review_id": "vault-1",
                },
                "supporting_metrics": {"avg_urgency_when_mentioned": 7.5},
                "mention_count_total": 4,
            },
        ],
    }
    out = briefing_mod._briefing_quotes_from_evidence_vault(vault)
    assert out == []


async def test_evidence_vault_quotes_stamp_verbatim_for_email_render():
    """Marked vault rows are re-emitted as vault-origin verbatim evidence."""
    from atlas_brain.autonomous.tasks import b2b_vendor_briefing as briefing_mod

    vault = {
        "weakness_evidence": [
            {
                "key": "pricing",
                "best_quote": "Vault quote text -- now marked verbatim.",
                "phrase_verbatim": True,
                "quote_origin": "review",
                "quote_source": {
                    "company": "Hack Club",
                    "reviewer_title": "VP Eng",
                    "source": "g2",
                    "review_id": "vault-1",
                },
                "supporting_metrics": {"avg_urgency_when_mentioned": 7.5},
                "mention_count_total": 4,
            },
        ],
    }
    out = briefing_mod._briefing_quotes_from_evidence_vault(vault)
    assert len(out) == 1
    assert out[0]["phrase_verbatim"] is True
    assert out[0]["quote_origin"] == "vault"

    briefing = _briefing_skeleton(evidence=out)
    html = render_vendor_briefing_html(briefing)
    assert "What Customers Are Saying" in html
    assert "Vault quote text -- now marked verbatim." in html


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


def test_archetype_confidence_accepts_string_numeric_payload():
    """DB/JSON payloads can carry archetype_confidence as a string.

    The renderer must coerce it before applying numeric format specs, otherwise
    scheduled vendor briefings fail with "Unknown format code 'f'".
    """
    briefing = _briefing_skeleton(
        archetype="pricing_shock",
        archetype_confidence="0.62",
    )

    html = render_vendor_briefing_html(briefing)

    assert "Pricing Shock" in html
    assert "Confidence: 62%" in html
