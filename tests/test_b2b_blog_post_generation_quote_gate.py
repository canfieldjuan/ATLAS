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
    _blog_quote_highlights,
    _blueprint_best_fit_guide,
    _blueprint_market_landscape,
    _blueprint_pain_point_roundup,
    _blueprint_pricing_reality_check,
    _blueprint_vendor_showdown,
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


def test_pricing_blueprint_preserves_legacy_vault_rows_until_vault_migration():
    vault_row = {
        "vendor": "Shopify",
        "urgency": 6.0,
        "role": "Finance",
        "text": "Vault-sourced quote text is still handled by the legacy path.",
        # Phase 2.3 wrapper marker: vault rows must carry this so they
        # are routed to the legacy truncation path while we wait on a
        # separate vault producer migration. Unmarked rows are dropped.
        "quote_origin": "vault",
    }
    blueprint = _blueprint_pricing_reality_check(
        _pricing_ctx(),
        {"pricing_reviews": [vault_row]},
    )
    assert blueprint.quotable_phrases == [{
        "phrase": "Vault-sourced quote text is still handled by the legacy path.",
        "vendor": "Shopify",
        "urgency": 6.0,
        "role": "Finance",
    }]


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
    }
    out = _split_and_gate_blog_quotes([row])
    assert len(out) == 1
    assert out[0]["phrase"] == "Vault-curated text passes through legacy truncation."
    assert out[0]["vendor"] == "Shopify"


def test_wrapper_handles_vault_phrase_field_too():
    """Vault rows from _merge_blog_quotes_with_evidence_vault carry
    'phrase' rather than 'text' -- both shapes accepted."""
    row = {
        "phrase": "Curated phrase from vault.",
        "vendor": "Shopify",
        "urgency": 4.0,
        "role": "VP",
        "quote_origin": "vault",
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


def test_blog_quote_highlights_preserves_explicit_vault_rows():
    row = {
        "vendor": "Shopify",
        "phrase": "Vault-curated quote text.",
        "role": "Finance",
        "quote_origin": "vault",
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
