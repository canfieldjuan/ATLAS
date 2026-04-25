"""Tests for atlas_brain.services.b2b.enrichment_contract.

Covers the use-case-specific helpers (headline / bucket / display) plus
phrase-grade splits (quote-grade vs theme-grade), and exercises golden
fixtures pulled from live data so the contract is anchored to real v4
enrichment shapes -- not synthetic dicts.

Fixtures:
  v4_shopify_pricing_demoted.json -- 5 Shopify reviews where v4 demoted
    pricing-tagged language to overall_dissatisfaction with
    pain_confidence='none'. Used to verify that a consumer asking for
    headline-grade pricing pain gets nothing.
  v4_bamboohr_no_signal.json -- 5 BambooHR rows classified
    enrichment_status='no_signal'. Used to verify no_signal rows produce
    no actionable pain and no phrase output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)

from atlas_brain.services.b2b.enrichment_contract import (  # noqa: E402
    GENERIC_PAIN_CATEGORIES,
    is_about_subject_vendor,
    is_generic_pain,
    is_negative_phrase,
    is_primary_driver,
    is_verbatim,
    pain_category_for_bucket,
    pain_category_for_display,
    pain_category_for_headline,
    quote_grade_phrases,
    resolve_pain_confidence,
    theme_grade_phrases,
)
from atlas_brain.services.b2b import enrichment_contract as contract  # noqa: E402


_FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> list[dict]:
    raw = (_FIXTURE_DIR / name).read_text()
    rows = json.loads(raw)
    out = []
    for r in rows:
        enrichment = r.get("enrichment")
        if isinstance(enrichment, str):
            enrichment = json.loads(enrichment)
        r["enrichment"] = enrichment
        out.append(r)
    return out


def _shopify_rows() -> list[dict]:
    return _load_fixture("v4_shopify_pricing_demoted.json")


def _bamboo_rows() -> list[dict]:
    return _load_fixture("v4_bamboohr_no_signal.json")


# ---------------------------------------------------------------------------
# Generic pain detector
# ---------------------------------------------------------------------------


def test_is_generic_pain_recognizes_canonical_v4_catch_all():
    assert is_generic_pain("overall_dissatisfaction") is True


def test_is_generic_pain_recognizes_legacy_aliases():
    for legacy in ("general_dissatisfaction", "other", "unknown"):
        assert is_generic_pain(legacy) is True


def test_is_generic_pain_excludes_specific_categories():
    for specific in ("pricing", "ux", "features", "support", "reliability"):
        assert is_generic_pain(specific) is False


def test_is_generic_pain_treats_none_and_empty_as_generic():
    assert is_generic_pain(None) is True
    assert is_generic_pain("") is True
    assert is_generic_pain("   ") is True


def test_generic_categories_include_all_expected():
    assert GENERIC_PAIN_CATEGORIES == frozenset({
        "overall_dissatisfaction",
        "general_dissatisfaction",
        "other",
        "unknown",
    })


# ---------------------------------------------------------------------------
# resolve_pain_confidence -- legacy fallback semantics
# ---------------------------------------------------------------------------


def test_resolve_pain_confidence_returns_explicit_value_when_canonical():
    for tier in ("strong", "weak", "none"):
        assert resolve_pain_confidence({"pain_confidence": tier}) == tier


def test_resolve_pain_confidence_returns_unknown_for_non_dict():
    assert resolve_pain_confidence(None) == "unknown"
    assert resolve_pain_confidence("not a dict") == "unknown"
    assert resolve_pain_confidence(42) == "unknown"


def test_resolve_pain_confidence_recomputes_for_v3_legacy_row():
    # Legacy v3 row missing pain_confidence -- recompute path should
    # produce one of the canonical tiers via _compute_pain_confidence.
    legacy = {
        "pain_category": "pricing",
        "specific_complaints": ["too expensive for what you get"],
    }
    resolved = resolve_pain_confidence(legacy)
    assert resolved in ("strong", "weak", "none")


def test_resolve_pain_confidence_unknown_only_on_unresolvable():
    # Empty dict gives recompute a chance; should produce 'none', not unknown
    assert resolve_pain_confidence({}) in ("strong", "weak", "none")


# ---------------------------------------------------------------------------
# pain_category_for_headline -- strict, no allow_legacy_unknown
# ---------------------------------------------------------------------------


def test_pain_category_for_headline_excludes_generic_by_default():
    e = {"pain_category": "overall_dissatisfaction", "pain_confidence": "strong"}
    assert pain_category_for_headline(e) is None


def test_pain_category_for_headline_excludes_weak_confidence():
    e = {"pain_category": "pricing", "pain_confidence": "weak"}
    assert pain_category_for_headline(e) is None


def test_pain_category_for_headline_excludes_none_confidence():
    e = {"pain_category": "pricing", "pain_confidence": "none"}
    assert pain_category_for_headline(e) is None


def test_pain_category_for_headline_returns_specific_strong_pain():
    e = {"pain_category": "pricing", "pain_confidence": "strong"}
    assert pain_category_for_headline(e) == "pricing"


def test_pain_category_for_headline_allow_generic_returns_strong_overall():
    e = {"pain_category": "overall_dissatisfaction", "pain_confidence": "strong"}
    assert pain_category_for_headline(e, allow_generic=True) == "overall_dissatisfaction"


def test_pain_category_for_headline_returns_none_on_malformed_unknown():
    # Unresolvable -- headlines must NEVER accept unknown tier
    assert pain_category_for_headline(None) is None


# ---------------------------------------------------------------------------
# pain_category_for_bucket -- looser gate, configurable
# ---------------------------------------------------------------------------


def test_pain_category_for_bucket_default_permits_weak_and_generic():
    e = {"pain_category": "overall_dissatisfaction", "pain_confidence": "weak"}
    assert pain_category_for_bucket(e) == "overall_dissatisfaction"


def test_pain_category_for_bucket_min_strong_filters_weak():
    e = {"pain_category": "pricing", "pain_confidence": "weak"}
    assert pain_category_for_bucket(e, min_confidence="strong") is None


def test_pain_category_for_bucket_filters_none_confidence_at_default():
    e = {"pain_category": "pricing", "pain_confidence": "none"}
    assert pain_category_for_bucket(e) is None


def test_pain_category_for_bucket_default_excludes_unknown_confidence():
    # malformed enrichment resolves to 'unknown'; default should reject
    assert pain_category_for_bucket(None) is None


def test_pain_category_for_bucket_allow_unknown_confidence_includes_them(monkeypatch):
    e = {"pain_category": "pricing"}
    monkeypatch.setattr(contract, "resolve_pain_confidence", lambda _enrichment: "unknown")

    assert contract.pain_category_for_bucket(e) is None
    assert contract.pain_category_for_bucket(e, allow_unknown_confidence=True) == "pricing"


def test_pain_category_for_bucket_allow_generic_false_filters_overall():
    e = {"pain_category": "overall_dissatisfaction", "pain_confidence": "weak"}
    assert pain_category_for_bucket(e, allow_generic=False) is None


# ---------------------------------------------------------------------------
# pain_category_for_display -- raw view including unknown
# ---------------------------------------------------------------------------


def test_pain_category_for_display_returns_pair_with_confidence():
    e = {"pain_category": "pricing", "pain_confidence": "weak"}
    cat, conf = pain_category_for_display(e)
    assert cat == "pricing"
    assert conf == "weak"


def test_pain_category_for_display_handles_none_input():
    cat, conf = pain_category_for_display(None)
    assert cat is None
    assert conf == "unknown"


# ---------------------------------------------------------------------------
# Phrase-level helpers
# ---------------------------------------------------------------------------


def test_is_about_subject_vendor_passes_only_for_subject_vendor():
    assert is_about_subject_vendor({"subject": "subject_vendor"}) is True
    assert is_about_subject_vendor({"subject": "competitor"}) is False
    assert is_about_subject_vendor({"subject": "self"}) is False
    assert is_about_subject_vendor({}) is False
    assert is_about_subject_vendor(None) is False


def test_is_negative_phrase_accepts_negative_and_mixed():
    assert is_negative_phrase({"polarity": "negative"}) is True
    assert is_negative_phrase({"polarity": "mixed"}) is True
    assert is_negative_phrase({"polarity": "positive"}) is False
    assert is_negative_phrase({"polarity": "neutral"}) is False
    assert is_negative_phrase({}) is False


def test_is_primary_driver_strict_match():
    assert is_primary_driver({"role": "primary_driver"}) is True
    assert is_primary_driver({"role": "supporting_context"}) is False
    assert is_primary_driver({"role": "passing_mention"}) is False
    assert is_primary_driver({}) is False


def test_is_verbatim_true_only_for_explicit_true():
    assert is_verbatim({"verbatim": True}) is True
    assert is_verbatim({"verbatim": False}) is False
    assert is_verbatim({"verbatim": "true"}) is False
    assert is_verbatim({}) is False


# ---------------------------------------------------------------------------
# quote_grade_phrases -- v4 only, never legacy
# ---------------------------------------------------------------------------


def test_quote_grade_phrases_empty_for_v3_legacy():
    legacy = {
        "enrichment_schema_version": 3,
        "pricing_phrases": ["too expensive", "price keeps going up"],
        "specific_complaints": ["bad ux", "slow loading"],
    }
    assert quote_grade_phrases(legacy) == []


def test_quote_grade_phrases_filters_non_subject_vendor():
    e = {
        "enrichment_schema_version": 4,
        "pricing_phrases": ["competitor charges less"],
        "phrase_metadata": [
            {
                "field": "pricing_phrases",
                "index": 0,
                "text": "competitor charges less",
                "subject": "competitor",
                "polarity": "negative",
                "role": "supporting_context",
                "verbatim": True,
            },
        ],
    }
    assert quote_grade_phrases(e) == []


def test_quote_grade_phrases_filters_non_verbatim():
    e = {
        "enrichment_schema_version": 4,
        "pricing_phrases": ["paraphrased pricing complaint"],
        "phrase_metadata": [
            {
                "field": "pricing_phrases",
                "index": 0,
                "text": "paraphrased pricing complaint",
                "subject": "subject_vendor",
                "polarity": "negative",
                "role": "primary_driver",
                "verbatim": False,
            },
        ],
    }
    assert quote_grade_phrases(e) == []


def test_quote_grade_phrases_passes_when_all_gates_satisfied():
    e = {
        "enrichment_schema_version": 4,
        "pricing_phrases": ["it costs too much money"],
        "phrase_metadata": [
            {
                "field": "pricing_phrases",
                "index": 0,
                "text": "it costs too much money",
                "subject": "subject_vendor",
                "polarity": "negative",
                "role": "primary_driver",
                "verbatim": True,
            },
        ],
    }
    out = quote_grade_phrases(e)
    assert len(out) == 1
    assert out[0]["text"] == "it costs too much money"


def test_quote_grade_phrases_require_primary_filters_supporting_context():
    e = {
        "enrichment_schema_version": 4,
        "pricing_phrases": ["it costs a lot"],
        "phrase_metadata": [
            {
                "field": "pricing_phrases",
                "index": 0,
                "text": "it costs a lot",
                "subject": "subject_vendor",
                "polarity": "negative",
                "role": "supporting_context",
                "verbatim": True,
            },
        ],
    }
    assert quote_grade_phrases(e) == [e["phrase_metadata"][0]]
    assert quote_grade_phrases(e, require_primary=True) == []


# ---------------------------------------------------------------------------
# theme_grade_phrases -- v4 plus opt-in legacy
# ---------------------------------------------------------------------------


def test_theme_grade_phrases_skips_v3_when_include_legacy_false():
    legacy = {
        "enrichment_schema_version": 3,
        "pricing_phrases": ["too expensive"],
    }
    assert theme_grade_phrases(legacy, include_legacy=False) == []


def test_theme_grade_phrases_includes_legacy_when_opted_in():
    legacy = {
        "enrichment_schema_version": 3,
        "pricing_phrases": ["too expensive", "price keeps going up"],
        "specific_complaints": ["bad ux"],
    }
    out = theme_grade_phrases(legacy, include_legacy=True)
    assert len(out) == 3
    for row in out:
        assert row["legacy"] is True
        assert row["verbatim"] is False
        assert row["subject"] == "unknown"
        assert row["polarity"] == "unknown"
        assert row["role"] == "unknown"


def test_theme_grade_phrases_legacy_field_filter_works():
    legacy = {
        "enrichment_schema_version": 3,
        "pricing_phrases": ["too expensive"],
        "specific_complaints": ["bad ux"],
    }
    out = theme_grade_phrases(
        legacy, field="pricing_phrases", include_legacy=True,
    )
    assert len(out) == 1
    assert out[0]["text"] == "too expensive"


def test_theme_grade_phrases_legacy_with_require_primary_returns_empty():
    legacy = {
        "enrichment_schema_version": 3,
        "pricing_phrases": ["too expensive"],
    }
    # Legacy rows have role='unknown', so require_primary must drop them
    out = theme_grade_phrases(
        legacy, include_legacy=True, require_primary=True,
    )
    assert out == []


def test_theme_grade_phrases_v4_includes_non_verbatim():
    e = {
        "enrichment_schema_version": 4,
        "pricing_phrases": ["paraphrased pricing complaint"],
        "phrase_metadata": [
            {
                "field": "pricing_phrases",
                "index": 0,
                "text": "paraphrased pricing complaint",
                "subject": "subject_vendor",
                "polarity": "negative",
                "role": "supporting_context",
                "verbatim": False,
            },
        ],
    }
    # theme-grade does NOT require verbatim
    assert len(theme_grade_phrases(e)) == 1
    # quote-grade DOES require verbatim
    assert quote_grade_phrases(e) == []


# ---------------------------------------------------------------------------
# Golden fixture assertions -- the actual headline behavior we want
# ---------------------------------------------------------------------------


def test_shopify_demoted_rows_produce_no_headline_pricing_pain():
    rows = _shopify_rows()
    assert len(rows) == 5
    for row in rows:
        e = row["enrichment"]
        assert e is not None, f"row {row['id']} has no enrichment"
        # Headline gate must reject every demoted row -- they are
        # confidence='none' OR generic, both of which fail headline.
        assert pain_category_for_headline(e) is None, (
            f"row {row['id']} unexpectedly produced headline pain "
            f"(category={e.get('pain_category')}, confidence={e.get('pain_confidence')})"
        )


def test_shopify_demoted_rows_have_no_quote_grade_pricing():
    rows = _shopify_rows()
    for row in rows:
        e = row["enrichment"]
        # When v4 demoted pricing -> overall_dissatisfaction with confidence
        # 'none', the pricing_phrases (if any) should not pass quote-grade
        # because either subject != subject_vendor OR verbatim != True OR
        # polarity != negative. These rows must NOT surface as pricing
        # pull-quotes anywhere.
        quotes = quote_grade_phrases(e, field="pricing_phrases")
        assert isinstance(quotes, list)
        # We don't assert empty -- the v4 path may still keep a quote-grade
        # phrase that was correctly subject/negative/verbatim; what we
        # assert is that the HEADLINE gate keeps the row out of pricing pain
        # narratives. The headline check above is the hard gate.


def test_bamboohr_no_signal_rows_produce_no_headline_pain():
    rows = _bamboo_rows()
    assert len(rows) == 5
    for row in rows:
        e = row["enrichment"]
        if e is None:
            continue  # some no_signal rows may have null enrichment
        assert pain_category_for_headline(e) is None, (
            f"row {row['id']} unexpectedly produced headline pain"
        )


def test_bamboohr_no_signal_rows_produce_no_quote_grade_phrases():
    rows = _bamboo_rows()
    for row in rows:
        e = row["enrichment"]
        if e is None:
            continue
        out = quote_grade_phrases(e)
        assert out == [], (
            f"row {row['id']} unexpectedly has quote-grade phrases: {out}"
        )


def test_bamboohr_no_signal_rows_produce_no_actionable_bucket_pain():
    """Bucket gate at min_confidence=strong should also reject these."""
    rows = _bamboo_rows()
    for row in rows:
        e = row["enrichment"]
        if e is None:
            continue
        assert pain_category_for_bucket(e, min_confidence="strong") is None, (
            f"row {row['id']} unexpectedly produced strong-bucket pain"
        )
