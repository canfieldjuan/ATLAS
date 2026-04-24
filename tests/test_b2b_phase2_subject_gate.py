"""Tests for Phase 2: subject-attribution gate (Layer 1).

Covers all three integration sites:
- `_derive_pain_categories` in b2b_enrichment.py
- `_classify_complaint_pain` in _b2b_witnesses.py (called by derive_evidence_spans)
- `derive_price_complaint` in evidence_engine.py

For each: verify the v1 (legacy) path keeps current behavior AND the v2
(tagged) path filters to subject='subject_vendor' before scoring.
"""

from __future__ import annotations

from typing import Any

from atlas_brain.autonomous.tasks._b2b_witnesses import (
    _classify_complaint_pain,
    derive_evidence_spans,
)
from atlas_brain.autonomous.tasks.b2b_enrichment import (
    _compute_derived_fields,
    _derive_pain_categories,
)
from atlas_brain.reasoning.evidence_engine import get_evidence_engine


# ---------------------------------------------------------------------------
# _derive_pain_categories: v1 path unchanged, v2 path filters by subject
# ---------------------------------------------------------------------------


def test_derive_pain_categories_v1_legacy_unchanged():
    """v1 enrichment (no enrichment_schema_version, no phrase_metadata)
    falls through to keyword-only scoring across all phrases."""
    result = {
        "specific_complaints": ["the pricing is brutal"],
        "pricing_phrases": ["a little expensive"],
        "feature_gaps": [],
        "quotable_phrases": [],
    }
    cats = _derive_pain_categories(result)
    # Both phrases scored; pricing wins on keyword count
    assert len(cats) >= 1
    assert cats[0]["category"] == "pricing"


def test_derive_pain_categories_v2_filters_self_attributed_phrases_out():
    """v2 enrichment: phrases tagged subject='self' must NOT contribute
    to pain scoring even when their text matches pricing keywords. This is
    the screenshot-review canary: '$1,500 a month' is the reviewer's own
    cost, not a complaint about the subject vendor."""
    result = {
        "enrichment_schema_version": 4,
        "specific_complaints": [],
        "pricing_phrases": [
            "I don't pay 1.500$ a month",  # subject=self -> filtered
            "I built my own for $30",       # subject=self -> filtered
        ],
        "feature_gaps": ["security on a higher standard"],  # subject=subject_vendor
        "quotable_phrases": [],
        "phrase_metadata": [
            {"field": "pricing_phrases", "index": 0,
             "text": "I don't pay 1.500$ a month",
             "subject": "self", "polarity": "positive",
             "role": "supporting_context", "verbatim": True},
            {"field": "pricing_phrases", "index": 1,
             "text": "I built my own for $30",
             "subject": "self", "polarity": "positive",
             "role": "supporting_context", "verbatim": True},
            {"field": "feature_gaps", "index": 0,
             "text": "security on a higher standard",
             "subject": "subject_vendor", "polarity": "negative",
             "role": "primary_driver", "verbatim": True},
        ],
    }
    cats = _derive_pain_categories(result)
    # Pricing phrases were the reviewer's own spending; only the feature
    # gap (subject_vendor) survives. Pricing must NOT be the primary pain.
    primary_categories = [
        c["category"] for c in cats
        if c.get("severity") == "primary"
    ]
    assert "pricing" not in primary_categories, \
        f"pricing leaked into primary pain via subject=self phrases: {cats}"


def test_derive_pain_categories_v2_keeps_subject_vendor_pricing():
    """When pricing_phrases ARE tagged subject_vendor, pricing should
    legitimately score and win."""
    result = {
        "enrichment_schema_version": 4,
        "specific_complaints": [],
        "pricing_phrases": ["their pricing killed our budget"],
        "feature_gaps": [],
        "quotable_phrases": [],
        "phrase_metadata": [
            {"field": "pricing_phrases", "index": 0,
             "text": "their pricing killed our budget",
             "subject": "subject_vendor", "polarity": "negative",
             "role": "primary_driver", "verbatim": True},
        ],
    }
    cats = _derive_pain_categories(result)
    primary = next((c for c in cats if c.get("severity") == "primary"), None)
    assert primary is not None
    assert primary["category"] == "pricing"


def test_derive_pain_categories_v2_all_phrases_filtered_returns_empty():
    """If every phrase is tagged non-subject_vendor, nothing scores;
    pain_categories list is empty (legacy behavior for no-evidence reviews)."""
    result = {
        "enrichment_schema_version": 4,
        "specific_complaints": ["expensive"],
        "pricing_phrases": [],
        "feature_gaps": [],
        "quotable_phrases": [],
        "phrase_metadata": [
            {"field": "specific_complaints", "index": 0, "text": "expensive",
             "subject": "alternative", "polarity": "negative",
             "role": "supporting_context", "verbatim": True},
        ],
    }
    cats = _derive_pain_categories(result)
    assert cats == []


def test_compute_derived_fields_v2_gate_runs_before_price_complaint_derivation():
    """Real pipeline path: the LLM may return phrase_metadata before the
    pipeline sets enrichment_schema_version=4. Phase 2 must still gate
    derived pain and price_complaint before persistence."""
    source_row = {
        "id": "rev-phase2-canary",
        "vendor_name": "Monday.com",
        "source": "manual",
        "rating": 2,
        "rating_max": 5,
        "summary": None,
        "review_text": "I said the pricing is brutal for my own setup.",
        "raw_metadata": {"source_weight": 1.0},
    }
    result = {
        "specific_complaints": [],
        "pricing_phrases": ["the pricing is brutal"],
        "feature_gaps": [],
        "quotable_phrases": [],
        "recommendation_language": [],
        "positive_aspects": [],
        "competitors_mentioned": [],
        "event_mentions": [],
        "churn_signals": {},
        "reviewer_context": {},
        "budget_signals": {},
        "use_case": {},
        "sentiment_trajectory": {},
        "buyer_authority": {},
        "timeline": {},
        "contract_context": {},
        "phrase_metadata": [
            {"field": "pricing_phrases", "index": 0,
             "text": "the pricing is brutal",
             "subject": "self", "polarity": "negative",
             "role": "supporting_context", "verbatim": True},
        ],
    }

    out = _compute_derived_fields(result, source_row)

    assert out["enrichment_schema_version"] == 4
    assert out["pain_categories"] == []
    assert out["pain_category"] == "overall_dissatisfaction"
    assert out["contract_context"]["price_complaint"] is False


# ---------------------------------------------------------------------------
# _classify_complaint_pain: subject keyword arg gates the classification
# ---------------------------------------------------------------------------


def test_classify_complaint_pain_no_subject_arg_is_legacy_behavior():
    result = _classify_complaint_pain("the pricing is brutal", "default")
    assert result == "pricing"


def test_classify_complaint_pain_subject_vendor_classifies_normally():
    result = _classify_complaint_pain(
        "the pricing is brutal", "default",
        subject="subject_vendor",
    )
    assert result == "pricing"


def test_classify_complaint_pain_subject_self_returns_none():
    """Self-attributed pricing complaint must NOT count as a vendor pain."""
    result = _classify_complaint_pain(
        "I pay 1500/month for my own setup", "default",
        subject="self",
    )
    assert result is None


def test_classify_complaint_pain_subject_alternative_returns_none():
    result = _classify_complaint_pain(
        "Salesforce charges twice that", "default",
        subject="alternative",
    )
    assert result is None


def test_classify_complaint_pain_subject_unclear_returns_none():
    """unclear is conservative: when the LLM didn't commit to subject,
    don't let the phrase contribute pain."""
    result = _classify_complaint_pain(
        "the pricing is brutal", "default",
        subject="unclear",
    )
    assert result is None


# ---------------------------------------------------------------------------
# derive_evidence_spans: passes subject from metadata to classifier
# ---------------------------------------------------------------------------


def _baseline_source_row() -> dict[str, Any]:
    return {
        "id": "rev-1",
        "summary": None,
        "review_text": "Some review body.",
        "pros": "",
        "cons": "",
        "reviewer_company": None,
        "reviewer_title": None,
        "rating": 3,
    }


def test_derive_evidence_spans_v2_filters_self_attributed_complaint_pain():
    """A v2 complaint phrase tagged subject=self should produce a span
    with pain_category=None (the gate stripped it), not a misleading
    pain like 'pricing'."""
    result = {
        "enrichment_schema_version": 4,
        "specific_complaints": ["I pay 1500 for my own setup"],
        "pricing_phrases": [],
        "feature_gaps": [],
        "quotable_phrases": [],
        "recommendation_language": [],
        "positive_aspects": [],
        "competitors_mentioned": [],
        "event_mentions": [],
        "churn_signals": {},
        "reviewer_context": {},
        "budget_signals": {},
        "use_case": {},
        "urgency_indicators": {},
        "sentiment_trajectory": {},
        "buyer_authority": {},
        "timeline": {},
        "contract_context": {},
        "pain_category": "pricing",  # legacy override would say pricing
        "phrase_metadata": [
            {"field": "specific_complaints", "index": 0,
             "text": "I pay 1500 for my own setup",
             "subject": "self", "polarity": "positive",
             "role": "supporting_context", "verbatim": True},
        ],
    }
    spans = derive_evidence_spans(result, _baseline_source_row())
    complaint_spans = [s for s in spans if s.get("signal_type") == "complaint"]
    assert complaint_spans, "expected a complaint span to be emitted"
    # Even though the legacy default_pain is 'pricing', the gate strips it
    # because subject=self.
    assert complaint_spans[0]["pain_category"] is None


def test_derive_evidence_spans_v2_subject_vendor_keeps_classified_pain():
    result = {
        "enrichment_schema_version": 4,
        "specific_complaints": ["the pricing is brutal"],
        "pricing_phrases": [],
        "feature_gaps": [],
        "quotable_phrases": [],
        "recommendation_language": [],
        "positive_aspects": [],
        "competitors_mentioned": [],
        "event_mentions": [],
        "churn_signals": {},
        "reviewer_context": {},
        "budget_signals": {},
        "use_case": {},
        "urgency_indicators": {},
        "sentiment_trajectory": {},
        "buyer_authority": {},
        "timeline": {},
        "contract_context": {},
        "pain_category": "pricing",
        "phrase_metadata": [
            {"field": "specific_complaints", "index": 0,
             "text": "the pricing is brutal",
             "subject": "subject_vendor", "polarity": "negative",
             "role": "primary_driver", "verbatim": True},
        ],
    }
    spans = derive_evidence_spans(result, _baseline_source_row())
    complaint_spans = [s for s in spans if s.get("signal_type") == "complaint"]
    assert complaint_spans
    assert complaint_spans[0]["pain_category"] == "pricing"


def test_derive_evidence_spans_v1_legacy_path_unaffected():
    """v1 enrichment (no metadata) goes through the existing classifier
    without the gate; legacy keyword-only behavior preserved."""
    result = {
        "specific_complaints": ["the pricing is brutal"],
        "pricing_phrases": [],
        "feature_gaps": [],
        "quotable_phrases": [],
        "recommendation_language": [],
        "positive_aspects": [],
        "competitors_mentioned": [],
        "event_mentions": [],
        "churn_signals": {},
        "reviewer_context": {},
        "budget_signals": {},
        "use_case": {},
        "urgency_indicators": {},
        "sentiment_trajectory": {},
        "buyer_authority": {},
        "timeline": {},
        "contract_context": {},
        "pain_category": "pricing",
    }
    spans = derive_evidence_spans(result, _baseline_source_row())
    complaint_spans = [s for s in spans if s.get("signal_type") == "complaint"]
    assert complaint_spans
    assert complaint_spans[0]["pain_category"] == "pricing"


# ---------------------------------------------------------------------------
# derive_price_complaint: filtered enrichment view for v2
# ---------------------------------------------------------------------------


def test_derive_price_complaint_v1_legacy_path_counts_all_phrases():
    """v1: any non-empty pricing_phrases entry can flip the flag (subject
    to the existing positive-sentiment heuristic)."""
    engine = get_evidence_engine()
    enrichment = {
        "pricing_phrases": ["the pricing is brutal"],
        "specific_complaints": [],
        "budget_signals": {},
    }
    assert engine.derive_price_complaint(enrichment) is True


def test_derive_price_complaint_v2_filters_self_attributed_pricing():
    """v2: a pricing phrase tagged subject=self does NOT trip the
    price_complaint flag for the subject vendor."""
    engine = get_evidence_engine()
    enrichment = {
        "enrichment_schema_version": 4,
        "pricing_phrases": ["I don't pay 1500 a month for my setup"],
        "specific_complaints": [],
        "budget_signals": {},
        "phrase_metadata": [
            {"field": "pricing_phrases", "index": 0,
             "text": "I don't pay 1500 a month for my setup",
             "subject": "self", "polarity": "positive",
             "role": "supporting_context", "verbatim": True},
        ],
    }
    assert engine.derive_price_complaint(enrichment) is False


def test_derive_price_complaint_v2_keeps_subject_vendor_pricing():
    engine = get_evidence_engine()
    enrichment = {
        "enrichment_schema_version": 4,
        "pricing_phrases": ["their pricing killed our budget"],
        "specific_complaints": [],
        "budget_signals": {},
        "phrase_metadata": [
            {"field": "pricing_phrases", "index": 0,
             "text": "their pricing killed our budget",
             "subject": "subject_vendor", "polarity": "negative",
             "role": "primary_driver", "verbatim": True},
        ],
    }
    assert engine.derive_price_complaint(enrichment) is True


def test_derive_price_complaint_v2_mixed_subjects_only_counts_subject_vendor():
    """A review with one self-pricing phrase AND one subject_vendor pricing
    phrase should still flip True. The gate filters out the self phrase
    but preserves the subject_vendor one."""
    engine = get_evidence_engine()
    enrichment = {
        "enrichment_schema_version": 4,
        "pricing_phrases": [
            "I built my own for $30",
            "Their pricing is unaffordable",
        ],
        "specific_complaints": [],
        "budget_signals": {},
        "phrase_metadata": [
            {"field": "pricing_phrases", "index": 0,
             "text": "I built my own for $30",
             "subject": "self", "polarity": "positive",
             "role": "supporting_context", "verbatim": True},
            {"field": "pricing_phrases", "index": 1,
             "text": "Their pricing is unaffordable",
             "subject": "subject_vendor", "polarity": "negative",
             "role": "primary_driver", "verbatim": True},
        ],
    }
    assert engine.derive_price_complaint(enrichment) is True
