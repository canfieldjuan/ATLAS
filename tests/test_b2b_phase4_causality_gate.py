"""Phase 4 (Layer 3 -- causality gate) tests.

These cover:
  - _count_corroborating_signals  (universal churn / sentiment count)
  - _count_pain_phrase_matches    (per-category phrase counter, gated by
                                    subject + polarity)
  - _compute_pain_confidence      (rubric: strong / weak / none)
  - _demote_primary_pain          (in-place pain_categories rewrite)
  - End-to-end demotion through _compute_derived_fields when the primary
    pain has only a single passing-mention keyword and no other
    corroboration.
"""

from __future__ import annotations

from atlas_brain.autonomous.tasks.b2b_enrichment import (
    _compute_derived_fields,
    _compute_pain_confidence,
    _count_corroborating_signals,
    _count_pain_phrase_matches,
    _demote_primary_pain,
)


# ---------------------------------------------------------------------------
# _count_corroborating_signals
# ---------------------------------------------------------------------------


def test_count_corroborating_signals_zero_for_clean_review():
    result = {
        "churn_signals": {},
        "would_recommend": True,
        "sentiment_trajectory": {"direction": "stable_positive"},
    }
    assert _count_corroborating_signals(result) == 0


def test_count_corroborating_signals_intent_to_leave():
    result = {
        "churn_signals": {"intent_to_leave": True},
        "would_recommend": None,
        "sentiment_trajectory": {"direction": "unknown"},
    }
    assert _count_corroborating_signals(result) == 1


def test_count_corroborating_signals_aggregates_multiple():
    result = {
        "churn_signals": {
            "intent_to_leave": True,
            "actively_evaluating": True,
            "migration_in_progress": False,
        },
        "would_recommend": False,
        "sentiment_trajectory": {"direction": "consistently_negative"},
    }
    # intent_to_leave + actively_evaluating + would_recommend=False +
    # consistently_negative direction = 4
    assert _count_corroborating_signals(result) == 4


def test_count_corroborating_signals_ignores_truthy_non_bool():
    # Strings that look truthy must not count; only is True triggers
    result = {
        "churn_signals": {"intent_to_leave": "true"},
        "would_recommend": 0,
        "sentiment_trajectory": {"direction": ""},
    }
    assert _count_corroborating_signals(result) == 0


# ---------------------------------------------------------------------------
# _count_pain_phrase_matches
# ---------------------------------------------------------------------------


def _v2(result: dict, phrase_metadata: list[dict]) -> dict:
    """Helper: stamp v2 schema + phrase_metadata onto a result dict."""
    return {
        **result,
        "enrichment_schema_version": 4,
        "phrase_metadata": phrase_metadata,
    }


def test_count_pain_phrase_matches_v2_filters_by_subject_and_polarity():
    result = _v2(
        {
            "specific_complaints": [
                "support takes forever to respond",   # subject=vendor, neg
                "support is great at our company",    # subject=self, pos
                "their support response is mediocre", # subject=vendor, mixed
            ]
        },
        [
            {
                "field": "specific_complaints",
                "index": 0,
                "subject": "subject_vendor",
                "polarity": "negative",
                "role": "primary_driver",
                "verbatim": True,
            },
            {
                "field": "specific_complaints",
                "index": 1,
                "subject": "self",
                "polarity": "positive",
                "role": "passing_mention",
                "verbatim": True,
            },
            {
                "field": "specific_complaints",
                "index": 2,
                "subject": "subject_vendor",
                "polarity": "mixed",
                "role": "supporting_context",
                "verbatim": True,
            },
        ],
    )
    # Only phrases that are subject_vendor AND in (negative, mixed) and
    # also keyword-match "support" should count.
    assert _count_pain_phrase_matches(result, "support") == 2


def test_count_pain_phrase_matches_v2_drops_positive_phrases():
    result = _v2(
        {
            "pricing_phrases": ["good value at $500/seat"],
        },
        [
            {
                "field": "pricing_phrases",
                "index": 0,
                "subject": "subject_vendor",
                "polarity": "positive",
                "role": "supporting_context",
                "verbatim": True,
            }
        ],
    )
    assert _count_pain_phrase_matches(result, "pricing") == 0


def test_count_pain_phrase_matches_unknown_category_returns_zero():
    result = _v2({"specific_complaints": ["some text"]}, [])
    assert _count_pain_phrase_matches(result, "not_a_real_pain_category") == 0


def test_count_pain_phrase_matches_v1_falls_back_to_keyword_scan():
    result = {
        "specific_complaints": [
            "pricing is a problem",
            "another expensive complaint",
        ],
        "pricing_phrases": ["expensive renewal"],
    }
    # v1 (no schema_version, no metadata) -> all phrases counted by
    # keyword pattern alone. "pricing" pattern matches the words
    # "pricing" and "expensive".
    assert _count_pain_phrase_matches(result, "pricing") >= 2


# ---------------------------------------------------------------------------
# _compute_pain_confidence
# ---------------------------------------------------------------------------


def test_compute_pain_confidence_strong_with_two_matching_phrases():
    result = _v2(
        {
            "specific_complaints": [
                "support takes forever to respond",
                "their support team never replies",
            ],
            "churn_signals": {},
            "would_recommend": None,
            "sentiment_trajectory": {"direction": "unknown"},
        },
        [
            {
                "field": "specific_complaints",
                "index": 0,
                "subject": "subject_vendor",
                "polarity": "negative",
                "role": "primary_driver",
                "verbatim": True,
            },
            {
                "field": "specific_complaints",
                "index": 1,
                "subject": "subject_vendor",
                "polarity": "negative",
                "role": "primary_driver",
                "verbatim": True,
            },
        ],
    )
    assert _compute_pain_confidence(result, "support") == "strong"


def test_compute_pain_confidence_weak_with_one_phrase_and_signal():
    result = _v2(
        {
            "specific_complaints": ["support is slow"],
            "churn_signals": {"intent_to_leave": True},
            "would_recommend": None,
            "sentiment_trajectory": {"direction": "unknown"},
        },
        [
            {
                "field": "specific_complaints",
                "index": 0,
                "subject": "subject_vendor",
                "polarity": "negative",
                "role": "primary_driver",
                "verbatim": True,
            }
        ],
    )
    assert _compute_pain_confidence(result, "support") == "weak"


def test_compute_pain_confidence_none_with_one_phrase_no_signals():
    result = _v2(
        {
            "specific_complaints": ["mentioned support once in passing"],
            "churn_signals": {},
            "would_recommend": True,
            "sentiment_trajectory": {"direction": "stable_positive"},
        },
        [
            {
                "field": "specific_complaints",
                "index": 0,
                "subject": "subject_vendor",
                "polarity": "negative",
                "role": "passing_mention",
                "verbatim": True,
            }
        ],
    )
    # 1 phrase + 0 universal signals + would_recommend=True
    # would_recommend=True does not count as a signal (only False does)
    assert _compute_pain_confidence(result, "support") == "none"


def test_compute_pain_confidence_overall_dissatisfaction_uses_signals_alone():
    # No keyword pattern for overall_dissatisfaction; signals drive grade.
    weak = {
        "churn_signals": {"intent_to_leave": True},
        "would_recommend": None,
        "sentiment_trajectory": {"direction": "unknown"},
    }
    strong = {
        "churn_signals": {"intent_to_leave": True, "actively_evaluating": True},
        "would_recommend": False,
        "sentiment_trajectory": {"direction": "consistently_negative"},
    }
    none = {
        "churn_signals": {},
        "would_recommend": True,
        "sentiment_trajectory": {"direction": "stable_positive"},
    }
    assert _compute_pain_confidence(weak, "overall_dissatisfaction") == "weak"
    assert _compute_pain_confidence(strong, "overall_dissatisfaction") == "strong"
    assert _compute_pain_confidence(none, "overall_dissatisfaction") == "none"


# ---------------------------------------------------------------------------
# _demote_primary_pain
# ---------------------------------------------------------------------------


def test_demote_primary_pain_replaces_primary_with_fallback():
    result = {
        "pain_categories": [
            {"category": "pricing", "severity": "primary"},
            {"category": "support", "severity": "secondary"},
        ]
    }
    _demote_primary_pain(result, "pricing")
    cats = result["pain_categories"]
    assert cats[0] == {"category": "overall_dissatisfaction", "severity": "primary"}
    # Original primary preserved as secondary so evidence is not lost.
    categories_present = [c["category"] for c in cats]
    assert "pricing" in categories_present
    assert "support" in categories_present
    # No duplicate primary entries
    primaries = [c for c in cats if c.get("severity") == "primary"]
    assert len(primaries) == 1


def test_demote_primary_pain_does_nothing_for_overall_dissatisfaction():
    result = {
        "pain_categories": [
            {"category": "overall_dissatisfaction", "severity": "primary"},
        ]
    }
    _demote_primary_pain(result, "overall_dissatisfaction")
    assert result["pain_categories"] == [
        {"category": "overall_dissatisfaction", "severity": "primary"},
    ]


def test_demote_primary_pain_handles_missing_or_empty_list():
    result = {}
    _demote_primary_pain(result, "pricing")
    cats = result["pain_categories"]
    assert cats[0] == {"category": "overall_dissatisfaction", "severity": "primary"}
    # The demoted pain still gets attached as secondary even when the
    # original list was empty -- preserves the keyword finding.
    assert any(c.get("category") == "pricing" for c in cats)


# ---------------------------------------------------------------------------
# End-to-end through _compute_derived_fields
# ---------------------------------------------------------------------------


def _base_source_row() -> dict:
    return {
        "id": "test-1",
        "vendor_name": "Acme",
        "rating": None,
        "rating_max": 5,
        "summary": "",
        "review_text": "",
        "pros": "",
        "cons": "",
        "raw_metadata": {"source_weight": 0.7},
        "content_type": "review",
    }


def test_compute_derived_fields_demotes_uncorroborated_passing_mention():
    """A review that mentions 'expensive' once but has would_recommend=True,
    no churn signals, and a high rating must NOT keep pain_category=pricing.
    The Layer 3 gate demotes it to overall_dissatisfaction."""
    result = _v2(
        {
            "specific_complaints": ["a bit expensive but worth it"],
            "pricing_phrases": [],
            "feature_gaps": [],
            "quotable_phrases": [],
            "churn_signals": {},
            "would_recommend": True,
            "recommendation_language": ["would recommend"],
            "reviewer_context": {},
            "buyer_authority": {},
            "timeline": {},
            "contract_context": {},
            "urgency_indicators": {},
            "event_mentions": [],
            "sentiment_trajectory": {},
        },
        [
            {
                "field": "specific_complaints",
                "index": 0,
                "subject": "subject_vendor",
                "polarity": "negative",
                "role": "passing_mention",
                "verbatim": True,
            }
        ],
    )
    source = {**_base_source_row(), "rating": 5.0}
    out = _compute_derived_fields(result, source)
    # Original keyword would have produced pain_category="pricing" but
    # confidence=none should demote.
    assert out["pain_category"] == "overall_dissatisfaction"
    assert out["pain_confidence"] in ("weak", "none")
    primaries = [
        c for c in out.get("pain_categories", []) if c.get("severity") == "primary"
    ]
    assert len(primaries) == 1
    assert primaries[0]["category"] == "overall_dissatisfaction"
    # Demoted category preserved as secondary
    secondary_cats = [
        c["category"]
        for c in out.get("pain_categories", [])
        if c.get("severity") == "secondary"
    ]
    assert "pricing" in secondary_cats


def test_compute_derived_fields_keeps_corroborated_pain_category():
    """A review with a clear pain phrase plus an intent_to_leave signal
    must keep pain_category and earn confidence='weak' (1 phrase + 1 signal)."""
    result = _v2(
        {
            "specific_complaints": ["pricing is way too expensive for what we got"],
            "pricing_phrases": [],
            "feature_gaps": [],
            "quotable_phrases": [],
            "churn_signals": {"intent_to_leave": True},
            "would_recommend": False,
            "recommendation_language": ["cannot recommend"],
            "reviewer_context": {},
            "buyer_authority": {},
            "timeline": {},
            "contract_context": {},
            "urgency_indicators": {},
            "event_mentions": [],
            "sentiment_trajectory": {},
        },
        [
            {
                "field": "specific_complaints",
                "index": 0,
                "subject": "subject_vendor",
                "polarity": "negative",
                "role": "primary_driver",
                "verbatim": True,
            }
        ],
    )
    source = {**_base_source_row(), "rating": 1.0}
    out = _compute_derived_fields(result, source)
    assert out["pain_category"] == "pricing"
    # 1 matching phrase + signals (intent_to_leave + would_recommend=False)
    # -> weak (signal_count >= 1 and phrase_count == 1)
    assert out["pain_confidence"] == "weak"


def test_compute_derived_fields_strong_confidence_with_multi_phrase_evidence():
    result = _v2(
        {
            "specific_complaints": [
                "the pricing is a ripoff",
                "pricing keeps going up unexpectedly",
            ],
            "pricing_phrases": ["expensive renewal of $50k"],
            "feature_gaps": [],
            "quotable_phrases": [],
            "churn_signals": {"intent_to_leave": True},
            "would_recommend": False,
            "recommendation_language": ["cannot recommend"],
            "reviewer_context": {},
            "buyer_authority": {},
            "timeline": {},
            "contract_context": {},
            "urgency_indicators": {},
            "event_mentions": [],
            "sentiment_trajectory": {},
        },
        [
            {
                "field": "specific_complaints",
                "index": 0,
                "subject": "subject_vendor",
                "polarity": "negative",
                "role": "primary_driver",
                "verbatim": True,
            },
            {
                "field": "specific_complaints",
                "index": 1,
                "subject": "subject_vendor",
                "polarity": "negative",
                "role": "primary_driver",
                "verbatim": True,
            },
            {
                "field": "pricing_phrases",
                "index": 0,
                "subject": "subject_vendor",
                "polarity": "negative",
                "role": "primary_driver",
                "verbatim": True,
            },
        ],
    )
    source = {**_base_source_row(), "rating": 1.0}
    out = _compute_derived_fields(result, source)
    assert out["pain_category"] == "pricing"
    assert out["pain_confidence"] == "strong"


def test_compute_derived_fields_v1_review_grades_with_keyword_scan():
    """v1 (legacy) reviews still get a pain_confidence value via the
    keyword-scan fallback in _count_pain_phrase_matches."""
    result = {
        "specific_complaints": ["pricing is a ripoff and expensive"],
        "pricing_phrases": [],
        "feature_gaps": [],
        "quotable_phrases": [],
        "churn_signals": {"intent_to_leave": True},
        "would_recommend": False,
        "recommendation_language": [],
        "reviewer_context": {},
        "buyer_authority": {},
        "timeline": {},
        "contract_context": {},
        "urgency_indicators": {},
        "event_mentions": [],
        "sentiment_trajectory": {},
    }
    source = {**_base_source_row(), "rating": 1.0}
    out = _compute_derived_fields(result, source)
    assert out["pain_category"] == "pricing"
    # v1 keyword scan finds 'pricing' and 'expensive' -> strong without
    # needing signals.
    assert out["pain_confidence"] in ("strong", "weak")
