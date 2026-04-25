"""F2: lazy pain_confidence recompute for v3-backed witnesses.

The Slack drilldown after Phase 7 Step B verification showed 11 of 12
v3-backed witnesses with empty pain_confidence column. _validate_enrichment
only triggers _compute_derived_fields when older 'primitives' are
missing (replacement_mode, evidence_spans, etc.); v3 enrichments that
have those primitives but were persisted before Phase 4 wired
pain_confidence will never get the field stamped via that path.

These tests lock the lazy-recompute helper that fills pain_confidence
at the witness build site without mutating the JSONB or forcing a
full _compute_derived_fields pass.
"""

from __future__ import annotations

from atlas_brain.autonomous.tasks._b2b_witnesses import _ensure_pain_confidence


def test_existing_strong_passes_through():
    enrichment = {"pain_confidence": "strong", "pain_category": "pricing"}
    assert _ensure_pain_confidence(enrichment) == "strong"


def test_existing_weak_passes_through():
    enrichment = {"pain_confidence": "weak", "pain_category": "support"}
    assert _ensure_pain_confidence(enrichment) == "weak"


def test_existing_none_passes_through():
    enrichment = {"pain_confidence": "none", "pain_category": "overall_dissatisfaction"}
    assert _ensure_pain_confidence(enrichment) == "none"


def test_unrecognised_value_recomputes():
    """A bogus string in the JSONB should not silently propagate; we
    recompute against the rubric to repair the value."""
    enrichment = {
        "pain_confidence": "garbage_value",
        "pain_category": "overall_dissatisfaction",
        "churn_signals": {},
        "would_recommend": True,
        "sentiment_trajectory": {"direction": "stable_positive"},
    }
    # No corroborating signals + non-existent pain pattern -> 'none'
    assert _ensure_pain_confidence(enrichment) == "none"


def test_missing_field_v3_pattern_recomputes_strong():
    """A v3 enrichment with no pain_confidence stamped, multiple
    keyword-matching phrases, and corroborating churn signals should
    recompute to 'strong' on the fly."""
    enrichment = {
        "pain_category": "pricing",
        "specific_complaints": [
            "pricing keeps going up dramatically",
            "another expensive complaint about pricing",
        ],
        "churn_signals": {"intent_to_leave": True},
        "would_recommend": False,
        "sentiment_trajectory": {"direction": "consistently_negative"},
    }
    assert _ensure_pain_confidence(enrichment) == "strong"


def test_missing_field_no_signals_recomputes_none():
    """v3 enrichment with no pain_confidence and no churn / sentiment
    corroboration on overall_dissatisfaction -> 'none'."""
    enrichment = {
        "pain_category": "overall_dissatisfaction",
        "churn_signals": {},
        "would_recommend": True,
        "sentiment_trajectory": {"direction": "stable_positive"},
    }
    assert _ensure_pain_confidence(enrichment) == "none"


def test_missing_field_one_signal_recomputes_weak():
    """overall_dissatisfaction + 1 corroborating signal -> 'weak'."""
    enrichment = {
        "pain_category": "overall_dissatisfaction",
        "churn_signals": {"intent_to_leave": True},
        "would_recommend": None,
        "sentiment_trajectory": {"direction": "unknown"},
    }
    assert _ensure_pain_confidence(enrichment) == "weak"


def test_does_not_mutate_input():
    """The helper must be pure -- callers may share the enrichment dict
    across multiple witness candidates per review and we don't want
    a stale recompute to bleed into other downstream readers."""
    enrichment = {
        "pain_category": "pricing",
        "specific_complaints": ["pricing too high"],
        "churn_signals": {"intent_to_leave": True},
    }
    before = dict(enrichment)
    _ensure_pain_confidence(enrichment)
    assert enrichment == before


def test_legacy_v3_with_zero_phrases_returns_none():
    """v3 enrichment with no pain phrases at all (e.g., a no_signal
    review that somehow ended up enriched) returns 'none' rather than
    crashing or returning None."""
    enrichment = {
        "pain_category": "overall_dissatisfaction",
        "churn_signals": {},
    }
    result = _ensure_pain_confidence(enrichment)
    assert result == "none"


def test_v4_passing_through_when_already_set():
    """v4 enrichments persisted post-Phase-4 should have
    pain_confidence already stamped; recompute is a no-op."""
    enrichment = {
        "enrichment_schema_version": 4,
        "phrase_metadata": [],
        "pain_confidence": "strong",
        "pain_category": "support",
    }
    assert _ensure_pain_confidence(enrichment) == "strong"
