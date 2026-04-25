"""Phase 5a (witness-level wiring) tests.

These cover:
  - phrase tags (subject / role / verbatim) propagated from v2 metadata
    onto evidence spans
  - candidate witness dicts carrying phrase_polarity, phrase_subject,
    phrase_role, phrase_verbatim, pain_confidence
  - _candidate_types blocking common_pattern when role=passing_mention
  - compute_witness_hash incorporating the new tags so updates propagate
  - _witness_row_payload emitting the new fields for downstream persistence
"""

from __future__ import annotations

from atlas_brain.autonomous.tasks._b2b_witnesses import (
    _candidate_types,
    build_vendor_witness_artifacts,
    compute_witness_hash,
    derive_evidence_spans,
)
from atlas_brain.autonomous.tasks.b2b_reasoning_synthesis import (
    _witness_row_payload,
)


# ---------------------------------------------------------------------------
# Span / candidate propagation
# ---------------------------------------------------------------------------


def _v2_enrichment(extra: dict | None = None, *, phrase_metadata=None) -> dict:
    base = {
        "enrichment_schema_version": 4,
        "specific_complaints": ["pricing keeps going up unexpectedly"],
        "pricing_phrases": [],
        "feature_gaps": [],
        "quotable_phrases": [],
        "recommendation_language": [],
        "positive_aspects": [],
        "competitors_mentioned": [],
        "event_mentions": [],
        "churn_signals": {"intent_to_leave": True},
        "would_recommend": False,
        "sentiment_trajectory": {"direction": "consistently_negative"},
        "reviewer_context": {"decision_maker": True, "company_name": "Acme"},
        "buyer_authority": {},
        "timeline": {},
        "contract_context": {},
        "urgency_indicators": {},
        "pain_category": "pricing",
        "pain_categories": [{"category": "pricing", "severity": "primary"}],
        "pain_confidence": "weak",
        "phrase_metadata": phrase_metadata
        or [
            {
                "field": "specific_complaints",
                "index": 0,
                "subject": "subject_vendor",
                "polarity": "negative",
                "role": "primary_driver",
                "verbatim": True,
            }
        ],
    }
    if extra:
        base.update(extra)
    return base


def _source_row() -> dict:
    return {
        "id": "review-1",
        "vendor_name": "Acme",
        "summary": "Acme review",
        "review_text": "Pricing keeps going up unexpectedly and I cannot recommend.",
        "pros": "",
        "cons": "",
        "rating": 1.0,
        "rating_max": 5,
        "raw_metadata": {},
        "reviewer_company": "Globex",
        "reviewer_title": "VP Engineering",
        "source": "g2",
        "reviewed_at": "2026-04-01T00:00:00+00:00",
    }


def test_derive_evidence_spans_v2_propagates_phrase_tags_to_span():
    enrichment = _v2_enrichment()
    spans = derive_evidence_spans(enrichment, _source_row())
    assert spans, "expected at least one span"
    span = spans[0]
    assert span["polarity"] == "negative"
    assert span["subject"] == "subject_vendor"
    assert span["role"] == "primary_driver"
    assert span["verbatim"] is True


def test_derive_evidence_spans_v1_leaves_phrase_tags_none():
    enrichment = {
        "specific_complaints": ["pricing keeps going up"],
        "pricing_phrases": [],
        "feature_gaps": [],
        "quotable_phrases": [],
        "recommendation_language": [],
        "positive_aspects": [],
        "competitors_mentioned": [],
        "event_mentions": [],
        "churn_signals": {},
        "reviewer_context": {},
        "buyer_authority": {},
        "timeline": {},
        "contract_context": {},
        "pain_category": "pricing",
    }
    spans = derive_evidence_spans(enrichment, _source_row())
    assert spans
    span = spans[0]
    assert span.get("polarity") is None
    assert span.get("subject") is None
    assert span.get("role") is None
    assert span.get("verbatim") is None


def test_build_vendor_witness_artifacts_carries_phrase_tags_and_pain_confidence():
    enrichment = _v2_enrichment()
    review = {**_source_row(), "enrichment": enrichment}
    selected, _ = build_vendor_witness_artifacts(
        "Acme",
        [review],
        max_witnesses=4,
    )
    assert selected, "expected at least one witness"
    candidate = selected[0]
    assert candidate.get("phrase_polarity") == "negative"
    assert candidate.get("phrase_subject") == "subject_vendor"
    assert candidate.get("phrase_role") == "primary_driver"
    assert candidate.get("phrase_verbatim") is True
    assert candidate.get("pain_confidence") == "weak"


def test_build_vendor_witness_artifacts_carries_selection_time_grounding_status():
    enrichment = _v2_enrichment()
    review = {**_source_row(), "summary": "", "enrichment": enrichment}
    selected, _ = build_vendor_witness_artifacts(
        "Acme",
        [review],
        max_witnesses=4,
    )
    assert selected, "expected at least one witness"
    assert selected[0].get("grounding_status") == "grounded"


# ---------------------------------------------------------------------------
# _candidate_types role gate
# ---------------------------------------------------------------------------


def test_candidate_types_blocks_common_pattern_when_role_is_passing_mention():
    review = {
        "reviewer_company": "Globex",
        "rating": 1,
        "rating_max": 5,
        "raw_metadata": {},
    }
    enrichment = {
        "reviewer_context": {"decision_maker": True},
        "churn_signals": {},
    }
    span = {
        "pain_category": "pricing",
        "competitor": None,
        "flags": [],
        "polarity": "negative",
        "role": "passing_mention",
        "signal_type": "complaint",
    }
    types = _candidate_types(review, enrichment, span, primary_pains={"pricing"})
    assert "common_pattern" not in types
    # other types unaffected
    assert "named_account" in types
    assert "flex" in types


def test_candidate_types_keeps_common_pattern_for_primary_driver_role():
    review = {
        "reviewer_company": "Globex",
        "rating": 1,
        "rating_max": 5,
        "raw_metadata": {},
    }
    enrichment = {
        "reviewer_context": {"decision_maker": True},
        "churn_signals": {},
    }
    span = {
        "pain_category": "pricing",
        "competitor": None,
        "flags": [],
        "polarity": "negative",
        "role": "primary_driver",
        "signal_type": "complaint",
    }
    types = _candidate_types(review, enrichment, span, primary_pains={"pricing"})
    assert "common_pattern" in types


def test_candidate_types_v1_span_with_none_role_keeps_common_pattern():
    """v1 spans have no role tag -- common_pattern must remain available."""
    review = {
        "reviewer_company": "Globex",
        "rating": 1,
        "rating_max": 5,
        "raw_metadata": {},
    }
    enrichment = {
        "reviewer_context": {"decision_maker": True},
        "churn_signals": {},
    }
    span = {
        "pain_category": "pricing",
        "competitor": None,
        "flags": [],
        "signal_type": "complaint",
    }
    types = _candidate_types(review, enrichment, span, primary_pains={"pricing"})
    assert "common_pattern" in types


# ---------------------------------------------------------------------------
# witness_hash sensitivity
# ---------------------------------------------------------------------------


def test_compute_witness_hash_changes_when_phrase_polarity_changes():
    base = {
        "witness_id": "w1",
        "review_id": "r1",
        "excerpt_text": "pricing keeps going up",
        "source": "g2",
        "signal_type": "complaint",
        "pain_category": "pricing",
        "competitor": None,
        "time_anchor": None,
        "replacement_mode": None,
        "operating_model_shift": None,
        "productivity_delta_claim": None,
        "signal_tags": [],
    }
    h1 = compute_witness_hash(base)
    h2 = compute_witness_hash({**base, "phrase_polarity": "negative"})
    assert h1 != h2


def test_compute_witness_hash_changes_when_pain_confidence_changes():
    base = {
        "witness_id": "w1",
        "review_id": "r1",
        "excerpt_text": "pricing keeps going up",
        "source": "g2",
        "signal_type": "complaint",
        "pain_category": "pricing",
        "competitor": None,
        "time_anchor": None,
        "replacement_mode": None,
        "operating_model_shift": None,
        "productivity_delta_claim": None,
        "signal_tags": [],
        "phrase_polarity": "negative",
    }
    h1 = compute_witness_hash({**base, "pain_confidence": "weak"})
    h2 = compute_witness_hash({**base, "pain_confidence": "strong"})
    assert h1 != h2


def test_compute_witness_hash_changes_when_grounding_status_changes():
    base = {
        "witness_id": "w1",
        "review_id": "r1",
        "excerpt_text": "pricing keeps going up",
        "source": "g2",
        "signal_type": "complaint",
        "pain_category": "pricing",
        "competitor": None,
        "time_anchor": None,
        "replacement_mode": None,
        "operating_model_shift": None,
        "productivity_delta_claim": None,
        "signal_tags": [],
    }
    h1 = compute_witness_hash({**base, "grounding_status": "grounded"})
    h2 = compute_witness_hash({**base, "grounding_status": "not_grounded"})
    assert h1 != h2


def test_compute_witness_hash_stable_for_legacy_witness_without_phrase_tags():
    """Existing v1 witness rows with no phrase tags must still produce
    deterministic hashes -- otherwise the unchanged-skip path would
    constantly rewrite all of them."""
    base = {
        "witness_id": "w1",
        "review_id": "r1",
        "excerpt_text": "pricing keeps going up",
        "source": "g2",
        "signal_type": "complaint",
        "pain_category": "pricing",
        "competitor": None,
        "time_anchor": None,
        "replacement_mode": None,
        "operating_model_shift": None,
        "productivity_delta_claim": None,
        "signal_tags": [],
    }
    assert compute_witness_hash(base) == compute_witness_hash({**base})


# ---------------------------------------------------------------------------
# _witness_row_payload
# ---------------------------------------------------------------------------


def test_witness_row_payload_exposes_phrase_tags_and_pain_confidence():
    witness = {
        "witness_id": "w1",
        "review_id": "r1",
        "witness_type": "common_pattern",
        "excerpt_text": "pricing keeps going up",
        "source": "g2",
        "reviewed_at": "2026-04-01T00:00:00+00:00",
        "reviewer_company": "Globex",
        "reviewer_title": "VP Engineering",
        "pain_category": "pricing",
        "competitor": None,
        "salience_score": 1.5,
        "selection_reason": "selected_for_common_pattern",
        "signal_tags": [],
        "_sid": "w1",
        "specificity_score": 2.0,
        "generic_reason": None,
        "witness_hash": "abc123",
        "source_span_id": "review:r1:span:0-10",
        "grounding_status": "grounded",
        "phrase_polarity": "negative",
        "phrase_subject": "subject_vendor",
        "phrase_role": "primary_driver",
        "phrase_verbatim": True,
        "pain_confidence": "strong",
    }
    payload = _witness_row_payload(witness)
    assert payload["phrase_polarity"] == "negative"
    assert payload["phrase_subject"] == "subject_vendor"
    assert payload["phrase_role"] == "primary_driver"
    assert payload["phrase_verbatim"] is True
    assert payload["pain_confidence"] == "strong"


def test_witness_row_payload_returns_none_for_missing_phrase_tags():
    witness = {
        "witness_id": "w1",
        "review_id": "r1",
        "witness_type": "flex",
        "excerpt_text": "some excerpt",
        "source": "g2",
        "reviewed_at": "2026-04-01T00:00:00+00:00",
        "reviewer_company": None,
        "reviewer_title": None,
        "pain_category": None,
        "competitor": None,
        "salience_score": 0.0,
        "selection_reason": "selected_for_flex",
        "signal_tags": [],
        "_sid": "w1",
        "specificity_score": 0.0,
        "generic_reason": None,
        "witness_hash": "def456",
        "source_span_id": None,
        "grounding_status": "pending",
    }
    payload = _witness_row_payload(witness)
    assert payload["phrase_polarity"] is None
    assert payload["phrase_subject"] is None
    assert payload["phrase_role"] is None
    assert payload["phrase_verbatim"] is None
    assert payload["pain_confidence"] is None
