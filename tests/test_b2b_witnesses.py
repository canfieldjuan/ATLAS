from atlas_brain.autonomous.tasks._b2b_witnesses import (
    build_vendor_witness_artifacts,
    derive_evidence_spans,
    derive_operating_model_shift,
    derive_org_pressure_type,
    derive_productivity_delta_claim,
    derive_salience_flags,
)


def _source_row(review_text: str, **overrides):
    row = {
        "id": "review-1",
        "summary": "",
        "review_text": review_text,
        "pros": "",
        "cons": "",
        "reviewer_title": "",
        "reviewer_company": "",
    }
    row.update(overrides)
    return row


def test_derive_productivity_delta_claim_recognizes_time_savings_language():
    row = _source_row("This saved time for the team and reduced manual work every week.")

    assert derive_productivity_delta_claim(row) == "more_productive"


def test_derive_productivity_delta_claim_recognizes_streamline_and_manual_entry_language():
    row = _source_row(
        "Its unified UI/UX transforms complex HR and IT tasks into a single, automated workflow, "
        "eliminating manual data entry and reducing administrative complexity."
    )

    assert derive_productivity_delta_claim(row) == "more_productive"


def test_derive_operating_model_shift_recognizes_all_in_one_source_of_truth_language():
    row = _source_row(
        "We use this as our all-in-one platform and single source of truth for cross-functional work."
    )

    assert derive_operating_model_shift({}, row) == "consolidation"


def test_derive_org_pressure_type_recognizes_procurement_review_language():
    row = _source_row("We had to pass security review and get on the approved software list.")

    assert derive_org_pressure_type(row) == "procurement_mandate"


def test_derive_evidence_spans_uses_decision_timeline_as_time_anchor():
    result = {
        "specific_complaints": ["Renewal pricing is too high"],
        "pricing_phrases": ["Renewal pricing is too high"],
        "feature_gaps": [],
        "recommendation_language": [],
        "positive_aspects": [],
        "competitors_mentioned": [],
        "event_mentions": [],
        "pain_category": "pricing",
        "timeline": {"decision_timeline": "within_quarter"},
        "churn_signals": {},
        "reviewer_context": {"company_name": "Acme Co"},
        "budget_signals": {},
    }
    row = _source_row("Renewal pricing is too high and we need an answer this quarter.")

    spans = derive_evidence_spans(result, row)

    assert spans
    assert spans[0]["time_anchor"] == "within_quarter"


def test_derive_salience_flags_uses_nested_company_and_decision_timeline():
    result = {
        "churn_signals": {},
        "budget_signals": {},
        "reviewer_context": {"company_name": "Acme Co", "decision_maker": False},
        "timeline": {"decision_timeline": "within_quarter"},
        "competitors_mentioned": [],
    }
    row = _source_row("We need to decide this quarter.")

    flags = derive_salience_flags(result, row)

    assert "named_account" in flags
    assert "renewal_window" in flags


def test_derive_salience_flags_requires_pricing_context_for_explicit_dollar():
    result = {
        "churn_signals": {},
        "budget_signals": {},
        "reviewer_context": {},
        "timeline": {},
        "competitors_mentioned": [],
        "pricing_phrases": [],
    }
    row = _source_row("Copper recently reached $13,000 per ton in commodity markets.")

    flags = derive_salience_flags(result, row)

    assert "explicit_dollar" not in flags


def test_derive_salience_flags_keeps_explicit_dollar_for_real_pricing_signal():
    result = {
        "churn_signals": {"contract_renewal_mentioned": True},
        "budget_signals": {},
        "reviewer_context": {},
        "timeline": {},
        "competitors_mentioned": [],
        "pricing_phrases": ["quoted $29 per seat at renewal"],
    }
    row = _source_row("We were quoted $29 per seat at renewal and the cost is too high.")

    flags = derive_salience_flags(result, row)

    assert "explicit_dollar" in flags


def test_derive_evidence_spans_only_marks_explicit_dollar_for_pricing_context():
    result = {
        "specific_complaints": ["Copper recently reached $13,000 per ton in commodity markets."],
        "pricing_phrases": [],
        "feature_gaps": [],
        "recommendation_language": [],
        "positive_aspects": [],
        "competitors_mentioned": [],
        "event_mentions": [],
        "pain_category": "pricing",
        "timeline": {},
        "churn_signals": {},
        "reviewer_context": {},
        "budget_signals": {},
    }
    row = _source_row("Copper recently reached $13,000 per ton in commodity markets.")

    spans = derive_evidence_spans(result, row)

    assert spans
    assert "explicit_dollar" not in spans[0]["flags"]


def test_positive_anchor_excerpt_uses_exact_phrase_without_adjacent_negative_context():
    result = {
        "specific_complaints": ["a little too expensive"],
        "pricing_phrases": ["a little too expensive"],
        "feature_gaps": [],
        "recommendation_language": [],
        "positive_aspects": ["it was helpful"],
        "competitors_mentioned": [],
        "event_mentions": [],
        "pain_category": "pricing",
        "timeline": {},
        "churn_signals": {},
        "reviewer_context": {},
        "budget_signals": {},
    }
    row = _source_row(
        "I used Monday few years ago, it was helpful but a little too expensive."
    )

    spans = derive_evidence_spans(result, row)

    positive_anchor = next(span for span in spans if span["signal_type"] == "positive_anchor")
    assert positive_anchor["text"] == "it was helpful"


def test_derive_evidence_spans_prefers_support_when_complaint_mentions_renewal_and_support():
    result = {
        "specific_complaints": ["talked to a robot who could not help me and support never replied before renewal"],
        "pricing_phrases": [],
        "feature_gaps": [],
        "recommendation_language": [],
        "positive_aspects": [],
        "competitors_mentioned": [],
        "event_mentions": [],
        "pain_category": "pricing",
        "timeline": {},
        "churn_signals": {},
        "reviewer_context": {},
        "budget_signals": {},
    }
    row = _source_row("At Acme I talked to a robot who could not help me and support never replied before renewal.")

    spans = derive_evidence_spans(result, row)

    complaint = next(span for span in spans if span["signal_type"] == "complaint")
    assert complaint["pain_category"] == "support"


def test_derive_evidence_spans_prefers_privacy_for_unsubscribe_complaint():
    result = {
        "specific_complaints": ["getting sales emails that I can't unsubscribe from"],
        "pricing_phrases": [],
        "feature_gaps": [],
        "recommendation_language": [],
        "positive_aspects": [],
        "competitors_mentioned": [],
        "event_mentions": [],
        "pain_category": "pricing",
        "timeline": {},
        "churn_signals": {},
        "reviewer_context": {},
        "budget_signals": {},
    }
    row = _source_row("I did not subscribe but I am still getting sales emails that I can't unsubscribe from.")

    spans = derive_evidence_spans(result, row)

    complaint = next(span for span in spans if span["signal_type"] == "complaint")
    assert complaint["pain_category"] == "privacy"


def test_mixed_negative_review_does_not_produce_counterevidence_witness():
    reviews = [{
        "id": "r1",
        "source": "trustpilot",
        "review_text": (
            "I used Monday few years ago, it was helpful but a little too expensive. "
            "Few days ago I visited monday.com to take a look as I am looking for similar app. "
            "I did not subscribe or give my email but still I'm getting sales emails that I can't unsubscribe from. "
            "Tried going to the site again and talked to a robot who could not help me and if I wanted to talk to a human "
            "I needed to start a 14day free trial. Needless to say I will never use their service and recommend staying away."
        ),
        "reviewed_at": "2026-03-29T09:47:13+00:00",
        "rating": 1,
        "enrichment": {
            "pain_categories": [{"category": "pricing", "severity": "primary"}],
            "would_recommend": False,
            "sentiment_trajectory": {"direction": "consistently_negative"},
            "churn_signals": {
                "intent_to_leave": True,
                "actively_evaluating": True,
                "migration_in_progress": False,
                "contract_renewal_mentioned": False,
            },
            "recommendation_language": [
                "I will never use their service",
                "recommend staying away",
            ],
            "evidence_spans": [
                {
                    "text": "it was helpful",
                    "pain_category": None,
                    "flags": [],
                    "signal_type": "positive_anchor",
                },
                {
                    "text": "talked to a robot who could not help me",
                    "pain_category": "support",
                    "flags": [],
                    "signal_type": "complaint",
                    "time_anchor": "few days",
                },
            ],
        },
    }]

    selected, section_packets = build_vendor_witness_artifacts(
        "Monday.com",
        reviews,
        fallback_min_witnesses=0,
    )

    assert all(item["witness_type"] != "counterevidence" for item in selected)
    assert section_packets["retention_packet"]["witness_ids"] == []
    assert section_packets["anchor_examples"]["counterevidence"] == []


def test_positive_review_can_surface_counterevidence_witness():
    reviews = [{
        "id": "r1",
        "source": "g2",
        "review_text": (
            "The team stayed with the product because the support team is excellent and "
            "the workflow is reliable for daily use."
        ),
        "reviewed_at": "2026-03-29T09:47:13+00:00",
        "rating": 5,
        "enrichment": {
            "pain_categories": [{"category": "pricing", "severity": "primary"}],
            "would_recommend": True,
            "sentiment_trajectory": {"direction": "stable_positive"},
            "churn_signals": {
                "intent_to_leave": False,
                "actively_evaluating": False,
                "migration_in_progress": False,
                "contract_renewal_mentioned": False,
            },
            "recommendation_language": [],
            "evidence_spans": [
                {
                    "text": "the support team is excellent and the workflow is reliable for daily use",
                    "pain_category": "support",
                    "flags": [],
                    "signal_type": "positive_anchor",
                    "productivity_delta_claim": "more_productive",
                },
            ],
        },
    }]

    selected, section_packets = build_vendor_witness_artifacts(
        "PositiveVendor",
        reviews,
        fallback_min_witnesses=0,
    )

    assert any(item["witness_type"] == "counterevidence" for item in selected)
    assert len(section_packets["retention_packet"]["witness_ids"]) == 1


def test_witness_builder_prefers_current_derived_spans_over_stale_persisted_spans():
    reviews = [{
        "id": "r3",
        "source": "trustpilot",
        "review_text": "At Acme I talked to a robot who could not help me and support never replied before renewal.",
        "reviewed_at": "2026-03-29T09:47:13+00:00",
        "rating": 1,
        "reviewer_company": "Acme",
        "enrichment": {
            "pain_category": "pricing",
            "pain_categories": [{"category": "support", "severity": "primary"}],
            "specific_complaints": ["talked to a robot who could not help me and support never replied before renewal"],
            "pricing_phrases": [],
            "recommendation_language": [],
            "positive_aspects": [],
            "competitors_mentioned": [],
            "event_mentions": [],
            "evidence_spans": [
                {
                    "text": "talked to a robot who could not help me and support never replied before renewal",
                    "pain_category": "pricing",
                    "signal_type": "complaint",
                    "flags": [],
                }
            ],
        },
    }]

    selected, section_packets = build_vendor_witness_artifacts(
        "TestVendor",
        reviews,
        fallback_min_witnesses=0,
    )

    assert any(item["pain_category"] == "support" for item in selected)
    gov = section_packets["_witness_governance"]
    assert gov["spans_persisted"] == 1
    assert gov["spans_fallback"] == 0
    assert gov["spans_refreshed"] == 1


# -- Witness fallback visibility tests --


def _review_with_spans():
    """Review with persisted evidence_spans."""
    return {
        "id": "rev-1",
        "enrichment": {
            "churn_signals": {"intent_to_leave": True},
            "reviewer_context": {"decision_maker": True},
            "specific_complaints": ["Too expensive"],
            "quotable_phrases": ["We are switching"],
            "evidence_spans": [
                {
                    "text": "We are switching to a competitor",
                    "pain_category": "pricing",
                    "signal_type": "churn",
                    "flags": ["explicit_cancel"],
                    "replacement_mode": "vendor_switch",
                    "operating_model_shift": "none",
                    "productivity_delta_claim": "unknown",
                }
            ],
            "salience_flags": ["explicit_cancel"],
            "replacement_mode": "vendor_switch",
            "urgency_score": 8,
        },
        "review_text": "We are switching to a competitor because pricing is too high.",
        "reviewer_company": "Acme",
        "reviewer_title": "VP Ops",
    }


def _review_without_spans():
    """Review missing persisted evidence_spans -- triggers fallback."""
    return {
        "id": "rev-2",
        "enrichment": {
            "churn_signals": {"intent_to_leave": True},
            "reviewer_context": {},
            "specific_complaints": ["Buggy product"],
            "quotable_phrases": [],
            "urgency_score": 6,
        },
        "review_text": "Buggy product, we are evaluating alternatives.",
        "reviewer_company": "Beta Inc",
        "reviewer_title": "Manager",
    }


def test_witness_governance_tracks_persisted_spans():
    reviews = [_review_with_spans()]
    _, packets = build_vendor_witness_artifacts("TestVendor", reviews)
    gov = packets["_witness_governance"]
    assert gov["spans_persisted"] == 1
    assert gov["spans_fallback"] == 0
    assert gov["spans_refreshed"] == 1


def test_witness_governance_tracks_fallback_spans():
    reviews = [_review_without_spans()]
    _, packets = build_vendor_witness_artifacts("TestVendor", reviews)
    gov = packets["_witness_governance"]
    assert gov["spans_persisted"] == 0
    assert gov["spans_fallback"] == 1
    assert gov["spans_refreshed"] == 0


def test_witness_governance_mixed_persisted_and_fallback():
    reviews = [_review_with_spans(), _review_without_spans()]
    _, packets = build_vendor_witness_artifacts("TestVendor", reviews)
    gov = packets["_witness_governance"]
    assert gov["spans_persisted"] == 1
    assert gov["spans_fallback"] == 1
    assert gov["spans_refreshed"] == 1
