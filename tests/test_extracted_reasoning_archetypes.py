from __future__ import annotations

from extracted_content_pipeline.reasoning.archetypes import (
    ARCHETYPES,
    best_match,
    enrich_evidence_with_archetypes,
    get_archetype,
    get_falsification_conditions,
    score_evidence,
    top_matches,
)


def _pricing_evidence(**overrides):
    evidence = {
        "avg_urgency": "7.2",
        "top_pain": "Renewal pricing jumped and the invoice became too expensive",
        "competitor_count": 4,
        "recommend_ratio": 0.2,
        "displacement_edge_count": 3,
        "positive_review_pct": "28%",
    }
    evidence.update(overrides)
    return evidence


def test_best_match_identifies_pricing_shock_from_mixed_numeric_shapes():
    match = best_match(_pricing_evidence())

    assert match is not None
    assert match.archetype == "pricing_shock"
    assert match.score >= 0.7
    assert match.risk_level == "high"
    assert set(match.matched_signals) >= {
        "avg_urgency",
        "top_pain",
        "competitor_count",
        "recommend_ratio",
        "displacement_edge_count",
        "positive_review_pct",
    }


def test_feature_gap_wins_when_missing_capability_signals_dominate():
    evidence = {
        "top_pain": "Customers keep asking for a missing workflow feature on the roadmap",
        "competitor_count": 3,
        "displacement_edge_count": 3,
        "pain_count": 5,
        "recommend_ratio": 0.25,
    }

    match = best_match(evidence)

    assert match is not None
    assert match.archetype == "feature_gap"
    assert "top_pain" in match.matched_signals


def test_keyword_matching_reads_nested_evidence_text():
    evidence = {
        "avg_urgency": 8,
        "high_intent_company_count": 4,
        "churn_density": 0.7,
        "weakness_evidence": [
            {
                "best_quote": "The API connector broke after the migration",
                "theme": "integration",
            }
        ],
    }

    match = best_match(evidence)

    assert match is not None
    assert match.archetype == "integration_break"


def test_score_evidence_returns_all_archetypes_sorted():
    matches = score_evidence(_pricing_evidence())

    assert len(matches) == len(ARCHETYPES)
    assert matches[0].score >= matches[-1].score
    assert matches[0].archetype == "pricing_shock"


def test_empty_or_malformed_evidence_does_not_emit_threshold_matches():
    evidence = {
        "avg_urgency": "not-a-number",
        "competitor_count": None,
        "top_pain": "   ",
    }

    assert best_match(evidence) is None
    assert top_matches(evidence) == []
    assert all(match.score == 0 for match in score_evidence(evidence))


def test_top_matches_honors_limit_and_threshold():
    matches = top_matches(_pricing_evidence(), limit=1)

    assert len(matches) == 1
    assert matches[0].archetype == "pricing_shock"
    assert top_matches(_pricing_evidence(), limit=0) == []


def test_temporal_velocity_increases_aligned_match_score():
    evidence = _pricing_evidence(competitor_count=3, recommend_ratio=0.35)
    base = next(match for match in score_evidence(evidence) if match.archetype == "pricing_shock")
    temporal = {
        "velocity_avg_urgency": 0.8,
        "accel_avg_urgency": 0.2,
        "velocity_competitor_count": 2.0,
        "velocity_recommend_ratio": -0.1,
    }

    boosted = next(
        match
        for match in score_evidence(evidence, temporal)
        if match.archetype == "pricing_shock"
    )

    assert boosted.score > base.score


def test_anomaly_bonus_increases_matching_numeric_signal_score():
    evidence = _pricing_evidence()
    base = next(match for match in score_evidence(evidence) if match.archetype == "pricing_shock")
    with_anomaly = dict(evidence)
    with_anomaly["anomalies"] = [{"metric": "competitor_count", "z_score": "2.4"}]

    boosted = next(
        match
        for match in score_evidence(with_anomaly)
        if match.archetype == "pricing_shock"
    )

    assert boosted.score > base.score


def test_enrich_evidence_adds_archetype_scores_without_mutating_input():
    evidence = _pricing_evidence()

    enriched = enrich_evidence_with_archetypes(evidence)

    assert "archetype_scores" not in evidence
    assert enriched["archetype_scores"][0]["archetype"] == "pricing_shock"
    assert enriched["archetype_scores"][0]["signal_score"] >= 0.7
    assert "matched_signals" in enriched["archetype_scores"][0]


def test_enrich_evidence_leaves_no_match_payload_unannotated():
    enriched = enrich_evidence_with_archetypes({"vendor_name": "NoSignal"})

    assert enriched == {"vendor_name": "NoSignal"}


def test_lookup_helpers_return_profiles_and_copy_falsification_lists():
    profile = get_archetype("compliance_gap")
    conditions = get_falsification_conditions("compliance_gap")

    assert profile is not None
    assert profile.typical_risk == "critical"
    assert conditions

    conditions.append("mutated")
    assert "mutated" not in get_falsification_conditions("compliance_gap")
    assert get_archetype("unknown") is None
    assert get_falsification_conditions("unknown") == []
