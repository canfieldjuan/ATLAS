"""Tests for the declarative evidence evaluation engine."""

from __future__ import annotations

import pytest
from pathlib import Path

from atlas_brain.reasoning.evidence_engine import EvidenceEngine, ConclusionResult, SuppressionResult


@pytest.fixture
def engine():
    return EvidenceEngine()


# -- Urgency Scoring ------------------------------------------------------


class TestUrgencyScoring:

    def test_zero_indicators_zero_score(self, engine):
        score = engine.compute_urgency({}, rating=None, rating_max=5, content_type="review", source_weight=0.7)
        assert score == 0.0

    def test_all_tier_a_indicators(self, engine):
        indicators = {
            "explicit_cancel_language": True,
            "active_migration_language": True,
            "active_evaluation_language": True,
            "completed_switch_language": True,
        }
        score = engine.compute_urgency(indicators, rating=None, rating_max=5, content_type="review", source_weight=0.7)
        # 3.0 + 2.5 + 2.5 + 3.0 = 11.0, clamped to 10
        assert score == 10.0

    def test_all_indicators_clamped_at_10(self, engine):
        indicators = {k: True for k in [
            "explicit_cancel_language", "active_migration_language",
            "active_evaluation_language", "completed_switch_language",
            "comparison_shopping_language", "named_alternative_with_reason",
            "frustration_without_alternative", "dollar_amount_mentioned",
            "timeline_mentioned", "decision_maker_language",
        ]}
        score = engine.compute_urgency(indicators, rating=None, rating_max=5, content_type="review", source_weight=0.7)
        assert score == 10.0

    def test_single_supporting_indicator(self, engine):
        score = engine.compute_urgency(
            {"frustration_without_alternative": True},
            rating=None, rating_max=5, content_type="review", source_weight=0.7,
        )
        # frustration_without_alternative = 1.5 in current YAML
        assert score == 1.5

    def test_rating_floor_one_star(self, engine):
        score = engine.compute_urgency({}, rating=1.0, rating_max=5.0, content_type="review", source_weight=0.7)
        assert score == 3.0

    def test_rating_floor_two_star(self, engine):
        score = engine.compute_urgency({}, rating=2.0, rating_max=5.0, content_type="review", source_weight=0.7)
        assert score == 2.0

    def test_rating_floor_does_not_lower_score(self, engine):
        indicators = {"explicit_cancel_language": True, "active_evaluation_language": True}
        score = engine.compute_urgency(indicators, rating=2.0, rating_max=5.0, content_type="review", source_weight=0.7)
        # 3.0 + 2.0 = 5.0, floor is 2.0 -- should keep 5.0
        assert score == 5.0

    def test_comment_adjustment(self, engine):
        score = engine.compute_urgency(
            {"frustration_without_alternative": True},
            rating=None, rating_max=5, content_type="comment", source_weight=0.7,
        )
        # 1.5 - 1.0 = 0.5
        assert score == 0.5

    def test_low_source_weight_gate(self, engine):
        indicators = {"explicit_cancel_language": True, "active_migration_language": True}
        score = engine.compute_urgency(indicators, rating=None, rating_max=5, content_type="review", source_weight=0.2)
        assert score == 0.0

    def test_moderate_signals_spread(self, engine):
        # Two moderate indicators should produce a meaningful mid-range score
        indicators = {
            "comparison_shopping_language": True,
            "named_alternative_with_reason": True,
        }
        score = engine.compute_urgency(indicators, rating=None, rating_max=5, content_type="review", source_weight=0.7)
        # 2.0 + 2.0 = 4.0
        assert score == 4.0

    def test_mixed_tiers(self, engine):
        indicators = {
            "active_evaluation_language": True,  # +2.5
            "timeline_mentioned": True,          # +1.5
            "decision_maker_language": True,      # +1.0
        }
        score = engine.compute_urgency(indicators, rating=None, rating_max=5, content_type="review", source_weight=0.7)
        # 2.5 + 1.5 + 1.0 = 5.0
        assert score == 5.0


# -- Pain Override --------------------------------------------------------


class TestPainOverride:

    def test_non_other_passes_through(self, engine):
        assert engine.override_pain("pricing", ["too expensive"]) == "pricing"

    def test_other_with_pricing_keywords(self, engine):
        result = engine.override_pain("other", ["The cost is way too expensive for what you get"])
        assert result == "pricing"

    def test_overall_dissatisfaction_triggers_override(self, engine):
        result = engine.override_pain(
            "overall_dissatisfaction",
            ["We need a dedicated Salesforce admin just to maintain workflows"],
        )
        assert result == "admin_burden"

    def test_other_with_support_keywords(self, engine):
        result = engine.override_pain("other", ["customer service is terrible", "ticket response time is awful"])
        assert result == "support"

    def test_other_with_no_keywords(self, engine):
        result = engine.override_pain("other", ["just not great overall"])
        assert result == "overall_dissatisfaction"

    def test_other_with_empty_complaints(self, engine):
        result = engine.override_pain("other", [])
        assert result == "overall_dissatisfaction"

    def test_highest_keyword_count_wins(self, engine):
        result = engine.override_pain("other", [
            "the price is too high and cost keeps going up",
            "also support is slow",
        ])
        # pricing: "price" + "cost" = 2 hits; support: "support" = 1 hit
        assert result == "pricing"

    def test_quotable_phrases_scanned(self, engine):
        result = engine.override_pain("other", [], quotable_phrases=["incredibly slow loading times"])
        assert result == "performance"

    def test_salesforce_customization_maps_to_admin_burden(self, engine):
        result = engine.override_pain(
            "other",
            ["Out of the box Salesforce does 60% of what we need and the rest requires consultants or Apex"],
        )
        assert result == "admin_burden"

    def test_salesforce_app_exchange_maps_to_integration_debt(self, engine):
        result = engine.override_pain(
            "other",
            ["AppExchange integrations keep breaking after updates and require constant maintenance"],
        )
        assert result == "integration_debt"


# -- Recommend Derivation --------------------------------------------------


class TestRecommendDerivation:

    def test_positive_language(self, engine):
        result = engine.derive_recommend(["I would highly recommend this tool"], rating=None, rating_max=5)
        assert result is True

    def test_negative_language(self, engine):
        result = engine.derive_recommend(["I would not recommend this to anyone"], rating=None, rating_max=5)
        assert result is False

    def test_mixed_language_negative_wins(self, engine):
        result = engine.derive_recommend(
            ["It's a great tool", "but I would not recommend it due to pricing", "stay away"],
            rating=None, rating_max=5,
        )
        assert result is False

    def test_no_language_high_rating(self, engine):
        result = engine.derive_recommend([], rating=4.5, rating_max=5.0)
        assert result is True

    def test_no_language_low_rating(self, engine):
        result = engine.derive_recommend([], rating=1.0, rating_max=5.0)
        assert result is False

    def test_no_language_mid_rating_returns_none(self, engine):
        result = engine.derive_recommend([], rating=3.0, rating_max=5.0)
        assert result is None

    def test_empty_everything_returns_none(self, engine):
        result = engine.derive_recommend([], rating=None, rating_max=5.0)
        assert result is None

    def test_avoid_pattern(self, engine):
        result = engine.derive_recommend(["avoid this product at all costs"], rating=None, rating_max=5)
        assert result is False


# -- Price Complaint Derivation --------------------------------------------


class TestPriceComplaintDerivation:

    def test_price_increase_mentioned(self, engine):
        enrichment = {"budget_signals": {"price_increase_mentioned": True}, "pricing_phrases": [], "specific_complaints": []}
        assert engine.derive_price_complaint(enrichment) is True

    def test_pricing_phrases_present(self, engine):
        enrichment = {"budget_signals": {"price_increase_mentioned": False}, "pricing_phrases": ["30% more expensive"], "specific_complaints": []}
        assert engine.derive_price_complaint(enrichment) is True

    def test_complaint_keywords(self, engine):
        enrichment = {"budget_signals": {"price_increase_mentioned": False}, "pricing_phrases": [], "specific_complaints": ["way too expensive"]}
        assert engine.derive_price_complaint(enrichment) is True

    def test_no_price_signal(self, engine):
        enrichment = {"budget_signals": {"price_increase_mentioned": False}, "pricing_phrases": [], "specific_complaints": ["slow and buggy"]}
        assert engine.derive_price_complaint(enrichment) is False

    def test_explicitly_positive_pricing_phrase_is_not_a_price_complaint(self, engine):
        enrichment = {
            "budget_signals": {"price_increase_mentioned": False},
            "pricing_phrases": ["I find the pricing reasonable."],
            "specific_complaints": [],
        }
        assert engine.derive_price_complaint(enrichment) is False


# -- Budget Authority Derivation -------------------------------------------


class TestBudgetAuthorityDerivation:

    def test_executive_role(self, engine):
        enrichment = {"reviewer_context": {"role_level": "executive", "decision_maker": False}, "urgency_indicators": {}, "budget_signals": {}}
        assert engine.derive_budget_authority(enrichment) is True

    def test_director_role(self, engine):
        enrichment = {"reviewer_context": {"role_level": "director", "decision_maker": False}, "urgency_indicators": {}, "budget_signals": {}}
        assert engine.derive_budget_authority(enrichment) is True

    def test_decision_maker_flag(self, engine):
        enrichment = {"reviewer_context": {"role_level": "ic", "decision_maker": True}, "urgency_indicators": {}, "budget_signals": {}}
        assert engine.derive_budget_authority(enrichment) is True

    def test_decision_maker_language(self, engine):
        enrichment = {"reviewer_context": {"role_level": "ic", "decision_maker": False}, "urgency_indicators": {"decision_maker_language": True}, "budget_signals": {}}
        assert engine.derive_budget_authority(enrichment) is True

    def test_budget_amount_present(self, engine):
        enrichment = {"reviewer_context": {"role_level": "ic", "decision_maker": False}, "urgency_indicators": {}, "budget_signals": {"annual_spend_estimate": "$50k/yr"}}
        assert engine.derive_budget_authority(enrichment) is True

    def test_no_authority_signals(self, engine):
        enrichment = {"reviewer_context": {"role_level": "ic", "decision_maker": False}, "urgency_indicators": {}, "budget_signals": {}}
        assert engine.derive_budget_authority(enrichment) is False


# -- Conclusion Gating ----------------------------------------------------


class TestConclusionGating:

    def test_insufficient_data_suppresses_all(self, engine):
        evidence = {"total_reviews": 10}
        results = engine.evaluate_conclusions(evidence)
        assert len(results) == 1
        assert results[0].conclusion_id == "insufficient_data"
        assert results[0].met is True

    def test_sufficient_data_evaluates_all(self, engine):
        evidence = {"total_reviews": 100}
        results = engine.evaluate_conclusions(evidence)
        assert len(results) >= 4
        assert all(r.conclusion_id != "insufficient_data" for r in results)

    def test_pricing_crisis_met(self, engine):
        evidence = {
            "total_reviews": 200,
            "pain_distribution": {"pricing": {"count": 30, "source_count": 4, "rank": 1}},
            "pricing_phrases_total": 10,
        }
        results = engine.evaluate_conclusions(evidence)
        pricing = next(r for r in results if r.conclusion_id == "pricing_crisis")
        assert pricing.met is True
        assert pricing.confidence == "high"

    def test_pricing_crisis_not_met_low_count(self, engine):
        evidence = {
            "total_reviews": 200,
            "pain_distribution": {"pricing": {"count": 5, "source_count": 4, "rank": 1}},
            "pricing_phrases_total": 10,
        }
        results = engine.evaluate_conclusions(evidence)
        pricing = next(r for r in results if r.conclusion_id == "pricing_crisis")
        assert pricing.met is False
        assert pricing.fallback_label is not None

    def test_losing_market_share_met(self, engine):
        evidence = {
            "total_reviews": 200,
            "displacement_edge": {
                "mention_count": 15,
                "signal_strength": "strong",
                "explicit_switch_count": 3,
                "net_flow": -10,
            },
        }
        results = engine.evaluate_conclusions(evidence)
        losing = next(r for r in results if r.conclusion_id == "losing_market_share")
        assert losing.met is True

    def test_active_churn_wave_met(self, engine):
        evidence = {
            "total_reviews": 100,
            "indicator_counts": {
                "active_evaluation_language": 8,
                "explicit_cancel_language": 4,
            },
        }
        results = engine.evaluate_conclusions(evidence)
        wave = next(r for r in results if r.conclusion_id == "active_churn_wave")
        assert wave.met is True
        assert wave.confidence == "high"


# -- Suppression ----------------------------------------------------------


class TestSuppression:

    def test_no_suppression_for_healthy_data(self, engine):
        result = engine.evaluate_suppression("executive_summary", {"total_reviews": 200, "confidence": "high"})
        assert result.suppress is False
        assert result.degrade is False

    def test_suppress_executive_summary_low_reviews(self, engine):
        result = engine.evaluate_suppression("executive_summary", {"total_reviews": 10})
        assert result.suppress is True

    def test_degrade_executive_summary_moderate_reviews(self, engine):
        result = engine.evaluate_suppression("executive_summary", {"total_reviews": 35})
        assert result.degrade is True
        assert result.disclaimer is not None

    def test_suppress_target_accounts_zero_companies(self, engine):
        result = engine.evaluate_suppression("target_accounts", {"named_company_count": 0})
        assert result.suppress is True
        assert result.fallback_label is not None

    def test_suppress_recommend_low_denominator(self, engine):
        result = engine.evaluate_suppression("recommend_ratio", {"recommend_denominator": 3})
        assert result.suppress is True

    def test_unknown_section_no_suppression(self, engine):
        result = engine.evaluate_suppression("nonexistent_section", {"total_reviews": 5})
        assert result.suppress is False


# -- Confidence Tiers -----------------------------------------------------


class TestConfidenceTiers:

    def test_high_confidence(self, engine):
        assert engine.get_confidence_tier(100) == "high"

    def test_medium_confidence(self, engine):
        assert engine.get_confidence_tier(30) == "medium"

    def test_low_confidence(self, engine):
        assert engine.get_confidence_tier(5) == "low"

    def test_insufficient(self, engine):
        assert engine.get_confidence_tier(0) == "insufficient"

    def test_label_includes_text(self, engine):
        label = engine.get_confidence_label(30)
        assert "confidence" in label.lower() or "moderate" in label.lower()
