"""Tests for cross-vendor synthesis packet builders and compatibility helpers."""

import json

import pytest

from atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis import (
    _sorted_vendors,
    build_category_council_packet,
    build_pairwise_battle_packet,
    build_resource_asymmetry_packet,
    compute_cross_vendor_evidence_hash,
    normalize_cross_vendor_contract,
    to_legacy_cross_vendor_conclusion,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_POOL_LAYERS = {
    "Zendesk": {
        "core": {
            "total_reviews": 120,
            "avg_urgency_score": 7.2,
            "churn_signal_density": 22.1,
            "price_complaint_rate": 0.28,
            "product_category": "Helpdesk",
            "top_competitors": [{"name": "Freshdesk"}],
        },
        "pain_distribution": [{"category": "pricing", "count": 30}],
        "budget_pressure": {"price_increase_rate": 0.03, "avg_seat_count": 50},
    },
    "Freshdesk": {
        "core": {
            "total_reviews": 80,
            "avg_urgency_score": 5.5,
            "churn_signal_density": 15.2,
            "price_complaint_rate": 0.24,
            "product_category": "Helpdesk",
            "top_competitors": [{"name": "Zendesk"}],
        },
        "pain_distribution": [{"category": "support", "count": 20}],
        "budget_pressure": {"price_increase_rate": 0.03, "avg_seat_count": 35},
    },
}

_PROFILES = {
    "Zendesk": {
        "product_category": "Helpdesk",
        "strengths": ["Enterprise features"],
        "weaknesses": ["Complex pricing"],
        "top_integrations": ["Salesforce"],
        "primary_use_cases": ["Ticketing"],
        "typical_company_size": "51-200",
        "typical_industries": ["SaaS"],
    },
    "Freshdesk": {
        "product_category": "Helpdesk",
        "strengths": ["Easy setup"],
        "weaknesses": ["Limited reporting"],
        "top_integrations": ["Slack"],
        "primary_use_cases": ["Support"],
        "typical_company_size": "11-50",
        "typical_industries": ["E-commerce"],
    },
}

_EDGE = {
    "from_vendor": "Zendesk",
    "to_vendor": "Freshdesk",
    "mention_count": 12,
    "signal_strength": "strong",
    "primary_driver": "pricing",
    "evidence_breakdown": {"explicit_switch": 2, "active_evaluation": 4, "implied_preference": 6},
    "velocity_7d": 3,
}


# ---------------------------------------------------------------------------
# Packet builder tests
# ---------------------------------------------------------------------------

class TestBuildPairwiseBattlePacket:
    def test_locked_direction_matches_edge(self):
        packet = build_pairwise_battle_packet(
            "Zendesk", "Freshdesk", _EDGE, _POOL_LAYERS, _PROFILES,
        )
        assert packet["locked_direction"]["winner"] == "Freshdesk"
        assert packet["locked_direction"]["loser"] == "Zendesk"

    def test_contains_both_vendor_pools(self):
        packet = build_pairwise_battle_packet(
            "Zendesk", "Freshdesk", _EDGE, _POOL_LAYERS, _PROFILES,
        )
        assert packet["vendor_a_pool"]["vendor"] == "Zendesk"
        assert packet["vendor_b_pool"]["vendor"] == "Freshdesk"
        assert packet["vendor_a_pool"]["total_reviews"] == 120
        assert packet["vendor_b_pool"]["total_reviews"] == 80

    def test_contains_both_profiles(self):
        packet = build_pairwise_battle_packet(
            "Zendesk", "Freshdesk", _EDGE, _POOL_LAYERS, _PROFILES,
        )
        assert packet["vendor_a_profile"]["category"] == "Helpdesk"
        assert "Enterprise features" in packet["vendor_a_profile"]["strengths"]

    def test_displacement_edge_fields(self):
        packet = build_pairwise_battle_packet(
            "Zendesk", "Freshdesk", _EDGE, _POOL_LAYERS, _PROFILES,
        )
        edge = packet["displacement_edge"]
        assert edge["mention_count"] == 12
        assert edge["signal_strength"] == "strong"
        assert edge["primary_driver"] == "pricing"


class TestBuildCategoryCouncilPacket:
    def test_category_vendors_resolved(self):
        packet = build_category_council_packet(
            "Helpdesk",
            {"hhi": 0.3, "displacement_intensity": 5.0},
            _POOL_LAYERS,
            _PROFILES,
        )
        assert packet["category"] == "Helpdesk"
        assert packet["vendor_count"] == 2
        assert len(packet["vendor_summaries"]) == 2

    def test_ecosystem_evidence_included(self):
        packet = build_category_council_packet(
            "Helpdesk",
            {"hhi": 0.3, "market_structure": "duopoly", "displacement_intensity": 5.0},
            _POOL_LAYERS,
            _PROFILES,
        )
        assert packet["ecosystem_evidence"]["hhi"] == 0.3
        assert packet["ecosystem_evidence"]["market_structure"] == "duopoly"


class TestBuildResourceAsymmetryPacket:
    def test_pressure_scores_included(self):
        packet = build_resource_asymmetry_packet(
            "Zendesk", "Freshdesk", _POOL_LAYERS, _PROFILES,
        )
        assert packet["pressure_scores"]["vendor_a_urgency"] == 7.2
        assert packet["pressure_scores"]["vendor_b_urgency"] == 5.5

    def test_resource_indicators_included(self):
        packet = build_resource_asymmetry_packet(
            "Zendesk", "Freshdesk", _POOL_LAYERS, _PROFILES,
        )
        ri = packet["resource_indicators"]
        assert ri["vendor_a_reviews"] == 120
        assert ri["vendor_b_reviews"] == 80

    def test_divergence_score_computed(self):
        packet = build_resource_asymmetry_packet(
            "Zendesk", "Freshdesk", _POOL_LAYERS, _PROFILES,
        )
        assert 0.0 < packet["divergence_score"] < 1.0


# ---------------------------------------------------------------------------
# Evidence hash tests
# ---------------------------------------------------------------------------

class TestEvidenceHash:
    def test_deterministic(self):
        packet = build_pairwise_battle_packet(
            "Zendesk", "Freshdesk", _EDGE, _POOL_LAYERS, _PROFILES,
        )
        h1 = compute_cross_vendor_evidence_hash(packet)
        h2 = compute_cross_vendor_evidence_hash(packet)
        assert h1 == h2

    def test_changes_with_input(self):
        packet = build_pairwise_battle_packet(
            "Zendesk", "Freshdesk", _EDGE, _POOL_LAYERS, _PROFILES,
        )
        h1 = compute_cross_vendor_evidence_hash(packet)
        modified_edge = {**_EDGE, "mention_count": 99}
        packet2 = build_pairwise_battle_packet(
            "Zendesk", "Freshdesk", modified_edge, _POOL_LAYERS, _PROFILES,
        )
        h2 = compute_cross_vendor_evidence_hash(packet2)
        assert h1 != h2


# ---------------------------------------------------------------------------
# Contract normalization tests
# ---------------------------------------------------------------------------

class TestNormalizeCrossVendorContract:
    def test_battle_fills_defaults(self):
        raw = {"winner": "A", "loser": "B", "conclusion": "A wins"}
        result = normalize_cross_vendor_contract(raw, "pairwise_battle")
        assert result["durability_assessment"] == "uncertain"
        assert result["confidence"] == 0.0
        assert isinstance(result["key_insights"], list)
        assert isinstance(result["falsification_conditions"], list)

    def test_council_fills_defaults(self):
        raw = {"market_regime": "price_competition", "conclusion": "Price war"}
        result = normalize_cross_vendor_contract(raw, "category_council")
        assert result["market_regime"] == "price_competition"
        assert result["winner"] is None

    def test_asymmetry_fills_defaults(self):
        raw = {"favored_vendor": "A", "disadvantaged_vendor": "B"}
        result = normalize_cross_vendor_contract(raw, "resource_asymmetry")
        assert result["pressure_delta"] == 0.0

    def test_confidence_clamped(self):
        raw = {"winner": "A", "loser": "B", "confidence": 5.0}
        result = normalize_cross_vendor_contract(raw, "pairwise_battle")
        assert result["confidence"] == 1.0


# ---------------------------------------------------------------------------
# Legacy mirror tests
# ---------------------------------------------------------------------------

class TestLegacyMirror:
    def test_battle_mirror_shape(self):
        synthesis = {
            "winner": "Freshdesk",
            "loser": "Zendesk",
            "conclusion": "Freshdesk gaining due to pricing",
            "confidence": 0.85,
            "durability_assessment": "structural",
            "key_insights": [{"insight": "Price gap", "evidence": "0.28 vs 0.24"}],
        }
        legacy = to_legacy_cross_vendor_conclusion(
            synthesis, "pairwise_battle",
            ["Freshdesk", "Zendesk"],
            evidence_hash="abc123",
            tokens_used=500,
        )
        assert legacy["analysis_type"] == "pairwise_battle"
        assert legacy["vendors"] == ["Freshdesk", "Zendesk"]
        assert legacy["confidence"] == 0.85
        assert legacy["conclusion"]["winner"] == "Freshdesk"
        assert legacy["conclusion"]["loser"] == "Zendesk"
        assert legacy["cached"] is False

    def test_council_mirror_shape(self):
        synthesis = {
            "market_regime": "price_competition",
            "conclusion": "Price war in Helpdesk",
            "winner": "Freshdesk",
            "loser": "Zendesk",
            "confidence": 0.7,
        }
        legacy = to_legacy_cross_vendor_conclusion(
            synthesis, "category_council",
            ["Freshdesk", "Zendesk"],
            category="Helpdesk",
        )
        assert legacy["category"] == "Helpdesk"
        assert legacy["conclusion"]["market_regime"] == "price_competition"

    def test_asymmetry_mirror_shape(self):
        synthesis = {
            "favored_vendor": "Zendesk",
            "disadvantaged_vendor": "Freshdesk",
            "conclusion": "Zendesk has larger base",
            "pressure_delta": 1.7,
            "confidence": 0.6,
        }
        legacy = to_legacy_cross_vendor_conclusion(
            synthesis, "resource_asymmetry",
            ["Freshdesk", "Zendesk"],
        )
        assert legacy["conclusion"]["resource_advantage"] == "Zendesk"
        assert legacy["conclusion"]["pressure_delta"] == 1.7


# ---------------------------------------------------------------------------
# Canonical vendor ordering tests
# ---------------------------------------------------------------------------

class TestVendorOrdering:
    def test_sorted_vendors_deduplicates(self):
        assert _sorted_vendors("B", "A", "B") == ["A", "B"]

    def test_sorted_vendors_strips_whitespace(self):
        assert _sorted_vendors(" Zendesk ", "Freshdesk") == ["Freshdesk", "Zendesk"]

    def test_sorted_vendors_filters_empty(self):
        assert _sorted_vendors("A", "", "B", None) == ["A", "B"]
