"""Tests for challenger_brief report type.

Covers weakness coverage, target account filtering, integration comparison,
brief assembly, executive summary format, and self-flow skip.
"""

import json
import sys
from datetime import date
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

# Pre-mock heavy deps before importing task module
_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)

for _mod in (
    "torch", "torchaudio", "transformers", "accelerate", "bitsandbytes",
    "PIL", "PIL.Image", "numpy", "cv2", "sounddevice", "soundfile",
    "nemo.collections", "nemo.collections.asr",
    "nemo.collections.asr.models",
    "playwright", "playwright.async_api",
    "playwright_stealth",
    "curl_cffi", "curl_cffi.requests",
    "pytrends", "pytrends.request",
):
    sys.modules.setdefault(_mod, MagicMock())

import atlas_brain.autonomous.tasks.b2b_challenger_brief as brief_mod
from atlas_brain.autonomous.tasks.b2b_challenger_brief import (
    _check_freshness,
    _compute_weakness_coverage,
    _extract_pain_quotes_from_reviews,
    _fetch_cross_vendor_battle,
    _resolve_cross_vendor_battle,
    _fetch_persisted_report_record,
    _filter_target_accounts,
    _build_integration_comparison,
    _build_challenger_brief,
    _normalize_product_profile_weaknesses,
    _retire_unselected_challenger_briefs,
)
from atlas_brain.autonomous.tasks._b2b_synthesis_reader import (
    load_synthesis_view,
)


# ---------------------------------------------------------------------------
# Weakness coverage tests
# ---------------------------------------------------------------------------

class TestWeaknessCoverage:
    def test_strong_match_exact(self):
        """Exact string match produces strong coverage."""
        weaknesses = [{"area": "pricing", "score": 3.1, "evidence_count": 30}]
        pain_addressed = [{"area": "pricing"}]
        result = _compute_weakness_coverage(weaknesses, pain_addressed)
        assert len(result) == 1
        assert result[0]["match_quality"] == "strong"
        assert result[0]["challenger_strength_score"] >= 0.8

    def test_strong_match_case_insensitive(self):
        """Matching is case-insensitive."""
        weaknesses = [{"area": "Pricing", "score": 2.0}]
        pain_addressed = [{"area": "pricing"}]
        result = _compute_weakness_coverage(weaknesses, pain_addressed)
        assert len(result) == 1
        assert result[0]["match_quality"] == "strong"

    def test_moderate_match_substring(self):
        """Substring match produces moderate or strong coverage."""
        weaknesses = [{"area": "pricing_transparency", "score": 2.0}]
        pain_addressed = [{"area": "pricing"}]
        result = _compute_weakness_coverage(weaknesses, pain_addressed)
        assert len(result) == 1
        assert result[0]["match_quality"] in ("strong", "moderate")
        assert result[0]["challenger_strength_score"] >= 0.6

    def test_no_match(self):
        """Unrelated areas produce no coverage entries."""
        weaknesses = [{"area": "pricing", "score": 3.0}]
        pain_addressed = [{"area": "support_quality"}]
        result = _compute_weakness_coverage(weaknesses, pain_addressed)
        assert len(result) == 0

    def test_empty_weaknesses(self):
        """Empty weaknesses returns empty list."""
        result = _compute_weakness_coverage([], [{"area": "pricing"}])
        assert result == []

    def test_empty_pain_addressed(self):
        """Empty pain_addressed returns empty list."""
        result = _compute_weakness_coverage(
            [{"area": "pricing", "score": 3.0}], [],
        )
        assert result == []

    def test_none_inputs(self):
        """None inputs handled gracefully."""
        assert _compute_weakness_coverage([], None) == []

    def test_multiple_weaknesses(self):
        """Multiple weaknesses with partial matches."""
        weaknesses = [
            {"area": "pricing", "score": 3.0},
            {"area": "support", "score": 2.5},
            {"area": "integrations", "score": 1.0},
        ]
        pain_addressed = [{"area": "pricing"}, {"area": "support"}]
        result = _compute_weakness_coverage(weaknesses, pain_addressed)
        matched_areas = {r["incumbent_weakness"] for r in result}
        assert "pricing" in matched_areas
        assert "support" in matched_areas
        assert "integrations" not in matched_areas

    def test_string_pain_addressed(self):
        """Pain addressed as plain strings (not dicts)."""
        weaknesses = [{"area": "pricing", "score": 3.0}]
        pain_addressed = ["pricing", "support"]
        result = _compute_weakness_coverage(weaknesses, pain_addressed)
        assert len(result) == 1
        assert result[0]["incumbent_weakness"] == "pricing"


# ---------------------------------------------------------------------------
# Target account filtering tests
# ---------------------------------------------------------------------------

class TestFilterTargetAccounts:
    def _make_accounts_data(self, accounts: list[dict]) -> dict:
        return {"accounts": accounts}

    def _make_account(self, company="Acme Corp", score=75, urgency=8.0, **kw):
        return {
            "company": company,
            "opportunity_score": score,
            "urgency": urgency,
            "buying_stage": kw.get("buying_stage", "evaluation"),
            "seat_count": kw.get("seat_count", 100),
            "contract_end": kw.get("contract_end"),
            "industry": kw.get("industry"),
            "domain": kw.get("domain"),
            "annual_revenue_range": kw.get("annual_revenue_range"),
            "top_quote": kw.get("top_quote"),
            "alternatives_considering": kw.get("alternatives", []),
        }

    def test_considers_challenger_flag(self):
        """Accounts with challenger in alternatives get considers_challenger=True."""
        data = self._make_accounts_data([
            self._make_account("A", 80, alternatives=["Freshdesk"]),
            self._make_account("B", 70, alternatives=["Help Scout"]),
        ])
        targets, total, considering = _filter_target_accounts(data, "Freshdesk", max_accounts=10)
        assert total == 2
        assert considering == 1
        a_target = next(t for t in targets if t["company"] == "A")
        b_target = next(t for t in targets if t["company"] == "B")
        assert a_target["considers_challenger"] is True
        assert b_target["considers_challenger"] is False

    def test_case_insensitive_challenger(self):
        """Challenger name matching is case-insensitive."""
        data = self._make_accounts_data([
            self._make_account("A", 80, alternatives=["freshdesk"]),
        ])
        _, _, considering = _filter_target_accounts(data, "Freshdesk", max_accounts=10)
        assert considering == 1

    def test_respects_max_limit(self):
        """Target accounts are capped at max_accounts."""
        accounts = [self._make_account(f"Co{i}", 100 - i) for i in range(20)]
        data = self._make_accounts_data(accounts)
        targets, total, _ = _filter_target_accounts(data, "X", max_accounts=5)
        assert len(targets) == 5
        assert total == 20

    def test_sorted_by_score_desc(self):
        """Accounts are sorted by opportunity_score descending."""
        data = self._make_accounts_data([
            self._make_account("Low", 30),
            self._make_account("High", 90),
            self._make_account("Mid", 60),
        ])
        targets, _, _ = _filter_target_accounts(data, "X", max_accounts=10)
        scores = [t["opportunity_score"] for t in targets]
        assert scores == sorted(scores, reverse=True)

    def test_none_data(self):
        """None accounts_data returns empty."""
        targets, total, considering = _filter_target_accounts(None, "X", max_accounts=10)
        assert targets == []
        assert total == 0
        assert considering == 0

    def test_domain_and_revenue_passed_through(self):
        """domain and annual_revenue_range are included in target output."""
        data = self._make_accounts_data([
            self._make_account(
                "Kroger", 90,
                domain="kroger.com",
                annual_revenue_range="$100B+",
                alternatives=["Freshdesk"],
            ),
        ])
        targets, _, _ = _filter_target_accounts(data, "Freshdesk", max_accounts=10)
        assert targets[0]["domain"] == "kroger.com"
        assert targets[0]["annual_revenue_range"] == "$100B+"

    def test_empty_accounts(self):
        """Empty accounts list returns empty."""
        targets, total, considering = _filter_target_accounts({"accounts": []}, "X", max_accounts=10)
        assert targets == []
        assert total == 0


# ---------------------------------------------------------------------------
# Integration comparison tests
# ---------------------------------------------------------------------------

class TestIntegrationComparison:
    def test_shared_and_exclusive(self):
        """Shared and exclusive integrations computed correctly."""
        inc = {"top_integrations": ["Zapier", "Slack", "Legacy CRM"]}
        chal = {"top_integrations": ["Zapier", "Slack", "Marketo"]}
        result = _build_integration_comparison(inc, chal)
        assert set(result["shared"]) == {"Zapier", "Slack"}
        assert result["challenger_exclusive"] == ["Marketo"]
        assert result["incumbent_exclusive"] == ["Legacy CRM"]

    def test_none_profiles(self):
        """None profiles produce empty sets."""
        result = _build_integration_comparison(None, None)
        assert result["shared"] == []
        assert result["challenger_exclusive"] == []
        assert result["incumbent_exclusive"] == []

    def test_one_none_profile(self):
        """One None profile still works."""
        result = _build_integration_comparison(
            {"top_integrations": ["Zapier"]}, None,
        )
        assert result["shared"] == []
        assert result["incumbent_exclusive"] == ["Zapier"]
        assert result["challenger_exclusive"] == []

    def test_dict_integrations(self):
        """Integrations as dicts with name key."""
        inc = {"top_integrations": [{"name": "Zapier"}, {"name": "Slack"}]}
        chal = {"top_integrations": [{"name": "Slack"}, {"name": "HubSpot"}]}
        result = _build_integration_comparison(inc, chal)
        assert result["shared"] == ["Slack"]

    def test_empty_integrations(self):
        """Empty integration lists produce empty sets."""
        result = _build_integration_comparison(
            {"top_integrations": []}, {"top_integrations": []},
        )
        assert result["shared"] == []


# ---------------------------------------------------------------------------
# Brief assembly tests
# ---------------------------------------------------------------------------

class TestBuildChallengerBrief:
    def _minimal_displacement(self):
        return {
            "total_mentions": 10,
            "signal_strength": "moderate",
            "confidence_score": 0.7,
            "primary_driver": "pricing",
            "key_quote": "We switched because of pricing",
            "source_distribution": {"g2": 5, "reddit": 5},
        }

    def test_minimal_brief(self):
        """Brief with only displacement detail (all other sources None)."""
        brief = _build_challenger_brief(
            incumbent="Zendesk",
            challenger="Freshdesk",
            displacement_detail=self._minimal_displacement(),
            battle_card=None,
            accounts_in_motion=None,
            incumbent_profile=None,
            challenger_profile=None,
            churn_signal=None,
            cross_vendor_battle=None,
            max_target_accounts=15,
        )
        assert brief["incumbent"] == "Zendesk"
        assert brief["challenger"] == "Freshdesk"
        assert brief["displacement_summary"]["total_mentions"] == 10
        assert brief["target_accounts"] == []
        assert brief["total_target_accounts"] == 0
        assert brief["data_sources"]["battle_card"] is False
        assert brief["data_sources"]["accounts_in_motion"] is False

    def test_full_brief(self):
        """Brief with all sources present."""
        battle_card = {
            "vendor_weaknesses": [
                {"area": "pricing", "score": 3.1, "evidence_count": 30},
                {"area": "support", "score": 2.5, "evidence_count": 15},
            ],
            "customer_pain_quotes": [
                {"quote": "Pricing tripled", "urgency": 9, "company": "Acme"},
            ],
            "churn_pressure_score": 72.5,
            "risk_level": "high",
            "objection_data": {
                "price_complaint_rate": 0.35,
                "dm_churn_rate": 0.42,
                "sentiment_direction": "negative",
            },
            "discovery_questions": ["What's your current cost per seat?"],
            "landmine_questions": ["Have you seen hidden fees?"],
            "objection_handlers": ["We offer transparent pricing"],
            "talk_track": "Freshdesk offers transparent pricing...",
            "recommended_plays": ["Price comparison worksheet"],
        }
        accounts = {
            "accounts": [
                {
                    "company": "Acme Corp",
                    "opportunity_score": 85,
                    "urgency": 8.5,
                    "buying_stage": "active_purchase",
                    "seat_count": 250,
                    "contract_end": "Q2 2026",
                    "industry": "SaaS",
                    "top_quote": "Looking to switch by Q3",
                    "alternatives_considering": ["Freshdesk", "Help Scout"],
                },
            ],
            "category_council": {
                "winner": "Zoho Desk",
                "loser": "Freshdesk",
                "conclusion": "Pricing pressure is fragmenting the helpdesk market.",
                "market_regime": "price_competition",
                "durability": "cyclical",
                "confidence": 0.58,
                "key_insights": [{"insight": "Pricing is the primary driver.", "evidence": "pricing"}],
            },
        }
        inc_profile = {
            "strengths": [{"area": "enterprise_features"}],
            "weaknesses": [{"area": "pricing"}],
            "pain_addressed": [],
            "commonly_switched_from": [],
            "top_integrations": ["Zapier", "Slack", "Legacy CRM"],
            "profile_summary": "Enterprise helpdesk...",
            "category": "Helpdesk",
        }
        chal_profile = {
            "strengths": [{"area": "pricing_transparency", "score": 8.5}],
            "weaknesses": [],
            "pain_addressed": [{"area": "pricing"}],
            "commonly_switched_from": [{"vendor": "Zendesk", "count": 15}],
            "top_integrations": ["Zapier", "Slack", "Marketo"],
            "profile_summary": "Freshdesk is a customer support platform...",
            "category": "Helpdesk",
        }
        churn_signal = {
            "archetype": "pricing_shock",
            "archetype_confidence": 0.82,
            "risk_level": "high",
            "key_signals": ["3x price increase"],
            "churn_pressure_score": None,
            "sentiment_direction": None,
            "price_complaint_rate": 0.35,
            "dm_churn_rate": 0.42,
        }
        cross_vendor = {
            "winner": "Freshdesk",
            "loser": "Zendesk",
            "conclusion": "Freshdesk wins on SMB pricing",
            "durability": "Durable",
            "key_insights": ["Freshdesk wins deals < $50k ARR"],
            "confidence": 0.78,
        }

        brief = _build_challenger_brief(
            incumbent="Zendesk",
            challenger="Freshdesk",
            displacement_detail=self._minimal_displacement(),
            battle_card=battle_card,
            accounts_in_motion=accounts,
            incumbent_profile=inc_profile,
            challenger_profile=chal_profile,
            churn_signal=churn_signal,
            cross_vendor_battle=cross_vendor,
            battle_card_metadata={"battle_card_date": "2026-03-16", "battle_card_stale": True},
            max_target_accounts=15,
        )

        # All 7 sections populated
        assert brief["displacement_summary"]["total_mentions"] == 10
        assert brief["incumbent_profile"]["archetype"] == "pricing_shock"
        assert brief["incumbent_profile"]["churn_pressure_score"] == 72.5
        assert len(brief["incumbent_profile"]["top_weaknesses"]) == 2
        assert len(brief["challenger_advantage"]["strengths"]) == 1
        assert len(brief["challenger_advantage"]["weakness_coverage"]) >= 1
        assert brief["head_to_head"]["winner"] == "Freshdesk"
        assert brief["head_to_head"]["loser"] == "Zendesk"
        assert brief["head_to_head"]["confidence"] == 0.78
        assert brief["category_council"]["winner"] == "Zoho Desk"
        assert brief["category_council"]["market_regime"] == "price_competition"
        assert len(brief["target_accounts"]) == 1
        assert brief["target_accounts"][0]["considers_challenger"] is True
        assert brief["accounts_considering_challenger"] == 1
        assert brief["battle_card_date"] == "2026-03-16"
        assert brief["battle_card_stale"] is True
        assert brief["sales_playbook"]["discovery_questions"] == ["What's your current cost per seat?"]
        assert "Marketo" in brief["integration_comparison"]["challenger_exclusive"]
        assert "Legacy CRM" in brief["integration_comparison"]["incumbent_exclusive"]
        # All original sources present; review_quotes is False when battle card exists
        for k in ("battle_card", "accounts_in_motion", "product_profiles", "cross_vendor_conclusion"):
            assert brief["data_sources"][k] is True
        assert brief["data_sources"]["evidence_vault"] is False
        assert brief["category"] == "Helpdesk"

    def test_evidence_vault_fallback_populates_weaknesses_and_quotes(self):
        """Vault fills incumbent evidence when battle card is absent."""
        vault = {
            "recent_window_days": 30,
            "weakness_evidence": [
                {
                    "key": "pricing",
                    "label": "Pricing opacity",
                    "evidence_type": "pain_category",
                    "best_quote": "Our actual bill was much higher than promised",
                    "quote_source": {
                        "company": "Acme",
                        "reviewer_title": "VP Ops",
                        "source": "g2",
                        "reviewed_at": "2026-03-10",
                        "rating": 2.0,
                    },
                    "mention_count_total": 18,
                    "mention_count_recent": 7,
                    "trend": {"direction": "accelerating"},
                    "supporting_metrics": {"avg_urgency_when_mentioned": 7.2},
                },
            ],
        }
        brief = _build_challenger_brief(
            incumbent="Zendesk",
            challenger="Freshdesk",
            displacement_detail=self._minimal_displacement(),
            battle_card=None,
            accounts_in_motion=None,
            incumbent_profile=None,
            challenger_profile=None,
            incumbent_evidence_vault=vault,
            churn_signal=None,
            cross_vendor_battle=None,
            max_target_accounts=15,
        )
        weakness = brief["incumbent_profile"]["top_weaknesses"][0]
        quote = brief["incumbent_profile"]["top_pain_quotes"][0]
        assert weakness["area"] == "Pricing opacity"
        assert weakness["count"] == 18
        assert "7 in last 30 days" in weakness["evidence"]
        assert quote["quote"] == "Our actual bill was much higher than promised"
        assert quote["company"] == "Acme"
        assert quote["role"] == "VP Ops"
        assert quote["source_site"] == "g2"
        assert brief["data_sources"]["evidence_vault"] is True

    def test_evidence_vault_quotes_fill_battle_card_quote_gap(self):
        """Vault quotes can fill a quote gap even when battle-card weaknesses exist."""
        battle_card = {
            "vendor_weaknesses": [{"area": "pricing", "score": 3.1, "evidence_count": 30}],
            "customer_pain_quotes": [],
        }
        vault = {
            "weakness_evidence": [
                {
                    "key": "pricing",
                    "label": "Pricing opacity",
                    "evidence_type": "pain_category",
                    "best_quote": "The add-ons made the contract much more expensive",
                    "quote_source": {"company": "Beta", "reviewer_title": "Director", "source": "reddit"},
                    "mention_count_total": 12,
                    "mention_count_recent": 4,
                    "supporting_metrics": {"avg_urgency_when_mentioned": 6.8},
                },
            ],
        }
        brief = _build_challenger_brief(
            incumbent="Zendesk",
            challenger="Freshdesk",
            displacement_detail=self._minimal_displacement(),
            battle_card=battle_card,
            accounts_in_motion=None,
            incumbent_profile=None,
            challenger_profile=None,
            incumbent_evidence_vault=vault,
            churn_signal=None,
            cross_vendor_battle=None,
            max_target_accounts=15,
        )
        assert brief["incumbent_profile"]["top_weaknesses"][0]["area"] == "pricing"
        assert brief["incumbent_profile"]["top_pain_quotes"][0]["company"] == "Beta"
        assert brief["data_sources"]["battle_card"] is True
        assert brief["data_sources"]["evidence_vault"] is True

    def test_synthesis_view_attaches_reasoning_contracts(self):
        """Challenger brief carries vendor/displacement reasoning contracts."""
        synthesis_view = load_synthesis_view(
            {
                "schema_version": "2.1",
                "reasoning_contracts": {
                    "vendor_core_reasoning": {
                        "schema_version": "v1",
                        "causal_narrative": {
                            "primary_wedge": "price_squeeze",
                            "confidence": "high",
                            "trigger": "Price hike",
                            "why_now": "AI bundle forcing higher per-seat spend",
                        },
                        "segment_playbook": {
                            "confidence": "medium",
                            "supporting_evidence": {
                                "top_strategic_roles": [
                                    {
                                        "role_type": "economic_buyer",
                                        "source_id": "segment:role:economic_buyer",
                                    },
                                ],
                                "top_departments": [
                                    {"department": "finance", "source_id": "segment:department:finance"},
                                ],
                                "top_contract_segments": [
                                    {"segment": "enterprise", "source_id": "segment:contract:enterprise"},
                                ],
                            },
                        },
                        "timing_intelligence": {
                            "confidence": "medium",
                            "best_timing_window": "Before renewal",
                            "active_eval_signals": {
                                "value": 2,
                                "source_id": "accounts:summary:active_eval_signal_count",
                            },
                            "sentiment_direction": "declining",
                            "immediate_triggers": [
                                {"trigger": "Q2 renewal", "type": "deadline"},
                            ],
                        },
                    },
                    "displacement_reasoning": {
                        "schema_version": "v1",
                        "migration_proof": {
                            "confidence": "medium",
                            "switch_volume": {
                                "value": 0,
                                "source_id": "displacement:aggregate:total_explicit_switches",
                            },
                        },
                        "competitive_reframes": {"confidence": "medium"},
                    },
                    "account_reasoning": {
                        "schema_version": "v1",
                        "market_summary": "Two accounts are actively evaluating alternatives while nine accounts show high intent signals overall.",
                        "total_accounts": {
                            "value": 9,
                            "source_id": "accounts:summary:total_accounts",
                        },
                        "high_intent_count": {
                            "value": 9,
                            "source_id": "accounts:summary:high_intent_count",
                        },
                        "active_eval_count": {
                            "value": 2,
                            "source_id": "accounts:summary:active_eval_signal_count",
                        },
                        "top_accounts": [
                            {
                                "name": "Acme Corp",
                                "intent_score": 0.9,
                                "source_id": "accounts:company:acme_corp",
                            },
                        ],
                    },
                },
                "meta": {
                    "evidence_window_start": "2026-03-01",
                    "evidence_window_end": "2026-03-18",
                },
            },
            "Zendesk",
        )

        brief = _build_challenger_brief(
            incumbent="Zendesk",
            challenger="Freshdesk",
            displacement_detail=self._minimal_displacement(),
            battle_card=None,
            accounts_in_motion=None,
            incumbent_profile=None,
            challenger_profile=None,
            incumbent_evidence_vault=None,
            churn_signal=None,
            incumbent_synthesis_view=synthesis_view,
            cross_vendor_battle=None,
            max_target_accounts=15,
        )

        assert brief["data_sources"]["reasoning_synthesis"] is True
        assert brief["data_sources"]["account_reasoning"] is True
        assert brief["reasoning_contracts"]["vendor_core_reasoning"]["causal_narrative"]["trigger"] == "Price hike"
        assert brief["reasoning_contracts"]["displacement_reasoning"]["migration_proof"]["confidence"] == "medium"
        assert brief["account_reasoning"]["top_accounts"][0]["name"] == "Acme Corp"
        assert brief["incumbent_profile"]["account_pressure_summary"] == (
            "Two accounts are actively evaluating alternatives while nine accounts show high intent signals overall."
        )
        assert brief["incumbent_profile"]["account_pressure_metrics"]["high_intent_count"] == 9
        assert brief["target_accounts_source"] == "account_reasoning"
        assert brief["target_accounts"][0]["company"] == "Acme Corp"
        assert brief["target_accounts"][0]["reasoning_backed"] is True
        assert brief["segment_playbook"]["supporting_evidence"]["top_departments"][0]["department"] == "finance"
        assert brief["timing_intelligence"]["best_timing_window"] == "Before renewal"
        assert brief["timing_summary"] == (
            "Before renewal. 2 active evaluation signals are visible right now. "
            "Review sentiment is skewing more negative."
        )
        assert brief["timing_metrics"]["active_eval_signals"] == 2
        assert brief["priority_timing_triggers"] == ["Q2 renewal"]
        assert brief["incumbent_profile"]["timing_intelligence"]["best_timing_window"] == "Before renewal"
        assert brief["incumbent_profile"]["timing_metrics"]["active_eval_signals"] == 2
        assert "economic buyers" in brief["segment_targeting_summary"]
        assert "Finance teams" in brief["segment_targeting_summary"]
        assert "enterprise contracts" in brief["segment_targeting_summary"]
        assert "causal_narrative" not in brief["incumbent_profile"]
        assert "synthesis_wedge" not in brief["incumbent_profile"]
        assert brief["evidence_window_days"] == 17
        assert brief["reasoning_source"] == "b2b_reasoning_synthesis"

    def test_empty_synthesis_view_does_not_claim_reasoning_source(self):
        """Presence of a synthesis row alone should not mark the brief as synthesis-backed."""
        synthesis_view = load_synthesis_view(
            {
                "schema_version": "2.1",
                "reasoning_contracts": {
                    "vendor_core_reasoning": {},
                    "displacement_reasoning": {},
                },
                "meta": {},
            },
            "Zendesk",
        )

        brief = _build_challenger_brief(
            incumbent="Zendesk",
            challenger="Freshdesk",
            displacement_detail=self._minimal_displacement(),
            battle_card=None,
            accounts_in_motion=None,
            incumbent_profile=None,
            challenger_profile=None,
            incumbent_evidence_vault=None,
            churn_signal=None,
            incumbent_synthesis_view=synthesis_view,
            cross_vendor_battle=None,
            max_target_accounts=15,
        )

        assert brief["data_sources"]["reasoning_synthesis"] is False
        assert "reasoning_source" not in brief

    def test_flat_section_synthesis_view_surfaces_segment_playbook(self):
        synthesis_view = load_synthesis_view(
            {
                "schema_version": "2.1",
                "causal_narrative": {
                    "primary_wedge": "price_squeeze",
                    "confidence": "high",
                    "trigger": "Price hike",
                },
                "segment_playbook": {
                    "confidence": "medium",
                    "supporting_evidence": {
                        "top_roles": [
                            {"role_type": "economic_buyer", "source_id": "segment:role:economic_buyer"},
                        ],
                    },
                },
                "timing_intelligence": {
                    "confidence": "medium",
                    "best_timing_window": "Before renewal",
                },
            },
            "Zendesk",
        )

        brief = _build_challenger_brief(
            incumbent="Zendesk",
            challenger="Freshdesk",
            displacement_detail=self._minimal_displacement(),
            battle_card=None,
            accounts_in_motion=None,
            incumbent_profile=None,
            challenger_profile=None,
            incumbent_evidence_vault=None,
            churn_signal=None,
            incumbent_synthesis_view=synthesis_view,
            cross_vendor_battle=None,
            max_target_accounts=15,
        )

        assert brief["data_sources"]["reasoning_synthesis"] is True
        assert brief["segment_playbook"]["supporting_evidence"]["top_roles"][0]["role_type"] == "economic_buyer"
        assert "economic buyers" in brief["segment_targeting_summary"]
        assert "Best tested before renewal." in brief["segment_targeting_summary"]
        assert brief["reasoning_contracts"]["vendor_core_reasoning"]["segment_playbook"]["confidence"] == "medium"

    def test_segment_targeting_summary_formats_duration_without_bad_preposition(self):
        synthesis_view = load_synthesis_view(
            {
                "schema_version": "2.1",
                "segment_playbook": {
                    "confidence": "medium",
                    "supporting_evidence": {
                        "top_roles": [
                            {"role_type": "economic_buyer", "source_id": "segment:role:economic_buyer"},
                        ],
                        "top_usage_durations": [
                            {"duration": "1 year", "source_id": "segment:duration:1_year"},
                        ],
                    },
                },
                "timing_intelligence": {
                    "confidence": "medium",
                    "best_timing_window": "Before renewal",
                },
            },
            "Zendesk",
        )

        brief = _build_challenger_brief(
            incumbent="Zendesk",
            challenger="Freshdesk",
            displacement_detail=self._minimal_displacement(),
            battle_card=None,
            accounts_in_motion=None,
            incumbent_profile=None,
            challenger_profile=None,
            incumbent_evidence_vault=None,
            churn_signal=None,
            incumbent_synthesis_view=synthesis_view,
            cross_vendor_battle=None,
            max_target_accounts=15,
        )

        assert "especially after 1 year of usage." in brief["segment_targeting_summary"]
        assert "in after" not in brief["segment_targeting_summary"]

    def test_segment_targeting_summary_normalizes_priority_segment_labels_and_uses_timing(self):
        synthesis_view = load_synthesis_view(
            {
                "schema_version": "2.1",
                "segment_playbook": {
                    "confidence": "medium",
                    "priority_segments": [
                        {
                            "segment": "end user role",
                            "best_opening_angle": "Offer a cost-control benchmark",
                        },
                    ],
                },
                "timing_intelligence": {
                    "confidence": "medium",
                    "best_timing_window": "Before renewal",
                },
            },
            "Zendesk",
        )

        brief = _build_challenger_brief(
            incumbent="Zendesk",
            challenger="Freshdesk",
            displacement_detail=self._minimal_displacement(),
            battle_card=None,
            accounts_in_motion=None,
            incumbent_profile=None,
            challenger_profile=None,
            incumbent_evidence_vault=None,
            churn_signal=None,
            incumbent_synthesis_view=synthesis_view,
            cross_vendor_battle=None,
            max_target_accounts=15,
        )

        assert "end users" in brief["segment_targeting_summary"]
        assert "end user role" not in brief["segment_targeting_summary"]
        assert "led with offer a cost control benchmark." in brief["segment_targeting_summary"]
        assert "Best tested before renewal." in brief["segment_targeting_summary"]

    def test_segment_targeting_summary_uses_company_size_when_available(self):
        synthesis_view = load_synthesis_view(
            {
                "schema_version": "2.1",
                "segment_playbook": {
                    "confidence": "medium",
                    "supporting_evidence": {
                        "top_roles": [
                            {"role_type": "economic_buyer", "source_id": "segment:role:economic_buyer"},
                        ],
                        "top_company_sizes": [
                            {"segment": "Mid-Market", "source_id": "segment:size:mid_market"},
                        ],
                        "top_contract_segments": [
                            {"segment": "smb", "source_id": "segment:contract:smb"},
                        ],
                    },
                },
            },
            "Zendesk",
        )

        brief = _build_challenger_brief(
            incumbent="Zendesk",
            challenger="Freshdesk",
            displacement_detail=self._minimal_displacement(),
            battle_card=None,
            accounts_in_motion=None,
            incumbent_profile=None,
            challenger_profile=None,
            incumbent_evidence_vault=None,
            churn_signal=None,
            incumbent_synthesis_view=synthesis_view,
            cross_vendor_battle=None,
            max_target_accounts=15,
        )

        assert "Mid Market accounts" in brief["segment_targeting_summary"]
        assert "smb contracts" in brief["segment_targeting_summary"]

    def test_segment_targeting_summary_dedupes_matching_size_and_contract_labels(self):
        synthesis_view = load_synthesis_view(
            {
                "schema_version": "2.1",
                "segment_playbook": {
                    "confidence": "medium",
                    "supporting_evidence": {
                        "top_roles": [
                            {"role_type": "evaluator", "source_id": "segment:role:evaluator"},
                        ],
                        "top_company_sizes": [
                            {"segment": "smb", "source_id": "segment:size:smb"},
                        ],
                        "top_contract_segments": [
                            {"segment": "SMB", "source_id": "segment:contract:smb"},
                        ],
                    },
                },
            },
            "Zendesk",
        )

        brief = _build_challenger_brief(
            incumbent="Zendesk",
            challenger="Freshdesk",
            displacement_detail=self._minimal_displacement(),
            battle_card=None,
            accounts_in_motion=None,
            incumbent_profile=None,
            challenger_profile=None,
            incumbent_evidence_vault=None,
            churn_signal=None,
            incumbent_synthesis_view=synthesis_view,
            cross_vendor_battle=None,
            max_target_accounts=15,
        )

        summary = brief["segment_targeting_summary"].lower()
        assert "smb accounts" in summary
        assert "smb contracts" not in summary

    def test_timing_summary_capitalizes_best_timing_window_text(self):
        synthesis_view = load_synthesis_view(
            {
                "schema_version": "2.1",
                "timing_intelligence": {
                    "confidence": "medium",
                    "best_timing_window": "immediate - pricing pressure is current",
                    "active_eval_signals": {
                        "value": 2,
                        "source_id": "accounts:summary:active_eval_signal_count",
                    },
                },
            },
            "Zendesk",
        )

        brief = _build_challenger_brief(
            incumbent="Zendesk",
            challenger="Freshdesk",
            displacement_detail=self._minimal_displacement(),
            battle_card=None,
            accounts_in_motion=None,
            incumbent_profile=None,
            challenger_profile=None,
            incumbent_evidence_vault=None,
            churn_signal=None,
            incumbent_synthesis_view=synthesis_view,
            cross_vendor_battle=None,
            max_target_accounts=15,
        )

        assert brief["timing_summary"].startswith(
            "Immediate - pricing pressure is current."
        )

    def test_timing_summary_adds_concrete_trigger_when_window_is_generic(self):
        synthesis_view = load_synthesis_view(
            {
                "schema_version": "2.1",
                "timing_intelligence": {
                    "confidence": "medium",
                    "best_timing_window": "Immediate - active evaluation signals are already present",
                    "active_eval_signals": {
                        "value": 3,
                        "source_id": "accounts:summary:active_eval_signal_count",
                    },
                    "immediate_triggers": [
                        {"trigger": "Active evaluation of BambooHR (3 accounts)", "type": "signal"},
                        {"trigger": "Support-related fee increase turning point", "type": "signal"},
                    ],
                },
            },
            "Zendesk",
        )

        brief = _build_challenger_brief(
            incumbent="Zendesk",
            challenger="Freshdesk",
            displacement_detail=self._minimal_displacement(),
            battle_card=None,
            accounts_in_motion=None,
            incumbent_profile=None,
            challenger_profile=None,
            incumbent_evidence_vault=None,
            churn_signal=None,
            incumbent_synthesis_view=synthesis_view,
            cross_vendor_battle=None,
            max_target_accounts=15,
        )

        assert brief["timing_summary"].startswith(
            "Immediate - buyers are already evaluating alternatives."
        )
        assert "Key trigger: Active evaluation of BambooHR (3 accounts)." in brief["timing_summary"]

    def test_timing_summary_drops_contradictory_no_signal_window_when_active_eval_exists(self):
        synthesis_view = load_synthesis_view(
            {
                "schema_version": "2.1",
                "timing_intelligence": {
                    "confidence": "medium",
                    "best_timing_window": "none - no active evaluation or deadline signals detected",
                    "active_eval_signals": {
                        "value": 3,
                        "source_id": "accounts:summary:active_eval_signal_count",
                    },
                },
            },
            "Zendesk",
        )

        brief = _build_challenger_brief(
            incumbent="Zendesk",
            challenger="Freshdesk",
            displacement_detail=self._minimal_displacement(),
            battle_card=None,
            accounts_in_motion=None,
            incumbent_profile=None,
            challenger_profile=None,
            incumbent_evidence_vault=None,
            churn_signal=None,
            incumbent_synthesis_view=synthesis_view,
            cross_vendor_battle=None,
            max_target_accounts=15,
        )

        assert brief["timing_summary"] == "3 active evaluation signals are visible right now."
        assert "no active evaluation" not in brief["timing_summary"].lower()

    def test_executive_summary_format(self):
        """Executive summary includes incumbent, challenger, mentions, accounts."""
        brief = _build_challenger_brief(
            incumbent="Zendesk",
            challenger="Freshdesk",
            displacement_detail=self._minimal_displacement(),
            battle_card=None,
            accounts_in_motion={"accounts": [
                {
                    "company": "A",
                    "opportunity_score": 80,
                    "urgency": 8,
                    "buying_stage": "eval",
                    "alternatives_considering": ["Freshdesk"],
                },
                {
                    "company": "B",
                    "opportunity_score": 60,
                    "urgency": 6,
                    "buying_stage": "eval",
                    "alternatives_considering": [],
                },
            ]},
            incumbent_profile=None,
            challenger_profile=None,
            churn_signal=None,
            cross_vendor_battle=None,
            max_target_accounts=15,
        )
        summary = brief["_executive_summary"]
        assert "Freshdesk" in summary
        assert "Zendesk" in summary
        assert "10 displacement mentions" in summary
        assert "2 target accounts" in summary
        assert "1 considering Freshdesk" in summary

    def test_category_from_challenger_profile_fallback(self):
        """Category falls back to challenger profile if incumbent has none."""
        brief = _build_challenger_brief(
            incumbent="Zendesk",
            challenger="Freshdesk",
            displacement_detail=self._minimal_displacement(),
            battle_card=None,
            accounts_in_motion=None,
            incumbent_profile=None,
            challenger_profile={"strengths": [], "weaknesses": [],
                                "pain_addressed": [], "commonly_switched_from": [],
                                "top_integrations": [], "profile_summary": "",
                                "category": "Support"},
            churn_signal=None,
            cross_vendor_battle=None,
            max_target_accounts=15,
        )
        assert brief["category"] == "Support"

    def test_zero_numeric_metrics_preserved(self):
        """Legitimate 0.0 in churn_signal must not fall through to battle card."""
        battle_card = {
            "vendor_weaknesses": [],
            "customer_pain_quotes": [],
            "churn_pressure_score": 50.0,
            "risk_level": "medium",
            "objection_data": {
                "price_complaint_rate": 0.999,
                "dm_churn_rate": 0.999,
                "sentiment_direction": "positive",
            },
        }
        churn_signal = {
            "archetype": "stable",
            "archetype_confidence": 0.9,
            "risk_level": "low",
            "key_signals": [],
            "churn_pressure_score": 0.0,
            "sentiment_direction": None,
            "price_complaint_rate": 0.0,
            "dm_churn_rate": 0.0,
        }
        brief = _build_challenger_brief(
            incumbent="Zendesk",
            challenger="Freshdesk",
            displacement_detail={
                "total_mentions": 5,
                "signal_strength": "emerging",
                "confidence_score": 0.5,
                "primary_driver": None,
                "key_quote": None,
                "source_distribution": {},
            },
            battle_card=battle_card,
            accounts_in_motion=None,
            incumbent_profile=None,
            challenger_profile=None,
            churn_signal=churn_signal,
            cross_vendor_battle=None,
            max_target_accounts=15,
        )
        inc = brief["incumbent_profile"]
        # 0.0 from churn_signal must win over battle card values
        assert inc["churn_pressure_score"] == 0.0
        assert inc["price_complaint_rate"] == 0.0
        assert inc["dm_churn_rate"] == 0.0
        assert inc["risk_level"] == "low"
        # sentiment_direction is None in churn_signal, should fall back
        assert inc["sentiment_direction"] == "positive"


# ---------------------------------------------------------------------------
# Cross-vendor battle fetch tests (async, mock pool)
# ---------------------------------------------------------------------------

def _make_xv_row(conclusion_dict: dict, confidence: float = 0.80):
    """Build a dict mimicking an asyncpg Record for b2b_cross_vendor_conclusions."""
    return {
        "conclusion": conclusion_dict,
        "confidence": confidence,
    }


class TestFetchCrossVendorBattle:
    """Verify _fetch_cross_vendor_battle picks the newest prior row
    when no same-day conclusion exists, and correctly parses the
    conclusion JSONB."""

    @pytest.mark.asyncio
    async def test_stale_conclusion_used_when_today_missing(self):
        """If no row exists for today, the most recent prior row is returned."""
        stale_conclusion = {
            "winner": "Freshdesk",
            "loser": "Zendesk",
            "conclusion": "Freshdesk wins on SMB pricing",
            "durability_assessment": "Durable: structural pricing gap",
            "key_insights": ["Wins deals under 50k ARR"],
        }
        pool = MagicMock()
        pool.fetchrow = AsyncMock(return_value=_make_xv_row(stale_conclusion, 0.75))

        today = date(2026, 3, 17)
        result = await _fetch_cross_vendor_battle(pool, "Zendesk", "Freshdesk", today)

        # Verify the query was called with <= today (not = today)
        sql = pool.fetchrow.call_args[0][0]
        assert "computed_date <= $1" in sql
        assert "ORDER BY computed_date DESC" in sql

        assert result is not None
        assert result["winner"] == "Freshdesk"
        assert result["loser"] == "Zendesk"
        assert result["conclusion"] == "Freshdesk wins on SMB pricing"
        assert result["durability"] == "Durable: structural pricing gap"
        assert result["key_insights"] == [{"insight": "Wins deals under 50k ARR", "evidence": "Wins deals under 50k ARR"}]
        assert result["confidence"] == 0.75

    @pytest.mark.asyncio
    async def test_returns_none_when_no_rows(self):
        """If no conclusion exists at all, returns None."""
        pool = MagicMock()
        pool.fetchrow = AsyncMock(return_value=None)

        result = await _fetch_cross_vendor_battle(
            pool, "Zendesk", "Freshdesk", date(2026, 3, 17),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_key_insights_dict_normalization(self):
        """key_insights containing dicts are normalized to insight/evidence objects."""
        conclusion = {
            "winner": "A",
            "conclusion": "A wins",
            "key_insights": [
                {"insight": "Point one"},
                "Point two",
                {"text": "Point three"},
                {"irrelevant": "skipped"},
            ],
        }
        pool = MagicMock()
        pool.fetchrow = AsyncMock(return_value=_make_xv_row(conclusion))

        result = await _fetch_cross_vendor_battle(
            pool, "A", "B", date(2026, 3, 17),
        )
        assert result["key_insights"] == [
            {"insight": "Point one", "evidence": "Point one"},
            {"insight": "Point two", "evidence": "Point two"},
            {"insight": "Point three", "evidence": "Point three"},
        ]

    @pytest.mark.asyncio
    async def test_durability_fallback_to_displacement_flows_nature(self):
        """durability falls back to displacement_flows_nature if durability_assessment absent."""
        conclusion = {
            "winner": "A",
            "conclusion": "A wins",
            "displacement_flows_nature": "Cyclical: tied to renewal cycles",
        }
        pool = MagicMock()
        pool.fetchrow = AsyncMock(return_value=_make_xv_row(conclusion))

        result = await _fetch_cross_vendor_battle(
            pool, "A", "B", date(2026, 3, 17),
        )
        assert result["durability"] == "Cyclical: tied to renewal cycles"

    @pytest.mark.asyncio
    async def test_string_conclusion_parsed_as_json(self):
        """If asyncpg returns conclusion as a string, it is JSON-parsed."""
        conclusion_str = json.dumps({
            "winner": "B",
            "conclusion": "B wins",
            "durability_assessment": "Stable",
            "key_insights": ["Insight 1"],
        })
        pool = MagicMock()
        pool.fetchrow = AsyncMock(return_value={
            "conclusion": conclusion_str,
            "confidence": 0.65,
        })

        result = await _fetch_cross_vendor_battle(
            pool, "A", "B", date(2026, 3, 17),
        )
        assert result["winner"] == "B"
        assert result["confidence"] == 0.65


class TestResolveCrossVendorBattle:
    @pytest.mark.asyncio
    async def test_prefers_synthesis_over_legacy_fallback(self, monkeypatch):
        fallback = AsyncMock(return_value={
            "conclusion": "Legacy battle",
            "winner": "Legacy",
        })
        monkeypatch.setattr(brief_mod, "_fetch_cross_vendor_battle", fallback)

        xv_lookup = {
            "battles": {
                ("Freshdesk", "Zendesk"): {
                    "conclusion": {
                        "conclusion": "Freshdesk wins on SMB pricing",
                        "winner": "Freshdesk",
                        "loser": "Zendesk",
                        "confidence": 0.82,
                        "durability_assessment": "structural",
                        "key_insights": ["Wins sub-$50k deals"],
                    },
                },
            },
        }

        result = await _resolve_cross_vendor_battle(
            MagicMock(), "Zendesk", "Freshdesk", date(2026, 3, 29), xv_lookup,
        )

        assert result["winner"] == "Freshdesk"
        assert result["conclusion"] == "Freshdesk wins on SMB pricing"
        assert result["key_insights"] == [
            {"insight": "Wins sub-$50k deals", "evidence": "Wins sub-$50k deals"},
        ]
        fallback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_falls_back_when_synthesis_missing(self, monkeypatch):
        fallback = AsyncMock(return_value={
            "conclusion": "Legacy battle",
            "winner": "Freshdesk",
            "loser": "Zendesk",
            "durability": "legacy",
            "key_insights": [],
            "confidence": 0.61,
        })
        monkeypatch.setattr(brief_mod, "_fetch_cross_vendor_battle", fallback)

        result = await _resolve_cross_vendor_battle(
            MagicMock(),
            "Zendesk",
            "Freshdesk",
            date(2026, 3, 29),
            {"battles": {}},
        )

        assert result["conclusion"] == "Legacy battle"
        fallback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_case_insensitive_synthesis_key_match(self, monkeypatch):
        fallback = AsyncMock(return_value=None)
        monkeypatch.setattr(brief_mod, "_fetch_cross_vendor_battle", fallback)

        xv_lookup = {
            "battles": {
                ("FreshDesk", "ZENdesk"): {
                    "conclusion": {
                        "conclusion": "Freshdesk is gaining on cost",
                        "winner": "Freshdesk",
                        "loser": "Zendesk",
                        "confidence": 0.7,
                        "durability_assessment": "durable",
                        "key_insights": [
                            {"text": "Lower admin burden"},
                            "Faster rollout",
                        ],
                    },
                },
            },
        }

        result = await _resolve_cross_vendor_battle(
            MagicMock(), "zendesk", "freshdesk", date(2026, 3, 29), xv_lookup,
        )

        assert result["winner"] == "Freshdesk"
        assert result["key_insights"] == [
            {"insight": "Lower admin burden", "evidence": "Lower admin burden"},
            {"insight": "Faster rollout", "evidence": "Faster rollout"},
        ]
        fallback.assert_not_awaited()


class TestChallengerBriefFallbacks:
    @pytest.mark.asyncio
    async def test_freshness_only_requires_core_run_marker(self):
        pool = MagicMock()
        pool.fetchval = AsyncMock(return_value=1)
        assert await _check_freshness(pool) == date.today()

    @pytest.mark.asyncio
    async def test_retire_unselected_challenger_briefs_deletes_stale_pairs(self):
        keep_id = uuid4()
        stale_id = uuid4()
        pool = MagicMock()
        pool.fetch = AsyncMock(return_value=[
            {
                "id": keep_id,
                "vendor_filter": "Zendesk",
                "category_filter": "Freshdesk",
            },
            {
                "id": stale_id,
                "vendor_filter": "Azure",
                "category_filter": "Google Workspace",
            },
        ])
        pool.execute = AsyncMock()

        retired = await _retire_unselected_challenger_briefs(
            pool,
            today=date(2026, 3, 21),
            pairs=[{"incumbent": "Zendesk", "challenger": "Freshdesk"}],
        )

        assert retired == 1
        args = pool.execute.await_args.args
        assert args[0] == "DELETE FROM b2b_intelligence WHERE id = ANY($1::uuid[])"
        assert args[1] == [stale_id]

    @pytest.mark.asyncio
    async def test_fetch_persisted_report_record_marks_stale_rows(self):
        pool = MagicMock()
        pool.fetchrow = AsyncMock(return_value={
            "intelligence_data": {"vendor": "Zendesk"},
            "report_date": date(2026, 3, 16),
        })
        record = await _fetch_persisted_report_record(
            pool, "battle_card", "Zendesk", date(2026, 3, 17), fallback_days=7,
        )
        assert record == {
            "data": {"vendor": "Zendesk"},
            "report_date": date(2026, 3, 16),
            "stale": True,
        }

    def test_extract_pain_quotes_from_reviews_keeps_rich_fields_and_dedupes(self):
        rows = [
            {
                "vendor_name": "Zendesk",
                "source": "reddit",
                "company": "Acme",
                "role": "Director",
                "pain_category": "support",
                "phrases": ["Support disappeared during onboarding", "Support disappeared during onboarding"],
                "urgency": 9.0,
            },
            {
                "vendor_name": "Zendesk",
                "source": "g2",
                "company": "Beta",
                "role": "VP Ops",
                "pain_category": "support",
                "phrases": ["Support disappeared during onboarding for our new team"],
                "urgency": 8.5,
            },
        ]
        quotes = _extract_pain_quotes_from_reviews(
            rows,
            vendor="Zendesk",
            max_quotes=5,
            min_urgency=6.0,
            similarity_threshold=0.5,
        )
        assert len(quotes) == 1
        assert quotes[0]["company"] == "Acme"
        assert quotes[0]["role"] == "Director"
        assert quotes[0]["pain_category"] == "support"

    def test_normalize_product_profile_weaknesses_matches_brief_shape(self):
        normalized = _normalize_product_profile_weaknesses([
            {"aspect": "support", "score": 2.8, "review_count": 45},
        ])
        assert normalized == [{
            "weakness": "support",
            "area": "support",
            "evidence": "Satisfaction score 2.8/5.0 across 45 reviews",
            "mention_count": 45,
            "count": 45,
            "source": "product_profile",
        }]


class TestChallengerBriefRunProgress:
    @pytest.mark.asyncio
    async def test_run_emits_progress_metadata_across_stages(self, monkeypatch):
        progress = AsyncMock()
        pool = type("Pool", (), {
            "is_initialized": True,
            "execute": AsyncMock(),
            "fetch": AsyncMock(return_value=[]),
        })()

        async def fake_gather(*_args, **_kwargs):
            for coro in _args:
                close = getattr(coro, "close", None)
                if close:
                    close()
            return (
                None,
                None,
                {"total_mentions": 3, "source_distribution": {"reddit": 3}},
                None,
                None,
                None,
                None,
                None,
                [],
            )

        monkeypatch.setattr(brief_mod.settings.b2b_churn, "enabled", True, raising=False)
        monkeypatch.setattr(brief_mod.settings.b2b_churn, "intelligence_enabled", True, raising=False)
        monkeypatch.setattr(brief_mod, "_update_execution_progress", progress)
        monkeypatch.setattr(brief_mod, "get_db_pool", lambda: pool)
        monkeypatch.setattr(brief_mod, "_check_freshness", AsyncMock(return_value=date(2026, 3, 18)))
        monkeypatch.setattr(brief_mod, "_fetch_latest_evidence_vault", AsyncMock(return_value={}))
        monkeypatch.setattr(
            brief_mod,
            "_select_displacement_pairs",
            AsyncMock(return_value=[{"incumbent": "Zendesk", "challenger": "Freshdesk"}]),
        )
        monkeypatch.setattr(brief_mod.asyncio, "gather", fake_gather)
        monkeypatch.setattr(
            brief_mod,
            "_build_challenger_brief",
            lambda **kwargs: {
                "_executive_summary": "summary",
                "displacement_summary": {"total_mentions": 3, "source_distribution": {"reddit": 3}},
                "data_sources": {"battle_card": False},
                "total_target_accounts": 0,
            },
        )

        task = type("Task", (), {"metadata": {"_execution_id": str(uuid4())}})()
        result = await brief_mod.run(task)

        assert result == {
            "_skip_synthesis": "Challenger briefs complete",
            "pairs": 1,
            "persisted": 1,
            "briefs_retired": 0,
        }
        stages = [call.kwargs["stage"] for call in progress.await_args_list]
        assert stages == [
            brief_mod._STAGE_SELECTING_PAIRS,
            brief_mod._STAGE_BUILDING_BRIEFS,
            brief_mod._STAGE_BUILDING_BRIEFS,
            brief_mod._STAGE_FINALIZING,
        ]

    @pytest.mark.asyncio
    async def test_run_scopes_pairs_to_test_vendors(self, monkeypatch):
        pool = type("Pool", (), {
            "is_initialized": True,
            "execute": AsyncMock(),
            "fetch": AsyncMock(return_value=[]),
        })()

        async def fake_gather(*_args, **_kwargs):
            for coro in _args:
                close = getattr(coro, "close", None)
                if close:
                    close()
            return (
                None,
                None,
                {"total_mentions": 3, "source_distribution": {"reddit": 3}},
                None,
                None,
                None,
                None,
                None,
                [],
            )

        monkeypatch.setattr(brief_mod.settings.b2b_churn, "enabled", True, raising=False)
        monkeypatch.setattr(brief_mod.settings.b2b_churn, "intelligence_enabled", True, raising=False)
        monkeypatch.setattr(brief_mod, "_update_execution_progress", AsyncMock())
        monkeypatch.setattr(brief_mod, "get_db_pool", lambda: pool)
        monkeypatch.setattr(brief_mod, "_check_freshness", AsyncMock(return_value=date(2026, 3, 18)))
        monkeypatch.setattr(brief_mod, "_fetch_latest_evidence_vault", AsyncMock(return_value={}))
        monkeypatch.setattr(
            brief_mod,
            "_select_displacement_pairs",
            AsyncMock(return_value=[
                {"incumbent": "Zendesk", "challenger": "Freshdesk"},
                {"incumbent": "HubSpot", "challenger": "Pipedrive"},
            ]),
        )
        monkeypatch.setattr(brief_mod.asyncio, "gather", fake_gather)
        monkeypatch.setattr(
            brief_mod,
            "_build_challenger_brief",
            lambda **kwargs: {
                "_executive_summary": "summary",
                "displacement_summary": {"total_mentions": 3, "source_distribution": {"reddit": 3}},
                "data_sources": {"battle_card": False},
                "total_target_accounts": 0,
            },
        )

        task = type("Task", (), {"metadata": {"_execution_id": str(uuid4()), "test_vendors": ["Zendesk"]}})()
        result = await brief_mod.run(task)

        assert result == {
            "_skip_synthesis": "Challenger briefs complete",
            "pairs": 1,
            "persisted": 1,
            "briefs_retired": 0,
        }


@pytest.mark.asyncio
class TestResolveCrossVendorBattle:
    """Unit tests for _resolve_cross_vendor_battle synthesis-first path."""

    _SYNTH_ENTRY = {
        "conclusion": {
            "conclusion": "Freshdesk gaining on pricing advantage",
            "winner": "Freshdesk",
            "loser": "Zendesk",
            "confidence": 0.85,
            "durability_assessment": "structural",
            "key_insights": [{"insight": "Price gap", "evidence": "0.28 vs 0.24"}],
        },
    }

    async def test_prefers_synthesis_over_legacy(self):
        xv = {"battles": {("Freshdesk", "Zendesk"): self._SYNTH_ENTRY}}
        pool = MagicMock()
        pool.fetchrow = AsyncMock(return_value=None)

        result = await brief_mod._resolve_cross_vendor_battle(
            pool, "Zendesk", "Freshdesk", date(2026, 3, 29), xv,
        )

        assert result is not None
        assert result["winner"] == "Freshdesk"
        assert "pricing" in result["conclusion"]
        pool.fetchrow.assert_not_called()

    async def test_falls_back_to_legacy_when_no_synthesis(self):
        xv = {"battles": {}}
        pool = MagicMock()
        pool.fetchrow = AsyncMock(return_value=None)

        result = await brief_mod._resolve_cross_vendor_battle(
            pool, "Zendesk", "Freshdesk", date(2026, 3, 29), xv,
        )

        assert result is None
        pool.fetchrow.assert_called_once()

    async def test_case_insensitive_key_match(self):
        xv = {"battles": {("Freshdesk", "Zendesk"): self._SYNTH_ENTRY}}
        pool = MagicMock()
        pool.fetchrow = AsyncMock(return_value=None)

        result = await brief_mod._resolve_cross_vendor_battle(
            pool, "zendesk", "freshdesk", date(2026, 3, 29), xv,
        )

        assert result is not None
        assert result["winner"] == "Freshdesk"
        pool.fetchrow.assert_not_called()

    async def test_output_shape_matches_legacy(self):
        xv = {"battles": {("Freshdesk", "Zendesk"): self._SYNTH_ENTRY}}
        pool = MagicMock()

        result = await brief_mod._resolve_cross_vendor_battle(
            pool, "Zendesk", "Freshdesk", date(2026, 3, 29), xv,
        )

        for key in ("conclusion", "winner", "loser", "durability", "key_insights", "confidence"):
            assert key in result

    async def test_string_key_insights_normalized(self):
        entry = {
            "conclusion": {
                "conclusion": "Test",
                "winner": "A",
                "loser": "B",
                "confidence": 0.7,
                "key_insights": ["Plain string", {"insight": "Dict", "evidence": "m: 42"}],
            },
        }
        xv = {"battles": {("a", "b"): entry}}
        pool = MagicMock()

        result = await brief_mod._resolve_cross_vendor_battle(
            pool, "A", "B", date(2026, 3, 29), xv,
        )

        assert len(result["key_insights"]) == 2
        assert result["key_insights"][0] == {"insight": "Plain string", "evidence": "Plain string"}
        assert result["key_insights"][1] == {"insight": "Dict", "evidence": "m: 42"}
