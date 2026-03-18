"""Tests for challenger_brief report type.

Covers weakness coverage, target account filtering, integration comparison,
brief assembly, executive summary format, and self-flow skip.
"""

import json
import sys
from datetime import date
from typing import Any
from unittest.mock import AsyncMock, MagicMock

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
    "starlette", "starlette.requests",
    "playwright", "playwright.async_api",
    "playwright_stealth",
    "curl_cffi", "curl_cffi.requests",
    "pytrends", "pytrends.request",
):
    sys.modules.setdefault(_mod, MagicMock())

from atlas_brain.autonomous.tasks.b2b_challenger_brief import (
    _compute_weakness_coverage,
    _fetch_cross_vendor_battle,
    _filter_target_accounts,
    _build_integration_comparison,
    _build_challenger_brief,
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
        assert brief["head_to_head"]["confidence"] == 0.78
        assert len(brief["target_accounts"]) == 1
        assert brief["target_accounts"][0]["considers_challenger"] is True
        assert brief["accounts_considering_challenger"] == 1
        assert brief["sales_playbook"]["discovery_questions"] == ["What's your current cost per seat?"]
        assert "Marketo" in brief["integration_comparison"]["challenger_exclusive"]
        assert "Legacy CRM" in brief["integration_comparison"]["incumbent_exclusive"]
        assert all(brief["data_sources"].values())
        assert brief["category"] == "Helpdesk"

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
        assert result["conclusion"] == "Freshdesk wins on SMB pricing"
        assert result["durability"] == "Durable: structural pricing gap"
        assert result["key_insights"] == ["Wins deals under 50k ARR"]
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
        """key_insights containing dicts with 'insight' key are normalized to strings."""
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
        assert result["key_insights"] == ["Point one", "Point two", "Point three"]

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
