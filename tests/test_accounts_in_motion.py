"""Tests for accounts_in_motion report type.

Covers scoring, merging, aggregate building, and short-circuit paths.
"""

import sys
from datetime import date
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

# Pre-mock heavy deps before importing task module
# asyncpg needs a real exception class for `except UndefinedTableError`
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

import atlas_brain.autonomous.tasks.b2b_accounts_in_motion as accounts_mod
import atlas_brain.autonomous.tasks.b2b_churn_intelligence as churn_mod
from atlas_brain.autonomous.tasks.b2b_accounts_in_motion import (
    _apply_account_quality_adjustments,
    _build_fallback_intent_rows,
    _build_signal_metadata_fallback_rows,
    _compute_account_opportunity_score,
    _merge_company_profiles,
    _normalize_company_key,
    _summarize_vendor_evidence,
    _build_vendor_aggregate,
)


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

class TestAccountOpportunityScore:
    def test_max_score(self):
        """DM + active_purchase + 500 seats + urgency 10 + 3 alternatives = 100."""
        account = {
            "urgency": 10,
            "decision_maker": True,
            "buying_stage": "active_purchase",
            "seat_count": 500,
            "alternatives_considering": ["A", "B", "C"],
        }
        score, components = _compute_account_opportunity_score(account)
        assert score == 100
        assert components["urgency"] == 30
        assert components["role"] == 20
        assert components["stage"] == 25
        assert components["seats"] == 15
        assert components["alternatives"] == 10

    def test_zero_score(self):
        """All defaults produce score 0."""
        score, components = _compute_account_opportunity_score({})
        assert score == 0
        assert all(v == 0 for v in components.values())

    def test_urgency_clamp_low(self):
        """Urgency below 5 should produce 0 points."""
        account = {"urgency": 3.0}
        score, components = _compute_account_opportunity_score(account)
        assert components["urgency"] == 0

    def test_urgency_clamp_high(self):
        """Urgency above 10 should cap at 30."""
        account = {"urgency": 15.0}
        score, components = _compute_account_opportunity_score(account)
        assert components["urgency"] == 30

    def test_partial_score(self):
        """Evaluator + evaluation + 50 seats + urgency 7 + 1 alternative."""
        account = {
            "urgency": 7,
            "role_level": "evaluator",
            "decision_maker": False,
            "buying_stage": "evaluation",
            "seat_count": 50,
            "alternatives_considering": ["Freshdesk"],
        }
        score, components = _compute_account_opportunity_score(account)
        assert components["urgency"] == 12  # (7-5)*6 = 12
        assert components["role"] == 10
        assert components["stage"] == 20
        assert components["seats"] == 5
        assert components["alternatives"] == 4
        assert score == 51

    def test_dm_overrides_role_level(self):
        """decision_maker=True should give 20 regardless of role_level."""
        account = {"decision_maker": True, "role_level": "evaluator"}
        _, components = _compute_account_opportunity_score(account)
        assert components["role"] == 20

    def test_seat_count_tiers(self):
        """Test the three seat count tiers."""
        assert _compute_account_opportunity_score({"seat_count": 500})[1]["seats"] == 15
        assert _compute_account_opportunity_score({"seat_count": 100})[1]["seats"] == 10
        assert _compute_account_opportunity_score({"seat_count": 20})[1]["seats"] == 5
        assert _compute_account_opportunity_score({"seat_count": 19})[1]["seats"] == 0

    def test_alternatives_tiers(self):
        """Test the three alternatives tiers."""
        assert _compute_account_opportunity_score(
            {"alternatives_considering": ["A", "B", "C", "D"]}
        )[1]["alternatives"] == 10
        assert _compute_account_opportunity_score(
            {"alternatives_considering": ["A", "B"]}
        )[1]["alternatives"] == 7
        assert _compute_account_opportunity_score(
            {"alternatives_considering": ["A"]}
        )[1]["alternatives"] == 4

    def test_renewal_decision_stage(self):
        """renewal_decision stage = 15 points."""
        _, c = _compute_account_opportunity_score({"buying_stage": "renewal_decision"})
        assert c["stage"] == 15


# ---------------------------------------------------------------------------
# Normalize company key
# ---------------------------------------------------------------------------

class TestNormalizeCompanyKey:
    def test_basic(self):
        assert _normalize_company_key("Acme Corp") == "acme corp"

    def test_case_insensitive(self):
        assert _normalize_company_key("ACME CORP") == "acme corp"

    def test_strips_whitespace(self):
        assert _normalize_company_key("  Acme Corp  ") == "acme corp"

    def test_empty(self):
        assert _normalize_company_key("") == ""

    def test_none(self):
        assert _normalize_company_key(None) == ""


class TestFallbackIntentRows:
    def test_build_fallback_intent_rows_filters_by_urgency(self):
        rows = _build_fallback_intent_rows(
            [
                {
                    "vendor": "Zendesk",
                    "companies": [
                        {"company": "Acme", "urgency": 4.9, "pain": "pricing"},
                        {"company": "Beta", "urgency": 5.0, "pain": "support"},
                    ],
                }
            ],
            min_urgency=5.0,
        )
        assert len(rows) == 1
        assert rows[0]["company"] == "Beta"
        assert rows[0]["source"] == "churning_companies_fallback"

    def test_build_fallback_intent_rows_carries_context_fields(self):
        rows = _build_fallback_intent_rows(
            [
                {
                    "vendor": "Zendesk",
                    "companies": [
                        {
                            "company": "Acme Corp",
                            "urgency": 7.0,
                            "role": "executive",
                            "title": "VP Support",
                            "company_size": "500-1000",
                            "industry": "SaaS",
                        }
                    ],
                }
            ],
            min_urgency=5.0,
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["title"] == "VP Support"
        assert row["company_size"] == "500-1000"
        assert row["industry"] == "SaaS"

    def test_build_signal_metadata_fallback_rows_uses_confidence_as_urgency(self):
        rows = _build_signal_metadata_fallback_rows(
            [
                {"company": "Acme", "vendor": "Zendesk", "confidence": 4.0},
                {"company": "Beta", "vendor": "Zendesk", "confidence": 6.2},
            ],
            min_urgency=5.0,
        )
        assert len(rows) == 1
        assert rows[0]["company"] == "Beta"
        assert rows[0]["source"] == "company_signals_fallback"


# ---------------------------------------------------------------------------
# Merging tests
# ---------------------------------------------------------------------------

class TestMergeCompanyProfiles:
    def _make_intent(self, company="Acme Corp", vendor="Zendesk", urgency=8.0, **kw):
        base = {
            "company": company,
            "vendor": vendor,
            "category": "Helpdesk",
            "title": kw.pop("title", None),
            "company_size": kw.pop("company_size", None),
            "industry": kw.pop("industry", None),
            "role_level": "executive",
            "decision_maker": True,
            "urgency": urgency,
            "pain": "pricing",
            "alternatives": [{"name": "Freshdesk"}],
            "review_id": kw.pop("review_id", "uuid-1"),
            "source": kw.pop("source", "g2"),
            "quotes": kw.pop("quotes", ["We need to switch ASAP"]),
            "seat_count": kw.pop("seat_count", 100),
            "contract_end": kw.pop("contract_end", None),
            "buying_stage": kw.pop("buying_stage", "evaluation"),
        }
        base.update(kw)
        return base

    def test_single_review(self):
        """A single high-intent row creates one profile."""
        merged = _merge_company_profiles(
            [self._make_intent()], [], [], [], [],
        )
        assert len(merged) == 1
        key = list(merged.keys())[0]
        assert key[0] == "acme corp"
        prof = merged[key]
        assert prof["company"] == "Acme Corp"
        assert prof["urgency"] == 8.0
        assert prof["evidence_count"] == 1
        assert "uuid-1" in prof["source_reviews"]

    def test_high_intent_context_fields_are_preserved(self):
        merged = _merge_company_profiles(
            [
                self._make_intent(
                    title="VP Support",
                    company_size="500-1000",
                    industry="SaaS",
                )
            ],
            [],
            [],
            [],
            [],
        )
        prof = list(merged.values())[0]
        assert prof["title"] == "VP Support"
        assert prof["company_size"] == "500-1000"
        assert prof["industry"] == "SaaS"

    def test_two_reviews_same_company_merge(self):
        """Two reviews for same (company, vendor) should merge."""
        r1 = self._make_intent(urgency=7.0, review_id="uuid-1")
        r2 = self._make_intent(urgency=9.0, review_id="uuid-2")
        r2["alternatives"] = [{"name": "Help Scout"}]
        merged = _merge_company_profiles([r1, r2], [], [], [], [])
        assert len(merged) == 1
        prof = list(merged.values())[0]
        assert prof["urgency"] == 9.0  # max
        assert prof["evidence_count"] == 2
        assert len(prof["source_reviews"]) == 2
        # Alternatives should be unioned
        assert "Freshdesk" in prof["alternatives_considering"]
        assert "Help Scout" in prof["alternatives_considering"]

    def test_company_name_normalization(self):
        """'Acme Corp' and 'ACME CORP' should merge into one profile."""
        r1 = self._make_intent(company="Acme Corp", urgency=7.0, review_id="uuid-1")
        r2 = self._make_intent(company="ACME CORP", urgency=8.0, review_id="uuid-2")
        merged = _merge_company_profiles([r1, r2], [], [], [], [])
        assert len(merged) == 1

    def test_timeline_fills_nulls(self):
        """Timeline signals fill nulls but don't overwrite existing data."""
        r = self._make_intent(contract_end="Q2 2026")
        timeline = [{
            "company": "Acme Corp",
            "vendor": "Zendesk",
            "contract_end": "Q3 2026",  # Should NOT overwrite
            "evaluation_deadline": "2026-04-15",
            "decision_timeline": "within_quarter",
            "urgency": 8,
            "title": "VP Engineering",
            "company_size": "500-1000",
            "industry": "SaaS",
        }]
        merged = _merge_company_profiles([r], timeline, [], [], [])
        prof = list(merged.values())[0]
        assert prof["contract_end"] == "Q2 2026"  # Not overwritten
        assert prof["evaluation_deadline"] == "2026-04-15"  # Filled
        assert prof["decision_timeline"] == "within_quarter"
        assert prof["title"] == "VP Engineering"

    def test_churning_fills_demographics(self):
        """Churning companies fill title, company_size, industry."""
        r = self._make_intent()
        churning = [{
            "vendor": "Zendesk",
            "companies": [{
                "company": "Acme Corp",
                "urgency": 8,
                "role": "executive",
                "pain": "pricing",
                "title": "CTO",
                "company_size": "1000+",
                "industry": "Fintech",
            }],
        }]
        merged = _merge_company_profiles([r], [], churning, [], [])
        prof = list(merged.values())[0]
        assert prof["title"] == "CTO"
        assert prof["company_size"] == "1000+"
        assert prof["industry"] == "Fintech"

    def test_no_match_vendors_empty(self):
        """Vendors without high-intent data produce no profiles."""
        timeline = [{
            "company": "Unknown Inc",
            "vendor": "SomeOther",
            "contract_end": "Q1 2027",
            "evaluation_deadline": None,
            "decision_timeline": None,
            "urgency": 5,
            "title": None,
            "company_size": None,
            "industry": None,
        }]
        merged = _merge_company_profiles([], timeline, [], [], [])
        assert len(merged) == 0

    def test_min_urgency_filter(self):
        """Accounts below min_urgency are excluded."""
        r = self._make_intent(urgency=4.0)
        merged = _merge_company_profiles([r], [], [], [], [], min_urgency=5.0)
        assert len(merged) == 0

    def test_quote_attachment(self):
        """Quotable evidence is attached to matching company."""
        r = self._make_intent()
        quotes = [{
            "vendor": "Zendesk",
            "quotes": [
                {"quote": "We need to switch ASAP", "urgency": 9, "company": "Acme Corp"},
                {"quote": "Pricing is bad", "urgency": 7, "company": "Other Inc"},
            ],
        }]
        merged = _merge_company_profiles([r], [], [], quotes, [])
        prof = list(merged.values())[0]
        assert prof["top_quote"] == "We need to switch ASAP"

    def test_vendor_level_quote_does_not_leak_to_other_company(self):
        """Vendor-level fallback quotes should not be attached to the wrong company."""
        r = self._make_intent(company="Acme Corp", quotes=[])
        quotes = [{
            "vendor": "Zendesk",
            "quotes": [{"quote": "Pricing is bad", "urgency": 7, "company": "Other Inc"}],
        }]
        merged = _merge_company_profiles([r], [], [], quotes, [])
        prof = list(merged.values())[0]
        assert prof["top_quote"] is None

    def test_alternative_cleanup_removes_self_and_invalid_terms(self):
        """Self references and configured non-vendor terms are dropped from alternatives."""
        r = self._make_intent(
            company="Acme Corp",
            vendor="Zendesk",
            alternatives=[{"name": "Acme Corp"}, {"name": "Zendesk"}, {"name": "bare metal"}, {"name": "Freshdesk"}],
        )
        merged = _merge_company_profiles(
            [r], [], [], [], [],
            invalid_alternative_terms=["bare metal"],
        )
        prof = list(merged.values())[0]
        assert prof["alternatives_considering"] == ["Freshdesk"]

    def test_signal_metadata_attachment(self):
        """Company signal metadata (first_seen, last_seen) is attached."""
        r = self._make_intent()
        from datetime import datetime
        meta = [{
            "company": "Acme Corp",
            "vendor": "Zendesk",
            "first_seen": "2026-03-01T00:00:00+00:00",
            "last_seen": "2026-03-15T00:00:00+00:00",
            "confidence": 0.85,
        }]
        merged = _merge_company_profiles([r], [], [], [], meta)
        prof = list(merged.values())[0]
        assert prof["first_seen"] == "2026-03-01T00:00:00+00:00"
        assert prof["last_seen"] == "2026-03-15T00:00:00+00:00"
        assert prof["confidence"] == 0.85

    def test_seed_rows_are_promoted_when_high_intent_overlay_arrives(self):
        seed_rows = [{
            "company": "Acme Corp",
            "vendor": "Zendesk",
            "category": "Helpdesk",
            "urgency": 7.0,
            "pain": "support",
            "role_level": "champion",
            "decision_maker": False,
            "buying_stage": "evaluation",
            "seat_count": 75,
            "contract_end": "2026-06-01",
            "first_seen": "2026-03-01T00:00:00+00:00",
            "last_seen": "2026-03-15T00:00:00+00:00",
            "confidence": 0.7,
        }]
        merged = _merge_company_profiles(
            [self._make_intent(review_id="uuid-9")], [], [], [], [], seed_rows=seed_rows,
        )
        prof = list(merged.values())[0]
        assert prof["evidence_count"] == 1
        assert prof["source_reviews"] == ["uuid-9"]
        assert prof["contract_end"] == "2026-06-01"
        assert prof["decision_maker"] is True
        assert prof["top_quote"] == "We need to switch ASAP"

    def test_seed_rows_preserve_pool_context_without_overlay(self):
        seed_rows = [{
            "company": "Acme Corp",
            "vendor": "Zendesk",
            "category": "Helpdesk",
            "urgency": 7.0,
            "pain": "pricing",
            "role_level": "champion",
            "decision_maker": False,
            "buying_stage": "evaluation",
            "alternatives": [{"name": "Freshdesk"}],
            "quotes": [{"quote": "We need to switch ASAP"}],
            "review_id": "uuid-1",
            "source": "g2",
        }]

        merged = _merge_company_profiles([], [], [], [], [], seed_rows=seed_rows)
        prof = list(merged.values())[0]

        assert prof["alternatives_considering"] == ["Freshdesk"]
        assert prof["source_reviews"] == ["uuid-1"]
        assert prof["top_quote"] == "We need to switch ASAP"
        assert prof["quote_match_type"] == "review"
        assert prof["source_distribution"] == {"g2": 1}

    def test_apollo_fills_industry_when_none(self):
        """Apollo industry fills when review data has no industry."""
        r = self._make_intent()
        apollo = {"acme corp": {
            "domain": "acme.com",
            "industry": "Retail",
            "employee_count": 5000,
            "annual_revenue_range": "$1B-$5B",
        }}
        merged = _merge_company_profiles([r], [], [], [], [], apollo_org_lookup=apollo)
        prof = list(merged.values())[0]
        assert prof["industry"] == "Retail"
        assert prof["company_size"] == "5000"
        assert prof["domain"] == "acme.com"
        assert prof["annual_revenue_range"] == "$1B-$5B"

    def test_apollo_does_not_overwrite_review_data(self):
        """Review industry wins over Apollo industry."""
        r = self._make_intent()
        timeline = [{
            "company": "Acme Corp",
            "vendor": "Zendesk",
            "contract_end": None,
            "evaluation_deadline": None,
            "decision_timeline": None,
            "urgency": 8,
            "title": None,
            "company_size": "200-500",
            "industry": "SaaS",
        }]
        apollo = {"acme corp": {
            "domain": "acme.com",
            "industry": "Retail",
            "employee_count": 5000,
            "annual_revenue_range": "$1B-$5B",
        }}
        merged = _merge_company_profiles([r], timeline, [], [], [], apollo_org_lookup=apollo)
        prof = list(merged.values())[0]
        assert prof["industry"] == "SaaS"  # review data wins
        assert prof["company_size"] == "200-500"  # review data wins
        assert prof["domain"] == "acme.com"  # only from Apollo
        assert prof["annual_revenue_range"] == "$1B-$5B"  # only from Apollo

    def test_apollo_adds_domain_and_revenue(self):
        """New fields populated from Apollo even when other fields exist."""
        r = self._make_intent()
        apollo = {"acme corp": {
            "domain": "acme.com",
            "industry": None,
            "employee_count": None,
            "annual_revenue_range": "$500M-$1B",
        }}
        merged = _merge_company_profiles([r], [], [], [], [], apollo_org_lookup=apollo)
        prof = list(merged.values())[0]
        assert prof["domain"] == "acme.com"
        assert prof["annual_revenue_range"] == "$500M-$1B"
        assert prof["industry"] is None  # Apollo has None, stays None

    def test_apollo_matches_via_normalized_company_name(self):
        """Apollo keyed by normalize_company_name (legal suffix stripped) still matches."""
        # normalize_company_name("Acme Corp") -> "acme" (strips "corp")
        # _normalize_company_key("Acme Corp") -> "acme corp"
        # The fallback path should find the match via normalize_company_name.
        r = self._make_intent()
        apollo = {"acme": {  # as normalize_company_name would produce
            "domain": "acme.com",
            "industry": "Retail",
            "employee_count": 5000,
            "annual_revenue_range": "$1B-$5B",
        }}
        merged = _merge_company_profiles([r], [], [], [], [], apollo_org_lookup=apollo)
        prof = list(merged.values())[0]
        assert prof["domain"] == "acme.com"
        assert prof["industry"] == "Retail"

    def test_no_apollo_match_no_change(self):
        """Missing Apollo match leaves profile unchanged."""
        r = self._make_intent()
        apollo = {"other corp": {
            "domain": "other.com",
            "industry": "Finance",
            "employee_count": 1000,
            "annual_revenue_range": "$100M-$500M",
        }}
        merged = _merge_company_profiles([r], [], [], [], [], apollo_org_lookup=apollo)
        prof = list(merged.values())[0]
        assert prof["domain"] is None
        assert prof["annual_revenue_range"] is None
        assert prof["industry"] is None


# ---------------------------------------------------------------------------
# Aggregate builder tests
# ---------------------------------------------------------------------------

class TestBuildVendorAggregate:
    def _make_account(self, **overrides):
        base = {
            "company": "Acme Corp",
            "vendor": "Zendesk",
            "urgency": 8.0,
            "opportunity_score": 75,
            "score_components": {"urgency": 18, "role": 20, "stage": 20, "seats": 10, "alternatives": 7},
            "pain_category": "pricing",
            "role_level": "executive",
            "decision_maker": True,
            "buying_stage": "evaluation",
            "seat_count": 100,
            "alternatives_considering": ["Freshdesk", "Help Scout"],
            "source_reviews": ["uuid-1"],
            "evidence_count": 1,
        }
        base.update(overrides)
        return base

    def test_basic_structure(self):
        """Aggregate has required top-level keys."""
        agg = _build_vendor_aggregate(
            "Zendesk",
            [self._make_account()],
            category="Helpdesk",
            reasoning_lookup={"Zendesk": {"archetype": "pricing_shock", "confidence": 0.85}},
            xv_lookup={"battles": {}, "councils": {}, "asymmetries": {}},
            feature_gap_lookup={},
            price_lookup={},
            budget_lookup={},
            competitor_lookup={},
        )
        assert agg["vendor"] == "Zendesk"
        assert agg["category"] == "Helpdesk"
        assert agg["archetype"] == "pricing_shock"
        assert agg["archetype_confidence"] == 0.85
        assert agg["total_accounts_in_motion"] == 1
        assert len(agg["accounts"]) == 1
        assert "pricing_pressure" in agg
        assert "feature_gaps" in agg
        assert "cross_vendor_context" in agg

    def test_feature_gaps_limited_to_10(self):
        """Feature gaps should be limited to top 10."""
        gaps = [{"feature_gap": f"gap_{i}", "mentions": 20 - i} for i in range(15)]
        agg = _build_vendor_aggregate(
            "Zendesk",
            [self._make_account()],
            category="Helpdesk",
            reasoning_lookup={},
            xv_lookup={"battles": {}, "councils": {}, "asymmetries": {}},
            feature_gap_lookup={"Zendesk": gaps},
            price_lookup={},
            budget_lookup={},
            competitor_lookup={},
        )
        assert len(agg["feature_gaps"]) == 10

    def test_pricing_pressure_assembled(self):
        """Pricing pressure section is assembled from price + budget lookups."""
        agg = _build_vendor_aggregate(
            "Zendesk",
            [self._make_account()],
            category="Helpdesk",
            reasoning_lookup={},
            xv_lookup={"battles": {}, "councils": {}, "asymmetries": {}},
            feature_gap_lookup={},
            price_lookup={"Zendesk": 0.234},
            budget_lookup={"Zendesk": {
                "avg_seat_count": 150.5,
                "price_increase_rate": 0.067,
            }},
            competitor_lookup={},
        )
        pp = agg["pricing_pressure"]
        assert pp["price_complaint_rate"] == 0.234
        assert pp["price_increase_rate"] == 0.067
        assert pp["avg_seat_count"] == 150  # rounded

    def test_vendor_aggregate_prefers_evidence_vault_feature_gaps_and_price(self):
        """Canonical vault feature gaps and price rate override sparse raw lookups."""
        agg = _build_vendor_aggregate(
            "Zendesk",
            [self._make_account()],
            category="Helpdesk",
            reasoning_lookup={},
            xv_lookup={"battles": {}, "councils": {}, "asymmetries": {}},
            feature_gap_lookup={"Zendesk": [{"feature": "Legacy Gap", "mentions": 2}]},
            price_lookup={},
            budget_lookup={"Zendesk": {"avg_seat_count": 90, "price_increase_rate": 0.02}},
            competitor_lookup={},
            evidence_vault_lookup={
                "Zendesk": {
                    "weakness_evidence": [
                        {
                            "evidence_type": "feature_gap",
                            "label": "Reporting",
                            "mention_count_total": 14,
                        }
                    ],
                    "metric_snapshot": {"price_complaint_rate": 0.19},
                }
            },
        )
        assert agg["feature_gaps"] == [{"feature": "Reporting", "mentions": 14}]
        assert agg["pricing_pressure"]["price_complaint_rate"] == 0.19
        assert agg["data_sources"]["evidence_vault"] is True

    def test_cross_vendor_context_top_destination(self):
        """top_destination comes from competitor_lookup."""
        agg = _build_vendor_aggregate(
            "Zendesk",
            [self._make_account()],
            category="Helpdesk",
            reasoning_lookup={},
            xv_lookup={"battles": {}, "councils": {}, "asymmetries": {}},
            feature_gap_lookup={},
            price_lookup={},
            budget_lookup={},
            competitor_lookup={"Zendesk": [{"name": "Freshdesk", "mentions": 10}]},
        )
        assert agg["cross_vendor_context"]["top_destination"] == "Freshdesk"

    def test_cross_vendor_context_falls_back_to_account_alternatives(self):
        account = self._make_account(alternatives_considering=["Intercom", "Intercom", "Freshdesk"])
        agg = _build_vendor_aggregate(
            "Zendesk",
            [account],
            category="Helpdesk",
            reasoning_lookup={},
            xv_lookup={"battles": {}, "councils": {}, "asymmetries": {}},
            feature_gap_lookup={},
            price_lookup={},
            budget_lookup={},
            competitor_lookup={},
        )
        assert agg["cross_vendor_context"]["top_destination"] == "Intercom"
        assert "active evaluation sets" in agg["cross_vendor_context"]["battle_conclusion"]

    def test_cross_vendor_context_has_non_empty_conclusion_without_flows(self):
        agg = _build_vendor_aggregate(
            "Zendesk",
            [self._make_account(alternatives_considering=[])],
            category="Helpdesk",
            reasoning_lookup={},
            xv_lookup={"battles": {}, "councils": {}, "asymmetries": {}},
            feature_gap_lookup={},
            price_lookup={},
            budget_lookup={},
            competitor_lookup={},
        )
        assert agg["cross_vendor_context"]["battle_conclusion"]

    def test_cross_vendor_context_prefers_battle_for_top_destination(self):
        agg = _build_vendor_aggregate(
            "Zendesk",
            [self._make_account(alternatives_considering=["Freshdesk"])],
            category="Helpdesk",
            reasoning_lookup={},
            xv_lookup={
                "battles": {
                    tuple(sorted(("Zendesk", "Intercom"))): {
                        "conclusion": {"conclusion": "Intercom is winning evaluations against Zendesk."}
                    },
                    tuple(sorted(("Zendesk", "Freshdesk"))): {
                        "conclusion": {"conclusion": "Freshdesk is displacing Zendesk on pricing."}
                    },
                },
                "councils": {},
                "asymmetries": {},
            },
            feature_gap_lookup={},
            price_lookup={},
            budget_lookup={},
            competitor_lookup={"Zendesk": [{"name": "Freshdesk", "mentions": 10}]},
        )
        assert agg["cross_vendor_context"]["top_destination"] == "Freshdesk"
        assert agg["cross_vendor_context"]["battle_conclusion"] == "Freshdesk is displacing Zendesk on pricing."

    def test_vendor_aggregate_attaches_category_council_when_richer(self):
        agg = _build_vendor_aggregate(
            "Zendesk",
            [self._make_account()],
            category="Helpdesk",
            reasoning_lookup={},
            xv_lookup={
                "battles": {},
                "councils": {
                    "Helpdesk": {
                        "confidence": 0.58,
                        "conclusion": {
                            "winner": "Zoho Desk",
                            "loser": "Freshdesk",
                            "conclusion": "Pricing pressure is fragmenting the helpdesk market.",
                            "market_regime": "price_competition",
                            "durability_assessment": "cyclical",
                            "key_insights": [{"insight": "Pricing is the primary driver.", "evidence": "pricing"}],
                        },
                    }
                },
                "asymmetries": {},
            },
            feature_gap_lookup={},
            price_lookup={},
            budget_lookup={},
            competitor_lookup={},
        )
        assert agg["category_council"]["winner"] == "Zoho Desk"
        assert agg["category_council"]["market_regime"] == "price_competition"
        assert agg["cross_vendor_context"]["market_regime"] == "price_competition"

    def test_vendor_aggregate_attaches_synthesis_contracts(self):
        class _Wedge:
            value = "pricing_shock"

        class _View:
            schema_version = "v2"
            primary_wedge = _Wedge()
            wedge_label = "Pricing Shock"
            meta = {"evidence_window_start": "2026-02-01", "evidence_window_end": "2026-03-01"}
            reference_ids = {
                "metric_ids": ["metric:zendesk:1"],
                "witness_ids": ["witness:zendesk:1"],
            }

            @staticmethod
            def contract(name):
                if name == "vendor_core_reasoning":
                    return {
                        "causal_narrative": {"confidence": "high"},
                        "segment_playbook": {
                            "confidence": "medium",
                            "supporting_evidence": {
                                "top_strategic_roles": [
                                    {
                                        "role_type": "economic_buyer",
                                        "source_id": "segment:role:economic_buyer",
                                    },
                                ],
                                "top_contract_segments": [
                                    {"segment": "smb", "source_id": "segment:contract:smb"},
                                ],
                            },
                        },
                        "timing_intelligence": {
                            "confidence": "medium",
                            "best_timing_window": "Before renewal review",
                            "active_eval_signals": {
                                "value": 2,
                                "source_id": "accounts:summary:active_eval_signal_count",
                            },
                            "sentiment_direction": "declining",
                            "immediate_triggers": [
                                {"trigger": "Q2 renewal", "type": "deadline"},
                            ],
                        },
                    }
                if name == "displacement_reasoning":
                    return {"competitive_reframes": {"confidence": "medium"}}
                if name == "account_reasoning":
                    return {
                        "market_summary": "Two accounts are actively evaluating alternatives while five accounts show high intent signals overall.",
                        "high_intent_count": {
                            "value": 5,
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
                    }
                return {}

        agg = _build_vendor_aggregate(
            "Zendesk",
            [self._make_account()],
            category="Helpdesk",
            reasoning_lookup={},
            xv_lookup={"battles": {}, "councils": {}, "asymmetries": {}},
            feature_gap_lookup={},
            price_lookup={},
            budget_lookup={},
            competitor_lookup={},
            synthesis_views={"Zendesk": _View()},
        )
        assert agg["reasoning_source"] == "b2b_reasoning_synthesis"
        assert agg["synthesis_wedge"] == "pricing_shock"
        assert "reasoning_contracts" in agg
        assert agg["reference_ids"]["metric_ids"] == ["metric:zendesk:1"]
        assert agg["reference_ids"]["witness_ids"] == ["witness:zendesk:1"]
        assert agg["reasoning_contracts"]["vendor_core_reasoning"]["causal_narrative"]["confidence"] == "high"
        assert agg["account_reasoning"]["top_accounts"][0]["name"] == "Acme Corp"
        assert agg["account_pressure_summary"] == (
            "Two accounts are actively evaluating alternatives while five accounts show high intent signals overall."
        )
        assert agg["account_pressure_metrics"]["high_intent_count"] == 5
        assert agg["account_pressure_metrics"]["active_eval_count"] == 2
        assert agg["priority_account_names"] == ["Acme Corp"]
        assert agg["segment_playbook"]["supporting_evidence"]["top_contract_segments"][0]["segment"] == "smb"
        assert agg["timing_intelligence"]["best_timing_window"] == "Before renewal review"
        assert agg["timing_summary"] == (
            "Before renewal review. 2 active evaluation signals are visible right now. "
            "Review sentiment is skewing more negative."
        )
        assert agg["timing_metrics"]["active_eval_signals"] == 2
        assert agg["priority_timing_triggers"] == ["Q2 renewal"]
        assert "economic buyers" in agg["segment_targeting_summary"]
        assert "smb contracts" in agg["segment_targeting_summary"]
        assert "vendor_core_reasoning" not in agg

    def test_empty_accounts(self):
        """Vendor with no accounts still builds a valid aggregate."""
        agg = _build_vendor_aggregate(
            "Zendesk",
            [],
            category="Helpdesk",
            reasoning_lookup={},
            xv_lookup={"battles": {}, "councils": {}, "asymmetries": {}},
            feature_gap_lookup={},
            price_lookup={},
            budget_lookup={},
            competitor_lookup={},
        )
        assert agg["total_accounts_in_motion"] == 0
        assert agg["accounts"] == []

    def test_vendor_evidence_summary_uses_unique_review_ids_and_sources(self):
        accounts = [
            self._make_account(source_reviews=["uuid-1", "uuid-2"], source_distribution={"g2": 2}),
            self._make_account(source_reviews=["uuid-2", "uuid-3"], source_distribution={"reddit": 1}),
        ]
        review_count, source_distribution = _summarize_vendor_evidence(accounts)
        assert review_count == 3
        assert source_distribution == {"g2": 2, "reddit": 1}

    def test_aggregate_includes_vendor_specific_evidence_metadata(self):
        agg = _build_vendor_aggregate(
            "Zendesk",
            [self._make_account(source_reviews=["uuid-1"], source_distribution={"g2": 1})],
            category="Helpdesk",
            reasoning_lookup={},
            xv_lookup={"battles": {}, "councils": {}, "asymmetries": {}},
            feature_gap_lookup={},
            price_lookup={},
            budget_lookup={},
            competitor_lookup={},
        )
        assert agg["source_review_count"] == 1
        assert agg["source_distribution"] == {"g2": 1}


class TestAccountQualityAdjustments:
    def test_quality_adjustments_penalize_missing_context_and_bonus_repeat_evidence(self):
        cfg = MagicMock(
            accounts_in_motion_repeat_evidence_bonus=3,
            accounts_in_motion_repeat_evidence_bonus_max=6,
            accounts_in_motion_low_confidence_threshold=6.0,
            accounts_in_motion_low_confidence_penalty=6,
            accounts_in_motion_missing_domain_penalty=8,
            accounts_in_motion_missing_title_penalty=4,
            accounts_in_motion_missing_quote_penalty=4,
        )
        account = {
            "evidence_count": 3,
            "confidence": 5.0,
            "domain": None,
            "title": None,
            "top_quote": None,
        }
        delta, components, flags = _apply_account_quality_adjustments(account, cfg)
        assert delta == -16
        assert components["repeat_evidence"] == 6
        assert components["low_confidence"] == -6
        assert "missing_domain" in flags


class TestAccountsInMotionRunProgress:
    @pytest.mark.asyncio
    async def test_run_emits_progress_metadata_across_stages(self, monkeypatch):
        progress = AsyncMock()
        pool = type("Pool", (), {"is_initialized": True, "execute": AsyncMock()})()

        async def fake_gather(*_args, **_kwargs):
            for coro in _args:
                close = getattr(coro, "close", None)
                if close:
                    close()
            return ([{"vendor": "Zendesk", "company": "Acme", "urgency": 8.0}], [], [], [], [], [], [], [], [], {}, {})

        monkeypatch.setattr(accounts_mod.settings.b2b_churn, "enabled", True, raising=False)
        monkeypatch.setattr(accounts_mod.settings.b2b_churn, "intelligence_enabled", True, raising=False)
        monkeypatch.setattr(accounts_mod, "_update_execution_progress", progress)
        monkeypatch.setattr(accounts_mod, "get_db_pool", lambda: pool)
        monkeypatch.setattr(accounts_mod, "_check_freshness", AsyncMock(return_value=date(2026, 3, 18)))
        monkeypatch.setattr(accounts_mod.asyncio, "gather", fake_gather)
        monkeypatch.setattr(accounts_mod, "_merge_company_profiles", lambda *args, **kwargs: {})
        monkeypatch.setattr(churn_mod, "reconstruct_reasoning_lookup", AsyncMock(return_value={}))
        monkeypatch.setattr(
            churn_mod,
            "reconstruct_cross_vendor_lookup",
            AsyncMock(return_value={"battles": {}, "councils": {}, "asymmetries": {}}),
        )

        task = type("Task", (), {"metadata": {"_execution_id": str(uuid4())}})()
        result = await accounts_mod.run(task)

        assert result == {
            "_skip_synthesis": "Accounts in motion complete",
            "vendors": 0,
            "total_accounts": 0,
            "persisted": 0,
        }
        stages = [call.kwargs["stage"] for call in progress.await_args_list]
        assert stages == [
            accounts_mod._STAGE_LOADING_INPUTS,
            accounts_mod._STAGE_BUILDING_ACCOUNTS,
            accounts_mod._STAGE_PERSISTING_REPORTS,
            accounts_mod._STAGE_FINALIZING,
        ]

    @pytest.mark.asyncio
    async def test_run_uses_churning_fallback_rows_when_high_intent_empty(self, monkeypatch):
        pool = type("Pool", (), {"is_initialized": True, "execute": AsyncMock()})()
        captured: dict[str, Any] = {}

        async def fake_gather(*_args, **_kwargs):
            for coro in _args:
                close = getattr(coro, "close", None)
                if close:
                    close()
            return (
                [],
                [],
                [{"vendor": "Zendesk", "companies": [{"company": "Acme", "urgency": 7.0, "pain": "pricing"}]}],
                [],
                [],
                [],
                [],
                [],
                [],
                {},
                {},
            )

        def fake_merge(high_intent_rows, *_args, **_kwargs):
            captured["rows"] = high_intent_rows
            return {}

        monkeypatch.setattr(accounts_mod.settings.b2b_churn, "enabled", True, raising=False)
        monkeypatch.setattr(accounts_mod.settings.b2b_churn, "intelligence_enabled", True, raising=False)
        monkeypatch.setattr(accounts_mod, "get_db_pool", lambda: pool)
        monkeypatch.setattr(accounts_mod, "_check_freshness", AsyncMock(return_value=date(2026, 3, 18)))
        monkeypatch.setattr(accounts_mod.asyncio, "gather", fake_gather)
        monkeypatch.setattr(accounts_mod, "_merge_company_profiles", fake_merge)
        monkeypatch.setattr(churn_mod, "reconstruct_reasoning_lookup", AsyncMock(return_value={}))
        monkeypatch.setattr(
            churn_mod,
            "reconstruct_cross_vendor_lookup",
            AsyncMock(return_value={"battles": {}, "councils": {}, "asymmetries": {}}),
        )
        monkeypatch.setattr(accounts_mod, "_fetch_latest_synthesis_views", AsyncMock(return_value={}))
        monkeypatch.setattr(accounts_mod, "_update_execution_progress", AsyncMock())

        task = type("Task", (), {"metadata": {"_execution_id": str(uuid4())}})()
        result = await accounts_mod.run(task)

        assert result["_skip_synthesis"] == "Accounts in motion complete"
        assert any(row.get("source") == "churning_companies_fallback" for row in captured["rows"])

    @pytest.mark.asyncio
    async def test_run_scopes_rows_to_test_vendors(self, monkeypatch):
        pool = type("Pool", (), {"is_initialized": True, "execute": AsyncMock()})()
        captured: dict[str, Any] = {}

        async def fake_gather(*_args, **_kwargs):
            for coro in _args:
                close = getattr(coro, "close", None)
                if close:
                    close()
            return (
                [
                    {"vendor": "Zendesk", "company": "Acme", "urgency": 8.0},
                    {"vendor": "Freshdesk", "company": "Beta", "urgency": 8.0},
                ],
                [], [], [], [], [], [], [], [], {}, {},
            )

        def fake_merge(high_intent_rows, *_args, **_kwargs):
            captured["rows"] = high_intent_rows
            return {}

        monkeypatch.setattr(accounts_mod.settings.b2b_churn, "enabled", True, raising=False)
        monkeypatch.setattr(accounts_mod.settings.b2b_churn, "intelligence_enabled", True, raising=False)
        monkeypatch.setattr(accounts_mod, "get_db_pool", lambda: pool)
        monkeypatch.setattr(accounts_mod, "_check_freshness", AsyncMock(return_value=date(2026, 3, 18)))
        monkeypatch.setattr(accounts_mod.asyncio, "gather", fake_gather)
        monkeypatch.setattr(accounts_mod, "_merge_company_profiles", fake_merge)
        monkeypatch.setattr(churn_mod, "reconstruct_reasoning_lookup", AsyncMock(return_value={}))
        monkeypatch.setattr(
            churn_mod,
            "reconstruct_cross_vendor_lookup",
            AsyncMock(return_value={"battles": {}, "councils": {}, "asymmetries": {}}),
        )
        monkeypatch.setattr(accounts_mod, "_fetch_latest_synthesis_views", AsyncMock(return_value={}))
        monkeypatch.setattr(accounts_mod, "_update_execution_progress", AsyncMock())

        task = type("Task", (), {"metadata": {"_execution_id": str(uuid4()), "test_vendors": ["Zendesk"]}})()
        result = await accounts_mod.run(task)

        assert result["_skip_synthesis"] == "Accounts in motion complete"
        assert [row["vendor"] for row in captured["rows"]] == ["Zendesk"]
