"""Tests for accounts_in_motion report type.

Covers scoring, merging, aggregate building, and short-circuit paths.
"""

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

# Pre-mock heavy deps before importing task module
for _mod in (
    "torch", "torchaudio", "transformers", "accelerate", "bitsandbytes",
    "PIL", "PIL.Image", "numpy", "cv2", "sounddevice", "soundfile",
    "nemo.collections", "nemo.collections.asr",
    "nemo.collections.asr.models",
    "starlette", "starlette.requests",
    "asyncpg",
    "playwright", "playwright.async_api",
    "playwright_stealth",
    "curl_cffi", "curl_cffi.requests",
    "pytrends", "pytrends.request",
):
    sys.modules.setdefault(_mod, MagicMock())

from atlas_brain.autonomous.tasks.b2b_accounts_in_motion import (
    _compute_account_opportunity_score,
    _merge_company_profiles,
    _normalize_company_key,
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


# ---------------------------------------------------------------------------
# Merging tests
# ---------------------------------------------------------------------------

class TestMergeCompanyProfiles:
    def _make_intent(self, company="Acme Corp", vendor="Zendesk", urgency=8.0, **kw):
        base = {
            "company": company,
            "vendor": vendor,
            "category": "Helpdesk",
            "role_level": "executive",
            "decision_maker": True,
            "urgency": urgency,
            "pain": "pricing",
            "alternatives": [{"name": "Freshdesk"}],
            "review_id": kw.pop("review_id", "uuid-1"),
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
