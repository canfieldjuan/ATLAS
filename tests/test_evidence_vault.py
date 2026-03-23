from datetime import date, datetime
import json
from unittest.mock import AsyncMock

import pytest

from atlas_brain.autonomous.tasks._b2b_shared import (
    _build_evidence_vault_pass2_rollups,
    _fetch_latest_evidence_vault,
    _fetch_product_profiles,
    _merge_company_lookup_with_evidence_vault,
    _merge_canonical_company_signals,
    _merge_feature_gap_lookup_with_evidence_vault,
    _merge_pain_lookup_with_evidence_vault,
    build_evidence_vault,
)


@pytest.mark.asyncio
async def test_fetch_product_profiles_normalizes_json_fields():
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=[{
        "vendor_name": "Acme",
        "product_category": "CRM",
        "strengths": '[{"area":"integrations","score":4.2,"evidence_count":12}]',
        "weaknesses": '[{"area":"pricing","score":2.8,"evidence_count":7}]',
        "pain_addressed": '{"pricing": 3}',
        "commonly_compared_to": '["HubSpot"]',
        "commonly_switched_from": '["Salesforce"]',
        "total_reviews_analyzed": 20,
        "avg_rating": 3.5,
        "recommend_rate": 0.6,
        "primary_use_cases": '["pipeline management"]',
        "typical_company_size": '["mid_market"]',
        "typical_industries": '["SaaS"]',
        "top_integrations": '["Slack"]',
        "profile_summary": "Summary",
    }])

    profiles = await _fetch_product_profiles(pool)

    assert profiles[0]["strengths"][0]["area"] == "integrations"
    assert profiles[0]["weaknesses"][0]["area"] == "pricing"
    assert profiles[0]["pain_addressed"]["pricing"] == 3
    assert profiles[0]["primary_use_cases"] == ["pipeline management"]


@pytest.mark.asyncio
async def test_fetch_latest_evidence_vault_normalizes_vendor_lookup():
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=[{
        "vendor_name": "Acme",
        "vault": '{"vendor_name":"Acme","weakness_evidence":[],"company_signals":[],"metric_snapshot":{},"provenance":{}}',
    }])

    lookup = await _fetch_latest_evidence_vault(
        pool,
        as_of=date(2026, 3, 19),
        analysis_window_days=30,
    )

    assert "Acme" in lookup
    assert lookup["Acme"]["vendor_name"] == "Acme"


def test_build_evidence_vault_accepts_stringified_product_profile_fields():
    vault = build_evidence_vault(
        vendor_name="Acme",
        vs={
            "total_reviews": 20,
            "recommend_yes": 12,
            "recommend_no": 8,
            "avg_urgency": 5.5,
        },
        pain_entries=[{"category": "pricing", "complaint_count": 6, "avg_urgency": 6.2}],
        feature_gap_entries=[],
        quotes=[{
            "quote": "Pricing got confusing fast.",
            "review_id": "r1",
            "company": "Acme Corp",
            "title": "VP Ops",
            "company_size": "mid_market",
            "industry": "SaaS",
        }],
        positive_entries=[{"aspect": "integrations", "mentions": 5}],
        product_profile={
            "strengths": json.dumps([{"area": "integrations", "score": 4.2, "evidence_count": 12}]),
            "weaknesses": json.dumps([{"area": "pricing", "score": 2.8, "evidence_count": 7}]),
        },
        keyword_spikes=None,
        company_signals=[],
        provenance=None,
        data_context=None,
        dm_rate=0.2,
        price_rate=0.3,
        analysis_window_days=30,
        recent_window_days=14,
    )

    weakness_keys = [item["key"] for item in vault["weakness_evidence"]]
    strength_keys = [item["key"] for item in vault["strength_evidence"]]

    assert "pricing" in weakness_keys
    assert "integrations" in strength_keys


def test_build_evidence_vault_pass2_rollups_populates_recent_trend_and_quote_provenance():
    rollups = _build_evidence_vault_pass2_rollups(
        [
            {
                "review_id": "r1",
                "vendor_name": "Acme",
                "reviewed_at": date(2026, 3, 18),
                "enriched_at": date(2026, 3, 18),
                "rating": 2.0,
                "rating_max": 5.0,
                "reviewer_title": "VP Ops",
                "company_size_raw": "mid_market",
                "role_level": "economic_buyer",
                "pain_category": "pricing",
                "feature_gaps": ["API access"],
                "positive_aspects": ["integrations"],
                "urgency": 8.0,
            },
            {
                "review_id": "r2",
                "vendor_name": "Acme",
                "reviewed_at": date(2026, 3, 10),
                "enriched_at": date(2026, 3, 10),
                "rating": 1.0,
                "rating_max": 5.0,
                "reviewer_title": "Finance Director",
                "company_size_raw": "mid_market",
                "role_level": "economic_buyer",
                "pain_category": "pricing",
                "feature_gaps": [],
                "positive_aspects": ["integrations"],
                "urgency": 7.0,
            },
            {
                "review_id": "r3",
                "vendor_name": "Acme",
                "reviewed_at": date(2026, 2, 1),
                "enriched_at": date(2026, 2, 1),
                "rating": 2.0,
                "rating_max": 5.0,
                "reviewer_title": "IT Manager",
                "company_size_raw": "enterprise",
                "role_level": "evaluator",
                "pain_category": "pricing",
                "feature_gaps": ["API access"],
                "positive_aspects": [],
                "urgency": 5.0,
            },
        ],
        {
            "Acme": [
                {
                    "quote": "Pricing got confusing fast.",
                    "review_id": "r1",
                    "source": "g2",
                    "reviewed_at": date(2026, 3, 18),
                    "rating": 2.0,
                    "company": "Acme Corp",
                    "title": "VP Ops",
                    "company_size": "mid_market",
                    "industry": "SaaS",
                    "urgency": 8.0,
                },
            ],
        },
        recent_window_days=30,
        today=date(2026, 3, 19),
    )

    pricing = rollups["Acme"]["weaknesses"][("pain_category", "pricing")]
    integrations = rollups["Acme"]["strengths"][("retention_signal", "integrations")]

    assert rollups["Acme"]["reviews_in_recent_window"] == 2
    assert pricing["mention_count_recent"] == 2
    assert pricing["trend"]["direction"] == "accelerating"
    assert pricing["affected_segments"][0] == {
        "segment": "mid_market",
        "count": 2,
        "recent_count": 2,
    }
    assert pricing["best_quote"] == "Pricing got confusing fast."
    assert pricing["quote_source"]["source"] == "g2"
    assert pricing["quote_source"]["reviewed_at"] == "2026-03-18"
    assert pricing["quote_source"]["rating"] == 2.0
    assert pricing["supporting_review_ids"] == ["r1", "r2", "r3"]
    assert integrations["mention_count_recent"] == 2
    assert integrations["trend"]["direction"] == "new"


def test_build_evidence_vault_merges_pass2_rollups_into_output():
    pass2_rollups = {
        "reviews_in_recent_window": 2,
        "weaknesses": {
            ("pain_category", "pricing"): {
                "best_quote": "Pricing got confusing fast.",
                "quote_source": {
                    "review_id": "r1",
                    "source": "g2",
                    "company": "Acme Corp",
                    "reviewer_title": "VP Ops",
                    "company_size": "mid_market",
                    "industry": "SaaS",
                    "reviewed_at": "2026-03-18",
                    "rating": 2.0,
                },
                "mention_count_recent": 2,
                "trend": {
                    "direction": "accelerating",
                    "prior_count": 1,
                    "recent_count": 2,
                    "delta_pct": 1.0,
                    "basis": "recent_vs_prior_window",
                },
                "affected_segments": [{"segment": "mid_market", "count": 2, "recent_count": 2}],
                "affected_roles": [{"role": "VP Ops", "count": 1, "recent_count": 1}],
                "supporting_metrics": {"avg_rating_when_mentioned": 1.67},
                "supporting_review_ids": ["r1", "r2", "r3"],
            },
        },
        "strengths": {
            ("retention_signal", "integrations"): {
                "best_quote": "The integrations save us time.",
                "quote_source": {
                    "review_id": "r2",
                    "source": "g2",
                    "company": "Acme Corp",
                    "reviewer_title": "Finance Director",
                    "company_size": "mid_market",
                    "industry": "SaaS",
                    "reviewed_at": "2026-03-10",
                    "rating": 1.0,
                },
                "mention_count_recent": 2,
                "trend": {
                    "direction": "new",
                    "prior_count": 0,
                    "recent_count": 2,
                    "delta_pct": None,
                    "basis": "recent_vs_prior_window",
                },
                "affected_segments": [{"segment": "mid_market", "count": 2, "recent_count": 2}],
                "affected_roles": [{"role": "Finance Director", "count": 1, "recent_count": 1}],
                "supporting_metrics": {"avg_urgency_when_mentioned": 7.5},
                "supporting_review_ids": ["r1", "r2"],
            },
        },
    }

    vault = build_evidence_vault(
        vendor_name="Acme",
        vs={
            "total_reviews": 20,
            "recommend_yes": 12,
            "recommend_no": 8,
            "avg_urgency": 5.5,
        },
        pain_entries=[{"category": "pricing", "complaint_count": 3, "avg_urgency": 6.7}],
        feature_gap_entries=[],
        quotes=[],
        positive_entries=[{"aspect": "integrations", "mentions": 2}],
        product_profile=None,
        keyword_spikes=None,
        company_signals=[],
        provenance=None,
        data_context=None,
        pass2_rollups=pass2_rollups,
        dm_rate=0.2,
        price_rate=0.3,
        analysis_window_days=30,
        recent_window_days=30,
    )

    pricing = vault["weakness_evidence"][0]
    integrations = vault["strength_evidence"][0]

    assert pricing["mention_count_recent"] == 2
    assert pricing["trend"]["direction"] == "accelerating"
    assert pricing["quote_source"]["source"] == "g2"
    assert pricing["supporting_review_ids"] == ["r1", "r2", "r3"]
    assert pricing["supporting_metrics"]["avg_urgency"] == 6.7
    assert pricing["supporting_metrics"]["avg_rating_when_mentioned"] == 1.67
    assert integrations["mention_count_recent"] == 2
    assert integrations["trend"]["direction"] == "new"
    assert vault["metric_snapshot"]["reviews_in_recent_window"] == 2


def test_merge_canonical_company_signals_preserves_strongest_current_signal():
    merged = _merge_canonical_company_signals(
        current_high_intent=[
            {
                "company": "Acme Corp",
                "vendor": "Zendesk",
                "urgency": 8.0,
                "pain": "pricing",
                "role_level": "VP Ops",
                "decision_maker": True,
                "seat_count": 120,
                "contract_end": "2026-06-30",
                "buying_stage": "evaluation",
                "review_id": "r-new",
                "source": "g2",
            },
            {
                "company": "Acme Corp",
                "vendor": "Zendesk",
                "urgency": 6.0,
                "pain": "support",
                "role_level": "Manager",
                "decision_maker": False,
                "seat_count": 80,
                "contract_end": "2026-07-31",
                "buying_stage": "research",
                "review_id": "r-old",
                "source": "reddit",
            },
        ],
        persisted_lookup={
            "Zendesk": [
                {
                    "company_name": "acme",
                    "vendor_name": "Zendesk",
                    "urgency_score": 5.0,
                    "pain_category": "pricing",
                    "buyer_role": "Ops Lead",
                    "decision_maker": False,
                    "seat_count": 50,
                    "contract_end": "2026-05-31",
                    "buying_stage": "monitoring",
                    "review_id": "r-persisted",
                    "source": "capterra",
                    "confidence_score": 0.74,
                    "first_seen_at": "2026-03-01T00:00:00+00:00",
                    "last_seen_at": "2026-03-10T00:00:00+00:00",
                },
            ],
        },
        as_of=datetime.fromisoformat("2026-03-19T12:00:00+00:00"),
    )

    signal = merged["Zendesk"][0]
    assert signal["company_name"] == "acme"
    assert signal["urgency_score"] == 8.0
    assert signal["pain_category"] == "pricing"
    assert signal["buyer_role"] == "VP Ops"
    assert signal["decision_maker"] is True
    assert signal["seat_count"] == 120
    assert signal["contract_end"] == "2026-06-30"
    assert signal["buying_stage"] == "evaluation"
    assert signal["review_id"] == "r-new"
    assert signal["source"] == "g2"
    assert signal["first_seen_at"] == "2026-03-01T00:00:00+00:00"
    assert signal["last_seen_at"] == "2026-03-19T12:00:00+00:00"
    assert signal["confidence_score"] == 0.74


def test_merge_canonical_company_signals_preserves_account_context_fields():
    merged = _merge_canonical_company_signals(
        current_high_intent=[
            {
                "company": "Acme Corp",
                "vendor": "Zendesk",
                "urgency": 8.0,
                "title": "VP Support",
                "company_size": "500-1000",
                "industry": "SaaS",
                "alternatives": [{"name": "Freshdesk"}],
                "quotes": [{"quote": "We need to switch ASAP"}],
            },
        ],
        persisted_lookup={
            "Zendesk": [
                {
                    "company_name": "acme",
                    "vendor_name": "Zendesk",
                    "urgency_score": 7.0,
                    "alternatives": [{"name": "Help Scout"}],
                    "quotes": [{"quote": "Support has slipped"}],
                },
            ],
        },
    )

    signal = merged["Zendesk"][0]
    assert signal["title"] == "VP Support"
    assert signal["company_size"] == "500-1000"
    assert signal["industry"] == "SaaS"
    assert signal["alternatives"] == [
        {"name": "Freshdesk"},
        {"name": "Help Scout"},
    ]
    assert signal["quotes"] == [
        {"quote": "We need to switch ASAP"},
        {"quote": "Support has slipped"},
    ]


def test_merge_pain_lookup_with_evidence_vault_prefers_canonical_rows():
    merged = _merge_pain_lookup_with_evidence_vault(
        {
            "Acme": [
                {"category": "pricing", "count": 2, "avg_urgency": 4.0},
                {"category": "support", "count": 1, "avg_urgency": 5.0},
            ],
        },
        {
            "Acme": {
                "weakness_evidence": [
                    {
                        "key": "pricing",
                        "evidence_type": "pain_category",
                        "mention_count_total": 9,
                        "supporting_metrics": {"avg_urgency": 6.7},
                    },
                ],
            },
        },
    )

    assert merged["Acme"][0] == {
        "category": "pricing",
        "count": 9,
        "avg_urgency": 6.7,
    }
    assert merged["Acme"][1]["category"] == "support"


def test_merge_feature_gap_lookup_with_evidence_vault_prefers_canonical_rows():
    merged = _merge_feature_gap_lookup_with_evidence_vault(
        {
            "Acme": [
                {"feature": "reporting", "mentions": 2},
                {"feature": "api access", "mentions": 1},
            ],
        },
        {
            "Acme": {
                "weakness_evidence": [
                    {
                        "key": "reporting",
                        "label": "Reporting",
                        "evidence_type": "feature_gap",
                        "mention_count_total": 7,
                    },
                ],
            },
        },
    )

    assert merged["Acme"][0] == {"feature": "Reporting", "mentions": 7}
    assert merged["Acme"][1]["feature"] == "api access"


def test_merge_company_lookup_with_evidence_vault_enriches_raw_entries():
    merged = _merge_company_lookup_with_evidence_vault(
        {
            "Acme": [
                {
                    "company": "Acme Corp",
                    "urgency": 7.0,
                    "title": "VP Ops",
                    "company_size": "mid_market",
                    "industry": "SaaS",
                },
            ],
        },
        {
            "Acme": {
                "company_signals": [
                    {
                        "company_name": "acme",
                        "urgency_score": 8.0,
                        "buyer_role": "executive",
                        "decision_maker": True,
                        "buying_stage": "evaluation",
                        "source": "g2",
                        "confidence_score": 0.83,
                        "first_seen_at": "2026-03-01T00:00:00+00:00",
                        "last_seen_at": "2026-03-19T00:00:00+00:00",
                        "review_id": "r1",
                    },
                ],
            },
        },
    )

    company = merged["Acme"][0]
    assert company["company"] == "Acme Corp"
    assert company["urgency"] == 8.0
    assert company["title"] == "VP Ops"
    assert company["company_size"] == "mid_market"
    assert company["industry"] == "SaaS"
    assert company["source"] == "g2"
    assert company["buying_stage"] == "evaluation"
    assert company["confidence_score"] == 0.83
    assert company["decision_maker"] is True
    assert company["first_seen_at"] == "2026-03-01T00:00:00+00:00"
    assert company["last_seen_at"] == "2026-03-19T00:00:00+00:00"
