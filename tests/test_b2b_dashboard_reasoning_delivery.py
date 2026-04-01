from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _make_view(vendor_name: str):
    from atlas_brain.autonomous.tasks._b2b_synthesis_reader import load_synthesis_view

    return load_synthesis_view(
        {
            "reasoning_contracts": {
                "vendor_core_reasoning": {
                    "causal_narrative": {
                        "primary_wedge": "price_squeeze",
                        "summary": "Pricing pressure is driving evaluation activity.",
                        "key_signals": ["Q1 price increase", "renewal scrutiny"],
                        "what_would_weaken_thesis": [{"condition": "Pricing stabilizes"}],
                        "data_gaps": ["thin enterprise sample"],
                        "confidence": "medium",
                    },
                },
            },
            "reference_ids": {
                "metric_ids": ["metric:test:1"],
                "witness_ids": ["witness:test:1"],
            },
            "meta": {
                "evidence_window_start": "2026-03-01",
                "evidence_window_end": "2026-03-31",
            },
        },
        vendor_name=vendor_name,
        schema_version="v2",
        as_of_date=date(2026, 3, 31),
    )


@pytest.mark.asyncio
async def test_dashboard_list_signals_overlays_synthesis_reasoning(monkeypatch):
    from atlas_brain.api import b2b_dashboard as mod

    pool = SimpleNamespace(
        fetch=AsyncMock(
            return_value=[
                {
                    "vendor_name": "Zendesk",
                    "product_category": "CRM",
                    "total_reviews": 100,
                    "churn_intent_count": 22,
                    "avg_urgency_score": 6.4,
                    "avg_rating_normalized": 0.4,
                    "nps_proxy": -0.2,
                    "price_complaint_rate": 0.18,
                    "decision_maker_churn_rate": 0.12,
                    "support_sentiment": -0.1,
                    "legacy_support_score": -0.2,
                    "new_feature_velocity": 0.3,
                    "employee_growth_rate": 0.04,
                    "archetype": "legacy_arch",
                    "archetype_confidence": 0.2,
                    "reasoning_mode": "legacy",
                    "reasoning_risk_level": "low",
                    "keyword_spike_count": 2,
                    "insider_signal_count": 1,
                    "last_computed_at": None,
                },
            ]
        ),
        fetchrow=AsyncMock(
            return_value={
                "total_vendors": 1,
                "high_urgency_count": 0,
                "total_signal_reviews": 100,
            }
        ),
    )
    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)
    monkeypatch.setattr(
        mod,
        "_load_reasoning_views_for_vendors",
        AsyncMock(return_value={"zendesk": _make_view("Zendesk")}),
    )

    result = await mod.list_signals(
        vendor_name=None,
        min_urgency=0,
        category=None,
        limit=20,
        user=None,
    )

    signal = result["signals"][0]
    assert signal["archetype"] == "price_squeeze"
    assert signal["reasoning_mode"] == "synthesis"
    assert signal["reasoning_risk_level"] == "high"


@pytest.mark.asyncio
async def test_dashboard_get_signal_overlays_synthesis_detail(monkeypatch):
    from atlas_brain.api import b2b_dashboard as mod

    pool = SimpleNamespace(
        fetchrow=AsyncMock(
            return_value={
                "id": "sig-1",
                "vendor_name": "Zendesk",
                "product_category": "CRM",
                "total_reviews": 100,
                "negative_reviews": 30,
                "churn_intent_count": 22,
                "avg_urgency_score": 6.4,
                "avg_rating_normalized": 0.4,
                "nps_proxy": -0.2,
                "price_complaint_rate": 0.18,
                "decision_maker_churn_rate": 0.12,
                "support_sentiment": -0.1,
                "legacy_support_score": -0.2,
                "new_feature_velocity": 0.3,
                "employee_growth_rate": 0.04,
                "top_pain_categories": [],
                "top_competitors": [],
                "top_feature_gaps": [],
                "company_churn_list": [],
                "quotable_evidence": [],
                "top_use_cases": [],
                "top_integration_stacks": [],
                "budget_signal_summary": {},
                "sentiment_distribution": {},
                "buyer_authority_summary": {},
                "timeline_summary": {},
                "source_distribution": {},
                "sample_review_ids": [],
                "review_window_start": None,
                "review_window_end": None,
                "confidence_score": 0.4,
                "archetype": "legacy_arch",
                "archetype_confidence": 0.2,
                "reasoning_mode": "legacy",
                "falsification_conditions": [],
                "reasoning_risk_level": "low",
                "reasoning_executive_summary": "legacy summary",
                "reasoning_key_signals": [],
                "reasoning_uncertainty_sources": [],
                "insider_signal_count": 1,
                "insider_org_health_summary": None,
                "insider_talent_drain_rate": None,
                "insider_quotable_evidence": [],
                "keyword_spike_count": 2,
                "keyword_spike_keywords": [],
                "keyword_trend_summary": None,
                "last_computed_at": None,
                "created_at": None,
            }
        ),
    )
    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)
    monkeypatch.setattr(
        mod,
        "_load_reasoning_views_for_vendors",
        AsyncMock(return_value={"zendesk": _make_view("Zendesk")}),
    )
    monkeypatch.setattr(
        mod,
        "_apply_field_overrides",
        AsyncMock(side_effect=lambda pool, entity_type, entity_id, payload: payload),
    )

    result = await mod.get_signal("Zendesk", product_category=None, user=None)

    assert result["reasoning_executive_summary"] == "Pricing pressure is driving evaluation activity."
    assert result["reasoning_reference_ids"]["witness_ids"] == ["witness:test:1"]
    assert result["reasoning_source"] == "b2b_reasoning_synthesis"


@pytest.mark.asyncio
async def test_tenant_get_vendor_detail_overlays_synthesis_detail(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    pool = SimpleNamespace(
        fetchrow=AsyncMock(
            side_effect=[
                {
                    "vendor_name": "Zendesk",
                    "avg_urgency_score": 6.4,
                    "churn_intent_count": 22,
                    "total_reviews": 100,
                    "nps_proxy": -0.2,
                    "price_complaint_rate": 0.18,
                    "decision_maker_churn_rate": 0.12,
                    "support_sentiment": -0.1,
                    "legacy_support_score": -0.2,
                    "new_feature_velocity": 0.3,
                    "employee_growth_rate": 0.04,
                    "top_pain_categories": [],
                    "top_competitors": [],
                    "top_feature_gaps": [],
                    "quotable_evidence": [],
                    "top_use_cases": [],
                    "top_integration_stacks": [],
                    "budget_signal_summary": {},
                    "sentiment_distribution": {},
                    "buyer_authority_summary": {},
                    "timeline_summary": {},
                    "last_computed_at": None,
                },
                {"total_reviews": 100, "enriched": 80},
            ]
        ),
        fetch=AsyncMock(return_value=[]),
    )
    user = SimpleNamespace(account_id="acct-1", product="b2b_retention", role="owner", is_admin=False)
    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", False, raising=False)
    monkeypatch.setattr(
        mod,
        "_load_reasoning_views_for_vendors",
        AsyncMock(return_value={"zendesk": _make_view("Zendesk")}),
    )

    result = await mod.get_vendor_detail("Zendesk", user=user)

    assert result["churn_signal"]["archetype"] == "price_squeeze"
    assert result["churn_signal"]["reasoning_executive_summary"] == "Pricing pressure is driving evaluation activity."
    assert result["churn_signal"]["reasoning_reference_ids"]["metric_ids"] == ["metric:test:1"]
