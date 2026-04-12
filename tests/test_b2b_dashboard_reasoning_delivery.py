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
async def test_dashboard_list_signals_normalizes_blank_filters(monkeypatch):
    from atlas_brain.api import b2b_dashboard as mod

    pool = SimpleNamespace(fetch=AsyncMock(return_value=[]), fetchrow=AsyncMock(return_value=None))
    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)
    read_rows = AsyncMock(return_value=[])
    read_summary = AsyncMock(return_value=None)
    monkeypatch.setattr(mod, "read_vendor_signal_rows", read_rows)
    monkeypatch.setattr(mod, "read_vendor_signal_summary", read_summary)
    monkeypatch.setattr(mod, "_load_reasoning_views_for_vendors", AsyncMock(return_value={}))

    result = await mod.list_signals(
        vendor_name="   ",
        min_urgency=0,
        category="",
        limit=20,
        user=None,
    )

    read_rows.assert_awaited_once_with(
        pool,
        vendor_name_query=None,
        min_urgency=0,
        product_category=None,
        tracked_account_id=None,
        include_snapshot_metrics=True,
        exclude_suppressed=True,
        limit=20,
    )
    read_summary.assert_awaited_once_with(
        pool,
        vendor_name_query=None,
        min_urgency=0,
        product_category=None,
        tracked_account_id=None,
        exclude_suppressed=True,
    )
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_dashboard_list_signals_does_not_fallback_to_legacy_reasoning(monkeypatch):
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
        AsyncMock(return_value={}),
    )

    result = await mod.list_signals(
        vendor_name=None,
        min_urgency=0,
        category=None,
        limit=20,
        user=None,
    )

    signal = result["signals"][0]
    assert signal["archetype"] is None
    assert signal["archetype_confidence"] is None
    assert signal["reasoning_mode"] is None
    assert signal["reasoning_risk_level"] is None


@pytest.mark.asyncio
async def test_dashboard_list_slow_burn_watchlist_normalizes_blank_filters(monkeypatch):
    from atlas_brain.api import b2b_dashboard as mod

    pool = SimpleNamespace()
    read_rows = AsyncMock(return_value=[])
    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)
    monkeypatch.setattr(mod, "read_ranked_vendor_signal_rows", read_rows)
    monkeypatch.setattr(mod, "_load_reasoning_views_for_vendors", AsyncMock(return_value={}))

    result = await mod.list_slow_burn_watchlist(
        vendor_name="   ",
        category="",
        limit=10,
        user=None,
    )

    read_rows.assert_awaited_once_with(
        pool,
        vendor_name_query=None,
        product_category=None,
        tracked_account_id=None,
        exclude_suppressed=True,
        require_snapshot_activity=True,
        limit=10,
    )
    assert result == {"signals": [], "count": 0}


@pytest.mark.asyncio
async def test_dashboard_get_signal_normalizes_blank_product_category(monkeypatch):
    from atlas_brain.api import b2b_dashboard as mod

    pool = SimpleNamespace()
    row = {
        "id": "sig-1",
        "vendor_name": "Zendesk",
        "product_category": "CRM",
        "total_reviews": 1,
        "negative_reviews": 0,
        "churn_intent_count": 0,
        "avg_urgency_score": 0.0,
        "avg_rating_normalized": 0.0,
        "nps_proxy": 0.0,
        "price_complaint_rate": 0.0,
        "decision_maker_churn_rate": 0.0,
        "support_sentiment": 0.0,
        "legacy_support_score": 0.0,
        "new_feature_velocity": 0.0,
        "employee_growth_rate": 0.0,
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
        "insider_signal_count": 0,
        "insider_org_health_summary": None,
        "insider_talent_drain_rate": None,
        "insider_quotable_evidence": [],
        "keyword_spike_count": 0,
        "keyword_spike_keywords": [],
        "keyword_trend_summary": None,
        "last_computed_at": None,
        "created_at": None,
    }
    read_detail = AsyncMock(return_value=row)
    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)
    monkeypatch.setattr(mod, "read_vendor_signal_detail", read_detail)
    monkeypatch.setattr(mod, "_load_reasoning_views_for_vendors", AsyncMock(return_value={}))
    monkeypatch.setattr(mod, "_apply_field_overrides", AsyncMock(side_effect=lambda pool, entity_type, entity_id, payload: payload))

    result = await mod.get_signal("Zendesk", product_category="   ", user=None)

    read_detail.assert_awaited_once_with(
        pool,
        vendor_name_query="Zendesk",
        product_category=None,
        tracked_account_id=None,
        include_snapshot_metrics=True,
        exclude_suppressed=True,
    )
    assert result["vendor_name"] == "Zendesk"


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
async def test_dashboard_export_signals_normalizes_blank_filters(monkeypatch):
    from atlas_brain.api import b2b_dashboard as mod

    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    pool = SimpleNamespace()
    read_rows = AsyncMock(return_value=[])

    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)
    monkeypatch.setattr(shared_mod, "read_vendor_signal_rows", read_rows)
    monkeypatch.setattr(mod, "_load_reasoning_views_for_vendors", AsyncMock(return_value={}))
    monkeypatch.setattr(mod, "_csv_response", lambda payload, filename: {"rows": payload, "filename": filename})

    result = await mod.export_signals(
        vendor_name="  ",
        min_urgency=5.0,
        category="",
        user=None,
    )

    read_rows.assert_awaited_once_with(
        pool,
        vendor_name_query=None,
        min_urgency=5.0,
        product_category=None,
        tracked_account_id=None,
        include_snapshot_metrics=True,
        exclude_suppressed=True,
        limit=mod.EXPORT_ROW_LIMIT,
    )
    assert result == {"rows": [], "filename": "churn_signals.csv"}


@pytest.mark.asyncio
async def test_dashboard_export_signals_uses_shared_signal_row_adapter(monkeypatch):
    from atlas_brain.api import b2b_dashboard as mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    pool = SimpleNamespace()
    rows = [
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
            "keyword_spike_count": 2,
            "insider_signal_count": 1,
            "last_computed_at": None,
        },
    ]
    read_rows = AsyncMock(return_value=rows)

    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)
    monkeypatch.setattr(shared_mod, "read_vendor_signal_rows", read_rows)
    monkeypatch.setattr(
        mod,
        "_load_reasoning_views_for_vendors",
        AsyncMock(return_value={"zendesk": _make_view("Zendesk")}),
    )
    monkeypatch.setattr(
        mod,
        "_csv_response",
        lambda payload, filename: {"rows": payload, "filename": filename},
    )

    user = SimpleNamespace(account_id="11111111-1111-1111-1111-111111111111", is_admin=False)
    result = await mod.export_signals(
        vendor_name="Zendesk",
        min_urgency=5.0,
        category="CRM",
        user=user,
    )

    read_rows.assert_awaited_once_with(
        pool,
        vendor_name_query="Zendesk",
        min_urgency=5.0,
        product_category="CRM",
        tracked_account_id="11111111-1111-1111-1111-111111111111",
        include_snapshot_metrics=True,
        exclude_suppressed=True,
        limit=mod.EXPORT_ROW_LIMIT,
    )
    assert result["filename"] == "churn_signals.csv"
    row = result["rows"][0]
    assert row["vendor_name"] == "Zendesk"
    assert row["support_sentiment"] == -0.1
    assert row["keyword_spike_count"] == 2
    assert row["archetype"] == "price_squeeze"
    assert row["reasoning_risk_level"] == "high"


@pytest.mark.asyncio
async def test_dashboard_get_signal_does_not_fallback_to_legacy_reasoning(monkeypatch):
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
                "falsification_conditions": [{"condition": "legacy"}],
                "reasoning_risk_level": "low",
                "reasoning_executive_summary": "legacy summary",
                "reasoning_key_signals": ["legacy signal"],
                "reasoning_uncertainty_sources": ["legacy uncertainty"],
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
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        mod,
        "_apply_field_overrides",
        AsyncMock(side_effect=lambda pool, entity_type, entity_id, payload: payload),
    )

    result = await mod.get_signal("Zendesk", product_category=None, user=None)

    assert result["archetype"] is None
    assert result["archetype_confidence"] is None
    assert result["reasoning_mode"] is None
    assert result["reasoning_risk_level"] is None
    assert result["falsification_conditions"] == []
    assert result["reasoning_executive_summary"] is None
    assert result["reasoning_key_signals"] == []
    assert result["reasoning_uncertainty_sources"] == []


@pytest.mark.asyncio
async def test_dashboard_get_signal_uses_filtered_reasoning_contracts(monkeypatch):
    from atlas_brain.api import b2b_dashboard as mod
    from atlas_brain.autonomous.tasks._b2b_synthesis_reader import load_synthesis_view

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
                "confidence_score": 0.42,
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
        "_apply_field_overrides",
        AsyncMock(side_effect=lambda pool, entity_type, entity_id, payload: payload),
    )
    monkeypatch.setattr(
        mod,
        "_load_reasoning_views_for_vendors",
        AsyncMock(
            return_value={
                "zendesk": load_synthesis_view(
                    {
                        "reasoning_contracts": {
                            "vendor_core_reasoning": {
                                "causal_narrative": {
                                    "primary_wedge": "price_squeeze",
                                    "summary": "Pricing pressure is driving evaluation activity.",
                                    "data_gaps": [],
                                    "confidence": "medium",
                                },
                            },
                            "account_reasoning": {
                                "confidence": "insufficient",
                                "data_gaps": ["Section missing from model output"],
                                "top_accounts": [],
                            },
                        },
                    },
                    vendor_name="Zendesk",
                    schema_version="v2",
                    as_of_date=date(2026, 3, 31),
                ),
            }
        ),
    )

    result = await mod.get_signal("Zendesk", product_category=None, user=None)

    assert "account_reasoning" not in result.get("reasoning_contracts", {})
    assert "account_reasoning:suppressed" in result["reasoning_contract_gaps"]


@pytest.mark.asyncio
async def test_dashboard_get_signal_surfaces_section_disclaimers(monkeypatch):
    from atlas_brain.api import b2b_dashboard as mod
    from atlas_brain.autonomous.tasks._b2b_synthesis_reader import load_synthesis_view

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
                "confidence_score": 0.42,
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
        "_apply_field_overrides",
        AsyncMock(side_effect=lambda pool, entity_type, entity_id, payload: payload),
    )
    monkeypatch.setattr(
        mod,
        "_load_reasoning_views_for_vendors",
        AsyncMock(
            return_value={
                "zendesk": load_synthesis_view(
                    {
                        "reasoning_contracts": {
                            "vendor_core_reasoning": {
                                "causal_narrative": {
                                    "primary_wedge": "price_squeeze",
                                    "summary": "Pricing pressure is driving evaluation activity.",
                                    "data_gaps": [],
                                    "confidence": "medium",
                                },
                                "timing_intelligence": {
                                    "best_timing_window": "Q2 renewal",
                                    "confidence": "low",
                                },
                            },
                        },
                    },
                    vendor_name="Zendesk",
                    schema_version="v2",
                    as_of_date=date(2026, 3, 31),
                ),
            }
        ),
    )

    result = await mod.get_signal("Zendesk", product_category=None, user=None)

    assert result["reasoning_section_disclaimers"]["timing_intelligence"]


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


@pytest.mark.asyncio
async def test_tenant_list_signals_does_not_fallback_to_legacy_reasoning(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

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
    user = SimpleNamespace(account_id="acct-1", product="b2b_retention", role="owner", is_admin=False)
    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)
    monkeypatch.setattr(mod.settings.saas_auth, "enabled", False, raising=False)
    monkeypatch.setattr(mod, "_tenant_params", lambda _user: [])
    monkeypatch.setattr(mod, "_vendor_scope_sql", lambda idx, _user: "TRUE")
    monkeypatch.setattr(
        mod,
        "_load_reasoning_views_for_vendors",
        AsyncMock(return_value={}),
    )

    result = await mod.list_tenant_signals(
        vendor_name=None,
        min_urgency=0,
        category=None,
        limit=20,
        user=user,
    )

    signal = result["signals"][0]
    assert signal["archetype"] is None
    assert signal["archetype_confidence"] is None
    assert signal["reasoning_mode"] is None


@pytest.mark.asyncio
async def test_dashboard_vendor_profile_only_uses_trusted_account_resolution(monkeypatch):
    from atlas_brain.api import b2b_dashboard as mod

    pool = SimpleNamespace(
        fetchrow=AsyncMock(
            side_effect=[
                None,
                {"total_reviews": 0, "pending_enrichment": 0, "enriched": 0},
            ]
        ),
        fetch=AsyncMock(side_effect=[[], []]),
    )
    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)

    result = await mod.get_vendor_profile("  Zendesk  ", user=None)

    assert result["vendor_name"] == "Zendesk"
    hi_sql = pool.fetch.await_args_list[0].args[0]
    counts_sql = pool.fetchrow.await_args_list[1].args[0]
    assert "WHEN ar.confidence_label IN ('high', 'medium')" in hi_sql
    assert "duplicate_of_review_id IS NULL" in counts_sql


@pytest.mark.asyncio
async def test_dashboard_vendor_profile_rejects_blank_vendor_name_without_db_touch(monkeypatch):
    from atlas_brain.api import b2b_dashboard as mod

    monkeypatch.setattr(
        mod,
        "_pool_or_503",
        lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")),
    )

    with pytest.raises(mod.HTTPException) as exc:
        await mod.get_vendor_profile("   ", user=None)

    assert exc.value.status_code == 400
    assert exc.value.detail == "vendor_name is required"


@pytest.mark.asyncio
async def test_reason_vendor_rejects_blank_vendor_name_without_db_touch(monkeypatch):
    from atlas_brain.api import b2b_dashboard as mod

    monkeypatch.setattr(
        mod,
        "_pool_or_503",
        lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")),
    )

    with pytest.raises(mod.HTTPException) as exc:
        await mod.reason_vendor("   ", user=None)

    assert exc.value.status_code == 400
    assert exc.value.detail == "vendor_name is required"


@pytest.mark.asyncio
async def test_reason_vendor_trims_vendor_name_before_reader_call(monkeypatch):
    from atlas_brain.api import b2b_dashboard as mod
    from atlas_brain.autonomous.tasks import _b2b_synthesis_reader as reader_mod

    pool = SimpleNamespace()
    load_mock = AsyncMock(return_value=_make_view("Zendesk"))
    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)
    monkeypatch.setattr(reader_mod, "load_best_reasoning_view", load_mock)
    monkeypatch.setattr(
        reader_mod,
        "synthesis_view_to_reasoning_entry",
        lambda _view: {
            "mode": "cached",
            "confidence": "medium",
            "archetype": "price_squeeze",
            "risk_level": "high",
            "executive_summary": "Summary",
            "key_signals": ["signal"],
            "falsification_conditions": ["condition"],
            "uncertainty_sources": ["source"],
        },
    )

    result = await mod.reason_vendor("  Zendesk  ", user=None)

    load_mock.assert_awaited_once_with(pool, "Zendesk")
    assert result["vendor_name"] == "Zendesk"
    assert result["cached"] is True


@pytest.mark.asyncio
async def test_compare_vendor_reasoning_rejects_blank_vendor_entries_without_db_touch(monkeypatch):
    from atlas_brain.api import b2b_dashboard as mod

    monkeypatch.setattr(
        mod,
        "_pool_or_503",
        lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")),
    )

    with pytest.raises(mod.HTTPException) as exc:
        await mod.compare_vendor_reasoning({"vendors": ["  ", "Zendesk"]}, user=None)

    assert exc.value.status_code == 400
    assert exc.value.detail == "vendors must be a list of 2-5 vendor names"


@pytest.mark.asyncio
async def test_compare_vendor_reasoning_trims_vendor_names_before_reader_call(monkeypatch):
    from atlas_brain.api import b2b_dashboard as mod
    from atlas_brain.autonomous.tasks import _b2b_synthesis_reader as reader_mod

    pool = SimpleNamespace()
    load_mock = AsyncMock(return_value={
        "Zendesk": _make_view("Zendesk"),
        "Intercom": _make_view("Intercom"),
    })
    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)
    monkeypatch.setattr(reader_mod, "load_best_reasoning_views", load_mock)
    monkeypatch.setattr(
        reader_mod,
        "synthesis_view_to_reasoning_entry",
        lambda _view: {
            "mode": "cached",
            "confidence": "medium",
            "archetype": "price_squeeze",
            "risk_level": "high",
            "executive_summary": "Summary",
            "key_signals": ["signal"],
            "falsification_conditions": ["condition"],
        },
    )

    result = await mod.compare_vendor_reasoning({"vendors": ["  Zendesk  ", " Intercom "]}, user=None)

    load_mock.assert_awaited_once_with(pool, ["Zendesk", "Intercom"])
    assert result["count"] == 2
    assert [row["vendor_name"] for row in result["vendors"]] == ["Zendesk", "Intercom"]


@pytest.mark.asyncio
async def test_dashboard_pipeline_excludes_cross_source_duplicates(monkeypatch):
    from atlas_brain.api import b2b_dashboard as mod

    pool = SimpleNamespace(
        fetch=AsyncMock(return_value=[]),
        fetchrow=AsyncMock(side_effect=[
            {"recent_imports_24h": 3, "last_enrichment_at": None},
            {"active_scrape_targets": 1, "last_scrape_at": None},
        ]),
    )
    monkeypatch.setattr(mod, "_pool_or_503", lambda: pool)

    result = await mod.get_pipeline_status(user=None)

    assert result["recent_imports_24h"] == 3
    status_sql = pool.fetch.await_args.args[0]
    stats_sql = pool.fetchrow.await_args_list[0].args[0]
    assert "duplicate_of_review_id IS NULL" in status_sql
    assert "duplicate_of_review_id IS NULL" in stats_sql
