from unittest.mock import AsyncMock

import pytest

from atlas_brain.autonomous.tasks import b2b_vendor_briefing as briefing_mod


def test_apply_evidence_vault_to_briefing_fills_sparse_fields():
    briefing = {
        "evidence": [],
        "pain_breakdown": [],
        "top_feature_gaps": [],
        "named_accounts": [],
        "review_count": 0,
        "avg_urgency": 0,
        "churn_signal_density": 0,
        "dm_churn_rate": 0,
    }
    vault = {
        "metric_snapshot": {
            "reviews_in_analysis_window": 42,
            "avg_urgency": 6.2,
            "churn_density": 18.5,
            "dm_churn_rate": 0.34,
        },
        "weakness_evidence": [
            {
                "key": "pricing",
                "label": "Pricing opacity",
                "evidence_type": "pain_category",
                "best_quote": "Bills kept climbing every quarter.",
                "quote_source": {
                    "company": "Acme Corp",
                    "reviewer_title": "VP Operations",
                    "company_size": "mid_market",
                    "industry": "SaaS",
                    "source": "g2",
                    "review_id": "r1",
                    "rating": 2.0,
                    "reviewed_at": "2026-03-10",
                },
                "mention_count_total": 12,
                "supporting_metrics": {"avg_urgency_when_mentioned": 7.1},
            },
            {
                "key": "reporting",
                "label": "Reporting gaps",
                "evidence_type": "feature_gap",
                "mention_count_total": 9,
            },
        ],
        "company_signals": [
            {
                "company_name": "Acme Corp",
                "buyer_role": "VP Operations",
                "seat_count": 250,
                "urgency_score": 8.0,
                "pain_category": "pricing",
                "buying_stage": "evaluation",
                "source": "g2",
                "confidence_score": 0.8,
            }
        ],
    }

    used = briefing_mod._apply_evidence_vault_to_briefing(briefing, vault)

    assert used is True
    assert briefing["review_count"] == 42
    assert briefing["avg_urgency"] == 6.2
    assert briefing["churn_signal_density"] == 18.5
    assert briefing["dm_churn_rate"] == 34.0
    assert briefing["pain_breakdown"][0]["category"] == "pricing"
    assert briefing["top_feature_gaps"] == ["Reporting gaps"]
    assert briefing["named_accounts"][0]["company"] == "Acme Corp"
    assert briefing["evidence"][0]["source"] == "g2"
    assert briefing["evidence"][0]["quote"] == "Bills kept climbing every quarter."


def test_apply_reasoning_synthesis_to_briefing_copies_contract_fields():
    briefing = {}
    feed_entry = {
        "reasoning_contracts": {
            "schema_version": "v1",
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "causal_narrative": {"trigger": "Price hike"},
                    "timing_intelligence": {
                        "best_timing_window": "Before renewal",
                        "active_eval_signals": {
                            "value": 3,
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
                "migration_proof": {"confidence": "medium"},
            },
        },
        "synthesis_wedge": "price_squeeze",
        "synthesis_wedge_label": "Price Squeeze",
        "synthesis_schema_version": "v2",
        "evidence_window": {
            "evidence_window_start": "2026-03-01",
            "evidence_window_end": "2026-03-18",
        },
        "evidence_window_days": 17,
        "reasoning_source": "b2b_reasoning_synthesis",
        "category_council": {
            "winner": "Zoho Desk",
            "loser": "Freshdesk",
            "market_regime": "price_competition",
        },
    }

    used = briefing_mod._apply_reasoning_synthesis_to_briefing(briefing, feed_entry)

    assert used is True
    assert briefing["vendor_core_reasoning"]["causal_narrative"]["trigger"] == "Price hike"
    assert briefing["displacement_reasoning"]["migration_proof"]["confidence"] == "medium"
    assert briefing["synthesis_wedge"] == "price_squeeze"
    assert briefing["evidence_window_days"] == 17
    assert briefing["reasoning_source"] == "b2b_reasoning_synthesis"
    assert briefing["category_council"]["winner"] == "Zoho Desk"
    assert briefing["timing_intelligence"]["best_timing_window"] == "Before renewal"
    assert briefing["timing_summary"] == (
        "Before renewal. 3 active evaluation signals are visible right now. "
        "Review sentiment is skewing more negative."
    )
    assert briefing["timing_metrics"]["active_eval_signals"] == 3
    assert briefing["priority_timing_triggers"] == ["Q2 renewal"]
    assert "causal_narrative" not in briefing


def test_apply_reasoning_synthesis_to_briefing_normalizes_flat_feed_sections():
    briefing = {}
    feed_entry = {
        "causal_narrative": {"trigger": "Legacy trigger"},
        "timing_intelligence": {
            "best_timing_window": "Before renewal",
            "active_eval_signals": {
                "value": 1,
                "source_id": "accounts:summary:active_eval_signal_count",
            },
        },
        "migration_proof": {"confidence": "medium"},
        "synthesis_wedge": "price_squeeze",
    }

    used = briefing_mod._apply_reasoning_synthesis_to_briefing(briefing, feed_entry)

    assert used is True
    assert briefing["vendor_core_reasoning"]["causal_narrative"]["trigger"] == "Legacy trigger"
    assert briefing["vendor_core_reasoning"]["timing_intelligence"]["best_timing_window"] == "Before renewal"
    assert briefing["timing_summary"] == (
        "Before renewal. 1 active evaluation signals are visible right now."
    )
    assert briefing["displacement_reasoning"]["migration_proof"]["confidence"] == "medium"
    assert "causal_narrative" not in briefing
    assert "migration_proof" not in briefing


def test_apply_reasoning_synthesis_to_briefing_promotes_account_reasoning():
    briefing = {"named_accounts": []}
    feed_entry = {
        "reasoning_contracts": {
            "schema_version": "v1",
            "vendor_core_reasoning": {
                "schema_version": "v1",
                "causal_narrative": {"trigger": "Price hike"},
            },
            "account_reasoning": {
                "schema_version": "v1",
                "market_summary": "Two accounts are actively evaluating alternatives.",
                "total_accounts": {"value": 6, "source_id": "accounts:summary:total_accounts"},
                "high_intent_count": {"value": 4, "source_id": "accounts:summary:high_intent_count"},
                "active_eval_count": {
                    "value": 2,
                    "source_id": "accounts:summary:active_eval_signal_count",
                },
                "top_accounts": [
                    {"name": "Acme Corp", "intent_score": 0.9, "source_id": "accounts:item:acme"},
                ],
            },
        },
    }

    used = briefing_mod._apply_reasoning_synthesis_to_briefing(briefing, feed_entry)

    assert used is True
    assert briefing["account_reasoning"]["top_accounts"][0]["name"] == "Acme Corp"
    assert briefing["account_pressure_summary"] == "Two accounts are actively evaluating alternatives."
    assert briefing["account_pressure_metrics"]["high_intent_count"] == 4
    assert briefing["priority_account_names"] == ["Acme Corp"]
    assert briefing["named_accounts"][0]["company"] == "Acme Corp"
    assert briefing["named_accounts"][0]["reasoning_backed"] is True


def test_apply_reasoning_synthesis_to_briefing_does_not_backfill_missing_explicit_contracts():
    briefing = {}
    feed_entry = {
        "reasoning_contracts": {
            "schema_version": "v1",
            "vendor_core_reasoning": {
                "schema_version": "v1",
                "causal_narrative": {"trigger": "Price hike"},
            },
        },
        "vendor_core_reasoning": {"causal_narrative": {"trigger": "Legacy mirror"}},
        "timing_intelligence": {"best_timing_window": "Before renewal"},
        "migration_proof": {"confidence": "medium"},
    }

    used = briefing_mod._apply_reasoning_synthesis_to_briefing(briefing, feed_entry)

    assert used is True
    assert briefing["vendor_core_reasoning"]["causal_narrative"]["trigger"] == "Price hike"
    assert "timing_intelligence" not in briefing["vendor_core_reasoning"]
    assert "displacement_reasoning" not in briefing


@pytest.mark.asyncio
async def test_build_vendor_briefing_uses_evidence_vault_overlay(monkeypatch):
    class DummyPool:
        is_initialized = True

        async def fetchrow(self, *args, **kwargs):
            return None

        async def fetch(self, *args, **kwargs):
            return []

    vault = {
        "metric_snapshot": {"reviews_in_analysis_window": 30, "avg_urgency": 5.5},
        "weakness_evidence": [
            {
                "key": "pricing",
                "label": "Pricing opacity",
                "evidence_type": "pain_category",
                "best_quote": "The price jumps were impossible to budget.",
                "quote_source": {"source": "capterra"},
                "mention_count_total": 11,
                "supporting_metrics": {"avg_urgency_when_mentioned": 6.4},
            }
        ],
        "company_signals": [
            {
                "company_name": "Northwind",
                "buyer_role": "VP Finance",
                "urgency_score": 7.0,
                "pain_category": "pricing",
                "source": "reddit",
            }
        ],
    }

    monkeypatch.setattr(briefing_mod, "get_db_pool", lambda: DummyPool())
    monkeypatch.setattr(
        briefing_mod,
        "_extract_feed_entry",
        AsyncMock(
            return_value={
                "churn_pressure_score": 48,
                "churn_signal_density": 12.5,
                "avg_urgency": 0,
                "total_reviews": 0,
                "dm_churn_rate": 0,
                "pain_breakdown": [],
                "top_displacement_targets": [],
                "evidence": [],
                "named_accounts": [],
                "top_feature_gaps": [],
                "category": "CRM",
            }
        ),
    )
    monkeypatch.setattr(briefing_mod, "_fetch_vendor_evidence_vault", AsyncMock(return_value=vault))
    monkeypatch.setattr(briefing_mod, "_fetch_product_profile", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "_fetch_high_urgency_quotes", AsyncMock(return_value=[]))
    monkeypatch.setattr(briefing_mod, "_enrich_with_analyst_summary", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "generate_account_cards", AsyncMock(return_value=[]))

    briefing = await briefing_mod.build_vendor_briefing("Zendesk")

    assert briefing is not None
    assert briefing["data_sources"]["weekly_churn_feed"] is True
    assert briefing["data_sources"]["evidence_vault"] is True
    assert briefing["pain_breakdown"][0]["category"] == "pricing"
    assert briefing["evidence"][0]["source"] == "capterra"
    assert briefing["named_accounts"][0]["company"] == "Northwind"
    assert briefing["avg_urgency"] == 5.5


@pytest.mark.asyncio
async def test_build_vendor_briefing_marks_reasoning_synthesis_from_feed(monkeypatch):
    class DummyPool:
        is_initialized = True

        async def fetchrow(self, *args, **kwargs):
            return None

        async def fetch(self, *args, **kwargs):
            return []

    monkeypatch.setattr(briefing_mod, "get_db_pool", lambda: DummyPool())
    monkeypatch.setattr(
        briefing_mod,
        "_extract_feed_entry",
        AsyncMock(
            return_value={
                "churn_pressure_score": 48,
                "churn_signal_density": 12.5,
                "avg_urgency": 4.4,
                "total_reviews": 30,
                "dm_churn_rate": 12.0,
                "pain_breakdown": [],
                "top_displacement_targets": [],
                "evidence": [],
                "named_accounts": [],
                "top_feature_gaps": [],
                "category": "CRM",
                "reasoning_contracts": {
                    "schema_version": "v1",
                    "vendor_core_reasoning": {
                        "schema_version": "v1",
                        "causal_narrative": {"trigger": "Price hike"},
                    },
                },
                "synthesis_wedge": "price_squeeze",
                "reasoning_source": "b2b_reasoning_synthesis",
            }
        ),
    )
    monkeypatch.setattr(briefing_mod, "_fetch_vendor_evidence_vault", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "_fetch_product_profile", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "_fetch_high_urgency_quotes", AsyncMock(return_value=[]))
    monkeypatch.setattr(briefing_mod, "_enrich_with_analyst_summary", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "generate_account_cards", AsyncMock(return_value=[]))

    briefing = await briefing_mod.build_vendor_briefing("Zendesk")

    assert briefing is not None
    assert briefing["data_sources"]["reasoning_synthesis"] is True
    assert briefing["vendor_core_reasoning"]["causal_narrative"]["trigger"] == "Price hike"
    assert "causal_narrative" not in briefing
    assert briefing["synthesis_wedge"] == "price_squeeze"


@pytest.mark.asyncio
async def test_build_vendor_briefing_uses_account_reasoning_named_account_fallback(monkeypatch):
    class DummyPool:
        is_initialized = True

        async def fetchrow(self, *args, **kwargs):
            return None

        async def fetch(self, *args, **kwargs):
            return []

    monkeypatch.setattr(briefing_mod, "get_db_pool", lambda: DummyPool())
    monkeypatch.setattr(
        briefing_mod,
        "_extract_feed_entry",
        AsyncMock(
            return_value={
                "churn_pressure_score": 48,
                "churn_signal_density": 12.5,
                "avg_urgency": 4.4,
                "total_reviews": 30,
                "dm_churn_rate": 12.0,
                "pain_breakdown": [],
                "top_displacement_targets": [],
                "evidence": [],
                "named_accounts": [],
                "top_feature_gaps": [],
                "category": "CRM",
                "reasoning_contracts": {
                    "schema_version": "v1",
                    "vendor_core_reasoning": {
                        "schema_version": "v1",
                        "causal_narrative": {"trigger": "Price hike"},
                    },
                    "account_reasoning": {
                        "schema_version": "v1",
                        "market_summary": "Two accounts are actively evaluating alternatives.",
                        "total_accounts": {"value": 6, "source_id": "accounts:summary:total_accounts"},
                        "high_intent_count": {"value": 4, "source_id": "accounts:summary:high_intent_count"},
                        "active_eval_count": {
                            "value": 2,
                            "source_id": "accounts:summary:active_eval_signal_count",
                        },
                        "top_accounts": [
                            {"name": "Acme Corp", "intent_score": 0.9, "source_id": "accounts:item:acme"},
                        ],
                    },
                },
                "synthesis_wedge": "price_squeeze",
                "reasoning_source": "b2b_reasoning_synthesis",
            }
        ),
    )
    monkeypatch.setattr(briefing_mod, "_fetch_vendor_evidence_vault", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "_fetch_product_profile", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "_fetch_high_urgency_quotes", AsyncMock(return_value=[]))
    monkeypatch.setattr(briefing_mod, "_enrich_with_analyst_summary", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "generate_account_cards", AsyncMock(return_value=[]))

    briefing = await briefing_mod.build_vendor_briefing("Zendesk")

    assert briefing is not None
    assert briefing["data_sources"]["reasoning_synthesis"] is True
    assert briefing["data_sources"]["account_reasoning"] is True
    assert briefing["account_pressure_summary"] == "Two accounts are actively evaluating alternatives."
    assert briefing["named_accounts"][0]["company"] == "Acme Corp"
    assert briefing["executive_summary"] == "Two accounts are actively evaluating alternatives."
