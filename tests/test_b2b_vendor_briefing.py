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
