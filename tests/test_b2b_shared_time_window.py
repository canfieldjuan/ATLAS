from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def test_eligible_review_filters_uses_stable_review_timestamp():
    from atlas_brain.autonomous.tasks import _b2b_shared as mod

    sql = mod._eligible_review_filters(window_param=1, source_param=2)

    assert "COALESCE(reviewed_at, imported_at, enriched_at)" in sql
    assert "enrichment_status IN ('enriched', 'no_signal', 'quarantined')" in sql
    assert "duplicate_of_review_id IS NULL" in sql
    assert "source = ANY($2::text[])" in sql


def test_eligible_review_filters_respects_alias_for_stable_timestamp():
    from atlas_brain.autonomous.tasks import _b2b_shared as mod

    sql = mod._eligible_review_filters(window_param=3, source_param=4, alias="r")

    assert "COALESCE(r.reviewed_at, r.imported_at, r.enriched_at)" in sql
    assert "r.duplicate_of_review_id IS NULL" in sql
    assert "r.source = ANY($4::text[])" in sql


@pytest.mark.asyncio
async def test_fetch_vendor_provenance_uses_stable_review_window(monkeypatch):
    from atlas_brain.autonomous.tasks import _b2b_shared as mod

    pool = SimpleNamespace(
        fetch=AsyncMock(
            side_effect=[
                [{"vendor_name": "Acme", "source": "g2", "cnt": 2}],
                [{
                    "vendor_name": "Acme",
                    "sample_ids": ["r1", "r2"],
                    "window_start": "2026-03-01T00:00:00+00:00",
                    "window_end": "2026-03-18T00:00:00+00:00",
                }],
            ],
        ),
    )
    monkeypatch.setattr(
        mod.settings.b2b_churn,
        "intelligence_source_allowlist",
        "g2,reddit",
        raising=False,
    )
    monkeypatch.setattr(
        mod.settings.b2b_churn,
        "deprecated_review_sources",
        "",
        raising=False,
    )

    result = await mod._fetch_vendor_provenance(pool, 90)

    assert result["Acme"]["source_distribution"] == {"g2": 2}
    dist_sql = pool.fetch.await_args_list[0].args[0]
    sample_sql = pool.fetch.await_args_list[1].args[0]
    assert "JOIN b2b_review_vendor_mentions vm" in dist_sql
    assert "GROUP BY vm.vendor_name, r.source" in dist_sql
    assert "MIN(COALESCE(r.reviewed_at, r.imported_at, r.enriched_at)) AS window_start" in sample_sql
    assert "MAX(COALESCE(r.reviewed_at, r.imported_at, r.enriched_at)) AS window_end" in sample_sql
    assert "JOIN b2b_review_vendor_mentions vm" in sample_sql
    assert "GROUP BY vm.vendor_name" in sample_sql


@pytest.mark.asyncio
async def test_fetch_vendor_churn_scores_reads_vendor_mentions(monkeypatch):
    from atlas_brain.autonomous.tasks import _b2b_shared as mod

    pool = SimpleNamespace(
        fetch=AsyncMock(
            return_value=[
                {
                    "vendor_name": "Acme",
                    "product_category": "CRM",
                    "total_reviews": 4,
                    "signal_reviews": 3,
                    "churn_intent": 2,
                    "avg_urgency": 7.2,
                    "avg_author_churn_score": 0.6,
                    "avg_rating_normalized": 0.7,
                    "recommend_yes": 2,
                    "recommend_no": 1,
                    "recommend_total": 3,
                    "positive_review_pct": 50.0,
                    "support_sentiment": 0.8,
                    "legacy_support_score": 0.6,
                    "new_feature_velocity": 0.25,
                    "indicator_cancel_count": 1,
                    "indicator_migration_count": 1,
                    "indicator_evaluation_count": 1,
                    "indicator_switch_count": 0,
                    "indicator_named_alt_count": 1,
                    "indicator_dm_language_count": 1,
                    "has_pricing_phrases_count": 2,
                    "has_recommendation_language_count": 1,
                    "employee_count": None,
                    "vendor_industry": None,
                    "annual_revenue_range": None,
                    "employee_growth_rate": None,
                },
            ]
        ),
    )
    monkeypatch.setattr(
        mod.settings.b2b_churn,
        "intelligence_source_allowlist",
        "g2,reddit",
        raising=False,
    )
    monkeypatch.setattr(
        mod.settings.b2b_churn,
        "deprecated_review_sources",
        "",
        raising=False,
    )

    rows = await mod._fetch_vendor_churn_scores(pool, 90, 2)

    sql = pool.fetch.await_args.args[0]
    assert "JOIN b2b_review_vendor_mentions vm" in sql
    assert "GROUP BY vm.vendor_name" in sql
    assert rows[0]["vendor_name"] == "Acme"


@pytest.mark.asyncio
async def test_fetch_buyer_profile_provenance_reads_vendor_mentions(monkeypatch):
    from atlas_brain.autonomous.tasks import _b2b_shared as mod

    pool = SimpleNamespace(
        fetch=AsyncMock(
            return_value=[
                {
                    "vendor_name": "Acme",
                    "role_type": "economic_buyer",
                    "buying_stage": "evaluation",
                    "source": "g2",
                    "cnt": 2,
                    "dm_cnt": 1,
                    "avg_urg": 7.0,
                    "review_ids": ["r1", "r2"],
                },
            ]
        ),
    )
    monkeypatch.setattr(
        mod.settings.b2b_churn,
        "intelligence_source_allowlist",
        "g2,reddit",
        raising=False,
    )
    monkeypatch.setattr(
        mod.settings.b2b_churn,
        "deprecated_review_sources",
        "",
        raising=False,
    )

    rows = await mod._fetch_buyer_profile_provenance(pool, 90)

    sql = pool.fetch.await_args.args[0]
    assert "JOIN b2b_review_vendor_mentions vm" in sql
    assert "GROUP BY vm.vendor_name" in sql
    assert rows[("Acme", "economic_buyer", "evaluation")]["source_distribution"] == {"g2": 2}
