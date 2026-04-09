from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_list_opportunities_uses_review_recency_and_self_match_filters(monkeypatch):
    from atlas_brain.api import b2b_affiliates as mod

    pool = SimpleNamespace(is_initialized=True, fetch=AsyncMock(return_value=[]))
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)

    result = await mod.list_opportunities(
        min_urgency=5,
        min_score=0,
        window_days=90,
        limit=50,
        vendor_name=None,
        dm_only=False,
    )
    assert result == {"basis": "canonical_reviews", "opportunities": [], "count": 0}

    sql = pool.fetch.await_args.args[0]
    assert "COALESCE(r.reviewed_at, r.imported_at, r.enriched_at)" in sql
    assert "r.duplicate_of_review_id IS NULL" in sql
    assert "rc.reviewer_company IS NOT NULL" in sql
    assert "LOWER(rc.competitor_name) <> LOWER(rc.vendor_name)" in sql
    assert "LOWER(rc.reviewer_company) <> LOWER(rc.competitor_name)" in sql


@pytest.mark.asyncio
async def test_list_opportunities_skips_rows_without_real_company(monkeypatch):
    from atlas_brain.api import b2b_affiliates as mod

    review_id = uuid4()
    partner_id = uuid4()
    reviewed_at = datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc)
    pool = SimpleNamespace(
        is_initialized=True,
        fetch=AsyncMock(
            return_value=[
                {
                    "review_id": review_id,
                    "vendor_name": "BigCommerce",
                    "reviewer_company": None,
                    "reviewer_name": "ShopifyOpsLead",
                    "product_category": "Ecommerce",
                    "source": "reddit",
                    "reviewed_at": reviewed_at,
                    "urgency": 8.0,
                    "is_dm": True,
                    "role_type": "decision_maker",
                    "buying_stage": "evaluation",
                    "seat_count": 120,
                    "contract_end": "Q2",
                    "decision_timeline": "30d",
                    "competitor_name": "Shopify",
                    "mention_context": "considering",
                    "mention_reason": "reliability concerns",
                    "partner_id": partner_id,
                    "partner_name": "Shopify Affiliate",
                    "affiliate_url": "https://example.com/a",
                    "commission_type": "cpa",
                    "commission_value": "$100",
                    "partner_category": "Ecommerce",
                }
            ]
        ),
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)

    result = await mod.list_opportunities(
        min_urgency=5,
        min_score=0,
        window_days=90,
        limit=50,
        vendor_name=None,
        dm_only=False,
    )
    assert result == {"basis": "canonical_reviews", "opportunities": [], "count": 0}
