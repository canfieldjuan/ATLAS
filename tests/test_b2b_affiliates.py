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


@pytest.mark.asyncio
async def test_list_opportunities_normalizes_blank_vendor_filter(monkeypatch):
    from atlas_brain.api import b2b_affiliates as mod

    pool = SimpleNamespace(is_initialized=True, fetch=AsyncMock(return_value=[]))
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)

    result = await mod.list_opportunities(
        min_urgency=5,
        min_score=0,
        window_days=90,
        limit=50,
        vendor_name="   ",
        dm_only=False,
    )

    assert result == {"basis": "canonical_reviews", "opportunities": [], "count": 0}
    sql, *params = pool.fetch.await_args.args
    assert "AND r.vendor_name ILIKE '%' || $4 || '%'" not in sql
    assert params == [90, 5, 50]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field_name", "body"),
    [
        ("name", lambda mod: mod.PartnerCreate(name="   ", product_name="Shopify", affiliate_url="https://example.com")),
        ("product_name", lambda mod: mod.PartnerCreate(name="Shopify", product_name="   ", affiliate_url="https://example.com")),
        ("affiliate_url", lambda mod: mod.PartnerCreate(name="Shopify", product_name="Shopify", affiliate_url="   ")),
        ("commission_type", lambda mod: mod.PartnerCreate(name="Shopify", product_name="Shopify", affiliate_url="https://example.com", commission_type="   ")),
    ],
)
async def test_create_partner_rejects_blank_required_text_before_db_touch(monkeypatch, field_name, body):
    from atlas_brain.api import b2b_affiliates as mod

    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.create_partner(body(mod))

    assert exc.value.status_code == 422
    assert exc.value.detail == f"{field_name} is required"


@pytest.mark.asyncio
async def test_create_partner_trims_and_normalizes_body_before_persistence(monkeypatch):
    from atlas_brain.api import b2b_affiliates as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(return_value={"id": uuid4(), "created_at": datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)}),
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)

    await mod.create_partner(
        mod.PartnerCreate(
            name="  Shopify Affiliate  ",
            product_name="  Shopify  ",
            product_aliases=["  shopify plus  ", "   ", "shopify commerce"],
            category="   ",
            affiliate_url="  https://example.com/affiliate  ",
            commission_type=" recurring ",
            commission_value="   ",
            notes="   ",
        )
    )

    _, name, product_name, aliases, category, affiliate_url, commission_type, commission_value, notes, enabled = pool.fetchrow.await_args.args
    assert name == "Shopify Affiliate"
    assert product_name == "Shopify"
    assert aliases == ["shopify plus", "shopify commerce"]
    assert category is None
    assert affiliate_url == "https://example.com/affiliate"
    assert commission_type == "recurring"
    assert commission_value is None
    assert notes is None
    assert enabled is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "field_name"),
    [
        (lambda mod: mod.PartnerUpdate(name="   "), "name"),
        (lambda mod: mod.PartnerUpdate(product_name="   "), "product_name"),
        (lambda mod: mod.PartnerUpdate(affiliate_url="   "), "affiliate_url"),
        (lambda mod: mod.PartnerUpdate(commission_type="   "), "commission_type"),
    ],
)
async def test_update_partner_rejects_blank_required_text_before_db_touch(monkeypatch, body, field_name):
    from atlas_brain.api import b2b_affiliates as mod

    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.update_partner(str(uuid4()), body(mod))

    assert exc.value.status_code == 422
    assert exc.value.detail == f"{field_name} is required"


@pytest.mark.asyncio
async def test_update_partner_trims_and_normalizes_fields_before_persistence(monkeypatch):
    from atlas_brain.api import b2b_affiliates as mod

    partner_id = uuid4()
    pool = SimpleNamespace(is_initialized=True, execute=AsyncMock(return_value="UPDATE 1"))
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)

    await mod.update_partner(
        str(partner_id),
        mod.PartnerUpdate(
            name="  Updated Partner  ",
            product_aliases=["  suite  ", "   "],
            category="   ",
            commission_value="   ",
            notes="  trimmed note  ",
        ),
    )

    sql, *params = pool.execute.await_args.args
    assert "name = $1" in sql
    assert "product_aliases = $2" in sql
    assert "category = $3" in sql
    assert "commission_value = $4" in sql
    assert "notes = $5" in sql
    assert params == ["Updated Partner", ["suite"], None, None, "trimmed note", partner_id]


@pytest.mark.asyncio
async def test_record_click_normalizes_blank_optional_fields(monkeypatch):
    from atlas_brain.api import b2b_affiliates as mod

    partner_id = uuid4()
    pool = SimpleNamespace(is_initialized=True, execute=AsyncMock(return_value="INSERT 0 1"))
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)

    await mod.record_click(
        mod.ClickRecord(partner_id=str(partner_id), review_id="   ", referrer="   ")
    )

    _, pid, rid, referrer = pool.execute.await_args.args
    assert pid == partner_id
    assert rid is None
    assert referrer == "dashboard"
