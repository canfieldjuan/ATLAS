from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from atlas_brain.api import seller_campaigns as mod


@pytest.mark.asyncio
async def test_list_seller_targets_normalizes_blank_filters(monkeypatch):
    class Pool:
        is_initialized = True

        async def fetch(self, query, *args):
            assert "status = $1" not in query
            assert "seller_type = $1" not in query
            assert "ANY(categories)" not in query
            assert args == (100, 0)
            return []

        async def fetchval(self, query, *args):
            assert args == ()
            return 0

    monkeypatch.setattr(mod, "get_db_pool", lambda: Pool())

    result = await mod.list_seller_targets(status="   ", seller_type="	", category="  ")

    assert result == {"targets": [], "total": 0}


@pytest.mark.asyncio
async def test_create_seller_target_rejects_blank_names_before_db_touch(monkeypatch):
    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.create_seller_target(
            mod.SellerTargetCreate(seller_name="   ", company_name="	", seller_type="private_label")
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "At least one of seller_name or company_name is required"


@pytest.mark.asyncio
async def test_create_seller_target_trims_and_normalizes_body(monkeypatch):
    row = {"id": uuid4(), "created_at": datetime(2026, 4, 12, 12, 0)}
    pool = SimpleNamespace(is_initialized=True, fetchrow=AsyncMock(return_value=row))
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)

    await mod.create_seller_target(
        mod.SellerTargetCreate(
            seller_name="  Acme Seller  ",
            company_name="   ",
            email="  ops@example.com  ",
            seller_type="  private_label  ",
            categories=["  supplements  ", " ", "beauty"],
            storefront_url="  https://amazon.example/store  ",
            notes="  note  ",
            source="  manual  ",
        )
    )

    args = pool.fetchrow.await_args.args
    assert args[1:] == (
        "Acme Seller",
        None,
        "ops@example.com",
        "private_label",
        ["supplements", "beauty"],
        "https://amazon.example/store",
        "note",
        "manual",
    )


@pytest.mark.asyncio
async def test_target_routes_reject_invalid_or_blank_uuid_before_db_touch(monkeypatch):
    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)

    for fn in (mod.get_seller_target, mod.update_seller_target, mod.delete_seller_target):
        with pytest.raises(mod.HTTPException):
            if fn is mod.update_seller_target:
                await fn("   ", mod.SellerTargetUpdate())
            else:
                await fn("   ")


@pytest.mark.asyncio
async def test_update_seller_target_trims_and_normalizes_fields(monkeypatch):
    target_id = uuid4()
    pool = SimpleNamespace(
        is_initialized=True,
        execute=AsyncMock(return_value="UPDATE 1"),
        fetchrow=AsyncMock(return_value={"id": target_id}),
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)

    await mod.update_seller_target(
        str(target_id),
        mod.SellerTargetUpdate(
            seller_name="  Seller  ",
            email="   ",
            seller_type="  agency  ",
            categories=["  beauty  ", "   "],
            storefront_url="   ",
            notes="  note  ",
            status="  active  ",
        ),
    )

    sql, *params = pool.execute.await_args.args
    assert "seller_name = $1" in sql
    assert "email = $2" in sql
    assert "seller_type = $3" in sql
    assert "storefront_url = $4" in sql
    assert "notes = $5" in sql
    assert "status = $6" in sql
    assert "categories = $7" in sql
    assert params == ["Seller", None, "agency", None, "note", "active", ["beauty"], target_id]


@pytest.mark.asyncio
async def test_list_seller_campaigns_normalizes_blank_filters(monkeypatch):
    class Pool:
        is_initialized = True

        async def fetch(self, query, *args):
            assert "status = $1" not in query
            assert "product_category = $1" not in query
            assert "channel = $1" not in query
            assert "batch_id = $1" not in query
            assert args == (50, 0)
            return []

        async def fetchval(self, query, *args):
            assert args == ()
            return 0

    monkeypatch.setattr(mod, "get_db_pool", lambda: Pool())

    result = await mod.list_seller_campaigns(status="   ", category=" ", channel="	", batch_id="  ")

    assert result == {"campaigns": [], "total": 0}


@pytest.mark.asyncio
async def test_generation_and_intelligence_normalize_blank_category(monkeypatch):
    pool = SimpleNamespace(is_initialized=True, fetch=AsyncMock(return_value=[]))
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.seller_campaign, "enabled", True, raising=False)

    called = {}

    async def _generate_campaigns(*, pool, category_filter, limit):
        called["generate"] = (pool, category_filter, limit)
        return {"generated": 0}

    async def _aggregate_category_intelligence(pool, cat):
        called.setdefault("cats", []).append(cat)
        return None

    async def _save_intelligence_snapshot(pool, intel):
        raise AssertionError("should not be called")

    import atlas_brain.autonomous.tasks.amazon_seller_campaign_generation as gen_mod
    monkeypatch.setattr(gen_mod, "generate_campaigns", _generate_campaigns)
    monkeypatch.setattr(gen_mod, "_aggregate_category_intelligence", _aggregate_category_intelligence)
    monkeypatch.setattr(gen_mod, "_save_intelligence_snapshot", _save_intelligence_snapshot)

    result = await mod.trigger_generation(mod.GenerateRequest(category="   ", limit=5))
    snapshots = await mod.list_category_intelligence(category="   ")
    refreshed = await mod.refresh_category_intelligence(category="   ")

    assert result == {"generated": 0}
    assert called["generate"] == (pool, None, 5)
    assert snapshots == {"snapshots": []}
    assert called.get("cats") is None
    assert pool.fetch.await_count == 2
    assert refreshed == {"refreshed": 0, "categories": 0, "errors": 0}
