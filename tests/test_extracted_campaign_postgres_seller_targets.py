from __future__ import annotations

from datetime import datetime, timezone

import pytest

from extracted_content_pipeline.campaign_postgres_seller_targets import (
    create_seller_target,
    delete_seller_target,
    get_seller_target,
    list_seller_targets,
    update_seller_target,
)


TARGET_ID = "00000000-0000-0000-0000-000000000001"


class _Pool:
    def __init__(self, rows=None, row=None, total: int = 0, execute_result="UPDATE 1"):
        self.rows = list(rows or [])
        self.row = row
        self.total = total
        self.execute_result = execute_result
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetchval_calls: list[tuple[str, tuple[object, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, query, *args):
        self.fetch_calls.append((str(query), args))
        return self.rows

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((str(query), args))
        return self.row

    async def fetchval(self, query, *args):
        self.fetchval_calls.append((str(query), args))
        return self.total

    async def execute(self, query, *args):
        self.execute_calls.append((str(query), args))
        return self.execute_result


def _target_row(**overrides):
    row = {
        "id": TARGET_ID,
        "seller_name": "Acme Seller",
        "company_name": "Acme",
        "email": "owner@example.com",
        "seller_type": "private_label",
        "categories": ["supplements"],
        "storefront_url": "https://example.com/store",
        "notes": "Top category seller",
        "status": "active",
        "source": "manual",
        "created_at": datetime(2026, 5, 4, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 5, 4, tzinfo=timezone.utc),
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_list_seller_targets_filters_rows() -> None:
    pool = _Pool(rows=[_target_row()], total=1)

    result = await list_seller_targets(
        pool,
        status="active",
        seller_type="private_label",
        category="supplements",
        limit=7,
        offset=3,
    )

    query, args = pool.fetch_calls[0]
    assert "FROM \"seller_targets\"" in query
    assert "status = $1" in query
    assert "seller_type = $2" in query
    assert "$3 = ANY(categories)" in query
    assert args == ("active", "private_label", "supplements", 7, 3)
    assert pool.fetchval_calls[0][1] == ("active", "private_label", "supplements")
    assert result.as_dict()["total"] == 1
    assert result.targets[0]["seller_name"] == "Acme Seller"


@pytest.mark.asyncio
async def test_create_seller_target_inserts_normalized_row() -> None:
    pool = _Pool(row=_target_row())

    result = await create_seller_target(
        pool,
        seller_name=" Acme Seller ",
        seller_type="private_label",
        categories=[" supplements ", ""],
        source="manual",
    )

    query, args = pool.fetchrow_calls[0]
    assert "INSERT INTO \"seller_targets\"" in query
    assert args[0] == "Acme Seller"
    assert args[3] == "private_label"
    assert args[4] == ["supplements"]
    assert result["id"] == TARGET_ID


@pytest.mark.asyncio
async def test_create_seller_target_requires_name() -> None:
    with pytest.raises(ValueError, match="seller_name or company_name"):
        await create_seller_target(_Pool(), seller_name=" ", company_name="")


@pytest.mark.asyncio
async def test_create_seller_target_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="unsupported seller_type"):
        await create_seller_target(_Pool(), seller_name="Acme", seller_type="unknown")


@pytest.mark.asyncio
async def test_get_seller_target_fetches_by_uuid() -> None:
    pool = _Pool(row=_target_row())

    result = await get_seller_target(pool, target_id=TARGET_ID)

    assert pool.fetchrow_calls[0][1] == (TARGET_ID,)
    assert result is not None
    assert result["email"] == "owner@example.com"


@pytest.mark.asyncio
async def test_update_seller_target_patches_allowed_fields() -> None:
    pool = _Pool(row=_target_row(status="paused"))

    result = await update_seller_target(
        pool,
        target_id=TARGET_ID,
        values={"status": "paused", "categories": ["supplements", "beauty"]},
    )

    query, args = pool.execute_calls[0]
    assert "UPDATE \"seller_targets\"" in query
    assert "status = $1" in query
    assert "categories = $2" in query
    assert args == ("paused", ["supplements", "beauty"], TARGET_ID)
    assert result is not None
    assert result["status"] == "paused"


@pytest.mark.asyncio
async def test_update_seller_target_returns_none_when_missing() -> None:
    pool = _Pool(execute_result="UPDATE 0")

    result = await update_seller_target(
        pool,
        target_id=TARGET_ID,
        values={"notes": "updated"},
    )

    assert result is None


@pytest.mark.asyncio
async def test_update_seller_target_requires_fields() -> None:
    with pytest.raises(ValueError, match="no fields to update"):
        await update_seller_target(_Pool(), target_id=TARGET_ID, values={})


@pytest.mark.asyncio
async def test_delete_seller_target_reports_result() -> None:
    pool = _Pool(execute_result="DELETE 1")

    assert await delete_seller_target(pool, target_id=TARGET_ID) is True

    query, args = pool.execute_calls[0]
    assert "DELETE FROM \"seller_targets\"" in query
    assert args == (TARGET_ID,)


@pytest.mark.asyncio
async def test_delete_seller_target_returns_false_when_missing() -> None:
    assert await delete_seller_target(
        _Pool(execute_result="DELETE 0"),
        target_id=TARGET_ID,
    ) is False
