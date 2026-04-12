from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from atlas_brain.api import b2b_evidence as mod


@pytest.mark.asyncio
async def test_list_witnesses_normalizes_query_defaults_on_direct_call(monkeypatch):
    captured = {}

    async def _latest_snapshot(pool, vendor_name, window_days, target_date):
        captured.update(vendor_name=vendor_name, window_days=window_days, target_date=target_date)
        return None

    pool = SimpleNamespace(is_initialized=True)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "resolve_vendor_name", AsyncMock(return_value=None))
    monkeypatch.setattr(mod, "_latest_witness_snapshot_date", _latest_snapshot)

    result = await mod.list_witnesses("Salesforce", user=SimpleNamespace(account_id="not-a-uuid"))

    assert captured["vendor_name"] == "Salesforce"
    assert captured["window_days"] == mod._default_analysis_window_days()
    assert result["analysis_window_days"] == mod._default_analysis_window_days()
    assert result["limit"] == mod.DEFAULT_WITNESS_LIMIT
    assert result["offset"] == 0
    assert result["as_of_date"] is None


@pytest.mark.asyncio
async def test_get_witness_normalizes_query_default_window_days(monkeypatch):
    captured = {}

    async def _latest_snapshot(pool, vendor_name, window_days, target_date):
        captured["window_days"] = window_days
        return date(2026, 4, 1)

    pool = SimpleNamespace(is_initialized=True, fetchrow=AsyncMock(return_value=None))
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "resolve_vendor_name", AsyncMock(return_value=None))
    monkeypatch.setattr(mod, "_latest_witness_snapshot_date", _latest_snapshot)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.get_witness("w1", "Salesforce", user=SimpleNamespace(account_id=str(uuid4())))

    assert exc.value.status_code == 404
    assert captured["window_days"] == mod._default_analysis_window_days()
    assert pool.fetchrow.await_args.args[3] == mod._default_analysis_window_days()
    assert pool.fetchrow.await_args.args[4] == date(2026, 4, 1)


@pytest.mark.asyncio
async def test_get_vault_normalizes_query_default_window_days(monkeypatch):
    captured = {}

    async def _read_record(pool, vendor_name, *, as_of, analysis_window_days):
        captured["analysis_window_days"] = analysis_window_days
        return None

    monkeypatch.setattr(mod, "get_db_pool", lambda: SimpleNamespace(is_initialized=True))
    monkeypatch.setattr(mod, "resolve_vendor_name", AsyncMock(return_value=None))
    monkeypatch.setattr(mod, "_read_vendor_intelligence_record", _read_record)

    result = await mod.get_vault("Salesforce", user=SimpleNamespace(account_id=str(uuid4())))

    assert captured["analysis_window_days"] == mod._default_analysis_window_days()
    assert result == {
        "vendor_name": "Salesforce",
        "vault": None,
        "message": "No evidence vault found for this vendor",
    }


@pytest.mark.asyncio
async def test_list_annotations_normalizes_query_defaults_on_direct_call(monkeypatch):
    account_id = str(uuid4())
    pool = SimpleNamespace(is_initialized=True, fetch=AsyncMock(return_value=[]))
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "resolve_vendor_name", AsyncMock(side_effect=AssertionError("resolve touched")))

    result = await mod.list_annotations(user=SimpleNamespace(account_id=account_id))

    assert pool.fetch.await_args.args[1:] == (UUID(account_id),)
    assert result == {"annotations": [], "count": 0}
