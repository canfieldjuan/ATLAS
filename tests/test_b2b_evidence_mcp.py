import json
import sys
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest


class _MockFastMCP:
    def __init__(self, *args, **kwargs):
        self.settings = MagicMock()

    def tool(self):
        def _passthrough(fn):
            return fn
        return _passthrough

    def run(self, **kwargs):
        return None


sys.modules.setdefault("asyncpg", MagicMock())
sys.modules.setdefault("asyncpg.exceptions", MagicMock())
sys.modules.setdefault("mcp", MagicMock())
sys.modules.setdefault("mcp.server", MagicMock())
_fastmcp_mod = MagicMock()
_fastmcp_mod.FastMCP = _MockFastMCP
sys.modules.setdefault("mcp.server.fastmcp", _fastmcp_mod)

import atlas_brain.mcp.b2b.evidence as evidence_mcp


@pytest.mark.asyncio
async def test_get_evidence_vault_uses_shared_vendor_intelligence_reader(monkeypatch):
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetchrow = AsyncMock(side_effect=AssertionError("direct fetchrow should not run"))
    monkeypatch.setattr(evidence_mcp, "get_pool", lambda: pool)

    reader = AsyncMock(
        return_value={
            "vendor_name": "Zendesk",
            "as_of_date": date(2026, 3, 31),
            "analysis_window_days": 30,
            "schema_version": 2,
            "created_at": datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc),
            "vault": {
                "weakness_evidence": [{"category": "pricing"}],
                "strength_evidence": [{"category": "support"}],
                "company_signals": [{"company_name": "Acme"}],
                "metric_snapshot": {"avg_urgency": 7.2},
                "provenance": {"sources": ["g2"]},
            },
        },
    )
    monkeypatch.setattr(evidence_mcp, "_search_vendor_intelligence_record", reader)

    body = json.loads(await evidence_mcp.get_evidence_vault("  zen  "))

    reader.assert_awaited_once_with(
        pool,
        vendor_query="zen",
        as_of=date.today(),
        analysis_window_days=30,
    )
    assert body["success"] is True
    assert body["vendor_name"] == "Zendesk"
    assert body["metric_snapshot"]["avg_urgency"] == 7.2


@pytest.mark.asyncio
async def test_list_evidence_vaults_sorts_before_limit(monkeypatch):
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(side_effect=AssertionError("direct fetch should not run"))
    monkeypatch.setattr(evidence_mcp, "get_pool", lambda: pool)

    reader = AsyncMock(
        return_value=[
            {
                "vendor_name": "Alpha",
                "as_of_date": date(2026, 3, 31),
                "analysis_window_days": 30,
                "schema_version": 2,
                "created_at": datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc),
                "vault": {"metric_snapshot": {"avg_urgency": 2.1}},
            },
            {
                "vendor_name": "Zendesk",
                "as_of_date": date(2026, 3, 31),
                "analysis_window_days": 30,
                "schema_version": 2,
                "created_at": datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc),
                "vault": {"metric_snapshot": {"avg_urgency": 7.2}},
            },
        ],
    )
    monkeypatch.setattr(evidence_mcp, "_search_vendor_intelligence_records", reader)

    body = json.loads(await evidence_mcp.list_evidence_vaults(limit=1))

    reader.assert_awaited_once_with(
        pool,
        as_of=date.today(),
        analysis_window_days=30,
        vendor_query=None,
    )
    assert body["count"] == 1
    assert body["vaults"][0]["vendor_name"] == "Zendesk"
    assert body["vaults"][0]["avg_urgency"] == 7.2


@pytest.mark.asyncio
async def test_list_evidence_vaults_normalizes_blank_vendor_filter(monkeypatch):
    pool = MagicMock()
    pool.is_initialized = True
    monkeypatch.setattr(evidence_mcp, "get_pool", lambda: pool)

    reader = AsyncMock(return_value=[])
    monkeypatch.setattr(evidence_mcp, "_search_vendor_intelligence_records", reader)

    body = json.loads(await evidence_mcp.list_evidence_vaults(vendor_name="   "))

    reader.assert_awaited_once_with(
        pool,
        as_of=date.today(),
        analysis_window_days=30,
        vendor_query=None,
    )
    assert body["count"] == 0


@pytest.mark.asyncio
async def test_list_segment_intelligence_normalizes_blank_vendor_filter(monkeypatch):
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(return_value=[])
    monkeypatch.setattr(evidence_mcp, "get_pool", lambda: pool)

    body = json.loads(await evidence_mcp.list_segment_intelligence(vendor_name="   "))

    pool.fetch.assert_awaited_once()
    query, *params = pool.fetch.await_args.args
    assert "vendor_name ILIKE" not in query
    assert params == [date.today(), 30]
    assert body["count"] == 0


@pytest.mark.asyncio
async def test_list_displacement_dynamics_normalizes_blank_vendor_filters(monkeypatch):
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(return_value=[])
    monkeypatch.setattr(evidence_mcp, "get_pool", lambda: pool)

    body = json.loads(
        await evidence_mcp.list_displacement_dynamics(from_vendor="   ", to_vendor="  ")
    )

    pool.fetch.assert_awaited_once()
    query, *params = pool.fetch.await_args.args
    assert "from_vendor ILIKE" not in query
    assert "to_vendor ILIKE" not in query
    assert params == [date.today(), 30]
    assert body["count"] == 0


@pytest.mark.asyncio
async def test_list_category_dynamics_normalizes_blank_category_filter(monkeypatch):
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(return_value=[])
    monkeypatch.setattr(evidence_mcp, "get_pool", lambda: pool)

    body = json.loads(await evidence_mcp.list_category_dynamics(category="   "))

    pool.fetch.assert_awaited_once()
    query, *params = pool.fetch.await_args.args
    assert "category ILIKE" not in query
    assert params == [date.today(), 30]
    assert body["count"] == 0


@pytest.mark.asyncio
async def test_list_account_intelligence_normalizes_blank_vendor_filter(monkeypatch):
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(return_value=[])
    monkeypatch.setattr(evidence_mcp, "get_pool", lambda: pool)

    body = json.loads(await evidence_mcp.list_account_intelligence(vendor_name="   "))

    pool.fetch.assert_awaited_once()
    query, *params = pool.fetch.await_args.args
    assert "vendor_name ILIKE" not in query
    assert params == [date.today(), 30]
    assert body["count"] == 0


@pytest.mark.asyncio
async def test_get_segment_intelligence_trims_vendor_name_before_query(monkeypatch):
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetchrow = AsyncMock(
        return_value={
            "vendor_name": "Zendesk",
            "as_of_date": date(2026, 3, 31),
            "analysis_window_days": 30,
            "schema_version": 2,
            "segments": {},
            "created_at": datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc),
        }
    )
    monkeypatch.setattr(evidence_mcp, "get_pool", lambda: pool)

    body = json.loads(await evidence_mcp.get_segment_intelligence("  zen  "))

    pool.fetchrow.assert_awaited_once()
    _query, vendor_query, as_of_date, window_days = pool.fetchrow.await_args.args
    assert vendor_query == "zen"
    assert as_of_date == date.today()
    assert window_days == 30
    assert body["success"] is True


@pytest.mark.asyncio
async def test_get_temporal_intelligence_trims_vendor_name_before_query(monkeypatch):
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetchrow = AsyncMock(
        return_value={
            "vendor_name": "Zendesk",
            "as_of_date": date(2026, 3, 31),
            "analysis_window_days": 30,
            "schema_version": 2,
            "temporal": {},
            "created_at": datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc),
        }
    )
    monkeypatch.setattr(evidence_mcp, "get_pool", lambda: pool)

    body = json.loads(await evidence_mcp.get_temporal_intelligence("  zen  "))

    pool.fetchrow.assert_awaited_once()
    _query, vendor_query, as_of_date, window_days = pool.fetchrow.await_args.args
    assert vendor_query == "zen"
    assert as_of_date == date.today()
    assert window_days == 30
    assert body["success"] is True


@pytest.mark.asyncio
async def test_get_displacement_dynamics_trims_vendor_names_before_query(monkeypatch):
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetchrow = AsyncMock(
        return_value={
            "from_vendor": "Zendesk",
            "to_vendor": "Freshdesk",
            "as_of_date": date(2026, 3, 31),
            "analysis_window_days": 30,
            "schema_version": 2,
            "dynamics": {},
            "created_at": datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc),
        }
    )
    monkeypatch.setattr(evidence_mcp, "get_pool", lambda: pool)

    body = json.loads(await evidence_mcp.get_displacement_dynamics("  zen  ", "  fresh  "))

    pool.fetchrow.assert_awaited_once()
    _query, from_vendor, to_vendor, as_of_date, window_days = pool.fetchrow.await_args.args
    assert from_vendor == "zen"
    assert to_vendor == "fresh"
    assert as_of_date == date.today()
    assert window_days == 30
    assert body["success"] is True


@pytest.mark.asyncio
async def test_get_category_dynamics_trims_category_before_query(monkeypatch):
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetchrow = AsyncMock(
        return_value={
            "category": "CRM",
            "as_of_date": date(2026, 3, 31),
            "analysis_window_days": 30,
            "schema_version": 2,
            "dynamics": {},
            "created_at": datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc),
        }
    )
    monkeypatch.setattr(evidence_mcp, "get_pool", lambda: pool)

    body = json.loads(await evidence_mcp.get_category_dynamics("  crm  "))

    pool.fetchrow.assert_awaited_once()
    _query, category, as_of_date, window_days = pool.fetchrow.await_args.args
    assert category == "crm"
    assert as_of_date == date.today()
    assert window_days == 30
    assert body["success"] is True


@pytest.mark.asyncio
async def test_get_account_intelligence_trims_vendor_name_before_query(monkeypatch):
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetchrow = AsyncMock(
        return_value={
            "vendor_name": "Zendesk",
            "as_of_date": date(2026, 3, 31),
            "analysis_window_days": 30,
            "schema_version": 2,
            "accounts": {},
            "created_at": datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc),
        }
    )
    monkeypatch.setattr(evidence_mcp, "get_pool", lambda: pool)

    body = json.loads(await evidence_mcp.get_account_intelligence("  zen  "))

    pool.fetchrow.assert_awaited_once()
    _query, vendor_query, as_of_date, window_days = pool.fetchrow.await_args.args
    assert vendor_query == "zen"
    assert as_of_date == date.today()
    assert window_days == 30
    assert body["success"] is True
