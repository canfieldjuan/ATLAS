import json
import sys
import types
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

import atlas_brain.mcp.b2b.products as products_mcp


def _install_product_matching_service(monkeypatch, **kwargs):
    service = types.ModuleType("atlas_brain.services.b2b.product_matching")
    for name, value in kwargs.items():
        setattr(service, name, value)
    monkeypatch.setitem(sys.modules, "atlas_brain.services.b2b.product_matching", service)


@pytest.mark.asyncio
async def test_get_product_profile_trims_vendor_name_before_query(monkeypatch):
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        return_value={
            "id": 1,
            "vendor_name": "Zendesk",
            "product_category": "helpdesk",
            "strengths": "[]",
            "weaknesses": "[]",
            "pain_addressed": "[]",
            "total_reviews_analyzed": 12,
            "avg_rating": 4.2,
            "recommend_rate": 0.8,
            "avg_urgency": 6.5,
            "primary_use_cases": "[]",
            "typical_company_size": "[]",
            "typical_industries": "[]",
            "top_integrations": "[]",
            "commonly_compared_to": "[]",
            "commonly_switched_from": "[]",
            "profile_summary": "summary",
            "confidence_score": 0.9,
            "last_computed_at": None,
            "created_at": None,
        }
    )
    monkeypatch.setattr(products_mcp, "get_pool", lambda: pool)

    body = json.loads(await products_mcp.get_product_profile("  zen  "))

    pool.fetchrow.assert_awaited_once()
    _query, vendor_name = pool.fetchrow.await_args.args
    assert vendor_name == "zen"
    assert body["success"] is True


@pytest.mark.asyncio
async def test_get_product_profile_history_trims_vendor_name_and_message(monkeypatch):
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(return_value=[])
    monkeypatch.setattr(products_mcp, "get_pool", lambda: pool)

    body = json.loads(await products_mcp.get_product_profile_history("  zen  ", days=30, limit=5))

    pool.fetch.assert_awaited_once()
    _query, vendor_name, days, limit = pool.fetch.await_args.args
    assert vendor_name == "zen"
    assert days == 30
    assert limit == 5
    assert body["vendor_name"] == "zen"
    assert body["message"] == "No product profile snapshots found for 'zen'"


@pytest.mark.asyncio
async def test_match_products_tool_trims_inputs_before_service_call(monkeypatch):
    pool = MagicMock()
    monkeypatch.setattr(products_mcp, "get_pool", lambda: pool)

    match_mock = AsyncMock(return_value=[{"vendor_name": "Freshdesk"}])
    _install_product_matching_service(monkeypatch, match_products=match_mock)

    body = json.loads(
        await products_mcp.match_products_tool(
            "  Zendesk  ",
            pain_categories='[{"category": "pricing", "severity": "primary"}]',
            company_size=250,
            industry="  saas  ",
            limit=3,
        )
    )

    match_mock.assert_awaited_once_with(
        churning_from="Zendesk",
        pain_categories=[{"category": "pricing", "severity": "primary"}],
        company_size=250,
        industry="saas",
        pool=pool,
        limit=3,
    )
    assert body["success"] is True
    assert body["count"] == 1


@pytest.mark.asyncio
async def test_match_products_tool_ignores_blank_optional_filters(monkeypatch):
    pool = MagicMock()
    monkeypatch.setattr(products_mcp, "get_pool", lambda: pool)

    match_mock = AsyncMock(return_value=[])
    _install_product_matching_service(monkeypatch, match_products=match_mock)

    body = json.loads(
        await products_mcp.match_products_tool(
            "  Zendesk  ",
            pain_categories="   ",
            industry="   ",
            limit=2,
        )
    )

    match_mock.assert_awaited_once_with(
        churning_from="Zendesk",
        pain_categories=[],
        company_size=None,
        industry=None,
        pool=pool,
        limit=2,
    )
    assert body["success"] is True
    assert body["count"] == 0
