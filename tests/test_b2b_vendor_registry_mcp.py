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

import atlas_brain.mcp.b2b.vendor_registry as vendor_registry_mcp


def _install_vendor_registry_service(monkeypatch, **kwargs):
    service = types.ModuleType("atlas_brain.services.vendor_registry")
    for name, value in kwargs.items():
        setattr(service, name, value)
    monkeypatch.setitem(sys.modules, "atlas_brain.services.vendor_registry", service)


@pytest.mark.asyncio
async def test_fuzzy_vendor_search_trims_query(monkeypatch):
    search_mock = AsyncMock(return_value=[{"canonical_name": "Zendesk"}])
    _install_vendor_registry_service(monkeypatch, fuzzy_search_vendors=search_mock)

    body = json.loads(await vendor_registry_mcp.fuzzy_vendor_search("  zend  "))

    search_mock.assert_awaited_once_with("zend", limit=10, min_similarity=0.3)
    assert body["query"] == "zend"
    assert body["count"] == 1


@pytest.mark.asyncio
async def test_fuzzy_company_search_trims_filters(monkeypatch):
    search_mock = AsyncMock(return_value=[{"company_name": "Acme"}])
    _install_vendor_registry_service(monkeypatch, fuzzy_search_companies=search_mock)

    body = json.loads(
        await vendor_registry_mcp.fuzzy_company_search("  acme  ", vendor_name="  zendesk  ")
    )

    search_mock.assert_awaited_once_with(
        "acme",
        vendor_name="zendesk",
        limit=10,
        min_similarity=0.3,
    )
    assert body["query"] == "acme"
    assert body["vendor_filter"] == "zendesk"


@pytest.mark.asyncio
async def test_fuzzy_company_search_normalizes_blank_vendor_filter(monkeypatch):
    search_mock = AsyncMock(return_value=[])
    _install_vendor_registry_service(monkeypatch, fuzzy_search_companies=search_mock)

    body = json.loads(
        await vendor_registry_mcp.fuzzy_company_search("  acme  ", vendor_name="   ")
    )

    search_mock.assert_awaited_once_with(
        "acme",
        vendor_name=None,
        limit=10,
        min_similarity=0.3,
    )
    assert body["vendor_filter"] is None


@pytest.mark.asyncio
async def test_add_vendor_to_registry_trims_canonical_name_and_aliases(monkeypatch):
    add_mock = AsyncMock(
        return_value={
            "id": 1,
            "canonical_name": "Salesforce",
            "aliases": ["sf", "sfdc"],
            "created_at": "2026-04-12T00:00:00Z",
            "updated_at": "2026-04-12T00:00:00Z",
        }
    )
    _install_vendor_registry_service(monkeypatch, add_vendor=add_mock)

    body = json.loads(
        await vendor_registry_mcp.add_vendor_to_registry(
            "  Salesforce  ",
            aliases=" sf ,   , sfdc ",
        )
    )

    add_mock.assert_awaited_once_with("Salesforce", ["sf", "sfdc"])
    assert body["success"] is True
    assert body["vendor"]["canonical_name"] == "Salesforce"


@pytest.mark.asyncio
async def test_add_vendor_alias_trims_values_before_service_call(monkeypatch):
    add_alias_mock = AsyncMock(
        return_value={
            "id": 1,
            "canonical_name": "Salesforce",
            "aliases": ["salesforce.com"],
            "updated_at": "2026-04-12T00:00:00Z",
        }
    )
    _install_vendor_registry_service(monkeypatch, add_alias=add_alias_mock)

    body = json.loads(
        await vendor_registry_mcp.add_vendor_alias("  Salesforce  ", "  salesforce.com  ")
    )

    add_alias_mock.assert_awaited_once_with("Salesforce", "salesforce.com")
    assert body["success"] is True
    assert body["vendor"]["canonical_name"] == "Salesforce"


@pytest.mark.asyncio
async def test_add_vendor_alias_not_found_uses_trimmed_name(monkeypatch):
    add_alias_mock = AsyncMock(return_value=None)
    _install_vendor_registry_service(monkeypatch, add_alias=add_alias_mock)

    body = json.loads(
        await vendor_registry_mcp.add_vendor_alias("  Salesforce  ", "  salesforce.com  ")
    )

    add_alias_mock.assert_awaited_once_with("Salesforce", "salesforce.com")
    assert body == {
        "success": False,
        "error": "Vendor 'Salesforce' not found in registry",
    }
