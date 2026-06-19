import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


for _heavy_mod in [
    "asyncpg",
    "asyncpg.exceptions",
    "mcp",
    "mcp.server",
]:
    sys.modules.setdefault(_heavy_mod, MagicMock())


class _MockFastMCP:
    def __init__(self, *args, **kwargs):
        self.settings = MagicMock()

    def tool(self):
        def _passthrough(fn):
            return fn
        return _passthrough

    def run(self, **kwargs):
        return None


_fastmcp_mod = MagicMock()
_fastmcp_mod.FastMCP = _MockFastMCP
sys.modules.setdefault("mcp.server.fastmcp", _fastmcp_mod)

import atlas_brain.mcp.b2b.scrape_targets as scrape_targets_mcp


def _mock_pool():
    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=[])
    pool.fetchrow = AsyncMock(return_value=None)
    return pool


@pytest.mark.asyncio
async def test_list_scrape_targets_normalizes_blank_optional_filters(monkeypatch):
    pool = _mock_pool()
    monkeypatch.setattr(scrape_targets_mcp, "get_pool", lambda: pool)

    body = json.loads(
        await scrape_targets_mcp.list_scrape_targets(source="   ", scrape_mode="\t", enabled_only=False)
    )

    pool.fetch.assert_awaited_once()
    query, limit = pool.fetch.await_args.args
    assert "source = $" not in query
    assert "scrape_mode = $" not in query
    assert limit == 20
    assert body == {"targets": [], "count": 0}


@pytest.mark.asyncio
async def test_list_scrape_targets_trims_active_filters(monkeypatch):
    pool = _mock_pool()
    monkeypatch.setattr(scrape_targets_mcp, "get_pool", lambda: pool)

    body = json.loads(
        await scrape_targets_mcp.list_scrape_targets(source="  G2  ", scrape_mode="  INCREMENTAL  ")
    )

    pool.fetch.assert_awaited_once()
    query, source, scrape_mode, limit = pool.fetch.await_args.args
    assert "source = $1" in query
    assert "scrape_mode = $2" in query
    assert source == "g2"
    assert scrape_mode == "incremental"
    assert limit == 20
    assert body == {"targets": [], "count": 0}


@pytest.mark.asyncio
async def test_add_scrape_target_trims_inputs_before_persistence(monkeypatch):
    pool = _mock_pool()
    pool.fetchrow = AsyncMock(
        side_effect=[
            None,
            {
                "id": uuid4(),
                "source": "g2",
                "vendor_name": "Zendesk",
                "product_slug": "zendesk-support",
                "enabled": True,
                "priority": 5,
                "scrape_mode": "incremental",
            },
        ]
    )
    monkeypatch.setattr(scrape_targets_mcp, "get_pool", lambda: pool)

    settings = MagicMock()
    settings.b2b_scrape.source_allowlist = ["g2"]
    monkeypatch.setattr("atlas_brain.config.settings", settings)
    monkeypatch.setattr(
        "atlas_brain.services.scraping.target_validation.is_source_allowed",
        lambda source, allowlist: True,
    )
    monkeypatch.setattr(
        "atlas_brain.services.scraping.target_validation.validate_target_input",
        lambda source, product_slug: (source, product_slug),
    )
    monkeypatch.setattr(
        "atlas_brain.services.vendor_registry.resolve_vendor_name",
        AsyncMock(return_value="Zendesk"),
    )

    body = json.loads(
        await scrape_targets_mcp.add_scrape_target(
            source="  G2  ",
            vendor_name="  Zendesk  ",
            product_slug="  zendesk-support  ",
            product_name="  Zendesk Support  ",
            product_category="  Helpdesk  ",
            priority=5,
            scrape_mode="  INCREMENTAL  ",
            metadata_json="   ",
        )
    )

    duplicate_query, source, product_slug, scrape_mode = pool.fetchrow.await_args_list[0].args
    assert source == "g2"
    assert product_slug == "zendesk-support"
    assert scrape_mode == "incremental"

    insert_query, insert_source, insert_vendor_name, insert_product_name, insert_product_slug, insert_product_category, _max_pages, _priority, _scrape_interval_hours, insert_scrape_mode, insert_metadata = pool.fetchrow.await_args_list[1].args
    assert insert_source == "g2"
    assert insert_vendor_name == "Zendesk"
    assert insert_product_name == "Zendesk Support"
    assert insert_product_slug == "zendesk-support"
    assert insert_product_category == "Helpdesk"
    assert insert_scrape_mode == "incremental"
    assert insert_metadata == "{}"
    assert body["success"] is True
    assert body["target"]["vendor_name"] == "Zendesk"
