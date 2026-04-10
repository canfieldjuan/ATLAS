"""Tests for source impact ledger mapping across service, API, and MCP layers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type(
    "UndefinedTableError",
    (Exception,),
    {},
)
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)

for _mod in (
    "torch",
    "torchaudio",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "PIL",
    "PIL.Image",
    "numpy",
    "cv2",
    "sounddevice",
    "soundfile",
    "httpx",
    "dateparser",
    "playwright",
    "playwright.async_api",
    "playwright_stealth",
    "curl_cffi",
    "curl_cffi.requests",
    "pytrends",
    "pytrends.request",
):
    sys.modules.setdefault(_mod, MagicMock())


class _MockFastMCP:
    def __init__(self, *args, **kwargs):
        self.settings = MagicMock()

    def tool(self):
        def _passthrough(fn):
            return fn

        return _passthrough

    def run(self, **kwargs):
        return None


_mcp_mod = MagicMock()
_mcp_server_mod = MagicMock()
_fastmcp_mod = MagicMock()
_fastmcp_mod.FastMCP = _MockFastMCP
sys.modules.setdefault("mcp", _mcp_mod)
sys.modules.setdefault("mcp.server", _mcp_server_mod)
sys.modules.setdefault("mcp.server.fastmcp", _fastmcp_mod)

from atlas_brain.mcp.b2b import pipeline as mcp_pipeline
from atlas_brain.services.b2b.source_impact import (
    build_source_impact_ledger,
    get_consumer_wiring_baseline,
    summarize_source_field_baseline,
)


def _mock_pool(fetch_return=None):
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(return_value=fetch_return or [])
    return pool


def test_build_source_impact_ledger_highlights_getapp_recovery():
    ledger = build_source_impact_ledger(source="getapp")

    assert ledger["summary"]["total_sources"] == 1
    entry = ledger["sources"][0]
    assert entry["source"] == "getapp"
    assert entry["expansion_stage"] == "recover_zero_row_core_source"
    assert "scrape_coverage" in entry["work_type"]
    assert "accounts" in entry["target_pools"]
    assert "watchlists_accounts_in_motion" in entry["expected_consumers"]


def test_consumer_wiring_baseline_flags_mixed_consumers():
    baseline = get_consumer_wiring_baseline()

    assert baseline["summary"]["total_consumers"] >= 5
    assert baseline["summary"]["mixed_consumers"] >= 1
    accounts = next(
        consumer
        for consumer in baseline["consumers"]
        if consumer["consumer"] == "b2b_accounts_in_motion"
    )
    assert accounts["legacy_fallback"] is True


@pytest.mark.asyncio
async def test_summarize_source_field_baseline_shapes_coverage():
    pool = _mock_pool(
        fetch_return=[
            {
                "source": "trustradius",
                "total_reviews": 10,
                "enriched_reviews": 8,
                "title_rows": 6,
                "company_rows": 5,
                "company_size_rows": 4,
                "industry_rows": 3,
                "decision_maker_rows": 2,
                "competitor_rows": 7,
                "timing_rows": 1,
                "quote_rows": 5,
                "pain_rows": 8,
            }
        ]
    )

    result = await summarize_source_field_baseline(
        pool,
        window_days=30,
        source="trustradius",
    )

    assert result["summary"]["total_sources"] == 1
    row = result["rows"][0]
    assert row["source"] == "trustradius"
    assert row["coverage"]["title"] == 0.6
    assert row["coverage"]["competitors"] == 0.7
    assert row["raw_counts"]["pain_rows"] == 8


def test_dashboard_source_contains_source_impact_route():
    source = (
        Path(__file__).resolve().parents[1] / "atlas_brain" / "api" / "b2b_dashboard.py"
    ).read_text()

    assert '@router.get("/source-impact-ledger")' in source
    assert "build_source_impact_ledger" in source
    assert "summarize_source_field_baseline" in source


@pytest.mark.asyncio
async def test_mcp_source_impact_ledger_returns_json_payload():
    pool = _mock_pool(
        fetch_return=[
            {
                "source": "getapp",
                "total_reviews": 5,
                "enriched_reviews": 5,
                "title_rows": 4,
                "company_rows": 4,
                "company_size_rows": 3,
                "industry_rows": 2,
                "decision_maker_rows": 2,
                "competitor_rows": 1,
                "timing_rows": 0,
                "quote_rows": 4,
                "pain_rows": 5,
            }
        ]
    )

    with patch.object(mcp_pipeline, "get_pool", return_value=pool):
        raw = await mcp_pipeline.get_source_impact_ledger(source="getapp")

    data = json.loads(raw)
    assert data["success"] is True
    assert data["impact_summary"]["total_sources"] == 1
    assert data["sources"][0]["source"] == "getapp"
    assert data["field_baseline"]["rows"][0]["coverage"]["title"] == 0.8
