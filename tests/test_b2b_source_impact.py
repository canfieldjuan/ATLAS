"""Tests for source impact ledger mapping across service, API, and MCP layers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
import types
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

import atlas_brain

from atlas_brain.services.b2b.source_impact import (
    _compute_coverage_ratio,
    build_source_impact_ledger,
    get_consumer_wiring_baseline,
    summarize_source_field_baseline,
)

_repo_root = Path(__file__).resolve().parents[1]
_api_pkg = types.ModuleType("atlas_brain.api")
_api_pkg.__path__ = [str(_repo_root / "atlas_brain" / "api")]
sys.modules.setdefault("atlas_brain.api", _api_pkg)

_dashboard_path = _repo_root / "atlas_brain" / "api" / "b2b_dashboard.py"
_dashboard_spec = importlib.util.spec_from_file_location(
    "atlas_brain.api.b2b_dashboard",
    _dashboard_path,
)
assert _dashboard_spec and _dashboard_spec.loader
b2b_dashboard = importlib.util.module_from_spec(_dashboard_spec)
sys.modules["atlas_brain.api.b2b_dashboard"] = b2b_dashboard
_dashboard_spec.loader.exec_module(b2b_dashboard)

from atlas_brain.mcp.b2b import pipeline as mcp_pipeline


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
    assert entry["operational_status"] == "infra_blocked"


def test_build_source_impact_ledger_exposes_quality_tier_alongside_source_family():
    ledger = build_source_impact_ledger(source="trustpilot")

    entry = ledger["sources"][0]
    assert entry["source_family"] == "community_signal"
    assert entry["scrape_data_quality"] == "verified"


def test_build_source_impact_ledger_marks_slashdot_as_deferred_inventory():
    ledger = build_source_impact_ledger(source="slashdot")

    assert ledger["summary"]["deferred_inventory_sources"] == ["slashdot"]
    entry = ledger["sources"][0]
    assert entry["source"] == "slashdot"
    assert entry["operational_status"] == "deferred_inventory"
    assert entry["expansion_stage"] == "deferred_conditional_inventory"


def test_consumer_wiring_baseline_flags_mixed_consumers():
    baseline = get_consumer_wiring_baseline()

    assert baseline["baseline_mode"] == "static_code_inventory"
    assert baseline["measured"] is False
    assert baseline["summary"]["total_consumers"] >= 5
    assert baseline["summary"]["mixed_consumers"] >= 1
    accounts = next(
        consumer
        for consumer in baseline["consumers"]
        if consumer["consumer"] == "b2b_accounts_in_motion"
    )
    assert accounts["legacy_fallback"] is True


def test_compute_coverage_ratio_keeps_three_decimal_precision():
    assert _compute_coverage_ratio(1, 3) == 0.333


@pytest.mark.asyncio
async def test_summarize_source_field_baseline_shapes_coverage():
    pool = _mock_pool(
        fetch_return=[
            {
                "source": "trustradius",
                "total_reviews": 10,
                "enriched_reviews": 8,
                "title_rows": 6,
                "enriched_title_rows": 6,
                "company_rows": 5,
                "enriched_company_rows": 5,
                "company_size_rows": 4,
                "enriched_company_size_rows": 4,
                "industry_rows": 3,
                "enriched_industry_rows": 3,
                "decision_maker_rows": 2,
                "enriched_decision_maker_rows": 2,
                "competitor_rows": 7,
                "enriched_competitor_rows": 7,
                "timing_rows": 1,
                "enriched_timing_rows": 1,
                "quote_rows": 5,
                "enriched_quote_rows": 5,
                "pain_rows": 8,
                "enriched_pain_rows": 8,
                "content_classification_rows": 8,
                "enriched_content_classification_rows": 8,
                "support_escalation_rows": 2,
                "enriched_support_escalation_rows": 2,
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
    assert row["enrichment_rate"] == 0.8
    assert row["coverage"]["title"] == 0.75
    assert row["coverage"]["competitors"] == 0.875
    assert row["coverage"]["content_classification"] == 1.0
    assert row["coverage"]["support_escalation"] == 0.25
    assert row["coverage_of_total_reviews"]["title"] == 0.6
    assert row["raw_counts"]["pain_rows"] == 8
    assert row["raw_counts"]["title_rows"] == 6
    assert row["raw_counts_of_total_reviews"]["title_rows"] == 6
    assert row["raw_counts"]["content_classification_rows"] == 8
    assert row["raw_counts"]["support_escalation_rows"] == 2


@pytest.mark.asyncio
async def test_summarize_source_field_baseline_scopes_coverage_to_enriched_rows():
    pool = _mock_pool(
        fetch_return=[
            {
                "source": "trustpilot",
                "total_reviews": 10,
                "enriched_reviews": 4,
                "title_rows": 6,
                "enriched_title_rows": 3,
                "company_rows": 5,
                "enriched_company_rows": 2,
                "company_size_rows": 4,
                "enriched_company_size_rows": 2,
                "industry_rows": 4,
                "enriched_industry_rows": 2,
                "decision_maker_rows": 2,
                "enriched_decision_maker_rows": 1,
                "competitor_rows": 5,
                "enriched_competitor_rows": 3,
                "timing_rows": 2,
                "enriched_timing_rows": 1,
                "quote_rows": 5,
                "enriched_quote_rows": 2,
                "pain_rows": 5,
                "enriched_pain_rows": 3,
                "content_classification_rows": 6,
                "enriched_content_classification_rows": 4,
                "support_escalation_rows": 2,
                "enriched_support_escalation_rows": 1,
            }
        ]
    )

    result = await summarize_source_field_baseline(pool, window_days=30, source="trustpilot")

    row = result["rows"][0]
    assert row["enrichment_rate"] == 0.4
    assert row["coverage"]["title"] == 0.75
    assert row["coverage"]["content_classification"] == 1.0
    assert row["coverage"]["competitors"] == 0.75
    assert row["coverage_of_total_reviews"]["title"] == 0.6
    assert row["coverage_of_total_reviews"]["content_classification"] == 0.6
    assert row["raw_counts"]["title_rows"] == 3
    assert row["raw_counts_of_total_reviews"]["title_rows"] == 6


@pytest.mark.asyncio
async def test_dashboard_source_health_normalizes_blank_source(monkeypatch):
    pool = _mock_pool(fetch_return=[])
    monkeypatch.setattr(b2b_dashboard, "_pool_or_503", lambda: pool)
    query_mock = MagicMock(return_value=("SELECT 1", []))
    monkeypatch.setattr(b2b_dashboard, "_build_source_health_query", query_mock)

    result = await b2b_dashboard.get_source_health(window_days=7, source="   ")

    query_mock.assert_called_once_with(None)
    pool.fetch.assert_awaited_once_with("SELECT 1", 7)
    assert result["summary"]["total_sources"] == 0


@pytest.mark.asyncio
async def test_dashboard_source_routes_reject_invalid_source_before_db_touch(monkeypatch):
    monkeypatch.setattr(b2b_dashboard, "_pool_or_503", lambda: (_ for _ in ()).throw(AssertionError("db should not be touched")))

    cases = [
        lambda: b2b_dashboard.get_source_health(window_days=7, source="invalid-source"),
        lambda: b2b_dashboard.get_source_telemetry(window_days=7, source="invalid-source", user=None),
        lambda: b2b_dashboard.get_telemetry_timeline(days=14, source="invalid-source", user=None),
        lambda: b2b_dashboard.export_source_health(window_days=7, source="invalid-source"),
    ]

    for call in cases:
        with pytest.raises(b2b_dashboard.HTTPException) as exc:
            await call()
        assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_dashboard_source_telemetry_normalizes_blank_source(monkeypatch):
    pool = _mock_pool(fetch_return=[])
    monkeypatch.setattr(b2b_dashboard, "_pool_or_503", lambda: pool)

    result = await b2b_dashboard.get_source_telemetry(window_days=7, source="   ", user=None)

    fetch_args = pool.fetch.await_args.args
    assert fetch_args[1:] == (7,)
    assert result["total_sources"] == 0


@pytest.mark.asyncio
async def test_dashboard_source_capabilities_normalizes_blank_source(monkeypatch):
    cap = MagicMock()
    cap.to_dict.return_value = {"mode": "read_only"}
    monkeypatch.setattr(b2b_dashboard, "get_capability", MagicMock(side_effect=AssertionError("unexpected single-source lookup")))

    with patch(
        "atlas_brain.services.scraping.capabilities.get_all_capabilities",
        return_value={"reddit": cap},
    ) as all_caps_mock:
        result = await b2b_dashboard.list_source_capabilities(source="   ", user=None)

    all_caps_mock.assert_called_once_with()
    assert result == {
        "sources": [{"source": "reddit", "capabilities": {"mode": "read_only"}}],
        "total": 1,
    }


@pytest.mark.asyncio
async def test_dashboard_source_impact_ledger_normalizes_blank_source(monkeypatch):
    monkeypatch.setattr(
        b2b_dashboard,
        "build_source_impact_ledger",
        MagicMock(return_value={"summary": {"total_sources": 0}, "sources": []}),
    )

    result = await b2b_dashboard.get_source_impact_ledger(
        source="   ",
        window_days=45,
        include_field_baseline=False,
        include_consumer_wiring=False,
        user=MagicMock(),
    )

    b2b_dashboard.build_source_impact_ledger.assert_called_once_with(source=None)
    assert result["source_filter"] is None
    assert result["impact_summary"]["total_sources"] == 0


@pytest.mark.asyncio
async def test_dashboard_telemetry_timeline_normalizes_blank_source(monkeypatch):
    pool = _mock_pool(fetch_return=[])
    monkeypatch.setattr(b2b_dashboard, "_pool_or_503", lambda: pool)

    result = await b2b_dashboard.get_telemetry_timeline(days=14, source="   ", user=None)

    fetch_args = pool.fetch.await_args.args
    assert fetch_args[1:] == (14,)
    assert result["timeline"] == []


@pytest.mark.asyncio
async def test_dashboard_export_source_health_normalizes_blank_source(monkeypatch):
    pool = _mock_pool(fetch_return=[])
    monkeypatch.setattr(b2b_dashboard, "_pool_or_503", lambda: pool)
    query_mock = MagicMock(return_value=("SELECT 1", []))
    monkeypatch.setattr(b2b_dashboard, "_build_source_health_query", query_mock)

    response = await b2b_dashboard.export_source_health(window_days=7, source="   ")

    query_mock.assert_called_once_with(None)
    pool.fetch.assert_awaited_once_with("SELECT 1", 7)
    assert response.media_type == "text/csv"


@pytest.mark.asyncio
async def test_dashboard_source_impact_ledger_includes_field_baseline(monkeypatch):
    pool = _mock_pool(
        fetch_return=[
            {
                "source": "reddit",
                "total_reviews": 20,
                "enriched_reviews": 15,
                "title_rows": 0,
                "enriched_title_rows": 0,
                "company_rows": 0,
                "enriched_company_rows": 0,
                "company_size_rows": 0,
                "enriched_company_size_rows": 0,
                "industry_rows": 1,
                "enriched_industry_rows": 1,
                "decision_maker_rows": 0,
                "enriched_decision_maker_rows": 0,
                "competitor_rows": 9,
                "enriched_competitor_rows": 9,
                "timing_rows": 4,
                "enriched_timing_rows": 4,
                "quote_rows": 7,
                "enriched_quote_rows": 7,
                "pain_rows": 12,
                "enriched_pain_rows": 12,
                "content_classification_rows": 15,
                "enriched_content_classification_rows": 15,
                "support_escalation_rows": 3,
                "enriched_support_escalation_rows": 3,
            }
        ]
    )
    monkeypatch.setattr(b2b_dashboard, "get_db_pool", lambda: pool)

    result = await b2b_dashboard.get_source_impact_ledger(
        source="reddit",
        window_days=45,
    )

    assert result["impact_summary"]["total_sources"] == 1
    assert result["sources"][0]["source"] == "reddit"
    assert result["field_baseline"]["rows"][0]["coverage"]["competitors"] == 0.6
    assert (
        result["field_baseline"]["rows"][0]["coverage_of_total_reviews"]["competitors"]
        == 0.45
    )
    assert result["consumer_wiring"]["summary"]["mixed_consumers"] >= 1


@pytest.mark.asyncio
async def test_mcp_source_impact_ledger_returns_json_payload():
    pool = _mock_pool(
        fetch_return=[
            {
                "source": "getapp",
                "total_reviews": 5,
                "enriched_reviews": 5,
                "title_rows": 4,
                "enriched_title_rows": 4,
                "company_rows": 4,
                "enriched_company_rows": 4,
                "company_size_rows": 3,
                "enriched_company_size_rows": 3,
                "industry_rows": 2,
                "enriched_industry_rows": 2,
                "decision_maker_rows": 2,
                "enriched_decision_maker_rows": 2,
                "competitor_rows": 1,
                "enriched_competitor_rows": 1,
                "timing_rows": 0,
                "enriched_timing_rows": 0,
                "quote_rows": 4,
                "enriched_quote_rows": 4,
                "pain_rows": 5,
                "enriched_pain_rows": 5,
                "content_classification_rows": 5,
                "enriched_content_classification_rows": 5,
                "support_escalation_rows": 1,
                "enriched_support_escalation_rows": 1,
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
    assert data["field_baseline"]["rows"][0]["coverage"]["support_escalation"] == 0.2
