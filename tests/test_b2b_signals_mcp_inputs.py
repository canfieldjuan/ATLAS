import json
import sys
import types
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


for _heavy_mod in [
    "PIL", "PIL.Image",
    "torch",
    "transformers",
    "numpy",
    "sentence_transformers",
    "asyncpg",
    "asyncpg.exceptions",
    "llama_cpp",
    "dateparser",
    "sse_starlette", "sse_starlette.sse",
    "uvicorn", "anyio",
    "httpx", "httpx_sse",
    "pydantic_settings",
    "curl_cffi", "curl_cffi.requests",
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


_mcp_mod = MagicMock()
_mcp_server_mod = MagicMock()
_fastmcp_mod = MagicMock()
_fastmcp_mod.FastMCP = _MockFastMCP
sys.modules.setdefault("mcp", _mcp_mod)
sys.modules.setdefault("mcp.server", _mcp_server_mod)
sys.modules.setdefault("mcp.server.fastmcp", _fastmcp_mod)

import atlas_brain.mcp.b2b.signals as signals_mcp


def _mock_pool():
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(return_value=[])
    pool.fetchrow = AsyncMock(return_value=None)
    return pool


def _make_signal_row(**overrides):
    row = {
        "id": str(uuid4()),
        "vendor_name": "Zendesk",
        "product_category": "Customer Support",
        "total_reviews": 120,
        "negative_reviews": 45,
        "churn_intent_count": 18,
        "avg_urgency_score": Decimal("7.2"),
        "avg_rating_normalized": Decimal("0.58"),
        "nps_proxy": Decimal("-15.3"),
        "price_complaint_rate": Decimal("0.2100"),
        "decision_maker_churn_rate": Decimal("0.1400"),
        "top_pain_categories": [],
        "top_competitors": [],
        "top_feature_gaps": [],
        "company_churn_list": [],
        "quotable_evidence": [],
        "top_use_cases": [],
        "top_integration_stacks": [],
        "budget_signal_summary": {},
        "sentiment_distribution": {},
        "buyer_authority_summary": {},
        "timeline_summary": {},
        "source_distribution": {},
        "sample_review_ids": [],
        "review_window_start": None,
        "review_window_end": None,
        "confidence_score": Decimal("0.85"),
        "last_computed_at": datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc),
        "created_at": datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc),
        "insider_signal_count": 0,
        "insider_org_health_summary": None,
        "insider_talent_drain_rate": None,
        "insider_quotable_evidence": [],
        "keyword_spike_count": 0,
        "keyword_spike_keywords": [],
        "keyword_trend_summary": None,
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_list_churn_signals_normalizes_blank_optional_filters(monkeypatch):
    pool = _mock_pool()
    monkeypatch.setattr(signals_mcp, "get_pool", lambda: pool)
    read_mock = AsyncMock(return_value=[])

    with patch("atlas_brain.autonomous.tasks._b2b_shared.read_vendor_signal_rows", new=read_mock), \
         patch.object(signals_mcp, "_load_reasoning_views_for_vendors", new=AsyncMock(return_value={})):
        body = json.loads(await signals_mcp.list_churn_signals(vendor_name="   ", category="\t"))

    read_mock.assert_awaited_once_with(
        pool,
        vendor_name_query=None,
        min_urgency=0.0,
        product_category=None,
        exclude_suppressed=True,
        limit=20,
    )
    assert body == {"signals": [], "count": 0}


@pytest.mark.asyncio
async def test_get_churn_signal_trims_vendor_name_and_blank_category(monkeypatch):
    pool = _mock_pool()
    monkeypatch.setattr(signals_mcp, "get_pool", lambda: pool)
    read_mock = AsyncMock(return_value=_make_signal_row())

    with patch("atlas_brain.autonomous.tasks._b2b_shared.read_vendor_signal_detail", new=read_mock), \
         patch.object(signals_mcp, "_load_reasoning_views_for_vendors", new=AsyncMock(return_value={})):
        body = json.loads(await signals_mcp.get_churn_signal("  Zendesk  ", product_category="   "))

    read_mock.assert_awaited_once_with(
        pool,
        vendor_name_query="Zendesk",
        product_category=None,
        exclude_suppressed=True,
    )
    assert body["success"] is True
    assert body["signal"]["vendor_name"] == "Zendesk"


@pytest.mark.asyncio
async def test_list_high_intent_companies_normalizes_blank_vendor_filter(monkeypatch):
    pool = _mock_pool()
    monkeypatch.setattr(signals_mcp, "get_pool", lambda: pool)
    read_mock = AsyncMock(return_value=[])

    with patch("atlas_brain.autonomous.tasks._b2b_shared.read_high_intent_companies", new=read_mock):
        body = json.loads(await signals_mcp.list_high_intent_companies(vendor_name="   "))

    read_mock.assert_awaited_once_with(
        pool,
        min_urgency=7.0,
        window_days=30,
        vendor_name=None,
        limit=20,
    )
    assert body == {"companies": [], "count": 0}


@pytest.mark.asyncio
async def test_get_vendor_profile_trims_vendor_name_before_queries(monkeypatch):
    pool = _mock_pool()
    pool.fetchrow = AsyncMock(return_value={"total_reviews": 5, "pending_enrichment": 1, "enriched": 4})
    pool.fetch = AsyncMock(return_value=[])
    monkeypatch.setattr(signals_mcp, "get_pool", lambda: pool)
    read_signal_mock = AsyncMock(return_value=None)
    read_high_intent_mock = AsyncMock(return_value=[])

    with patch("atlas_brain.autonomous.tasks._b2b_shared.read_vendor_signal_detail", new=read_signal_mock), \
         patch("atlas_brain.autonomous.tasks._b2b_shared.read_high_intent_companies", new=read_high_intent_mock):
        body = json.loads(await signals_mcp.get_vendor_profile("  Zendesk  "))

    read_signal_mock.assert_awaited_once_with(
        pool,
        vendor_name_query="Zendesk",
        exclude_suppressed=True,
    )
    read_high_intent_mock.assert_awaited_once_with(
        pool,
        min_urgency=7.0,
        window_days=3650,
        vendor_name="Zendesk",
        limit=5,
    )
    counts_sql = pool.fetchrow.await_args_list[0].args[0]
    pain_sql = pool.fetch.await_args_list[0].args[0]
    assert "b2b_review_vendor_mentions vm" in counts_sql
    assert "b2b_review_vendor_mentions vm" in pain_sql
    counts_query, counts_vendor = pool.fetchrow.await_args.args
    pain_query, pain_vendor = pool.fetch.await_args.args
    assert counts_vendor == "Zendesk"
    assert pain_vendor == "Zendesk"
    assert body["success"] is True
    assert body["profile"]["vendor_name"] == "Zendesk"


@pytest.mark.asyncio
async def test_reason_vendor_rejects_blank_vendor_name_without_pool(monkeypatch):
    def _boom():
        raise AssertionError("DB pool should not be acquired")

    monkeypatch.setattr(signals_mcp, "get_pool", _boom)

    body = json.loads(await signals_mcp.reason_vendor("   "))

    assert body == {"success": False, "error": "vendor_name is required"}


@pytest.mark.asyncio
async def test_reason_vendor_trims_vendor_name_before_reader_call(monkeypatch):
    pool = _mock_pool()
    monkeypatch.setattr(signals_mcp, "get_pool", lambda: pool)
    view = types.SimpleNamespace(vendor_name="Zendesk")
    load_mock = AsyncMock(return_value=view)

    with patch("atlas_brain.autonomous.tasks._b2b_synthesis_reader.load_best_reasoning_view", new=load_mock), \
         patch("atlas_brain.autonomous.tasks._b2b_synthesis_reader.synthesis_view_to_reasoning_entry", return_value={
             "mode": "replacement",
             "confidence": 0.8,
             "archetype": "pricing_pressure",
             "risk_level": "high",
             "executive_summary": "summary",
             "key_signals": ["pricing"],
             "falsification_conditions": [],
             "uncertainty_sources": [],
         }):
        body = json.loads(await signals_mcp.reason_vendor("  Zendesk  "))

    load_mock.assert_awaited_once_with(pool, "Zendesk")
    assert body["success"] is True
    assert body["vendor_name"] == "Zendesk"


@pytest.mark.asyncio
async def test_compare_vendors_trims_and_filters_vendor_list(monkeypatch):
    pool = _mock_pool()
    monkeypatch.setattr(signals_mcp, "get_pool", lambda: pool)
    zendesk_view = types.SimpleNamespace(vendor_name="Zendesk")
    freshdesk_view = types.SimpleNamespace(vendor_name="Freshdesk")
    load_mock = AsyncMock(return_value={"Zendesk": zendesk_view, "Freshdesk": freshdesk_view})

    def _entry_for_view(view):
        return {
            "mode": "replacement",
            "confidence": 0.8,
            "archetype": f"{view.vendor_name.lower()}_archetype",
            "risk_level": "high",
            "executive_summary": f"{view.vendor_name} summary",
            "key_signals": [],
            "falsification_conditions": [],
        }

    with patch("atlas_brain.autonomous.tasks._b2b_synthesis_reader.load_best_reasoning_views", new=load_mock), \
         patch("atlas_brain.autonomous.tasks._b2b_synthesis_reader.synthesis_view_to_reasoning_entry", side_effect=_entry_for_view):
        body = json.loads(await signals_mcp.compare_vendors(["  Zendesk  ", " ", " Freshdesk "]))

    load_mock.assert_awaited_once_with(pool, ["Zendesk", "Freshdesk"])
    assert body["success"] is True
    assert body["count"] == 2
    assert [row["vendor_name"] for row in body["vendors"]] == ["Zendesk", "Freshdesk"]
