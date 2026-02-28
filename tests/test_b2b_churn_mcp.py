"""
Unit tests for the B2B Churn Intelligence MCP server tools.

All external dependencies (DB pool, GPU packages) are mocked so these
tests run without any real services or GPU hardware.
"""

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock heavy/unavailable dependencies at import time.
# ---------------------------------------------------------------------------
for _heavy_mod in [
    "PIL", "PIL.Image",
    "torch",
    "transformers",
    "numpy",
    "sentence_transformers",
    "asyncpg",
    "llama_cpp",
    "dateparser",
    "starlette", "starlette.applications", "starlette.middleware",
    "starlette.requests", "starlette.responses", "starlette.routing",
    "sse_starlette", "sse_starlette.sse",
    "uvicorn", "anyio",
    "httpx", "httpx_sse",
    "pydantic_settings",
]:
    sys.modules.setdefault(_heavy_mod, MagicMock())


# Mock the mcp package with a FastMCP whose @tool() decorator is a passthrough.
class _MockFastMCP:
    def __init__(self, *args, **kwargs):
        self.settings = MagicMock()

    def tool(self):
        def _passthrough(fn):
            return fn
        return _passthrough

    def run(self, **kwargs):
        pass


_mcp_mod = MagicMock()
_mcp_server_mod = MagicMock()
_fastmcp_mod = MagicMock()
_fastmcp_mod.FastMCP = _MockFastMCP
sys.modules.setdefault("mcp", _mcp_mod)
sys.modules.setdefault("mcp.server", _mcp_server_mod)
sys.modules.setdefault("mcp.server.fastmcp", _fastmcp_mod)

import json
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_pool(fetch_return=None, fetchrow_return=None):
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(return_value=fetch_return or [])
    pool.fetchrow = AsyncMock(return_value=fetchrow_return)
    return pool


def _patch_pool(mock_pool):
    return patch(
        "atlas_brain.storage.database.get_db_pool",
        return_value=mock_pool,
    )


def _make_churn_signal(**kwargs) -> dict:
    return {
        "id": str(uuid4()),
        "vendor_name": kwargs.get("vendor_name", "Zendesk"),
        "product_category": kwargs.get("product_category", "Customer Support"),
        "total_reviews": kwargs.get("total_reviews", 120),
        "negative_reviews": kwargs.get("negative_reviews", 45),
        "churn_intent_count": kwargs.get("churn_intent_count", 18),
        "avg_urgency_score": Decimal("7.2"),
        "avg_rating_normalized": Decimal("0.58"),
        "nps_proxy": Decimal("-15.3"),
        "price_complaint_rate": Decimal("0.2100"),
        "decision_maker_churn_rate": Decimal("0.1400"),
        "top_pain_categories": [{"category": "pricing", "count": 25}],
        "top_competitors": [{"name": "Freshdesk", "mentions": 12}],
        "top_feature_gaps": ["reporting", "automation"],
        "company_churn_list": [{"company": "Acme Corp", "urgency": 9}],
        "quotable_evidence": ["Too expensive for what you get"],
        "last_computed_at": datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc),
        "created_at": datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc),
    }


def _make_review_row(**kwargs) -> dict:
    return {
        "id": uuid4(),
        "dedup_key": f"g2-{uuid4().hex[:8]}",
        "source": kwargs.get("source", "g2"),
        "source_url": "https://g2.com/review/123",
        "source_review_id": "123",
        "vendor_name": kwargs.get("vendor_name", "Zendesk"),
        "product_name": "Zendesk Support",
        "product_category": kwargs.get("product_category", "Customer Support"),
        "rating": Decimal("2.0"),
        "rating_max": Decimal("5"),
        "summary": "Overpriced and slow",
        "review_text": "We switched after 2 years. Too expensive.",
        "pros": "Good UI",
        "cons": "Expensive, slow support",
        "reviewer_name": "John D.",
        "reviewer_title": "VP Operations",
        "reviewer_company": kwargs.get("reviewer_company", "Acme Corp"),
        "company_size_raw": "201-500",
        "reviewer_industry": "SaaS",
        "reviewed_at": datetime(2026, 2, 15, tzinfo=timezone.utc),
        "imported_at": datetime(2026, 2, 20, tzinfo=timezone.utc),
        "import_batch_id": "batch-001",
        "raw_metadata": {"source_weight": 1.0},
        "enrichment": {
            "urgency_score": 8,
            "pain_category": "pricing",
            "churn_signals": {"intent_to_leave": True},
            "reviewer_context": {"role_level": "vp", "decision_maker": True},
            "competitors_mentioned": [{"name": "Freshdesk", "context": "considering"}],
            "feature_gaps": ["reporting"],
            "contract_context": {"contract_value_signal": "mid_market"},
        },
        "enrichment_status": "enriched",
        "enrichment_attempts": 1,
        "enriched_at": datetime(2026, 2, 21, tzinfo=timezone.utc),
    }


def _make_high_intent_row(**kwargs) -> dict:
    return {
        "reviewer_company": kwargs.get("reviewer_company", "Acme Corp"),
        "vendor_name": kwargs.get("vendor_name", "Zendesk"),
        "product_category": "Customer Support",
        "role_level": "vp",
        "is_dm": True,
        "urgency": Decimal("8.5"),
        "pain": "pricing",
        "alternatives": [{"name": "Freshdesk"}],
        "value_signal": "mid_market",
    }


def _make_report_row(**kwargs) -> dict:
    rid = uuid4()
    return {
        "id": rid,
        "report_date": datetime(2026, 2, 28).date(),
        "report_type": kwargs.get("report_type", "weekly_churn_feed"),
        "vendor_filter": kwargs.get("vendor_filter", "Zendesk"),
        "category_filter": None,
        "executive_summary": "Zendesk showing elevated churn signals.",
        "intelligence_data": {"vendors": [{"name": "Zendesk"}]},
        "data_density": {"reviews": 120, "signals": 5},
        "status": "published",
        "llm_model": "qwen3:14b",
        "created_at": datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc),
    }


def _make_scrape_target(**kwargs) -> dict:
    return {
        "id": uuid4(),
        "source": kwargs.get("source", "g2"),
        "vendor_name": kwargs.get("vendor_name", "Zendesk"),
        "product_name": "Zendesk Support",
        "product_category": "Customer Support",
        "enabled": kwargs.get("enabled", True),
        "priority": kwargs.get("priority", 5),
        "last_scraped_at": datetime(2026, 2, 27, 6, 0, tzinfo=timezone.utc),
        "last_scrape_status": "success",
        "last_scrape_reviews": 15,
    }


# ===========================================================================
# B2B Churn MCP Tools
# ===========================================================================

class TestB2BChurnMCPTools:
    """Tests for B2B Churn Intelligence MCP server tools."""

    # -- list_churn_signals ------------------------------------------------

    async def test_list_churn_signals_returns_results(self):
        from atlas_brain.mcp.b2b_churn_server import list_churn_signals

        signal = _make_churn_signal()
        pool = _mock_pool(fetch_return=[signal])

        with _patch_pool(pool):
            raw = await list_churn_signals()

        data = json.loads(raw)
        assert data["count"] == 1
        assert data["signals"][0]["vendor_name"] == "Zendesk"
        assert data["signals"][0]["avg_urgency_score"] == 7.2

    async def test_list_churn_signals_empty(self):
        from atlas_brain.mcp.b2b_churn_server import list_churn_signals

        pool = _mock_pool(fetch_return=[])

        with _patch_pool(pool):
            raw = await list_churn_signals()

        data = json.loads(raw)
        assert data["count"] == 0
        assert data["signals"] == []

    async def test_list_churn_signals_with_vendor_filter(self):
        from atlas_brain.mcp.b2b_churn_server import list_churn_signals

        pool = _mock_pool(fetch_return=[_make_churn_signal()])

        with _patch_pool(pool):
            raw = await list_churn_signals(vendor_name="Zendesk")

        data = json.loads(raw)
        assert data["count"] == 1
        # Verify the SQL included ILIKE filter
        call_args = pool.fetch.call_args
        assert "ILIKE" in call_args[0][0]
        assert "Zendesk" in call_args[0][1:]

    async def test_list_churn_signals_with_min_urgency(self):
        from atlas_brain.mcp.b2b_churn_server import list_churn_signals

        pool = _mock_pool(fetch_return=[])

        with _patch_pool(pool):
            raw = await list_churn_signals(min_urgency=5.0)

        data = json.loads(raw)
        call_args = pool.fetch.call_args
        assert "avg_urgency_score >=" in call_args[0][0]

    async def test_list_churn_signals_limit_capped(self):
        from atlas_brain.mcp.b2b_churn_server import list_churn_signals

        pool = _mock_pool(fetch_return=[])

        with _patch_pool(pool):
            await list_churn_signals(limit=500)

        call_args = pool.fetch.call_args
        # Last positional param is the capped limit
        assert call_args[0][-1] == 100

    async def test_list_churn_signals_error(self):
        from atlas_brain.mcp.b2b_churn_server import list_churn_signals

        pool = _mock_pool()
        pool.fetch = AsyncMock(side_effect=RuntimeError("DB down"))

        with _patch_pool(pool):
            raw = await list_churn_signals()

        data = json.loads(raw)
        assert "error" in data
        assert data["signals"] == []
        assert data["count"] == 0

    # -- get_churn_signal --------------------------------------------------

    async def test_get_churn_signal_found(self):
        from atlas_brain.mcp.b2b_churn_server import get_churn_signal

        signal = _make_churn_signal()
        pool = _mock_pool(fetchrow_return=signal)

        with _patch_pool(pool):
            raw = await get_churn_signal(vendor_name="Zendesk")

        data = json.loads(raw)
        assert data["success"] is True
        assert data["signal"]["vendor_name"] == "Zendesk"
        assert isinstance(data["signal"]["top_pain_categories"], list)
        assert isinstance(data["signal"]["top_competitors"], list)

    async def test_get_churn_signal_not_found(self):
        from atlas_brain.mcp.b2b_churn_server import get_churn_signal

        pool = _mock_pool(fetchrow_return=None)

        with _patch_pool(pool):
            raw = await get_churn_signal(vendor_name="NonExistent")

        data = json.loads(raw)
        assert data["success"] is False
        assert "no churn signal found" in data["error"].lower()

    async def test_get_churn_signal_empty_name(self):
        from atlas_brain.mcp.b2b_churn_server import get_churn_signal

        raw = await get_churn_signal(vendor_name="")
        data = json.loads(raw)
        assert data["success"] is False
        assert "required" in data["error"]

    async def test_get_churn_signal_with_category(self):
        from atlas_brain.mcp.b2b_churn_server import get_churn_signal

        signal = _make_churn_signal()
        pool = _mock_pool(fetchrow_return=signal)

        with _patch_pool(pool):
            raw = await get_churn_signal(
                vendor_name="Zendesk", product_category="Customer Support"
            )

        data = json.loads(raw)
        assert data["success"] is True
        call_args = pool.fetchrow.call_args
        assert "product_category = $2" in call_args[0][0]

    # -- list_high_intent_companies ----------------------------------------

    async def test_list_high_intent_companies_found(self):
        from atlas_brain.mcp.b2b_churn_server import list_high_intent_companies

        row = _make_high_intent_row()
        pool = _mock_pool(fetch_return=[row])

        with _patch_pool(pool):
            raw = await list_high_intent_companies()

        data = json.loads(raw)
        assert data["count"] == 1
        assert data["companies"][0]["company"] == "Acme Corp"
        assert data["companies"][0]["decision_maker"] is True

    async def test_list_high_intent_companies_with_vendor(self):
        from atlas_brain.mcp.b2b_churn_server import list_high_intent_companies

        pool = _mock_pool(fetch_return=[_make_high_intent_row()])

        with _patch_pool(pool):
            raw = await list_high_intent_companies(vendor_name="Zendesk")

        data = json.loads(raw)
        assert data["count"] == 1
        call_args = pool.fetch.call_args
        assert "ILIKE" in call_args[0][0]

    async def test_list_high_intent_companies_error(self):
        from atlas_brain.mcp.b2b_churn_server import list_high_intent_companies

        pool = _mock_pool()
        pool.fetch = AsyncMock(side_effect=RuntimeError("timeout"))

        with _patch_pool(pool):
            raw = await list_high_intent_companies()

        data = json.loads(raw)
        assert "error" in data
        assert data["companies"] == []
        assert data["count"] == 0

    # -- get_vendor_profile ------------------------------------------------

    async def test_get_vendor_profile_full(self):
        from atlas_brain.mcp.b2b_churn_server import get_vendor_profile

        signal = _make_churn_signal()
        counts_row = {"total_reviews": 120, "pending_enrichment": 5, "enriched": 115}
        hi_row = {"reviewer_company": "Acme Corp", "urgency": Decimal("9"), "pain": "pricing"}
        pain_row = {"pain": "pricing", "cnt": 25}

        pool = _mock_pool()
        pool.fetchrow = AsyncMock(side_effect=[signal, counts_row])
        pool.fetch = AsyncMock(side_effect=[[hi_row], [pain_row]])

        with _patch_pool(pool):
            raw = await get_vendor_profile(vendor_name="Zendesk")

        data = json.loads(raw)
        assert data["success"] is True
        profile = data["profile"]
        assert profile["vendor_name"] == "Zendesk"
        assert profile["churn_signal"] is not None
        assert profile["review_counts"]["total"] == 120
        assert profile["review_counts"]["enriched"] == 115
        assert len(profile["high_intent_companies"]) == 1
        assert len(profile["pain_distribution"]) == 1

    async def test_get_vendor_profile_no_signal(self):
        from atlas_brain.mcp.b2b_churn_server import get_vendor_profile

        counts_row = {"total_reviews": 10, "pending_enrichment": 10, "enriched": 0}

        pool = _mock_pool()
        pool.fetchrow = AsyncMock(side_effect=[None, counts_row])
        pool.fetch = AsyncMock(side_effect=[[], []])

        with _patch_pool(pool):
            raw = await get_vendor_profile(vendor_name="Unknown")

        data = json.loads(raw)
        assert data["success"] is True
        assert data["profile"]["churn_signal"] is None
        assert data["profile"]["review_counts"]["total"] == 10

    async def test_get_vendor_profile_empty_name(self):
        from atlas_brain.mcp.b2b_churn_server import get_vendor_profile

        raw = await get_vendor_profile(vendor_name="  ")
        data = json.loads(raw)
        assert data["success"] is False
        assert "required" in data["error"]

    # -- list_reports ------------------------------------------------------

    async def test_list_reports_returns_results(self):
        from atlas_brain.mcp.b2b_churn_server import list_reports

        report = _make_report_row()
        pool = _mock_pool(fetch_return=[report])

        with _patch_pool(pool):
            raw = await list_reports()

        data = json.loads(raw)
        assert data["count"] == 1
        assert data["reports"][0]["report_type"] == "weekly_churn_feed"

    async def test_list_reports_with_type_filter(self):
        from atlas_brain.mcp.b2b_churn_server import list_reports

        pool = _mock_pool(fetch_return=[_make_report_row()])

        with _patch_pool(pool):
            raw = await list_reports(report_type="vendor_scorecard")

        call_args = pool.fetch.call_args
        assert "report_type = $1" in call_args[0][0]

    async def test_list_reports_invalid_type(self):
        from atlas_brain.mcp.b2b_churn_server import list_reports

        raw = await list_reports(report_type="invalid_type")
        data = json.loads(raw)
        assert "error" in data
        assert data["reports"] == []
        assert data["count"] == 0

    # -- get_report --------------------------------------------------------

    async def test_get_report_found(self):
        from atlas_brain.mcp.b2b_churn_server import get_report

        report = _make_report_row()
        pool = _mock_pool(fetchrow_return=report)

        with _patch_pool(pool):
            raw = await get_report(report_id=str(report["id"]))

        data = json.loads(raw)
        assert data["success"] is True
        assert data["report"]["report_type"] == "weekly_churn_feed"
        assert isinstance(data["report"]["intelligence_data"], dict)

    async def test_get_report_not_found(self):
        from atlas_brain.mcp.b2b_churn_server import get_report

        pool = _mock_pool(fetchrow_return=None)

        with _patch_pool(pool):
            raw = await get_report(report_id=str(uuid4()))

        data = json.loads(raw)
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    async def test_get_report_invalid_uuid(self):
        from atlas_brain.mcp.b2b_churn_server import get_report

        raw = await get_report(report_id="not-a-uuid")
        data = json.loads(raw)
        assert data["success"] is False
        assert "UUID" in data["error"]

    # -- search_reviews ----------------------------------------------------

    async def test_search_reviews_returns_results(self):
        from atlas_brain.mcp.b2b_churn_server import search_reviews

        row = {
            "id": uuid4(),
            "vendor_name": "Zendesk",
            "product_category": "Customer Support",
            "reviewer_company": "Acme Corp",
            "rating": Decimal("2.0"),
            "urgency_score": Decimal("8"),
            "pain_category": "pricing",
            "intent_to_leave": True,
            "decision_maker": True,
            "enriched_at": datetime(2026, 2, 21, tzinfo=timezone.utc),
        }
        pool = _mock_pool(fetch_return=[row])

        with _patch_pool(pool):
            raw = await search_reviews(vendor_name="Zendesk")

        data = json.loads(raw)
        assert data["count"] == 1
        assert data["reviews"][0]["vendor_name"] == "Zendesk"
        assert data["reviews"][0]["intent_to_leave"] is True

    async def test_search_reviews_all_filters(self):
        from atlas_brain.mcp.b2b_churn_server import search_reviews

        pool = _mock_pool(fetch_return=[])

        with _patch_pool(pool):
            raw = await search_reviews(
                vendor_name="Zendesk",
                pain_category="pricing",
                min_urgency=7.0,
                company="Acme",
                has_churn_intent=True,
                window_days=60,
                limit=10,
            )

        data = json.loads(raw)
        assert data["count"] == 0
        call_args = pool.fetch.call_args
        sql = call_args[0][0]
        assert "ILIKE" in sql
        assert "pain_category" in sql
        assert "urgency_score" in sql
        assert "reviewer_company ILIKE" in sql
        assert "intent_to_leave" in sql

    async def test_search_reviews_limit_capped(self):
        from atlas_brain.mcp.b2b_churn_server import search_reviews

        pool = _mock_pool(fetch_return=[])

        with _patch_pool(pool):
            await search_reviews(limit=999)

        call_args = pool.fetch.call_args
        assert call_args[0][-1] == 100

    async def test_search_reviews_error(self):
        from atlas_brain.mcp.b2b_churn_server import search_reviews

        pool = _mock_pool()
        pool.fetch = AsyncMock(side_effect=RuntimeError("connection lost"))

        with _patch_pool(pool):
            raw = await search_reviews()

        data = json.loads(raw)
        assert "error" in data
        assert data["reviews"] == []
        assert data["count"] == 0

    # -- get_review --------------------------------------------------------

    async def test_get_review_found(self):
        from atlas_brain.mcp.b2b_churn_server import get_review

        row = _make_review_row()
        pool = _mock_pool(fetchrow_return=row)

        with _patch_pool(pool):
            raw = await get_review(review_id=str(row["id"]))

        data = json.loads(raw)
        assert data["success"] is True
        assert data["review"]["vendor_name"] == "Zendesk"
        assert data["review"]["review_text"] is not None
        assert isinstance(data["review"]["enrichment"], dict)

    async def test_get_review_not_found(self):
        from atlas_brain.mcp.b2b_churn_server import get_review

        pool = _mock_pool(fetchrow_return=None)

        with _patch_pool(pool):
            raw = await get_review(review_id=str(uuid4()))

        data = json.loads(raw)
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    async def test_get_review_invalid_uuid(self):
        from atlas_brain.mcp.b2b_churn_server import get_review

        raw = await get_review(review_id="bad-id")
        data = json.loads(raw)
        assert data["success"] is False
        assert "UUID" in data["error"]

    # -- get_pipeline_status -----------------------------------------------

    async def test_get_pipeline_status_success(self):
        from atlas_brain.mcp.b2b_churn_server import get_pipeline_status

        status_rows = [
            {"enrichment_status": "pending", "cnt": 50},
            {"enrichment_status": "enriched", "cnt": 200},
        ]
        stats_row = {
            "recent_imports_24h": 12,
            "last_enrichment_at": datetime(2026, 2, 28, 10, 0, tzinfo=timezone.utc),
        }
        scrape_row = {
            "active_scrape_targets": 8,
            "last_scrape_at": datetime(2026, 2, 27, 6, 0, tzinfo=timezone.utc),
        }

        pool = _mock_pool(fetch_return=status_rows)
        pool.fetchrow = AsyncMock(side_effect=[stats_row, scrape_row])

        with _patch_pool(pool):
            raw = await get_pipeline_status()

        data = json.loads(raw)
        assert data["success"] is True
        assert data["enrichment_counts"]["pending"] == 50
        assert data["enrichment_counts"]["enriched"] == 200
        assert data["recent_imports_24h"] == 12
        assert data["active_scrape_targets"] == 8

    async def test_get_pipeline_status_error(self):
        from atlas_brain.mcp.b2b_churn_server import get_pipeline_status

        pool = _mock_pool()
        pool.fetch = AsyncMock(side_effect=RuntimeError("DB unavailable"))

        with _patch_pool(pool):
            raw = await get_pipeline_status()

        data = json.loads(raw)
        assert data["success"] is False
        assert "error" in data

    # -- list_scrape_targets -----------------------------------------------

    async def test_list_scrape_targets_returns_results(self):
        from atlas_brain.mcp.b2b_churn_server import list_scrape_targets

        target = _make_scrape_target()
        pool = _mock_pool(fetch_return=[target])

        with _patch_pool(pool):
            raw = await list_scrape_targets()

        data = json.loads(raw)
        assert data["count"] == 1
        assert data["targets"][0]["vendor_name"] == "Zendesk"
        assert data["targets"][0]["source"] == "g2"

    async def test_list_scrape_targets_source_filter(self):
        from atlas_brain.mcp.b2b_churn_server import list_scrape_targets

        pool = _mock_pool(fetch_return=[])

        with _patch_pool(pool):
            raw = await list_scrape_targets(source="capterra")

        call_args = pool.fetch.call_args
        assert "source = $1" in call_args[0][0]

    async def test_list_scrape_targets_invalid_source(self):
        from atlas_brain.mcp.b2b_churn_server import list_scrape_targets

        raw = await list_scrape_targets(source="yelp")
        data = json.loads(raw)
        assert "error" in data
        assert data["targets"] == []
        assert data["count"] == 0

    async def test_list_scrape_targets_disabled_included(self):
        from atlas_brain.mcp.b2b_churn_server import list_scrape_targets

        pool = _mock_pool(fetch_return=[])

        with _patch_pool(pool):
            raw = await list_scrape_targets(enabled_only=False)

        call_args = pool.fetch.call_args
        sql = call_args[0][0]
        assert "enabled = true" not in sql

    async def test_list_scrape_targets_error(self):
        from atlas_brain.mcp.b2b_churn_server import list_scrape_targets

        pool = _mock_pool()
        pool.fetch = AsyncMock(side_effect=RuntimeError("connection refused"))

        with _patch_pool(pool):
            raw = await list_scrape_targets()

        data = json.loads(raw)
        assert "error" in data
        assert data["targets"] == []
        assert data["count"] == 0
