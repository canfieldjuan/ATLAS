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
from datetime import date, datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_pool(fetch_return=None, fetchrow_return=None, fetchval_return=None):
    pool = MagicMock()
    pool.is_initialized = True
    pool.fetch = AsyncMock(return_value=fetch_return or [])
    pool.fetchrow = AsyncMock(return_value=fetchrow_return)
    pool.fetchval = AsyncMock(return_value=fetchval_return)
    pool.execute = AsyncMock(return_value=None)
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
        "top_use_cases": None,
        "top_integration_stacks": None,
        "budget_signal_summary": None,
        "sentiment_distribution": None,
        "buyer_authority_summary": None,
        "timeline_summary": None,
        "source_distribution": None,
        "sample_review_ids": [],
        "review_window_start": None,
        "review_window_end": None,
        "confidence_score": Decimal("0.85"),
        "archetype": kwargs.get("archetype", "pricing_pressure"),
        "archetype_confidence": Decimal("0.82"),
        "reasoning_mode": kwargs.get("reasoning_mode", "replacement"),
        "reasoning_risk_level": kwargs.get("reasoning_risk_level", "high"),
        "reasoning_executive_summary": "Pricing pressure is driving active replacement motion.",
        "reasoning_key_signals": ["price fatigue", "competitor mentions"],
        "reasoning_uncertainty_sources": ["limited enterprise sample"],
        "falsification_conditions": ["renewal sentiment stabilizes"],
        "insider_signal_count": 2,
        "insider_org_health_summary": "Minor talent churn in support org.",
        "insider_talent_drain_rate": Decimal("0.11"),
        "insider_quotable_evidence": ["Support leaders are leaving."],
        "keyword_spike_count": 3,
        "keyword_spike_keywords": ["migration", "pricing", "switch"],
        "keyword_trend_summary": "Migration and pricing complaints are rising.",
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
        "content_type": kwargs.get("content_type", "review"),
        "thread_id": kwargs.get("thread_id", None),
        "relevance_score": Decimal("0.88"),
        "author_churn_score": Decimal("0.73"),
        "low_fidelity": False,
        "low_fidelity_reasons": [],
    }


def _make_adapter_high_intent_result(**kwargs) -> dict:
    """Result shape returned by read_high_intent_companies() adapter."""
    return {
        "company": kwargs.get("company", "Acme Corp"),
        "raw_company": kwargs.get("raw_company", "Acme Corp"),
        "resolution_confidence": kwargs.get("resolution_confidence", "high"),
        "vendor": kwargs.get("vendor", "Zendesk"),
        "category": "Customer Support",
        "title": "VP Operations",
        "company_size": "201-500",
        "industry": "SaaS",
        "verified_employee_count": 320,
        "company_country": "US",
        "company_domain": "acme.example",
        "revenue_range": "$50M-$100M",
        "founded_year": 2015,
        "total_funding": 25000000,
        "funding_stage": "Series B",
        "headcount_growth_6m": 0.12,
        "headcount_growth_12m": 0.25,
        "headcount_growth_24m": 0.50,
        "publicly_traded": None,
        "ticker": None,
        "company_description": "B2B SaaS operations platform.",
        "role_level": "vp",
        "decision_maker": True,
        "urgency": 8.5,
        "pain": "pricing",
        "alternatives": [{"name": "Freshdesk"}],
        "quotes": [],
        "contract_signal": "mid_market",
        "review_id": None,
        "source": "g2",
        "seat_count": None,
        "lock_in_level": None,
        "contract_end": None,
        "buying_stage": None,
        "relevance_score": 0.8,
        "author_churn_score": None,
        "intent_signals": {
            "cancel": False,
            "migration": False,
            "evaluation": True,
            "completed_switch": False,
        },
    }


def _make_high_intent_row(**kwargs) -> dict:
    return {
        "company": kwargs.get("company", "Acme Corp"),
        "raw_company": kwargs.get("raw_company", kwargs.get("company", "Acme Corp")),
        "resolution_confidence": kwargs.get("resolution_confidence", "high"),
        "vendor_name": kwargs.get("vendor_name", "Zendesk"),
        "product_category": "Customer Support",
        "role_level": "vp",
        "is_dm": True,
        "urgency": Decimal("8.5"),
        "pain": "pricing",
        "alternatives": [{"name": "Freshdesk"}],
        "value_signal": "mid_market",
        "seat_count": None,
        "lock_in_level": None,
        "contract_end": None,
        "buying_stage": None,
        "reviewer_title": "VP Operations",
        "company_size_raw": "201-500",
        "industry": "SaaS",
        "verified_employee_count": 320,
        "company_country": "US",
        "company_domain": "acme.example",
        "revenue_range": "$50M-$100M",
        "founded_year": 2015,
        "total_funding": 25000000,
        "latest_funding_stage": "Series B",
        "headcount_growth_6m": Decimal("0.12"),
        "headcount_growth_12m": Decimal("0.25"),
        "headcount_growth_24m": Decimal("0.50"),
        "publicly_traded_exchange": None,
        "publicly_traded_symbol": None,
        "company_description": "B2B SaaS operations platform.",
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
        "scrape_mode": kwargs.get("scrape_mode", "incremental"),
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
        from atlas_brain.mcp.b2b.signals import list_churn_signals

        signal = _make_churn_signal()
        pool = _mock_pool(fetch_return=[signal])

        with _patch_pool(pool):
            raw = await list_churn_signals()

        data = json.loads(raw)
        assert data["count"] == 1
        assert data["signals"][0]["vendor_name"] == "Zendesk"
        assert data["signals"][0]["avg_urgency_score"] == 7.2

    async def test_list_churn_signals_empty(self):
        from atlas_brain.mcp.b2b.signals import list_churn_signals

        pool = _mock_pool(fetch_return=[])

        with _patch_pool(pool):
            raw = await list_churn_signals()

        data = json.loads(raw)
        assert data["count"] == 0
        assert data["signals"] == []

    async def test_list_churn_signals_with_vendor_filter(self):
        from atlas_brain.mcp.b2b.signals import list_churn_signals

        pool = _mock_pool(fetch_return=[_make_churn_signal()])

        with _patch_pool(pool), \
             patch("atlas_brain.mcp.b2b.signals._load_reasoning_views_for_vendors",
                   new=AsyncMock(return_value={})):
            raw = await list_churn_signals(vendor_name="Zendesk")

        data = json.loads(raw)
        assert data["count"] == 1
        # Verify the SQL included ILIKE filter
        call_args = pool.fetch.call_args
        assert "ILIKE" in call_args[0][0]
        assert "Zendesk" in call_args[0][1:]
        assert data["signals"][0]["archetype"] is None
        assert data["signals"][0]["reasoning_risk_level"] is None

    async def test_list_churn_signals_with_min_urgency(self):
        from atlas_brain.mcp.b2b.signals import list_churn_signals

        pool = _mock_pool(fetch_return=[])

        with _patch_pool(pool):
            raw = await list_churn_signals(min_urgency=5.0)

        data = json.loads(raw)
        call_args = pool.fetch.call_args
        assert "avg_urgency_score >=" in call_args[0][0]

    async def test_list_churn_signals_limit_capped(self):
        from atlas_brain.mcp.b2b.signals import list_churn_signals

        pool = _mock_pool(fetch_return=[])

        with _patch_pool(pool):
            await list_churn_signals(limit=500)

        call_args = pool.fetch.call_args
        # Last positional param is the capped limit
        assert call_args[0][-1] == 100

    async def test_list_churn_signals_error(self):
        from atlas_brain.mcp.b2b.signals import list_churn_signals

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
        from atlas_brain.mcp.b2b.signals import get_churn_signal

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
        from atlas_brain.mcp.b2b.signals import get_churn_signal

        pool = _mock_pool(fetchrow_return=None)

        with _patch_pool(pool):
            raw = await get_churn_signal(vendor_name="NonExistent")

        data = json.loads(raw)
        assert data["success"] is False
        assert "no churn signal found" in data["error"].lower()

    async def test_get_churn_signal_empty_name(self):
        from atlas_brain.mcp.b2b.signals import get_churn_signal

        raw = await get_churn_signal(vendor_name="")
        data = json.loads(raw)
        assert data["success"] is False
        assert "required" in data["error"]

    async def test_get_churn_signal_with_category(self):
        from atlas_brain.mcp.b2b.signals import get_churn_signal

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
        from atlas_brain.mcp.b2b.signals import list_high_intent_companies

        adapter_result = [_make_adapter_high_intent_result()]
        pool = _mock_pool()

        with _patch_pool(pool), \
             patch("atlas_brain.autonomous.tasks._b2b_shared.read_high_intent_companies",
                   new=AsyncMock(return_value=adapter_result)):
            raw = await list_high_intent_companies()

        data = json.loads(raw)
        assert data["count"] == 1
        assert data["companies"][0]["company"] == "Acme Corp"
        assert data["companies"][0]["decision_maker"] is True

    async def test_list_high_intent_companies_with_vendor(self):
        from atlas_brain.mcp.b2b.signals import list_high_intent_companies

        adapter_result = [_make_adapter_high_intent_result()]
        pool = _mock_pool()
        mock_adapter = AsyncMock(return_value=adapter_result)

        with _patch_pool(pool), \
             patch("atlas_brain.autonomous.tasks._b2b_shared.read_high_intent_companies",
                   new=mock_adapter):
            raw = await list_high_intent_companies(vendor_name="Zendesk")

        data = json.loads(raw)
        assert data["count"] == 1
        mock_adapter.assert_called_once()
        call_kwargs = mock_adapter.call_args[1]
        assert call_kwargs["vendor_name"] == "Zendesk"

    async def test_list_high_intent_companies_error(self):
        from atlas_brain.mcp.b2b.signals import list_high_intent_companies

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
        from atlas_brain.mcp.b2b.signals import get_vendor_profile

        signal = _make_churn_signal()
        counts_row = {"total_reviews": 120, "pending_enrichment": 5, "enriched": 115}
        hi_row = {
            "reviewer_company": "Acme Corp",
            "urgency": Decimal("9"),
            "pain": "pricing",
            "reviewer_title": "VP Operations",
            "company_size_raw": "201-500",
            "industry": "SaaS",
            "verified_employee_count": 320,
            "company_country": "US",
            "company_domain": "acme.example",
            "revenue_range": "$50M-$100M",
            "founded_year": 2015,
            "total_funding": 25000000,
            "latest_funding_stage": "Series B",
            "headcount_growth_6m": Decimal("0.12"),
            "headcount_growth_12m": Decimal("0.25"),
            "headcount_growth_24m": Decimal("0.50"),
            "publicly_traded_exchange": None,
            "publicly_traded_symbol": None,
            "company_description": "B2B SaaS operations platform.",
        }
        pain_row = {"pain": "pricing", "cnt": 25}

        pool = _mock_pool()
        pool.fetchrow = AsyncMock(side_effect=[signal, counts_row])
        # side_effect order: pain_rows fetch, _apply_field_overrides fetch
        pool.fetch = AsyncMock(side_effect=[[pain_row], []])

        adapter_result = [_make_adapter_high_intent_result()]

        with _patch_pool(pool), \
             patch("atlas_brain.mcp.b2b.signals._load_reasoning_views_for_vendors",
                   new=AsyncMock(return_value={})), \
             patch("atlas_brain.autonomous.tasks._b2b_shared.read_high_intent_companies",
                   new=AsyncMock(return_value=adapter_result)):
            raw = await get_vendor_profile(vendor_name="Zendesk")

        data = json.loads(raw)
        assert data["success"] is True
        profile = data["profile"]
        assert profile["vendor_name"] == "Zendesk"
        assert profile["churn_signal"] is not None
        assert profile["churn_signal"]["archetype"] is None
        assert profile["churn_signal"]["reasoning_risk_level"] is None
        assert profile["review_counts"]["total"] == 120
        assert profile["review_counts"]["enriched"] == 115
        assert len(profile["high_intent_companies"]) == 1
        assert profile["high_intent_companies"][0]["company"] == "Acme Corp"
        assert len(profile["pain_distribution"]) == 1
        counts_sql = pool.fetchrow.call_args_list[1][0][0]
        pain_sql = pool.fetch.call_args_list[0][0][0]
        assert "duplicate_of_review_id IS NULL" in counts_sql
        assert "duplicate_of_review_id IS NULL" in pain_sql

    async def test_get_vendor_profile_no_signal(self):
        from atlas_brain.mcp.b2b.signals import get_vendor_profile

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
        from atlas_brain.mcp.b2b.signals import get_vendor_profile

        raw = await get_vendor_profile(vendor_name="  ")
        data = json.loads(raw)
        assert data["success"] is False
        assert "required" in data["error"]

    # -- list_reports ------------------------------------------------------

    async def test_list_reports_returns_results(self):
        from atlas_brain.mcp.b2b.reports import list_reports

        report = _make_report_row()
        report["intelligence_data"] = {
            "account_reasoning_preview_only": True,
            "account_reasoning_preview": {
                "disclaimer": "Early account signal only.",
                "account_pressure_summary": "A single named account is showing early evaluation pressure.",
                "priority_account_names": ["Concentrix", "Concentrix"],
            },
        }
        pool = _mock_pool(fetch_return=[report])

        with _patch_pool(pool):
            raw = await list_reports()

        data = json.loads(raw)
        assert data["count"] == 1
        assert data["reports"][0]["report_type"] == "weekly_churn_feed"
        assert data["reports"][0]["account_reasoning_preview_only"] is True
        assert data["reports"][0]["account_pressure_summary"] == (
            "A single named account is showing early evaluation pressure."
        )
        assert data["reports"][0]["priority_account_names"] == ["Concentrix"]
        assert data["reports"][0]["account_pressure_disclaimer"] == "Early account signal only."

    async def test_list_reports_with_type_filter(self):
        from atlas_brain.mcp.b2b.reports import list_reports

        pool = _mock_pool(fetch_return=[_make_report_row()])

        with _patch_pool(pool):
            raw = await list_reports(report_type="vendor_scorecard")

        call_args = pool.fetch.call_args
        assert "report_type = $1" in call_args[0][0]

    async def test_list_reports_invalid_type(self):
        from atlas_brain.mcp.b2b.reports import list_reports

        raw = await list_reports(report_type="invalid_type")
        data = json.loads(raw)
        assert "error" in data
        assert data["reports"] == []
        assert data["count"] == 0

    # -- get_report --------------------------------------------------------

    async def test_get_report_found(self):
        from atlas_brain.mcp.b2b.reports import get_report

        report = _make_report_row()
        pool = _mock_pool(fetchrow_return=report)

        with _patch_pool(pool):
            raw = await get_report(report_id=str(report["id"]))

        data = json.loads(raw)
        assert data["success"] is True
        assert data["report"]["report_type"] == "weekly_churn_feed"
        assert isinstance(data["report"]["intelligence_data"], dict)

    async def test_get_report_not_found(self):
        from atlas_brain.mcp.b2b.reports import get_report

        pool = _mock_pool(fetchrow_return=None)

        with _patch_pool(pool):
            raw = await get_report(report_id=str(uuid4()))

        data = json.loads(raw)
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    async def test_get_report_invalid_uuid(self):
        from atlas_brain.mcp.b2b.reports import get_report

        raw = await get_report(report_id="not-a-uuid")
        data = json.loads(raw)
        assert data["success"] is False
        assert "UUID" in data["error"]

    # -- search_reviews ----------------------------------------------------

    async def test_search_reviews_returns_results(self):
        from atlas_brain.mcp.b2b.reviews import search_reviews

        adapter_result = [{
            "id": str(uuid4()),
            "vendor_name": "Zendesk",
            "product_category": "Customer Support",
            "reviewer_company": "Acme Corp",
            "rating": 2.0,
            "source": "g2",
            "reviewed_at": None,
            "urgency_score": 8.0,
            "pain_category": "pricing",
            "intent_to_leave": True,
            "decision_maker": True,
            "role_level": "vp",
            "buying_stage": "evaluation",
            "sentiment_direction": "declining",
            "enriched_at": datetime(2026, 2, 21, tzinfo=timezone.utc),
            "reviewer_title": "VP Operations",
            "company_size": "201-500",
            "industry": "SaaS",
            "content_type": "review",
            "thread_id": None,
            "competitors_mentioned": [],
            "quotable_phrases": [],
            "positive_aspects": [],
            "specific_complaints": [],
            "relevance_score": 0.88,
            "author_churn_score": 0.73,
            "low_fidelity": False,
            "low_fidelity_reasons": [],
        }]
        pool = _mock_pool()

        with _patch_pool(pool), \
             patch("atlas_brain.autonomous.tasks._b2b_shared.read_review_details",
                   new=AsyncMock(return_value=adapter_result)):
            raw = await search_reviews(vendor_name="Zendesk")

        data = json.loads(raw)
        assert data["count"] == 1
        assert data["reviews"][0]["vendor_name"] == "Zendesk"
        assert data["reviews"][0]["intent_to_leave"] is True

    async def test_search_reviews_all_filters(self):
        from atlas_brain.mcp.b2b.reviews import search_reviews

        pool = _mock_pool()
        mock_adapter = AsyncMock(return_value=[])

        with _patch_pool(pool), \
             patch("atlas_brain.autonomous.tasks._b2b_shared.read_review_details",
                   new=mock_adapter):
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
        mock_adapter.assert_called_once()
        kw = mock_adapter.call_args[1]
        assert kw["vendor_name"] == "Zendesk"
        assert kw["pain_category"] == "pricing"
        assert kw["min_urgency"] == 7.0
        assert kw["company"] == "Acme"
        assert kw["has_churn_intent"] is True
        assert kw["window_days"] == 60
        assert kw["limit"] == 10

    async def test_search_reviews_limit_capped(self):
        from atlas_brain.mcp.b2b.reviews import search_reviews

        pool = _mock_pool()
        mock_adapter = AsyncMock(return_value=[])

        with _patch_pool(pool), \
             patch("atlas_brain.autonomous.tasks._b2b_shared.read_review_details",
                   new=mock_adapter):
            await search_reviews(limit=999)

        kw = mock_adapter.call_args[1]
        assert kw["limit"] == 100

    async def test_search_reviews_error(self):
        from atlas_brain.mcp.b2b.reviews import search_reviews

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
        from atlas_brain.mcp.b2b.reviews import get_review

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
        from atlas_brain.mcp.b2b.reviews import get_review

        pool = _mock_pool(fetchrow_return=None)

        with _patch_pool(pool):
            raw = await get_review(review_id=str(uuid4()))

        data = json.loads(raw)
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    async def test_get_review_invalid_uuid(self):
        from atlas_brain.mcp.b2b.reviews import get_review

        raw = await get_review(review_id="bad-id")
        data = json.loads(raw)
        assert data["success"] is False
        assert "UUID" in data["error"]

    # -- get_pipeline_status -----------------------------------------------

    async def test_get_pipeline_status_success(self):
        from atlas_brain.mcp.b2b.pipeline import get_pipeline_status

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
        status_sql = pool.fetch.call_args[0][0]
        stats_sql = pool.fetchrow.call_args_list[0][0][0]
        assert "duplicate_of_review_id IS NULL" in status_sql
        assert "duplicate_of_review_id IS NULL" in stats_sql

    async def test_get_operational_overview_excludes_cross_source_duplicates(self):
        from atlas_brain.mcp.b2b.pipeline import get_operational_overview

        pipeline_row = {
            "pending": 5,
            "enriched": 20,
            "failed": 1,
            "total": 26,
        }
        telemetry_row = {
            "captcha_total": 3,
            "blocks_total": 1,
        }
        review_row = {
            "total_reviews": 26,
            "vendors_tracked": 4,
        }
        health_rows = [
            {"source": "g2", "total": 8, "success": 7, "blocked": 1},
        ]
        event_rows = [
            {
                "vendor_name": "Zendesk",
                "event_type": "vendor_alert",
                "event_date": date(2026, 2, 28),
                "description": "Urgency spiked",
            }
        ]

        pool = _mock_pool()
        pool.fetchrow = AsyncMock(side_effect=[pipeline_row, telemetry_row, review_row])
        pool.fetch = AsyncMock(side_effect=[health_rows, event_rows])

        with _patch_pool(pool):
            raw = await get_operational_overview()

        data = json.loads(raw)
        assert data["pipeline"]["total"] == 26
        assert data["data_summary"]["total_reviews"] == 26
        pipeline_sql = pool.fetchrow.call_args_list[0][0][0]
        review_sql = pool.fetchrow.call_args_list[2][0][0]
        assert "duplicate_of_review_id IS NULL" in pipeline_sql
        assert "duplicate_of_review_id IS NULL" in review_sql

    async def test_get_pipeline_status_error(self):
        from atlas_brain.mcp.b2b.pipeline import get_pipeline_status

        pool = _mock_pool()
        pool.fetch = AsyncMock(side_effect=RuntimeError("DB unavailable"))

        with _patch_pool(pool):
            raw = await get_pipeline_status()

        data = json.loads(raw)
        assert data["success"] is False
        assert "error" in data

    async def test_get_source_correction_impact_is_labeled_raw_provenance(self):
        from atlas_brain.mcp.b2b.corrections import get_source_correction_impact

        pool = _mock_pool(fetch_return=[{
            "source_name": "reddit",
            "vendor_scope": None,
            "reason": "noise",
            "affected_review_count": 12,
            "created_at": datetime(2026, 2, 28, 10, 0, tzinfo=timezone.utc),
        }])

        with _patch_pool(pool):
            raw = await get_source_correction_impact()

        data = json.loads(raw)
        assert data["success"] is True
        assert data["basis"] == "raw_source_provenance"
        assert data["total"] == 1

    # -- list_scrape_targets -----------------------------------------------

    async def test_list_scrape_targets_returns_results(self):
        from atlas_brain.mcp.b2b.scrape_targets import list_scrape_targets

        target = _make_scrape_target()
        pool = _mock_pool(fetch_return=[target])

        with _patch_pool(pool):
            raw = await list_scrape_targets()

        data = json.loads(raw)
        assert data["count"] == 1
        assert data["targets"][0]["vendor_name"] == "Zendesk"
        assert data["targets"][0]["source"] == "g2"

    async def test_list_scrape_targets_source_filter(self):
        from atlas_brain.mcp.b2b.scrape_targets import list_scrape_targets

        pool = _mock_pool(fetch_return=[])

        with _patch_pool(pool):
            raw = await list_scrape_targets(source="capterra")

        call_args = pool.fetch.call_args
        assert "source = $1" in call_args[0][0]

    async def test_list_scrape_targets_invalid_source(self):
        from atlas_brain.mcp.b2b.scrape_targets import list_scrape_targets

        raw = await list_scrape_targets(source="yelp")
        data = json.loads(raw)
        assert "error" in data
        assert data["targets"] == []
        assert data["count"] == 0

    async def test_list_scrape_targets_disabled_included(self):
        from atlas_brain.mcp.b2b.scrape_targets import list_scrape_targets

        pool = _mock_pool(fetch_return=[])

        with _patch_pool(pool):
            raw = await list_scrape_targets(enabled_only=False)

        call_args = pool.fetch.call_args
        sql = call_args[0][0]
        assert "enabled = true" not in sql

    async def test_list_scrape_targets_error(self):
        from atlas_brain.mcp.b2b.scrape_targets import list_scrape_targets

        pool = _mock_pool()
        pool.fetch = AsyncMock(side_effect=RuntimeError("connection refused"))

        with _patch_pool(pool):
            raw = await list_scrape_targets()

        data = json.loads(raw)
        assert "error" in data
        assert data["targets"] == []
        assert data["count"] == 0


# ---------------------------------------------------------------------------
# Regression tests for surgical fixes
# ---------------------------------------------------------------------------


class TestNoduplicateToolDefinitions:
    """Meta-regression: fail fast if known-duplicate tools are reintroduced."""

    def test_trigger_score_calibration_defined_once(self):
        import ast
        import pathlib

        src = (pathlib.Path(__file__).parent.parent /
               "atlas_brain/mcp/b2b/calibration.py").read_text()
        tree = ast.parse(src)
        names = [n.name for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)]
        count = names.count("trigger_score_calibration")
        assert count == 1, (
            f"trigger_score_calibration defined {count} times in atlas_brain/mcp/b2b/calibration.py "
            "(expected exactly 1)"
        )


def _patch_calibrate(fake_fn):
    """Patch the b2b_score_calibration module via sys.modules (import is inline)."""
    import sys as _sys
    mod = MagicMock()
    mod.calibrate = fake_fn
    return patch.dict(_sys.modules, {
        "atlas_brain.autonomous": MagicMock(),
        "atlas_brain.autonomous.tasks": MagicMock(),
        "atlas_brain.autonomous.tasks.b2b_score_calibration": mod,
    })


@pytest.mark.asyncio
class TestTriggerScoreCalibration:
    async def test_response_shape(self):
        from atlas_brain.mcp.b2b.calibration import trigger_score_calibration

        pool = _mock_pool()
        calibrate_result = {"weights_updated": 3, "sequences_used": 42}

        with _patch_pool(pool), _patch_calibrate(AsyncMock(return_value=calibrate_result)):
            raw = await trigger_score_calibration(window_days=180)

        data = json.loads(raw)
        assert data["success"] is True
        assert "window_days" in data
        assert "triggered_by" in data
        assert data["triggered_by"] == "mcp"

    async def test_window_clamped_to_floor(self):
        from atlas_brain.mcp.b2b.calibration import trigger_score_calibration

        pool = _mock_pool()
        captured = {}

        async def _fake_calibrate(pool, window_days):
            captured["window_days"] = window_days
            return {}

        with _patch_pool(pool), _patch_calibrate(_fake_calibrate):
            await trigger_score_calibration(window_days=1)

        assert captured["window_days"] >= 30

    async def test_window_clamped_to_ceiling(self):
        from atlas_brain.mcp.b2b.calibration import trigger_score_calibration

        pool = _mock_pool()
        captured = {}

        async def _fake_calibrate(pool, window_days):
            captured["window_days"] = window_days
            return {}

        with _patch_pool(pool), _patch_calibrate(_fake_calibrate):
            await trigger_score_calibration(window_days=9999)

        assert captured["window_days"] <= 730

    async def test_db_not_ready(self):
        from atlas_brain.mcp.b2b.calibration import trigger_score_calibration

        pool = _mock_pool()
        pool.is_initialized = False

        with _patch_pool(pool), _patch_calibrate(AsyncMock(return_value={})):
            raw = await trigger_score_calibration()

        data = json.loads(raw)
        assert data.get("success") is False or "error" in data


@pytest.mark.asyncio
class TestGetParserHealth:
    async def test_no_nameerror(self):
        """Regression: _pool_or_fail must not be referenced."""
        from atlas_brain.mcp.b2b.pipeline import get_parser_health

        pool = _mock_pool(fetch_return=[])
        with _patch_pool(pool):
            raw = await get_parser_health()
        # Must return valid JSON with expected shape (regression for _pool_or_fail NameError)
        data = json.loads(raw)
        assert data.get("success") is True
        assert "sources" in data

    async def test_db_not_ready(self):
        from atlas_brain.mcp.b2b.pipeline import get_parser_health

        pool = _mock_pool()
        pool.is_initialized = False

        with _patch_pool(pool):
            raw = await get_parser_health()

        data = json.loads(raw)
        assert data.get("success") is False
        assert "Database not ready" in data.get("error", "")

    async def test_success_shape(self):
        from atlas_brain.mcp.b2b.pipeline import get_parser_health

        fake_row = {
            "source": "reddit",
            "parser_version": "reddit:3",
            "review_count": 500,
            "latest_version": "reddit:3",
            "is_stale": False,
        }
        pool = _mock_pool(fetch_return=[fake_row])
        with _patch_pool(pool):
            raw = await get_parser_health()

        data = json.loads(raw)
        assert data["success"] is True
        assert "sources" in data
        assert "total_stale_reviews" in data
        assert data["total_sources"] == 1


@pytest.mark.asyncio
class TestCreateDataCorrectionSuppressSource:
    async def test_suppress_source_without_entity_id(self):
        """suppress_source must succeed without a client-supplied UUID."""
        from atlas_brain.mcp.b2b.corrections import create_data_correction

        inserted_row = {
            "id": str(uuid4()),
            "entity_type": "source",
            "entity_id": str(uuid4()),
            "correction_type": "suppress_source",
            "status": "applied",
            "created_at": datetime(2026, 3, 13, tzinfo=timezone.utc),
        }
        pool = _mock_pool(fetchrow_return=inserted_row)

        with _patch_pool(pool):
            raw = await create_data_correction(
                entity_type="source",
                entity_id="",  # client provides nothing
                correction_type="suppress_source",
                reason="Spam source",
                metadata='{"source_name": "reddit"}',
                corrected_by="test",
            )

        data = json.loads(raw)
        assert data.get("success") is True, f"Expected success, got: {data}"
        # Verify the deterministic sentinel UUID was passed to the INSERT
        import uuid as _uuid_mod
        expected_uuid = _uuid_mod.uuid5(_uuid_mod.NAMESPACE_DNS, "source:reddit")
        call_args = pool.fetchrow.call_args
        entity_uuid_arg = call_args[0][2]  # $2 in the INSERT: entity_id
        assert entity_uuid_arg == expected_uuid, (
            f"Sentinel UUID mismatch: got {entity_uuid_arg}, expected {expected_uuid}"
        )

    async def test_suppress_source_missing_source_name(self):
        """suppress_source without metadata.source_name should fail gracefully."""
        from atlas_brain.mcp.b2b.corrections import create_data_correction

        pool = _mock_pool()
        with _patch_pool(pool):
            raw = await create_data_correction(
                entity_type="source",
                entity_id="",
                correction_type="suppress_source",
                reason="Test",
                metadata="{}",
            )

        data = json.loads(raw)
        assert data.get("success") is False
        assert "source_name" in data.get("error", "").lower()


@pytest.mark.asyncio
class TestGetDisplacementHistorySuppression:
    async def test_suppression_predicate_in_query(self):
        """get_displacement_history must include suppression filtering."""
        from atlas_brain.mcp.b2b.displacement import get_displacement_history

        pool = _mock_pool(fetch_return=[])
        with _patch_pool(pool):
            await get_displacement_history(from_vendor="Salesforce", to_vendor="HubSpot")

        call_args = pool.fetch.call_args
        sql = call_args[0][0]
        # The suppression predicate injects a NOT EXISTS subquery against data_corrections
        assert "NOT EXISTS (SELECT 1 FROM data_corrections dc" in sql, (
            "get_displacement_history SQL must include NOT EXISTS suppression subquery"
        )
        assert "dc.correction_type = 'suppress'" in sql, (
            "suppression predicate must check correction_type = 'suppress'"
        )
        assert "dc.status = 'applied'" in sql, (
            "suppression predicate must check status = 'applied'"
        )


@pytest.mark.asyncio
class TestDraftCampaignBlogIntegration:
    """draft_campaign should fetch blog posts, store them in metadata, and return them."""

    _BLOG_POST = {"title": "Why Users Leave Zendesk", "url": "https://example.com/blog/zendesk", "topic_type": "vendor_alternative"}

    async def _run_draft_campaign(self, pool, blog_posts):
        from atlas_brain.mcp.b2b.write_intelligence import draft_campaign

        campaign_row = {"id": uuid4(), "created_at": datetime(2026, 3, 28, tzinfo=timezone.utc)}
        pool.fetchrow = AsyncMock(return_value=campaign_row)

        async def _fake_blogs(*args, **kwargs):
            return blog_posts

        with _patch_pool(pool):
            with patch("atlas_brain.mcp.b2b.write_intelligence._get_blog_matcher", return_value=_fake_blogs):
                result_json = await draft_campaign(
                    company_name="Acme Corp",
                    vendor_name="Zendesk",
                    channel="email_cold",
                    subject="Why teams leave Zendesk",
                    body="Hi, I noticed your team is evaluating Zendesk alternatives.",
                    llm_model="claude-sonnet-4-6",
                )
        return json.loads(result_json)

    async def test_blog_posts_included_in_response_when_found(self):
        pool = _mock_pool()
        result = await self._run_draft_campaign(pool, [self._BLOG_POST])

        assert result["success"] is True
        assert "blog_posts" in result
        assert result["blog_posts"] == [self._BLOG_POST]

    async def test_blog_posts_absent_from_response_when_none_found(self):
        pool = _mock_pool()
        result = await self._run_draft_campaign(pool, [])

        assert result["success"] is True
        assert "blog_posts" not in result

    async def test_blog_posts_stored_in_metadata_jsonb(self):
        pool = _mock_pool()
        await self._run_draft_campaign(pool, [self._BLOG_POST])

        insert_call = pool.fetchrow.call_args
        sql = insert_call[0][0]
        args = insert_call[0][1:]
        # metadata is the 10th positional arg ($10::jsonb)
        assert "metadata" in sql.lower()
        metadata_arg = next(
            (a for a in args if isinstance(a, str) and "blog_posts" in a), None
        )
        assert metadata_arg is not None
        stored = json.loads(metadata_arg)
        assert stored["blog_posts"] == [self._BLOG_POST]

    async def test_blog_fetch_failure_does_not_abort_campaign_creation(self):
        pool = _mock_pool()
        campaign_row = {"id": uuid4(), "created_at": datetime(2026, 3, 28, tzinfo=timezone.utc)}
        pool.fetchrow = AsyncMock(return_value=campaign_row)

        async def _failing_blogs(*args, **kwargs):
            raise RuntimeError("DB connection lost")

        from atlas_brain.mcp.b2b.write_intelligence import draft_campaign
        with _patch_pool(pool):
            with patch("atlas_brain.mcp.b2b.write_intelligence._get_blog_matcher", return_value=_failing_blogs):
                result_json = await draft_campaign(
                    company_name="Acme Corp",
                    vendor_name="Zendesk",
                    channel="email_cold",
                    subject="Subject",
                    body="Body",
                    llm_model="claude-sonnet-4-6",
                )
        result = json.loads(result_json)
        assert result["success"] is True
        assert "blog_posts" not in result


@pytest.mark.asyncio
class TestBuildChallengerBriefMCP:
    async def test_uses_merged_cross_vendor_lookup(self, monkeypatch):
        from atlas_brain.mcp.b2b.write_intelligence import build_challenger_brief
        import atlas_brain.autonomous.tasks.b2b_challenger_brief as brief_mod
        import atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis as xv_mod

        pool = _mock_pool(fetchrow_return=None)
        merged_lookup = {
            "battles": {
                ("Freshdesk", "Zendesk"): {
                    "conclusion": {
                        "conclusion": "Freshdesk gaining on pricing advantage",
                        "winner": "Freshdesk",
                    },
                },
            },
            "councils": {},
            "asymmetries": {},
        }
        resolve_battle = AsyncMock(return_value={"winner": "Freshdesk"})
        build_brief = MagicMock(return_value={"_executive_summary": "summary", "displacement_summary": {}})

        monkeypatch.setattr(
            brief_mod,
            "_fetch_persisted_report_record",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(brief_mod, "_fetch_persisted_report", AsyncMock(return_value=None))
        monkeypatch.setattr(
            brief_mod,
            "_fetch_displacement_detail",
            AsyncMock(return_value={"total_mentions": 2, "source_distribution": {"g2": 2}}),
        )
        monkeypatch.setattr(brief_mod, "_fetch_product_profile", AsyncMock(return_value=None))
        monkeypatch.setattr(brief_mod, "_fetch_churn_signal", AsyncMock(return_value=None))
        monkeypatch.setattr(brief_mod, "_fetch_review_pain_quotes", AsyncMock(return_value=[]))
        monkeypatch.setattr(brief_mod, "_resolve_cross_vendor_battle", resolve_battle)
        monkeypatch.setattr(brief_mod, "_build_challenger_brief", build_brief)
        monkeypatch.setattr(
            xv_mod,
            "load_best_cross_vendor_lookup",
            AsyncMock(return_value=merged_lookup),
        )

        with _patch_pool(pool):
            raw = await build_challenger_brief(
                incumbent="Zendesk",
                challenger="Freshdesk",
                persist=False,
                max_target_accounts=25,
            )

        payload = json.loads(raw)
        assert payload["success"] is True
        xv_mod.load_best_cross_vendor_lookup.assert_awaited_once()
        resolve_args = resolve_battle.await_args.args
        assert resolve_args[1:4] == ("Zendesk", "Freshdesk", date.today())
        assert resolve_args[4] == merged_lookup
        build_brief.assert_called_once()


@pytest.mark.asyncio
async def test_build_accounts_in_motion_uses_canonical_vault_reader(monkeypatch):
    from atlas_brain.mcp.b2b.write_intelligence import build_accounts_in_motion
    import atlas_brain.autonomous.tasks._b2b_shared as shared_mod
    import atlas_brain.autonomous.tasks.b2b_accounts_in_motion as aim_mod
    import atlas_brain.autonomous.tasks._b2b_synthesis_reader as synth_mod
    import atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis as xv_mod

    pool = _mock_pool()
    read_vaults = AsyncMock(return_value={"Zendesk": {"metric_snapshot": {"avg_urgency": 7.2}}})
    monkeypatch.setattr(shared_mod, "read_vendor_intelligence_map", read_vaults)
    monkeypatch.setattr(
        shared_mod,
        "_fetch_latest_evidence_vault",
        AsyncMock(side_effect=AssertionError("deprecated wrapper should not run")),
    )
    monkeypatch.setattr(shared_mod, "_aggregate_competitive_disp", lambda rows: rows)
    monkeypatch.setattr(shared_mod, "_build_competitor_lookup", lambda rows: {})
    monkeypatch.setattr(shared_mod, "_build_feature_gap_lookup", lambda rows: {})
    monkeypatch.setattr(shared_mod, "_canonicalize_vendor", lambda raw: str(raw or "").strip())
    monkeypatch.setattr(shared_mod, "_fetch_budget_signals", AsyncMock(return_value=[]))
    monkeypatch.setattr(shared_mod, "_fetch_churning_companies", AsyncMock(return_value=[]))
    monkeypatch.setattr(shared_mod, "_fetch_competitive_displacement", AsyncMock(return_value=[]))
    monkeypatch.setattr(shared_mod, "_fetch_feature_gaps", AsyncMock(return_value=[]))
    monkeypatch.setattr(shared_mod, "_fetch_high_intent_companies", AsyncMock(return_value=[]))
    monkeypatch.setattr(shared_mod, "_fetch_latest_account_intelligence", AsyncMock(return_value={}))
    monkeypatch.setattr(shared_mod, "_fetch_price_complaint_rates", AsyncMock(return_value=[]))
    monkeypatch.setattr(shared_mod, "_fetch_quotable_evidence", AsyncMock(return_value=[]))
    monkeypatch.setattr(shared_mod, "_fetch_timeline_signals", AsyncMock(return_value=[]))
    monkeypatch.setattr(aim_mod, "_fetch_company_signal_metadata", AsyncMock(return_value=[]))
    monkeypatch.setattr(aim_mod, "_fetch_apollo_org_lookup", AsyncMock(return_value={}))
    monkeypatch.setattr(aim_mod, "_merge_company_profiles", lambda *args, **kwargs: {
        "acct-1": {"vendor": "Zendesk", "category": "Helpdesk"},
    })
    monkeypatch.setattr(aim_mod, "_compute_account_opportunity_score", lambda acct: (50, {}))
    monkeypatch.setattr(aim_mod, "_apply_account_quality_adjustments", lambda acct, cfg: (0, {}, []))
    monkeypatch.setattr(aim_mod, "_build_vendor_aggregate", lambda *args, **kwargs: {"archetype": "pricing_pressure"})
    monkeypatch.setattr(synth_mod, "load_best_reasoning_view", AsyncMock(return_value=None))
    monkeypatch.setattr(synth_mod, "build_reasoning_lookup_from_views", lambda views: views)
    monkeypatch.setattr(xv_mod, "load_best_cross_vendor_lookup", AsyncMock(return_value={}))

    with _patch_pool(pool):
        raw = await build_accounts_in_motion(
            vendor_name="Zendesk",
            persist=False,
            min_urgency=5.0,
            max_accounts=10,
        )

    payload = json.loads(raw)
    read_vaults.assert_awaited_once()
    assert payload["success"] is True
    assert payload["vendor_name"] == "Zendesk"



@pytest.mark.asyncio
async def test_crm_mcp_list_events_validates_filters_before_pool(monkeypatch):
    from atlas_brain.mcp.b2b.crm_events import list_crm_events

    def _boom():
        raise AssertionError("DB pool should not be acquired")

    monkeypatch.setattr("atlas_brain.storage.database.get_db_pool", _boom)

    status_raw = await list_crm_events(status="not_a_status")
    provider_raw = await list_crm_events(crm_provider="not_a_provider")
    date_raw = await list_crm_events(start_date="not-a-date")

    assert json.loads(status_raw)["error"].startswith("Invalid status.")
    assert json.loads(provider_raw)["error"].startswith("Invalid crm_provider.")
    assert json.loads(date_raw)["error"] == "Invalid start_date (ISO 8601 expected)"


@pytest.mark.asyncio
async def test_crm_mcp_list_events_normalizes_blank_optional_filters():
    from atlas_brain.mcp.b2b.crm_events import list_crm_events

    pool = _mock_pool(fetch_return=[])
    with _patch_pool(pool):
        raw = await list_crm_events(
            status="   ",
            crm_provider="  ",
            company_name="\t",
            start_date="   ",
            end_date="  ",
            limit=50,
        )

    payload = json.loads(raw)
    assert payload["events"] == []
    query, *params = pool.fetch.await_args.args
    assert "status = $" not in query
    assert "crm_provider = $" not in query
    assert "LOWER(company_name) LIKE" not in query
    assert "received_at >= $" not in query
    assert "received_at < $" not in query
    assert params == [50]


@pytest.mark.asyncio
async def test_crm_mcp_ingest_event_trims_text_fields_before_persistence():
    from atlas_brain.mcp.b2b.crm_events import ingest_crm_event

    pool = _mock_pool(fetchval_return=uuid4())
    with _patch_pool(pool):
        raw = await ingest_crm_event(
            crm_provider=" hubspot ",
            event_type=" deal_won ",
            company_name=" Acme Corp ",
            contact_email="   ",
            deal_stage=" closedwon ",
            deal_id=" deal-1 ",
            notes=" note ",
        )

    payload = json.loads(raw)
    assert payload["success"] is True
    args = pool.fetchval.await_args.args
    assert args[1] == "hubspot"
    assert args[2] == "deal_won"
    assert args[3] == "Acme Corp"
    assert args[4] is None
    assert args[5] == "deal-1"
    assert args[6] == "closedwon"
    assert json.loads(args[8]) == {"notes": "note"}


@pytest.mark.asyncio
async def test_crm_mcp_ingest_event_rejects_blank_required_text_before_pool(monkeypatch):
    from atlas_brain.mcp.b2b.crm_events import ingest_crm_event

    def _boom():
        raise AssertionError("DB pool should not be acquired")

    monkeypatch.setattr("atlas_brain.storage.database.get_db_pool", _boom)

    raw = await ingest_crm_event(
        crm_provider="   ",
        event_type=" deal_won ",
        company_name=" Acme Corp ",
    )

    assert json.loads(raw)["error"].startswith("crm_provider must be one of")



@pytest.mark.asyncio
async def test_crm_mcp_list_pushes_validates_subscription_before_pool(monkeypatch):
    from atlas_brain.mcp.b2b.crm_events import list_crm_pushes

    def _boom():
        raise AssertionError("DB pool should not be acquired")

    monkeypatch.setattr("atlas_brain.storage.database.get_db_pool", _boom)

    raw = await list_crm_pushes(subscription_id="not-a-uuid")

    assert json.loads(raw)["error"] == "subscription_id must be a valid UUID"


@pytest.mark.asyncio
async def test_crm_mcp_list_pushes_normalizes_blank_vendor_filter():
    from atlas_brain.mcp.b2b.crm_events import list_crm_pushes

    pool = _mock_pool(fetch_return=[])
    with _patch_pool(pool):
        raw = await list_crm_pushes(vendor_name="   ", limit=50)

    payload = json.loads(raw)
    assert payload["pushes"] == []
    query, *params = pool.fetch.await_args.args
    assert "pl.vendor_name ILIKE" not in query
    assert params == [50]


@pytest.mark.asyncio
async def test_webhook_mcp_update_rejects_embedded_credentials_url():
    from atlas_brain.mcp.b2b.webhooks import update_webhook

    pool = _mock_pool()
    with _patch_pool(pool):
        raw = await update_webhook(
            subscription_id=str(uuid4()),
            url="https://user:pass@hooks.example.com/churn",
        )

    payload = json.loads(raw)
    assert payload == {"error": "Webhook URL must not include embedded credentials"}
    pool.fetchrow.assert_not_awaited()


@pytest.mark.asyncio
async def test_webhook_mcp_update_allows_high_intent_push_for_crm_channel():
    from atlas_brain.mcp.b2b.webhooks import update_webhook

    subscription_id = uuid4()
    pool = _mock_pool()
    pool.fetchrow = AsyncMock(side_effect=[
        {"channel": "crm_hubspot"},
        {
            "id": subscription_id,
            "url": "https://hooks.example.com/crm",
            "event_types": ["high_intent_push", "report_generated"],
            "channel": "crm_hubspot",
            "enabled": True,
            "description": "CRM webhook",
            "updated_at": datetime(2026, 4, 12, tzinfo=timezone.utc),
        },
    ])
    with _patch_pool(pool):
        raw = await update_webhook(
            subscription_id=str(subscription_id),
            event_types="high_intent_push, report_generated",
        )

    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["webhook"]["event_types"] == ["high_intent_push", "report_generated"]
    validate_query, validate_id = pool.fetchrow.await_args_list[0].args
    assert "SELECT COALESCE(channel, 'generic') AS channel" in validate_query
    assert validate_id == subscription_id


@pytest.mark.asyncio
async def test_webhook_mcp_update_rejects_high_intent_push_for_generic_channel():
    from atlas_brain.mcp.b2b.webhooks import update_webhook

    subscription_id = uuid4()
    pool = _mock_pool()
    pool.fetchrow = AsyncMock(return_value={"channel": "generic"})
    with _patch_pool(pool):
        raw = await update_webhook(
            subscription_id=str(subscription_id),
            event_types="high_intent_push",
        )

    payload = json.loads(raw)
    assert payload == {
        "error": "Invalid event_types for channel generic: ['high_intent_push'] require a CRM channel",
    }
    assert pool.fetchrow.await_count == 1


@pytest.mark.asyncio
async def test_webhook_mcp_list_subscriptions_excludes_manual_tests_from_recent_stats():
    from atlas_brain.mcp.b2b.webhooks import list_webhook_subscriptions

    pool = _mock_pool(fetch_return=[
        {
            "id": uuid4(),
            "account_id": uuid4(),
            "account_name": "Acme",
            "url": "https://hooks.example.com/churn",
            "event_types": ["churn_alert"],
            "channel": "generic",
            "enabled": True,
            "description": "Ops webhook",
            "created_at": datetime(2026, 4, 12, tzinfo=timezone.utc),
            "recent_deliveries": 3,
            "recent_successes": 2,
        }
    ])
    with _patch_pool(pool):
        raw = await list_webhook_subscriptions()

    payload = json.loads(raw)
    query = pool.fetch.await_args.args[0]
    assert "dl.event_type <> 'test'" in query
    assert "dl2.event_type <> 'test'" in query
    assert payload["count"] == 1
    assert payload["subscriptions"][0]["recent_success_rate_7d"] == 0.667


@pytest.mark.asyncio
async def test_webhook_mcp_delivery_summary_excludes_manual_tests():
    from atlas_brain.mcp.b2b.webhooks import get_webhook_delivery_summary

    pool = _mock_pool(fetchrow_return={
        "active_subscriptions": 2,
        "total_deliveries": 4,
        "successful": 3,
        "failed": 1,
        "avg_success_duration_ms": 120.4,
        "last_delivery_at": datetime(2026, 4, 12, tzinfo=timezone.utc),
    })
    with _patch_pool(pool):
        raw = await get_webhook_delivery_summary(days=30)

    payload = json.loads(raw)
    query = pool.fetchrow.await_args.args[0]
    assert "dl.event_type <> 'test'" in query
    assert payload["success"] is True
    assert payload["total_deliveries"] == 4
    assert payload["success_rate"] == 0.75
