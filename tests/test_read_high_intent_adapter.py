"""Direct tests for read_high_intent_companies adapter.

Exercises the real adapter and _fetch_high_intent_companies with a mock
pool that returns DB-shaped rows, verifying column mapping, type coercion,
filtering, and the full output contract.
"""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod
from atlas_brain.autonomous.tasks._b2b_shared import read_high_intent_companies


def _make_db_row(**overrides):
    """Return a dict mimicking an asyncpg Record from the high-intent SQL."""
    row = {
        "review_id": uuid4(),
        "source": "g2",
        "reviewer_company": "Acme Corp",
        "raw_reviewer_company": "Acme Corp",
        "resolution_confidence": "high",
        "vendor_name": "Zendesk",
        "product_category": "Customer Support",
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
        "company_description": "B2B SaaS ops platform.",
        "role_level": "vp",
        "is_dm": True,
        "urgency": Decimal("8.5"),
        "pain": "pricing",
        "alternatives": json.dumps([{"name": "Freshdesk", "reason": "cheaper"}]),
        "quotes": json.dumps(["We are leaving", "Too expensive"]),
        "value_signal": "mid_market",
        "seat_count": "150",
        "lock_in_level": "medium",
        "contract_end": "2026-06-30",
        "buying_stage": "evaluation",
        "relevance_score": Decimal("0.85"),
        "author_churn_score": Decimal("0.7"),
        "intent_to_leave": False,
        "actively_evaluating": True,
        "contract_renewal_mentioned": False,
        "indicator_cancel": False,
        "indicator_migration": True,
        "indicator_evaluation": True,
        "indicator_switch": False,
    }
    row.update(overrides)
    return row


class FakePool:
    """Minimal pool mock that records fetch calls."""

    def __init__(self, rows):
        self._rows = rows
        self.fetch = AsyncMock(return_value=rows)


# Patch the source allowlist and eligibility filters so the adapter
# does not need a real DB connection for those helpers.
@pytest.fixture(autouse=True)
def _patch_shared_helpers():
    with patch(
        "atlas_brain.autonomous.tasks._b2b_shared._intelligence_source_allowlist",
        return_value=["g2", "capterra", "trustpilot"],
    ), patch(
        "atlas_brain.autonomous.tasks._b2b_shared._eligible_review_filters",
        return_value=(
            "r.enrichment_status = 'enriched'"
            " AND r.enriched_at > NOW() - make_interval(days => $2)"
            " AND r.source = ANY($3::text[])"
        ),
    ):
        yield


@pytest.mark.asyncio
async def test_output_shape_has_all_required_keys():
    """Adapter output contains every documented key."""
    pool = FakePool([_make_db_row()])
    results = await read_high_intent_companies(pool, min_urgency=7.0, window_days=30)

    assert len(results) == 1
    r = results[0]
    required_keys = {
        "company", "raw_company", "resolution_confidence",
        "vendor", "category", "title", "company_size", "industry",
        "verified_employee_count", "company_country", "company_domain",
        "revenue_range", "founded_year", "total_funding", "funding_stage",
        "headcount_growth_6m", "headcount_growth_12m", "headcount_growth_24m",
        "publicly_traded", "ticker", "company_description",
        "role_level", "decision_maker", "urgency", "pain",
        "alternatives", "quotes", "contract_signal",
        "review_id", "source", "seat_count", "lock_in_level",
        "contract_end", "buying_stage", "relevance_score",
        "author_churn_score", "intent_signals",
    }
    assert set(r.keys()) == required_keys


@pytest.mark.asyncio
async def test_urgency_is_float():
    pool = FakePool([_make_db_row(urgency=Decimal("7.9"))])
    results = await read_high_intent_companies(pool, min_urgency=7.0, window_days=30)
    assert isinstance(results[0]["urgency"], float)
    assert results[0]["urgency"] == 7.9


@pytest.mark.asyncio
async def test_seat_count_parsed_to_int():
    pool = FakePool([_make_db_row(seat_count="250")])
    results = await read_high_intent_companies(pool, min_urgency=7.0, window_days=30)
    assert results[0]["seat_count"] == 250


@pytest.mark.asyncio
async def test_seat_count_none_when_empty():
    pool = FakePool([_make_db_row(seat_count=None)])
    results = await read_high_intent_companies(pool, min_urgency=7.0, window_days=30)
    assert results[0]["seat_count"] is None


@pytest.mark.asyncio
async def test_seat_count_none_when_non_numeric():
    pool = FakePool([_make_db_row(seat_count="many")])
    results = await read_high_intent_companies(pool, min_urgency=7.0, window_days=30)
    assert results[0]["seat_count"] is None


@pytest.mark.asyncio
async def test_alternatives_parsed_from_json_string():
    alts = [{"name": "Freshdesk"}, {"name": "Intercom"}]
    pool = FakePool([_make_db_row(alternatives=json.dumps(alts))])
    results = await read_high_intent_companies(pool, min_urgency=7.0, window_days=30)
    assert results[0]["alternatives"] == alts


@pytest.mark.asyncio
async def test_quotes_parsed_from_json_string():
    quotes = ["We are leaving", "Too expensive"]
    pool = FakePool([_make_db_row(quotes=json.dumps(quotes))])
    results = await read_high_intent_companies(pool, min_urgency=7.0, window_days=30)
    assert results[0]["quotes"] == quotes


@pytest.mark.asyncio
async def test_headcount_growth_converted_to_float():
    pool = FakePool([_make_db_row(
        headcount_growth_6m=Decimal("0.15"),
        headcount_growth_12m=None,
        headcount_growth_24m=Decimal("-0.05"),
    )])
    results = await read_high_intent_companies(pool, min_urgency=7.0, window_days=30)
    r = results[0]
    assert r["headcount_growth_6m"] == 0.15
    assert r["headcount_growth_12m"] is None
    assert r["headcount_growth_24m"] == -0.05


@pytest.mark.asyncio
async def test_intent_signals_mapped():
    pool = FakePool([_make_db_row(
        indicator_cancel=True,
        indicator_migration=False,
        indicator_evaluation=True,
        indicator_switch=False,
    )])
    results = await read_high_intent_companies(pool, min_urgency=7.0, window_days=30)
    assert results[0]["intent_signals"] == {
        "cancel": True,
        "migration": False,
        "evaluation": True,
        "completed_switch": False,
    }


@pytest.mark.asyncio
async def test_rows_without_explicit_signal_evidence_are_filtered_out():
    pool = FakePool([_make_db_row(
        intent_to_leave=False,
        actively_evaluating=False,
        contract_renewal_mentioned=False,
        indicator_cancel=False,
        indicator_migration=False,
        indicator_evaluation=False,
        indicator_switch=False,
    )])
    results = await read_high_intent_companies(pool, min_urgency=7.0, window_days=30)
    assert results == []


@pytest.mark.asyncio
async def test_signal_evidence_clause_is_present_in_sql():
    pool = FakePool([_make_db_row()])
    await read_high_intent_companies(pool, min_urgency=7.0, window_days=30)
    sql = pool.fetch.call_args[0][0]
    assert "intent_to_leave" in sql
    assert "actively_evaluating" in sql
    assert "contract_renewal_mentioned" in sql


@pytest.mark.asyncio
async def test_publicly_traded_none_when_empty():
    pool = FakePool([_make_db_row(publicly_traded_exchange="", publicly_traded_symbol="")])
    results = await read_high_intent_companies(pool, min_urgency=7.0, window_days=30)
    assert results[0]["publicly_traded"] is None
    assert results[0]["ticker"] is None


@pytest.mark.asyncio
async def test_review_id_stringified():
    rid = uuid4()
    pool = FakePool([_make_db_row(review_id=rid)])
    results = await read_high_intent_companies(pool, min_urgency=7.0, window_days=30)
    assert results[0]["review_id"] == str(rid)


@pytest.mark.asyncio
async def test_vendor_name_filter_in_sql():
    """vendor_name param should appear in the SQL query, not filtered in Python."""
    pool = FakePool([_make_db_row()])
    await read_high_intent_companies(pool, min_urgency=7.0, window_days=30, vendor_name="Zendesk")
    sql = pool.fetch.call_args[0][0]
    assert "ILIKE" in sql


@pytest.mark.asyncio
async def test_scoped_vendors_filter_in_sql():
    """scoped_vendors param should appear in the SQL query as ANY()."""
    pool = FakePool([_make_db_row()])
    await read_high_intent_companies(pool, min_urgency=7.0, window_days=30, scoped_vendors=["Zendesk", "HubSpot"])
    sql = pool.fetch.call_args[0][0]
    assert "ANY(" in sql


@pytest.mark.asyncio
async def test_limit_in_sql():
    """limit param should appear in the SQL query as LIMIT."""
    pool = FakePool([_make_db_row()])
    await read_high_intent_companies(pool, min_urgency=7.0, window_days=30, limit=10)
    sql = pool.fetch.call_args[0][0]
    assert "LIMIT" in sql


@pytest.mark.asyncio
async def test_fractional_urgency_not_truncated():
    """min_urgency=7.9 should pass 7.9 to SQL, not 7."""
    pool = FakePool([])
    await read_high_intent_companies(pool, min_urgency=7.9, window_days=30)
    args = pool.fetch.call_args[0]
    # First positional arg after the SQL string is urgency_threshold
    urgency_param = args[1]
    assert urgency_param == 7.9


@pytest.mark.asyncio
async def test_empty_result_set():
    pool = FakePool([])
    results = await read_high_intent_companies(pool, min_urgency=7.0, window_days=30)
    assert results == []


@pytest.mark.asyncio
async def test_ineligible_company_filtered_out():
    """Companies matching vendor name should be excluded by eligibility check."""
    # reviewer_company == vendor_name triggers the eligibility filter
    pool = FakePool([_make_db_row(reviewer_company="Zendesk", vendor_name="Zendesk")])
    results = await read_high_intent_companies(pool, min_urgency=7.0, window_days=30)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_url_like_company_filtered_out():
    pool = FakePool([_make_db_row(reviewer_company="https://chatgpt.com/g/g-LsO4PHxnv-robert-on-ai-and-craftsmanship")])
    results = await read_high_intent_companies(pool, min_urgency=7.0, window_days=30)
    assert results == []


@pytest.mark.asyncio
async def test_low_trust_source_uses_provisional_confidence(monkeypatch):
    monkeypatch.setattr(
        shared_mod.settings.b2b_churn,
        "company_signal_low_trust_min_confidence",
        0.2,
    )
    pool = FakePool([_make_db_row(source="reddit")])
    results = await read_high_intent_companies(pool, min_urgency=7.0, window_days=30)
    assert len(results) == 1
    assert results[0]["source"] == "reddit"


@pytest.mark.asyncio
async def test_scoped_vendors_empty_returns_zero_rows():
    """Empty scoped_vendors means scoped user with no tracked vendors = zero results."""
    pool = FakePool([_make_db_row()])
    results = await read_high_intent_companies(pool, min_urgency=7.0, window_days=30, scoped_vendors=[])
    assert results == []
    pool.fetch.assert_not_called()


@pytest.mark.asyncio
async def test_scoped_vendors_none_means_unscoped():
    """scoped_vendors=None means no scoping (admin/public); query runs without vendor ANY()."""
    pool = FakePool([_make_db_row()])
    results = await read_high_intent_companies(pool, min_urgency=7.0, window_days=30, scoped_vendors=None)
    assert len(results) == 1
    sql = pool.fetch.call_args[0][0]
    assert "r.vendor_name = ANY(" not in sql


@pytest.mark.asyncio
async def test_null_urgency_defaults_to_zero():
    pool = FakePool([_make_db_row(urgency=None)])
    results = await read_high_intent_companies(pool, min_urgency=7.0, window_days=30)
    assert results[0]["urgency"] == 0
