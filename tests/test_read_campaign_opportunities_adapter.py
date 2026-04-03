"""Direct tests for read_campaign_opportunities adapter."""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from atlas_brain.autonomous.tasks._b2b_shared import read_campaign_opportunities


def _make_db_row(**overrides):
    row = {
        "review_id": uuid4(),
        "vendor_name": "Zendesk",
        "reviewer_company": "Acme Corp",
        "reviewer_name": "Jane Doe",
        "product_category": "Customer Support",
        "source": "g2",
        "reviewed_at": None,
        "urgency": Decimal("8.0"),
        "is_dm": True,
        "role_type": "economic_buyer",
        "buying_stage": "evaluation",
        "seat_count": 150,
        "contract_end": "2026-06-30",
        "decision_timeline": "within_quarter",
        "competitors_json": json.dumps([{"name": "Freshdesk", "context": "comparing pricing"}]),
        "pain_json": json.dumps([{"category": "pricing", "severity": "primary"}]),
        "quotable_phrases": json.dumps(["Way too expensive"]),
        "feature_gaps": json.dumps(["Better automation"]),
        "primary_workflow": "ticket management",
        "integration_stack": json.dumps(["Slack", "Jira"]),
        "sentiment_direction": "declining",
        "industry": "SaaS",
        "reviewer_title": "VP Operations",
        "company_size_raw": "201-500",
    }
    row.update(overrides)
    return row


class FakePool:
    def __init__(self, rows):
        self.fetch = AsyncMock(return_value=rows)


@pytest.fixture(autouse=True)
def _patch_suppress():
    with patch(
        "atlas_brain.services.b2b.corrections.suppress_predicate",
        return_value="TRUE",
    ):
        yield


@pytest.mark.asyncio
async def test_output_shape():
    pool = FakePool([_make_db_row()])
    results = await read_campaign_opportunities(pool)
    assert len(results) == 1
    r = results[0]
    required = {
        "review_id", "vendor_name", "reviewer_company", "reviewer_name",
        "product_category", "source", "reviewed_at",
        "urgency", "is_dm", "role_type", "buying_stage", "seat_count",
        "contract_end", "decision_timeline",
        "competitors", "competitors_json", "pain_json",
        "quotable_phrases", "feature_gaps",
        "primary_workflow", "integration_stack",
        "sentiment_direction", "industry", "reviewer_title", "company_size_raw",
    }
    assert set(r.keys()) == required


@pytest.mark.asyncio
async def test_competitors_parsed():
    comps = [{"name": "Freshdesk"}, {"name": "Intercom"}]
    pool = FakePool([_make_db_row(competitors_json=json.dumps(comps))])
    results = await read_campaign_opportunities(pool)
    assert results[0]["competitors"] == comps


@pytest.mark.asyncio
async def test_urgency_null_preserved():
    pool = FakePool([_make_db_row(urgency=None)])
    results = await read_campaign_opportunities(pool)
    assert results[0]["urgency"] is None


@pytest.mark.asyncio
async def test_dm_null_preserved():
    pool = FakePool([_make_db_row(is_dm=None)])
    results = await read_campaign_opportunities(pool)
    assert results[0]["is_dm"] is None


@pytest.mark.asyncio
async def test_vendor_filter_in_sql():
    pool = FakePool([])
    await read_campaign_opportunities(pool, vendor_name="Zendesk")
    sql = pool.fetch.call_args[0][0]
    assert "ILIKE" in sql


@pytest.mark.asyncio
async def test_company_filter_in_sql():
    pool = FakePool([])
    await read_campaign_opportunities(pool, company="Acme")
    sql = pool.fetch.call_args[0][0]
    assert "reviewer_company" in sql
    assert "ILIKE" in sql


@pytest.mark.asyncio
async def test_dm_only_in_sql():
    pool = FakePool([])
    await read_campaign_opportunities(pool, dm_only=True)
    sql = pool.fetch.call_args[0][0]
    assert "decision_maker" in sql


@pytest.mark.asyncio
async def test_dm_only_false_omitted():
    pool = FakePool([])
    await read_campaign_opportunities(pool, dm_only=False)
    sql = pool.fetch.call_args[0][0]
    where_clause = sql.split("WHERE")[1].split("ORDER")[0]
    assert "decision_maker" not in where_clause


@pytest.mark.asyncio
async def test_limit_in_sql():
    pool = FakePool([])
    await read_campaign_opportunities(pool, limit=25)
    sql = pool.fetch.call_args[0][0]
    assert "LIMIT" in sql


@pytest.mark.asyncio
async def test_min_urgency_passed():
    pool = FakePool([])
    await read_campaign_opportunities(pool, min_urgency=6.5)
    args = pool.fetch.call_args[0]
    assert 6.5 in args


@pytest.mark.asyncio
async def test_suppress_predicate_applied():
    """Adapter applies suppress_predicate to exclude corrected/suppressed reviews."""
    pool = FakePool([])
    with patch(
        "atlas_brain.services.b2b.corrections.suppress_predicate",
        return_value="r.id != 'blocked'",
    ) as mock_sp:
        await read_campaign_opportunities(pool)
    mock_sp.assert_called_once()
    sql = pool.fetch.call_args[0][0]
    assert "r.id != 'blocked'" in sql


@pytest.mark.asyncio
async def test_empty_result():
    pool = FakePool([])
    results = await read_campaign_opportunities(pool)
    assert results == []


@pytest.mark.asyncio
async def test_quotable_phrases_parsed():
    pool = FakePool([_make_db_row(quotable_phrases=json.dumps(["quote1", "quote2"]))])
    results = await read_campaign_opportunities(pool)
    assert results[0]["quotable_phrases"] == ["quote1", "quote2"]


@pytest.mark.asyncio
async def test_integration_stack_parsed():
    pool = FakePool([_make_db_row(integration_stack=json.dumps(["Slack", "Jira"]))])
    results = await read_campaign_opportunities(pool)
    assert results[0]["integration_stack"] == ["Slack", "Jira"]
