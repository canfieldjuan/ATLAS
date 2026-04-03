"""Direct tests for read_review_details adapter.

Exercises the real adapter with a mock pool returning DB-shaped rows,
verifying column mapping, type coercion, filtering in SQL, and the
full output contract.
"""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from atlas_brain.autonomous.tasks._b2b_shared import read_review_details


def _make_db_row(**overrides):
    """Return a dict mimicking an asyncpg Record from the review details SQL."""
    row = {
        "id": uuid4(),
        "vendor_name": "Zendesk",
        "product_category": "Customer Support",
        "reviewer_company": "Acme Corp",
        "rating": Decimal("3.0"),
        "source": "g2",
        "reviewed_at": None,
        "enriched_at": "2026-03-28T12:00:00+00:00",
        "urgency_score": Decimal("7.5"),
        "pain_category": "pricing",
        "intent_to_leave": True,
        "decision_maker": True,
        "role_level": "vp",
        "buying_stage": "evaluation",
        "sentiment_direction": "declining",
        "industry": "SaaS",
        "reviewer_title": "VP Engineering",
        "company_size_raw": "201-500",
        "content_type": "review",
        "thread_id": None,
        "competitors_raw": json.dumps([{"name": "Freshdesk"}]),
        "quotable_raw": json.dumps(["Too expensive for what you get"]),
        "positive_raw": json.dumps(["Good integrations"]),
        "complaints_raw": json.dumps(["Pricing went up 40%"]),
        "relevance_score": Decimal("0.85"),
        "author_churn_score": Decimal("0.6"),
        "low_fidelity": False,
        "low_fidelity_reasons": None,
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
async def test_output_shape_has_all_required_keys():
    pool = FakePool([_make_db_row()])
    results = await read_review_details(pool, window_days=30)

    assert len(results) == 1
    r = results[0]
    required_keys = {
        "id", "vendor_name", "product_category", "reviewer_company",
        "rating", "source", "reviewed_at", "enriched_at",
        "urgency_score", "pain_category", "intent_to_leave", "decision_maker",
        "role_level", "buying_stage", "sentiment_direction", "industry",
        "reviewer_title", "company_size", "content_type", "thread_id",
        "competitors_mentioned", "quotable_phrases", "positive_aspects",
        "specific_complaints", "relevance_score", "author_churn_score",
        "low_fidelity", "low_fidelity_reasons",
    }
    assert set(r.keys()) == required_keys


@pytest.mark.asyncio
async def test_urgency_is_float():
    pool = FakePool([_make_db_row(urgency_score=Decimal("8.3"))])
    results = await read_review_details(pool, window_days=30)
    assert isinstance(results[0]["urgency_score"], float)
    assert results[0]["urgency_score"] == 8.3


@pytest.mark.asyncio
async def test_null_urgency_defaults_to_zero():
    pool = FakePool([_make_db_row(urgency_score=None)])
    results = await read_review_details(pool, window_days=30)
    assert results[0]["urgency_score"] == 0.0


@pytest.mark.asyncio
async def test_json_fields_parsed():
    pool = FakePool([_make_db_row(
        competitors_raw=json.dumps([{"name": "Intercom"}]),
        quotable_raw=json.dumps(["We switched"]),
        positive_raw=json.dumps(["Fast support"]),
        complaints_raw=json.dumps(["Slow API"]),
    )])
    results = await read_review_details(pool, window_days=30)
    r = results[0]
    assert r["competitors_mentioned"] == [{"name": "Intercom"}]
    assert r["quotable_phrases"] == ["We switched"]
    assert r["positive_aspects"] == ["Fast support"]
    assert r["specific_complaints"] == ["Slow API"]


@pytest.mark.asyncio
async def test_unknown_role_level_becomes_none():
    pool = FakePool([_make_db_row(role_level="unknown")])
    results = await read_review_details(pool, window_days=30)
    assert results[0]["role_level"] is None


@pytest.mark.asyncio
async def test_unknown_buying_stage_becomes_none():
    pool = FakePool([_make_db_row(buying_stage="unknown")])
    results = await read_review_details(pool, window_days=30)
    assert results[0]["buying_stage"] is None


@pytest.mark.asyncio
async def test_review_id_stringified():
    rid = uuid4()
    pool = FakePool([_make_db_row(id=rid)])
    results = await read_review_details(pool, window_days=30)
    assert results[0]["id"] == str(rid)


@pytest.mark.asyncio
async def test_vendor_name_filter_in_sql():
    pool = FakePool([])
    await read_review_details(pool, window_days=30, vendor_name="Zendesk")
    sql = pool.fetch.call_args[0][0]
    assert "ILIKE" in sql


@pytest.mark.asyncio
async def test_scoped_vendors_filter_in_sql():
    pool = FakePool([])
    await read_review_details(pool, window_days=30, scoped_vendors=["Zendesk"])
    sql = pool.fetch.call_args[0][0]
    assert "ANY(" in sql


@pytest.mark.asyncio
async def test_pain_category_filter_in_sql():
    pool = FakePool([])
    await read_review_details(pool, window_days=30, pain_category="pricing")
    sql = pool.fetch.call_args[0][0]
    assert "pain_category" in sql


@pytest.mark.asyncio
async def test_min_urgency_filter_in_sql():
    pool = FakePool([])
    await read_review_details(pool, window_days=30, min_urgency=6.5)
    sql = pool.fetch.call_args[0][0]
    assert "urgency_score" in sql
    # Verify the param is passed as float
    args = pool.fetch.call_args[0]
    assert 6.5 in args


@pytest.mark.asyncio
async def test_company_filter_in_sql():
    pool = FakePool([])
    await read_review_details(pool, window_days=30, company="Acme")
    sql = pool.fetch.call_args[0][0]
    assert "reviewer_company ILIKE" in sql


@pytest.mark.asyncio
async def test_churn_intent_filter_in_sql():
    pool = FakePool([])
    await read_review_details(pool, window_days=30, has_churn_intent=True)
    sql = pool.fetch.call_args[0][0]
    assert "intent_to_leave" in sql


@pytest.mark.asyncio
async def test_min_relevance_filter_in_sql():
    pool = FakePool([])
    await read_review_details(pool, window_days=30, min_relevance=0.5)
    sql = pool.fetch.call_args[0][0]
    assert "relevance_score" in sql


@pytest.mark.asyncio
async def test_exclude_low_fidelity_filter_in_sql():
    pool = FakePool([])
    await read_review_details(pool, window_days=30, exclude_low_fidelity=True)
    sql = pool.fetch.call_args[0][0]
    assert "low_fidelity" in sql


@pytest.mark.asyncio
async def test_limit_in_sql():
    pool = FakePool([])
    await read_review_details(pool, window_days=30, limit=25)
    sql = pool.fetch.call_args[0][0]
    assert "LIMIT" in sql


@pytest.mark.asyncio
async def test_empty_result_set():
    pool = FakePool([])
    results = await read_review_details(pool, window_days=30)
    assert results == []


@pytest.mark.asyncio
async def test_boolean_fields_coerced():
    pool = FakePool([_make_db_row(intent_to_leave=None, decision_maker=None, low_fidelity=None)])
    results = await read_review_details(pool, window_days=30)
    r = results[0]
    assert r["intent_to_leave"] is False
    assert r["decision_maker"] is False
    assert r["low_fidelity"] is False


@pytest.mark.asyncio
async def test_relevance_score_converted_to_float():
    pool = FakePool([_make_db_row(relevance_score=Decimal("0.92"), author_churn_score=None)])
    results = await read_review_details(pool, window_days=30)
    r = results[0]
    assert r["relevance_score"] == 0.92
    assert r["author_churn_score"] is None
