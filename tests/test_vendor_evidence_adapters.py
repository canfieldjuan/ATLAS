"""Direct tests for Phase 4 vendor evidence adapters and SQL fragment helpers."""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from atlas_brain.autonomous.tasks._b2b_shared import (
    _competitor_unnest_sql,
    _integration_stack_unnest_sql,
    _vendor_evidence_base_filters,
    read_category_quote_evidence,
    read_vendor_quote_evidence,
)


# ---------------------------------------------------------------------------
# Class 2: SQL fragment helper tests
# ---------------------------------------------------------------------------


def test_base_filters_default_enriched_at():
    sql = _vendor_evidence_base_filters(alias="r", window_param=1)
    assert "r.enrichment_status = 'enriched'" in sql
    assert "r.enriched_at > NOW()" in sql
    assert "COALESCE" not in sql


def test_base_filters_coalesce_recency():
    sql = _vendor_evidence_base_filters(alias="r", window_param=1, recency_column="coalesce")
    assert "COALESCE(r.reviewed_at, r.imported_at, r.enriched_at)" in sql


def test_base_filters_custom_alias():
    sql = _vendor_evidence_base_filters(alias="rv", window_param=3)
    assert "rv.enrichment_status" in sql
    assert "$3" in sql


def test_base_filters_includes_suppress_predicate():
    with patch(
        "atlas_brain.services.b2b.corrections.suppress_predicate",
        return_value="NOT EXISTS (SELECT 1 FROM data_corrections)",
    ):
        sql = _vendor_evidence_base_filters(alias="r", window_param=1)
    assert "data_corrections" in sql


def test_competitor_unnest_sql():
    sql = _competitor_unnest_sql(alias="r")
    assert "jsonb_array_elements" in sql
    assert "competitors_mentioned" in sql
    assert "CROSS JOIN LATERAL" in sql


def test_integration_stack_unnest_sql():
    sql = _integration_stack_unnest_sql(alias="r")
    assert "jsonb_array_elements_text" in sql
    assert "integration_stack" in sql


# ---------------------------------------------------------------------------
# Class 1: Row-level evidence reader tests
# ---------------------------------------------------------------------------


def _make_quote_row(**overrides):
    row = {
        "vendor_name": "Zendesk",
        "source": "g2",
        "reviewer_company": "Acme Corp",
        "reviewer_title": "VP Support",
        "role_level": "vp",
        "pain_category": "pricing",
        "urgency": Decimal("8.5"),
        "review_text": "Way too expensive for what we get.",
        "rating": Decimal("2.0"),
        "quotable_raw": json.dumps(["Way too expensive", "Considering alternatives"]),
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
async def test_vendor_quote_output_shape():
    pool = FakePool([_make_quote_row()])
    results = await read_vendor_quote_evidence(pool, vendor_name="Zendesk")
    assert len(results) == 1
    required = {
        "vendor_name", "source", "reviewer_company", "reviewer_title",
        "role_level", "pain_category", "urgency", "review_text",
        "rating", "quotable_phrases",
    }
    assert set(results[0].keys()) == required


@pytest.mark.asyncio
async def test_vendor_quote_urgency_float():
    pool = FakePool([_make_quote_row(urgency=Decimal("7.3"))])
    results = await read_vendor_quote_evidence(pool, vendor_name="Zendesk")
    assert results[0]["urgency"] == 7.3
    assert isinstance(results[0]["urgency"], float)


@pytest.mark.asyncio
async def test_vendor_quote_null_urgency():
    pool = FakePool([_make_quote_row(urgency=None)])
    results = await read_vendor_quote_evidence(pool, vendor_name="Zendesk")
    assert results[0]["urgency"] is None


@pytest.mark.asyncio
async def test_vendor_quote_phrases_parsed():
    phrases = ["Too expensive", "Switching soon"]
    pool = FakePool([_make_quote_row(quotable_raw=json.dumps(phrases))])
    results = await read_vendor_quote_evidence(pool, vendor_name="Zendesk")
    assert results[0]["quotable_phrases"] == phrases


@pytest.mark.asyncio
async def test_vendor_quote_sources_filter():
    pool = FakePool([])
    await read_vendor_quote_evidence(pool, vendor_name="Zendesk", sources=["g2", "capterra"])
    sql = pool.fetch.call_args[0][0]
    assert "ANY(" in sql


@pytest.mark.asyncio
async def test_vendor_quote_pain_filter():
    pool = FakePool([])
    await read_vendor_quote_evidence(pool, vendor_name="Zendesk", pain_filter="pricing")
    sql = pool.fetch.call_args[0][0]
    assert "pain_categories" in sql
    assert "ILIKE" in sql


@pytest.mark.asyncio
async def test_vendor_quote_min_urgency_in_sql():
    pool = FakePool([])
    await read_vendor_quote_evidence(pool, vendor_name="Zendesk", min_urgency=7.5)
    args = pool.fetch.call_args[0]
    assert 7.5 in args


@pytest.mark.asyncio
async def test_vendor_quote_limit_in_sql():
    pool = FakePool([])
    await read_vendor_quote_evidence(pool, vendor_name="Zendesk", limit=5)
    sql = pool.fetch.call_args[0][0]
    assert "LIMIT" in sql


@pytest.mark.asyncio
async def test_vendor_quote_suppress_applied():
    pool = FakePool([])
    with patch(
        "atlas_brain.services.b2b.corrections.suppress_predicate",
        return_value="r.id != 'blocked'",
    ) as mock_sp:
        await read_vendor_quote_evidence(pool, vendor_name="Zendesk")
    mock_sp.assert_called_once()
    sql = pool.fetch.call_args[0][0]
    assert "r.id != 'blocked'" in sql


@pytest.mark.asyncio
async def test_category_quote_output_shape():
    pool = FakePool([_make_quote_row()])
    results = await read_category_quote_evidence(pool, product_category="Customer Support")
    assert len(results) == 1
    assert set(results[0].keys()) == {
        "vendor_name", "source", "reviewer_company", "reviewer_title",
        "role_level", "pain_category", "urgency", "review_text",
        "rating", "quotable_phrases",
    }


@pytest.mark.asyncio
async def test_category_quote_filters_by_category():
    pool = FakePool([])
    await read_category_quote_evidence(pool, product_category="CRM")
    sql = pool.fetch.call_args[0][0]
    assert "product_category = $2" in sql


@pytest.mark.asyncio
async def test_empty_results():
    pool = FakePool([])
    v = await read_vendor_quote_evidence(pool, vendor_name="X")
    c = await read_category_quote_evidence(pool, product_category="X")
    assert v == []
    assert c == []
