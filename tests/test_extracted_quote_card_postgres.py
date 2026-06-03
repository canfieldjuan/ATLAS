from __future__ import annotations

import json

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.quote_card_ports import QuoteCardDraft
from extracted_content_pipeline.quote_card_postgres import (
    PostgresQuoteCardRepository,
)


class _Pool:
    def __init__(self) -> None:
        self.fetchval_results: list[object] = []
        self.fetch_rows: list[dict[str, object]] = []
        self.fetchval_calls: list[dict[str, object]] = []
        self.fetch_calls: list[dict[str, object]] = []
        self.execute_calls: list[dict[str, object]] = []
        self.execute_result: object = "UPDATE 1"

    async def fetchval(self, query, *args):
        self.fetchval_calls.append({"query": str(query), "args": args})
        return self.fetchval_results.pop(0)

    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": str(query), "args": args})
        return self.fetch_rows

    async def execute(self, query, *args):
        self.execute_calls.append({"query": str(query), "args": args})
        return self.execute_result


def _compact_sql(query: object) -> str:
    return " ".join(str(query).split())


def _draft() -> QuoteCardDraft:
    return QuoteCardDraft(
        target_id="review-1",
        target_mode="vendor_retention",
        theme="customer_proof",
        quote="Pricing became hard to justify after renewal.",
        attribution="Acme Logistics",
        headline="Customer proof for HubSpot",
        supporting_text="Use this quote to frame pricing pressure.",
        source_id="review-1",
        source_type="review",
        company_name="Acme Logistics",
        vendor_name="HubSpot",
        pain_points=("pricing pressure",),
        metadata={"confidence": 0.88},
    )


@pytest.mark.asyncio
async def test_save_drafts_persists_quote_cards_and_returns_ids() -> None:
    pool = _Pool()
    pool.fetchval_results = ["quote-card-uuid-1"]
    repo = PostgresQuoteCardRepository(pool)

    saved = await repo.save_drafts(
        [_draft()],
        scope=TenantScope(account_id="acct-1", user_id="user-1"),
    )

    assert saved == ("quote-card-uuid-1",)
    call = pool.fetchval_calls[0]
    assert "INSERT INTO quote_card_drafts" in call["query"]
    args = call["args"]
    assert args[:12] == (
        "acct-1",
        "review-1",
        "vendor_retention",
        "customer_proof",
        "Pricing became hard to justify after renewal.",
        "Acme Logistics",
        "Customer proof for HubSpot",
        "Use this quote to frame pricing pressure.",
        "review-1",
        "review",
        "Acme Logistics",
        "HubSpot",
    )
    assert json.loads(args[12]) == ["pricing pressure"]
    metadata = json.loads(args[13])
    assert metadata["target_id"] == "review-1"
    assert metadata["target_mode"] == "vendor_retention"
    assert metadata["scope"] == {"account_id": "acct-1", "user_id": "user-1"}
    assert metadata["confidence"] == 0.88


@pytest.mark.asyncio
async def test_list_drafts_filters_by_scope_status_target_mode_and_theme() -> None:
    pool = _Pool()
    pool.fetch_rows = [{
        "id": "quote-card-uuid-1",
        "target_id": "review-1",
        "target_mode": "vendor_retention",
        "theme": "customer_proof",
        "quote": "Pricing became hard to justify after renewal.",
        "attribution": "Acme Logistics",
        "headline": "Customer proof for HubSpot",
        "supporting_text": "Use this quote to frame pricing pressure.",
        "source_id": "review-1",
        "source_type": "review",
        "company_name": "Acme Logistics",
        "vendor_name": "HubSpot",
        "pain_points": json.dumps(["pricing pressure"]),
        "metadata": json.dumps({"confidence": 0.88}),
        "status": "draft",
    }]
    repo = PostgresQuoteCardRepository(pool)

    drafts = await repo.list_drafts(
        scope=TenantScope(account_id="acct-1"),
        status="draft",
        target_mode="vendor_retention",
        theme="customer_proof",
        limit=10,
    )

    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.id == "quote-card-uuid-1"
    assert draft.theme == "customer_proof"
    assert draft.quote == "Pricing became hard to justify after renewal."
    assert draft.pain_points == ("pricing pressure",)
    assert draft.metadata == {"confidence": 0.88}
    call = pool.fetch_calls[0]
    assert "FROM quote_card_drafts" in call["query"]
    assert "account_id = $1" in call["query"]
    assert call["args"] == (
        "acct-1",
        "draft",
        "vendor_retention",
        "customer_proof",
        10,
    )


@pytest.mark.asyncio
async def test_update_status_is_tenant_scoped_and_reports_misses() -> None:
    pool = _Pool()
    pool.execute_result = "UPDATE 0"
    repo = PostgresQuoteCardRepository(pool)

    updated = await repo.update_status(
        "quote-card-uuid-1",
        "approved",
        scope=TenantScope(account_id="acct-1"),
    )

    assert updated is False
    call = pool.execute_calls[0]
    sql = _compact_sql(call["query"])
    assert "UPDATE quote_card_drafts" in sql
    assert "WHERE id = $1::uuid AND account_id = $3" in sql
    assert call["args"] == ("quote-card-uuid-1", "approved", "acct-1")


@pytest.mark.asyncio
async def test_update_statuses_returns_scoped_matches() -> None:
    pool = _Pool()
    pool.fetch_rows = [{"id": "quote-card-uuid-1"}]
    repo = PostgresQuoteCardRepository(pool)

    updated = await repo.update_statuses(
        ["quote-card-uuid-1", ""],
        "rejected",
        scope=TenantScope(account_id="acct-1"),
    )

    assert updated == ("quote-card-uuid-1",)
    call = pool.fetch_calls[0]
    sql = _compact_sql(call["query"])
    assert "UPDATE quote_card_drafts" in sql
    assert "WHERE id = ANY($1::uuid[]) AND account_id = $3" in sql
    assert call["args"] == (["quote-card-uuid-1"], "rejected", "acct-1")
