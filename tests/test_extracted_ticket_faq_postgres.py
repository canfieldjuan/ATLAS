from __future__ import annotations

import json

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.ticket_faq_ports import TicketFAQDraft
from extracted_content_pipeline.ticket_faq_postgres import PostgresTicketFAQRepository


class _Pool:
    def __init__(self) -> None:
        self.fetchval_results: list[object] = []
        self.fetch_rows: list[dict] = []
        self.fetchval_calls: list[dict] = []
        self.fetch_calls: list[dict] = []
        self.execute_calls: list[dict] = []
        self.execute_result: object = "UPDATE 1"

    async def fetchval(self, query, *args):
        self.fetchval_calls.append({"query": query, "args": args})
        return self.fetchval_results.pop(0)

    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": query, "args": args})
        return self.fetch_rows

    async def execute(self, query, *args):
        self.execute_calls.append({"query": query, "args": args})
        return self.execute_result


def _draft() -> TicketFAQDraft:
    return TicketFAQDraft(
        target_id="ticket-1",
        target_mode="vendor_retention",
        title="Support FAQ",
        markdown="# Support FAQ\n\n## Question",
        items=({"question": "What changed?", "source_ids": ["ticket-1"]},),
        source_count=3,
        ticket_source_count=2,
        output_checks={"has_action_items": True},
        warnings=({"code": "missing_source_text", "row_index": 2},),
        metadata={"source_types": ["ticket"]},
    )


@pytest.mark.asyncio
async def test_save_drafts_persists_markdown_document_and_returns_ids() -> None:
    pool = _Pool()
    pool.fetchval_results = ["faq-uuid-1"]
    repo = PostgresTicketFAQRepository(pool)

    saved = await repo.save_drafts([_draft()], scope=TenantScope(account_id="acct-1"))

    assert saved == ("faq-uuid-1",)
    query = pool.fetchval_calls[0]["query"]
    args = pool.fetchval_calls[0]["args"]
    assert "INSERT INTO ticket_faq_markdown" in query
    assert args[:5] == (
        "acct-1",
        "ticket-1",
        "vendor_retention",
        "Support FAQ",
        "# Support FAQ\n\n## Question",
    )
    assert json.loads(args[5])[0]["question"] == "What changed?"
    assert args[6:8] == (3, 2)
    assert json.loads(args[8]) == {"has_action_items": True}
    assert json.loads(args[9])[0]["code"] == "missing_source_text"
    metadata = json.loads(args[10])
    assert metadata["source_types"] == ["ticket"]
    assert metadata["scope"]["account_id"] == "acct-1"


@pytest.mark.asyncio
async def test_list_drafts_filters_by_status_target_mode_and_limit() -> None:
    pool = _Pool()
    pool.fetch_rows = [{
        "id": "faq-uuid-1",
        "target_id": "ticket-1",
        "target_mode": "vendor_retention",
        "title": "Support FAQ",
        "markdown": "# Support FAQ",
        "items": json.dumps([{"question": "Q"}]),
        "source_count": 3,
        "ticket_source_count": 2,
        "output_checks": json.dumps({"condensed": True}),
        "warnings": json.dumps([]),
        "metadata": json.dumps({"source_types": ["ticket"]}),
        "status": "draft",
    }]
    repo = PostgresTicketFAQRepository(pool)

    drafts = await repo.list_drafts(
        scope=TenantScope(account_id="acct-1"),
        status="draft",
        target_mode="vendor_retention",
        limit=20,
    )

    assert len(drafts) == 1
    assert drafts[0].id == "faq-uuid-1"
    assert drafts[0].items[0]["question"] == "Q"
    assert drafts[0].output_checks == {"condensed": True}
    query = pool.fetch_calls[0]["query"]
    assert "status = $2" in query
    assert "target_mode = $3" in query
    assert "LIMIT $4" in query
    assert pool.fetch_calls[0]["args"] == ("acct-1", "draft", "vendor_retention", 20)


@pytest.mark.asyncio
async def test_update_status_returns_false_on_miss() -> None:
    pool = _Pool()
    pool.execute_result = "UPDATE 0"
    repo = PostgresTicketFAQRepository(pool)

    updated = await repo.update_status(
        "faq-uuid-1",
        "approved",
        scope=TenantScope(account_id="acct-1"),
    )

    assert updated is False
    assert "UPDATE ticket_faq_markdown" in pool.execute_calls[0]["query"]
    assert pool.execute_calls[0]["args"] == ("faq-uuid-1", "approved", "acct-1")


@pytest.mark.asyncio
async def test_update_statuses_returns_matched_ids() -> None:
    pool = _Pool()
    pool.fetch_rows = [{"id": "faq-uuid-1"}]
    repo = PostgresTicketFAQRepository(pool)

    updated = await repo.update_statuses(
        ["faq-uuid-1", ""],
        "approved",
        scope=TenantScope(account_id="acct-1"),
    )

    assert updated == ("faq-uuid-1",)
    assert "id = ANY($1::uuid[])" in pool.fetch_calls[0]["query"]
    assert pool.fetch_calls[0]["args"] == (["faq-uuid-1"], "approved", "acct-1")
