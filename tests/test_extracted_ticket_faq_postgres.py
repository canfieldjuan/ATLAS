from __future__ import annotations

import json

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.ticket_faq_ports import TicketFAQDraft
from extracted_content_pipeline.ticket_faq_postgres import (
    PostgresTicketFAQRepository,
    backfill_ticket_faq_search_documents,
)


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


class _AsyncContext:
    def __init__(self, value=None, enter=None) -> None:
        self.value = value
        self.enter = enter

    async def __aenter__(self):
        if self.enter is not None:
            self.enter()
        return self.value

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _Connection(_Pool):
    def __init__(self) -> None:
        super().__init__()
        self.transaction_entries = 0

    def transaction(self):
        return _AsyncContext(enter=lambda: setattr(
            self,
            "transaction_entries",
            self.transaction_entries + 1,
        ))


class _AcquirePool(_Pool):
    def __init__(self, connection: _Connection) -> None:
        super().__init__()
        self.connection = connection
        self.acquire_entries = 0

    def acquire(self):
        return _AsyncContext(
            self.connection,
            enter=lambda: setattr(self, "acquire_entries", self.acquire_entries + 1),
        )


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


def _draft_row(
    *,
    draft_id: str = "11111111-1111-1111-1111-111111111111",
    account_id: str = "acct-1",
    status: str = "approved",
) -> dict:
    return {
        "id": draft_id,
        "account_id": account_id,
        "target_id": "ticket-1",
        "target_mode": "vendor_retention",
        "title": "Support FAQ",
        "markdown": "# Support FAQ",
        "items": json.dumps([{
            "rank": 1,
            "topic": "Login access",
            "question": "How do I reset login?",
            "answer": "Use the reset link.",
            "source_ids": ["ticket-1"],
            "ticket_count": 2,
        }]),
        "source_count": 3,
        "ticket_source_count": 2,
        "output_checks": json.dumps({"condensed": True}),
        "warnings": json.dumps([]),
        "metadata": json.dumps({"corpus_id": "corpus-1"}),
        "status": status,
    }


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
    pool.fetch_rows = []
    repo = PostgresTicketFAQRepository(pool)

    updated = await repo.update_status(
        "faq-uuid-1",
        "approved",
        scope=TenantScope(account_id="acct-1"),
    )

    assert updated is False
    assert "UPDATE ticket_faq_markdown" in pool.fetch_calls[0]["query"]
    assert pool.fetch_calls[0]["args"] == ("faq-uuid-1", "approved", "acct-1")
    assert pool.execute_calls == []


@pytest.mark.asyncio
async def test_update_status_replaces_search_projection() -> None:
    pool = _Pool()
    pool.fetch_rows = [_draft_row()]
    repo = PostgresTicketFAQRepository(pool)

    updated = await repo.update_status(
        "11111111-1111-1111-1111-111111111111",
        "approved",
        scope=TenantScope(account_id="acct-1"),
    )

    assert updated is True
    assert "RETURNING id, target_id" in pool.fetch_calls[0]["query"]
    assert "DELETE FROM ticket_faq_search_documents" in pool.execute_calls[0]["query"]
    assert pool.execute_calls[0]["args"] == (
        "acct-1",
        "corpus-1",
        "11111111-1111-1111-1111-111111111111",
    )
    assert "INSERT INTO ticket_faq_search_documents" in pool.execute_calls[1]["query"]
    assert pool.execute_calls[1]["args"][:10] == (
        "acct-1",
        "corpus-1",
        "11111111-1111-1111-1111-111111111111",
        "ticket-1",
        "vendor_retention",
        "approved",
        1,
        "Login access",
        "How do I reset login?",
        "Use the reset link.",
    )


@pytest.mark.asyncio
async def test_update_status_indexes_projection_on_acquired_transaction() -> None:
    connection = _Connection()
    connection.fetch_rows = [_draft_row()]
    pool = _AcquirePool(connection)
    repo = PostgresTicketFAQRepository(pool)

    updated = await repo.update_status(
        "11111111-1111-1111-1111-111111111111",
        "approved",
        scope=TenantScope(account_id="acct-1"),
    )

    assert updated is True
    assert pool.acquire_entries == 1
    assert connection.transaction_entries == 2
    assert pool.fetch_calls == []
    assert "UPDATE ticket_faq_markdown" in connection.fetch_calls[0]["query"]
    assert "INSERT INTO ticket_faq_search_documents" in connection.execute_calls[1]["query"]


@pytest.mark.asyncio
async def test_update_status_clears_stale_projection_for_empty_items() -> None:
    pool = _Pool()
    row = _draft_row()
    row["items"] = json.dumps([])
    pool.fetch_rows = [row]
    repo = PostgresTicketFAQRepository(pool)

    updated = await repo.update_status(
        "11111111-1111-1111-1111-111111111111",
        "rejected",
        scope=TenantScope(account_id="acct-1"),
    )

    assert updated is True
    assert len(pool.execute_calls) == 1
    assert "DELETE FROM ticket_faq_search_documents" in pool.execute_calls[0]["query"]


@pytest.mark.asyncio
async def test_update_status_without_projection_scope_keeps_review_update() -> None:
    pool = _Pool()
    row = _draft_row()
    row["metadata"] = json.dumps({})
    pool.fetch_rows = [row]
    repo = PostgresTicketFAQRepository(pool)

    updated = await repo.update_status(
        "11111111-1111-1111-1111-111111111111",
        "approved",
        scope=TenantScope(),
    )

    assert updated is True
    assert pool.execute_calls == []


@pytest.mark.asyncio
async def test_update_statuses_returns_matched_ids() -> None:
    pool = _Pool()
    pool.fetch_rows = [_draft_row(draft_id="22222222-2222-2222-2222-222222222222")]
    repo = PostgresTicketFAQRepository(pool)

    updated = await repo.update_statuses(
        ["22222222-2222-2222-2222-222222222222", ""],
        "approved",
        scope=TenantScope(account_id="acct-1"),
    )

    assert updated == ("22222222-2222-2222-2222-222222222222",)
    assert "id = ANY($1::uuid[])" in pool.fetch_calls[0]["query"]
    assert pool.fetch_calls[0]["args"] == (
        ["22222222-2222-2222-2222-222222222222"],
        "approved",
        "acct-1",
    )
    assert "INSERT INTO ticket_faq_search_documents" in pool.execute_calls[1]["query"]


@pytest.mark.asyncio
async def test_update_statuses_skips_projection_when_no_rows_match() -> None:
    pool = _Pool()
    pool.fetch_rows = []
    repo = PostgresTicketFAQRepository(pool)

    updated = await repo.update_statuses(
        ["22222222-2222-2222-2222-222222222222"],
        "approved",
        scope=TenantScope(account_id="acct-1"),
    )

    assert updated == ()
    assert pool.execute_calls == []


@pytest.mark.asyncio
async def test_backfill_ticket_faq_search_documents_dry_run_reports_without_writing() -> None:
    pool = _Pool()
    pool.fetch_rows = [_draft_row()]

    result = await backfill_ticket_faq_search_documents(pool)

    assert result.as_dict() == {
        "account_id": None,
        "applied_documents": 0,
        "applied_rows": 0,
        "apply": False,
        "eligible_rows": 1,
        "limit": None,
        "projected_documents": 1,
        "scanned": 1,
        "skipped_missing_key": 0,
        "status": "approved",
    }
    assert "FROM ticket_faq_markdown" in pool.fetch_calls[0]["query"]
    assert pool.fetch_calls[0]["args"] == ("approved",)
    assert pool.execute_calls == []


@pytest.mark.asyncio
async def test_backfill_ticket_faq_search_documents_applies_projection() -> None:
    pool = _Pool()
    pool.fetch_rows = [_draft_row()]

    result = await backfill_ticket_faq_search_documents(pool, apply=True)

    assert result.applied_rows == 1
    assert result.applied_documents == 1
    assert "DELETE FROM ticket_faq_search_documents" in pool.execute_calls[0]["query"]
    assert pool.execute_calls[0]["args"] == (
        "acct-1",
        "corpus-1",
        "11111111-1111-1111-1111-111111111111",
    )
    assert "INSERT INTO ticket_faq_search_documents" in pool.execute_calls[1]["query"]
    assert pool.execute_calls[1]["args"][:6] == (
        "acct-1",
        "corpus-1",
        "11111111-1111-1111-1111-111111111111",
        "ticket-1",
        "vendor_retention",
        "approved",
    )


@pytest.mark.asyncio
async def test_backfill_ticket_faq_search_documents_uses_each_rows_account_id() -> None:
    pool = _Pool()
    pool.fetch_rows = [
        _draft_row(
            draft_id="11111111-1111-1111-1111-111111111111",
            account_id="acct-a",
        ),
        _draft_row(
            draft_id="22222222-2222-2222-2222-222222222222",
            account_id="acct-b",
        ),
    ]

    result = await backfill_ticket_faq_search_documents(pool, apply=True)

    assert result.scanned == 2
    assert result.applied_rows == 2
    assert pool.execute_calls[0]["args"][:3] == (
        "acct-a",
        "corpus-1",
        "11111111-1111-1111-1111-111111111111",
    )
    assert pool.execute_calls[1]["args"][:3] == (
        "acct-a",
        "corpus-1",
        "11111111-1111-1111-1111-111111111111",
    )
    assert pool.execute_calls[2]["args"][:3] == (
        "acct-b",
        "corpus-1",
        "22222222-2222-2222-2222-222222222222",
    )
    assert pool.execute_calls[3]["args"][:3] == (
        "acct-b",
        "corpus-1",
        "22222222-2222-2222-2222-222222222222",
    )


@pytest.mark.asyncio
async def test_backfill_ticket_faq_search_documents_clears_empty_items() -> None:
    pool = _Pool()
    row = _draft_row()
    row["items"] = json.dumps([])
    pool.fetch_rows = [row]

    result = await backfill_ticket_faq_search_documents(pool, apply=True)

    assert result.eligible_rows == 1
    assert result.projected_documents == 0
    assert result.applied_rows == 1
    assert result.applied_documents == 0
    assert len(pool.execute_calls) == 1
    assert "DELETE FROM ticket_faq_search_documents" in pool.execute_calls[0]["query"]


@pytest.mark.asyncio
async def test_backfill_ticket_faq_search_documents_rejects_blank_status() -> None:
    pool = _Pool()

    with pytest.raises(ValueError, match="requires status"):
        await backfill_ticket_faq_search_documents(pool, status="  ")

    assert pool.fetch_calls == []
    assert pool.execute_calls == []


@pytest.mark.asyncio
async def test_backfill_ticket_faq_search_documents_filters_account_and_limit() -> None:
    pool = _Pool()
    pool.fetch_rows = [_draft_row(account_id="acct-2", status="rejected")]

    result = await backfill_ticket_faq_search_documents(
        pool,
        status="rejected",
        account_id="acct-2",
        limit=5,
    )

    assert result.status == "rejected"
    assert result.account_id == "acct-2"
    assert result.limit == 5
    assert "account_id = $2" in pool.fetch_calls[0]["query"]
    assert "LIMIT $3" in pool.fetch_calls[0]["query"]
    assert pool.fetch_calls[0]["args"] == ("rejected", "acct-2", 5)


@pytest.mark.asyncio
async def test_backfill_ticket_faq_search_documents_skips_incomplete_projection_key() -> None:
    pool = _Pool()
    pool.fetch_rows = [_draft_row(account_id="")]

    result = await backfill_ticket_faq_search_documents(pool, apply=True)

    assert result.scanned == 1
    assert result.eligible_rows == 0
    assert result.skipped_missing_key == 1
    assert result.applied_rows == 0
    assert pool.execute_calls == []
