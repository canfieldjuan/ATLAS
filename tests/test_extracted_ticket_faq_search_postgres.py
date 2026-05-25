from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from extracted_content_pipeline.ticket_faq_search import (
    PostgresTicketFAQSearchRepository,
    TicketFAQSearchDocument,
    TicketFAQSearchProjectionKey,
)
from extracted_content_pipeline.ticket_faq_postgres import (
    PostgresTicketFAQRepository,
    backfill_ticket_faq_search_documents,
)
from extracted_content_pipeline.campaign_ports import TenantScope


class _Transaction:
    def __init__(self, pool: _Pool) -> None:
        self.pool = pool

    async def __aenter__(self):
        self.pool.transaction_enters += 1
        return self.pool

    async def __aexit__(self, exc_type, exc, traceback):
        self.pool.transaction_exits.append(exc_type)
        return False


class _Pool:
    def __init__(self) -> None:
        self.execute_calls: list[dict[str, object]] = []
        self.fetch_calls: list[dict[str, object]] = []
        self.fetch_rows: list[dict[str, object]] = []
        self.transaction_enters = 0
        self.transaction_exits: list[type[BaseException] | None] = []
        self.fail_on_insert = False

    async def execute(self, query, *args):
        if self.fail_on_insert and "INSERT INTO ticket_faq_search_documents" in str(query):
            raise RuntimeError("insert failed")
        self.execute_calls.append({"query": query, "args": args})
        return "INSERT 0 1"

    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": query, "args": args})
        return self.fetch_rows

    def transaction(self):
        return _Transaction(self)


def _document(
    *,
    faq_id: str = "11111111-1111-1111-1111-111111111111",
    rank: int = 1,
    account_id: str = "acct-1",
    corpus_id: str = "corpus-1",
    status: str = "approved",
    question: str = "How do I reset my password?",
) -> TicketFAQSearchDocument:
    return TicketFAQSearchDocument(
        account_id=account_id,
        corpus_id=corpus_id,
        faq_id=faq_id,
        target_id="support-account-1",
        target_mode="support_account",
        status=status,
        rank=rank,
        topic="password reset",
        question=question,
        answer_summary="Customers cannot find the password reset email.",
        source_ids=("ticket-1", "ticket-2"),
        ticket_count=2,
        search_text="password reset email login support",
    )


def _projection_key(
    *,
    account_id: str = "acct-1",
    corpus_id: str = "corpus-1",
    faq_id: str = "11111111-1111-1111-1111-111111111111",
) -> TicketFAQSearchProjectionKey:
    return TicketFAQSearchProjectionKey(
        account_id=account_id,
        corpus_id=corpus_id,
        faq_id=faq_id,
    )


@pytest.mark.asyncio
async def test_replace_documents_deletes_existing_projection_then_inserts() -> None:
    pool = _Pool()
    repo = PostgresTicketFAQSearchRepository(pool)

    replaced = await repo.replace_documents([
        _document(rank=1),
        _document(rank=2, question="Where is my reset email?"),
    ])

    assert replaced == 2
    assert pool.transaction_enters == 1
    assert pool.transaction_exits == [None]
    assert len(pool.execute_calls) == 3
    delete_call = pool.execute_calls[0]
    assert "DELETE FROM ticket_faq_search_documents" in str(delete_call["query"])
    assert delete_call["args"] == (
        "acct-1",
        "corpus-1",
        "11111111-1111-1111-1111-111111111111",
    )
    insert_call = pool.execute_calls[1]
    assert "INSERT INTO ticket_faq_search_documents" in str(insert_call["query"])
    assert "ON CONFLICT (account_id, corpus_id, faq_id, rank)" in str(insert_call["query"])
    assert insert_call["args"][:10] == (
        "acct-1",
        "corpus-1",
        "11111111-1111-1111-1111-111111111111",
        "support-account-1",
        "support_account",
        "approved",
        1,
        "password reset",
        "How do I reset my password?",
        "Customers cannot find the password reset email.",
    )
    assert json.loads(insert_call["args"][10]) == ["ticket-1", "ticket-2"]
    assert insert_call["args"][11:] == (2, "password reset email login support")


@pytest.mark.asyncio
async def test_replace_documents_groups_delete_by_account_corpus_and_faq() -> None:
    pool = _Pool()
    repo = PostgresTicketFAQSearchRepository(pool)

    await repo.replace_documents([
        _document(faq_id="11111111-1111-1111-1111-111111111111"),
        _document(faq_id="22222222-2222-2222-2222-222222222222"),
        _document(faq_id="22222222-2222-2222-2222-222222222222", rank=2),
    ])

    delete_args = [
        call["args"]
        for call in pool.execute_calls
        if "DELETE FROM ticket_faq_search_documents" in str(call["query"])
    ]
    assert delete_args == [
        ("acct-1", "corpus-1", "11111111-1111-1111-1111-111111111111"),
        ("acct-1", "corpus-1", "22222222-2222-2222-2222-222222222222"),
    ]


@pytest.mark.asyncio
async def test_replace_documents_can_clear_existing_projection_for_empty_results() -> None:
    pool = _Pool()
    repo = PostgresTicketFAQSearchRepository(pool)

    replaced = await repo.replace_documents(
        [],
        replace_keys=[
            _projection_key()
        ],
    )

    assert replaced == 0
    assert pool.transaction_enters == 1
    assert pool.transaction_exits == [None]
    assert len(pool.execute_calls) == 1
    delete_call = pool.execute_calls[0]
    assert "DELETE FROM ticket_faq_search_documents" in str(delete_call["query"])
    assert delete_call["args"] == (
        "acct-1",
        "corpus-1",
        "11111111-1111-1111-1111-111111111111",
    )


@pytest.mark.asyncio
async def test_replace_documents_uses_transaction_for_delete_and_insert() -> None:
    pool = _Pool()
    pool.fail_on_insert = True
    repo = PostgresTicketFAQSearchRepository(pool)

    with pytest.raises(RuntimeError, match="insert failed"):
        await repo.replace_documents([_document()])

    assert pool.transaction_enters == 1
    assert pool.transaction_exits == [RuntimeError]
    assert len(pool.execute_calls) == 1
    assert "DELETE FROM ticket_faq_search_documents" in str(pool.execute_calls[0]["query"])


@pytest.mark.asyncio
async def test_replace_documents_rejects_duplicate_ranks_before_database_write() -> None:
    pool = _Pool()
    repo = PostgresTicketFAQSearchRepository(pool)

    with pytest.raises(ValueError, match="requires distinct ranks"):
        await repo.replace_documents([
            _document(rank=1),
            _document(rank=1, question="Where is my reset email?"),
        ])

    assert pool.execute_calls == []
    assert pool.transaction_enters == 0


@pytest.mark.asyncio
async def test_replace_documents_rejects_blank_tenant_before_database_write() -> None:
    pool = _Pool()
    repo = PostgresTicketFAQSearchRepository(pool)

    with pytest.raises(ValueError, match="requires account_id"):
        await repo.replace_documents([_document(account_id="   ")])

    assert pool.execute_calls == []


@pytest.mark.asyncio
async def test_search_filters_by_tenant_corpus_status_and_uses_full_text_query() -> None:
    pool = _Pool()
    pool.fetch_rows = [
        {
            "account_id": "acct-1",
            "corpus_id": "corpus-1",
            "faq_id": "11111111-1111-1111-1111-111111111111",
            "target_id": "support-account-1",
            "target_mode": "support_account",
            "status": "approved",
            "rank": 1,
            "topic": "password reset",
            "question": "How do I reset my password?",
            "answer_summary": "Use the reset email.",
            "source_ids": json.dumps(["ticket-1"]),
            "ticket_count": 1,
            "search_text": "password reset email",
            "score": 84,
        }
    ]
    repo = PostgresTicketFAQSearchRepository(pool)

    response = await repo.search(
        query="reset email",
        account_id="acct-1",
        corpus_id="corpus-1",
        status="approved",
        limit=5,
    )

    sql = str(pool.fetch_calls[0]["query"])
    assert "websearch_to_tsquery('english', $1)" in sql
    assert "account_id = $2" in sql
    assert "corpus_id = $3" in sql
    assert "status = $4" in sql
    assert "LIMIT $5" in sql
    assert pool.fetch_calls[0]["args"] == (
        "reset email",
        "acct-1",
        "corpus-1",
        "approved",
        5,
    )
    assert response.as_dict() == {
        "query": "reset email",
        "count": 1,
        "results": [{
            "account_id": "acct-1",
            "corpus_id": "corpus-1",
            "faq_id": "11111111-1111-1111-1111-111111111111",
            "target_id": "support-account-1",
            "target_mode": "support_account",
            "status": "approved",
            "rank": 1,
            "topic": "password reset",
            "question": "How do I reset my password?",
            "answer_summary": "Use the reset email.",
            "source_ids": ["ticket-1"],
            "ticket_count": 1,
            "score": 84,
        }],
    }


@pytest.mark.asyncio
async def test_search_allows_optional_corpus_and_status_filters() -> None:
    pool = _Pool()
    repo = PostgresTicketFAQSearchRepository(pool)

    await repo.search(
        query="invoice",
        account_id="acct-1",
        corpus_id=None,
        status=None,
        limit=10,
    )

    sql = str(pool.fetch_calls[0]["query"])
    assert "account_id = $2" in sql
    assert "corpus_id =" not in sql
    assert "status =" not in sql
    assert "LIMIT $3" in sql
    assert pool.fetch_calls[0]["args"] == ("invoice", "acct-1", 10)


@pytest.mark.asyncio
async def test_search_short_circuits_blank_query_and_zero_limit() -> None:
    pool = _Pool()
    repo = PostgresTicketFAQSearchRepository(pool)

    blank = await repo.search(query="   ", account_id="acct-1")
    limited = await repo.search(query="password", account_id="acct-1", limit=0)
    blank_tenant = await repo.search(query="password", account_id="   ")

    assert blank.as_dict() == {"query": "", "results": [], "count": 0}
    assert limited.as_dict() == {"query": "password", "results": [], "count": 0}
    assert blank_tenant.as_dict() == {"query": "password", "results": [], "count": 0}
    assert pool.fetch_calls == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ticket_faq_search_contract_against_postgres() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("EXTRACTED_DATABASE_URL or DATABASE_URL is required")

    root = Path(__file__).resolve().parents[1]
    faq_id_a = str(uuid4())
    faq_id_b = str(uuid4())
    faq_id_rollback = str(uuid4())
    account_a = f"acct-a-{uuid4().hex}"
    account_b = f"acct-b-{uuid4().hex}"
    account_rollback = f"acct-rb-{uuid4().hex}"
    corpus_id = f"corpus-{uuid4().hex}"
    function_name = f"test_fail_ticket_faq_insert_{uuid4().hex}"
    trigger_name = f"test_fail_ticket_faq_insert_{uuid4().hex}"
    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=2)
    repo = PostgresTicketFAQSearchRepository(pool)

    try:
        await pool.execute((root / "atlas_brain/storage/migrations/325_ticket_faq_markdown.sql").read_text())
        await pool.execute((root / "atlas_brain/storage/migrations/327_ticket_faq_search_documents.sql").read_text())
        for faq_id, account_id in (
            (faq_id_a, account_a),
            (faq_id_b, account_b),
            (faq_id_rollback, account_rollback),
        ):
            await _insert_parent_faq(pool, faq_id=faq_id, account_id=account_id)

        await repo.replace_documents([
            _document(
                faq_id=faq_id_a,
                account_id=account_a,
                corpus_id=corpus_id,
                question="How do I reset my password?",
            ),
            _document(
                faq_id=faq_id_b,
                account_id=account_b,
                corpus_id=corpus_id,
                question="How do I reset my password?",
            ),
        ])

        account_a_hit = await repo.search(
            query="password reset",
            account_id=account_a,
            corpus_id=corpus_id,
        )
        account_b_hit = await repo.search(
            query="password reset",
            account_id=account_b,
            corpus_id=corpus_id,
        )
        miss = await repo.search(
            query="escrow shortage",
            account_id=account_a,
            corpus_id=corpus_id,
        )

        assert [row["account_id"] for row in account_a_hit.as_dict()["results"]] == [account_a]
        assert [row["account_id"] for row in account_b_hit.as_dict()["results"]] == [account_b]
        assert miss.as_dict()["results"] == []

        await repo.replace_documents([
            _document(
                faq_id=faq_id_rollback,
                account_id=account_rollback,
                corpus_id=corpus_id,
                question="How do I update my billing address?",
            )
        ])
        await pool.execute(
            f"""
            CREATE OR REPLACE FUNCTION {function_name}()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            BEGIN
                IF NEW.account_id = '{account_rollback}' THEN
                    RAISE EXCEPTION 'forced ticket FAQ search projection insert failure';
                END IF;
                RETURN NEW;
            END;
            $$
            """
        )
        await pool.execute(
            f"""
            CREATE TRIGGER {trigger_name}
            BEFORE INSERT ON ticket_faq_search_documents
            FOR EACH ROW
            EXECUTE FUNCTION {function_name}()
            """
        )

        with pytest.raises(Exception, match="forced ticket FAQ search projection insert failure"):
            await repo.replace_documents([
                _document(
                    faq_id=faq_id_rollback,
                    account_id=account_rollback,
                    corpus_id=corpus_id,
                    question="How do I change my billing address?",
                )
            ])

        rollback_count = await pool.fetchval(
            """
            SELECT COUNT(*)
              FROM ticket_faq_search_documents
             WHERE account_id = $1
               AND corpus_id = $2
               AND faq_id = $3::uuid
               AND question = $4
            """,
            account_rollback,
            corpus_id,
            faq_id_rollback,
            "How do I update my billing address?",
        )
        assert rollback_count == 1
    finally:
        await pool.execute(f"DROP TRIGGER IF EXISTS {trigger_name} ON ticket_faq_search_documents")
        await pool.execute(f"DROP FUNCTION IF EXISTS {function_name}()")
        await pool.execute(
            "DELETE FROM ticket_faq_markdown WHERE account_id = ANY($1::text[])",
            [account_a, account_b, account_rollback],
        )
        await pool.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ticket_faq_review_status_indexes_search_projection_against_postgres() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("EXTRACTED_DATABASE_URL or DATABASE_URL is required")

    root = Path(__file__).resolve().parents[1]
    faq_id = str(uuid4())
    account_a = f"acct-a-{uuid4().hex}"
    account_b = f"acct-b-{uuid4().hex}"
    corpus_id = f"corpus-{uuid4().hex}"
    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=2)
    faq_repo = PostgresTicketFAQRepository(pool)
    search_repo = PostgresTicketFAQSearchRepository(pool)

    try:
        await pool.execute((root / "atlas_brain/storage/migrations/325_ticket_faq_markdown.sql").read_text())
        await pool.execute((root / "atlas_brain/storage/migrations/327_ticket_faq_search_documents.sql").read_text())
        await _insert_reviewable_faq(
            pool,
            faq_id=faq_id,
            account_id=account_a,
            corpus_id=corpus_id,
        )

        approved = await faq_repo.update_status(
            faq_id,
            "approved",
            scope=TenantScope(account_id=account_a),
        )
        account_a_hit = await search_repo.search(
            query="password reset",
            account_id=account_a,
            corpus_id=corpus_id,
        )
        account_b_miss = await search_repo.search(
            query="password reset",
            account_id=account_b,
            corpus_id=corpus_id,
        )
        account_b_update = await faq_repo.update_status(
            faq_id,
            "approved",
            scope=TenantScope(account_id=account_b),
        )
        rejected = await faq_repo.update_status(
            faq_id,
            "rejected",
            scope=TenantScope(account_id=account_a),
        )
        approved_after_reject = await search_repo.search(
            query="password reset",
            account_id=account_a,
            corpus_id=corpus_id,
        )
        rejected_after_reject = await search_repo.search(
            query="password reset",
            account_id=account_a,
            corpus_id=corpus_id,
            status="rejected",
        )

        assert approved is True
        assert account_a_hit.as_dict()["count"] == 1
        assert account_b_miss.as_dict()["count"] == 0
        assert account_b_update is False
        assert rejected is True
        assert approved_after_reject.as_dict()["count"] == 0
        assert rejected_after_reject.as_dict()["count"] == 1
        assert rejected_after_reject.as_dict()["results"][0]["status"] == "rejected"
    finally:
        await pool.execute(
            "DELETE FROM ticket_faq_markdown WHERE account_id = ANY($1::text[])",
            [account_a, account_b],
        )
        await pool.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ticket_faq_search_backfill_projects_each_account_against_postgres() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("EXTRACTED_DATABASE_URL or DATABASE_URL is required")

    root = Path(__file__).resolve().parents[1]
    run_id = uuid4().hex
    status = f"approved_backfill_{run_id}"
    account_a = f"acct-a-{run_id}"
    account_b = f"acct-b-{run_id}"
    corpus_a = f"corpus-a-{run_id}"
    corpus_b = f"corpus-b-{run_id}"
    faq_a = str(uuid4())
    faq_b = str(uuid4())
    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=2)
    search_repo = PostgresTicketFAQSearchRepository(pool)

    try:
        await pool.execute((root / "atlas_brain/storage/migrations/325_ticket_faq_markdown.sql").read_text())
        await pool.execute((root / "atlas_brain/storage/migrations/327_ticket_faq_search_documents.sql").read_text())
        await _insert_reviewable_faq(
            pool,
            faq_id=faq_a,
            account_id=account_a,
            corpus_id=corpus_a,
            status=status,
        )
        await _insert_reviewable_faq(
            pool,
            faq_id=faq_b,
            account_id=account_b,
            corpus_id=corpus_b,
            status=status,
        )

        dry_run = await backfill_ticket_faq_search_documents(pool, status=status)
        dry_run_count = await pool.fetchval(
            "SELECT COUNT(*) FROM ticket_faq_search_documents WHERE status = $1",
            status,
        )
        applied = await backfill_ticket_faq_search_documents(pool, status=status, apply=True)
        account_a_hit = await search_repo.search(
            query="password reset",
            account_id=account_a,
            corpus_id=corpus_a,
            status=status,
        )
        account_a_cross_miss = await search_repo.search(
            query="password reset",
            account_id=account_a,
            corpus_id=corpus_b,
            status=status,
        )
        account_b_hit = await search_repo.search(
            query="password reset",
            account_id=account_b,
            corpus_id=corpus_b,
            status=status,
        )

        assert dry_run.scanned == 2
        assert dry_run.applied_rows == 0
        assert dry_run_count == 0
        assert applied.scanned == 2
        assert applied.applied_rows == 2
        assert applied.applied_documents == 2
        assert account_a_hit.as_dict()["count"] == 1
        assert account_a_cross_miss.as_dict()["count"] == 0
        assert account_b_hit.as_dict()["count"] == 1
        assert account_a_hit.as_dict()["results"][0]["account_id"] == account_a
        assert account_b_hit.as_dict()["results"][0]["account_id"] == account_b
    finally:
        await pool.execute(
            "DELETE FROM ticket_faq_markdown WHERE account_id = ANY($1::text[])",
            [account_a, account_b],
        )
        await pool.close()


async def _insert_parent_faq(pool, *, faq_id: str, account_id: str) -> None:
    await pool.execute(
        """
        INSERT INTO ticket_faq_markdown (
            id, account_id, target_id, target_mode, title, markdown,
            items, source_count, ticket_source_count, output_checks,
            warnings, metadata, status
        )
        VALUES (
            $1::uuid, $2, 'support-account-1', 'support_account',
            'Support FAQ', '# Support FAQ', '[]'::jsonb, 1, 1,
            '{}'::jsonb, '[]'::jsonb, '{}'::jsonb, 'approved'
        )
        """,
        faq_id,
        account_id,
    )


async def _insert_reviewable_faq(
    pool,
    *,
    faq_id: str,
    account_id: str,
    corpus_id: str,
    status: str = "draft",
) -> None:
    await pool.execute(
        """
        INSERT INTO ticket_faq_markdown (
            id, account_id, target_id, target_mode, title, markdown,
            items, source_count, ticket_source_count, output_checks,
            warnings, metadata, status
        )
        VALUES (
            $1::uuid, $2, 'support-account-1', 'support_account',
            'Support FAQ', '# Support FAQ', $3::jsonb, 1, 1,
            '{}'::jsonb, '[]'::jsonb, $4::jsonb, $5
        )
        """,
        faq_id,
        account_id,
        json.dumps([{
            "rank": 1,
            "topic": "password reset",
            "question": "How do I reset my password?",
            "answer": "Use the reset link.",
            "source_ids": ["ticket-1"],
            "ticket_count": 1,
        }]),
        json.dumps({"corpus_id": corpus_id}),
        status,
    )
