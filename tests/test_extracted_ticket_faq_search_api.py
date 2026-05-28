from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

pytest.importorskip("fastapi")

from fastapi import Depends, FastAPI, Header
from fastapi.testclient import TestClient

from atlas_brain._content_ops_scope import (
    build_content_ops_scope,
    set_current_auth_user,
)
from extracted_content_pipeline.api.faq_search import (
    FAQDeflectionSearchApiConfig,
    create_faq_deflection_search_router,
)
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.ticket_faq_search import (
    PostgresTicketFAQSearchRepository,
    TicketFAQSearchDocument,
)


class _Pool:
    def __init__(
        self,
        rows=None,
        *,
        initialized: bool = True,
    ) -> None:
        self.rows = list(rows or [])
        self.is_initialized = initialized
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, query, *args):
        self.fetch_calls.append((str(query), args))
        return self.rows


def _row() -> dict[str, object]:
    return {
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


def _faq_row() -> dict[str, object]:
    return {
        "id": "11111111-1111-1111-1111-111111111111",
        "target_id": "support-account-1",
        "target_mode": "support_account",
        "title": "Support FAQ",
        "markdown": "# Support FAQ\n\nUse the reset email.",
        "items": json.dumps([{
            "rank": 1,
            "topic": "password reset",
            "question": "How do I reset my password?",
            "answer": "Use the reset email.",
            "source_ids": ["ticket-1"],
            "ticket_count": 1,
        }]),
        "source_count": 1,
        "ticket_source_count": 1,
        "output_checks": json.dumps({"has_action_steps": True}),
        "warnings": json.dumps([]),
        "metadata": json.dumps({"corpus_id": "corpus-1"}),
        "status": "approved",
    }


def _client(
    pool: _Pool,
    *,
    scope: TenantScope | dict[str, object] | None = None,
    config: FAQDeflectionSearchApiConfig | None = None,
) -> TestClient:
    app = FastAPI()
    app.include_router(
        create_faq_deflection_search_router(
            pool_provider=lambda: pool,
            scope_provider=lambda: scope,
            config=config,
        )
    )
    return TestClient(app)


def test_faq_deflection_search_route_returns_documented_envelope() -> None:
    pool = _Pool(rows=[_row()])

    response = _client(
        pool,
        scope=TenantScope(account_id="acct-1", user_id="user-1"),
    ).get(
        "/content-ops/faq-deflection-search"
        "?q=reset%20email&corpus_id=corpus-1&limit=5"
    )

    assert response.status_code == 200
    assert response.json() == {
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
    sql, args = pool.fetch_calls[0]
    assert "FROM ticket_faq_search_documents" in sql
    assert "account_id = $2" in sql
    assert "corpus_id = $3" in sql
    assert "status = $4" in sql
    assert "LIMIT $5" in sql
    assert args == ("reset email", "acct-1", "corpus-1", "approved", 5)


def test_faq_deflection_search_route_allows_all_statuses_and_caps_limit() -> None:
    pool = _Pool(rows=[])

    response = _client(
        pool,
        scope={"account_id": "acct-1"},
        config=FAQDeflectionSearchApiConfig(max_limit=3, default_limit=2),
    ).get("/content-ops/faq-deflection-search?q=invoice&status=&limit=99")

    assert response.status_code == 200
    sql, args = pool.fetch_calls[0]
    assert "status =" not in sql
    assert "LIMIT $3" in sql
    assert args == ("invoice", "acct-1", 3)


def test_faq_deflection_search_route_rejects_blank_query_before_database() -> None:
    pool = _Pool(rows=[_row()])

    response = _client(pool, scope={"account_id": "acct-1"}).get(
        "/content-ops/faq-deflection-search?q=%20%20%20"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "q is required"
    assert pool.fetch_calls == []


def test_faq_deflection_search_route_rejects_overlong_query_before_database() -> None:
    pool = _Pool(rows=[_row()])

    response = _client(
        pool,
        scope={"account_id": "acct-1"},
        config=FAQDeflectionSearchApiConfig(max_query_chars=5),
    ).get("/content-ops/faq-deflection-search?q=too-long")

    assert response.status_code == 400
    assert response.json()["detail"] == "q must be 5 characters or fewer"
    assert pool.fetch_calls == []


def test_faq_deflection_search_route_requires_tenant_scope_before_database() -> None:
    pool = _Pool(rows=[_row()])

    response = _client(pool, scope=None).get(
        "/content-ops/faq-deflection-search?q=reset"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "account_id is required"
    assert pool.fetch_calls == []


def test_faq_deflection_search_route_reports_unavailable_database() -> None:
    pool = _Pool(rows=[_row()], initialized=False)

    response = _client(pool, scope={"account_id": "acct-1"}).get(
        "/content-ops/faq-deflection-search?q=reset"
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Database unavailable"
    assert pool.fetch_calls == []


def test_faq_deflection_detail_route_returns_full_generated_faq() -> None:
    pool = _Pool(rows=[_faq_row()])

    response = _client(pool, scope={"account_id": "acct-1"}).get(
        "/content-ops/faq-deflection-search/11111111-1111-1111-1111-111111111111"
    )

    assert response.status_code == 200
    assert response.json() == {
        "account_id": "acct-1",
        "id": "11111111-1111-1111-1111-111111111111",
        "target_id": "support-account-1",
        "target_mode": "support_account",
        "title": "Support FAQ",
        "markdown": "# Support FAQ\n\nUse the reset email.",
        "items": [{
            "rank": 1,
            "topic": "password reset",
            "question": "How do I reset my password?",
            "answer": "Use the reset email.",
            "source_ids": ["ticket-1"],
            "ticket_count": 1,
        }],
        "source_count": 1,
        "ticket_source_count": 1,
        "output_checks": {"has_action_steps": True},
        "warnings": [],
        "metadata": {"corpus_id": "corpus-1"},
        "status": "approved",
    }
    sql, args = pool.fetch_calls[0]
    assert "FROM ticket_faq_markdown" in sql
    assert "account_id = $2" in sql
    assert args == ("11111111-1111-1111-1111-111111111111", "acct-1")


def test_faq_deflection_detail_route_requires_tenant_scope_before_database() -> None:
    pool = _Pool(rows=[_faq_row()])

    response = _client(pool, scope=None).get(
        "/content-ops/faq-deflection-search/11111111-1111-1111-1111-111111111111"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "account_id is required"
    assert pool.fetch_calls == []


def test_faq_deflection_detail_route_rejects_malformed_faq_id_before_database() -> None:
    pool = _Pool(rows=[_faq_row()])

    response = _client(pool, scope={"account_id": "acct-1"}).get(
        "/content-ops/faq-deflection-search/not-a-uuid"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "faq_id must be a valid UUID"
    assert pool.fetch_calls == []


def test_faq_deflection_detail_route_returns_404_on_scoped_miss() -> None:
    pool = _Pool(rows=[])

    response = _client(pool, scope={"account_id": "acct-2"}).get(
        "/content-ops/faq-deflection-search/11111111-1111-1111-1111-111111111111"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "FAQ not found"


@pytest.mark.integration
def test_faq_deflection_search_route_queries_real_postgres_projection_and_detail() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("EXTRACTED_DATABASE_URL or DATABASE_URL is required")

    account_a = f"acct-a-{uuid4().hex}"
    account_b = f"acct-b-{uuid4().hex}"
    corpus_id = f"corpus-{uuid4().hex}"
    faq_id = str(uuid4())
    scope_account = {"value": account_a}
    pool_provider = _RoutePoolProvider(asyncpg, database_url)

    _run_async(
        _seed_faq_search_route_projection(
            asyncpg,
            database_url,
            account_id=account_a,
            corpus_id=corpus_id,
            faq_id=faq_id,
        )
    )

    app = FastAPI()
    app.add_event_handler("shutdown", pool_provider.close)
    app.include_router(
        create_faq_deflection_search_router(
            pool_provider=pool_provider,
            scope_provider=lambda: TenantScope(account_id=scope_account["value"]),
        )
    )

    try:
        with TestClient(app) as client:
            response = client.get(
                "/content-ops/faq-deflection-search"
                f"?q=password%20reset&corpus_id={corpus_id}&limit=5"
            )
            detail_response = client.get(
                f"/content-ops/faq-deflection-search/{faq_id}"
            )
            scope_account["value"] = account_b
            cross_tenant_response = client.get(
                "/content-ops/faq-deflection-search"
                f"?q=password%20reset&corpus_id={corpus_id}&limit=5"
            )
            cross_tenant_detail_response = client.get(
                f"/content-ops/faq-deflection-search/{faq_id}"
            )

        assert response.status_code == 200
        assert response.json() == {
            "query": "password reset",
            "count": 1,
            "results": [{
                "account_id": account_a,
                "corpus_id": corpus_id,
                "faq_id": faq_id,
                "target_id": "support-account-1",
                "target_mode": "support_account",
                "status": "approved",
                "rank": 1,
                "topic": "password reset",
                "question": "How do I reset my password?",
                "answer_summary": "Use the password reset email.",
                "source_ids": ["ticket-1"],
                "ticket_count": 1,
                "score": response.json()["results"][0]["score"],
            }],
        }
        assert isinstance(response.json()["results"][0]["score"], int)
        assert response.json()["results"][0]["score"] > 0
        assert cross_tenant_response.status_code == 200
        assert cross_tenant_response.json() == {
            "query": "password reset",
            "results": [],
            "count": 0,
        }
        assert detail_response.status_code == 200
        assert detail_response.json() == {
            "account_id": account_a,
            "target_id": "support-account-1",
            "target_mode": "support_account",
            "title": "Support FAQ",
            "markdown": "# Support FAQ",
            "items": [_postgres_faq_item(corpus_id=corpus_id)],
            "source_count": 1,
            "ticket_source_count": 1,
            "output_checks": {},
            "warnings": [],
            "metadata": {"corpus_id": corpus_id},
            "id": faq_id,
            "status": "approved",
        }
        assert cross_tenant_detail_response.status_code == 404
        assert cross_tenant_detail_response.json()["detail"] == "FAQ not found"
    finally:
        _run_async(
            _cleanup_faq_search_route_projection(
                asyncpg,
                database_url,
                account_ids=(account_a, account_b),
            )
        )


@pytest.mark.integration
def test_faq_deflection_search_route_keeps_concurrent_context_scopes_isolated() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("EXTRACTED_DATABASE_URL or DATABASE_URL is required")

    account_a = f"acct-a-{uuid4().hex}"
    account_b = f"acct-b-{uuid4().hex}"
    corpus_id = f"shared-corpus-{uuid4().hex}"
    faq_a = str(uuid4())
    faq_b = str(uuid4())
    pool_provider = _RoutePoolProvider(asyncpg, database_url)

    _run_async(
        _seed_faq_search_route_projection(
            asyncpg,
            database_url,
            account_id=account_a,
            corpus_id=corpus_id,
            faq_id=faq_a,
            target_id="support-account-a",
        )
    )
    _run_async(
        _seed_faq_search_route_projection(
            asyncpg,
            database_url,
            account_id=account_b,
            corpus_id=corpus_id,
            faq_id=faq_b,
            target_id="support-account-b",
        )
    )

    async def _capture_test_auth_user(
        x_test_account_id: str = Header(...),
    ):
        user = SimpleNamespace(
            account_id=x_test_account_id,
            user_id=f"user-{x_test_account_id}",
        )
        set_current_auth_user(user)
        try:
            yield user
        finally:
            set_current_auth_user(None)

    app = FastAPI()
    app.add_event_handler("shutdown", pool_provider.close)
    app.include_router(
        create_faq_deflection_search_router(
            pool_provider=pool_provider,
            scope_provider=build_content_ops_scope,
            dependencies=[Depends(_capture_test_auth_user)],
        )
    )

    expected = {
        account_a: {"faq_id": faq_a, "target_id": "support-account-a"},
        account_b: {"faq_id": faq_b, "target_id": "support-account-b"},
    }

    try:
        with TestClient(app) as client:
            warmup = client.get(
                "/content-ops/faq-deflection-search"
                f"?q=password%20reset&corpus_id={corpus_id}&limit=5",
                headers={"x-test-account-id": account_a},
            )
            assert warmup.status_code == 200

            def _request(account_id: str) -> dict[str, object]:
                headers = {"x-test-account-id": account_id}
                search = client.get(
                    "/content-ops/faq-deflection-search"
                    f"?q=password%20reset&corpus_id={corpus_id}&limit=5",
                    headers=headers,
                )
                assert search.status_code == 200
                search_payload = search.json()
                first = search_payload["results"][0]
                detail = client.get(
                    f"/content-ops/faq-deflection-search/{first['faq_id']}",
                    headers=headers,
                )
                assert detail.status_code == 200
                return {
                    "account_id": account_id,
                    "search": search_payload,
                    "detail": detail.json(),
                }

            request_accounts = [account_a, account_b] * 6
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                responses = list(executor.map(_request, request_accounts))

        assert len(responses) == 12
        for response in responses:
            account_id = str(response["account_id"])
            search_payload = response["search"]
            detail_payload = response["detail"]
            assert search_payload["count"] == 1
            first = search_payload["results"][0]
            assert first["account_id"] == account_id
            assert first["corpus_id"] == corpus_id
            assert first["faq_id"] == expected[account_id]["faq_id"]
            assert first["target_id"] == expected[account_id]["target_id"]
            assert detail_payload["account_id"] == account_id
            assert detail_payload["id"] == expected[account_id]["faq_id"]
            assert detail_payload["target_id"] == expected[account_id]["target_id"]
            assert detail_payload["items"] == [_postgres_faq_item(corpus_id=corpus_id)]
    finally:
        _run_async(
            _cleanup_faq_search_route_projection(
                asyncpg,
                database_url,
                account_ids=(account_a, account_b),
            )
        )


class _RoutePoolProvider:
    def __init__(self, asyncpg, database_url: str) -> None:
        self.asyncpg = asyncpg
        self.database_url = database_url
        self.pool = None

    async def __call__(self):
        if self.pool is None:
            self.pool = await self.asyncpg.create_pool(self.database_url, min_size=1, max_size=2)
        return self.pool

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None


def _run_async(awaitable):
    import asyncio

    return asyncio.run(awaitable)


async def _seed_faq_search_route_projection(
    asyncpg,
    database_url: str,
    *,
    account_id: str,
    corpus_id: str,
    faq_id: str,
    target_id: str = "support-account-1",
) -> None:
    root = Path(__file__).resolve().parents[1]
    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=2)
    try:
        await pool.execute((root / "atlas_brain/storage/migrations/325_ticket_faq_markdown.sql").read_text())
        await pool.execute((root / "atlas_brain/storage/migrations/327_ticket_faq_search_documents.sql").read_text())
        await pool.execute(
            """
            INSERT INTO ticket_faq_markdown (
                id, account_id, target_id, target_mode, title, markdown,
                items, source_count, ticket_source_count, output_checks,
                warnings, metadata, status
            )
            VALUES (
                $1::uuid, $2, $3, 'support_account',
                'Support FAQ', '# Support FAQ', $4::jsonb, 1, 1,
                '{}'::jsonb, '[]'::jsonb, $5::jsonb, 'approved'
            )
            """,
            faq_id,
            account_id,
            target_id,
            json.dumps([_postgres_faq_item(corpus_id=corpus_id)]),
            json.dumps({"corpus_id": corpus_id}),
        )
        await PostgresTicketFAQSearchRepository(pool).replace_documents([
            TicketFAQSearchDocument(
                account_id=account_id,
                corpus_id=corpus_id,
                faq_id=faq_id,
                target_id=target_id,
                target_mode="support_account",
                status="approved",
                rank=1,
                topic="password reset",
                question="How do I reset my password?",
                answer_summary="Use the password reset email.",
                source_ids=("ticket-1",),
                ticket_count=1,
                search_text="password reset email login support",
            )
        ])
    finally:
        await pool.close()


def _postgres_faq_item(*, corpus_id: str) -> dict[str, object]:
    return {
        "topic": "password reset",
        "question": "How do I reset my password?",
        "answer": "Use the password reset email.",
        "source_ids": [f"{corpus_id}-ticket-1"],
        "ticket_count": 1,
    }


async def _cleanup_faq_search_route_projection(
    asyncpg,
    database_url: str,
    *,
    account_ids: tuple[str, ...],
) -> None:
    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=2)
    try:
        await pool.execute(
            "DELETE FROM ticket_faq_markdown WHERE account_id = ANY($1::text[])",
            list(account_ids),
        )
    finally:
        await pool.close()
