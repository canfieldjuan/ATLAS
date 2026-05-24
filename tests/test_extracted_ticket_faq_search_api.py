from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from extracted_content_pipeline.api.faq_search import (
    FAQDeflectionSearchApiConfig,
    create_faq_deflection_search_router,
)
from extracted_content_pipeline.campaign_ports import TenantScope


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
