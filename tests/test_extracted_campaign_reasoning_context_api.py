"""Tests for the hosted campaign reasoning context admin API."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from extracted_content_pipeline.api.reasoning_contexts import (
    create_reasoning_context_admin_router,
)
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.campaign_reasoning_postgres import (
    CampaignReasoningContextListResult,
)


class _Pool:
    is_initialized = True


class _Repository:
    def __init__(self) -> None:
        self.list_calls: list[dict[str, Any]] = []
        self.save_calls: list[dict[str, Any]] = []

    async def list_contexts(self, **kwargs: Any) -> CampaignReasoningContextListResult:
        self.list_calls.append(kwargs)
        return CampaignReasoningContextListResult(
            rows=(
                {
                    "id": "ctx-1",
                    "account_id": "acct-1",
                    "target_mode": "vendor_retention",
                    "selectors": ["opp-1", "Acme"],
                    "selector_key": "abc123",
                    "updated_at": "2026-05-15T00:00:00Z",
                    "payload": {"top_theses": [{"summary": "Renewal pressure"}]},
                },
            ),
            limit=kwargs["limit"],
            filters={"target_mode": kwargs.get("target_mode")},
        )

    async def save_context(self, **kwargs: Any) -> str:
        self.save_calls.append(kwargs)
        return "ctx-1"


def _client(
    repository: _Repository,
    *,
    scope: TenantScope | dict[str, Any] | None = TenantScope(account_id="acct-1"),
    dependencies: list[Any] | None = None,
) -> TestClient:
    app = FastAPI()
    pool = _Pool()

    async def scope_provider() -> TenantScope | dict[str, Any] | None:
        return scope

    app.include_router(
        create_reasoning_context_admin_router(
            pool_provider=lambda: pool,
            scope_provider=scope_provider if scope is not None else None,
            dependencies=dependencies,
            repository_factory=lambda pool_arg, table: repository,
        )
    )
    return TestClient(app)


def test_reasoning_context_admin_lists_rows_with_scope_and_filters() -> None:
    repository = _Repository()

    response = _client(repository).get(
        "/campaign-reasoning-contexts?target_mode=vendor_retention&selector=Acme&limit=5"
    )

    assert response.status_code == 200
    assert response.json()["rows"][0]["id"] == "ctx-1"
    call = repository.list_calls[0]
    assert call["scope"].account_id == "acct-1"
    assert call["target_mode"] == "vendor_retention"
    assert call["selectors"] == ("Acme",)
    assert call["limit"] == 5


def test_reasoning_context_admin_upserts_context_with_scoped_account() -> None:
    repository = _Repository()

    response = _client(repository, scope=TenantScope(account_id="acct-scope")).post(
        "/campaign-reasoning-contexts",
        json={
            "account_id": "acct-payload",
            "target_mode": "Vendor_Retention",
            "selectors": ["opp-1"],
            "context": {"top_theses": [{"summary": "Renewal pressure"}]},
        },
    )

    assert response.status_code == 200
    assert response.json()["account_id"] == "acct-scope"
    call = repository.save_calls[0]
    assert call["scope"].account_id == "acct-scope"
    assert call["target_mode"] == "vendor_retention"
    assert call["selectors"] == ("opp-1",)


def test_reasoning_context_admin_upserts_selector_fields_without_scope() -> None:
    repository = _Repository()

    response = _client(repository, scope=None).post(
        "/campaign-reasoning-contexts",
        json={
            "account_id": "acct-payload",
            "target_id": "opp-1",
            "company_name": "Acme",
            "context": {"proof_points": [{"label": "pricing"}]},
        },
    )

    assert response.status_code == 200
    call = repository.save_calls[0]
    assert call["scope"].account_id == "acct-payload"
    assert call["selectors"] == ("opp-1", "Acme")
    assert call["context"]["proof_points"][0]["label"] == "pricing"


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"context": {"top_theses": [{"summary": "Renewal pressure"}]}}, "selectors are required"),
        ({"selectors": ["opp-1"], "target_mode": "vendor_retention"}, "context is required"),
        ({"selectors": ["opp-1"], "contextt": {"typo": True}}, "context is required"),
    ],
)
def test_reasoning_context_admin_rejects_invalid_payloads(
    payload: dict[str, Any],
    detail: str,
) -> None:
    response = _client(_Repository()).post("/campaign-reasoning-contexts", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == detail
