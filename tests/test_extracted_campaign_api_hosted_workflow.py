from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

import extracted_content_pipeline.api.b2b_campaigns as b2b_api
from extracted_content_pipeline.api.b2b_campaigns import create_b2b_campaign_router
import extracted_content_pipeline.api.campaign_operations as operations_api
from extracted_content_pipeline.api.campaign_operations import (
    CampaignOperationsApiConfig,
    create_campaign_operations_router,
)
from extracted_content_pipeline.campaign_ports import TenantScope


CAMPAIGN_ID = "00000000-0000-0000-0000-000000000011"


class _Pool:
    is_initialized = True


class _Result:
    def __init__(self, **values: Any) -> None:
        self.values = values

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)


class _Sender:
    pass


class _LLM:
    pass


class _Skills:
    pass


class _Reasoning:
    pass


def _host_client(
    *,
    pool: Any,
    sender: Any,
    llm: Any,
    skills: Any,
    reasoning: Any,
    scope: TenantScope,
    counters: dict[str, int],
) -> TestClient:
    app = FastAPI()

    async def pool_provider() -> Any:
        counters["pool"] = counters.get("pool", 0) + 1
        return pool

    async def scope_provider() -> TenantScope:
        counters["scope"] = counters.get("scope", 0) + 1
        return scope

    async def sender_provider() -> Any:
        counters["sender"] = counters.get("sender", 0) + 1
        return sender

    async def llm_provider() -> Any:
        counters["llm"] = counters.get("llm", 0) + 1
        return llm

    async def skills_provider() -> Any:
        counters["skills"] = counters.get("skills", 0) + 1
        return skills

    async def reasoning_context_provider() -> Any:
        counters["reasoning"] = counters.get("reasoning", 0) + 1
        return reasoning

    def require_content_ops_admin() -> None:
        counters["auth"] = counters.get("auth", 0) + 1

    dependencies = [Depends(require_content_ops_admin)]
    app.include_router(
        create_campaign_operations_router(
            pool_provider=pool_provider,
            sender_provider=sender_provider,
            scope_provider=scope_provider,
            llm_provider=llm_provider,
            skills_provider=skills_provider,
            reasoning_context_provider=reasoning_context_provider,
            config=CampaignOperationsApiConfig(
                max_generation_limit=20,
                max_send_limit=20,
                send_default_from_email="audit@example.com",
                send_company_address="123 Customer St",
                sequence_from_email="audit@example.com",
            ),
            dependencies=dependencies,
        )
    )
    app.include_router(
        create_b2b_campaign_router(
            pool_provider=pool_provider,
            scope_provider=scope_provider,
            dependencies=dependencies,
        )
    )
    return TestClient(app)


def test_hosted_campaign_api_workflow_uses_shared_scope_and_provider_ports(
    monkeypatch,
) -> None:
    pool = _Pool()
    sender = _Sender()
    llm = _LLM()
    skills = _Skills()
    reasoning = _Reasoning()
    scope = TenantScope(account_id="acct_1", user_id="user_1")
    counters: dict[str, int] = {}
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _generate(received_pool, **kwargs):
        calls.append(("generate", {"pool": received_pool, **kwargs}))
        return _Result(
            requested=1,
            generated=1,
            skipped=0,
            saved_ids=[CAMPAIGN_ID],
            errors=[],
        )

    async def _list(received_pool, **kwargs):
        calls.append(("list", {"pool": received_pool, **kwargs}))
        return _Result(
            count=1,
            limit=5,
            rows=[
                {
                    "id": CAMPAIGN_ID,
                    "status": "draft",
                    "company_name": "Acme",
                    "vendor_name": "LegacyCRM",
                }
            ],
        )

    async def _review(received_pool, **kwargs):
        calls.append(("review", {"pool": received_pool, **kwargs}))
        return _Result(updated=1, status="queued", dry_run=False)

    async def _send(received_pool, **kwargs):
        calls.append(("send", {"pool": received_pool, **kwargs}))
        return _Result(sent=1, failed=0, suppressed=0, skipped=0)

    async def _refresh(received_pool):
        calls.append(("analytics", {"pool": received_pool}))
        return _Result(refreshed=True, error=None)

    monkeypatch.setattr(
        operations_api,
        "generate_campaign_drafts_from_postgres",
        _generate,
    )
    monkeypatch.setattr(b2b_api, "list_campaign_drafts", _list)
    monkeypatch.setattr(b2b_api, "review_campaign_drafts", _review)
    monkeypatch.setattr(operations_api, "send_due_campaigns_from_postgres", _send)
    monkeypatch.setattr(
        operations_api,
        "refresh_campaign_analytics_from_postgres",
        _refresh,
    )

    client = _host_client(
        pool=pool,
        sender=sender,
        llm=llm,
        skills=skills,
        reasoning=reasoning,
        scope=scope,
        counters=counters,
    )

    generate_response = client.post(
        "/campaigns/operations/drafts/generate",
        json={
            "account_id": "acct_1",
            "target_mode": "vendor_retention",
            "channel": "email",
            "channels": ["email_cold", "email_followup"],
            "filters": {"vendor_name": "LegacyCRM"},
            "limit": 1,
        },
    )
    list_response = client.get("/b2b/campaigns/drafts?statuses=draft&limit=5")
    review_response = client.post(
        "/b2b/campaigns/drafts/review",
        json={
            "campaign_ids": [CAMPAIGN_ID],
            "status": "queued",
            "from_statuses": ["draft"],
            "from_email": "audit@example.com",
            "reviewed_by": "ops@example.com",
        },
    )
    send_response = client.post("/campaigns/operations/send/queued", json={"limit": 1})
    analytics_response = client.post("/campaigns/operations/analytics/refresh")

    assert generate_response.status_code == 200
    assert list_response.status_code == 200
    assert review_response.status_code == 200
    assert send_response.status_code == 200
    assert analytics_response.status_code == 200
    assert [name for name, _ in calls] == [
        "generate",
        "list",
        "review",
        "send",
        "analytics",
    ]
    assert generate_response.json()["saved_ids"] == [CAMPAIGN_ID]
    assert list_response.json()["rows"][0]["status"] == "draft"
    assert review_response.json()["status"] == "queued"
    assert send_response.json()["sent"] == 1
    assert analytics_response.json()["refreshed"] is True

    for _, call in calls:
        assert call["pool"] is pool
    assert calls[0][1]["scope"] == scope
    assert calls[0][1]["llm"] is llm
    assert calls[0][1]["skills"] is skills
    assert calls[0][1]["reasoning_context"] is reasoning
    assert calls[1][1]["scope"] == scope
    assert calls[2][1]["scope"] == scope
    assert calls[3][1]["sender"] is sender
    assert counters == {
        "auth": 5,
        "scope": 3,
        "pool": 5,
        "llm": 1,
        "skills": 1,
        "reasoning": 1,
        "sender": 1,
    }


def test_hosted_campaign_api_workflow_keeps_host_auth_before_product_calls(
    monkeypatch,
) -> None:
    pool = _Pool()
    calls: list[str] = []

    async def pool_provider() -> Any:
        calls.append("pool")
        return pool

    def require_content_ops_admin() -> None:
        raise HTTPException(status_code=403, detail="forbidden")

    app = FastAPI()
    app.include_router(
        create_campaign_operations_router(
            pool_provider=pool_provider,
            dependencies=[Depends(require_content_ops_admin)],
        )
    )

    response = TestClient(app).post("/campaigns/operations/analytics/refresh")

    assert response.status_code == 403
    assert response.json()["detail"] == "forbidden"
    assert calls == []
