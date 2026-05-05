from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

import extracted_content_pipeline.api.campaign_operations as operations_api
from extracted_content_pipeline.api.campaign_operations import (
    CampaignOperationsApiConfig,
    create_campaign_operations_router,
)


class _Pool:
    def __init__(self, *, initialized: bool = True) -> None:
        self.is_initialized = initialized


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


def _client(
    pool: Any,
    *,
    sender: Any = None,
    llm: Any = None,
    skills: Any = None,
    config: CampaignOperationsApiConfig | None = None,
    dependencies: list[Any] | None = None,
    counters: dict[str, int] | None = None,
) -> TestClient:
    app = FastAPI()
    counts = counters if counters is not None else {}

    async def pool_provider():
        counts["pool"] = counts.get("pool", 0) + 1
        return pool

    async def sender_provider():
        counts["sender"] = counts.get("sender", 0) + 1
        return sender

    async def llm_provider():
        counts["llm"] = counts.get("llm", 0) + 1
        return llm

    async def skills_provider():
        counts["skills"] = counts.get("skills", 0) + 1
        return skills

    app.include_router(
        create_campaign_operations_router(
            pool_provider=pool_provider,
            sender_provider=sender_provider if sender is not None else None,
            llm_provider=llm_provider if llm is not None else None,
            skills_provider=skills_provider if skills is not None else None,
            config=config,
            dependencies=dependencies,
        )
    )
    return TestClient(app)


def test_campaign_operations_router_sends_queued_campaigns(monkeypatch) -> None:
    pool = _Pool()
    sender = _Sender()
    calls: list[tuple[Any, dict[str, Any]]] = []
    counters: dict[str, int] = {}

    async def _send(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(sent=2, failed=0, suppressed=1, skipped=0)

    monkeypatch.setattr(operations_api, "send_due_campaigns_from_postgres", _send)

    response = _client(
        pool,
        sender=sender,
        counters=counters,
        config=CampaignOperationsApiConfig(
            max_send_limit=60,
            send_default_from_email="audit@example.com",
            send_default_reply_to="reply@example.com",
            send_unsubscribe_base_url="https://example.com/unsubscribe",
            send_unsubscribe_token_secret="secret",
            send_company_address="123 Main St",
        ),
    ).post("/campaigns/operations/send/queued", json={"limit": 7})

    assert response.status_code == 200
    assert response.json() == {
        "sent": 2,
        "failed": 0,
        "suppressed": 1,
        "skipped": 0,
    }
    assert counters == {"pool": 1, "sender": 1}
    assert calls[0][0] is pool
    assert calls[0][1]["sender"] is sender
    assert calls[0][1]["limit"] == 7
    send_config = calls[0][1]["config"]
    assert send_config.default_from_email == "audit@example.com"
    assert send_config.default_reply_to == "reply@example.com"
    assert send_config.unsubscribe_base_url == "https://example.com/unsubscribe"
    assert send_config.unsubscribe_token_secret == "secret"
    assert send_config.company_address == "123 Main St"
    assert send_config.limit == 7


@pytest.mark.parametrize(
    ("payload", "detail"),
    (
        ({"limit": True}, "limit must be an integer"),
        ({"limit": 3.5}, "limit must be an integer"),
        ({"limit": 0}, "limit must be greater than 0"),
        ({"limit": 61}, "limit must be less than or equal to 60"),
    ),
)
def test_campaign_operations_router_rejects_invalid_send_limits(
    monkeypatch,
    payload,
    detail,
) -> None:
    calls = []

    async def _send(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(sent=1)

    monkeypatch.setattr(operations_api, "send_due_campaigns_from_postgres", _send)

    response = _client(
        _Pool(),
        sender=_Sender(),
        config=CampaignOperationsApiConfig(max_send_limit=60),
    ).post("/campaigns/operations/send/queued", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == detail
    assert calls == []


def test_campaign_operations_router_requires_sender_before_send(monkeypatch) -> None:
    calls = []

    async def _send(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(sent=1)

    monkeypatch.setattr(operations_api, "send_due_campaigns_from_postgres", _send)

    response = _client(_Pool()).post("/campaigns/operations/send/queued")

    assert response.status_code == 503
    assert response.json()["detail"] == "Campaign sender unavailable"
    assert calls == []


def test_campaign_operations_router_progresses_sequences(monkeypatch) -> None:
    pool = _Pool()
    llm = _LLM()
    skills = _Skills()
    calls: list[tuple[Any, dict[str, Any]]] = []
    counters: dict[str, int] = {}

    async def _progress(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(due_sequences=3, progressed=2, skipped=1, disabled=False)

    monkeypatch.setattr(
        operations_api,
        "progress_campaign_sequences_from_postgres",
        _progress,
    )

    response = _client(
        pool,
        llm=llm,
        skills=skills,
        counters=counters,
        config=CampaignOperationsApiConfig(
            max_sequence_limit=40,
            max_sequence_steps=9,
            sequence_from_email="audit@example.com",
            sequence_onboarding_product_name="Content Ops",
            sequence_temperature=0.4,
        ),
    ).post(
        "/campaigns/operations/sequences/progress",
        json={"limit": 8, "max_steps": 4},
    )

    assert response.status_code == 200
    assert response.json() == {
        "due_sequences": 3,
        "progressed": 2,
        "skipped": 1,
        "disabled": False,
    }
    assert counters == {"pool": 1, "llm": 1, "skills": 1}
    assert calls[0][0] is pool
    assert calls[0][1]["llm"] is llm
    assert calls[0][1]["skills"] is skills
    sequence_config = calls[0][1]["config"]
    assert sequence_config.batch_limit == 8
    assert sequence_config.max_steps == 4
    assert sequence_config.from_email == "audit@example.com"
    assert sequence_config.onboarding_product_name == "Content Ops"
    assert sequence_config.temperature == 0.4


def test_campaign_operations_router_rejects_invalid_sequence_limits(
    monkeypatch,
) -> None:
    calls = []

    async def _progress(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(progressed=1)

    monkeypatch.setattr(
        operations_api,
        "progress_campaign_sequences_from_postgres",
        _progress,
    )

    response = _client(
        _Pool(),
        config=CampaignOperationsApiConfig(
            default_sequence_limit=10,
            max_sequence_limit=10,
            max_sequence_steps=5,
            sequence_from_email="audit@example.com",
        ),
    ).post(
        "/campaigns/operations/sequences/progress",
        json={"limit": 11, "max_steps": 2},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "limit must be less than or equal to 10"
    assert calls == []

    response = _client(
        _Pool(),
        config=CampaignOperationsApiConfig(
            default_sequence_limit=10,
            max_sequence_limit=10,
            max_sequence_steps=5,
            sequence_from_email="audit@example.com",
        ),
    ).post(
        "/campaigns/operations/sequences/progress",
        json={"limit": 3, "max_steps": 6},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "max_steps must be less than or equal to 5"
    assert calls == []


def test_campaign_operations_router_requires_sequence_from_email(monkeypatch) -> None:
    counters: dict[str, int] = {}
    calls = []

    async def _progress(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(progressed=1)

    monkeypatch.setattr(
        operations_api,
        "progress_campaign_sequences_from_postgres",
        _progress,
    )

    response = _client(
        _Pool(),
        counters=counters,
    ).post("/campaigns/operations/sequences/progress", json={"limit": 3})

    assert response.status_code == 503
    assert response.json()["detail"] == "Campaign sequence from_email is not configured"
    assert counters == {}
    assert calls == []


def test_campaign_operations_router_refreshes_analytics(monkeypatch) -> None:
    pool = _Pool()
    calls = []

    async def _refresh(received_pool):
        calls.append(received_pool)
        return _Result(refreshed=True, error=None)

    monkeypatch.setattr(
        operations_api,
        "refresh_campaign_analytics_from_postgres",
        _refresh,
    )

    response = _client(pool).post("/campaigns/operations/analytics/refresh")

    assert response.status_code == 200
    assert response.json() == {"refreshed": True, "error": None}
    assert calls == [pool]


def test_campaign_operations_router_sanitizes_analytics_errors(monkeypatch) -> None:
    async def _refresh(_pool):
        return _Result(refreshed=False, error="SELECT * FROM private_table failed")

    monkeypatch.setattr(
        operations_api,
        "refresh_campaign_analytics_from_postgres",
        _refresh,
    )

    response = _client(_Pool()).post("/campaigns/operations/analytics/refresh")

    assert response.status_code == 200
    assert response.json() == {
        "refreshed": False,
        "error": operations_api._ANALYTICS_ERROR_SUMMARY,
    }


def test_campaign_operations_router_requires_database() -> None:
    response = _client(_Pool(initialized=False), sender=_Sender()).post(
        "/campaigns/operations/send/queued"
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Database unavailable"


def test_campaign_operations_router_honors_host_dependencies() -> None:
    def require_auth():
        raise HTTPException(status_code=403, detail="forbidden")

    response = _client(
        _Pool(),
        sender=_Sender(),
        dependencies=[Depends(require_auth)],
    ).post("/campaigns/operations/send/queued")

    assert response.status_code == 403


def test_campaign_operations_api_config_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="default_send_limit must be positive"):
        CampaignOperationsApiConfig(default_send_limit=0)

    with pytest.raises(ValueError, match="max_send_limit must be positive"):
        CampaignOperationsApiConfig(max_send_limit=0)

    with pytest.raises(ValueError, match="default_sequence_limit must be less"):
        CampaignOperationsApiConfig(default_sequence_limit=5, max_sequence_limit=4)


def test_campaign_operations_router_requires_fastapi(monkeypatch) -> None:
    monkeypatch.setattr(
        operations_api,
        "_FASTAPI_IMPORT_ERROR",
        ImportError("missing"),
    )

    with pytest.raises(RuntimeError, match="FastAPI is required"):
        create_campaign_operations_router(pool_provider=lambda: None)
