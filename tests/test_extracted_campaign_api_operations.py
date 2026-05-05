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
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.services.single_pass_reasoning_provider import (
    SinglePassCampaignReasoningProvider,
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


class _Reasoning:
    pass


class _Visibility:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.fail:
            raise RuntimeError("visibility down")
        self.events.append((event_type, dict(payload)))


def _client(
    pool: Any,
    *,
    sender: Any = None,
    scope: Any = None,
    llm: Any = None,
    skills: Any = None,
    reasoning: Any = None,
    visibility: Any = None,
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

    async def scope_provider():
        counts["scope"] = counts.get("scope", 0) + 1
        return scope

    async def llm_provider():
        counts["llm"] = counts.get("llm", 0) + 1
        return llm

    async def skills_provider():
        counts["skills"] = counts.get("skills", 0) + 1
        return skills

    async def reasoning_context_provider():
        counts["reasoning"] = counts.get("reasoning", 0) + 1
        return reasoning

    async def visibility_provider():
        counts["visibility"] = counts.get("visibility", 0) + 1
        return visibility

    app.include_router(
        create_campaign_operations_router(
            pool_provider=pool_provider,
            sender_provider=sender_provider if sender is not None else None,
            scope_provider=scope_provider if scope is not None else None,
            llm_provider=llm_provider if llm is not None else None,
            skills_provider=skills_provider if skills is not None else None,
            reasoning_context_provider=(
                reasoning_context_provider if reasoning is not None else None
            ),
            visibility_provider=visibility_provider if visibility is not None else None,
            config=config,
            dependencies=dependencies,
        )
    )
    return TestClient(app)


def test_campaign_operations_router_generates_campaign_drafts(monkeypatch) -> None:
    pool = _Pool()
    llm = _LLM()
    skills = _Skills()
    reasoning = _Reasoning()
    calls: list[tuple[Any, dict[str, Any]]] = []
    counters: dict[str, int] = {}

    async def _generate(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(
            requested=2,
            generated=2,
            skipped=0,
            saved_ids=["campaign-1", "campaign-2"],
            errors=[],
        )

    monkeypatch.setattr(
        operations_api,
        "generate_campaign_drafts_from_postgres",
        _generate,
    )

    response = _client(
        pool,
        scope=TenantScope(account_id="acct_1", user_id="user_1"),
        llm=llm,
        skills=skills,
        reasoning=reasoning,
        counters=counters,
        config=CampaignOperationsApiConfig(
            max_generation_limit=50,
            generation_opportunity_table="opps",
            generation_vendor_targets_table="targets",
            generation_skill_name="digest/custom_campaign_generation",
            generation_max_tokens=900,
            generation_temperature=0.3,
        ),
    ).post(
        "/campaigns/operations/drafts/generate",
        json={
            "account_id": "acct_1",
            "target_mode": "vendor_retention",
            "channel": "email",
            "channels": ["email_cold", "email_followup"],
            "filters": {"vendor_name": "HubSpot"},
            "limit": 2,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "requested": 2,
        "generated": 2,
        "skipped": 0,
        "saved_ids": ["campaign-1", "campaign-2"],
        "errors": [],
    }
    assert counters == {
        "scope": 1,
        "pool": 1,
        "llm": 1,
        "skills": 1,
        "reasoning": 1,
    }
    assert calls[0][0] is pool
    kwargs = calls[0][1]
    assert kwargs["scope"] == TenantScope(account_id="acct_1", user_id="user_1")
    assert kwargs["target_mode"] == "vendor_retention"
    assert kwargs["channel"] == "email"
    assert kwargs["channels"] == ("email_cold", "email_followup")
    assert kwargs["limit"] == 2
    assert kwargs["filters"] == {"vendor_name": "HubSpot"}
    assert kwargs["llm"] is llm
    assert kwargs["skills"] is skills
    assert kwargs["reasoning_context"] is reasoning
    assert kwargs["opportunity_table"] == "opps"
    assert kwargs["vendor_targets_table"] == "targets"
    config = kwargs["config"]
    assert config.skill_name == "digest/custom_campaign_generation"
    assert config.max_tokens == 900
    assert config.temperature == 0.3


def test_campaign_operations_router_emits_visibility_for_generation(monkeypatch) -> None:
    visibility = _Visibility()

    async def _generate(_pool, **_kwargs):
        return _Result(
            requested=2,
            generated=2,
            skipped=0,
            saved_ids=["campaign-1", "campaign-2"],
            errors=[],
        )

    monkeypatch.setattr(
        operations_api,
        "generate_campaign_drafts_from_postgres",
        _generate,
    )

    response = _client(
        _Pool(),
        scope=TenantScope(account_id="acct_1"),
        visibility=visibility,
    ).post(
        "/campaigns/operations/drafts/generate",
        json={
            "account_id": "acct_1",
            "target_mode": "vendor_retention",
            "channel": "email",
            "channels": ["email_cold", "email_followup"],
            "limit": 2,
        },
    )

    assert response.status_code == 200
    assert visibility.events == [
        (
            operations_api._OPERATION_STARTED_EVENT,
            {
                "operation": "draft_generation",
                "limit": 2,
                "target_mode": "vendor_retention",
                "channel": "email",
                "channels": ["email_cold", "email_followup"],
                "account_id": "acct_1",
            },
        ),
        (
            operations_api._OPERATION_COMPLETED_EVENT,
            {
                "operation": "draft_generation",
                "limit": 2,
                "target_mode": "vendor_retention",
                "channel": "email",
                "channels": ["email_cold", "email_followup"],
                "account_id": "acct_1",
                "result": {
                    "requested": 2,
                    "generated": 2,
                    "skipped": 0,
                    "saved_ids_count": 2,
                    "error_count": 0,
                },
            },
        ),
    ]


def test_campaign_operations_router_reports_status() -> None:
    counters: dict[str, int] = {}
    response = _client(
        _Pool(),
        sender=_Sender(),
        llm=_LLM(),
        skills=_Skills(),
        reasoning=_Reasoning(),
        visibility=_Visibility(),
        counters=counters,
        config=CampaignOperationsApiConfig(
            default_generation_limit=7,
            max_generation_limit=70,
            generation_target_mode="vendor_retention",
            generation_channel="email",
            generation_channels=("email_cold", "email_followup"),
            default_send_limit=8,
            max_send_limit=80,
            default_sequence_limit=9,
            max_sequence_limit=90,
            default_sequence_max_steps=4,
            max_sequence_steps=12,
            sequence_from_email="audit@example.com",
        ),
    ).get("/campaigns/operations/status")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "database": {"configured": True, "available": True},
        "providers": {
            "database": True,
            "sender": True,
            "llm": True,
            "skills": True,
            "reasoning": True,
            "visibility": True,
        },
        "reasoning": {
            "mode": "explicit_provider",
            "single_pass_configured": False,
            "single_pass_ready": False,
        },
        "features": {
            "draft_generation": True,
            "send_queued": True,
            "sequence_progression": True,
            "analytics_refresh": True,
        },
        "limits": {
            "generation": {
                "default_limit": 7,
                "max_limit": 70,
                "target_mode": "vendor_retention",
                "channel": "email",
                "channels": ["email_cold", "email_followup"],
            },
            "send": {"default_limit": 8, "max_limit": 80},
            "sequence": {
                "default_limit": 9,
                "max_limit": 90,
                "default_max_steps": 4,
                "max_steps": 12,
            },
        },
    }
    assert counters == {"pool": 1}


def test_campaign_operations_router_status_reports_degraded_database() -> None:
    response = _client(_Pool(initialized=False)).get("/campaigns/operations/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["database"] == {
        "configured": True,
        "available": False,
        "reason": "pool_uninitialized",
    }
    assert payload["features"] == {
        "draft_generation": False,
        "send_queued": False,
        "sequence_progression": False,
        "analytics_refresh": False,
    }


def test_campaign_operations_router_status_reports_single_pass_readiness() -> None:
    response = _client(
        _Pool(),
        config=CampaignOperationsApiConfig(generation_single_pass_reasoning=True),
    ).get("/campaigns/operations/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["reasoning"] == {
        "mode": "single_pass",
        "single_pass_configured": True,
        "single_pass_ready": False,
    }
    assert payload["features"]["draft_generation"] is False


def test_campaign_operations_router_status_marks_single_pass_ready() -> None:
    response = _client(
        _Pool(),
        llm=_LLM(),
        skills=_Skills(),
        config=CampaignOperationsApiConfig(generation_single_pass_reasoning=True),
    ).get("/campaigns/operations/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["reasoning"] == {
        "mode": "single_pass",
        "single_pass_configured": True,
        "single_pass_ready": True,
    }
    assert payload["features"]["draft_generation"] is True


def test_campaign_operations_router_status_treats_explicit_reasoning_as_ready() -> None:
    response = _client(
        _Pool(),
        reasoning=_Reasoning(),
        config=CampaignOperationsApiConfig(generation_single_pass_reasoning=True),
    ).get("/campaigns/operations/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["reasoning"] == {
        "mode": "explicit_provider",
        "single_pass_configured": True,
        "single_pass_ready": False,
    }
    assert payload["features"]["draft_generation"] is True


def test_campaign_operations_router_generates_with_payload_scope(monkeypatch) -> None:
    calls = []

    async def _generate(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(requested=1, generated=1, skipped=0, saved_ids=[], errors=[])

    monkeypatch.setattr(
        operations_api,
        "generate_campaign_drafts_from_postgres",
        _generate,
    )

    response = _client(_Pool()).post(
        "/campaigns/operations/drafts/generate",
        json={"account_id": "acct_1", "user_id": "user_1"},
    )

    assert response.status_code == 200
    assert calls[0][1]["scope"] == {"account_id": "acct_1", "user_id": "user_1"}


def test_campaign_operations_router_builds_single_pass_reasoning(monkeypatch) -> None:
    pool = _Pool()
    llm = _LLM()
    skills = _Skills()
    calls: list[tuple[Any, dict[str, Any]]] = []

    async def _generate(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(requested=1, generated=1, skipped=0, saved_ids=[], errors=[])

    monkeypatch.setattr(
        operations_api,
        "generate_campaign_drafts_from_postgres",
        _generate,
    )

    response = _client(
        pool,
        llm=llm,
        skills=skills,
        config=CampaignOperationsApiConfig(
            generation_single_pass_reasoning=True,
            generation_reasoning_skill_name="digest/custom_reasoning",
            generation_reasoning_max_tokens=321,
            generation_reasoning_temperature=0.3,
            generation_reasoning_include_source_opportunity=False,
        ),
    ).post("/campaigns/operations/drafts/generate", json={"limit": 1})

    assert response.status_code == 200
    provider = calls[0][1]["reasoning_context"]
    assert isinstance(provider, SinglePassCampaignReasoningProvider)
    assert provider.llm is llm
    assert provider.skills is skills
    assert provider.config.skill_name == "digest/custom_reasoning"
    assert provider.config.max_tokens == 321
    assert provider.config.temperature == 0.3
    assert provider.config.include_source_opportunity is False


def test_campaign_operations_router_prefers_explicit_reasoning_provider(
    monkeypatch,
) -> None:
    reasoning = _Reasoning()
    calls = []

    async def _generate(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(requested=1, generated=1, skipped=0, saved_ids=[], errors=[])

    monkeypatch.setattr(
        operations_api,
        "generate_campaign_drafts_from_postgres",
        _generate,
    )

    response = _client(
        _Pool(),
        reasoning=reasoning,
        config=CampaignOperationsApiConfig(generation_single_pass_reasoning=True),
    ).post("/campaigns/operations/drafts/generate", json={"limit": 1})

    assert response.status_code == 200
    assert calls[0][1]["reasoning_context"] is reasoning


def test_campaign_operations_router_rejects_single_pass_without_llm(
    monkeypatch,
) -> None:
    calls = []

    async def _generate(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(generated=1)

    monkeypatch.setattr(
        operations_api,
        "generate_campaign_drafts_from_postgres",
        _generate,
    )

    response = _client(
        _Pool(),
        skills=_Skills(),
        config=CampaignOperationsApiConfig(generation_single_pass_reasoning=True),
    ).post("/campaigns/operations/drafts/generate", json={"limit": 1})

    assert response.status_code == 503
    assert response.json()["detail"] == "Campaign reasoning LLM unavailable"
    assert calls == []


def test_campaign_operations_router_rejects_single_pass_without_skills(
    monkeypatch,
) -> None:
    calls = []

    async def _generate(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(generated=1)

    monkeypatch.setattr(
        operations_api,
        "generate_campaign_drafts_from_postgres",
        _generate,
    )

    response = _client(
        _Pool(),
        llm=_LLM(),
        config=CampaignOperationsApiConfig(generation_single_pass_reasoning=True),
    ).post("/campaigns/operations/drafts/generate", json={"limit": 1})

    assert response.status_code == 503
    assert response.json()["detail"] == "Campaign reasoning skills unavailable"
    assert calls == []


def test_campaign_operations_router_maps_single_pass_config_errors(
    monkeypatch,
) -> None:
    calls = []

    async def _generate(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(generated=1)

    monkeypatch.setattr(
        operations_api,
        "generate_campaign_drafts_from_postgres",
        _generate,
    )

    response = _client(
        _Pool(),
        llm=_LLM(),
        skills=_Skills(),
        config=CampaignOperationsApiConfig(
            generation_single_pass_reasoning=True,
            generation_reasoning_temperature="not-a-number",  # type: ignore[arg-type]
        ),
    ).post("/campaigns/operations/drafts/generate", json={"limit": 1})

    assert response.status_code == 400
    assert "not-a-number" in response.json()["detail"]
    assert calls == []


def test_campaign_operations_router_rejects_generation_scope_mismatch(
    monkeypatch,
) -> None:
    calls = []

    async def _generate(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(generated=1)

    monkeypatch.setattr(
        operations_api,
        "generate_campaign_drafts_from_postgres",
        _generate,
    )

    response = _client(
        _Pool(),
        scope={"account_id": "acct_1"},
    ).post(
        "/campaigns/operations/drafts/generate",
        json={"account_id": "acct_2"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "account_id does not match scope"
    assert calls == []


@pytest.mark.parametrize(
    ("payload", "detail"),
    (
        ({"limit": True}, "limit must be an integer"),
        ({"limit": 0}, "limit must be greater than 0"),
        ({"limit": 51}, "limit must be less than or equal to 50"),
        ({"filters": ["vendor_name", "HubSpot"]}, "filters must be an object"),
        ({"channels": {"name": "email"}}, "channels must be a list or string"),
    ),
)
def test_campaign_operations_router_rejects_invalid_generation_payload(
    monkeypatch,
    payload,
    detail,
) -> None:
    calls = []

    async def _generate(received_pool, **kwargs):
        calls.append((received_pool, kwargs))
        return _Result(generated=1)

    monkeypatch.setattr(
        operations_api,
        "generate_campaign_drafts_from_postgres",
        _generate,
    )

    response = _client(
        _Pool(),
        config=CampaignOperationsApiConfig(
            default_generation_limit=50,
            max_generation_limit=50,
        ),
    ).post("/campaigns/operations/drafts/generate", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == detail
    assert calls == []


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


def test_campaign_operations_router_emits_visibility_for_send(monkeypatch) -> None:
    visibility = _Visibility()

    async def _send(_pool, **_kwargs):
        return _Result(sent=2, failed=0, suppressed=1, skipped=0)

    monkeypatch.setattr(operations_api, "send_due_campaigns_from_postgres", _send)

    response = _client(
        _Pool(),
        sender=_Sender(),
        visibility=visibility,
    ).post("/campaigns/operations/send/queued", json={"limit": 7})

    assert response.status_code == 200
    assert visibility.events == [
        (
            operations_api._OPERATION_STARTED_EVENT,
            {"operation": "send_queued", "limit": 7},
        ),
        (
            operations_api._OPERATION_COMPLETED_EVENT,
            {
                "operation": "send_queued",
                "limit": 7,
                "result": {
                    "sent": 2,
                    "failed": 0,
                    "suppressed": 1,
                    "skipped": 0,
                },
            },
        ),
    ]


def test_campaign_operations_router_visibility_failures_do_not_break_send(
    monkeypatch,
) -> None:
    async def _send(_pool, **_kwargs):
        return _Result(sent=1, failed=0)

    monkeypatch.setattr(operations_api, "send_due_campaigns_from_postgres", _send)

    response = _client(
        _Pool(),
        sender=_Sender(),
        visibility=_Visibility(fail=True),
    ).post("/campaigns/operations/send/queued", json={"limit": 1})

    assert response.status_code == 200
    assert response.json() == {"sent": 1, "failed": 0}


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


def test_campaign_operations_router_emits_visibility_for_sequence_progression(
    monkeypatch,
) -> None:
    visibility = _Visibility()

    async def _progress(_pool, **_kwargs):
        return _Result(due_sequences=3, progressed=2, skipped=1, disabled=False)

    monkeypatch.setattr(
        operations_api,
        "progress_campaign_sequences_from_postgres",
        _progress,
    )

    response = _client(
        _Pool(),
        visibility=visibility,
        config=CampaignOperationsApiConfig(sequence_from_email="audit@example.com"),
    ).post(
        "/campaigns/operations/sequences/progress",
        json={"limit": 8, "max_steps": 4},
    )

    assert response.status_code == 200
    assert visibility.events == [
        (
            operations_api._OPERATION_STARTED_EVENT,
            {"operation": "sequence_progression", "limit": 8, "max_steps": 4},
        ),
        (
            operations_api._OPERATION_COMPLETED_EVENT,
            {
                "operation": "sequence_progression",
                "limit": 8,
                "max_steps": 4,
                "result": {
                    "due_sequences": 3,
                    "progressed": 2,
                    "skipped": 1,
                    "disabled": False,
                },
            },
        ),
    ]


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


def test_campaign_operations_router_emits_visibility_for_analytics(monkeypatch) -> None:
    visibility = _Visibility()

    async def _refresh(_pool):
        return _Result(refreshed=True, error=None)

    monkeypatch.setattr(
        operations_api,
        "refresh_campaign_analytics_from_postgres",
        _refresh,
    )

    response = _client(
        _Pool(),
        visibility=visibility,
    ).post("/campaigns/operations/analytics/refresh")

    assert response.status_code == 200
    assert visibility.events == [
        (
            operations_api._OPERATION_STARTED_EVENT,
            {"operation": "analytics_refresh"},
        ),
        (
            operations_api._OPERATION_COMPLETED_EVENT,
            {
                "operation": "analytics_refresh",
                "result": {"refreshed": True, "error": None},
            },
        ),
    ]


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


def test_campaign_operations_router_emits_failed_visibility_for_analytics_error(
    monkeypatch,
) -> None:
    visibility = _Visibility()

    async def _refresh(_pool):
        return _Result(refreshed=False, error="SELECT * FROM private_table failed")

    monkeypatch.setattr(
        operations_api,
        "refresh_campaign_analytics_from_postgres",
        _refresh,
    )

    response = _client(
        _Pool(),
        visibility=visibility,
    ).post("/campaigns/operations/analytics/refresh")

    assert response.status_code == 200
    assert response.json() == {
        "refreshed": False,
        "error": operations_api._ANALYTICS_ERROR_SUMMARY,
    }
    assert visibility.events == [
        (
            operations_api._OPERATION_STARTED_EVENT,
            {"operation": "analytics_refresh"},
        ),
        (
            operations_api._OPERATION_FAILED_EVENT,
            {
                "operation": "analytics_refresh",
                "error_type": operations_api._ANALYTICS_REPORTED_ERROR_TYPE,
                "result": {
                    "refreshed": False,
                },
            },
        ),
    ]


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
