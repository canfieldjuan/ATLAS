from __future__ import annotations

import base64
import hashlib
import hmac
import json

import pytest

pytest.importorskip("fastapi")

from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from extracted_content_pipeline.api.campaign_webhooks import (
    CampaignWebhookApiConfig,
    create_campaign_webhook_router,
)


def _secret() -> str:
    return "whsec_" + base64.b64encode(b"secret").decode("utf-8")


def _body(event_type: str = "email.opened", **data_overrides) -> bytes:
    data = {
        "email_id": "email_1",
        "to": "Buyer@Example.com",
        "created_at": "2026-05-01T12:00:00Z",
    }
    data.update(data_overrides)
    return json.dumps({"type": event_type, "data": data}).encode("utf-8")


def _headers(body: bytes, *, secret: str | None = None, msg_id: str = "msg_1"):
    secret_text = secret or _secret()
    raw_secret = secret_text[6:] if secret_text.startswith("whsec_") else secret_text
    secret_bytes = base64.b64decode(raw_secret)
    timestamp = "1714550400"
    to_sign = f"{msg_id}.{timestamp}.".encode("utf-8") + body
    signature = base64.b64encode(
        hmac.new(secret_bytes, to_sign, hashlib.sha256).digest()
    ).decode("utf-8")
    return {
        "svix-id": msg_id,
        "svix-timestamp": timestamp,
        "svix-signature": f"v1,{signature}",
    }


def _token_verifier(email: str, token: str) -> bool:
    return email == "buyer@example.com" and token == "token_1"


class _Pool:
    def __init__(self, *, initialized: bool = True) -> None:
        self.is_initialized = initialized
        self.execute_calls: list[dict[str, object]] = []

    async def execute(self, query, *args):
        self.execute_calls.append({"query": str(query), "args": args})
        return "OK"


def _client(
    pool,
    *,
    secret: str | None = None,
    config: CampaignWebhookApiConfig | None = None,
    dependencies=None,
    unsubscribe_token_verifier=None,
) -> TestClient:
    app = FastAPI()

    async def pool_provider():
        return pool

    def signing_secret_provider():
        return _secret() if secret is None else secret

    app.include_router(
        create_campaign_webhook_router(
            pool_provider=pool_provider,
            signing_secret_provider=signing_secret_provider,
            unsubscribe_token_verifier=unsubscribe_token_verifier,
            config=config,
            dependencies=dependencies,
        )
    )
    return TestClient(app)


def test_campaign_webhook_router_ingests_signed_resend_event() -> None:
    pool = _Pool()
    body = _body("email.opened")

    response = _client(pool).post(
        "/webhooks/campaign-email",
        content=body,
        headers=_headers(body),
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "event_type": "opened",
        "message_id": "email_1",
        "reason": None,
        "suppressed": False,
    }
    assert any(
        "UPDATE b2b_campaigns" in call["query"]
        for call in pool.execute_calls
    )
    assert sum(
        "INSERT INTO campaign_audit_log" in call["query"]
        for call in pool.execute_calls
    ) == 2


def test_campaign_webhook_router_unsubscribe_records_suppression() -> None:
    pool = _Pool()

    response = _client(
        pool,
        unsubscribe_token_verifier=_token_verifier,
    ).get("/webhooks/unsubscribe?email=Buyer@Example.com&token=token_1")

    assert response.status_code == 200
    assert "You have been unsubscribed" in response.text
    assert len(pool.execute_calls) == 1
    assert "INSERT INTO campaign_suppressions" in pool.execute_calls[0]["query"]
    assert pool.execute_calls[0]["args"][:3] == (
        "buyer@example.com",
        "unsubscribe",
        "recipient",
    )


def test_campaign_webhook_router_one_click_post_records_suppression() -> None:
    pool = _Pool()

    response = _client(
        pool,
        unsubscribe_token_verifier=_token_verifier,
    ).post("/webhooks/unsubscribe?email=buyer@example.com&token=token_1")

    assert response.status_code == 200
    assert len(pool.execute_calls) == 1
    assert "INSERT INTO campaign_suppressions" in pool.execute_calls[0]["query"]


def test_campaign_webhook_router_rejects_unsubscribe_without_token() -> None:
    pool = _Pool()

    response = _client(
        pool,
        unsubscribe_token_verifier=_token_verifier,
    ).get("/webhooks/unsubscribe?email=buyer@example.com")

    assert response.status_code == 401
    assert response.json()["detail"] == "Unsubscribe token is required"
    assert pool.execute_calls == []


def test_campaign_webhook_router_rejects_invalid_unsubscribe_token() -> None:
    pool = _Pool()

    response = _client(
        pool,
        unsubscribe_token_verifier=_token_verifier,
    ).get("/webhooks/unsubscribe?email=buyer@example.com&token=wrong")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid unsubscribe token"
    assert pool.execute_calls == []


def test_campaign_webhook_router_requires_unsubscribe_token_verifier() -> None:
    pool = _Pool()

    response = _client(pool).get(
        "/webhooks/unsubscribe?email=buyer@example.com&token=token_1"
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Unsubscribe token verifier is not configured"
    assert pool.execute_calls == []


def test_campaign_webhook_router_can_disable_unsubscribe_token_requirement() -> None:
    pool = _Pool()

    response = _client(
        pool,
        config=CampaignWebhookApiConfig(require_unsubscribe_token=False),
    ).get("/webhooks/unsubscribe?email=buyer@example.com")

    assert response.status_code == 200
    assert len(pool.execute_calls) == 1


def test_campaign_webhook_router_rejects_invalid_signature() -> None:
    pool = _Pool()
    body = _body("email.opened")

    response = _client(pool).post("/webhooks/campaign-email", content=body)

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid webhook signature"
    assert pool.execute_calls == []


def test_campaign_webhook_router_rejects_invalid_json() -> None:
    pool = _Pool()
    body = b"{not-json"

    response = _client(
        pool,
        config=CampaignWebhookApiConfig(verify_signatures=False),
    ).post("/webhooks/campaign-email", content=body)

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid JSON"
    assert pool.execute_calls == []


def test_campaign_webhook_router_requires_database() -> None:
    body = _body("email.opened")

    response = _client(_Pool(initialized=False)).post(
        "/webhooks/campaign-email",
        content=body,
        headers=_headers(body),
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Database unavailable"


def test_campaign_webhook_router_rejects_unsupported_provider() -> None:
    pool = _Pool()
    body = _body("email.opened")

    response = _client(pool).post(
        "/webhooks/campaign-email?provider=sendgrid",
        content=body,
        headers=_headers(body),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported webhook provider: sendgrid"
    assert pool.execute_calls == []


def test_campaign_webhook_router_requires_secret_when_verifying() -> None:
    pool = _Pool()
    body = _body("email.opened")

    response = _client(pool, secret="").post(
        "/webhooks/campaign-email",
        content=body,
        headers=_headers(body),
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Webhook signing secret is not configured"
    assert pool.execute_calls == []


def test_campaign_webhook_api_config_requires_positive_soft_bounce_days() -> None:
    with pytest.raises(ValueError, match="soft_bounce_suppression_days"):
        CampaignWebhookApiConfig(soft_bounce_suppression_days=0)


def test_campaign_webhook_router_accepts_host_dependencies() -> None:
    pool = _Pool()

    def require_host_auth():
        raise HTTPException(status_code=403, detail="host auth required")

    response = _client(pool, dependencies=[Depends(require_host_auth)]).get(
        "/webhooks/unsubscribe?email=buyer@example.com"
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "host auth required"
    assert pool.execute_calls == []
