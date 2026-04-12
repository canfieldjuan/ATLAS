import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.requests import Request

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)


def _request(body: bytes, headers: list[tuple[str, str]] | None = None) -> Request:
    encoded_headers = [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in (headers or [])
    ]

    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/webhooks/campaign-email",
            "headers": encoded_headers,
        },
        receive=receive,
    )


@pytest.mark.asyncio
async def test_unsubscribe_rejects_blank_email_before_db_touch(monkeypatch):
    from atlas_brain.api import campaign_webhooks as mod

    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.unsubscribe("   ")

    assert exc.value.status_code == 422
    assert exc.value.detail == "email is required"


@pytest.mark.asyncio
async def test_unsubscribe_trims_email_before_suppression(monkeypatch):
    from atlas_brain.api import campaign_webhooks as mod
    from atlas_brain.autonomous.tasks import campaign_suppression as suppression_mod

    pool = SimpleNamespace(is_initialized=True)
    add_suppression = AsyncMock()

    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(suppression_mod, "add_suppression", add_suppression)

    html = await mod.unsubscribe("  person@example.com  ")

    add_suppression.assert_awaited_once_with(
        pool,
        email="person@example.com",
        reason="unsubscribe",
        source="recipient",
    )
    assert "You have been unsubscribed" in html


@pytest.mark.asyncio
async def test_campaign_email_webhook_rejects_invalid_signature_before_db_touch(monkeypatch):
    from atlas_brain.api import campaign_webhooks as mod

    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)
    monkeypatch.setattr(
        mod.settings.campaign_sequence,
        "resend_webhook_signing_secret",
        "whsec_ZmFrZVNlY3JldA==",
    )

    with pytest.raises(mod.HTTPException) as exc:
        await mod.campaign_email_webhook(
            _request(b'{"type":"email.delivered","data":{"email_id":"msg-1"}}')
        )

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid webhook signature"


@pytest.mark.asyncio
async def test_campaign_email_webhook_rejects_invalid_json_before_db_touch(monkeypatch):
    from atlas_brain.api import campaign_webhooks as mod

    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)
    monkeypatch.setattr(mod.settings.campaign_sequence, "resend_webhook_signing_secret", "")

    with pytest.raises(mod.HTTPException) as exc:
        await mod.campaign_email_webhook(_request(b"{not-json"))

    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid JSON"


@pytest.mark.asyncio
async def test_campaign_email_webhook_ignores_missing_email_id_before_db_touch(monkeypatch):
    from atlas_brain.api import campaign_webhooks as mod

    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)
    monkeypatch.setattr(mod.settings.campaign_sequence, "resend_webhook_signing_secret", "")

    result = await mod.campaign_email_webhook(
        _request(json.dumps({"type": "email.delivered", "data": {}}).encode("utf-8"))
    )

    assert result == {"status": "ignored", "reason": "no email_id"}


@pytest.mark.asyncio
async def test_campaign_email_webhook_uses_db_after_prechecks(monkeypatch):
    from atlas_brain.api import campaign_webhooks as mod

    pool = SimpleNamespace(
        is_initialized=True,
        fetchrow=AsyncMock(return_value=None),
    )
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.campaign_sequence, "resend_webhook_signing_secret", "")

    result = await mod.campaign_email_webhook(
        _request(json.dumps({"type": "email.delivered", "data": {"email_id": "msg-1"}}).encode("utf-8"))
    )

    pool.fetchrow.assert_awaited_once()
    assert result == {"status": "ignored", "reason": "unknown campaign"}
