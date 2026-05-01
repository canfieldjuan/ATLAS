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
            _request(b'{"type":"email.delivered","data":{"email_id":"msg-1"}}'),
            provider="resend",
        )

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid webhook signature"


@pytest.mark.asyncio
async def test_campaign_email_webhook_rejects_unknown_provider(monkeypatch):
    from atlas_brain.api import campaign_webhooks as mod

    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.campaign_email_webhook(
            _request(b'{}'),
            provider="not-a-real-esp",
        )

    assert exc.value.status_code == 400
    assert "unknown email webhook provider" in exc.value.detail


@pytest.mark.asyncio
async def test_campaign_email_webhook_returns_501_for_stubbed_provider(monkeypatch):
    """SES provider is registered but not yet implemented; route must
    surface NotImplementedError as HTTP 501 instead of crashing."""
    from atlas_brain.api import campaign_webhooks as mod

    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.campaign_email_webhook(
            _request(b'{"Type":"Notification","Message":"{}"}'),
            provider="ses",
        )

    assert exc.value.status_code == 501
    assert "not yet implemented" in exc.value.detail


@pytest.mark.asyncio
async def test_campaign_email_webhook_returns_no_events_for_invalid_json(monkeypatch):
    """Malformed JSON yields zero canonical events; the route returns 200
    with status='ignored' so the ESP does not retry indefinitely."""
    from atlas_brain.api import campaign_webhooks as mod

    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)
    monkeypatch.setattr(mod.settings.campaign_sequence, "resend_webhook_signing_secret", "")

    result = await mod.campaign_email_webhook(_request(b"{not-json"), provider="resend")

    assert result["status"] == "ignored"
    assert result["reason"] == "no canonical events parsed"


@pytest.mark.asyncio
async def test_campaign_email_webhook_ignores_missing_email_id_before_db_touch(monkeypatch):
    from atlas_brain.api import campaign_webhooks as mod

    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)
    monkeypatch.setattr(mod.settings.campaign_sequence, "resend_webhook_signing_secret", "")

    result = await mod.campaign_email_webhook(
        _request(json.dumps({"type": "email.delivered", "data": {}}).encode("utf-8")),
        provider="resend",
    )

    assert result["status"] == "ignored"
    assert result["reason"] == "no canonical events parsed"


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
        _request(json.dumps({"type": "email.delivered", "data": {"email_id": "msg-1"}}).encode("utf-8")),
        provider="resend",
    )

    pool.fetchrow.assert_awaited_once()
    assert result["status"] == "ok"
    assert result["provider"] == "resend"
    assert result["processed"] == 0
    assert result["skipped"] == 1
    assert result["events"][0]["reason"] == "unknown campaign"


@pytest.mark.asyncio
async def test_resend_provider_signature_skipped_when_no_secret():
    from atlas_brain.services.email_webhooks.resend import ResendProvider

    provider = ResendProvider()
    assert provider.verify_signature(b"body", {}, secret="") is True


@pytest.mark.asyncio
async def test_resend_provider_normalizes_opened_event():
    from atlas_brain.services.email_webhooks.resend import ResendProvider

    provider = ResendProvider()
    payload = json.dumps({
        "type": "email.opened",
        "created_at": "2026-04-30T12:00:00Z",
        "data": {
            "email_id": "msg-9",
            "to": ["alice@example.com"],
        },
    }).encode("utf-8")

    events = provider.normalize_event(payload)
    assert len(events) == 1
    event = events[0]
    assert event.event_type == "opened"
    assert event.message_id == "msg-9"
    assert event.recipient_email == "alice@example.com"
    assert event.provider == "resend"


@pytest.mark.asyncio
async def test_resend_provider_returns_empty_for_unknown_type():
    from atlas_brain.services.email_webhooks.resend import ResendProvider

    provider = ResendProvider()
    payload = json.dumps({
        "type": "email.spammed",
        "data": {"email_id": "msg-9"},
    }).encode("utf-8")
    assert provider.normalize_event(payload) == []


@pytest.mark.asyncio
async def test_provider_registry_resolves_known_names():
    from atlas_brain.services.email_webhooks import resolve

    for name in ("resend", "ses", "sendgrid", "postmark", "mailgun"):
        provider = resolve(name)
        assert provider.name == name


@pytest.mark.asyncio
async def test_provider_registry_defaults_to_resend_when_blank():
    from atlas_brain.services.email_webhooks import resolve

    provider = resolve("")
    assert provider.name == "resend"


@pytest.mark.asyncio
async def test_webhook_lookup_passes_provider_to_query(monkeypatch):
    """Migration 311 added esp_provider to b2b_campaigns; the webhook
    lookup must scope by (esp_message_id, esp_provider) so a Resend
    webhook cannot match an SES-sent campaign that happens to share an
    esp_message_id value once a second provider goes live."""
    from atlas_brain.api import campaign_webhooks as mod

    fetchrow = AsyncMock(return_value=None)
    pool = SimpleNamespace(is_initialized=True, fetchrow=fetchrow)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.campaign_sequence, "resend_webhook_signing_secret", "")

    await mod.campaign_email_webhook(
        _request(json.dumps({
            "type": "email.delivered",
            "data": {"email_id": "msg-77"},
        }).encode("utf-8")),
        provider="resend",
    )

    fetchrow.assert_awaited_once()
    args = fetchrow.await_args.args
    # Args: (sql, message_id, provider)
    assert args[1] == "msg-77"
    assert args[2] == "resend"
    assert "esp_provider" in args[0]
