import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas_brain.services.b2b import webhook_dispatcher


@pytest.mark.asyncio
async def test_validate_webhook_destination_rejects_non_global_literal_when_disabled():
    ok, error = await webhook_dispatcher.validate_webhook_destination(
        "http://127.0.0.1:8080/hook",
        SimpleNamespace(allow_non_global_destinations=False),
        resolve_dns=False,
    )

    assert ok is False
    assert error == "Webhook destination cannot target non-global address: 127.0.0.1"


@pytest.mark.asyncio
async def test_validate_webhook_destination_allows_non_global_literal_when_configured():
    ok, error = await webhook_dispatcher.validate_webhook_destination(
        "http://127.0.0.1:8080/hook",
        SimpleNamespace(allow_non_global_destinations=True),
        resolve_dns=True,
    )

    assert ok is True
    assert error is None


@pytest.mark.asyncio
async def test_validate_webhook_destination_blocks_public_hostname_resolving_to_non_global_address():
    with patch.object(
        webhook_dispatcher,
        "_resolve_webhook_destination_ip_strings",
        AsyncMock(return_value={"127.0.0.1", "203.0.113.10"}),
    ):
        ok, error = await webhook_dispatcher.validate_webhook_destination(
            "https://hooks.example.com/churn",
            SimpleNamespace(allow_non_global_destinations=False),
            resolve_dns=True,
        )

    assert ok is False
    assert error == "Webhook destination cannot target non-global address: 127.0.0.1"


@pytest.mark.asyncio
async def test_validate_webhook_destination_rejects_embedded_credentials():
    ok, error = await webhook_dispatcher.validate_webhook_destination(
        "https://user:pass@hooks.example.com/churn",
        SimpleNamespace(allow_non_global_destinations=False),
        resolve_dns=False,
    )

    assert ok is False
    assert error == "Webhook URL must not include embedded credentials"


@pytest.mark.asyncio
async def test_deliver_single_logs_blocked_destination_without_http_request():
    pool = MagicMock()
    pool.execute = AsyncMock(return_value=None)
    sub = {
        "id": "sub-1",
        "url": "http://127.0.0.1:8080/hook",
        "secret": "atlas-secret",
        "channel": "generic",
        "auth_header": None,
    }
    envelope = {
        "event": "signal_update",
        "vendor": "Acme",
        "data": {"company_name": "Acme Bank"},
    }
    cfg = SimpleNamespace(
        timeout_seconds=10,
        max_retries=3,
        retry_delay_seconds=1,
        allow_non_global_destinations=False,
    )

    with patch("atlas_brain.services.b2b.webhook_dispatcher.httpx.AsyncClient") as client_cls:
        ok = await webhook_dispatcher._deliver_single(
            pool,
            sub,
            "signal_update",
            envelope,
            json.dumps(envelope).encode(),
            cfg,
        )

    assert ok is False
    client_cls.assert_not_called()
    assert pool.execute.await_count == 1
    args = pool.execute.await_args.args
    assert "INSERT INTO b2b_webhook_delivery_log" in args[0]
    assert args[1] == "sub-1"
    assert args[2] == "signal_update"
    assert args[6] == 0
    assert args[7] == 1
    assert args[8] is False
    assert args[9] == "Webhook destination cannot target non-global address: 127.0.0.1"


@pytest.mark.asyncio
async def test_deliver_single_logs_embedded_credentials_error_without_http_request():
    pool = MagicMock()
    pool.execute = AsyncMock(return_value=None)
    sub = {
        "id": "sub-credentials",
        "url": "https://user:pass@hooks.example.com/churn",
        "secret": "atlas-secret",
        "channel": "generic",
        "auth_header": None,
    }
    envelope = {
        "event": "signal_update",
        "vendor": "Acme",
        "data": {"company_name": "Acme Bank"},
    }
    cfg = SimpleNamespace(
        timeout_seconds=10,
        max_retries=3,
        retry_delay_seconds=1,
        allow_non_global_destinations=False,
    )

    with patch("atlas_brain.services.b2b.webhook_dispatcher.httpx.AsyncClient") as client_cls:
        ok = await webhook_dispatcher._deliver_single(
            pool,
            sub,
            "signal_update",
            envelope,
            json.dumps(envelope).encode(),
            cfg,
        )

    assert ok is False
    client_cls.assert_not_called()
    args = pool.execute.await_args.args
    assert args[1] == "sub-credentials"
    assert args[9] == "Webhook URL must not include embedded credentials"


@pytest.mark.asyncio
async def test_send_test_webhook_returns_destination_error_when_blocked():
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        return_value={
            "id": "sub-1",
            "url": "http://127.0.0.1:8080/hook",
            "secret": "atlas-secret",
            "account_id": "account-1",
            "channel": "generic",
            "auth_header": None,
        }
    )
    cfg = SimpleNamespace(
        timeout_seconds=10,
        max_retries=3,
        retry_delay_seconds=1,
        allow_non_global_destinations=False,
    )

    with patch("atlas_brain.services.b2b.webhook_dispatcher.settings", create=True) as settings_mock, patch.object(
        webhook_dispatcher,
        "validate_webhook_destination",
        AsyncMock(return_value=(False, "Webhook destination cannot target non-global address: 127.0.0.1")),
    ), patch.object(
        webhook_dispatcher,
        "_deliver_single",
        AsyncMock(return_value=False),
    ) as deliver_single:
        settings_mock.b2b_webhook = cfg
        result = await webhook_dispatcher.send_test_webhook(pool, "sub-1")

    assert result == {
        "success": False,
        "subscription_id": "sub-1",
        "channel": "generic",
        "error": "Webhook destination cannot target non-global address: 127.0.0.1",
    }
    deliver_single.assert_awaited_once()
