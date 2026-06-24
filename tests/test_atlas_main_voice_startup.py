from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from atlas_brain import main


def _paid_funnel_settings(
    *,
    stripe_secret_key: str = "sk_test_paid_deflection",
    stripe_webhook_secret: str = "whsec_test_paid_deflection",
    stripe_content_ops_deflection_report_amount_cents: int = 150000,
    stripe_content_ops_deflection_report_price_id: str = "price_paid_deflection",
    stripe_content_ops_deflection_report_allowed_amount_cents: str = "",
    stripe_content_ops_deflection_report_currency: str = "usd",
    alerts_enabled: bool = True,
    ntfy_enabled: bool = True,
    ntfy_url: str = "https://ntfy.example",
    ntfy_topic: str = "atlas-paid-funnel",
):
    return SimpleNamespace(
        saas_auth=SimpleNamespace(
            stripe_secret_key=stripe_secret_key,
            stripe_webhook_secret=stripe_webhook_secret,
            stripe_content_ops_deflection_report_amount_cents=(
                stripe_content_ops_deflection_report_amount_cents
            ),
            stripe_content_ops_deflection_report_price_id=(
                stripe_content_ops_deflection_report_price_id
            ),
            stripe_content_ops_deflection_report_allowed_amount_cents=(
                stripe_content_ops_deflection_report_allowed_amount_cents
            ),
            stripe_content_ops_deflection_report_currency=(
                stripe_content_ops_deflection_report_currency
            ),
        ),
        alerts=SimpleNamespace(
            enabled=alerts_enabled,
            ntfy_enabled=ntfy_enabled,
            ntfy_url=ntfy_url,
            ntfy_topic=ntfy_topic,
        ),
    )


def test_security_txt_body_contains_required_fields():
    body = main._security_txt_body(
        now=datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)
    )

    assert (
        "Contact: https://github.com/canfieldjuan/ATLAS/security/advisories/new\n"
        in body
    )
    assert "Policy: https://github.com/canfieldjuan/ATLAS/blob/main/SECURITY.md\n" in body
    assert "Preferred-Languages: en\n" in body
    assert "Expires: 2026-12-20T12:00:00Z\n" in body


def test_security_txt_route_serves_plain_text_without_auth():
    response = TestClient(main.app).get("/.well-known/security.txt")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert (
        "Contact: https://github.com/canfieldjuan/ATLAS/security/advisories/new"
        in response.text
    )
    assert "Policy: https://github.com/canfieldjuan/ATLAS/blob/main/SECURITY.md" in response.text
    assert "Preferred-Languages: en" in response.text
    assert "Expires:" in response.text


def test_paid_funnel_alert_preflight_skips_unrelated_stripe_without_deflection_price():
    config = _paid_funnel_settings(
        stripe_secret_key="sk_test_subscription_billing",
        stripe_webhook_secret="whsec_test_subscription_billing",
        stripe_content_ops_deflection_report_price_id="",
        ntfy_enabled=False,
    )

    assert main._paid_funnel_stripe_deflection_configured(config) is False
    assert main._paid_funnel_alert_channel_errors(config) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"stripe_content_ops_deflection_report_amount_cents": 0},
        {"stripe_content_ops_deflection_report_currency": "us"},
        {"stripe_content_ops_deflection_report_currency": "us1"},
        {"stripe_content_ops_deflection_report_allowed_amount_cents": "9900"},
        {"stripe_content_ops_deflection_report_allowed_amount_cents": "150000,bad"},
    ],
)
def test_paid_funnel_alert_preflight_skips_when_checkout_terms_reject_charge(
    overrides,
):
    config = _paid_funnel_settings(ntfy_enabled=False, **overrides)

    assert main._paid_funnel_stripe_deflection_configured(config) is False
    assert main._paid_funnel_alert_channel_errors(config) == []


def test_paid_funnel_alert_preflight_requires_ntfy_when_log_only():
    config = _paid_funnel_settings(ntfy_enabled=False)

    errors = main._paid_funnel_alert_channel_errors(config)

    assert main._paid_funnel_stripe_deflection_configured(config) is True
    assert errors == [
        "ATLAS_ALERTS_NTFY_ENABLED must be true when Stripe paid "
        "deflection reports are configured"
    ]


def test_paid_funnel_alert_preflight_does_not_require_atlas_webhook_secret():
    config = _paid_funnel_settings(
        stripe_webhook_secret="",
        ntfy_enabled=False,
    )

    errors = main._paid_funnel_alert_channel_errors(config)

    assert main._paid_funnel_stripe_deflection_configured(config) is True
    assert errors == [
        "ATLAS_ALERTS_NTFY_ENABLED must be true when Stripe paid "
        "deflection reports are configured"
    ]


def test_paid_funnel_alert_preflight_rejects_alerts_disabled():
    config = _paid_funnel_settings(alerts_enabled=False)

    errors = main._paid_funnel_alert_channel_errors(config)

    assert errors == [
        "ATLAS_ALERTS_ENABLED must be true when Stripe paid deflection "
        "reports are configured"
    ]


def test_paid_funnel_alert_preflight_rejects_missing_ntfy_channel():
    config = _paid_funnel_settings(ntfy_url=" ", ntfy_topic="")

    errors = main._paid_funnel_alert_channel_errors(config)

    assert errors == [
        "ATLAS_ALERTS_NTFY_URL must be set when Stripe paid deflection "
        "reports are configured",
        "ATLAS_ALERTS_NTFY_TOPIC must be set when Stripe paid deflection "
        "reports are configured",
    ]


def test_paid_funnel_alert_preflight_rejects_non_absolute_ntfy_url():
    config = _paid_funnel_settings(ntfy_url="localhost:8090")

    errors = main._paid_funnel_alert_channel_errors(config)

    assert errors == [
        "ATLAS_ALERTS_NTFY_URL must be an absolute HTTP(S) URL when "
        "Stripe paid deflection reports are configured"
    ]


def test_paid_funnel_alert_preflight_allows_configured_ntfy_channel():
    config = _paid_funnel_settings(ntfy_url="https://ntfy.example")

    assert main._paid_funnel_stripe_deflection_configured(config) is True
    assert main._paid_funnel_alert_channel_errors(config) == []
    main._enforce_paid_funnel_alert_channel(config)


def test_paid_funnel_alert_preflight_raises_specific_startup_error():
    config = _paid_funnel_settings(ntfy_enabled=False, ntfy_url="localhost:8090")

    with pytest.raises(
        RuntimeError,
        match="requires a configured ntfy alert channel",
    ) as exc:
        main._enforce_paid_funnel_alert_channel(config)

    assert "ATLAS_ALERTS_NTFY_ENABLED" in str(exc.value)
    assert "ATLAS_ALERTS_NTFY_URL" in str(exc.value)


def test_asr_autostart_allows_cpu_device():
    assert main._asr_autostart_blocked_reason("cpu") is None


def test_asr_autostart_blocks_cuda_when_unavailable(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    reason = main._asr_autostart_blocked_reason("cuda")

    assert reason is not None
    assert "requires CUDA" in reason
    assert "CUDA is not available" in reason


def test_start_asr_server_skips_popen_when_cuda_unavailable(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            raise RuntimeError("ASR endpoint is down")

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

    fake_httpx = SimpleNamespace(AsyncClient=FakeAsyncClient)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(main.settings.voice, "asr_url", "http://localhost:8081")
    monkeypatch.setattr(main.settings.voice, "asr_device", "cuda")
    monkeypatch.setattr(
        main.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("ASR subprocess should not start"),
    )

    assert asyncio.run(main._start_asr_server()) is None


def test_start_asr_server_keeps_running_external_asr_before_cuda_guard(monkeypatch):
    class FakeResponse:
        status_code = 200

    class FakeAsyncClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

        async def get(self, _url):
            return FakeResponse()

    fake_httpx = SimpleNamespace(AsyncClient=FakeAsyncClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setattr(main.settings.voice, "asr_url", "http://localhost:8081")
    monkeypatch.setattr(main.settings.voice, "asr_device", "cuda")
    monkeypatch.setattr(
        main,
        "_asr_autostart_blocked_reason",
        lambda _device: pytest.fail("CUDA guard should not run for healthy external ASR"),
    )
    monkeypatch.setattr(
        main.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("ASR subprocess should not start"),
    )

    assert asyncio.run(main._start_asr_server()) is None
