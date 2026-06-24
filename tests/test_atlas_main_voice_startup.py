from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from atlas_brain import main
from atlas_brain.logging_config import (
    AtlasJsonFormatter,
    TEXT_LOG_FORMAT,
    build_log_formatter,
    configure_logging,
)


def test_build_log_formatter_keeps_text_default():
    formatter = build_log_formatter("text")

    assert isinstance(formatter, logging.Formatter)
    assert formatter._style._fmt == TEXT_LOG_FORMAT


def test_build_log_formatter_rejects_unknown_format():
    with pytest.raises(ValueError, match="ATLAS_LOG_FORMAT"):
        build_log_formatter("xml")


def test_json_log_formatter_emits_stable_fields_and_extra():
    record = logging.LogRecord(
        name="atlas.tests",
        level=logging.INFO,
        pathname=__file__,
        lineno=42,
        msg="hello %s",
        args=("world",),
        exc_info=None,
        func="test_func",
    )
    record.account_id = "acct-123"
    record.not_json = object()

    payload = json.loads(AtlasJsonFormatter().format(record))

    assert payload["timestamp"].endswith("Z")
    assert payload["level"] == "INFO"
    assert payload["logger"] == "atlas.tests"
    assert payload["message"] == "hello world"
    assert payload["module"] == "test_atlas_main_voice_startup"
    assert payload["function"] == "test_func"
    assert payload["line"] == 42
    assert payload["extra"]["account_id"] == "acct-123"
    assert isinstance(payload["extra"]["not_json"], str)


def test_json_log_formatter_namespaces_reserved_extra_fields():
    record = logging.LogRecord(
        name="atlas.tests",
        level=logging.ERROR,
        pathname=__file__,
        lineno=45,
        msg="canonical message",
        args=(),
        exc_info=None,
        func="test_func",
    )
    record.level = "business-level"
    record.logger = "business-logger"
    record.timestamp = "yesterday"

    payload = json.loads(AtlasJsonFormatter().format(record))

    assert payload["level"] == "ERROR"
    assert payload["logger"] == "atlas.tests"
    assert payload["timestamp"] != "yesterday"
    assert payload["extra"]["level"] == "business-level"
    assert payload["extra"]["logger"] == "business-logger"
    assert payload["extra"]["timestamp"] == "yesterday"


def test_json_log_formatter_emits_strict_json_for_non_finite_numbers():
    record = logging.LogRecord(
        name="atlas.tests",
        level=logging.INFO,
        pathname=__file__,
        lineno=52,
        msg="metric",
        args=(),
        exc_info=None,
        func="test_func",
    )
    record.positive_infinity = float("inf")
    record.not_a_number = float("nan")

    line = AtlasJsonFormatter().format(record)
    payload = json.loads(line)

    assert "Infinity" not in line
    assert "NaN" not in line
    assert payload["extra"]["positive_infinity"] == "inf"
    assert payload["extra"]["not_a_number"] == "nan"


def test_json_log_formatter_normalizes_nested_mixed_key_dicts():
    record = logging.LogRecord(
        name="atlas.tests",
        level=logging.INFO,
        pathname=__file__,
        lineno=61,
        msg="nested",
        args=(),
        exc_info=None,
        func="test_func",
    )
    record.details = {"b": "two", 1: "one", "nested": {2: "two"}}

    payload = json.loads(AtlasJsonFormatter().format(record))

    assert payload["extra"]["details"]["1"] == "one"
    assert payload["extra"]["details"]["b"] == "two"
    assert payload["extra"]["details"]["nested"]["2"] == "two"


def test_json_log_formatter_includes_exception_details():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="atlas.tests",
        level=logging.ERROR,
        pathname=__file__,
        lineno=51,
        msg="failed",
        args=(),
        exc_info=exc_info,
        func="test_func",
    )

    payload = json.loads(AtlasJsonFormatter().format(record))

    assert payload["exception"]["type"] == "RuntimeError"
    assert payload["exception"]["message"] == "boom"
    assert "RuntimeError: boom" in payload["exception"]["traceback"]


def test_json_log_formatter_includes_cached_exception_text():
    record = logging.LogRecord(
        name="atlas.tests",
        level=logging.ERROR,
        pathname=__file__,
        lineno=57,
        msg="failed from cached text",
        args=(),
        exc_info=None,
        func="test_func",
    )
    record.exc_text = "Traceback (most recent call last):\nValueError: cached boom"

    payload = json.loads(AtlasJsonFormatter().format(record))

    assert payload["exception"]["type"] == "unknown"
    assert payload["exception"]["message"] == "ValueError: cached boom"
    assert "ValueError: cached boom" in payload["exception"]["traceback"]


def test_configure_logging_installs_json_formatter_on_root_and_uvicorn_handlers():
    root_logger = logging.getLogger()
    old_handlers = list(root_logger.handlers)
    old_level = root_logger.level
    uvicorn_logger = logging.getLogger("uvicorn.access")
    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    old_uvicorn_handlers = list(uvicorn_logger.handlers)
    old_uvicorn_level = uvicorn_logger.level
    old_uvicorn_propagate = uvicorn_logger.propagate
    old_uvicorn_error_handlers = list(uvicorn_error_logger.handlers)
    old_uvicorn_error_level = uvicorn_error_logger.level
    old_uvicorn_error_propagate = uvicorn_error_logger.propagate
    root_handler = logging.StreamHandler()
    uvicorn_handler = logging.StreamHandler()
    uvicorn_error_handler = logging.StreamHandler()

    try:
        root_logger.handlers[:] = [root_handler]
        uvicorn_logger.handlers[:] = [uvicorn_handler]
        uvicorn_logger.propagate = False
        uvicorn_error_logger.handlers[:] = [uvicorn_error_handler]
        uvicorn_error_logger.propagate = False
        configure_logging(level="WARNING", log_format="json")

        assert root_logger.level == logging.WARNING
        assert isinstance(root_handler.formatter, AtlasJsonFormatter)
        assert isinstance(uvicorn_handler.formatter, AtlasJsonFormatter)
        assert isinstance(uvicorn_error_handler.formatter, AtlasJsonFormatter)
    finally:
        root_logger.handlers[:] = old_handlers
        root_logger.setLevel(old_level)
        uvicorn_logger.handlers[:] = old_uvicorn_handlers
        uvicorn_logger.setLevel(old_uvicorn_level)
        uvicorn_logger.propagate = old_uvicorn_propagate
        uvicorn_error_logger.handlers[:] = old_uvicorn_error_handlers
        uvicorn_error_logger.setLevel(old_uvicorn_error_level)
        uvicorn_error_logger.propagate = old_uvicorn_error_propagate


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
