from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, patch

import pytest

from atlas_brain.alerts import get_alert_manager, reset_alert_manager
from atlas_brain.content_ops_deflection_incidents import (
    INCIDENT_LOG_MARKER,
    MAX_INCIDENT_FIELD_CHARS,
    emit_deflection_paid_funnel_incident,
    emit_deflection_paid_funnel_incident_alert,
)


def _incident_payload(record: logging.LogRecord) -> dict[str, str]:
    marker, payload = record.getMessage().split(INCIDENT_LOG_MARKER, 1)
    assert marker == ""
    return json.loads(payload.strip())


def test_emit_deflection_paid_funnel_incident_logs_bounded_json(
    caplog,
) -> None:
    logger = logging.getLogger("tests.deflection.incidents")
    caplog.set_level(logging.WARNING, logger=logger.name)

    emit_deflection_paid_funnel_incident(
        logger,
        incident_type="paid_report_delivery_send_failed",
        severity="warning",
        account_id="acct-123",
        request_id="req-123",
        error="x" * 700,
        empty="",
        amount_total=0,
        async_payment=False,
        missing=None,
    )

    record = caplog.records[0]
    payload = _incident_payload(record)
    assert record.levelno == logging.WARNING
    assert payload["incident_type"] == "paid_report_delivery_send_failed"
    assert payload["severity"] == "warning"
    assert payload["account_id"] == "acct-123"
    assert payload["request_id"] == "req-123"
    assert payload["amount_total"] == "0"
    assert payload["async_payment"] == "False"
    assert "empty" not in payload
    assert "missing" not in payload
    assert len(payload["error"]) == MAX_INCIDENT_FIELD_CHARS
    assert payload["error"].endswith("...")


@pytest.mark.asyncio
async def test_emit_deflection_paid_funnel_incident_alert_routes_to_alert_manager(
    caplog,
) -> None:
    reset_alert_manager()
    logger = logging.getLogger("tests.deflection.incidents.alerts")
    caplog.set_level(logging.ERROR, logger=logger.name)
    callback = AsyncMock()
    manager = get_alert_manager()
    manager.register_callback(callback)

    with patch("atlas_brain.alerts.manager.settings") as mock_settings:
        mock_settings.alerts.enabled = True
        mock_settings.alerts.persist_alerts = False
        payload = await emit_deflection_paid_funnel_incident_alert(
            logger,
            incident_type="paid_report_missing_after_payment",
            severity="error",
            account_id="acct-123",
            request_id="req-123",
            error="provider echoed buyer@example.com",
        )

    assert payload["incident_type"] == "paid_report_missing_after_payment"
    incident_record = next(
        record for record in caplog.records if INCIDENT_LOG_MARKER in record.getMessage()
    )
    assert _incident_payload(incident_record)["request_id"] == "req-123"
    callback.assert_awaited_once()
    message, rule, event = callback.await_args.args
    assert rule.name == "deflection_paid_funnel_incident"
    assert event.event_type == "deflection_paid_funnel_incident"
    assert event.metadata["error"] == "provider echoed buyer@example.com"
    assert message == (
        "Paid deflection funnel incident: paid_report_missing_after_payment "
        "(error) account=acct-123 request=req-123"
    )
    assert "buyer@example.com" not in message


@pytest.mark.asyncio
async def test_emit_deflection_paid_funnel_incident_alert_fails_best_effort(
    caplog,
) -> None:
    reset_alert_manager()
    logger = logging.getLogger("tests.deflection.incidents.alert_failures")
    caplog.set_level(logging.WARNING, logger=logger.name)

    with patch(
        "atlas_brain.alerts.manager.AlertManager.process_event",
        new=AsyncMock(side_effect=RuntimeError("sink down")),
    ):
        payload = await emit_deflection_paid_funnel_incident_alert(
            logger,
            incident_type="paid_report_delivery_send_failed",
            severity="warning",
            account_id="acct-123",
            request_id="req-123",
        )

    assert payload["incident_type"] == "paid_report_delivery_send_failed"
    assert any(INCIDENT_LOG_MARKER in record.getMessage() for record in caplog.records)
    assert any(
        "Deflection paid-funnel alert dispatch failed: sink down" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_emit_deflection_paid_funnel_incident_alert_times_out_best_effort(
    caplog,
) -> None:
    reset_alert_manager()
    logger = logging.getLogger("tests.deflection.incidents.alert_timeouts")
    caplog.set_level(logging.WARNING, logger=logger.name)

    async def _hang(*_args, **_kwargs) -> None:
        import asyncio

        await asyncio.sleep(1)

    with (
        patch(
            "atlas_brain.content_ops_deflection_incidents.INCIDENT_ALERT_DISPATCH_TIMEOUT_SECONDS",
            0.01,
        ),
        patch(
            "atlas_brain.alerts.manager.AlertManager.process_event",
            new=AsyncMock(side_effect=_hang),
        ),
    ):
        payload = await emit_deflection_paid_funnel_incident_alert(
            logger,
            incident_type="paid_report_missing_after_payment",
            severity="error",
            account_id="acct-123",
            request_id="req-123",
        )

    assert payload["incident_type"] == "paid_report_missing_after_payment"
    assert any(INCIDENT_LOG_MARKER in record.getMessage() for record in caplog.records)
    assert any(
        "Deflection paid-funnel alert dispatch timed out after 0.0s"
        in record.getMessage()
        for record in caplog.records
    )
