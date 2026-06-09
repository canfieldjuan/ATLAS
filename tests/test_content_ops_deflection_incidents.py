from __future__ import annotations

import json
import logging

from atlas_brain.content_ops_deflection_incidents import (
    INCIDENT_LOG_MARKER,
    MAX_INCIDENT_FIELD_CHARS,
    emit_deflection_paid_funnel_incident,
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
