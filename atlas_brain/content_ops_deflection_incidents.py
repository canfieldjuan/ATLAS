"""Structured incident records for the paid FAQ-deflection funnel."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal


INCIDENT_LOG_MARKER = "DEFLECTION_PAID_FUNNEL_INCIDENT"
INCIDENT_ALERT_DISPATCH_TIMEOUT_SECONDS = 2.0
MAX_INCIDENT_FIELD_CHARS = 240
_Severity = Literal["error", "warning", "info"]


def emit_deflection_paid_funnel_incident(
    logger: logging.Logger,
    *,
    incident_type: str,
    severity: _Severity = "error",
    **fields: Any,
) -> dict[str, str]:
    """Emit a bounded structured incident record through an existing logger."""

    payload = _incident_payload(incident_type=incident_type, severity=severity, **fields)
    message = f"{INCIDENT_LOG_MARKER} {json.dumps(payload, sort_keys=True)}"
    if severity == "error":
        logger.error(message)
    elif severity == "warning":
        logger.warning(message)
    else:
        logger.info(message)
    return payload


async def emit_deflection_paid_funnel_incident_alert(
    logger: logging.Logger,
    *,
    incident_type: str,
    severity: _Severity = "error",
    **fields: Any,
) -> dict[str, str]:
    """Emit a paid-funnel incident log and route it to configured alert sinks."""
    payload = emit_deflection_paid_funnel_incident(
        logger,
        incident_type=incident_type,
        severity=severity,
        **fields,
    )
    try:
        from .alerts import PaidFunnelIncidentAlertEvent, get_alert_manager

        await asyncio.wait_for(
            get_alert_manager().process_event(
                PaidFunnelIncidentAlertEvent.from_payload(payload)
            ),
            timeout=INCIDENT_ALERT_DISPATCH_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Deflection paid-funnel alert dispatch timed out after %.1fs",
            INCIDENT_ALERT_DISPATCH_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.warning("Deflection paid-funnel alert dispatch failed: %s", exc)
    return payload


def _incident_payload(
    *,
    incident_type: str,
    severity: str,
    **fields: Any,
) -> dict[str, str]:
    payload = {
        "incident_type": _bounded_text(incident_type),
        "severity": _bounded_text(severity),
    }
    for key, value in fields.items():
        text = _bounded_text(value)
        if text:
            payload[str(key)] = text
    return payload


def _bounded_text(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if len(text) <= MAX_INCIDENT_FIELD_CHARS:
        return text
    return text[: MAX_INCIDENT_FIELD_CHARS - 3] + "..."
