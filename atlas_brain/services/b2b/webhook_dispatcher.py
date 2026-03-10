"""Outbound webhook delivery for B2B intelligence events.

Dispatches signed payloads to tenant webhook subscriptions when intelligence
events fire (change events, churn alerts, report generation, signal updates).

Usage::

    from atlas_brain.services.b2b.webhook_dispatcher import dispatch_webhooks

    await dispatch_webhooks(pool, "change_event", "Zendesk", {
        "event_type": "urgency_spike",
        "vendor_name": "Zendesk",
        "delta": 2.1,
    })
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone

import httpx

from ...services.tracing import (
    build_business_trace_context,
    build_reasoning_trace_context,
    tracer,
)

logger = logging.getLogger("atlas.services.b2b.webhook_dispatcher")

VALID_EVENT_TYPES = {"change_event", "churn_alert", "report_generated", "signal_update"}


async def dispatch_webhooks(
    pool,
    event_type: str,
    vendor_name: str,
    payload: dict,
) -> int:
    """Fan out a webhook delivery to all subscriptions matching the event.

    Finds accounts that track *vendor_name*, then delivers to each enabled
    subscription that includes *event_type* in its ``event_types`` array.

    Returns the number of successful deliveries.  Never raises -- all errors
    are logged and delivery failures are recorded in the log table.
    """
    try:
        from ...config import settings
        cfg = settings.b2b_webhook
    except Exception:
        return 0

    if not cfg.enabled:
        return 0

    if event_type not in VALID_EVENT_TYPES:
        logger.warning("dispatch_webhooks: invalid event_type %r", event_type)
        return 0

    span = tracer.start_span(
        span_name="b2b.webhook.dispatch",
        operation_type="business_operation",
        metadata={
            "business": build_business_trace_context(
                workflow="webhook_dispatch",
                event_type=event_type,
                vendor_name=vendor_name,
            ),
        },
    )

    try:
        # Find subscriptions: accounts tracking this vendor with matching event type
        subs = await pool.fetch(
            """
            SELECT ws.id, ws.url, ws.secret, ws.account_id,
                   COALESCE(ws.channel, 'generic') AS channel,
                   ws.auth_header
            FROM b2b_webhook_subscriptions ws
            JOIN tracked_vendors tv ON tv.account_id = ws.account_id
            WHERE ws.enabled = true
              AND tv.vendor_name ILIKE $1
              AND $2 = ANY(ws.event_types)
            """,
            vendor_name,
            event_type,
        )

        if not subs:
            tracer.end_span(span, status="completed", output_data={"delivered": 0, "subscriptions": 0})
            return 0

        envelope = _build_envelope(event_type, vendor_name, payload)

        delivered = 0
        for sub in subs:
            channel = sub["channel"]
            channel_bytes = _format_for_channel(channel, envelope)

            if len(channel_bytes) > cfg.max_payload_bytes:
                logger.warning(
                    "Webhook payload too large (%d bytes, max %d) for %s/%s channel=%s",
                    len(channel_bytes), cfg.max_payload_bytes,
                    event_type, vendor_name, channel,
                )
                continue

            ok = await _deliver_single(
                pool, sub, event_type, envelope, channel_bytes, cfg,
            )
            if ok:
                delivered += 1

        tracer.end_span(
            span,
            status="completed",
            output_data={"delivered": delivered, "subscriptions": len(subs)},
            metadata={
                "reasoning": build_reasoning_trace_context(
                    decision={"event_type": event_type},
                    evidence={"subscriptions": len(subs), "payload_keys": list(payload.keys())[:10]},
                ),
            },
        )
        return delivered

    except Exception:
        tracer.end_span(
            span,
            status="failed",
            error_message="webhook dispatch error",
            error_type="WebhookDispatchError",
        )
        logger.exception("dispatch_webhooks error for %s/%s", event_type, vendor_name)
        return 0


async def dispatch_webhooks_multi(
    pool,
    event_type: str,
    events: list[tuple[str, dict]],
) -> int:
    """Dispatch webhooks for multiple (vendor_name, payload) pairs.

    Convenience wrapper for batch event dispatch (e.g., multiple change events
    from a single intelligence run).  Returns total successful deliveries.
    """
    total = 0
    for vendor_name, payload in events:
        total += await dispatch_webhooks(pool, event_type, vendor_name, payload)
    return total


def _build_envelope(event_type: str, vendor_name: str, payload: dict) -> dict:
    """Wrap the raw payload in a standard envelope."""
    return {
        "event": event_type,
        "vendor": vendor_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": payload,
    }


def _sign_payload(payload_bytes: bytes, secret: str) -> str:
    """Compute HMAC-SHA256 hex digest for payload signing."""
    return hmac.new(
        secret.encode(), payload_bytes, hashlib.sha256,
    ).hexdigest()


# ---------------------------------------------------------------------------
# Channel-specific payload formatting
# ---------------------------------------------------------------------------

VALID_CHANNELS = {"generic", "slack", "teams", "crm_hubspot", "crm_salesforce", "crm_pipedrive"}


def _format_slack_payload(envelope: dict) -> dict:
    """Convert envelope to Slack Block Kit format (incoming webhook)."""
    event_type = envelope.get("event", "unknown")
    vendor = envelope.get("vendor", "Unknown")
    ts = envelope.get("timestamp", "")
    data = envelope.get("data", {})

    # Header
    header = f":bell: *Atlas Intelligence Alert*  |  `{event_type}`"
    # Summary line
    summary = f"*Vendor:* {vendor}"
    if ts:
        summary += f"  |  {ts[:19]}"

    # Build detail fields from data dict
    fields = []
    for k, v in list(data.items())[:10]:
        fields.append({"type": "mrkdwn", "text": f"*{k}:* {v}"})

    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": header}},
        {"type": "section", "text": {"type": "mrkdwn", "text": summary}},
    ]
    if fields:
        # Slack allows max 10 fields per section
        blocks.append({"type": "section", "fields": fields})

    return {"blocks": blocks, "text": f"Atlas: {event_type} for {vendor}"}


def _format_teams_payload(envelope: dict) -> dict:
    """Convert envelope to Microsoft Teams Adaptive Card format."""
    event_type = envelope.get("event", "unknown")
    vendor = envelope.get("vendor", "Unknown")
    ts = envelope.get("timestamp", "")
    data = envelope.get("data", {})

    # Build fact set from data dict
    facts = []
    for k, v in list(data.items())[:10]:
        facts.append({"title": str(k), "value": str(v)})

    card = {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "contentUrl": None,
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.4",
                "body": [
                    {
                        "type": "TextBlock",
                        "size": "Medium",
                        "weight": "Bolder",
                        "text": f"Atlas Intelligence: {event_type}",
                    },
                    {
                        "type": "FactSet",
                        "facts": [
                            {"title": "Vendor", "value": vendor},
                            {"title": "Time", "value": ts[:19] if ts else "N/A"},
                        ] + facts,
                    },
                ],
            },
        }],
    }
    return card


def _format_crm_hubspot_payload(envelope: dict) -> dict:
    """Format as HubSpot CRM API payload.

    Maps to HubSpot Deals API (POST /crm/v3/objects/deals) or
    Notes API (POST /crm/v3/objects/notes) depending on event type.
    Customers point their subscription URL at the HubSpot API endpoint.
    """
    event_type = envelope.get("event", "unknown")
    vendor = envelope.get("vendor", "Unknown")
    data = envelope.get("data", {})

    if event_type in ("churn_alert", "signal_update"):
        # Create/update a deal or note with intelligence data
        properties = {
            "dealname": f"Churn Signal: {vendor}",
            "pipeline": "default",
            "dealstage": "qualifiedtobuy",
            "atlas_vendor": vendor,
            "atlas_event_type": event_type,
        }
        for k, v in list(data.items())[:15]:
            key = f"atlas_{k}".lower().replace(" ", "_")[:50]
            properties[key] = str(v)[:65535]
        return {"properties": properties}

    # Default: note-style payload
    lines = [f"Atlas Intelligence: {event_type} for {vendor}"]
    for k, v in list(data.items())[:20]:
        lines.append(f"  {k}: {v}")
    return {
        "properties": {
            "hs_note_body": "\n".join(lines),
            "hs_timestamp": envelope.get("timestamp", ""),
        },
    }


def _format_crm_salesforce_payload(envelope: dict) -> dict:
    """Format as Salesforce REST API payload.

    Maps to Salesforce sObject API (PATCH /services/data/v59.0/sobjects/...).
    Customers point their subscription URL at a Salesforce composite or
    sObject endpoint.
    """
    event_type = envelope.get("event", "unknown")
    vendor = envelope.get("vendor", "Unknown")
    data = envelope.get("data", {})

    fields = {
        "Atlas_Vendor__c": vendor,
        "Atlas_Event_Type__c": event_type,
        "Atlas_Timestamp__c": envelope.get("timestamp", ""),
    }
    for k, v in list(data.items())[:15]:
        field_name = f"Atlas_{k}__c"[:40]
        fields[field_name] = str(v)[:255]

    return fields


def _format_crm_pipedrive_payload(envelope: dict) -> dict:
    """Format as Pipedrive API payload.

    Maps to Pipedrive Deals API (POST /v1/deals) or Notes API (POST /v1/notes).
    """
    event_type = envelope.get("event", "unknown")
    vendor = envelope.get("vendor", "Unknown")
    data = envelope.get("data", {})

    if event_type in ("churn_alert", "signal_update"):
        return {
            "title": f"Churn Signal: {vendor}",
            "status": "open",
            "atlas_vendor": vendor,
            "atlas_event_type": event_type,
            **{f"atlas_{k}": str(v) for k, v in list(data.items())[:10]},
        }

    lines = [f"Atlas Intelligence: {event_type} for {vendor}"]
    for k, v in list(data.items())[:20]:
        lines.append(f"  {k}: {v}")
    return {"content": "\n".join(lines)}


def _format_for_channel(channel: str, envelope: dict) -> bytes:
    """Format envelope for the subscription's channel type.

    Returns the JSON-encoded bytes to POST.  For generic channels,
    the envelope is sent as-is.  Notification channels (Slack, Teams)
    get their native formats.  CRM channels format for provider APIs.
    """
    if channel == "slack":
        payload = _format_slack_payload(envelope)
    elif channel == "teams":
        payload = _format_teams_payload(envelope)
    elif channel == "crm_hubspot":
        payload = _format_crm_hubspot_payload(envelope)
    elif channel == "crm_salesforce":
        payload = _format_crm_salesforce_payload(envelope)
    elif channel == "crm_pipedrive":
        payload = _format_crm_pipedrive_payload(envelope)
    else:
        payload = envelope
    return json.dumps(payload, default=str).encode()


async def _deliver_single(
    pool,
    sub,
    event_type: str,
    envelope: dict,
    payload_bytes: bytes,
    cfg,
) -> bool:
    """Deliver to a single subscription with retry.  Returns True on success."""
    signature = _sign_payload(payload_bytes, sub["secret"])
    headers = {
        "Content-Type": "application/json",
        "X-Atlas-Signature": f"sha256={signature}",
        "X-Atlas-Event": event_type,
        "User-Agent": "Atlas-Webhook/1.0",
    }
    # CRM channels: add Authorization header if configured
    auth_header = sub.get("auth_header")
    if auth_header:
        headers["Authorization"] = auth_header

    for attempt in range(1, cfg.max_retries + 1):
        status_code = None
        response_body = None
        duration_ms = None
        error_msg = None
        success = False

        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=cfg.timeout_seconds) as client:
                resp = await client.post(
                    sub["url"], content=payload_bytes, headers=headers,
                )
            duration_ms = int((time.monotonic() - t0) * 1000)
            status_code = resp.status_code
            response_body = resp.text[:500] if resp.text else None
            success = 200 <= resp.status_code < 300
        except httpx.TimeoutException:
            duration_ms = int((time.monotonic() - t0) * 1000)
            error_msg = "timeout"
        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            error_msg = str(exc)[:500]

        # Log delivery attempt
        try:
            await pool.execute(
                """
                INSERT INTO b2b_webhook_delivery_log
                    (subscription_id, event_type, payload, status_code,
                     response_body, duration_ms, attempt, success, error)
                VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7, $8, $9)
                """,
                sub["id"],
                event_type,
                json.dumps(envelope, default=str),
                status_code,
                response_body,
                duration_ms,
                attempt,
                success,
                error_msg,
            )
        except Exception:
            logger.exception("Failed to log webhook delivery")

        if success:
            # Log CRM push for CRM channels
            channel = sub.get("channel", "generic")
            if channel.startswith("crm_"):
                await _log_crm_push(
                    pool, sub["id"], event_type, envelope,
                    crm_record_id=None, response_body=response_body,
                )
            return True

        if attempt < cfg.max_retries:
            logger.info(
                "Webhook delivery attempt %d/%d failed for %s (status=%s, error=%s), retrying",
                attempt, cfg.max_retries, sub["url"], status_code, error_msg,
            )
            # Simple linear backoff
            await asyncio.sleep(cfg.retry_delay_seconds * attempt)

    logger.warning(
        "Webhook delivery exhausted %d retries for subscription %s -> %s",
        cfg.max_retries, sub["id"], sub["url"],
    )
    return False


async def _log_crm_push(
    pool,
    subscription_id,
    event_type: str,
    envelope: dict,
    *,
    crm_record_id: str | None = None,
    response_body: str | None = None,
) -> None:
    """Log a successful CRM push to b2b_crm_push_log."""
    try:
        vendor = envelope.get("vendor", "")
        data = envelope.get("data", {})
        # Try to extract CRM record ID from response body
        if not crm_record_id and response_body:
            try:
                resp_data = json.loads(response_body)
                crm_record_id = (
                    resp_data.get("id")
                    or resp_data.get("vid")
                    or resp_data.get("data", {}).get("id")
                )
                if crm_record_id:
                    crm_record_id = str(crm_record_id)
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        signal_type = "change_event" if event_type == "change_event" else (
            "company_signal" if data.get("company_name") else "churn_signal"
        )
        await pool.execute(
            """
            INSERT INTO b2b_crm_push_log
                (subscription_id, signal_type, vendor_name, company_name,
                 crm_record_id, crm_record_type)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            subscription_id,
            signal_type,
            vendor,
            data.get("company_name"),
            crm_record_id,
            "deal" if event_type in ("churn_alert", "signal_update") else "note",
        )
    except Exception:
        logger.debug("Failed to log CRM push")


async def send_test_webhook(pool, subscription_id) -> dict:
    """Send a test payload to a specific subscription.  Returns delivery result."""
    try:
        from ...config import settings
        cfg = settings.b2b_webhook
    except Exception:
        return {"success": False, "error": "Config not available"}

    sub = await pool.fetchrow(
        """SELECT id, url, secret, account_id,
                  COALESCE(channel, 'generic') AS channel,
                  auth_header
           FROM b2b_webhook_subscriptions WHERE id = $1""",
        subscription_id,
    )
    if not sub:
        return {"success": False, "error": "Subscription not found"}

    test_payload = {
        "test": True,
        "message": "This is a test webhook from Atlas B2B Intelligence",
    }
    envelope = _build_envelope("test", "test_vendor", test_payload)
    payload_bytes = _format_for_channel(sub["channel"], envelope)
    span = tracer.start_span(
        span_name="b2b.webhook.test",
        operation_type="business_operation",
        session_id=str(subscription_id),
        metadata={
            "business": build_business_trace_context(
                workflow="webhook_test",
                subscription_id=str(subscription_id),
            ),
        },
    )

    ok = await _deliver_single(pool, sub, "test", envelope, payload_bytes, cfg)
    response = {"success": ok, "subscription_id": str(subscription_id), "channel": sub["channel"]}
    tracer.end_span(
        span,
        status="completed" if ok else "failed",
        output_data=response,
        metadata={
            "reasoning": build_reasoning_trace_context(
                decision={"channel": sub["channel"]},
                evidence={"success": ok},
            ),
        },
        error_message=None if ok else "test webhook delivery failed",
        error_type=None if ok else "WebhookTestFailure",
    )
    return response
