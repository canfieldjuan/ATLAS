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

    try:
        # Find subscriptions: accounts tracking this vendor with matching event type
        subs = await pool.fetch(
            """
            SELECT ws.id, ws.url, ws.secret, ws.account_id
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
            return 0

        envelope = _build_envelope(event_type, vendor_name, payload)
        envelope_bytes = json.dumps(envelope, default=str).encode()

        if len(envelope_bytes) > cfg.max_payload_bytes:
            logger.warning(
                "Webhook payload too large (%d bytes, max %d) for %s/%s",
                len(envelope_bytes), cfg.max_payload_bytes, event_type, vendor_name,
            )
            return 0

        delivered = 0
        for sub in subs:
            ok = await _deliver_single(
                pool, sub, event_type, envelope, envelope_bytes, cfg,
            )
            if ok:
                delivered += 1

        return delivered

    except Exception:
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


async def _deliver_single(
    pool,
    sub,
    event_type: str,
    envelope: dict,
    envelope_bytes: bytes,
    cfg,
) -> bool:
    """Deliver to a single subscription with retry.  Returns True on success."""
    signature = _sign_payload(envelope_bytes, sub["secret"])
    headers = {
        "Content-Type": "application/json",
        "X-Atlas-Signature": f"sha256={signature}",
        "X-Atlas-Event": event_type,
        "User-Agent": "Atlas-Webhook/1.0",
    }

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
                    sub["url"], content=envelope_bytes, headers=headers,
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


async def send_test_webhook(pool, subscription_id) -> dict:
    """Send a test payload to a specific subscription.  Returns delivery result."""
    try:
        from ...config import settings
        cfg = settings.b2b_webhook
    except Exception:
        return {"success": False, "error": "Config not available"}

    sub = await pool.fetchrow(
        "SELECT id, url, secret, account_id FROM b2b_webhook_subscriptions WHERE id = $1",
        subscription_id,
    )
    if not sub:
        return {"success": False, "error": "Subscription not found"}

    test_payload = {
        "test": True,
        "message": "This is a test webhook from Atlas B2B Intelligence",
    }
    envelope = _build_envelope("test", "test_vendor", test_payload)
    envelope_bytes = json.dumps(envelope, default=str).encode()

    ok = await _deliver_single(pool, sub, "test", envelope, envelope_bytes, cfg)
    return {"success": ok, "subscription_id": str(subscription_id)}
