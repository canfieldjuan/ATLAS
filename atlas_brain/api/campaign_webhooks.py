"""
Webhook endpoint for campaign email ESP events (Resend).

Receives open, click, bounce, and complaint events and updates
campaign + sequence state accordingly.
"""

import base64
import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from ..config import settings
from ..storage.database import get_db_pool
from ..autonomous.tasks.campaign_audit import log_campaign_event

logger = logging.getLogger("atlas.api.campaign_webhooks")

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def _verify_svix_signature(
    payload_bytes: bytes,
    headers: dict[str, str],
    secret: str,
) -> bool:
    """Verify Resend webhook signature (Svix format).

    Returns True if valid or if no secret is configured (skip verification).
    """
    if not secret:
        logger.warning("Webhook signing secret not configured -- signature verification disabled")
        return True

    msg_id = headers.get("svix-id", "")
    timestamp = headers.get("svix-timestamp", "")
    signature_header = headers.get("svix-signature", "")

    if not msg_id or not timestamp or not signature_header:
        return False

    # Svix signs: "{msg_id}.{timestamp}.{body}"
    to_sign = f"{msg_id}.{timestamp}.".encode() + payload_bytes

    # Secret may be prefixed with "whsec_"
    raw_secret = secret
    if raw_secret.startswith("whsec_"):
        raw_secret = raw_secret[6:]

    secret_bytes = base64.b64decode(raw_secret)
    expected = base64.b64encode(
        hmac.new(secret_bytes, to_sign, hashlib.sha256).digest()
    ).decode()

    # signature_header can contain multiple signatures separated by spaces
    for sig in signature_header.split(" "):
        # Each signature is "v1,<base64>"
        parts = sig.split(",", 1)
        if len(parts) == 2 and parts[0] == "v1" and hmac.compare_digest(parts[1], expected):
            return True
    return False


@router.post("/campaign-email")
async def campaign_email_webhook(request: Request):
    """Receive Resend ESP events: opened, clicked, bounced, complained."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database unavailable")

    payload_bytes = await request.body()

    # Verify signature
    signing_secret = settings.campaign_sequence.resend_webhook_signing_secret
    svix_headers = {
        "svix-id": request.headers.get("svix-id", ""),
        "svix-timestamp": request.headers.get("svix-timestamp", ""),
        "svix-signature": request.headers.get("svix-signature", ""),
    }
    if not _verify_svix_signature(payload_bytes, svix_headers, signing_secret):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event_type = payload.get("type", "")
    data = payload.get("data", {})
    esp_message_id = data.get("email_id", "")

    if not esp_message_id:
        return {"status": "ignored", "reason": "no email_id"}

    # Look up campaign by ESP message ID
    campaign = await pool.fetchrow(
        """
        SELECT id, sequence_id, step_number, recipient_email
        FROM b2b_campaigns
        WHERE esp_message_id = $1
        """,
        esp_message_id,
    )
    if not campaign:
        logger.debug("Webhook for unknown esp_message_id: %s", esp_message_id)
        return {"status": "ignored", "reason": "unknown campaign"}

    campaign_id = campaign["id"]
    sequence_id = campaign["sequence_id"]
    now = datetime.now(timezone.utc)

    if event_type == "email.opened":
        await pool.execute(
            "UPDATE b2b_campaigns SET opened_at = COALESCE(opened_at, $1) WHERE id = $2",
            now, campaign_id,
        )
        if sequence_id:
            await pool.execute(
                """
                UPDATE campaign_sequences
                SET last_opened_at = $1,
                    open_count = open_count + 1,
                    updated_at = $2
                WHERE id = $3
                """,
                now, now, sequence_id,
            )
        await log_campaign_event(
            pool, event_type="opened", source="webhook",
            campaign_id=campaign_id, sequence_id=sequence_id,
            esp_message_id=esp_message_id,
            step_number=campaign["step_number"],
        )

    elif event_type == "email.clicked":
        await pool.execute(
            "UPDATE b2b_campaigns SET clicked_at = COALESCE(clicked_at, $1) WHERE id = $2",
            now, campaign_id,
        )
        if sequence_id:
            await pool.execute(
                """
                UPDATE campaign_sequences
                SET last_clicked_at = $1,
                    click_count = click_count + 1,
                    updated_at = $2
                WHERE id = $3
                """,
                now, now, sequence_id,
            )
        await log_campaign_event(
            pool, event_type="clicked", source="webhook",
            campaign_id=campaign_id, sequence_id=sequence_id,
            esp_message_id=esp_message_id,
            step_number=campaign["step_number"],
            metadata={"url": data.get("click", {}).get("link", "")},
        )

    elif event_type == "email.bounced":
        bounce_type = data.get("bounce", {}).get("type", "hard")
        if sequence_id:
            await pool.execute(
                """
                UPDATE campaign_sequences
                SET status = 'bounced',
                    bounced_at = $1,
                    bounce_type = $2,
                    updated_at = $1
                WHERE id = $3
                """,
                now, bounce_type, sequence_id,
            )
        await log_campaign_event(
            pool, event_type="bounced", source="webhook",
            campaign_id=campaign_id, sequence_id=sequence_id,
            esp_message_id=esp_message_id,
            step_number=campaign["step_number"],
            metadata={"bounce_type": bounce_type},
        )

        # Mark campaign as cancelled (bounced emails are not valid sends)
        await pool.execute(
            "UPDATE b2b_campaigns SET status = 'cancelled' WHERE id = $1",
            campaign_id,
        )

        # Cancel any remaining queued campaigns in the sequence
        if sequence_id:
            await pool.execute(
                "UPDATE b2b_campaigns SET status = 'cancelled' WHERE sequence_id = $1 AND status = 'queued'",
                sequence_id,
            )

        # Global suppression: prevent future campaigns to this address
        from ..autonomous.tasks.campaign_suppression import add_suppression
        from datetime import timedelta

        await add_suppression(
            pool, email=campaign["recipient_email"],
            reason="bounce_hard" if bounce_type == "hard" else "bounce_soft",
            source="webhook", campaign_id=campaign_id,
            expires_at=now + timedelta(days=30) if bounce_type != "hard" else None,
        )

    elif event_type == "email.complained":
        if sequence_id:
            await pool.execute(
                "UPDATE campaign_sequences SET status = 'unsubscribed', updated_at = $1 WHERE id = $2",
                now, sequence_id,
            )
        await log_campaign_event(
            pool, event_type="complained", source="webhook",
            campaign_id=campaign_id, sequence_id=sequence_id,
            esp_message_id=esp_message_id,
            step_number=campaign["step_number"],
        )

        # Mark campaign as cancelled and cancel remaining queued campaigns
        await pool.execute(
            "UPDATE b2b_campaigns SET status = 'cancelled' WHERE id = $1",
            campaign_id,
        )
        if sequence_id:
            await pool.execute(
                "UPDATE b2b_campaigns SET status = 'cancelled' WHERE sequence_id = $1 AND status = 'queued'",
                sequence_id,
            )

        # Global suppression: permanent block on complaint
        from ..autonomous.tasks.campaign_suppression import add_suppression

        await add_suppression(
            pool, email=campaign["recipient_email"],
            reason="complaint", source="webhook", campaign_id=campaign_id,
        )

    elif event_type == "email.delivered":
        await log_campaign_event(
            pool, event_type="delivered", source="webhook",
            campaign_id=campaign_id, sequence_id=sequence_id,
            esp_message_id=esp_message_id,
            step_number=campaign["step_number"],
        )

    else:
        logger.debug("Unhandled webhook event type: %s", event_type)
        return {"status": "ignored", "reason": f"unhandled type: {event_type}"}

    logger.info(
        "Webhook %s processed: campaign=%s sequence=%s",
        event_type, campaign_id, sequence_id,
    )
    return {"status": "ok", "event_type": event_type}
