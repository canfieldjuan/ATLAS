"""
Webhook endpoint for campaign email ESP events.

Dispatches by ``?provider=`` query param to one of the WebhookProvider
plugins in atlas_brain/services/email_webhooks/. Resend remains the
default for backwards compatibility with existing webhook URLs. The route
itself is provider-agnostic: it consumes a CanonicalEvent and updates
b2b_campaigns / campaign_sequences uniformly.

See atlas_brain/services/email_webhooks/__init__.py for the provider
abstraction and atlas_brain/schemas/campaigns.py for the CanonicalEvent
schema.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from ..autonomous.tasks.campaign_audit import log_campaign_event
from ..config import settings
from ..schemas.campaigns import CanonicalEvent
from ..services.email_webhooks import (
    UnknownProviderError,
    resolve as resolve_provider,
)
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.campaign_webhooks")

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def _clean_required_text(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail=f"{field_name} is required")
    return text


@router.get("/unsubscribe", response_class=HTMLResponse)
async def unsubscribe(email: str = Query(..., description="Email to unsubscribe")):
    """One-click unsubscribe endpoint. Adds email to suppression list."""
    email = _clean_required_text(email, "email")
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database unavailable")

    from ..autonomous.tasks.campaign_suppression import add_suppression

    await add_suppression(pool, email=email, reason="unsubscribe", source="recipient")

    logger.info("Unsubscribe processed for %s", email)
    return (
        "<html><body style='font-family:sans-serif;text-align:center;padding:60px;'>"
        "<h2>You have been unsubscribed</h2>"
        "<p>You will no longer receive campaign emails from us.</p>"
        "</body></html>"
    )


@router.post("/campaign-email")
async def campaign_email_webhook(
    request: Request,
    provider: str = Query(
        "resend",
        description="ESP name. One of: resend|ses|sendgrid|postmark|mailgun",
    ),
):
    """Receive ESP engagement events.

    Dispatches by ?provider=. Body parsing and signature verification are
    delegated to the provider plugin; the table-update logic is canonical.
    """
    try:
        plugin = resolve_provider(provider)
    except UnknownProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    payload_bytes = await request.body()
    headers = {k.lower(): v for k, v in request.headers.items()}
    signing_secret = _signing_secret_for(plugin.name)

    try:
        if not plugin.verify_signature(payload_bytes, headers, signing_secret):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))

    try:
        events = plugin.normalize_event(payload_bytes)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))

    if not events:
        return {"status": "ignored", "reason": "no canonical events parsed"}

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database unavailable")

    processed: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for event in events:
        result = await _apply_canonical_event(pool, event)
        (processed if result["status"] == "ok" else skipped).append(result)

    return {
        "status": "ok",
        "provider": plugin.name,
        "processed": len(processed),
        "skipped": len(skipped),
        "events": processed + skipped,
    }


def _signing_secret_for(provider_name: str) -> str:
    """Look up the signing secret for the named provider.

    Today only Resend's secret is wired through ``CampaignSequenceConfig``.
    Other providers will gain their own settings as their plugin lands;
    until then this returns an empty string and the plugin's
    verify_signature() falls back to skip-mode (or, for the stubbed
    providers, raises NotImplementedError before signature checks run).
    """
    if provider_name == "resend":
        return settings.campaign_sequence.resend_webhook_signing_secret or ""
    return ""


async def _apply_canonical_event(pool, event: CanonicalEvent) -> dict[str, str]:
    """Apply a CanonicalEvent to b2b_campaigns + campaign_sequences.

    Mirrors the pre-Gap-1 Resend handler's table updates verbatim, just
    keyed off the canonical fields rather than provider-specific shapes.
    """
    now = datetime.now(timezone.utc)
    if not event.message_id:
        return {"status": "ignored", "reason": "missing message_id"}

    campaign = await pool.fetchrow(
        """
        SELECT id, sequence_id, step_number, recipient_email
        FROM b2b_campaigns
        WHERE esp_message_id = $1
        """,
        event.message_id,
    )
    if not campaign:
        logger.debug("Webhook for unknown esp_message_id: %s", event.message_id)
        return {"status": "ignored", "reason": "unknown campaign"}

    campaign_id = campaign["id"]
    sequence_id = campaign["sequence_id"]
    event_type = event.event_type

    if event_type == "opened":
        detail = await pool.fetchrow(
            "SELECT sent_at, opened_at FROM b2b_campaigns WHERE id = $1",
            campaign_id,
        )
        hours_to_open = None
        if detail and detail["opened_at"] is None and detail["sent_at"]:
            delta = (now - detail["sent_at"]).total_seconds() / 3600
            hours_to_open = round(delta, 2)

        await pool.execute(
            """
            UPDATE b2b_campaigns
            SET opened_at = COALESCE(opened_at, $1),
                hours_to_first_open = COALESCE(hours_to_first_open, $2)
            WHERE id = $3
            """,
            now, hours_to_open, campaign_id,
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
            esp_message_id=event.message_id,
            step_number=campaign["step_number"],
        )

    elif event_type == "clicked":
        detail = await pool.fetchrow(
            "SELECT sent_at, clicked_at FROM b2b_campaigns WHERE id = $1",
            campaign_id,
        )
        hours_to_click = None
        if detail and detail["clicked_at"] is None and detail["sent_at"]:
            delta = (now - detail["sent_at"]).total_seconds() / 3600
            hours_to_click = round(delta, 2)

        await pool.execute(
            """
            UPDATE b2b_campaigns
            SET clicked_at = COALESCE(clicked_at, $1),
                hours_to_first_click = COALESCE(hours_to_first_click, $2)
            WHERE id = $3
            """,
            now, hours_to_click, campaign_id,
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
            esp_message_id=event.message_id,
            step_number=campaign["step_number"],
            metadata={"url": event.click_url or ""},
        )

    elif event_type == "bounced":
        bounce_type = event.bounce_type or "hard"
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
            esp_message_id=event.message_id,
            step_number=campaign["step_number"],
            metadata={"bounce_type": bounce_type},
        )

        await pool.execute(
            "UPDATE b2b_campaigns SET status = 'cancelled' WHERE id = $1",
            campaign_id,
        )
        if sequence_id:
            await pool.execute(
                "UPDATE b2b_campaigns SET status = 'cancelled' WHERE sequence_id = $1 AND status = 'queued'",
                sequence_id,
            )

        from datetime import timedelta

        from ..autonomous.tasks.campaign_suppression import add_suppression

        await add_suppression(
            pool, email=campaign["recipient_email"],
            reason="bounce_hard" if bounce_type == "hard" else "bounce_soft",
            source="webhook", campaign_id=campaign_id,
            expires_at=now + timedelta(days=30) if bounce_type != "hard" else None,
        )

    elif event_type == "complained":
        if sequence_id:
            await pool.execute(
                "UPDATE campaign_sequences SET status = 'unsubscribed', updated_at = $1 WHERE id = $2",
                now, sequence_id,
            )
        await log_campaign_event(
            pool, event_type="complained", source="webhook",
            campaign_id=campaign_id, sequence_id=sequence_id,
            esp_message_id=event.message_id,
            step_number=campaign["step_number"],
        )

        await pool.execute(
            "UPDATE b2b_campaigns SET status = 'cancelled' WHERE id = $1",
            campaign_id,
        )
        if sequence_id:
            await pool.execute(
                "UPDATE b2b_campaigns SET status = 'cancelled' WHERE sequence_id = $1 AND status = 'queued'",
                sequence_id,
            )

        from ..autonomous.tasks.campaign_suppression import add_suppression

        await add_suppression(
            pool, email=campaign["recipient_email"],
            reason="complaint", source="webhook", campaign_id=campaign_id,
        )

    elif event_type == "delivered":
        await log_campaign_event(
            pool, event_type="delivered", source="webhook",
            campaign_id=campaign_id, sequence_id=sequence_id,
            esp_message_id=event.message_id,
            step_number=campaign["step_number"],
        )

    elif event_type == "unsubscribed":
        # Resend reports unsubscribes as 'complained'; SendGrid emits a
        # distinct 'unsubscribe' event per recipient. This branch is wired
        # for the SendGrid normalize_event() to map onto when that provider
        # is implemented (see services/email_webhooks/sendgrid.py).
        if sequence_id:
            await pool.execute(
                "UPDATE campaign_sequences SET status = 'unsubscribed', updated_at = $1 WHERE id = $2",
                now, sequence_id,
            )
        from ..autonomous.tasks.campaign_suppression import add_suppression

        await add_suppression(
            pool, email=campaign["recipient_email"],
            reason="unsubscribe", source="webhook", campaign_id=campaign_id,
        )
        await log_campaign_event(
            pool, event_type="unsubscribed", source="webhook",
            campaign_id=campaign_id, sequence_id=sequence_id,
            esp_message_id=event.message_id,
            step_number=campaign["step_number"],
        )

    else:
        return {"status": "ignored", "reason": f"unhandled event_type: {event_type}"}

    logger.info(
        "Webhook %s processed: campaign=%s sequence=%s",
        event_type, campaign_id, sequence_id,
    )
    return {"status": "ok", "event_type": event_type}
