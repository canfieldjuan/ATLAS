"""
REST API for intent-specific email actions.

Provides endpoints triggered by ntfy action buttons for each business intent:
- /quote      (estimate_request) -- generate a quote draft
- /escalate   (complaint)        -- escalate with urgent notification + reminder
- /slots      (reschedule)       -- show available calendar slots  [stub]
- /send-info  (info_admin)       -- generate an info reply draft   [stub]
- /archive    (info_admin)       -- archive the email              [stub]
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("atlas.api.email_actions")

router = APIRouter(prefix="/email/actions", tags=["email-actions"])


async def _load_email(gmail_message_id: str) -> dict:
    """Load a processed email row by gmail_message_id.  Raises 404 if not found."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    row = await pool.fetchrow(
        "SELECT * FROM processed_emails WHERE gmail_message_id = $1",
        gmail_message_id,
    )
    if not row:
        raise HTTPException(404, f"Email {gmail_message_id} not found")
    return dict(row)


# ---------------------------------------------------------------------------
# POST /email/actions/{gmail_message_id}/quote  -- [Get Quote]
# ---------------------------------------------------------------------------

@router.post("/{gmail_message_id}/quote")
async def generate_quote(gmail_message_id: str):
    """Generate a quote/estimate draft reply for the email.

    Internally delegates to the email draft generation endpoint with
    context primed for quoting.
    """
    email_row = await _load_email(gmail_message_id)

    # Trigger draft generation (reuses the existing draft pipeline)
    from .email_drafts import generate_draft

    try:
        result = await generate_draft(gmail_message_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Quote draft generation failed for %s: %s", gmail_message_id, e)
        raise HTTPException(500, "Failed to generate quote draft")

    logger.info(
        "Quote draft triggered for %s (contact=%s)",
        gmail_message_id, email_row.get("contact_id"),
    )
    return {
        "action": "quote",
        "gmail_message_id": gmail_message_id,
        "draft": result,
    }


# ---------------------------------------------------------------------------
# POST /email/actions/{gmail_message_id}/escalate  -- [Escalate]
# ---------------------------------------------------------------------------

@router.post("/{gmail_message_id}/escalate")
async def escalate_email(gmail_message_id: str):
    """Escalate a complaint email.

    - Sets priority to 'urgent' on the processed_emails row
    - Logs escalation as a CRM interaction
    - Sends an urgent ntfy with full email body + customer history
    - Creates a 1-hour callback reminder
    """
    import httpx

    from ..config import settings
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    email_row = await _load_email(gmail_message_id)

    # 1. Set priority to urgent
    await pool.execute(
        "UPDATE processed_emails SET priority = 'urgent' WHERE gmail_message_id = $1",
        gmail_message_id,
    )

    # 2. Log CRM escalation
    contact_id = email_row.get("contact_id")
    if contact_id:
        try:
            from ..services.crm_provider import get_crm_provider

            crm = get_crm_provider()
            await crm.log_interaction(
                contact_id=str(contact_id),
                interaction_type="note",
                summary=f"ESCALATED: Complaint about {email_row.get('subject', '(no subject)')}",
                intent="complaint",
            )
        except Exception as e:
            logger.warning("CRM escalation log failed for %s: %s", gmail_message_id, e)

    # 3. Send urgent ntfy
    if settings.alerts.ntfy_enabled:
        ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"
        customer_summary = email_row.get("customer_context_summary") or ""
        subject = email_row.get("subject", "(no subject)")

        message_parts = [
            f"ESCALATED COMPLAINT",
            f"From: {email_row.get('sender', 'unknown')}",
            f"Subject: {subject}",
        ]
        if customer_summary:
            message_parts.append(f"\n{customer_summary}")

        api_url = settings.email_draft.atlas_api_url.rstrip("/")
        headers = {
            "Title": f"ESCALATED: {subject[:50]}",
            "Priority": "urgent",
            "Tags": "email,rotating_light,exclamation",
            "Actions": (
                f"http, Draft Reply, {api_url}/api/v1/email/drafts/generate/{gmail_message_id}, "
                f"method=POST, clear=true; "
                f"view, View Email, https://mail.google.com/mail/u/0/#inbox"
            ),
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    ntfy_url, content="\n".join(message_parts), headers=headers,
                )
                resp.raise_for_status()
        except Exception as e:
            logger.warning("Escalation ntfy failed for %s: %s", gmail_message_id, e)

    # 4. Create a 1-hour callback reminder
    try:
        from datetime import timedelta

        from ..services.reminders import get_reminder_service

        svc = get_reminder_service()
        if svc:
            sender = email_row.get("sender", "unknown")
            due_at = datetime.now(timezone.utc) + timedelta(hours=1)
            await svc.create_reminder(
                message=f"Follow up on escalated complaint from {sender}: {email_row.get('subject', '')}",
                due_at=due_at,
            )
    except Exception as e:
        logger.warning("Escalation reminder creation failed: %s", e)

    logger.info("Email %s escalated (contact=%s)", gmail_message_id, contact_id)
    return {
        "action": "escalate",
        "gmail_message_id": gmail_message_id,
        "priority": "urgent",
        "status": "escalated",
    }


# ---------------------------------------------------------------------------
# POST /email/actions/{gmail_message_id}/slots  -- [Show Slots]
# ---------------------------------------------------------------------------

@router.post("/{gmail_message_id}/slots")
async def show_slots(gmail_message_id: str):
    """Show available calendar slots for rescheduling.  Stub -- returns 200."""
    _ = await _load_email(gmail_message_id)
    logger.info("Slots requested for %s (stub)", gmail_message_id)
    return {
        "action": "slots",
        "gmail_message_id": gmail_message_id,
        "status": "stub",
        "message": "Calendar slot lookup not yet implemented.",
    }


# ---------------------------------------------------------------------------
# POST /email/actions/{gmail_message_id}/send-info  -- [Send Info]
# ---------------------------------------------------------------------------

@router.post("/{gmail_message_id}/send-info")
async def send_info(gmail_message_id: str):
    """Generate an informational reply draft.  Stub -- triggers standard draft."""
    _ = await _load_email(gmail_message_id)

    from .email_drafts import generate_draft

    try:
        result = await generate_draft(gmail_message_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Info draft generation failed for %s: %s", gmail_message_id, e)
        raise HTTPException(500, "Failed to generate info draft")

    logger.info("Info draft triggered for %s", gmail_message_id)
    return {
        "action": "send_info",
        "gmail_message_id": gmail_message_id,
        "draft": result,
    }


# ---------------------------------------------------------------------------
# POST /email/actions/{gmail_message_id}/archive  -- [Archive]
# ---------------------------------------------------------------------------

@router.post("/{gmail_message_id}/archive")
async def archive_email(gmail_message_id: str):
    """Archive an email -- sets priority to 'archived'.  Stub."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    result = await pool.execute(
        "UPDATE processed_emails SET priority = 'archived' WHERE gmail_message_id = $1",
        gmail_message_id,
    )

    logger.info("Email %s archived", gmail_message_id)
    return {
        "action": "archive",
        "gmail_message_id": gmail_message_id,
        "status": "archived",
    }
