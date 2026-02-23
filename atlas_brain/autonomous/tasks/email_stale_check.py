"""
Stale email re-engagement task.

Runs every 2 hours (configurable).  Detects three scenarios:
1. Pending drafts not approved/rejected after N hours -- re-notify owner
2. action_required emails with no draft or action after N hours -- escalation
3. Sent estimate replies with no customer response after N days -- auto-generate follow-up

Complaint emails are excluded from all scenarios.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.email_stale_check")


async def run(task: ScheduledTask) -> dict:
    """Detect stale email situations and take action."""
    cfg = settings.email_stale_check
    if not cfg.enabled:
        return {"_skip_synthesis": "Stale check disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "Database not available"}

    s1 = await _check_stale_drafts(pool, cfg)
    s2 = await _check_unactioned_high_priority(pool, cfg)
    s3 = await _check_unanswered_estimates(pool, cfg)

    total = s1["reminded"] + s2["escalated"] + s3["followups"]
    logger.info(
        "Stale check cycle: %d stale drafts reminded, %d unactioned escalated, %d follow-ups generated",
        s1["reminded"], s2["escalated"], s3["followups"],
    )

    return {
        "stale_drafts_reminded": s1["reminded"],
        "stale_drafts_checked": s1["checked"],
        "unactioned_escalated": s2["escalated"],
        "unactioned_checked": s2["checked"],
        "followups_generated": s3["followups"],
        "followups_checked": s3["checked"],
        "_skip_synthesis": True if total == 0 else None,
    }


# ---------------------------------------------------------------------------
# Scenario 1: Stale pending drafts
# ---------------------------------------------------------------------------

async def _check_stale_drafts(pool, cfg) -> dict[str, int]:
    """Re-notify owner about pending drafts that have been sitting too long."""
    rows = await pool.fetch(
        """
        SELECT ed.id, ed.gmail_message_id, ed.original_from, ed.draft_subject,
               ed.draft_body, ed.created_at, ed.metadata, pe.intent
        FROM email_drafts ed
        LEFT JOIN processed_emails pe ON ed.gmail_message_id = pe.gmail_message_id
        WHERE ed.status = 'pending'
          AND ed.parent_draft_id IS NULL
          AND ed.created_at <= NOW() - make_interval(hours => $1)
          AND COALESCE((ed.metadata->>'stale_reminders_sent')::int, 0) < $2
        ORDER BY ed.created_at ASC
        """,
        cfg.stale_draft_hours,
        cfg.max_reminders,
    )

    reminded = 0
    for row in rows:
        intent = row["intent"]
        if intent == "complaint":
            continue

        draft_id = str(row["id"])
        meta = row["metadata"] or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}

        reminders_sent = int(meta.get("stale_reminders_sent", 0))

        await _send_stale_draft_notification(
            draft_id=draft_id,
            original_from=row["original_from"],
            draft_subject=row["draft_subject"],
            draft_body=row["draft_body"] or "",
            reminder_number=reminders_sent + 1,
        )

        # Increment counter in metadata
        meta["stale_reminders_sent"] = reminders_sent + 1
        meta["last_stale_reminder_at"] = datetime.now(timezone.utc).isoformat()
        await pool.execute(
            "UPDATE email_drafts SET metadata = $1::jsonb WHERE id = $2",
            json.dumps(meta), row["id"],
        )
        reminded += 1

    return {"reminded": reminded, "checked": len(rows)}


async def _send_stale_draft_notification(
    draft_id: str,
    original_from: str,
    draft_subject: str,
    draft_body: str,
    reminder_number: int,
) -> None:
    """Send ntfy reminder for a stale pending draft."""
    if not settings.alerts.ntfy_enabled:
        return

    api_url = settings.email_draft.atlas_api_url.rstrip("/")
    ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"

    sender_name = original_from.split("<")[0].strip().strip('"') or original_from
    preview = draft_body[:300] + ("..." if len(draft_body) > 300 else "")

    message = (
        f"Pending draft waiting for review ({reminder_number}x reminded)\n"
        f"From: {sender_name}\n"
        f"Subject: {draft_subject}\n\n"
        f"{preview}"
    )

    actions = (
        f"http, Approve, {api_url}/api/v1/email/drafts/{draft_id}/approve, method=POST, clear=true; "
        f"http, Reject, {api_url}/api/v1/email/drafts/{draft_id}/reject, method=POST, clear=true; "
        f"view, View Full, {api_url}/api/v1/email/drafts/{draft_id}"
    )

    priority = "high" if reminder_number >= 2 else "default"

    headers = {
        "Title": f"Stale Draft: {draft_subject[:60]}",
        "Priority": priority,
        "Tags": "email,hourglass",
        "Actions": actions,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(ntfy_url, content=message, headers=headers)
            resp.raise_for_status()
        logger.info("Stale draft reminder sent for %s (reminder #%d)", draft_id, reminder_number)
    except Exception as e:
        logger.warning("Failed to send stale draft notification: %s", e)


# ---------------------------------------------------------------------------
# Scenario 2: Unactioned high-priority emails
# ---------------------------------------------------------------------------

async def _check_unactioned_high_priority(pool, cfg) -> dict[str, int]:
    """Escalate action_required emails with no draft or action taken."""
    rows = await pool.fetch(
        """
        SELECT pe.id, pe.gmail_message_id, pe.sender, pe.subject,
               pe.intent, pe.processed_at, pe.stale_check_metadata
        FROM processed_emails pe
        LEFT JOIN email_drafts ed
            ON pe.gmail_message_id = ed.gmail_message_id
            AND ed.status IN ('pending', 'approved', 'sent')
        WHERE pe.priority = 'action_required'
          AND pe.processed_at <= NOW() - make_interval(hours => $1)
          AND ed.id IS NULL
          AND COALESCE(pe.intent, '') != 'complaint'
          AND COALESCE((pe.stale_check_metadata->>'stale_reminders_sent')::int, 0) < $2
        ORDER BY pe.processed_at ASC
        """,
        cfg.unactioned_hours,
        cfg.max_reminders,
    )

    escalated = 0
    for row in rows:
        msg_id = row["gmail_message_id"]
        meta = row["stale_check_metadata"] or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}

        reminders_sent = int(meta.get("stale_reminders_sent", 0))

        await _send_unactioned_notification(
            gmail_message_id=msg_id,
            sender=row["sender"] or "unknown",
            subject=row["subject"] or "(no subject)",
            intent=row["intent"],
            reminder_number=reminders_sent + 1,
        )

        meta["stale_reminders_sent"] = reminders_sent + 1
        meta["last_stale_reminder_at"] = datetime.now(timezone.utc).isoformat()
        await pool.execute(
            "UPDATE processed_emails SET stale_check_metadata = $1::jsonb WHERE id = $2",
            json.dumps(meta), row["id"],
        )
        escalated += 1

    return {"escalated": escalated, "checked": len(rows)}


async def _send_unactioned_notification(
    gmail_message_id: str,
    sender: str,
    subject: str,
    intent: str | None,
    reminder_number: int,
) -> None:
    """Send ntfy escalation for an unactioned high-priority email."""
    if not settings.alerts.ntfy_enabled:
        return

    api_url = settings.email_draft.atlas_api_url.rstrip("/")
    ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"

    sender_name = sender.split("<")[0].strip().strip('"') or sender
    intent_label = intent or "unknown"

    message = (
        f"Action-required email has no draft or action ({reminder_number}x reminded)\n"
        f"From: {sender_name}\n"
        f"Subject: {subject}\n"
        f"Intent: {intent_label}"
    )

    actions = (
        f"http, Draft Reply, {api_url}/api/v1/email/drafts/generate/{gmail_message_id}, method=POST, clear=true; "
        f"view, View, {api_url}/api/v1/email/drafts/generate/{gmail_message_id}"
    )

    headers = {
        "Title": f"Unactioned: {subject[:60]}",
        "Priority": "high",
        "Tags": "email,warning,hourglass",
        "Actions": actions,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(ntfy_url, content=message, headers=headers)
            resp.raise_for_status()
        logger.info("Unactioned escalation sent for %s (reminder #%d)", gmail_message_id, reminder_number)
    except Exception as e:
        logger.warning("Failed to send unactioned notification: %s", e)


# ---------------------------------------------------------------------------
# Scenario 3: Unanswered estimate replies
# ---------------------------------------------------------------------------

async def _check_unanswered_estimates(pool, cfg) -> dict[str, int]:
    """Generate 'checking in' follow-up drafts for unanswered sent replies."""
    rows = await pool.fetch(
        """
        SELECT ed.id AS draft_id, ed.gmail_message_id, ed.thread_id,
               ed.original_message_id, ed.original_from,
               ed.original_subject, ed.draft_body, ed.sent_at, ed.metadata,
               pe.intent, pe.contact_id
        FROM email_drafts ed
        JOIN processed_emails pe ON ed.gmail_message_id = pe.gmail_message_id
        LEFT JOIN processed_emails follow ON follow.followup_of_draft_id = ed.id
        WHERE ed.status = 'sent'
          AND pe.intent = ANY($1::text[])
          AND pe.intent != 'complaint'
          AND ed.sent_at <= NOW() - make_interval(days => $2)
          AND follow.id IS NULL
          AND COALESCE((ed.metadata->>'followup_generated')::boolean, false) = false
        ORDER BY ed.sent_at ASC
        LIMIT $3
        """,
        cfg.unanswered_intents,
        cfg.unanswered_days,
        cfg.max_followups_per_cycle,
    )

    followups = 0
    for row in rows:
        parent_draft_id = row["draft_id"]
        try:
            await _generate_followup_draft(pool, row)
            followups += 1
        except Exception as exc:
            logger.warning(
                "Failed to generate follow-up for draft %s: %s",
                parent_draft_id, exc,
            )

    return {"followups": followups, "checked": len(rows)}


async def _generate_followup_draft(pool, row: Any) -> None:
    """Generate a 'checking in' draft via LLM and insert it."""
    from ...services.llm_router import get_llm
    from ...services.protocols import Message
    from ...skills import get_skill_registry
    from .email_draft import _insert_draft, _parse_draft_output, _send_draft_notification

    cfg = settings.email_draft
    llm = get_llm("email_draft")
    if llm is None:
        raise RuntimeError("Draft LLM not available")

    skill = get_skill_registry().get("digest/email_followup_checkin")
    system_prompt = skill.content if skill else ""

    sent_at = row["sent_at"]
    days_since = (datetime.now(timezone.utc) - sent_at).days if sent_at else 0

    user_input = json.dumps({
        "original_from": row["original_from"],
        "original_subject": row["original_subject"],
        "our_previous_reply": (row["draft_body"] or "")[:500],
        "days_since_reply": days_since,
        "user_name": settings.persona.owner_name,
    }, indent=2)

    result = await asyncio.to_thread(
        llm.chat,
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_input),
        ],
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    output = result.get("response", "")
    if not output.strip():
        raise RuntimeError("LLM returned empty follow-up draft")

    draft_subject, draft_body = _parse_draft_output(output)
    if not draft_subject:
        draft_subject = f"Re: {row['original_subject']}"

    # Insert with confidence=0.0 so auto-approve won't pick it up
    draft_id = await _insert_draft({
        "gmail_message_id": row["gmail_message_id"],
        "thread_id": row["thread_id"],
        "original_message_id": row["original_message_id"],
        "original_from": row["original_from"],
        "original_subject": row["original_subject"],
        "draft_subject": draft_subject,
        "draft_body": draft_body,
        "model_provider": cfg.model_provider,
        "model_name": cfg.model_name,
    })

    # Notify with standard draft buttons (human review required)
    await _send_draft_notification(
        draft_id=draft_id,
        original_from=row["original_from"],
        draft_subject=draft_subject,
        draft_body=draft_body,
        intent=row["intent"],
        confidence=0.0,
    )

    # Mark the original draft so we don't generate duplicates
    parent_draft_id = row["draft_id"]
    meta = row["metadata"] or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            meta = {}

    meta["followup_generated"] = True
    meta["followup_draft_id"] = draft_id
    meta["followup_generated_at"] = datetime.now(timezone.utc).isoformat()
    await pool.execute(
        "UPDATE email_drafts SET metadata = $1::jsonb WHERE id = $2",
        json.dumps(meta), parent_draft_id,
    )

    logger.info(
        "Generated follow-up draft %s for parent %s (%s)",
        draft_id, parent_draft_id, row["original_subject"],
    )
