"""
Email draft generation builtin task.

For action_required emails that don't already have a draft,
generates a reply draft via the Anthropic LLM and sends an
ntfy notification with approve/reject buttons.
"""

import json
import logging
from typing import Any

from ...config import settings
from ...services.llm_router import get_llm
from ...services.protocols import Message
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.email_draft")


async def _get_draftable_emails() -> list[dict[str, Any]]:
    """Find action_required emails from the last 48h with no existing draft."""
    pool = get_db_pool()
    if not pool.is_initialized:
        return []

    priorities = settings.email_draft.auto_draft_priorities

    rows = await pool.fetch(
        """
        SELECT pe.gmail_message_id, pe.sender, pe.subject, pe.category, pe.priority, pe.replyable
        FROM processed_emails pe
        LEFT JOIN email_drafts ed ON pe.gmail_message_id = ed.gmail_message_id
            AND ed.status IN ('pending', 'approved', 'sent')
        WHERE pe.priority = ANY($1::text[])
          AND pe.processed_at > CURRENT_TIMESTAMP - INTERVAL '48 hours'
          AND pe.replyable IS NOT FALSE
          AND ed.id IS NULL
        ORDER BY pe.processed_at DESC
        """,
        priorities,
    )

    return [dict(r) for r in rows]


async def _triage_ambiguous_emails(
    ambiguous: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use LLM triage to classify ambiguous emails as replyable or not.

    For each email where replyable IS NULL, calls the triage LLM (Haiku)
    for a cheap yes/no classification, then caches the result in the DB.

    Returns the subset of emails that are replyable.
    """
    if not ambiguous:
        return []

    llm = get_llm("email_triage")
    if llm is None:
        # No triage LLM -- default to replyable (draft anyway, user can reject)
        logger.warning("Triage LLM not available; defaulting %d ambiguous emails to replyable", len(ambiguous))
        return ambiguous

    from ...skills import get_skill_registry

    skill = get_skill_registry().get("digest/email_replyable")
    system_prompt = skill.content if skill else "Answer YES or NO: does this email expect a human reply?"

    # Fetch full messages for body snippets
    from .gmail_digest import _get_gmail_client

    gmail = await _get_gmail_client()
    pool = get_db_pool()
    replyable_emails: list[dict[str, Any]] = []

    logger.info("Triage: %d ambiguous emails to classify", len(ambiguous))

    for email_row in ambiguous:
        msg_id = email_row["gmail_message_id"]

        # Get body snippet (first 200 chars)
        body_snippet = ""
        try:
            full_msg = await gmail.get_message_full(msg_id)
            body_snippet = (full_msg.get("body_text") or "")[:200]
        except Exception as e:
            logger.warning("Triage: failed to fetch body for %s: %s", msg_id, e)

        user_input = json.dumps({
            "sender": email_row.get("sender", ""),
            "subject": email_row.get("subject", ""),
            "body_snippet": body_snippet,
        })

        try:
            result = llm.chat(
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=user_input),
                ],
                max_tokens=settings.email_draft.triage_max_tokens,
                temperature=0.0,
            )
            answer = (result.get("response") or "").strip().upper()
            is_replyable = answer.startswith("YES")
        except Exception as e:
            logger.warning("Triage LLM failed for %s: %s; defaulting to replyable", msg_id, e)
            is_replyable = True

        logger.info("Triage result for %s: %s", msg_id, "YES" if is_replyable else "NO")

        # Cache result in DB
        if pool.is_initialized:
            try:
                await pool.execute(
                    "UPDATE processed_emails SET replyable = $1 WHERE gmail_message_id = $2",
                    is_replyable,
                    msg_id,
                )
                logger.debug("Updated replyable for %s", msg_id)
            except Exception as e:
                logger.warning("Failed to cache triage result for %s: %s", msg_id, e)

        if is_replyable:
            replyable_emails.append(email_row)

    return replyable_emails


async def _insert_draft(draft: dict[str, Any]) -> str:
    """Insert a draft row and return its UUID."""
    pool = get_db_pool()

    expiry_hours = settings.email_draft.draft_expiry_hours
    row = await pool.fetchrow(
        """
        INSERT INTO email_drafts (
            gmail_message_id, thread_id, original_message_id,
            original_from, original_subject,
            original_body_text, draft_subject, draft_body,
            model_provider, model_name, expires_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            CURRENT_TIMESTAMP + make_interval(hours => $11)
        )
        RETURNING id
        """,
        draft["gmail_message_id"],
        draft.get("thread_id"),
        draft.get("original_message_id"),
        draft["original_from"],
        draft["original_subject"],
        draft.get("original_body_text"),
        draft["draft_subject"],
        draft["draft_body"],
        draft["model_provider"],
        draft["model_name"],
        expiry_hours,
    )
    return str(row["id"])


def _parse_draft_output(output: str) -> tuple[str, str]:
    """Parse the LLM output into (subject, body).

    Expected format:
        SUBJECT: Re: ...
        ---
        body text here
    """
    # Find the --- separator
    parts = output.split("---", 1)
    if len(parts) == 2:
        subject_line = parts[0].strip()
        body = parts[1].strip()

        # Extract subject from "SUBJECT: ..." line
        if subject_line.upper().startswith("SUBJECT:"):
            subject = subject_line[len("SUBJECT:"):].strip()
        else:
            subject = subject_line

        return subject, body

    # Fallback: use entire output as body
    return "", output.strip()


async def _send_draft_notification(
    draft_id: str,
    original_from: str,
    draft_subject: str,
    draft_body: str,
) -> None:
    """Send ntfy notification with approve/reject action buttons."""
    if not settings.email_draft.notify_drafts:
        return
    if not settings.alerts.ntfy_enabled:
        return

    import httpx

    api_url = settings.email_draft.atlas_api_url.rstrip("/")
    ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"

    # Truncate body for notification
    preview = draft_body[:300] + ("..." if len(draft_body) > 300 else "")
    sender_name = original_from.split("<")[0].strip().strip('"') or original_from

    message = f"From: {sender_name}\nSubject: {draft_subject}\n\n{preview}"

    # ntfy Actions header: semicolon-separated action definitions
    actions = (
        f"http, Approve, {api_url}/api/v1/email/drafts/{draft_id}/approve, method=POST, clear=true; "
        f"http, Reject, {api_url}/api/v1/email/drafts/{draft_id}/reject, method=POST, clear=true; "
        f"view, View Full, {api_url}/api/v1/email/drafts/{draft_id}"
    )

    headers = {
        "Title": f"Email Draft: {draft_subject[:60]}",
        "Priority": "default",
        "Tags": "email,draft",
        "Actions": actions,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(ntfy_url, content=message, headers=headers)
            resp.raise_for_status()
        logger.info("Draft notification sent for %s", draft_id)
    except Exception as e:
        logger.warning("Failed to send draft notification: %s", e)


async def run(task: ScheduledTask) -> dict:
    """Generate reply drafts for action-required emails."""
    cfg = settings.email_draft

    if not cfg.enabled:
        return {
            "drafts_generated": 0,
            "drafts": [],
            "_skip_synthesis": "Email draft generation is disabled.",
        }

    if not cfg.auto_draft_enabled:
        return {
            "drafts_generated": 0,
            "drafts": [],
            "_skip_synthesis": "Auto-draft disabled (user-initiated mode via ntfy).",
        }

    # Get the draft LLM
    llm = get_llm("email_draft")
    if llm is None:
        return {
            "drafts_generated": 0,
            "drafts": [],
            "_skip_synthesis": "Draft LLM not available.",
        }

    # Find emails that need drafts (replyable=True or NULL)
    candidates = await _get_draftable_emails()
    if not candidates:
        return {
            "drafts_generated": 0,
            "drafts": [],
            "_skip_synthesis": "No action-required emails need drafting.",
        }

    # Split into confirmed replyable and ambiguous (needs triage)
    confirmed = [e for e in candidates if e.get("replyable") is True]
    ambiguous = [e for e in candidates if e.get("replyable") is None]

    # Triage ambiguous emails via LLM
    triaged = await _triage_ambiguous_emails(ambiguous)
    draftable = confirmed + triaged

    if not draftable:
        return {
            "drafts_generated": 0,
            "drafts": [],
            "_skip_synthesis": "No replyable emails found after triage.",
        }

    logger.info("Found %d emails to draft replies for (%d confirmed, %d triaged)",
                len(draftable), len(confirmed), len(triaged))

    # Load the draft skill prompt
    from ...skills import get_skill_registry
    skill = get_skill_registry().get("digest/email_draft")
    system_prompt = skill.content if skill else ""

    # Fetch full message bodies from Gmail
    from .gmail_digest import _get_gmail_client

    gmail = await _get_gmail_client()
    drafts_created = []

    for email_meta in draftable:
        msg_id = email_meta["gmail_message_id"]
        try:
            full_msg = await gmail.get_message_full(msg_id)
        except Exception as e:
            logger.warning("Failed to fetch full message %s: %s", msg_id, e)
            continue

        # Build LLM prompt
        user_input = json.dumps({
            "original_from": full_msg.get("from", ""),
            "original_subject": full_msg.get("subject", ""),
            "original_body": full_msg.get("body_text", ""),
            "user_name": settings.persona.owner_name,
            "user_timezone": settings.reminder.default_timezone,
        }, indent=2)

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_input),
        ]

        try:
            result = llm.chat(
                messages=messages,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            )
            output = result.get("response", "")
        except Exception as e:
            logger.error("LLM draft generation failed for %s: %s", msg_id, e)
            continue

        if not output.strip():
            logger.warning("LLM returned empty draft for %s", msg_id)
            continue

        # Parse subject and body from output
        draft_subject, draft_body = _parse_draft_output(output)
        if not draft_subject:
            draft_subject = f"Re: {full_msg.get('subject', '')}"

        # Insert draft into DB
        draft_id = await _insert_draft({
            "gmail_message_id": msg_id,
            "thread_id": full_msg.get("thread_id"),
            "original_message_id": full_msg.get("message_id"),
            "original_from": full_msg.get("from", ""),
            "original_subject": full_msg.get("subject", ""),
            "original_body_text": full_msg.get("body_text", ""),
            "draft_subject": draft_subject,
            "draft_body": draft_body,
            "model_provider": cfg.model_provider,
            "model_name": cfg.model_name,
        })

        # Send notification
        await _send_draft_notification(
            draft_id=draft_id,
            original_from=full_msg.get("from", ""),
            draft_subject=draft_subject,
            draft_body=draft_body,
        )

        drafts_created.append({
            "draft_id": draft_id,
            "gmail_message_id": msg_id,
            "subject": draft_subject,
        })

        logger.info("Created draft %s for message %s", draft_id, msg_id)

    return {
        "drafts_generated": len(drafts_created),
        "drafts": drafts_created,
        "_skip_synthesis": None if drafts_created else "No drafts generated.",
    }
