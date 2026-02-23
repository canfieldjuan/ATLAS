"""
REST API for email draft management.

Provides list, view, approve, reject, edit, generate, redraft, and skip
operations for LLM-generated email reply drafts.
"""

import asyncio
import email.utils
import json
import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger("atlas.api.email_drafts")

router = APIRouter(prefix="/email/drafts", tags=["email-drafts"])

_REDRAFT_REASONS: dict[str, str] = {
    "soften_tone": (
        "The user found the tone too harsh or direct. "
        "Write a warmer, more empathetic reply with softer language and a collaborative tone."
    ),
    "be_shorter": (
        "The user found the draft too long. "
        "Write a much shorter reply - aim for half the length or less. "
        "Keep only the single most important point and a clear next step."
    ),
    "wrong_info": (
        "The user believes the draft contains incorrect or outdated information. "
        "Use the graph_context facts (verified from memory) to correct any details. "
        "Only assert facts you can verify from the original email or graph_context."
    ),
}
_REDRAFT_GUIDANCE_DEFAULT = (
    "The user rejected the previous draft. Write a substantially different reply - "
    "different opening, different tone or formality level, different proposed action."
)

def _affected_rows(result: str) -> int:
    """Parse affected row count from asyncpg execute result."""
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError, AttributeError):
        return -1


def _row_to_dict(row) -> dict:
    """Convert a DB row to a response dict."""
    d = {}
    for key in row.keys():
        val = row[key]
        if isinstance(val, UUID):
            d[key] = str(val)
        elif isinstance(val, datetime):
            d[key] = val.isoformat()
        else:
            d[key] = val
    return d


class EditDraftBody(BaseModel):
    draft_body: str | None = None
    draft_subject: str | None = None


@router.get("/")
async def list_drafts(
    status: str = Query(default="pending", description="Filter by status: pending, approved, sent, rejected, expired, or all"),
    limit: int = Query(default=50, ge=1, le=500),
):
    """List email drafts with optional status filter."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"count": 0, "drafts": []}

    if status == "all":
        rows = await pool.fetch(
            """
            SELECT id, gmail_message_id, thread_id, original_from, original_subject,
                   draft_subject, draft_body, model_provider, model_name,
                   status, approved_at, sent_at, gmail_sent_id,
                   created_at, expires_at
            FROM email_drafts
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT id, gmail_message_id, thread_id, original_from, original_subject,
                   draft_subject, draft_body, model_provider, model_name,
                   status, approved_at, sent_at, gmail_sent_id,
                   created_at, expires_at
            FROM email_drafts
            WHERE status = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            status, limit,
        )

    drafts = [_row_to_dict(r) for r in rows]
    return {"count": len(drafts), "drafts": drafts}


@router.get("/{draft_id}")
async def get_draft(draft_id: UUID):
    """Get full draft detail including original body."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    row = await pool.fetchrow(
        "SELECT * FROM email_drafts WHERE id = $1", draft_id,
    )
    if not row:
        raise HTTPException(404, "Draft not found")

    return _row_to_dict(row)


@router.post("/{draft_id}/approve")
async def approve_draft(draft_id: UUID):
    """Approve a pending draft and send it via Gmail."""
    from ..storage.database import get_db_pool
    from ..tools.gmail import get_gmail_transport

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    # Fetch and validate
    row = await pool.fetchrow(
        "SELECT * FROM email_drafts WHERE id = $1", draft_id,
    )
    if not row:
        raise HTTPException(404, "Draft not found")
    if row["status"] != "pending":
        raise HTTPException(409, f"Draft is already '{row['status']}', cannot approve")

    # Mark as approved
    now = datetime.now(timezone.utc)
    await pool.execute(
        "UPDATE email_drafts SET status = 'approved', approved_at = $1 WHERE id = $2",
        now, draft_id,
    )

    # Extract reply-to address from original_from
    original_from = row["original_from"]
    # Parse email from "Name <email@domain.com>" format
    if "<" in original_from and ">" in original_from:
        reply_to_addr = original_from.split("<")[1].split(">")[0].strip()
    else:
        reply_to_addr = original_from.strip()

    # Send via Gmail with threading
    # Use RFC 2822 Message-ID for In-Reply-To/References headers
    # (not the Gmail API hex ID which won't thread correctly cross-client)
    message_id = row.get("original_message_id") or row["gmail_message_id"]
    try:
        transport = get_gmail_transport()
        send_result = await transport.send(
            to=[reply_to_addr],
            subject=row["draft_subject"],
            body=row["draft_body"],
            thread_id=row.get("thread_id"),
            in_reply_to=message_id,
            references=message_id,
        )

        gmail_sent_id = send_result.get("id", "")
        sent_at = datetime.now(timezone.utc)

        # Fetch RFC 2822 Message-ID of the email we just sent
        sent_rfc_message_id = None
        if gmail_sent_id:
            try:
                sent_rfc_message_id = await transport.get_sent_message_id(gmail_sent_id)
            except Exception as exc:
                logger.warning("Could not retrieve sent Message-ID for %s: %s", gmail_sent_id, exc)

        await pool.execute(
            """
            UPDATE email_drafts
            SET status = 'sent', sent_at = $1, gmail_sent_id = $2, sent_message_id = $3
            WHERE id = $4
            """,
            sent_at, gmail_sent_id, sent_rfc_message_id, draft_id,
        )

        logger.info("Draft %s approved and sent: gmail_id=%s", draft_id, gmail_sent_id)

        # Log CRM interaction if the recipient is a known contact
        try:
            await _log_email_crm_interaction(reply_to_addr, row["draft_subject"], row["draft_body"])
        except Exception as crm_exc:
            logger.warning("CRM interaction log failed for draft %s: %s", draft_id, crm_exc)

        # Fire-and-forget confirmation notification (never raises)
        await _send_approval_notification(
            draft_id=str(draft_id),
            original_from=original_from,
            draft_subject=row["draft_subject"],
            sent_to=reply_to_addr,
        )

        # Emit event for reasoning agent
        from ..reasoning.producers import emit_if_enabled
        await emit_if_enabled(
            "email.draft_sent", "email_drafts",
            {"draft_id": str(draft_id), "sent_to": reply_to_addr,
             "subject": row["draft_subject"]},
        )

        return {
            "draft_id": str(draft_id),
            "status": "sent",
            "gmail_sent_id": gmail_sent_id,
            "sent_to": reply_to_addr,
        }

    except Exception as e:
        # Revert to pending on send failure
        await pool.execute(
            "UPDATE email_drafts SET status = 'pending', approved_at = NULL WHERE id = $1",
            draft_id,
        )
        logger.error("Failed to send approved draft %s: %s", draft_id, e)
        raise HTTPException(502, f"Failed to send email: {e}")


@router.post("/{draft_id}/reject")
async def reject_draft(draft_id: UUID):
    """Reject a pending draft and offer redraft option."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    # Fetch draft details before rejecting (needed for notification)
    row = await pool.fetchrow(
        "SELECT original_from, original_subject FROM email_drafts WHERE id = $1 AND status = 'pending'",
        draft_id,
    )
    if not row:
        raise HTTPException(404, "Draft not found or not pending")

    result = await pool.execute(
        """
        UPDATE email_drafts SET status = 'rejected'
        WHERE id = $1 AND status = 'pending'
        """,
        draft_id,
    )

    if _affected_rows(result) == 0:
        raise HTTPException(404, "Draft not found or not pending")

    logger.info("Draft %s rejected", draft_id)

    # Send follow-up notification with redraft option
    await _send_rejection_notification(
        draft_id=str(draft_id),
        original_from=row["original_from"],
        original_subject=row["original_subject"],
    )

    return {"draft_id": str(draft_id), "status": "rejected"}


@router.post("/{draft_id}/edit")
async def edit_draft(draft_id: UUID, body: EditDraftBody):
    """Edit a pending draft's body and/or subject."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    row = await pool.fetchrow(
        "SELECT status FROM email_drafts WHERE id = $1", draft_id,
    )
    if not row:
        raise HTTPException(404, "Draft not found")
    if row["status"] != "pending":
        raise HTTPException(409, f"Draft is '{row['status']}', can only edit pending drafts")

    updates = []
    params = []
    idx = 1

    if body.draft_body is not None:
        updates.append(f"draft_body = ${idx}")
        params.append(body.draft_body)
        idx += 1

    if body.draft_subject is not None:
        updates.append(f"draft_subject = ${idx}")
        params.append(body.draft_subject)
        idx += 1

    if not updates:
        raise HTTPException(400, "No fields to update")

    params.append(draft_id)
    query = f"UPDATE email_drafts SET {', '.join(updates)} WHERE id = ${idx}"
    await pool.execute(query, *params)

    updated = await pool.fetchrow(
        "SELECT * FROM email_drafts WHERE id = $1", draft_id,
    )
    return _row_to_dict(updated)


@router.post("/generate/{gmail_message_id}")
async def generate_draft(gmail_message_id: str):
    """On-demand draft generation triggered by ntfy [Draft Reply] button."""
    from ..config import settings
    from ..services.llm_router import get_llm
    from ..services.protocols import Message
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    cfg = settings.email_draft
    if not cfg.enabled:
        raise HTTPException(503, "Email draft system is disabled")

    # Dedup: check for existing pending/approved/sent draft for this message
    existing = await pool.fetchrow(
        """
        SELECT id, status, draft_subject FROM email_drafts
        WHERE gmail_message_id = $1 AND status IN ('pending', 'approved', 'sent')
        ORDER BY created_at DESC LIMIT 1
        """,
        gmail_message_id,
    )
    if existing:
        logger.info("Draft already exists for %s: %s", gmail_message_id, existing["id"])
        return {
            "draft_id": str(existing["id"]),
            "status": existing["status"],
            "draft_subject": existing["draft_subject"],
            "already_exists": True,
            "message": "A draft already exists for this email.",
        }

    # Get draft LLM
    llm = get_llm("email_draft")
    if llm is None:
        raise HTTPException(503, "Draft LLM not available")

    # Fetch full email via IMAP
    from ..services.email_provider import IMAPEmailProvider

    imap = IMAPEmailProvider()
    try:
        full_msg = await imap.get_message(gmail_message_id)
        if full_msg.get("error"):
            raise RuntimeError(full_msg["error"])
    except Exception as e:
        logger.error("Failed to fetch message %s via IMAP: %s", gmail_message_id, e)
        raise HTTPException(502, f"Failed to fetch email via IMAP: {e}")

    # Load skill prompt
    from ..skills import get_skill_registry

    skill = get_skill_registry().get("digest/email_draft")
    system_prompt = skill.content if skill else ""

    # Enrich with CRM context
    customer_context_str = ""
    try:
        _, sender_email = email.utils.parseaddr(full_msg.get("from", ""))
        if sender_email:
            from ..services.customer_context import get_customer_context_service
            from ..comms.action_planner import _format_customer_context
            ctx = await get_customer_context_service().get_context_by_email(sender_email)
            if not ctx.is_empty:
                customer_context_str = _format_customer_context(ctx)
    except Exception:
        logger.debug("CRM context lookup failed for draft, continuing without")

    # Enrich with graph facts
    graph_facts = await _get_sender_graph_context(full_msg.get("from", ""))

    # Check if this is a follow-up to a thread Atlas participated in
    thread_context = None
    msg_in_reply_to = full_msg.get("in_reply_to", "").strip()
    msg_references = full_msg.get("references", "").strip()
    if msg_in_reply_to or msg_references:
        candidate_ids: set[str] = set()
        if msg_in_reply_to:
            candidate_ids.add(msg_in_reply_to)
        if msg_references:
            candidate_ids.update(msg_references.split())
        if candidate_ids:
            try:
                thread_row = await pool.fetchrow(
                    """
                    SELECT ed.draft_subject, ed.draft_body, ed.sent_at,
                           pe.intent AS original_intent
                    FROM email_drafts ed
                    LEFT JOIN processed_emails pe
                        ON pe.gmail_message_id = ed.gmail_message_id
                    WHERE ed.status = 'sent'
                      AND (ed.original_message_id = ANY($1::text[])
                           OR ed.sent_message_id = ANY($1::text[]))
                    ORDER BY ed.sent_at DESC NULLS LAST
                    LIMIT 1
                    """,
                    list(candidate_ids),
                )
                if thread_row:
                    thread_context = {
                        "is_followup": True,
                        "original_intent": thread_row["original_intent"] or "unknown",
                        "our_previous_reply": (thread_row["draft_body"] or "")[:500],
                        "our_previous_subject": thread_row["draft_subject"] or "",
                    }
            except Exception as exc:
                logger.warning("Thread context lookup failed for %s: %s", gmail_message_id, exc)

    # Build LLM input
    user_input_dict = {
        "original_from": full_msg.get("from", ""),
        "original_subject": full_msg.get("subject", ""),
        "original_body": full_msg.get("body_text", ""),
        "user_name": settings.persona.owner_name,
        "user_timezone": settings.reminder.default_timezone,
    }
    if customer_context_str:
        user_input_dict["customer_context"] = customer_context_str
    if graph_facts:
        user_input_dict["graph_context"] = graph_facts
    if thread_context:
        user_input_dict["thread_context"] = thread_context

    user_input = json.dumps(user_input_dict, indent=2)

    try:
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
    except Exception as e:
        logger.error("LLM draft generation failed for %s: %s", gmail_message_id, e)
        raise HTTPException(502, f"Draft generation failed: {e}")

    if not output.strip():
        raise HTTPException(502, "LLM returned empty draft")

    # Parse and insert
    from ..autonomous.tasks.email_draft import (
        _insert_draft,
        _parse_draft_output,
        _send_draft_notification,
    )

    draft_subject, draft_body = _parse_draft_output(output)
    if not draft_subject:
        draft_subject = f"Re: {full_msg.get('subject', '')}"

    draft_id = await _insert_draft({
        "gmail_message_id": gmail_message_id,
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

    # Look up intent + confidence for auto-approve notification
    _intent = None
    _confidence = 0.0
    pe_row = await pool.fetchrow(
        "SELECT intent, action_plan FROM processed_emails WHERE gmail_message_id = $1",
        gmail_message_id,
    )
    if pe_row:
        _intent = pe_row["intent"]
        ap_raw = pe_row["action_plan"]
        if ap_raw:
            try:
                ap = json.loads(ap_raw) if isinstance(ap_raw, str) else ap_raw
                if isinstance(ap, dict):
                    _confidence = float(ap.get("confidence", 0.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    # Send notification with approve/reject buttons
    await _send_draft_notification(
        draft_id=draft_id,
        original_from=full_msg.get("from", ""),
        draft_subject=draft_subject,
        draft_body=draft_body,
        intent=_intent,
        confidence=_confidence,
    )

    logger.info("On-demand draft %s generated for message %s", draft_id, gmail_message_id)

    # Emit event for reasoning agent
    from ..reasoning.producers import emit_if_enabled
    await emit_if_enabled(
        "email.draft_generated", "email_drafts",
        {"draft_id": str(draft_id), "gmail_message_id": gmail_message_id,
         "subject": draft_subject, "from": full_msg.get("from", "")},
    )

    return {
        "draft_id": draft_id,
        "status": "pending",
        "draft_subject": draft_subject,
        "message": "Draft generated successfully.",
    }


@router.post("/{draft_id}/redraft")
async def redraft(draft_id: UUID, reason: str | None = Query(default=None)):
    """Generate a new draft after rejection with optional reason-specific guidance.

    reason values: soften_tone | be_shorter | wrong_info | None (generic)
    """
    from ..config import settings
    from ..services.llm_router import get_llm
    from ..services.protocols import Message
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    cfg = settings.email_draft
    if not cfg.enabled:
        raise HTTPException(503, "Email draft system is disabled")

    # Fetch the rejected parent draft
    parent = await pool.fetchrow(
        "SELECT * FROM email_drafts WHERE id = $1", draft_id,
    )
    if not parent:
        raise HTTPException(404, "Draft not found")
    if parent["status"] != "rejected":
        raise HTTPException(409, f"Draft is '{parent['status']}', can only redraft rejected drafts")

    # Dedup: check for existing pending draft with this parent
    existing = await pool.fetchrow(
        """
        SELECT id, status, draft_subject FROM email_drafts
        WHERE parent_draft_id = $1 AND status = 'pending'
        LIMIT 1
        """,
        draft_id,
    )
    if existing:
        return {
            "draft_id": str(existing["id"]),
            "parent_draft_id": str(draft_id),
            "status": existing["status"],
            "already_exists": True,
            "message": "A redraft already exists for this rejected draft.",
        }

    # Get draft LLM
    llm = get_llm("email_draft")
    if llm is None:
        raise HTTPException(503, "Draft LLM not available")

    # Re-fetch full email via IMAP
    from ..services.email_provider import IMAPEmailProvider

    imap = IMAPEmailProvider()
    try:
        full_msg = await imap.get_message(parent["gmail_message_id"])
        if full_msg.get("error"):
            raise RuntimeError(full_msg["error"])
    except Exception as e:
        logger.error("Failed to fetch message %s via IMAP: %s", parent["gmail_message_id"], e)
        raise HTTPException(502, f"Failed to fetch email via IMAP: {e}")

    # Load skill prompt
    from ..skills import get_skill_registry

    skill = get_skill_registry().get("digest/email_draft")
    system_prompt = skill.content if skill else ""

    parent_attempt = parent.get("attempt_number") or 1
    new_attempt = parent_attempt + 1

    # Record rejection reason on parent (only on first redraft; guard prevents overwrite)
    if reason:
        await pool.execute(
            "UPDATE email_drafts SET rejection_reason = $1 WHERE id = $2 AND rejection_reason IS NULL",
            reason, draft_id,
        )

    # Build reason-aware guidance
    guidance = _REDRAFT_REASONS.get(reason or "", _REDRAFT_GUIDANCE_DEFAULT)

    # Enrich with CRM context
    customer_context_str = ""
    try:
        _, sender_email = email.utils.parseaddr(parent["original_from"])
        if sender_email:
            from ..services.customer_context import get_customer_context_service
            from ..comms.action_planner import _format_customer_context
            ctx = await get_customer_context_service().get_context_by_email(sender_email)
            if not ctx.is_empty:
                customer_context_str = _format_customer_context(ctx)
    except Exception:
        logger.debug("CRM context lookup failed for redraft, continuing without")

    # Enrich with graph facts (all reasons, not just wrong_info)
    graph_facts = await _get_sender_graph_context(parent["original_from"])

    # Build LLM input with rejection context
    user_input_dict: dict = {
        "original_from": full_msg.get("from", ""),
        "original_subject": full_msg.get("subject", ""),
        "original_body": full_msg.get("body_text", ""),
        "user_name": settings.persona.owner_name,
        "user_timezone": settings.reminder.default_timezone,
        "redraft": True,
        "attempt_number": new_attempt,
        "previous_draft_rejected": (parent["draft_body"] or "")[:500],
        "redraft_guidance": guidance,
    }
    if customer_context_str:
        user_input_dict["customer_context"] = customer_context_str
    if graph_facts:
        user_input_dict["graph_context"] = graph_facts
        logger.info(
            "Injecting %d graph facts for redraft of %s",
            len(graph_facts), draft_id,
        )

    user_input = json.dumps(user_input_dict, indent=2)

    # Use slightly higher temperature for variety
    temperature = min(cfg.temperature + 0.1, 0.9)

    try:
        result = await asyncio.to_thread(
            llm.chat,
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_input),
            ],
            max_tokens=cfg.max_tokens,
            temperature=temperature,
        )
        output = result.get("response", "")
    except Exception as e:
        logger.error("LLM redraft failed for %s: %s", parent["gmail_message_id"], e)
        raise HTTPException(502, f"Redraft generation failed: {e}")

    if not output.strip():
        raise HTTPException(502, "LLM returned empty redraft")

    # Parse and insert with chain fields
    from ..autonomous.tasks.email_draft import (
        _parse_draft_output,
        _send_draft_notification,
    )

    draft_subject, draft_body = _parse_draft_output(output)
    if not draft_subject:
        draft_subject = f"Re: {full_msg.get('subject', '')}"

    expiry_hours = cfg.draft_expiry_hours
    new_row = await pool.fetchrow(
        """
        INSERT INTO email_drafts (
            gmail_message_id, thread_id, original_message_id,
            original_from, original_subject,
            original_body_text, draft_subject, draft_body,
            model_provider, model_name, expires_at,
            attempt_number, parent_draft_id
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            CURRENT_TIMESTAMP + make_interval(hours => $11),
            $12, $13
        )
        RETURNING id
        """,
        parent["gmail_message_id"],
        parent.get("thread_id"),
        parent.get("original_message_id"),
        parent["original_from"],
        parent["original_subject"],
        parent.get("original_body_text"),
        draft_subject,
        draft_body,
        cfg.model_provider,
        cfg.model_name,
        expiry_hours,
        new_attempt,
        draft_id,
    )
    new_draft_id = str(new_row["id"])

    # Look up intent + confidence for auto-approve notification
    _rd_intent = None
    _rd_confidence = 0.0
    pe_row2 = await pool.fetchrow(
        "SELECT intent, action_plan FROM processed_emails WHERE gmail_message_id = $1",
        parent["gmail_message_id"],
    )
    if pe_row2:
        _rd_intent = pe_row2["intent"]
        ap_raw2 = pe_row2["action_plan"]
        if ap_raw2:
            try:
                ap2 = json.loads(ap_raw2) if isinstance(ap_raw2, str) else ap_raw2
                if isinstance(ap2, dict):
                    _rd_confidence = float(ap2.get("confidence", 0.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    # Send notification with approve/reject buttons
    await _send_draft_notification(
        draft_id=new_draft_id,
        original_from=parent["original_from"],
        draft_subject=draft_subject,
        draft_body=draft_body,
        intent=_rd_intent,
        confidence=_rd_confidence,
    )

    logger.info("Redraft %s (attempt %d) generated for parent %s", new_draft_id, new_attempt, draft_id)
    return {
        "draft_id": new_draft_id,
        "parent_draft_id": str(draft_id),
        "attempt_number": new_attempt,
        "status": "pending",
        "message": "Redraft generated successfully.",
    }


@router.post("/{draft_id}/skip")
async def skip_draft(draft_id: UUID):
    """Acknowledge 'don't redraft' -- clears the ntfy notification."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if pool.is_initialized:
        row = await pool.fetchrow(
            "SELECT id FROM email_drafts WHERE id = $1", draft_id,
        )
        if not row:
            raise HTTPException(404, "Draft not found")

    logger.info("Draft %s skipped (no redraft)", draft_id)
    return {"draft_id": str(draft_id), "status": "skipped"}


async def _log_email_crm_interaction(
    reply_to_addr: str, draft_subject: str, draft_body: str,
) -> None:
    """Log a CRM interaction when a draft reply is sent to a known contact."""
    from ..services.crm_provider import get_crm_provider

    _, clean_email = email.utils.parseaddr(reply_to_addr)
    if not clean_email:
        return
    crm = get_crm_provider()
    matches = await crm.search_contacts(email=clean_email)
    if not matches:
        return  # Only log if contact exists -- no auto-create here
    await crm.log_interaction(
        contact_id=str(matches[0]["id"]),
        interaction_type="email",
        summary=f"Sent reply: {draft_subject}. {draft_body[:200]}",
    )


async def _send_approval_notification(
    draft_id: str,
    original_from: str,
    draft_subject: str,
    sent_to: str,
) -> None:
    """Push a 'sent' confirmation to ntfy after a draft is approved and sent."""
    from ..config import settings

    if not settings.email_draft.notify_drafts:
        return
    if not settings.alerts.ntfy_enabled:
        return

    import httpx

    ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"
    sender_name = original_from.split("<")[0].strip().strip('"') or original_from

    headers = {
        "Title": f"Sent: {draft_subject[:60]}",
        "Priority": "low",
        "Tags": "email,white_check_mark",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                ntfy_url,
                content=f"Reply sent to {sender_name} ({sent_to})",
                headers=headers,
            )
            resp.raise_for_status()
        logger.info("Approval confirmation sent for draft %s", draft_id)
    except Exception as e:
        logger.warning("Failed to send approval notification: %s", e)


async def _get_sender_graph_context(sender: str) -> list[str]:
    """Query Graphiti for known facts about the sender. Returns [] on any failure."""
    from ..config import settings
    try:
        from ..memory.rag_client import get_rag_client
        client = get_rag_client()
        if not await client.health_check():
            return []
        sender_name = sender.split("<")[0].strip().strip('"') or sender
        group_id = settings.memory.email_graph_group_id or settings.memory.group_id
        result = await client.search(
            query=f"facts about {sender_name}",
            group_id=group_id,
            max_facts=5,
        )
        return [s.fact for s in result.facts if s.fact]
    except Exception as e:
        logger.warning("Graph context lookup failed for %r: %s", sender, e)
        return []


async def _send_rejection_notification(
    draft_id: str,
    original_from: str,
    original_subject: str,
) -> None:
    """Send ntfy notification after rejection with feedback buttons."""
    from ..config import settings

    if not settings.email_draft.notify_drafts:
        return
    if not settings.alerts.ntfy_enabled:
        return

    import httpx

    api_url = settings.email_draft.atlas_api_url.rstrip("/")
    ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"

    sender_name = original_from.split("<")[0].strip().strip('"') or original_from

    message = (
        f"Draft rejected for: {sender_name}\n"
        f"Subject: {original_subject}\n\n"
        "How should it be fixed?"
    )

    actions = (
        f"http, Soften Tone, {api_url}/api/v1/email/drafts/{draft_id}/redraft?reason=soften_tone, method=POST, clear=true; "
        f"http, Be Shorter, {api_url}/api/v1/email/drafts/{draft_id}/redraft?reason=be_shorter, method=POST, clear=true; "
        f"http, Wrong Info, {api_url}/api/v1/email/drafts/{draft_id}/redraft?reason=wrong_info, method=POST, clear=true"
    )

    headers = {
        "Title": f"Draft Rejected: {original_subject[:60]}",
        "Priority": "default",
        "Tags": "email,x",
        "Actions": actions,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(ntfy_url, content=message, headers=headers)
            resp.raise_for_status()
        logger.info("Rejection notification sent for draft %s", draft_id)
    except Exception as e:
        logger.warning("Failed to send rejection notification: %s", e)
