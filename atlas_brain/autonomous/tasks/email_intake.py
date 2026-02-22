"""
Near-real-time email intake pipeline.

Polls the inbox via IMAP every N minutes, classifies emails,
cross-references action_required + lead emails against the CRM,
generates LLM action plans for known-contact emails, and sends
enriched ntfy notifications.
"""

import asyncio
import email.utils
import json
import logging
import re
from typing import Any

import httpx

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from .gmail_digest import (
    _get_processed_message_ids,
    _process_lead_emails,
    _send_action_email_notifications,
)
from .email_classifier import get_email_classifier

logger = logging.getLogger("atlas.autonomous.tasks.email_intake")


# ---------------------------------------------------------------------------
# CRM cross-reference
# ---------------------------------------------------------------------------

async def _crm_cross_reference(emails: list[dict[str, Any]]) -> int:
    """Cross-reference action_required/lead emails against the CRM.

    For each match, stashes _contact_id, _customer_context, and
    _customer_summary on the email dict.  Returns count of matches.
    Fail-open per email.
    """
    from ...services.customer_context import get_customer_context_service

    svc = get_customer_context_service()
    matched = 0

    for e in emails:
        priority = e.get("priority", "")
        category = e.get("category", "")
        if priority != "action_required" and category != "lead":
            continue

        try:
            _, sender_email = email.utils.parseaddr(e.get("from", ""))
            if not sender_email:
                continue

            ctx = await svc.get_context_by_email(sender_email)
            if ctx.is_empty:
                continue

            e["_contact_id"] = ctx.contact_id
            e["_customer_context"] = ctx

            name = ctx.display_name
            ctype = ctx.contact.get("contact_type", "customer")
            n_interactions = len(ctx.interactions) if ctx.interactions else 0
            e["_customer_summary"] = (
                f"Returning customer: {name}, {ctype}, {n_interactions} past interactions"
            )
            matched += 1

        except Exception as exc:
            logger.warning(
                "CRM cross-ref failed for email %s: %s", e.get("id"), exc
            )

    return matched


# ---------------------------------------------------------------------------
# LLM action plan generation
# ---------------------------------------------------------------------------

async def _generate_action_plans(emails: list[dict[str, Any]]) -> int:
    """Generate LLM action plans for CRM-matched emails.

    Only processes emails that have _customer_context set.
    Respects max_action_plans_per_cycle cap.  Returns count of plans generated.
    """
    from ...comms.action_planner import _format_customer_context, _parse_plan
    from ...skills import get_skill_registry
    from ...services.llm_router import get_triage_llm
    from ...services import llm_registry
    from ...services.protocols import Message

    cfg = settings.email_intake
    if not cfg.action_plan_enabled:
        return 0

    llm = get_triage_llm() or llm_registry.get_active()
    if not llm:
        logger.warning("No LLM available for email action planning")
        return 0

    skill = get_skill_registry().get("digest/email_action_planning")
    plan_count = 0

    for e in emails:
        if plan_count >= cfg.max_action_plans_per_cycle:
            break

        ctx = e.get("_customer_context")
        if ctx is None:
            continue

        try:
            customer_context_str = _format_customer_context(ctx)
            body = e.get("body_text", "")[:1500]

            if skill:
                system_prompt = (
                    skill.content
                    .replace("{customer_context}", customer_context_str)
                    .replace("{email_from}", e.get("from", ""))
                    .replace("{email_subject}", e.get("subject", ""))
                    .replace("{email_category}", e.get("category", "other"))
                    .replace("{email_body}", body)
                )
            else:
                system_prompt = (
                    f"You are Atlas, planning follow-up actions for an email from a known customer.\n"
                    f"Customer: {customer_context_str}\n"
                    f"From: {e.get('from', '')}\n"
                    f"Subject: {e.get('subject', '')}\n"
                    f"Category: {e.get('category', 'other')}\n"
                    f"Body: {body}\n"
                    f"Return a JSON array of actions: [{{action, priority, params, rationale}}]"
                )

            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content="Generate the action plan for this email."),
            ]

            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda msgs=messages: llm.chat(
                        messages=msgs,
                        max_tokens=512,
                        temperature=0.3,
                    ),
                ),
                timeout=30,
            )

            text = result.get("response", "").strip()
            if not text:
                continue

            # Strip <think> tags (Qwen3)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            plan = _parse_plan(text, {})
            if plan and plan[0].get("action") != "none":
                e["_action_plan"] = plan
                plan_count += 1
                logger.info(
                    "Action plan generated for email %s: %d actions",
                    e.get("id"), len(plan),
                )

        except Exception as exc:
            logger.warning(
                "Action plan generation failed for email %s: %s",
                e.get("id"), exc,
            )

    return plan_count


# ---------------------------------------------------------------------------
# Enriched notifications
# ---------------------------------------------------------------------------

async def _send_enriched_notifications(emails: list[dict[str, Any]]) -> None:
    """Send ntfy notifications with CRM context and action plan summaries.

    Enriches action_required/lead emails that have CRM matches with
    customer context and suggested actions.  Non-CRM emails fall through
    to the standard notification path.
    """
    if not settings.alerts.ntfy_enabled:
        return
    if not settings.email_draft.enabled:
        return

    api_url = settings.email_draft.atlas_api_url.rstrip("/")
    ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"

    # Non-enriched emails: delegate to the standard notifier
    standard_emails = []

    for e in emails:
        if e.get("priority") != "action_required" and e.get("category") != "lead":
            continue
        if e.get("replyable") is False:
            continue

        customer_summary = e.get("_customer_summary")
        action_plan = e.get("_action_plan")

        # No CRM enrichment -- use standard notification path
        if not customer_summary and not action_plan:
            standard_emails.append(e)
            continue

        msg_id = e.get("id", "")
        if not msg_id:
            continue

        sender = e.get("from", "unknown")
        subject = e.get("subject", "(no subject)")
        body_snippet = (e.get("body_text") or e.get("snippet") or "")[:200]
        if len(body_snippet) == 200:
            body_snippet += "..."

        sender_name = sender.split("<")[0].strip().strip('"') or sender

        is_lead = e.get("category") == "lead"
        if is_lead:
            lead_name = e.get("_lead_name", "")
            lead_email = e.get("_lead_email", "")
            if lead_name or lead_email:
                sender_name = lead_name or lead_email

        # Build enriched message body
        parts = [f"From: {sender_name}", f"Subject: {subject}"]
        if customer_summary:
            parts.append(f"\n{customer_summary}")
        if action_plan:
            actions_str = ", ".join(a.get("action", "?") for a in action_plan[:3])
            parts.append(f"Suggested: {actions_str}")
        parts.append(f"\n{body_snippet}")
        message = "\n".join(parts)

        rfc_msg_id = e.get("message_id", "").strip("<>")
        view_url = (
            f"https://mail.google.com/mail/u/0/#search/rfc822msgid:{rfc_msg_id}"
            if rfc_msg_id
            else "https://mail.google.com/mail/u/0/#inbox"
        )
        actions = (
            f"http, Draft Reply, {api_url}/api/v1/email/drafts/generate/{msg_id}, method=POST, clear=true; "
            f"view, View Email, {view_url}"
        )

        if is_lead:
            ntfy_title = f"New Lead: {subject[:60]}"
            ntfy_tags = "email,star"
        else:
            ntfy_title = f"Action Required: {subject[:60]}"
            ntfy_tags = "email,warning"

        headers = {
            "Title": ntfy_title,
            "Priority": "high",
            "Tags": ntfy_tags,
            "Actions": actions,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(ntfy_url, content=message, headers=headers)
                resp.raise_for_status()
            logger.info(
                "Enriched notification sent for %s: %s", msg_id, subject[:40]
            )
        except Exception as exc:
            logger.warning(
                "Failed to send enriched notification for %s: %s",
                msg_id, exc,
            )

    # Send standard notifications for non-enriched emails
    if standard_emails:
        await _send_action_email_notifications(standard_emails)


# ---------------------------------------------------------------------------
# Extended DB recording
# ---------------------------------------------------------------------------

async def _record_with_action_plans(emails: list[dict[str, Any]]) -> None:
    """Record processed emails with action_plan and customer_context_summary columns."""
    pool = get_db_pool()
    if not pool.is_initialized or not emails:
        return

    try:
        async with pool.transaction() as conn:
            await conn.executemany(
                """
                INSERT INTO processed_emails
                    (gmail_message_id, sender, subject, category, priority,
                     replyable, contact_id, action_plan, customer_context_summary)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (gmail_message_id) DO NOTHING
                """,
                [
                    (
                        e["id"],
                        e.get("from", ""),
                        e.get("subject", ""),
                        e.get("category"),
                        e.get("priority"),
                        e.get("replyable"),
                        e.get("_contact_id"),
                        json.dumps(e["_action_plan"]) if e.get("_action_plan") else None,
                        e.get("_customer_summary"),
                    )
                    for e in emails
                ],
            )
        logger.debug("Recorded %d processed email IDs (with action plans)", len(emails))
    except Exception as e:
        logger.warning("Failed to record processed emails: %s", e)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run(task: ScheduledTask) -> dict:
    """Near-real-time email intake: fetch via IMAP, classify, CRM xref, action plan, notify."""
    cfg = settings.email_intake
    if not cfg.enabled:
        return {"_skip_synthesis": "Email intake disabled"}

    # Check IMAP is configured
    from ...services.email_provider import IMAPEmailProvider

    provider = IMAPEmailProvider()
    if not provider.is_configured():
        return {"_skip_synthesis": "IMAP not configured"}

    query = settings.tools.gmail_query
    max_results = settings.tools.gmail_max_results

    # 1. Fetch unread email metadata via IMAP
    try:
        messages = await provider.list_messages(
            query=query, max_results=max_results,
        )
    except Exception as e:
        logger.error("Email intake: failed to list messages: %s", e)
        return {"_skip_synthesis": f"IMAP error: {type(e).__name__}"}

    if not messages:
        return {"_skip_synthesis": True, "new_emails": 0}

    # 2. Dedup against already-processed (by IMAP UID stored in gmail_message_id)
    all_ids = [m["id"] for m in messages]
    already_processed = await _get_processed_message_ids(all_ids)
    new_messages = [m for m in messages if m["id"] not in already_processed]

    if not new_messages:
        return {"_skip_synthesis": True, "new_emails": 0, "already_processed": len(already_processed)}

    logger.info(
        "Email intake: %d unread, %d already processed, %d new",
        len(messages), len(already_processed), len(new_messages),
    )

    # 3. Fetch full body for new messages via IMAP
    max_body = settings.tools.gmail_body_max_chars
    emails: list[dict[str, Any]] = []
    batch_size = settings.autonomous.gmail_digest_batch_size
    for i in range(0, len(new_messages), batch_size):
        batch = new_messages[i : i + batch_size]
        tasks = [provider.get_message(m["id"]) for m in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Failed to fetch message: %s", r)
            elif r.get("error"):
                logger.warning("IMAP fetch error: %s", r["error"])
            else:
                # Truncate body to configured limit
                body = r.get("body_text", "")
                if len(body) > max_body:
                    r["body_text"] = body[:max_body] + "..."
                emails.append(r)

    if not emails:
        return {"_skip_synthesis": True, "new_emails": 0}

    # 4. Classify (rule-based, no LLM)
    classifier = get_email_classifier()
    emails = classifier.classify_batch(emails)

    # 5. Process leads (CRM contact creation for web form submissions)
    await _process_lead_emails(emails)

    # 6. CRM cross-reference for action_required + lead emails
    crm_count = 0
    if cfg.crm_enabled:
        crm_count = await _crm_cross_reference(emails)

    # 7. Generate action plans for CRM-matched emails
    plan_count = 0
    if cfg.action_plan_enabled and crm_count > 0:
        plan_count = await _generate_action_plans(emails)

    # 8. Record to DB (extended INSERT with action_plan + customer_context_summary)
    await _record_with_action_plans(emails)

    # 9. Send enriched ntfy notifications
    action_emails = [
        e for e in emails
        if (e.get("priority") == "action_required" or e.get("category") == "lead")
        and e.get("replyable") is not False
    ]
    if action_emails:
        await _send_enriched_notifications(action_emails)

    logger.info(
        "Email intake: %d new, %d CRM matched, %d action plans",
        len(emails), crm_count, plan_count,
    )

    return {
        "new_emails": len(emails),
        "crm_matched": crm_count,
        "action_plans": plan_count,
        "_skip_synthesis": True,
    }
