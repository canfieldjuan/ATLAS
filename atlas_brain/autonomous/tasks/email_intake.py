"""
Near-real-time email intake pipeline.

Polls the inbox via IMAP every N minutes, classifies emails (Stage 1 rule-based),
cross-references against the CRM, runs Stage 2 LLM intent classification +
action planning, and sends intent-aware ntfy notifications.
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
# Valid business intents (Stage 2)
# ---------------------------------------------------------------------------

VALID_INTENTS = {"estimate_request", "reschedule", "complaint", "info_admin"}

_INTENT_LABELS = {
    "estimate_request": "Estimate Request",
    "reschedule": "Reschedule",
    "complaint": "Complaint",
    "info_admin": "Info/Admin",
}

_INTENT_NTFY = {
    "estimate_request": {
        "buttons": [
            ("http", "Get Quote", "/api/v1/email/actions/{id}/quote"),
            ("http", "Draft Reply", "/api/v1/email/drafts/generate/{id}"),
        ],
        "tags": "email,dollar",
        "priority": "high",
    },
    "reschedule": {
        "buttons": [
            ("http", "Show Slots", "/api/v1/email/actions/{id}/slots"),
            ("http", "Draft Reply", "/api/v1/email/drafts/generate/{id}"),
        ],
        "tags": "email,calendar",
        "priority": "high",
    },
    "complaint": {
        "buttons": [
            ("http", "Escalate", "/api/v1/email/actions/{id}/escalate"),
            ("http", "Draft Reply", "/api/v1/email/drafts/generate/{id}"),
        ],
        "tags": "email,rotating_light",
        "priority": "urgent",
    },
    "info_admin": {
        "buttons": [
            ("http", "Send Info", "/api/v1/email/actions/{id}/send-info"),
            ("http", "Archive", "/api/v1/email/actions/{id}/archive"),
        ],
        "tags": "email,information_source",
        "priority": "default",
    },
}


# ---------------------------------------------------------------------------
# CRM cross-reference (expanded scope)
# ---------------------------------------------------------------------------

async def _crm_cross_reference(emails: list[dict[str, Any]]) -> int:
    """Cross-reference emails against the CRM.

    Runs on ALL emails where replyable != False (expanded from just
    action_required/lead).  For each match, stashes _contact_id,
    _customer_context, and _customer_summary on the email dict.
    Returns count of matches.  Fail-open per email.
    """
    from ...services.customer_context import get_customer_context_service
    from ...services.crm_provider import get_crm_provider

    svc = get_customer_context_service()
    crm = get_crm_provider()
    matched = 0

    for e in emails:
        if e.get("replyable") is False:
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

            # Log inbound email as CRM interaction
            subject = e.get("subject", "(no subject)")
            body_preview = (e.get("body_text") or "")[:200]
            await crm.log_interaction(
                contact_id=ctx.contact_id,
                interaction_type="email",
                summary=f"Received email: {subject}. {body_preview}".strip(),
            )

        except Exception as exc:
            logger.warning(
                "CRM cross-ref failed for email %s: %s", e.get("id"), exc
            )

    return matched


# ---------------------------------------------------------------------------
# Stage 2: LLM intent classification + action plan (merged)
# ---------------------------------------------------------------------------

def _parse_intent_plan(text: str) -> dict | None:
    """Parse LLM output into {intent, sentiment, confidence, actions[]}.

    Returns None if parsing fails or intent is invalid.
    """
    # Strip <think> tags (Qwen3)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Find first JSON object
    obj_start = text.find("{")
    if obj_start < 0:
        return None

    # Find matching closing brace
    depth = 0
    in_string = False
    escape = False
    obj_end = -1
    for i in range(obj_start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj_end = i
                break

    if obj_end < 0:
        return None

    try:
        data = json.loads(text[obj_start : obj_end + 1])
    except json.JSONDecodeError:
        logger.warning("Failed to parse intent plan JSON")
        return None

    if not isinstance(data, dict):
        return None

    intent = data.get("intent", "").lower().strip()
    if intent not in VALID_INTENTS:
        logger.warning("Invalid intent '%s', discarding", intent)
        return None

    # Validate and normalize actions array
    raw_actions = data.get("actions", [])
    actions = []
    if isinstance(raw_actions, list):
        for a in raw_actions:
            if isinstance(a, dict) and a.get("action"):
                actions.append({
                    "action": a["action"],
                    "priority": a.get("priority", 99),
                    "params": a.get("params", {}),
                    "rationale": a.get("rationale", ""),
                })
        actions.sort(key=lambda x: x["priority"])

    # Coerce confidence to float (LLM may return string)
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5

    return {
        "intent": intent,
        "sentiment": data.get("sentiment", "neutral"),
        "confidence": confidence,
        "actions": actions,
    }


async def _classify_and_plan(emails: list[dict[str, Any]]) -> int:
    """Stage 2: LLM intent classification + action planning.

    Runs on ALL emails where replyable != False.  Uses the merged
    email_intent_planning skill to get both intent and actions in one
    LLM call.  Returns count of emails classified.
    """
    from ...comms.action_planner import _format_customer_context
    from ...skills import get_skill_registry
    from ...services.llm_router import get_triage_llm
    from ...services import llm_registry
    from ...services.protocols import Message

    cfg = settings.email_intake
    if not cfg.action_plan_enabled:
        return 0

    llm = get_triage_llm() or llm_registry.get_active()
    if not llm:
        logger.warning("No LLM available for email intent classification")
        return 0

    skill = get_skill_registry().get("digest/email_intent_planning")
    classified = 0

    for e in emails:
        if classified >= cfg.max_action_plans_per_cycle:
            break

        if e.get("replyable") is False:
            continue

        try:
            # Build customer context string (or fallback for unknown senders)
            ctx = e.get("_customer_context")
            if ctx is not None:
                customer_context_str = _format_customer_context(ctx)
            else:
                customer_context_str = "No prior customer history available."

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
                    f"You are Atlas, classifying and planning follow-up actions for an email.\n"
                    f"Customer: {customer_context_str}\n"
                    f"From: {e.get('from', '')}\n"
                    f"Subject: {e.get('subject', '')}\n"
                    f"Category: {e.get('category', 'other')}\n"
                    f"Body: {body}\n"
                    f'Return a JSON object: {{"intent": "estimate_request|reschedule|complaint|info_admin", '
                    f'"sentiment": "...", "confidence": 0.0-1.0, '
                    f'"actions": [{{action, priority, params, rationale}}]}}'
                )

            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content="Classify the intent and generate the action plan for this email."),
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

            parsed = _parse_intent_plan(text)
            if parsed is None:
                continue

            e["_intent"] = parsed["intent"]
            e["_sentiment"] = parsed["sentiment"]

            # Store action plan if there are real actions
            if parsed["actions"] and parsed["actions"][0].get("action") != "none":
                e["_action_plan"] = parsed["actions"]

            classified += 1
            logger.info(
                "Intent classified for email %s: %s (confidence=%.2f, actions=%d)",
                e.get("id"), parsed["intent"], parsed["confidence"],
                len(parsed["actions"]),
            )

        except Exception as exc:
            logger.warning(
                "Intent classification failed for email %s: %s",
                e.get("id"), exc,
            )

    return classified


# ---------------------------------------------------------------------------
# Enriched notifications (intent-aware)
# ---------------------------------------------------------------------------

async def _send_enriched_notifications(emails: list[dict[str, Any]]) -> None:
    """Send ntfy notifications with intent-specific buttons and CRM context.

    Uses _INTENT_NTFY mapping for intent-classified emails.  Non-classified
    emails fall through to the standard notification path.
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

        intent = e.get("_intent")
        customer_summary = e.get("_customer_summary")
        action_plan = e.get("_action_plan")

        # No intent and no CRM enrichment -- use standard notification path
        if not intent and not customer_summary and not action_plan:
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

        is_new_lead = e.get("_is_new_lead", False)

        # Build enriched message body
        parts = [f"From: {sender_name}", f"Subject: {subject}"]
        if customer_summary:
            parts.append(f"\n{customer_summary}")
        if action_plan:
            actions_str = ", ".join(a.get("action", "?") for a in action_plan[:3])
            parts.append(f"Suggested: {actions_str}")
        parts.append(f"\n{body_snippet}")
        message = "\n".join(parts)

        # Build View Email URL
        rfc_msg_id = e.get("message_id", "").strip("<>")
        view_url = (
            f"https://mail.google.com/mail/u/0/#search/rfc822msgid:{rfc_msg_id}"
            if rfc_msg_id
            else "https://mail.google.com/mail/u/0/#inbox"
        )

        # Intent-aware buttons and metadata
        intent_cfg = _INTENT_NTFY.get(intent) if intent else None

        if intent_cfg:
            # Build action buttons from intent config
            action_parts = []
            for btn_type, label, url_template in intent_cfg["buttons"]:
                btn_url = api_url + url_template.replace("{id}", msg_id)
                action_parts.append(
                    f"{btn_type}, {label}, {btn_url}, method=POST, clear=true"
                )
            action_parts.append(f"view, View Email, {view_url}")
            actions = "; ".join(action_parts)

            intent_label = _INTENT_LABELS.get(intent, intent.replace("_", " ").title())
            if is_new_lead or (is_lead and e.get("_contact_id") is None):
                ntfy_title = f"NEW LEAD: {intent_label} - {subject[:50]}"
            else:
                ntfy_title = f"{intent_label}: {subject[:50]}"

            ntfy_tags = intent_cfg["tags"]
            ntfy_priority = intent_cfg["priority"]
        else:
            # Fallback: generic buttons (pre-intent behavior)
            actions = (
                f"http, Draft Reply, {api_url}/api/v1/email/drafts/generate/{msg_id}, method=POST, clear=true; "
                f"view, View Email, {view_url}"
            )
            if is_new_lead or is_lead:
                ntfy_title = f"New Lead: {subject[:60]}"
                ntfy_tags = "email,star"
            else:
                ntfy_title = f"Action Required: {subject[:60]}"
                ntfy_tags = "email,warning"
            ntfy_priority = "high"

        headers = {
            "Title": ntfy_title,
            "Priority": ntfy_priority,
            "Tags": ntfy_tags,
            "Actions": actions,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(ntfy_url, content=message, headers=headers)
                resp.raise_for_status()
            logger.info(
                "Enriched notification sent for %s: intent=%s title=%s",
                msg_id, intent or "none", ntfy_title[:40],
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
    """Record processed emails with action_plan, customer_context_summary, and intent."""
    pool = get_db_pool()
    if not pool.is_initialized or not emails:
        return

    try:
        async with pool.transaction() as conn:
            await conn.executemany(
                """
                INSERT INTO processed_emails
                    (gmail_message_id, sender, subject, category, priority,
                     replyable, contact_id, action_plan, customer_context_summary,
                     intent)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
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
                        e.get("_intent"),
                    )
                    for e in emails
                ],
            )
        logger.debug("Recorded %d processed email IDs (with intent + action plans)", len(emails))
    except Exception as e:
        logger.warning("Failed to record processed emails: %s", e)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run(task: ScheduledTask) -> dict:
    """Near-real-time email intake: fetch via IMAP, classify, CRM xref, intent classify, notify."""
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

    # 4. Stage 1: Classify (rule-based, no LLM)
    classifier = get_email_classifier()
    emails = classifier.classify_batch(emails)

    # 5. Process leads (CRM contact creation for web form submissions)
    await _process_lead_emails(emails)

    # 6. CRM cross-reference (expanded: all replyable emails)
    crm_count = 0
    if cfg.crm_enabled:
        crm_count = await _crm_cross_reference(emails)

    # 7. Stage 2: LLM intent classification + action planning
    intent_count = 0
    if cfg.action_plan_enabled:
        intent_count = await _classify_and_plan(emails)

    # 8. Derive new_lead flag (intent classified + no existing CRM contact)
    for e in emails:
        if e.get("_intent") and e.get("_contact_id") is None:
            e["_is_new_lead"] = True

    # 9. Record to DB (extended INSERT with intent + action_plan + customer_context_summary)
    await _record_with_action_plans(emails)

    # 10. Send intent-aware ntfy notifications
    action_emails = [
        e for e in emails
        if (e.get("priority") == "action_required" or e.get("category") == "lead")
        and e.get("replyable") is not False
    ]
    if action_emails:
        await _send_enriched_notifications(action_emails)

    logger.info(
        "Email intake: %d new, %d CRM matched, %d intent classified",
        len(emails), crm_count, intent_count,
    )

    return {
        "new_emails": len(emails),
        "crm_matched": crm_count,
        "intent_classified": intent_count,
        "_skip_synthesis": True,
    }
