"""
SMS intelligence pipeline.

After an inbound SMS is persisted:
1. LLM classify intent + extract structured data
2. Look up or create CRM contact, link SMS record
3. Generate action plan (reuses call action_planner)
4. Push ntfy notification with intent-aware action buttons

Modeled after call_intelligence.py -- each step is fail-open.
"""

import asyncio
import json
import logging
import re
from typing import Optional
from uuid import UUID

import httpx

from ..config import settings
from ..skills import get_skill_registry
from ..storage.repositories.sms_message import get_sms_message_repo

logger = logging.getLogger("atlas.comms.sms_intelligence")


# ---------------------------------------------------------------------------
# SMS intent -> business intent normalization
# ---------------------------------------------------------------------------

_SMS_INTENT_MAP = {
    "estimate_request": "estimate_request",
    "booking": "estimate_request",
    "reschedule": "reschedule",
    "cancel": "reschedule",
    "complaint": "complaint",
    "inquiry": "info_admin",
    "follow_up": "info_admin",
    "stop": None,
    "spam": None,
}

_INTENT_LABELS = {
    "estimate_request": "Estimate Request",
    "reschedule": "Reschedule",
    "complaint": "Complaint",
    "info_admin": "Info/Admin",
}

_SMS_INTENT_NTFY = {
    "estimate_request": {
        "buttons": [
            ("http", "Book Appointment", "/api/v1/comms/sms-actions/{id}/book"),
            ("http", "Reply SMS", "/api/v1/comms/sms-actions/{id}/reply"),
        ],
        "tags": "speech_balloon,dollar",
        "priority": "high",
    },
    "reschedule": {
        "buttons": [
            ("http", "Book Appointment", "/api/v1/comms/sms-actions/{id}/book"),
            ("http", "Reply SMS", "/api/v1/comms/sms-actions/{id}/reply"),
        ],
        "tags": "speech_balloon,calendar",
        "priority": "high",
    },
    "complaint": {
        "buttons": [
            ("http", "Reply SMS", "/api/v1/comms/sms-actions/{id}/reply"),
        ],
        "tags": "speech_balloon,rotating_light",
        "priority": "urgent",
    },
    "info_admin": {
        "buttons": [
            ("http", "Reply SMS", "/api/v1/comms/sms-actions/{id}/reply"),
        ],
        "tags": "speech_balloon,information_source",
        "priority": "default",
    },
}


async def process_inbound_sms(
    sms_id: Optional[UUID],
    from_number: str,
    to_number: str,
    body: str,
    business_context_id: str,
    business_context=None,
    media_urls: Optional[list] = None,
) -> None:
    """
    Process an inbound SMS through the intelligence pipeline.

    Each step is fail-open -- partial results are better than no results.
    """
    cfg = settings.sms_intelligence

    if not cfg.enabled:
        logger.info("SMS intelligence disabled, skipping")
        return

    if not body.strip():
        logger.info("Empty SMS body, skipping intelligence pipeline")
        return

    repo = get_sms_message_repo()

    # Step 1: LLM classify + extract
    summary = None
    extracted_data = {}
    raw_intent = None
    try:
        if sms_id:
            await repo.update_status(sms_id, "processing")

        summary, extracted_data, raw_intent = await _extract_sms_data(body, business_context)

        if sms_id:
            await repo.update_extraction(
                sms_id,
                summary=summary,
                extracted_data=extracted_data,
                intent=raw_intent,
            )
        logger.info(
            "Step 1/4 OK: SMS extraction intent=%s summary=%s",
            raw_intent, (summary or "")[:80],
        )
    except Exception as e:
        logger.error("Step 1/4 FAIL: SMS extraction: %s", e)

    # Skip further processing for opt-out and spam
    business_intent = _SMS_INTENT_MAP.get(raw_intent or "")
    if raw_intent in ("stop", "spam"):
        logger.info("SMS intent=%s, skipping CRM/plan/notify", raw_intent)
        if sms_id:
            try:
                await repo.update_status(sms_id, "ready")
            except Exception:
                pass
        return

    # Step 2: CRM link
    contact_id = None
    is_new_lead = False
    try:
        contact_id, is_new_lead = await _link_to_crm(
            repo, sms_id, from_number, business_context_id,
            extracted_data, summary or f"Inbound SMS: {body[:100]}",
        )
        if contact_id:
            logger.info("Step 2/4 OK: Linked SMS to contact %s (new=%s)", contact_id, is_new_lead)
        else:
            logger.info("Step 2/4 OK: No CRM link created (insufficient data)")
    except Exception as e:
        logger.error("Step 2/4 FAIL: CRM link: %s", e)

    # Step 3: Action plan
    action_plan = []
    try:
        action_plan = await _generate_action_plan(
            sms_id, contact_id, summary or body[:200],
            extracted_data, business_context,
        )
        if action_plan and sms_id:
            # Store actions in extracted_data
            merged = dict(extracted_data)
            merged["proposed_actions"] = action_plan
            await repo.update_extraction(sms_id, extracted_data=merged)
        logger.info("Step 3/4 OK: Action plan %d actions", len(action_plan))
    except Exception as e:
        logger.error("Step 3/4 FAIL: Action plan: %s", e)

    # Step 4: ntfy notification
    notified = False
    try:
        await _notify_sms_summary(
            repo, sms_id, from_number, body,
            summary, extracted_data, action_plan,
            business_context,
            is_new_lead=is_new_lead,
        )
        notified = True
        logger.info("Step 4/4 OK: Notification sent")
    except Exception as e:
        logger.error("Step 4/4 FAIL: Notification: %s", e)

    # Mark ready only if notification didn't already set "notified"
    if sms_id and not notified:
        try:
            await repo.update_status(sms_id, "ready")
        except Exception:
            pass


async def _extract_sms_data(
    body: str,
    business_context=None,
) -> tuple[Optional[str], dict, Optional[str]]:
    """Use LLM to classify intent and extract structured data from SMS."""
    from ..services.protocols import Message
    from ..services.llm_router import get_triage_llm
    from ..services import llm_registry

    llm = get_triage_llm() or llm_registry.get_active()
    if not llm:
        logger.warning("No active LLM for SMS extraction")
        return None, {}, None

    # Build business context string
    ctx_parts = []
    if business_context:
        if hasattr(business_context, "name"):
            ctx_parts.append(f"Business: {business_context.name}")
        if hasattr(business_context, "business_type") and business_context.business_type:
            ctx_parts.append(f"Type: {business_context.business_type}")
        if hasattr(business_context, "services") and business_context.services:
            ctx_parts.append(f"Services: {', '.join(business_context.services)}")
    business_context_str = "\n".join(ctx_parts) if ctx_parts else "General business"

    # Load skill prompt
    skill = get_skill_registry().get("sms/sms_extraction")
    if skill:
        system_prompt = skill.content.replace("{business_context}", business_context_str)
    else:
        system_prompt = (
            "Extract customer info and intent from this SMS message. "
            f"Business context: {business_context_str}\n"
            "Return a JSON object with: customer_name, customer_phone, customer_email, "
            "intent, services_mentioned, address, preferred_date, preferred_time, "
            "urgency, follow_up_needed, notes."
        )

    cfg = settings.sms_intelligence
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=body),
    ]

    loop = asyncio.get_running_loop()
    result = await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: llm.chat(
                messages=messages,
                max_tokens=cfg.llm_max_tokens,
                temperature=cfg.llm_temperature,
            ),
        ),
        timeout=cfg.llm_timeout,
    )

    text = result.get("response", "").strip()
    if not text:
        return None, {}, None

    # Strip <think> tags (Qwen3 models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    return _parse_extraction(text, body)


def _parse_extraction(text: str, body: str) -> tuple[Optional[str], dict, Optional[str]]:
    """Parse LLM output into (summary, extracted_data, intent)."""
    extracted_data = {}

    # Find JSON object
    obj_start = text.find("{")
    obj_end = _find_matching_brace(text, obj_start) if obj_start >= 0 else -1

    if obj_start >= 0 and obj_end > obj_start:
        try:
            extracted_data = json.loads(text[obj_start : obj_end + 1])
        except json.JSONDecodeError:
            pass

    intent = extracted_data.get("intent")

    # Build summary
    parts = []
    name = extracted_data.get("customer_name")
    if name:
        parts.append(f"Customer: {name}")
    if intent:
        parts.append(f"Intent: {intent.replace('_', ' ')}")
    services = extracted_data.get("services_mentioned")
    if services:
        parts.append(f"Services: {', '.join(services)}")
    notes = extracted_data.get("notes")
    if notes:
        parts.append(notes)

    summary = ". ".join(parts) if parts else body[:200]

    return summary, extracted_data, intent


def _find_matching_brace(text: str, start: int) -> int:
    """Find the closing brace matching the opening brace at `start`."""
    if start < 0 or start >= len(text) or text[start] != "{":
        return -1
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
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
                return i
    return -1


async def _link_to_crm(
    repo,
    sms_id: Optional[UUID],
    from_number: str,
    context_id: str,
    extracted_data: dict,
    summary: str,
) -> tuple[Optional[str], bool]:
    """Look up or create a CRM contact from SMS extraction data.

    Returns (contact_id, is_new_lead). Fail-open.
    """
    from ..services.crm_provider import get_crm_provider
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return None, False

    phone = extracted_data.get("customer_phone") or from_number
    email_addr = extracted_data.get("customer_email")
    name = extracted_data.get("customer_name")

    if not phone and not email_addr:
        return None, False

    crm = get_crm_provider()
    contact = await crm.find_or_create_contact(
        full_name=name or phone or "Unknown",
        phone=phone,
        email=email_addr,
        address=extracted_data.get("address"),
        business_context_id=context_id,
        contact_type="customer",
        source="sms",
        source_ref=str(sms_id) if sms_id else None,
    )
    if not contact.get("id"):
        return None, False

    contact_id = str(contact["id"])
    is_new_lead = contact.get("_was_created", False)

    # Link the SMS to the contact
    if sms_id:
        await repo.link_contact(sms_id, contact_id)

    # Log the interaction with normalized business intent
    raw_intent = extracted_data.get("intent", "")
    business_intent = _SMS_INTENT_MAP.get(raw_intent)
    await crm.log_interaction(
        contact_id=contact_id,
        interaction_type="sms",
        summary=summary or f"Inbound SMS from {from_number}",
        intent=business_intent,
    )

    return contact_id, is_new_lead


async def _generate_action_plan(
    sms_id: Optional[UUID],
    contact_id: Optional[str],
    summary: str,
    extracted_data: dict,
    business_context=None,
) -> list[dict]:
    """Reuse the call action planner for SMS."""
    from .action_planner import generate_action_plan
    from ..services.customer_context import CustomerContext, get_customer_context_service

    intent = extracted_data.get("intent", "")
    if intent in ("stop", "spam", "other"):
        return []

    if contact_id:
        ctx = await get_customer_context_service().get_context(contact_id)
    else:
        ctx = CustomerContext()

    return await generate_action_plan(
        transcript_id=sms_id or UUID(int=0),
        call_summary=summary,
        extracted_data=extracted_data,
        customer_context=ctx,
        business_context=business_context,
    )


async def _notify_sms_summary(
    repo,
    sms_id: Optional[UUID],
    from_number: str,
    body: str,
    summary: Optional[str],
    extracted_data: dict,
    proposed_actions: list,
    business_context=None,
    is_new_lead: bool = False,
) -> None:
    """Send ntfy notification with SMS summary and intent-aware buttons."""
    cfg = settings.sms_intelligence
    if not cfg.notify_enabled:
        return
    if not settings.alerts.ntfy_enabled:
        return
    if not settings.alerts.ntfy_url or not settings.alerts.ntfy_topic:
        return

    from ..comms import comms_settings as _comms_cfg

    ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"
    api_url = _comms_cfg.webhook_base_url.rstrip("/")

    # Build message
    biz_name = business_context.name if business_context and hasattr(business_context, "name") else "Business"
    lines = [f"From: {from_number}"]

    name = extracted_data.get("customer_name")
    if name:
        lines.append(f"Customer: {name}")

    raw_intent = extracted_data.get("intent")
    business_intent = _SMS_INTENT_MAP.get(raw_intent or "") if raw_intent else None
    if raw_intent:
        lines.append(f"Intent: {raw_intent.replace('_', ' ').title()}")

    services = extracted_data.get("services_mentioned")
    if services:
        lines.append(f"Services: {', '.join(services)}")

    lines.append(f"\nMessage: {body[:300]}")

    # Format action plan if present
    if proposed_actions:
        plan_lines = []
        for a in proposed_actions:
            atype = a.get("action") or a.get("type", "none")
            if atype == "none":
                continue
            rationale = a.get("rationale") or a.get("label", "")
            plan_lines.append(f"  {atype.replace('_', ' ').title()}: {rationale}")
        if plan_lines:
            lines.append("\nAction Plan:")
            lines.extend(plan_lines)

    message = "\n".join(lines)

    # Build ntfy action buttons
    sid = str(sms_id) if sms_id else "unknown"
    intent_cfg = _SMS_INTENT_NTFY.get(business_intent) if business_intent else None

    if intent_cfg:
        action_parts = []
        for btn_type, label, url_template in intent_cfg["buttons"]:
            btn_url = api_url + url_template.replace("{id}", sid)
            action_parts.append(
                f"{btn_type}, {label}, {btn_url}, method=POST, clear=true"
            )
        actions_header = "; ".join(action_parts)

        intent_label = _INTENT_LABELS.get(business_intent, business_intent)
        if is_new_lead:
            title = f"NEW LEAD SMS: {intent_label} - {biz_name}"
        else:
            title = f"SMS {intent_label}: {biz_name}"

        headers = {
            "Title": title,
            "Priority": intent_cfg["priority"],
            "Tags": intent_cfg["tags"],
            "Actions": actions_header,
        }
    else:
        if is_new_lead:
            title = f"NEW LEAD SMS: {biz_name}"
        else:
            title = f"SMS: {biz_name}"

        headers = {
            "Title": title,
            "Priority": "high",
            "Tags": "speech_balloon",
        }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(ntfy_url, content=message, headers=headers)
            resp.raise_for_status()

        if sms_id:
            await repo.mark_notified(sms_id)
            await repo.update_status(sms_id, "notified")
        logger.info("SMS notification sent for %s", from_number)
    except Exception as e:
        logger.warning("Failed to send SMS notification: %s", e)
