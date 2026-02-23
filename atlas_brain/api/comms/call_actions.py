"""
Call action endpoints triggered by ntfy notification buttons.

POST /comms/call-actions/{transcript_id}/book          -> create Google Calendar event
POST /comms/call-actions/{transcript_id}/sms           -> send confirmation SMS to customer
GET  /comms/call-actions/{transcript_id}/view          -> return transcript + extracted data
POST /comms/call-actions/{transcript_id}/draft-email   -> LLM drafts confirmation email
POST /comms/call-actions/{transcript_id}/draft-sms     -> LLM drafts confirmation SMS
POST /comms/call-actions/{transcript_id}/send-email    -> send the drafted email via Resend
POST /comms/call-actions/{transcript_id}/send-sms      -> send the drafted SMS via SignalWire
POST /comms/call-actions/{transcript_id}/approve-plan  -> execute all actions in the plan
POST /comms/call-actions/{transcript_id}/reject-plan   -> reject the action plan
POST /comms/call-actions/{transcript_id}/discard       -> discard draft (ntfy clear handler)
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID
from zoneinfo import ZoneInfo

import dateparser
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from ...comms.context import get_context_router
from ...config import settings
from ...services.llm_router import get_draft_llm, get_triage_llm
from ...services.protocols import Message
from ...skills import get_skill_registry
from ...storage.repositories.call_transcript import get_call_transcript_repo

logger = logging.getLogger("atlas.api.comms.call_actions")

router = APIRouter(prefix="/call-actions", tags=["call-actions"])


def _ntfy_url() -> str:
    return f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"


def _api_url() -> str:
    from ...comms import comms_settings
    return comms_settings.webhook_base_url.rstrip("/")


def _get_business_name(record: dict) -> str:
    """Look up the business name from the transcript's context ID."""
    ctx_id = record.get("business_context_id") or ""
    if ctx_id:
        ctx = get_context_router().get_context(ctx_id)
        if ctx and ctx.name:
            return ctx.name
    return "Your Business"


def _parse_event_datetime(date_str: str, time_str: str, tz_name: str = "America/Chicago") -> datetime:
    """Parse extracted date/time strings into a timezone-aware datetime.

    Uses dateparser for natural language support (relative dates, time-of-day words).
    Falls back to tomorrow at 9 AM in the given timezone if strings cannot be parsed.
    """
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    fallback = now.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)

    dp_settings = {
        "TIMEZONE": tz_name,
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "future",
        "PREFER_DAY_OF_MONTH": "first",
    }

    combined = f"{date_str} {time_str}".strip()
    if combined:
        result = dateparser.parse(combined, settings=dp_settings)
        if result:
            return result

    if date_str:
        result = dateparser.parse(date_str, settings=dp_settings)
        if result:
            if time_str:
                t = dateparser.parse(time_str, settings=dp_settings)
                if t:
                    return result.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
            return result

    return fallback


def _build_customer_info(data: dict, from_number: str) -> str:
    parts = []
    if data.get("customer_name"):
        parts.append(f"Name: {data['customer_name']}")
    phone = data.get("customer_phone") or from_number
    if phone:
        parts.append(f"Phone: {phone}")
    if data.get("customer_email"):
        parts.append(f"Email: {data['customer_email']}")
    if data.get("address"):
        parts.append(f"Address: {data['address']}")
    services = ", ".join(data.get("services_mentioned") or [])
    if services:
        parts.append(f"Services: {services}")
    date_str = data.get("preferred_date", "")
    time_str = data.get("preferred_time", "")
    if date_str or time_str:
        parts.append(f"Requested: {(date_str + ' ' + time_str).strip()}")
    frequency = data.get("frequency")
    if frequency:
        parts.append(f"Frequency: {frequency.replace('_', ' ').title()}")
    return "\n".join(parts) if parts else "No customer details available"


async def _generate_draft(draft_type: str, record: dict, business_name: str) -> str:
    """Use the draft LLM to generate a confirmation email or SMS draft."""
    data = record.get("extracted_data") or {}
    customer_info = _build_customer_info(data, record.get("from_number", ""))

    skill = get_skill_registry().get(f"call/confirmation_{draft_type}")
    if skill:
        system_prompt = (
            skill.content
            .replace("{business_name}", business_name)
            .replace("{customer_info}", customer_info)
        )
    elif draft_type == "email":
        system_prompt = (
            f"Draft a short confirmation email for {business_name}.\n"
            f"Customer info:\n{customer_info}\n"
            "Format: SUBJECT: [subject]\n\n[body]. Under 180 words."
        )
    else:
        system_prompt = (
            f"Draft a brief SMS confirmation for {business_name}.\n"
            f"Customer info:\n{customer_info}\n"
            "Under 300 characters. End with 'Reply STOP to opt out.'"
        )

    llm = get_draft_llm() or get_triage_llm()
    if not llm:
        logger.warning("No LLM available for %s draft generation", draft_type)
        return ""

    label = "email" if draft_type == "email" else "SMS confirmation"
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=f"Please draft the {label}."),
    ]

    loop = asyncio.get_running_loop()
    result = await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: llm.chat(messages=messages, max_tokens=512, temperature=0.4),
        ),
        timeout=30.0,
    )

    text = result.get("response", "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


async def _notify_booking_confirmed(
    transcript_id: UUID,
    record: dict,
    business_name: str,
) -> None:
    """Fire 'Appointment Booked' ntfy with Draft Email + Draft SMS buttons."""
    if not settings.alerts.ntfy_enabled:
        return

    data = record.get("extracted_data") or {}
    customer = data.get("customer_name") or "Customer"
    services = ", ".join(data.get("services_mentioned") or []) or "cleaning service"
    date_str = data.get("preferred_date", "")
    time_str = data.get("preferred_time", "")

    lines = [f"Customer: {customer}", f"Services: {services}"]
    if date_str or time_str:
        lines.append(f"Requested: {(date_str + ' ' + time_str).strip()}")
    message = "\n".join(lines)

    base = _api_url()
    tid = transcript_id
    actions = (
        f"http, Draft Email, {base}/api/v1/comms/call-actions/{tid}/draft-email, method=POST, clear=true; "
        f"http, Draft SMS, {base}/api/v1/comms/call-actions/{tid}/draft-sms, method=POST, clear=true; "
        f"view, View Transcript, {base}/api/v1/comms/call-actions/{tid}/view"
    )

    headers = {
        "Title": f"{business_name}: Appointment Booked",
        "Priority": "high",
        "Tags": "calendar,white_check_mark",
        "Actions": actions,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(_ntfy_url(), content=message, headers=headers)
        resp.raise_for_status()
    logger.info("Booking confirmed notification sent for %s", transcript_id)


async def _notify_draft_ready(
    transcript_id: UUID,
    draft_type: str,
    content: str,
    customer_name: str,
    business_name: str,
) -> None:
    """Fire a draft-ready ntfy with Send + Discard buttons."""
    if not settings.alerts.ntfy_enabled:
        return

    label = "Email" if draft_type == "email" else "SMS"
    preview = content[:400] + ("..." if len(content) > 400 else "")

    base = _api_url()
    tid = transcript_id
    actions = (
        f"http, Send, {base}/api/v1/comms/call-actions/{tid}/send-{draft_type}, method=POST, clear=true; "
        f"http, Discard, {base}/api/v1/comms/call-actions/{tid}/discard, method=POST, clear=true"
    )

    headers = {
        "Title": f"{business_name}: {label} Draft - {customer_name}",
        "Priority": "default",
        "Tags": "email" if draft_type == "email" else "phone",
        "Actions": actions,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(_ntfy_url(), content=preview, headers=headers)
        resp.raise_for_status()
    logger.info("%s draft notification sent for %s", label, transcript_id)


async def _get_transcript_or_404(transcript_id: UUID) -> dict:
    repo = get_call_transcript_repo()
    record = await repo.get_by_id(transcript_id)
    if not record:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return record


@router.post("/{transcript_id}/book")
async def book_appointment(transcript_id: UUID):
    """Create a Google Calendar event from the call's extracted data."""
    record = await _get_transcript_or_404(transcript_id)
    data = record.get("extracted_data") or {}

    customer = data.get("customer_name") or "Customer"
    phone = data.get("customer_phone") or record.get("from_number", "")
    email = data.get("customer_email") or ""
    address = data.get("address", "")
    services = ", ".join(data.get("services_mentioned") or []) or "Cleaning service"
    date_str = data.get("preferred_date") or ""
    time_str = data.get("preferred_time") or ""
    frequency = data.get("frequency") or ""

    try:
        from ...tools.calendar import calendar_tool

        biz_name = _get_business_name(record)
        ctx_id = record.get("business_context_id") or ""
        ctx = get_context_router().get_context(ctx_id) if ctx_id else None
        calendar_id = (ctx.scheduling.calendar_id if ctx else None) or None

        summary = f"Estimate: {customer}"
        desc_lines = [
            f"Customer: {customer}",
            f"Phone: {phone}",
        ]
        if email:
            desc_lines.append(f"Email: {email}")
        if address:
            desc_lines.append(f"Address: {address}")
        desc_lines.append(f"Services: {services}")
        if frequency:
            desc_lines.append(f"Frequency: {frequency.replace('_', ' ').title()}")
        if date_str or time_str:
            desc_lines.append(f"Requested: {(date_str + ' ' + time_str).strip()}")
        description = "\n".join(desc_lines)

        tz_name = ctx.hours.timezone if (ctx and ctx.hours) else settings.reminder.default_timezone
        start_dt = _parse_event_datetime(date_str, time_str, tz_name)
        end_dt = start_dt + timedelta(hours=1)

        result = await calendar_tool.create_event(
            summary=summary,
            start=start_dt,
            end=end_dt,
            location=address or None,
            description=description,
            calendar_id=calendar_id,
        )
        logger.info("Calendar event created for transcript %s: %s", transcript_id, result)

        try:
            await _notify_booking_confirmed(transcript_id, record, biz_name)
        except Exception as _notify_err:
            logger.warning("Booking confirmed ntfy failed for %s: %s", transcript_id, _notify_err)

        return JSONResponse({"status": "ok", "event": result.data if result.success else result.message})

    except Exception as e:
        logger.error("Failed to book appointment for %s: %s", transcript_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{transcript_id}/sms")
async def send_sms(transcript_id: UUID):
    """Send a confirmation SMS to the customer's phone number."""
    record = await _get_transcript_or_404(transcript_id)
    data = record.get("extracted_data") or {}

    to_number = data.get("customer_phone") or record.get("from_number")
    if not to_number:
        raise HTTPException(status_code=400, detail="No customer phone number available")

    customer = data.get("customer_name") or "there"
    date_str = data.get("preferred_date", "")
    time_str = data.get("preferred_time", "")
    appt = f" for {(date_str + ' ' + time_str).strip()}" if (date_str or time_str) else ""

    biz_name = _get_business_name(record)
    body = (
        f"Hi {customer}, this is {biz_name} following up on your call{appt}. "
        f"We'll be in touch shortly to confirm details. Reply STOP to opt out."
    )

    try:
        from ...comms import get_comms_service
        svc = get_comms_service()
        from_number = record.get("to_number", "")
        msg = await svc.provider.send_sms(
            to_number=to_number,
            from_number=from_number,
            body=body,
        )
        logger.info("SMS sent to %s for transcript %s", to_number, transcript_id)

        # Persist outbound SMS (fail-open)
        try:
            from ...storage.repositories.sms_message import get_sms_message_repo
            sms_repo = get_sms_message_repo()
            await sms_repo.create(
                message_sid=getattr(msg, "provider_message_id", "") or f"call_action_{transcript_id.hex[:12]}",
                from_number=from_number,
                to_number=to_number,
                direction="outbound",
                body=body,
                business_context_id=record.get("business_context_id"),
                status="sent",
                source="call_action",
                source_ref=str(transcript_id),
            )
        except Exception as persist_err:
            logger.warning("Failed to persist outbound SMS: %s", persist_err)

        return JSONResponse({"status": "ok", "message_sid": msg.provider_message_id})

    except Exception as e:
        logger.error("Failed to send SMS for %s: %s", transcript_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{transcript_id}/view")
async def view_transcript(transcript_id: UUID):
    """Return the transcript, extracted data, action plan, and customer context."""
    record = await _get_transcript_or_404(transcript_id)
    data = record.get("extracted_data") or {}

    mins, secs = divmod(record.get("duration_seconds", 0), 60)
    dur_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    lines = [
        f"Call: {record.get('call_sid', '')}",
        f"From: {record.get('from_number', '')}",
        f"Duration: {dur_str}",
        f"Status: {record.get('status', '')}",
        "",
        "--- EXTRACTED DATA ---",
    ]
    for k, v in data.items():
        if v not in (None, "", [], False):
            lines.append(f"{k}: {v}")

    # Action plan
    actions = record.get("proposed_actions") or []
    actionable = [a for a in actions if (a.get("action") or a.get("type", "none")) != "none"]
    if actionable:
        lines += ["", "--- ACTION PLAN ---"]
        for i, a in enumerate(actionable, 1):
            atype = a.get("action") or a.get("type", "")
            rationale = a.get("rationale") or a.get("label", "")
            lines.append(f"{i}. {atype.replace('_', ' ').title()}: {rationale}")

    # Customer context (if linked)
    contact_id = record.get("contact_id")
    if contact_id:
        try:
            from ...services.customer_context import get_customer_context_service
            ctx = await get_customer_context_service().get_context(str(contact_id))
            if not ctx.is_empty:
                lines += ["", "--- CUSTOMER ---"]
                c = ctx.contact
                lines.append(f"Name: {c.get('full_name', 'Unknown')}")
                if c.get("phone"):
                    lines.append(f"Phone: {c['phone']}")
                if c.get("email"):
                    lines.append(f"Email: {c['email']}")
                if c.get("contact_type"):
                    lines.append(f"Type: {c['contact_type']}")
                if ctx.appointments:
                    lines.append(f"Appointments: {len(ctx.appointments)}")
                if ctx.call_transcripts:
                    lines.append(f"Past calls: {len(ctx.call_transcripts)}")
                if ctx.interactions:
                    lines.append(f"Interactions: {len(ctx.interactions)}")
        except Exception as e:
            logger.warning("Customer context fetch failed for %s: %s", transcript_id, e)

    lines += ["", "--- TRANSCRIPT ---", record.get("transcript") or "(none)"]
    return PlainTextResponse("\n".join(lines))


@router.post("/{transcript_id}/draft-email")
async def draft_email(transcript_id: UUID):
    """Generate a confirmation email draft using LLM and notify via ntfy."""
    record = await _get_transcript_or_404(transcript_id)
    data = record.get("extracted_data") or {}
    customer = data.get("customer_name") or "Customer"

    biz_name = _get_business_name(record)

    try:
        content = await _generate_draft("email", record, biz_name)
        if not content:
            raise HTTPException(status_code=500, detail="LLM did not return a draft")

        repo = get_call_transcript_repo()
        await repo.save_draft(transcript_id, "email", content)
        await _notify_draft_ready(transcript_id, "email", content, customer, biz_name)

        return JSONResponse({"status": "ok", "draft": content})
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to draft email for %s: %s", transcript_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{transcript_id}/draft-sms")
async def draft_sms_confirmation(transcript_id: UUID):
    """Generate a confirmation SMS draft using LLM and notify via ntfy."""
    record = await _get_transcript_or_404(transcript_id)
    data = record.get("extracted_data") or {}
    customer = data.get("customer_name") or "Customer"

    biz_name = _get_business_name(record)

    try:
        content = await _generate_draft("sms", record, biz_name)
        if not content:
            raise HTTPException(status_code=500, detail="LLM did not return a draft")

        repo = get_call_transcript_repo()
        await repo.save_draft(transcript_id, "sms", content)
        await _notify_draft_ready(transcript_id, "sms", content, customer, biz_name)

        return JSONResponse({"status": "ok", "draft": content})
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to draft SMS for %s: %s", transcript_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{transcript_id}/send-email")
async def send_drafted_email(transcript_id: UUID):
    """Send the stored email draft via Resend."""
    record = await _get_transcript_or_404(transcript_id)
    drafts = record.get("drafts") or {}
    content = drafts.get("email", "")
    if not content:
        raise HTTPException(status_code=400, detail="No email draft found. Generate one first.")

    data = record.get("extracted_data") or {}
    to_email = data.get("customer_email")
    if not to_email:
        raise HTTPException(status_code=400, detail="No customer email address in extracted data")

    subject = "Following up on your call"
    body = content
    lines = content.split("\n")
    first_nonempty = next((l for l in lines if l.strip()), "")
    if first_nonempty.upper().startswith("SUBJECT:"):
        subject = first_nonempty[8:].strip()
        idx = lines.index(first_nonempty)
        body = "\n".join(lines[idx + 1:]).strip()

    try:
        from ...comms import get_email_service, EmailMessage
        svc = get_email_service()
        msg = EmailMessage(to=to_email, subject=subject, body_text=body)
        sent = await svc.send_email(msg)
        if not sent:
            raise HTTPException(status_code=500, detail="Email service returned failure")
        logger.info("Drafted email sent to %s for transcript %s", to_email, transcript_id)
        return JSONResponse({"status": "ok", "to": to_email})
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to send email for %s: %s", transcript_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{transcript_id}/send-sms")
async def send_drafted_sms(transcript_id: UUID):
    """Send the stored SMS draft via SignalWire."""
    record = await _get_transcript_or_404(transcript_id)
    drafts = record.get("drafts") or {}
    body = drafts.get("sms", "")
    if not body:
        raise HTTPException(status_code=400, detail="No SMS draft found. Generate one first.")

    data = record.get("extracted_data") or {}
    to_number = data.get("customer_phone") or record.get("from_number")
    if not to_number:
        raise HTTPException(status_code=400, detail="No customer phone number available")

    from_number = record.get("to_number", "")

    try:
        from ...comms import get_comms_service
        svc = get_comms_service()
        msg = await svc.provider.send_sms(
            to_number=to_number,
            from_number=from_number,
            body=body,
        )
        logger.info("Drafted SMS sent to %s for transcript %s", to_number, transcript_id)

        # Persist outbound SMS (fail-open)
        try:
            from ...storage.repositories.sms_message import get_sms_message_repo
            sms_repo = get_sms_message_repo()
            await sms_repo.create(
                message_sid=getattr(msg, "provider_message_id", "") or f"call_action_draft_{transcript_id.hex[:12]}",
                from_number=from_number,
                to_number=to_number,
                direction="outbound",
                body=body,
                business_context_id=record.get("business_context_id"),
                status="sent",
                source="call_action",
                source_ref=str(transcript_id),
            )
        except Exception as persist_err:
            logger.warning("Failed to persist drafted SMS: %s", persist_err)

        return JSONResponse({"status": "ok", "message_sid": msg.provider_message_id})
    except Exception as e:
        logger.error("Failed to send SMS for %s: %s", transcript_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{transcript_id}/approve-plan")
async def approve_plan(transcript_id: UUID):
    """Execute all actions in the proposed action plan.

    Iterates through proposed_actions, executes each one, and logs
    results to contact_interactions. Sends a summary notification
    when complete.
    """
    record = await _get_transcript_or_404(transcript_id)

    # Idempotency: skip if already approved/executed
    plan_status = record.get("plan_status", "pending")
    if plan_status in ("approved", "executed"):
        return JSONResponse({
            "status": "ok",
            "message": "Plan already executed",
            "results": record.get("plan_results") or [],
        })
    if plan_status == "rejected":
        return JSONResponse({"status": "ok", "message": "Plan already rejected"})

    actions = record.get("proposed_actions") or []
    data = record.get("extracted_data") or {}
    biz_name = _get_business_name(record)

    actionable = [a for a in actions if (a.get("action") or a.get("type", "none")) != "none"]
    if not actionable:
        return JSONResponse({"status": "ok", "message": "No actionable items in plan"})

    results = []
    for action in actionable:
        atype = action.get("action") or action.get("type", "")
        params = action.get("params") or {}
        try:
            result = await _execute_plan_action(
                atype, params, transcript_id, record, data, biz_name,
            )
            results.append({"action": atype, "status": "ok", "detail": result})
            logger.info("Plan action OK: %s for %s", atype, transcript_id)
        except Exception as e:
            results.append({"action": atype, "status": "error", "detail": str(e)})
            logger.error("Plan action FAIL: %s for %s: %s", atype, transcript_id, e)

    # Log approval to CRM interaction
    contact_id = record.get("contact_id")
    if contact_id:
        try:
            from ...services.crm_provider import get_crm_provider
            action_summary = ", ".join(r["action"] for r in results if r["status"] == "ok")
            await get_crm_provider().log_interaction(
                contact_id=str(contact_id),
                interaction_type="plan_approved",
                summary=f"Action plan approved: {action_summary}" if action_summary else "Plan approved (no actions succeeded)",
            )
        except Exception as e:
            logger.warning("Failed to log plan approval interaction: %s", e)

    # Persist plan status + results
    try:
        repo = get_call_transcript_repo()
        await repo.update_plan_status(transcript_id, "executed", results)
    except Exception as e:
        logger.error("Failed to persist plan status for %s: %s", transcript_id, e)

    # Send completion notification
    try:
        await _notify_plan_executed(transcript_id, results, biz_name, data)
    except Exception as e:
        logger.warning("Plan execution notification failed: %s", e)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    fail_count = len(results) - ok_count
    if fail_count:
        failed_names = [r["action"] for r in results if r["status"] == "error"]
        logger.warning(
            "Plan execution for %s: %d OK, %d failed (%s)",
            transcript_id, ok_count, fail_count, ", ".join(failed_names),
        )
    else:
        logger.info("Plan execution for %s: all %d actions succeeded", transcript_id, ok_count)
    return JSONResponse({
        "status": "ok",
        "executed": ok_count,
        "total": len(results),
        "results": results,
    })


@router.post("/{transcript_id}/reject-plan")
async def reject_plan(transcript_id: UUID):
    """Mark the action plan as rejected."""
    record = await _get_transcript_or_404(transcript_id)

    # Idempotency: skip if already decided
    plan_status = record.get("plan_status", "pending")
    if plan_status == "rejected":
        return JSONResponse({"status": "ok", "message": "Plan already rejected"})
    if plan_status in ("approved", "executed"):
        return JSONResponse({"status": "ok", "message": "Plan already executed, cannot reject"})

    contact_id = record.get("contact_id")
    if contact_id:
        try:
            from ...services.crm_provider import get_crm_provider
            await get_crm_provider().log_interaction(
                contact_id=str(contact_id),
                interaction_type="plan_rejected",
                summary="Action plan rejected by user",
            )
        except Exception as e:
            logger.warning("Failed to log plan rejection: %s", e)

    # Persist plan status
    try:
        repo = get_call_transcript_repo()
        await repo.update_plan_status(transcript_id, "rejected")
    except Exception as e:
        logger.error("Failed to persist plan rejection for %s: %s", transcript_id, e)

    logger.info("Action plan rejected for transcript %s", transcript_id)
    return JSONResponse({"status": "ok", "message": "Plan rejected"})


async def _execute_plan_action(
    action_type: str,
    params: dict,
    transcript_id: UUID,
    record: dict,
    extracted_data: dict,
    business_name: str,
) -> str:
    """Execute a single action from the plan. Returns a status message."""
    if action_type in ("book_appointment", "book_estimate", "create_appointment"):
        return await _exec_book(transcript_id, record, extracted_data, business_name)

    elif action_type == "send_email":
        return await _exec_email(transcript_id, record, extracted_data, business_name)

    elif action_type == "send_sms":
        return await _exec_sms(transcript_id, record, extracted_data)

    elif action_type == "update_contact":
        return await _exec_update_contact(record, params)

    elif action_type == "schedule_callback":
        return await _exec_callback(record, extracted_data, params)

    else:
        return f"Unknown action type: {action_type}"


async def _exec_book(
    transcript_id: UUID, record: dict, data: dict, biz_name: str,
) -> str:
    """Book an appointment from plan action."""
    from ...tools.calendar import calendar_tool

    customer = data.get("customer_name") or "Customer"
    phone = data.get("customer_phone") or record.get("from_number", "")
    email = data.get("customer_email") or ""
    address = data.get("address", "")
    services = ", ".join(data.get("services_mentioned") or []) or "Service"
    date_str = data.get("preferred_date") or ""
    time_str = data.get("preferred_time") or ""
    frequency = data.get("frequency") or ""

    ctx_id = record.get("business_context_id") or ""
    ctx = get_context_router().get_context(ctx_id) if ctx_id else None
    calendar_id = (ctx.scheduling.calendar_id if ctx else None) or None
    tz_name = ctx.hours.timezone if (ctx and ctx.hours) else settings.reminder.default_timezone

    summary = f"Estimate: {customer}"
    desc_lines = [f"Customer: {customer}", f"Phone: {phone}"]
    if email:
        desc_lines.append(f"Email: {email}")
    if address:
        desc_lines.append(f"Address: {address}")
    desc_lines.append(f"Services: {services}")
    if frequency:
        desc_lines.append(f"Frequency: {frequency.replace('_', ' ').title()}")
    if date_str or time_str:
        desc_lines.append(f"Requested: {(date_str + ' ' + time_str).strip()}")

    start_dt = _parse_event_datetime(date_str, time_str, tz_name)
    end_dt = start_dt + timedelta(hours=1)

    result = await calendar_tool.create_event(
        summary=summary, start=start_dt, end=end_dt,
        location=address or None, description="\n".join(desc_lines),
        calendar_id=calendar_id,
    )
    if not result.success:
        raise Exception(f"Calendar creation failed: {result.message}")
    return f"Booked: {start_dt.strftime('%Y-%m-%d %H:%M')}"


async def _exec_email(
    transcript_id: UUID, record: dict, data: dict, biz_name: str,
) -> str:
    """Draft and send a confirmation email."""
    to_email = data.get("customer_email")
    if not to_email:
        return "Skipped: no customer email"

    content = await _generate_draft("email", record, biz_name)
    if not content:
        return "Skipped: LLM draft generation failed"

    # Save draft for audit
    repo = get_call_transcript_repo()
    await repo.save_draft(transcript_id, "email", content)

    # Parse subject from draft
    subject = "Following up on your call"
    body = content
    lines = content.split("\n")
    first = next((l for l in lines if l.strip()), "")
    if first.upper().startswith("SUBJECT:"):
        subject = first[8:].strip()
        idx = lines.index(first)
        body = "\n".join(lines[idx + 1:]).strip()

    from ...comms import get_email_service, EmailMessage
    svc = get_email_service()
    msg = EmailMessage(to=to_email, subject=subject, body_text=body)
    sent = await svc.send_email(msg)
    if not sent:
        return "Email send failed"
    return f"Email sent to {to_email}"


async def _exec_sms(
    transcript_id: UUID, record: dict, data: dict,
) -> str:
    """Draft and send a confirmation SMS."""
    to_number = data.get("customer_phone") or record.get("from_number")
    if not to_number:
        return "Skipped: no customer phone"

    biz_name = _get_business_name(record)
    content = await _generate_draft("sms", record, biz_name)
    if not content:
        return "Skipped: LLM draft generation failed"

    repo = get_call_transcript_repo()
    await repo.save_draft(transcript_id, "sms", content)

    from ...comms import get_comms_service
    svc = get_comms_service()
    from_number = record.get("to_number", "")
    msg = await svc.provider.send_sms(
        to_number=to_number, from_number=from_number, body=content,
    )
    return f"SMS sent to {to_number}"


async def _exec_update_contact(record: dict, params: dict) -> str:
    """Update CRM contact with new info from the plan."""
    contact_id = record.get("contact_id")
    if not contact_id:
        return "Skipped: no linked contact"
    if not params:
        return "Skipped: no update params"

    from ...services.crm_provider import get_crm_provider
    await get_crm_provider().update_contact(str(contact_id), params)
    return f"Contact {contact_id} updated"


async def _exec_callback(record: dict, extracted_data: dict, params: dict) -> str:
    """Schedule a callback reminder via the reminder service."""
    from ...services.reminders import get_reminder_service

    customer = extracted_data.get("customer_name") or "Customer"
    phone = extracted_data.get("customer_phone") or record.get("from_number", "")
    reason = params.get("reason") or extracted_data.get("intent", "follow-up")
    time_str = params.get("time") or params.get("when") or "tomorrow at 9am"

    message = f"Call back {customer}"
    if phone:
        message += f" ({phone})"
    message += f" -- {reason}"

    # Parse the callback time
    import dateparser
    from datetime import datetime as _dt, timezone as _tz

    parser_settings = {
        "PREFER_DATES_FROM": "future",
        "RETURN_AS_TIMEZONE_AWARE": True,
        "TIMEZONE": settings.reminder.default_timezone,
    }
    due_at = dateparser.parse(time_str, settings=parser_settings)
    if not due_at:
        # Fallback: tomorrow 9 AM in configured timezone
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(settings.reminder.default_timezone)
        now = _dt.now(tz)
        due_at = now.replace(hour=9, minute=0, second=0, microsecond=0)
        if due_at <= now:
            from datetime import timedelta
            due_at += timedelta(days=1)

    service = get_reminder_service()
    reminder = await service.create_reminder(
        message=message,
        due_at=due_at,
        source="call_plan",
        metadata={
            "transcript_id": str(record.get("id", "")),
            "contact_id": str(record.get("contact_id", "")),
            "phone": phone,
        },
    )
    if not reminder:
        raise Exception("Reminder service returned None (disabled or at limit)")
    return f"Callback scheduled: {due_at.strftime('%Y-%m-%d %H:%M')} -- {message}"


async def _notify_plan_executed(
    transcript_id: UUID,
    results: list[dict],
    business_name: str,
    extracted_data: dict,
) -> None:
    """Send ntfy notification summarizing plan execution results."""
    if not settings.alerts.ntfy_enabled:
        return

    customer = extracted_data.get("customer_name") or "Customer"
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "error"]

    lines = [f"Customer: {customer}"]
    if ok:
        lines.append(f"Completed ({len(ok)}):")
        for r in ok:
            lines.append(f"  {r['action'].replace('_', ' ').title()}: {r['detail']}")
    if failed:
        lines.append(f"Failed ({len(failed)}):")
        for r in failed:
            lines.append(f"  {r['action'].replace('_', ' ').title()}: {r['detail']}")

    message = "\n".join(lines)
    api_url = _api_url()
    actions = f"view, View Transcript, {api_url}/api/v1/comms/call-actions/{transcript_id}/view"

    headers = {
        "Title": f"{business_name}: Plan Executed",
        "Priority": "default",
        "Tags": "white_check_mark" if not failed else "warning",
        "Actions": actions,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(_ntfy_url(), content=message, headers=headers)
        resp.raise_for_status()


@router.post("/{transcript_id}/discard")
async def discard_draft(transcript_id: UUID):
    """Acknowledge draft discard (ntfy clear=true button handler)."""
    await _get_transcript_or_404(transcript_id)
    logger.info("Draft discarded for transcript %s", transcript_id)
    return JSONResponse({"status": "ok"})
