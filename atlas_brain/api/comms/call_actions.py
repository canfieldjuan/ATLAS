"""
Call action endpoints triggered by ntfy notification buttons.

POST /comms/call-actions/{transcript_id}/book         -> create Google Calendar event
POST /comms/call-actions/{transcript_id}/sms          -> send confirmation SMS to customer
GET  /comms/call-actions/{transcript_id}/view         -> return transcript + extracted data
POST /comms/call-actions/{transcript_id}/draft-email  -> LLM drafts confirmation email
POST /comms/call-actions/{transcript_id}/draft-sms    -> LLM drafts confirmation SMS
POST /comms/call-actions/{transcript_id}/send-email   -> send the drafted email via Resend
POST /comms/call-actions/{transcript_id}/send-sms     -> send the drafted SMS via SignalWire
POST /comms/call-actions/{transcript_id}/discard      -> discard draft (ntfy clear handler)
"""

import asyncio
import logging
import re
from uuid import UUID

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

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
        from ...tools.calendar import get_calendar_tool
        cal = get_calendar_tool()

        title = f"Estimate: {customer}"
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
            desc_lines.append(f"Requested: {date_str} {time_str}".strip())
        description = "\n".join(desc_lines)

        result = await cal.create_event(
            title=title,
            description=description,
            start=f"{date_str} {time_str}".strip() or "TBD",
            calendar_hint="effingham_maids",
        )
        logger.info("Calendar event created for transcript %s: %s", transcript_id, result)

        try:
            from ...comms import EFFINGHAM_MAIDS_CONTEXT
            biz_name = EFFINGHAM_MAIDS_CONTEXT.name
            await _notify_booking_confirmed(transcript_id, record, biz_name)
        except Exception as _notify_err:
            logger.warning("Booking confirmed ntfy failed for %s: %s", transcript_id, _notify_err)

        return JSONResponse({"status": "ok", "event": result})

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
    appt = f" for {date_str} {time_str}".strip() if (date_str or time_str) else ""

    body = (
        f"Hi {customer}, this is Effingham Office Maids following up on your call{appt}. "
        f"We'll be in touch shortly to confirm details. Reply STOP to opt out."
    )

    try:
        from ...comms import get_comms_service
        svc = get_comms_service()
        from_number = "+16183683696"
        msg = await svc.provider.send_sms(
            to_number=to_number,
            from_number=from_number,
            body=body,
        )
        logger.info("SMS sent to %s for transcript %s", to_number, transcript_id)
        return JSONResponse({"status": "ok", "message_sid": msg.provider_message_id})

    except Exception as e:
        logger.error("Failed to send SMS for %s: %s", transcript_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{transcript_id}/view")
async def view_transcript(transcript_id: UUID):
    """Return the transcript and extracted data as plain text."""
    record = await _get_transcript_or_404(transcript_id)
    data = record.get("extracted_data") or {}

    lines = [
        f"Call: {record.get('call_sid', '')}",
        f"From: {record.get('from_number', '')}",
        f"Duration: {record.get('duration_seconds', 0)}s",
        f"Status: {record.get('status', '')}",
        "",
        "--- EXTRACTED ---",
    ]
    for k, v in data.items():
        if v not in (None, "", [], False):
            lines.append(f"{k}: {v}")

    lines += ["", "--- TRANSCRIPT ---", record.get("transcript") or "(none)"]
    return PlainTextResponse("\n".join(lines))


@router.post("/{transcript_id}/draft-email")
async def draft_email(transcript_id: UUID):
    """Generate a confirmation email draft using LLM and notify via ntfy."""
    record = await _get_transcript_or_404(transcript_id)
    data = record.get("extracted_data") or {}
    customer = data.get("customer_name") or "Customer"

    from ...comms import EFFINGHAM_MAIDS_CONTEXT
    biz_name = EFFINGHAM_MAIDS_CONTEXT.name

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

    from ...comms import EFFINGHAM_MAIDS_CONTEXT
    biz_name = EFFINGHAM_MAIDS_CONTEXT.name

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
    if lines and lines[0].upper().startswith("SUBJECT:"):
        subject = lines[0][8:].strip()
        body = "\n".join(lines[1:]).strip()

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
        return JSONResponse({"status": "ok", "message_sid": msg.provider_message_id})
    except Exception as e:
        logger.error("Failed to send SMS for %s: %s", transcript_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{transcript_id}/discard")
async def discard_draft(transcript_id: UUID):
    """Acknowledge draft discard (ntfy clear=true button handler)."""
    await _get_transcript_or_404(transcript_id)
    logger.info("Draft discarded for transcript %s", transcript_id)
    return JSONResponse({"status": "ok"})
