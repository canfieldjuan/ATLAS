"""
Communications management API.

Provides endpoints for:
- Making outbound calls
- Sending SMS messages
- Managing business contexts
- Checking availability and scheduling
- Recording reconciliation (missed webhooks)
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ...comms import (
    comms_settings,
    BusinessContext,
    EFFINGHAM_MAIDS_CONTEXT,
    get_context_router,
    get_provider,
    list_providers,
)
from atlas_comms.services import scheduling_service, TimeSlot

logger = logging.getLogger("atlas.api.comms.management")

router = APIRouter()


# === Request/Response Models ===


class MakeCallRequest(BaseModel):
    """Request to make an outbound call."""
    to_number: str = Field(..., max_length=20, description="Phone number to call (E.164 format)")
    from_number: Optional[str] = Field(None, max_length=20, description="Caller ID number")
    context_id: Optional[str] = Field(None, max_length=100, description="Business context ID")


class MakeCallResponse(BaseModel):
    """Response from making a call."""
    call_id: str
    provider_call_id: str
    status: str
    from_number: str
    to_number: str


class SendSMSRequest(BaseModel):
    """Request to send an SMS."""
    to_number: str = Field(..., max_length=20, description="Phone number (E.164 format)")
    from_number: Optional[str] = Field(None, max_length=20, description="Sender number")
    body: str = Field(..., max_length=1600, description="Message text")
    media_urls: Optional[list[str]] = Field(None, max_length=10, description="MMS attachment URLs")
    context_id: Optional[str] = Field(None, max_length=100, description="Business context ID")


class SendSMSResponse(BaseModel):
    """Response from sending SMS."""
    message_id: str
    provider_message_id: str
    status: str
    to_number: str


class AvailabilityRequest(BaseModel):
    """Request to check appointment availability."""
    context_id: str = Field(..., description="Business context ID")
    date: Optional[str] = Field(None, description="Specific date (YYYY-MM-DD)")
    duration_minutes: Optional[int] = Field(None, description="Appointment duration")
    days_ahead: int = Field(7, ge=1, le=90, description="Days to search ahead")


class TimeSlotResponse(BaseModel):
    """An available time slot."""
    start: str
    end: str
    duration_minutes: int
    display: str


class AvailabilityResponse(BaseModel):
    """Response with available time slots."""
    context_id: str
    slots: list[TimeSlotResponse]
    message: str


class BookAppointmentRequest(BaseModel):
    """Request to book an appointment."""
    context_id: str = Field(..., max_length=100)
    start_time: str = Field(..., max_length=40, description="ISO format start time")
    end_time: str = Field(..., max_length=40, description="ISO format end time")
    customer_name: str = Field(..., max_length=200)
    customer_phone: str = Field(..., max_length=20)
    customer_email: Optional[str] = Field(None, max_length=254)
    service_type: Optional[str] = Field(None, max_length=200)
    location: Optional[str] = Field(None, max_length=500)
    notes: Optional[str] = Field(None, max_length=2000)


class BookAppointmentResponse(BaseModel):
    """Response from booking appointment."""
    appointment_id: str
    summary: str
    start: str
    end: str
    customer_name: str
    success: bool
    message: str


class ContextResponse(BaseModel):
    """Business context info."""
    id: str
    name: str
    description: str
    phone_numbers: list[str]
    is_open: bool
    greeting: str
    services: list[str]


# === Endpoints ===


@router.get("/status")
async def get_comms_status():
    """Get communications system status."""
    return {
        "enabled": comms_settings.enabled,
        "provider": comms_settings.provider,
        "providers_available": list_providers(),
        "webhook_base_url": comms_settings.webhook_base_url or None,
    }


@router.get("/contexts")
async def list_contexts() -> list[ContextResponse]:
    """List all registered business contexts."""
    context_router = get_context_router()
    contexts = context_router.list_contexts()

    results = []
    for ctx in contexts:
        status = context_router.get_business_status(ctx)
        results.append(ContextResponse(
            id=ctx.id,
            name=ctx.name,
            description=ctx.description,
            phone_numbers=ctx.phone_numbers,
            is_open=status["is_open"],
            greeting=ctx.greeting,
            services=ctx.services,
        ))

    return results


@router.get("/contexts/{context_id}")
async def get_context(context_id: str) -> ContextResponse:
    """Get a specific business context."""
    context_router = get_context_router()
    ctx = context_router.get_context(context_id)

    if not ctx:
        raise HTTPException(status_code=404, detail="Context not found")

    status = context_router.get_business_status(ctx)

    return ContextResponse(
        id=ctx.id,
        name=ctx.name,
        description=ctx.description,
        phone_numbers=ctx.phone_numbers,
        is_open=status["is_open"],
        greeting=ctx.greeting,
        services=ctx.services,
    )


@router.post("/calls")
async def make_call(request: MakeCallRequest) -> MakeCallResponse:
    """Make an outbound call."""
    if not comms_settings.enabled:
        raise HTTPException(status_code=503, detail="Communications not enabled")

    try:
        provider = get_provider()

        if not provider.is_connected:
            await provider.connect()

        # Determine from number
        from_number = request.from_number
        if not from_number and request.context_id:
            context_router = get_context_router()
            ctx = context_router.get_context(request.context_id)
            if ctx and ctx.phone_numbers:
                from_number = ctx.phone_numbers[0]

        if not from_number:
            raise HTTPException(
                status_code=400,
                detail="from_number required or context must have phone numbers"
            )

        call = await asyncio.wait_for(
            provider.make_call(
                to_number=request.to_number,
                from_number=from_number,
                context_id=request.context_id,
            ),
            timeout=30.0,
        )

        return MakeCallResponse(
            call_id=str(call.id),
            provider_call_id=call.provider_call_id,
            status=call.state.value,
            from_number=call.from_number,
            to_number=call.to_number,
        )

    except Exception as e:
        logger.error("Failed to make call: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to make call")


@router.post("/sms")
async def send_sms(request: SendSMSRequest) -> SendSMSResponse:
    """Send an SMS message."""
    if not comms_settings.enabled:
        raise HTTPException(status_code=503, detail="Communications not enabled")

    try:
        provider = get_provider()

        if not provider.is_connected:
            await provider.connect()

        # Determine from number
        from_number = request.from_number
        if not from_number and request.context_id:
            context_router = get_context_router()
            ctx = context_router.get_context(request.context_id)
            if ctx and ctx.phone_numbers:
                from_number = ctx.phone_numbers[0]

        if not from_number:
            raise HTTPException(
                status_code=400,
                detail="from_number required or context must have phone numbers"
            )

        message = await asyncio.wait_for(
            provider.send_sms(
                to_number=request.to_number,
                from_number=from_number,
                body=request.body,
                media_urls=request.media_urls[:10] if request.media_urls else None,
                context_id=request.context_id,
            ),
            timeout=30.0,
        )

        return SendSMSResponse(
            message_id=str(message.id),
            provider_message_id=message.provider_message_id,
            status=message.status,
            to_number=message.to_number,
        )

    except Exception as e:
        logger.error("Failed to send SMS: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to send SMS")


@router.post("/availability")
async def check_availability(request: AvailabilityRequest) -> AvailabilityResponse:
    """Check appointment availability for a business context."""
    context_router = get_context_router()
    ctx = context_router.get_context(request.context_id)

    if not ctx:
        raise HTTPException(status_code=404, detail="Context not found")

    if not ctx.scheduling.enabled:
        raise HTTPException(status_code=400, detail="Scheduling not enabled for this context")

    # Parse date if provided
    date = None
    if request.date:
        try:
            date = datetime.strptime(request.date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")

    try:
        slots = await scheduling_service.get_available_slots(
            context=ctx,
            date=date,
            duration_minutes=request.duration_minutes,
            days_ahead=request.days_ahead,
        )

        slot_responses = [
            TimeSlotResponse(
                start=slot.start.isoformat(),
                end=slot.end.isoformat(),
                duration_minutes=slot.duration_minutes,
                display=str(slot),
            )
            for slot in slots
        ]

        message = scheduling_service.format_slots_for_speech(slots)

        return AvailabilityResponse(
            context_id=request.context_id,
            slots=slot_responses,
            message=message,
        )

    except Exception as e:
        logger.error("Failed to check availability: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to check availability")


@router.post("/appointments")
async def book_appointment(request: BookAppointmentRequest) -> BookAppointmentResponse:
    """Book an appointment."""
    context_router = get_context_router()
    ctx = context_router.get_context(request.context_id)

    if not ctx:
        raise HTTPException(status_code=404, detail="Context not found")

    if not ctx.scheduling.enabled:
        raise HTTPException(status_code=400, detail="Scheduling not enabled")

    try:
        start = datetime.fromisoformat(request.start_time)
        end = datetime.fromisoformat(request.end_time)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid datetime format")

    slot = TimeSlot(start=start, end=end)

    try:
        appointment = await scheduling_service.book_appointment(
            context=ctx,
            slot=slot,
            customer_name=request.customer_name,
            customer_phone=request.customer_phone,
            customer_email=request.customer_email,
            service_type=request.service_type,
            location=request.location,
            notes=request.notes,
        )

        if not appointment:
            return BookAppointmentResponse(
                appointment_id="",
                summary="",
                start=request.start_time,
                end=request.end_time,
                customer_name=request.customer_name,
                success=False,
                message="Failed to book appointment. Please try again.",
            )

        return BookAppointmentResponse(
            appointment_id=appointment.id,
            summary=appointment.summary,
            start=appointment.start.isoformat(),
            end=appointment.end.isoformat(),
            customer_name=appointment.customer_name,
            success=True,
            message=f"Appointment booked for {slot}",
        )

    except Exception as e:
        logger.error("Failed to book appointment: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to book appointment")


@router.delete("/appointments/{appointment_id}")
async def cancel_appointment(
    appointment_id: str,
    context_id: str = Query(..., description="Business context ID"),
):
    """Cancel an appointment."""
    context_router = get_context_router()
    ctx = context_router.get_context(context_id)

    if not ctx:
        raise HTTPException(status_code=404, detail="Context not found")

    try:
        success = await scheduling_service.cancel_appointment(ctx, appointment_id)

        if success:
            return {"success": True, "message": "Appointment cancelled"}
        else:
            raise HTTPException(status_code=500, detail="Failed to cancel appointment")

    except Exception as e:
        logger.error("Failed to cancel appointment: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to cancel appointment")


@router.post("/recordings/reconcile")
async def reconcile_recordings(
    limit: int = Query(default=50, ge=1, le=200, description="Max recordings to check"),
):
    """Check SignalWire for recent recordings missing from our DB.

    Queries the provider's recording list, cross-references against
    call_transcripts, and spawns the intelligence pipeline for any
    recordings that were missed (e.g. webhook never arrived).
    """
    if not comms_settings.enabled:
        raise HTTPException(status_code=503, detail="Communications not enabled")

    try:
        provider = get_provider()
        if not provider.is_connected:
            await provider.connect()
    except Exception as e:
        logger.error("Provider unavailable: %s", e)
        raise HTTPException(status_code=503, detail="Provider unavailable")

    # List recent recordings from SignalWire (Twilio-compatible API)
    try:
        client = getattr(provider, "_client", None)
        if not client:
            raise HTTPException(status_code=503, detail="No REST client available")
        recordings = client.recordings.list(limit=limit)
    except HTTPException:
        raise
    except Exception as e:
        # SignalWire's Twilio-compatible SDK can throw JSON parse errors
        # even when the REST API returned 200. This is a known issue.
        if "Expecting value" in str(e):
            logger.warning("SignalWire recordings.list() SDK parse error: %s", e)
            return {
                "checked": 0,
                "already_processed": 0,
                "missing": 0,
                "spawned": 0,
                "missing_recordings": [],
                "note": "SignalWire SDK parse error -- recordings API may be unavailable",
            }
        logger.error("Failed to list recordings from provider: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list recordings")

    from ...storage.repositories.call_transcript import get_call_transcript_repo
    repo = get_call_transcript_repo()

    missing = []
    already_have = 0
    spawned = 0

    for rec in recordings:
        call_sid = getattr(rec, "call_sid", None)
        if not call_sid:
            continue

        # Check if we already have this recording
        try:
            existing = await repo.get_by_call_sid(call_sid)
            if existing:
                already_have += 1
                continue
        except Exception as e:
            logger.warning("DB check failed for call_sid %s: %s", call_sid, e)
            continue

        duration = int(getattr(rec, "duration", 0) or 0)
        recording_url = getattr(rec, "uri", "") or ""
        # Build the actual media URL from the recording SID
        rec_sid = getattr(rec, "sid", "")
        if rec_sid and not recording_url.startswith("http"):
            space = comms_settings.signalwire_space or ""
            recording_url = (
                f"https://{space}.signalwire.com"
                f"/api/laml/2010-04-01/Accounts/"
                f"{comms_settings.signalwire_project_id}/Recordings/{rec_sid}"
            )

        missing.append({
            "call_sid": call_sid,
            "recording_sid": rec_sid,
            "duration": duration,
            "recording_url": recording_url,
        })

        # Spawn processing
        from .webhooks import _spawn_recording_processing
        _spawn_recording_processing(call_sid, recording_url, duration)
        spawned += 1

    logger.info(
        "Recording reconciliation: checked=%d already_have=%d missing=%d spawned=%d",
        len(recordings), already_have, len(missing), spawned,
    )

    return {
        "checked": len(recordings),
        "already_processed": already_have,
        "missing": len(missing),
        "spawned": spawned,
        "missing_recordings": missing,
    }
