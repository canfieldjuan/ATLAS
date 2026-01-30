"""
Appointment scheduling endpoints.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..context import get_context_router
from ..services import scheduling_service, TimeSlot, Appointment

logger = logging.getLogger("atlas.comms.api.scheduling")
router = APIRouter()


class TimeSlotResponse(BaseModel):
    """Available time slot response."""
    start: str
    end: str
    duration_minutes: int


class AvailableSlotsResponse(BaseModel):
    """Response with available appointment slots."""
    context_id: str
    slots: list[TimeSlotResponse]
    speech_text: str


class BookAppointmentRequest(BaseModel):
    """Request to book an appointment."""
    context_id: str = Field(..., description="Business context ID")
    start_time: str = Field(..., description="ISO format start time")
    end_time: str = Field(..., description="ISO format end time")
    customer_name: str = Field(..., description="Customer name")
    customer_phone: str = Field(..., description="Customer phone (E.164)")
    customer_email: Optional[str] = Field(None, description="Customer email")
    service_type: Optional[str] = Field(None, description="Type of service")
    location: Optional[str] = Field(None, description="Appointment location")
    notes: Optional[str] = Field(None, description="Additional notes")


class AppointmentResponse(BaseModel):
    """Booked appointment response."""
    id: str
    calendar_event_id: Optional[str]
    start: str
    end: str
    customer_name: str
    service_type: str
    business_context_id: str


class CancelAppointmentRequest(BaseModel):
    """Request to cancel an appointment."""
    context_id: str = Field(..., description="Business context ID")
    appointment_id: str = Field(..., description="Appointment/event ID")


@router.get("/available", response_model=AvailableSlotsResponse)
async def get_available_slots(
    context_id: str = Query(..., description="Business context ID"),
    date: Optional[str] = Query(None, description="Specific date (YYYY-MM-DD)"),
    duration_minutes: Optional[int] = Query(None, description="Appointment duration"),
    days_ahead: int = Query(7, description="Days ahead to search"),
):
    """Get available appointment slots for a business context."""
    context_router = get_context_router()
    context = context_router.get_context(context_id)

    if context is None:
        raise HTTPException(status_code=404, detail="Context not found")

    if not context.scheduling.enabled:
        raise HTTPException(status_code=400, detail="Scheduling not enabled")

    # Parse date if provided
    parsed_date = None
    if date:
        try:
            parsed_date = datetime.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")

    slots = await scheduling_service.get_available_slots(
        context=context,
        date=parsed_date,
        duration_minutes=duration_minutes,
        days_ahead=days_ahead,
    )

    slot_responses = [
        TimeSlotResponse(
            start=s.start.isoformat(),
            end=s.end.isoformat(),
            duration_minutes=s.duration_minutes,
        )
        for s in slots
    ]

    speech_text = scheduling_service.format_slots_for_speech(slots)

    return AvailableSlotsResponse(
        context_id=context_id,
        slots=slot_responses,
        speech_text=speech_text,
    )


@router.post("/book", response_model=AppointmentResponse)
async def book_appointment(request: BookAppointmentRequest):
    """Book an appointment."""
    context_router = get_context_router()
    context = context_router.get_context(request.context_id)

    if context is None:
        raise HTTPException(status_code=404, detail="Context not found")

    if not context.scheduling.enabled:
        raise HTTPException(status_code=400, detail="Scheduling not enabled")

    # Parse times
    try:
        start_time = datetime.fromisoformat(request.start_time)
        end_time = datetime.fromisoformat(request.end_time)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid time format")

    slot = TimeSlot(start=start_time, end=end_time)

    appointment = await scheduling_service.book_appointment(
        context=context,
        slot=slot,
        customer_name=request.customer_name,
        customer_phone=request.customer_phone,
        customer_email=request.customer_email,
        service_type=request.service_type,
        location=request.location,
        notes=request.notes,
    )

    if appointment is None:
        raise HTTPException(status_code=500, detail="Failed to book appointment")

    return AppointmentResponse(
        id=str(appointment.id),
        calendar_event_id=appointment.calendar_event_id,
        start=appointment.start.isoformat(),
        end=appointment.end.isoformat(),
        customer_name=appointment.customer_name,
        service_type=appointment.service_type,
        business_context_id=appointment.business_context_id,
    )


@router.post("/cancel")
async def cancel_appointment(request: CancelAppointmentRequest):
    """Cancel an appointment."""
    context_router = get_context_router()
    context = context_router.get_context(request.context_id)

    if context is None:
        raise HTTPException(status_code=404, detail="Context not found")

    success = await scheduling_service.cancel_appointment(
        context=context,
        appointment_id=request.appointment_id,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to cancel appointment")

    return {"success": True, "message": "Appointment cancelled"}
