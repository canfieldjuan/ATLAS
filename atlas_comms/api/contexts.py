"""
Business context management endpoints.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..context import get_context_router

logger = logging.getLogger("atlas.comms.api.contexts")
router = APIRouter()


class BusinessHoursResponse(BaseModel):
    """Business hours response."""
    monday_open: Optional[str] = None
    monday_close: Optional[str] = None
    tuesday_open: Optional[str] = None
    tuesday_close: Optional[str] = None
    wednesday_open: Optional[str] = None
    wednesday_close: Optional[str] = None
    thursday_open: Optional[str] = None
    thursday_close: Optional[str] = None
    friday_open: Optional[str] = None
    friday_close: Optional[str] = None
    saturday_open: Optional[str] = None
    saturday_close: Optional[str] = None
    sunday_open: Optional[str] = None
    sunday_close: Optional[str] = None
    timezone: str


class ContextResponse(BaseModel):
    """Business context response."""
    id: str
    name: str
    description: str
    phone_numbers: list[str]
    greeting: str
    voice_name: str
    business_type: str
    services: list[str]
    service_area: str
    hours: BusinessHoursResponse
    scheduling_enabled: bool
    sms_enabled: bool


class ContextStatusResponse(BaseModel):
    """Current status of a business context."""
    context_id: str
    is_open: bool
    next_open: Optional[str] = None
    message: str


@router.get("", response_model=list[ContextResponse])
async def list_contexts():
    """List all registered business contexts."""
    router = get_context_router()
    contexts = router.list_contexts()

    return [
        ContextResponse(
            id=ctx.id,
            name=ctx.name,
            description=ctx.description,
            phone_numbers=ctx.phone_numbers,
            greeting=ctx.greeting,
            voice_name=ctx.voice_name,
            business_type=ctx.business_type,
            services=ctx.services,
            service_area=ctx.service_area,
            hours=BusinessHoursResponse(
                monday_open=ctx.hours.monday_open,
                monday_close=ctx.hours.monday_close,
                tuesday_open=ctx.hours.tuesday_open,
                tuesday_close=ctx.hours.tuesday_close,
                wednesday_open=ctx.hours.wednesday_open,
                wednesday_close=ctx.hours.wednesday_close,
                thursday_open=ctx.hours.thursday_open,
                thursday_close=ctx.hours.thursday_close,
                friday_open=ctx.hours.friday_open,
                friday_close=ctx.hours.friday_close,
                saturday_open=ctx.hours.saturday_open,
                saturday_close=ctx.hours.saturday_close,
                sunday_open=ctx.hours.sunday_open,
                sunday_close=ctx.hours.sunday_close,
                timezone=ctx.hours.timezone,
            ),
            scheduling_enabled=ctx.scheduling.enabled,
            sms_enabled=ctx.sms_enabled,
        )
        for ctx in contexts
    ]


@router.get("/{context_id}", response_model=ContextResponse)
async def get_context(context_id: str):
    """Get a specific business context."""
    router = get_context_router()
    ctx = router.get_context(context_id)

    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")

    return ContextResponse(
        id=ctx.id,
        name=ctx.name,
        description=ctx.description,
        phone_numbers=ctx.phone_numbers,
        greeting=ctx.greeting,
        voice_name=ctx.voice_name,
        business_type=ctx.business_type,
        services=ctx.services,
        service_area=ctx.service_area,
        hours=BusinessHoursResponse(
            monday_open=ctx.hours.monday_open,
            monday_close=ctx.hours.monday_close,
            tuesday_open=ctx.hours.tuesday_open,
            tuesday_close=ctx.hours.tuesday_close,
            wednesday_open=ctx.hours.wednesday_open,
            wednesday_close=ctx.hours.wednesday_close,
            thursday_open=ctx.hours.thursday_open,
            thursday_close=ctx.hours.thursday_close,
            friday_open=ctx.hours.friday_open,
            friday_close=ctx.hours.friday_close,
            saturday_open=ctx.hours.saturday_open,
            saturday_close=ctx.hours.saturday_close,
            sunday_open=ctx.hours.sunday_open,
            sunday_close=ctx.hours.sunday_close,
            timezone=ctx.hours.timezone,
        ),
        scheduling_enabled=ctx.scheduling.enabled,
        sms_enabled=ctx.sms_enabled,
    )


@router.get("/{context_id}/status", response_model=ContextStatusResponse)
async def get_context_status(context_id: str):
    """Get the current status of a business context (open/closed)."""
    router = get_context_router()
    ctx = router.get_context(context_id)

    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Context {context_id} not found")

    status = router.get_business_status(ctx)

    return ContextStatusResponse(
        context_id=context_id,
        is_open=status["is_open"],
        next_open=status["next_open"].isoformat() if status["next_open"] else None,
        message=status["message"],
    )
