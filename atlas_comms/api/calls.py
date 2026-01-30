"""
Call management endpoints.
"""

import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..service import get_comms_service
from ..core.protocols import CallState, CallDirection

logger = logging.getLogger("atlas.comms.api.calls")
router = APIRouter()


class CallResponse(BaseModel):
    """Call information response."""
    id: str
    provider_call_id: str
    from_number: str
    to_number: str
    direction: str
    state: str
    context_id: Optional[str] = None
    initiated_at: datetime
    answered_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None


class MakeCallRequest(BaseModel):
    """Request to make an outbound call."""
    to_number: str = Field(..., description="Phone number to call (E.164 format)")
    context_id: Optional[str] = Field(None, description="Business context ID")


class MakeCallResponse(BaseModel):
    """Response from making a call."""
    success: bool
    call_id: Optional[str] = None
    provider_call_id: Optional[str] = None
    error: Optional[str] = None


@router.get("", response_model=list[CallResponse])
async def list_calls():
    """List all active calls."""
    service = get_comms_service()
    calls = service.get_active_calls()

    return [
        CallResponse(
            id=str(call.id),
            provider_call_id=call.provider_call_id,
            from_number=call.from_number,
            to_number=call.to_number,
            direction=call.direction.value,
            state=call.state.value,
            context_id=call.context_id,
            initiated_at=call.initiated_at,
            answered_at=call.answered_at,
            ended_at=call.ended_at,
            duration_seconds=call.duration_seconds,
        )
        for call in calls
    ]


@router.get("/{call_id}", response_model=CallResponse)
async def get_call(call_id: str):
    """Get a specific call by ID."""
    service = get_comms_service()
    call = service.get_call(call_id)

    if call is None:
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")

    return CallResponse(
        id=str(call.id),
        provider_call_id=call.provider_call_id,
        from_number=call.from_number,
        to_number=call.to_number,
        direction=call.direction.value,
        state=call.state.value,
        context_id=call.context_id,
        initiated_at=call.initiated_at,
        answered_at=call.answered_at,
        ended_at=call.ended_at,
        duration_seconds=call.duration_seconds,
    )


@router.post("/outbound", response_model=MakeCallResponse)
async def make_call(request: MakeCallRequest):
    """Make an outbound call."""
    service = get_comms_service()

    if not service.is_connected:
        raise HTTPException(
            status_code=503,
            detail="Communications service not connected"
        )

    call = await service.make_call(
        to_number=request.to_number,
        context_id=request.context_id,
    )

    if call is None:
        return MakeCallResponse(
            success=False,
            error="Failed to initiate call"
        )

    return MakeCallResponse(
        success=True,
        call_id=str(call.id),
        provider_call_id=call.provider_call_id,
    )


@router.post("/{call_id}/hangup")
async def hangup_call(call_id: str):
    """Hang up an active call."""
    service = get_comms_service()

    success = await service.hangup_call(call_id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to hangup call {call_id}"
        )

    return {"success": True, "call_id": call_id}


@router.post("/{call_id}/transfer")
async def transfer_call(call_id: str, to_number: str = Query(...)):
    """Transfer an active call to another number."""
    service = get_comms_service()

    success = await service.transfer_call(call_id, to_number)

    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to transfer call {call_id}"
        )

    return {"success": True, "call_id": call_id, "transferred_to": to_number}
