"""
SMS management endpoints.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..service import get_comms_service

logger = logging.getLogger("atlas.comms.api.sms")
router = APIRouter()


class SendSMSRequest(BaseModel):
    """Request to send an SMS."""
    to_number: str = Field(..., description="Recipient phone number (E.164 format)")
    body: str = Field(..., description="Message text")
    context_id: Optional[str] = Field(None, description="Business context ID")
    media_urls: Optional[list[str]] = Field(None, description="MMS attachment URLs")


class SendSMSResponse(BaseModel):
    """Response from sending an SMS."""
    success: bool
    message_id: Optional[str] = None
    provider_message_id: Optional[str] = None
    error: Optional[str] = None


@router.post("/send", response_model=SendSMSResponse)
async def send_sms(request: SendSMSRequest):
    """Send an SMS message."""
    service = get_comms_service()

    if not service.is_connected:
        raise HTTPException(
            status_code=503,
            detail="Communications service not connected"
        )

    message = await service.send_sms(
        to_number=request.to_number,
        body=request.body,
        context_id=request.context_id,
        media_urls=request.media_urls,
    )

    if message is None:
        return SendSMSResponse(
            success=False,
            error="Failed to send SMS"
        )

    return SendSMSResponse(
        success=True,
        message_id=str(message.id),
        provider_message_id=message.provider_message_id,
    )
