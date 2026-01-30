"""
Health check endpoints for Atlas Comms.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..core.config import comms_settings
from ..service import get_comms_service

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str = "atlas_comms"
    version: str = "0.1.0"
    timestamp: datetime
    comms_enabled: bool
    comms_connected: bool
    provider: Optional[str] = None
    active_calls: int = 0


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Get service health status."""
    service = get_comms_service()

    return HealthResponse(
        status="healthy" if service.is_connected or not comms_settings.enabled else "degraded",
        timestamp=datetime.utcnow(),
        comms_enabled=comms_settings.enabled,
        comms_connected=service.is_connected,
        provider=service.provider.name if service.provider else None,
        active_calls=len(service.get_active_calls()),
    )


@router.get("/ping")
async def ping():
    """Simple ping endpoint."""
    return {"pong": True}
