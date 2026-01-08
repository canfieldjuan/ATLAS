"""
FastAPI dependencies for service injection.

These dependencies provide the active AI service instances to API endpoints,
raising appropriate HTTP errors if services are not available.
"""

from fastapi import HTTPException, status

from ..services import stt_registry, vlm_registry
from ..services.protocols import STTService, VLMService


def get_vlm() -> VLMService:
    """
    FastAPI dependency that provides the active VLM service.

    Raises:
        HTTPException: 503 if no VLM model is currently loaded
    """
    service = vlm_registry.get_active()
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No VLM model is currently loaded. Use /models/vlm/activate to load one.",
        )
    return service


def get_stt() -> STTService:
    """
    FastAPI dependency that provides the active STT service.

    Raises:
        HTTPException: 503 if no STT model is currently loaded
    """
    service = stt_registry.get_active()
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No STT model is currently loaded. Use /models/stt/activate to load one.",
        )
    return service
