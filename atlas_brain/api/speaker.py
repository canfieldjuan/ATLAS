"""
Speaker identification API endpoints.

Provides voice enrollment and verification endpoints.
"""

import base64
import logging
from typing import Optional
from uuid import UUID, uuid4

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..config import settings
from ..storage import db_settings
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.speaker")

router = APIRouter(prefix="/speaker", tags=["speaker"])


def _check_enabled():
    """Check if speaker ID is enabled."""
    if not settings.speaker_id.enabled:
        raise HTTPException(
            status_code=503,
            detail="Speaker identification is disabled"
        )


def _check_db_enabled():
    """Check if database is enabled."""
    if not db_settings.enabled:
        raise HTTPException(
            status_code=503,
            detail="Database persistence is disabled"
        )
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Database not initialized"
        )


class StartEnrollmentRequest(BaseModel):
    """Request to start voice enrollment."""
    user_name: str


class StartEnrollmentResponse(BaseModel):
    """Response from starting enrollment."""
    session_id: str
    user_id: str
    user_name: str
    samples_collected: int
    samples_needed: int


class AddSampleRequest(BaseModel):
    """Request to add voice sample."""
    session_id: str
    audio_base64: str
    sample_rate: int = 16000


class AddSampleResponse(BaseModel):
    """Response from adding sample."""
    session_id: str
    samples_collected: int
    samples_needed: int
    is_ready: bool
    error: Optional[str] = None


class CompleteEnrollmentRequest(BaseModel):
    """Request to complete enrollment."""
    session_id: str


class CompleteEnrollmentResponse(BaseModel):
    """Response from completing enrollment."""
    success: bool
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    samples_used: Optional[int] = None
    error: Optional[str] = None


class VerifyRequest(BaseModel):
    """Request to verify speaker."""
    audio_base64: str
    sample_rate: int = 16000


class VerifyResponse(BaseModel):
    """Response from speaker verification."""
    matched: bool
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    confidence: float
    is_known: bool
    threshold: float


class EnrolledUserResponse(BaseModel):
    """Enrolled user info."""
    id: str
    name: str
    created_at: str
    updated_at: str


@router.get("/status")
async def get_status():
    """Get speaker ID service status."""
    return {
        "enabled": settings.speaker_id.enabled,
        "require_known_speaker": settings.speaker_id.require_known_speaker,
        "confidence_threshold": settings.speaker_id.confidence_threshold,
        "model": settings.speaker_id.default_model,
    }


@router.post("/enroll/start", response_model=StartEnrollmentResponse)
async def start_enrollment(request: StartEnrollmentRequest):
    """
    Start voice enrollment for a user.

    Creates or finds user and starts enrollment session.
    Call /enroll/sample to add voice samples.
    """
    _check_enabled()
    _check_db_enabled()

    from ..services.speaker_id import get_speaker_id_service
    from ..storage.repositories.speaker import get_speaker_repo

    try:
        repo = get_speaker_repo()
        service = get_speaker_id_service()

        # Get or create user
        user_id = await repo.get_or_create_user(request.user_name)

        # Start enrollment session
        session_id = str(uuid4())
        result = service.start_enrollment(session_id, user_id, request.user_name)

        return StartEnrollmentResponse(
            session_id=result["session_id"],
            user_id=result["user_id"],
            user_name=result["user_name"],
            samples_collected=result["samples_collected"],
            samples_needed=result["samples_needed"],
        )
    except Exception as e:
        logger.error("Failed to start enrollment: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enroll/sample", response_model=AddSampleResponse)
async def add_enrollment_sample(request: AddSampleRequest):
    """
    Add voice sample to enrollment session.

    Audio should be base64-encoded PCM (int16, mono).
    Repeat until samples_needed samples are collected.
    """
    _check_enabled()

    from ..services.speaker_id import get_speaker_id_service

    try:
        service = get_speaker_id_service()

        # Decode audio
        audio_bytes = base64.b64decode(request.audio_base64)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio.astype(np.float32) / 32768.0

        # Add sample
        result = await service.add_enrollment_sample(
            request.session_id,
            audio_float,
            request.sample_rate,
        )

        if "error" in result:
            return AddSampleResponse(
                session_id=request.session_id,
                samples_collected=result.get("samples_collected", 0),
                samples_needed=result.get("samples_needed", 3),
                is_ready=False,
                error=result["error"],
            )

        return AddSampleResponse(
            session_id=result["session_id"],
            samples_collected=result["samples_collected"],
            samples_needed=result["samples_needed"],
            is_ready=result["is_ready"],
        )
    except Exception as e:
        logger.error("Failed to add enrollment sample: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enroll/complete", response_model=CompleteEnrollmentResponse)
async def complete_enrollment(request: CompleteEnrollmentRequest):
    """
    Complete enrollment and store voice embedding.

    Only call when is_ready=True from /enroll/sample.
    """
    _check_enabled()
    _check_db_enabled()

    from ..services.speaker_id import get_speaker_id_service

    try:
        service = get_speaker_id_service()
        result = await service.complete_enrollment(request.session_id)

        if not result.get("success"):
            return CompleteEnrollmentResponse(
                success=False,
                error=result.get("error", "Unknown error"),
            )

        return CompleteEnrollmentResponse(
            success=True,
            user_id=result["user_id"],
            user_name=result["user_name"],
            samples_used=result["samples_used"],
        )
    except Exception as e:
        logger.error("Failed to complete enrollment: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enroll/cancel")
async def cancel_enrollment(session_id: str = Query(...)):
    """Cancel an in-progress enrollment session."""
    _check_enabled()

    from ..services.speaker_id import get_speaker_id_service

    service = get_speaker_id_service()
    result = service.cancel_enrollment(session_id)

    return result


@router.post("/verify", response_model=VerifyResponse)
async def verify_speaker(request: VerifyRequest):
    """
    Verify speaker identity from audio.

    Audio should be base64-encoded PCM (int16, mono).
    Returns match result and confidence score.
    """
    _check_enabled()
    _check_db_enabled()

    from ..services.speaker_id import get_speaker_id_service

    try:
        service = get_speaker_id_service()

        # Decode audio
        audio_bytes = base64.b64decode(request.audio_base64)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio.astype(np.float32) / 32768.0

        # Identify speaker
        result = await service.identify_speaker(audio_float, request.sample_rate)

        return VerifyResponse(
            matched=result.matched,
            user_id=str(result.user_id) if result.user_id else None,
            user_name=result.user_name,
            confidence=result.confidence,
            is_known=result.is_known,
            threshold=service.threshold,
        )
    except Exception as e:
        logger.error("Failed to verify speaker: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enrolled")
async def list_enrolled_users():
    """List all users with voice enrollment."""
    _check_enabled()
    _check_db_enabled()

    from ..storage.repositories.speaker import get_speaker_repo

    try:
        repo = get_speaker_repo()
        users = await repo.get_enrolled_users()

        return {
            "count": len(users),
            "users": [
                {
                    "id": u["id"],
                    "name": u["name"],
                    "created_at": u["created_at"].isoformat(),
                    "updated_at": u["updated_at"].isoformat(),
                }
                for u in users
            ],
        }
    except Exception as e:
        logger.error("Failed to list enrolled users: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{user_id}")
async def delete_enrollment(user_id: str):
    """Delete a user's voice enrollment."""
    _check_enabled()
    _check_db_enabled()

    from ..services.speaker_id import get_speaker_id_service

    try:
        service = get_speaker_id_service()
        user_uuid = UUID(user_id)
        success = await service.delete_enrollment(user_uuid)

        return {
            "success": success,
            "user_id": user_id,
            "message": "Enrollment deleted" if success else "Delete failed",
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    except Exception as e:
        logger.error("Failed to delete enrollment: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
