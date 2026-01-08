"""
Speaker identification API endpoints.

Provides REST API for:
- Enrolling speakers (registering known voices)
- Identifying speakers (who is speaking?)
- Managing enrolled speakers
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ..services import speaker_id_registry

router = APIRouter(prefix="/speaker-id", tags=["speaker-id"])


class ActivateRequest(BaseModel):
    name: str = "resemblyzer"


class IdentifyRequest(BaseModel):
    threshold: float = 0.75


class SpeakerResponse(BaseModel):
    name: str
    enrolled_at: float
    sample_count: int


class IdentifyResponse(BaseModel):
    name: str
    confidence: float
    is_known: bool


class EnrollResponse(BaseModel):
    success: bool
    speaker: SpeakerResponse


class ListResponse(BaseModel):
    speakers: list[SpeakerResponse]
    count: int


@router.get("/available")
async def list_available():
    """List available speaker ID implementations."""
    return {
        "available": speaker_id_registry.list_available(),
        "active": speaker_id_registry.get_active_name(),
    }


@router.post("/activate")
async def activate_speaker_id(request: ActivateRequest):
    """Activate a speaker ID implementation."""
    try:
        service = speaker_id_registry.activate(request.name)
        return {
            "success": True,
            "message": f"Activated speaker ID: {request.name}",
            "model_info": service.model_info.to_dict(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate: {e}")


@router.post("/deactivate")
async def deactivate_speaker_id():
    """Deactivate speaker ID to free resources."""
    speaker_id_registry.deactivate()
    return {"success": True, "message": "Speaker ID deactivated"}


@router.post("/enroll")
async def enroll_speaker(
    audio: UploadFile = File(...),
    name: str = Form(...),
    merge_existing: bool = Form(True),
) -> EnrollResponse:
    """
    Enroll a speaker with a voice sample.

    For best results, use 5-10 seconds of clear speech.
    Multiple enrollments are averaged for better accuracy.
    """
    service = speaker_id_registry.get_active()
    if service is None:
        raise HTTPException(
            status_code=400,
            detail="No speaker ID service active. Call /speaker-id/activate first.",
        )

    try:
        audio_bytes = await audio.read()
        speaker_info = service.enroll(name, audio_bytes, merge_existing)

        return EnrollResponse(
            success=True,
            speaker=SpeakerResponse(
                name=speaker_info.name,
                enrolled_at=speaker_info.enrolled_at,
                sample_count=speaker_info.sample_count,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {e}")


@router.post("/identify")
async def identify_speaker(
    audio: UploadFile = File(...),
    threshold: float = Form(0.75),
) -> IdentifyResponse:
    """
    Identify who is speaking in an audio sample.

    Returns the best matching enrolled speaker if above threshold,
    otherwise returns "unknown".
    """
    service = speaker_id_registry.get_active()
    if service is None:
        raise HTTPException(
            status_code=400,
            detail="No speaker ID service active. Call /speaker-id/activate first.",
        )

    try:
        audio_bytes = await audio.read()
        match = service.identify(audio_bytes, threshold)

        return IdentifyResponse(
            name=match.name,
            confidence=match.confidence,
            is_known=match.is_known,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identification failed: {e}")


@router.post("/verify")
async def verify_speaker(
    audio: UploadFile = File(...),
    claimed_name: str = Form(...),
    threshold: float = Form(0.80),
):
    """
    Verify if the speaker matches a claimed identity.

    Use for authentication: "Is this person really Juan?"
    """
    service = speaker_id_registry.get_active()
    if service is None:
        raise HTTPException(
            status_code=400,
            detail="No speaker ID service active. Call /speaker-id/activate first.",
        )

    try:
        audio_bytes = await audio.read()
        is_verified, similarity = service.verify(audio_bytes, claimed_name, threshold)

        return {
            "verified": is_verified,
            "claimed_name": claimed_name,
            "similarity": similarity,
            "threshold": threshold,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {e}")


@router.get("/speakers")
async def list_enrolled_speakers() -> ListResponse:
    """List all enrolled speakers."""
    service = speaker_id_registry.get_active()
    if service is None:
        raise HTTPException(
            status_code=400,
            detail="No speaker ID service active. Call /speaker-id/activate first.",
        )

    speakers = service.list_enrolled()
    return ListResponse(
        speakers=[
            SpeakerResponse(
                name=s.name,
                enrolled_at=s.enrolled_at,
                sample_count=s.sample_count,
            )
            for s in speakers
        ],
        count=len(speakers),
    )


@router.delete("/speakers/{name}")
async def remove_speaker(name: str):
    """Remove an enrolled speaker."""
    service = speaker_id_registry.get_active()
    if service is None:
        raise HTTPException(
            status_code=400,
            detail="No speaker ID service active. Call /speaker-id/activate first.",
        )

    if service.remove_speaker(name):
        return {"success": True, "message": f"Removed speaker: {name}"}
    else:
        raise HTTPException(status_code=404, detail=f"Speaker not found: {name}")
