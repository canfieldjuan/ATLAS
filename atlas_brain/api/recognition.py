"""
Person recognition API endpoints.

Proxies requests to atlas_vision service for recognition operations.
"""

import logging
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..config import settings

logger = logging.getLogger("atlas.api.recognition")

router = APIRouter(prefix="/recognition", tags=["recognition"])


def _get_vision_url() -> str:
    """Get atlas_vision service URL."""
    return settings.security.video_processing_url


async def _proxy_request(
    method: str,
    path: str,
    json_data: Optional[dict] = None,
    params: Optional[dict] = None,
) -> dict:
    """
    Proxy a request to atlas_vision service.

    Args:
        method: HTTP method (GET, POST, DELETE, etc.)
        path: API path (e.g., /recognition/persons)
        json_data: JSON body for POST/PUT requests
        params: Query parameters

    Returns:
        Response JSON as dict
    """
    base_url = _get_vision_url()
    url = f"{base_url}{path}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method=method,
                url=url,
                json=json_data,
                params=params,
            )

            if response.status_code >= 400:
                detail = response.text
                try:
                    error_json = response.json()
                    detail = error_json.get("detail", response.text)
                except Exception:
                    pass
                raise HTTPException(
                    status_code=response.status_code,
                    detail=detail,
                )

            return response.json()

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to atlas_vision at {base_url}",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Request to atlas_vision timed out",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Proxy error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class CreatePersonRequest(BaseModel):
    name: str
    is_known: bool = True
    metadata: Optional[dict[str, Any]] = None


class PersonResponse(BaseModel):
    id: str
    name: str
    is_known: bool
    auto_created: bool
    created_at: str
    last_seen_at: Optional[str]


class UpdatePersonRequest(BaseModel):
    name: Optional[str] = None
    is_known: Optional[bool] = None
    metadata: Optional[dict[str, Any]] = None


@router.post("/persons")
async def create_person(request: CreatePersonRequest):
    """Create a new person for enrollment."""
    return await _proxy_request(
        "POST",
        "/recognition/persons",
        json_data=request.model_dump(),
    )


@router.get("/persons")
async def list_persons(include_unknown: bool = Query(default=True)):
    """List all registered persons."""
    return await _proxy_request(
        "GET",
        "/recognition/persons",
        params={"include_unknown": include_unknown},
    )


@router.get("/persons/{person_id}")
async def get_person(person_id: str):
    """Get person details by ID."""
    return await _proxy_request("GET", f"/recognition/persons/{person_id}")


@router.patch("/persons/{person_id}")
async def update_person(person_id: str, request: UpdatePersonRequest):
    """Update a person's details."""
    return await _proxy_request(
        "PATCH",
        f"/recognition/persons/{person_id}",
        json_data=request.model_dump(exclude_none=True),
    )


@router.delete("/persons/{person_id}")
async def delete_person(person_id: str):
    """Delete a person and all their embeddings."""
    return await _proxy_request("DELETE", f"/recognition/persons/{person_id}")


@router.get("/persons/{person_id}/embeddings")
async def get_person_embeddings(person_id: str):
    """Get embedding counts for a person."""
    return await _proxy_request("GET", f"/recognition/persons/{person_id}/embeddings")


@router.get("/events")
async def get_recognition_events(
    person_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Get recent recognition events."""
    params = {"limit": limit}
    if person_id:
        params["person_id"] = person_id
    return await _proxy_request("GET", "/recognition/events", params=params)


# Enrollment and identification endpoints
# These require camera access which is now handled by atlas_vision

class EnrollFaceRequest(BaseModel):
    person_id: str
    camera_id: str = "webcam_office"
    source: str = "enrollment"


class IdentifyRequest(BaseModel):
    camera_id: str = "webcam_office"
    threshold: Optional[float] = None
    auto_enroll_unknown: Optional[bool] = None


@router.post("/enroll/face")
async def enroll_face(request: EnrollFaceRequest):
    """
    Enroll a face for a person.

    Uses atlas_vision camera for frame capture.
    """
    return await _proxy_request(
        "POST",
        "/recognition/enroll/face",
        json_data=request.model_dump(),
    )


@router.post("/identify/face")
async def identify_face(request: IdentifyRequest):
    """
    Identify a person by their face.

    Uses atlas_vision camera for frame capture.
    """
    return await _proxy_request(
        "POST",
        "/recognition/identify/face",
        json_data=request.model_dump(),
    )


class StartGaitEnrollRequest(BaseModel):
    person_id: str
    camera_id: str = "webcam_office"


@router.post("/enroll/gait/start")
async def start_gait_enrollment(request: StartGaitEnrollRequest):
    """Start gait enrollment for a person."""
    return await _proxy_request(
        "POST",
        "/recognition/enroll/gait/start",
        json_data=request.model_dump(),
    )


@router.post("/enroll/gait/frame")
async def add_gait_frame(camera_id: str = Query(default="webcam_office")):
    """Capture a frame and add pose to gait buffer."""
    return await _proxy_request(
        "POST",
        "/recognition/enroll/gait/frame",
        params={"camera_id": camera_id},
    )


@router.post("/enroll/gait/complete")
async def complete_gait_enrollment(
    walking_direction: Optional[str] = Query(default=None),
):
    """Complete gait enrollment using collected frames."""
    params = {}
    if walking_direction:
        params["walking_direction"] = walking_direction
    return await _proxy_request(
        "POST",
        "/recognition/enroll/gait/complete",
        params=params if params else None,
    )


@router.get("/enroll/gait/status")
async def get_gait_enrollment_status():
    """Get current gait enrollment status."""
    return await _proxy_request("GET", "/recognition/enroll/gait/status")


@router.post("/enroll/gait/cancel")
async def cancel_gait_enrollment():
    """Cancel ongoing gait enrollment."""
    return await _proxy_request("POST", "/recognition/enroll/gait/cancel")


class GaitIdentifyRequest(BaseModel):
    camera_id: str = "webcam_office"
    threshold: Optional[float] = None


@router.post("/identify/gait/start")
async def start_gait_identification():
    """Start gait identification (clears buffer)."""
    return await _proxy_request("POST", "/recognition/identify/gait/start")


@router.post("/identify/gait/frame")
async def add_gait_identify_frame(camera_id: str = Query(default="webcam_office")):
    """Add a frame for gait identification."""
    return await _proxy_request(
        "POST",
        "/recognition/identify/gait/frame",
        params={"camera_id": camera_id},
    )


@router.post("/identify/gait/match")
async def match_gait(request: GaitIdentifyRequest):
    """Match collected gait sequence against enrolled gaits."""
    return await _proxy_request(
        "POST",
        "/recognition/identify/gait/match",
        json_data=request.model_dump(),
    )


class CombinedIdentifyRequest(BaseModel):
    camera_id: str = "webcam_office"
    face_threshold: Optional[float] = None
    gait_threshold: Optional[float] = None


@router.post("/identify/combined")
async def identify_combined(request: CombinedIdentifyRequest):
    """Combined face + gait identification."""
    return await _proxy_request(
        "POST",
        "/recognition/identify/combined",
        json_data=request.model_dump(),
    )
