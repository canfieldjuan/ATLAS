"""
Person recognition API endpoints.

Provides face and gait enrollment/recognition.
"""

import logging
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..core.config import settings
from ..devices.registry import device_registry
from ..storage import db_settings
from ..storage.database import get_db_pool
from ..recognition import (
    get_face_service,
    get_gait_service,
    get_person_repository,
)

logger = logging.getLogger("atlas.vision.api.recognition")

router = APIRouter()


class CreatePersonRequest(BaseModel):
    name: str
    is_known: bool = True
    metadata: Optional[dict[str, Any]] = None


class UpdatePersonRequest(BaseModel):
    name: Optional[str] = None
    is_known: Optional[bool] = None
    metadata: Optional[dict[str, Any]] = None


class PersonResponse(BaseModel):
    id: str
    name: str
    is_known: bool
    auto_created: bool
    created_at: str
    last_seen_at: Optional[str]


class EnrollFaceRequest(BaseModel):
    person_id: str
    camera_id: str = "webcam_office"
    source: str = "enrollment"


class IdentifyRequest(BaseModel):
    camera_id: str = "webcam_office"
    threshold: Optional[float] = None
    auto_enroll_unknown: Optional[bool] = None


class StartGaitEnrollRequest(BaseModel):
    person_id: str
    camera_id: str = "webcam_office"


class GaitIdentifyRequest(BaseModel):
    camera_id: str = "webcam_office"
    threshold: Optional[float] = None


class CombinedIdentifyRequest(BaseModel):
    camera_id: str = "webcam_office"
    face_threshold: Optional[float] = None
    gait_threshold: Optional[float] = None


def _check_db_enabled():
    """Check if database is enabled."""
    if not db_settings.enabled:
        raise HTTPException(status_code=503, detail="Database disabled")
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not initialized")


def _check_recognition_enabled():
    """Check if recognition is enabled."""
    if not settings.recognition.enabled:
        raise HTTPException(status_code=503, detail="Recognition disabled")


async def _get_camera_frame(camera_id: str):
    """Get a frame from the specified camera."""
    camera = device_registry.get(camera_id)
    if not camera:
        raise HTTPException(
            status_code=404,
            detail=f"Camera {camera_id} not found"
        )
    frame = await camera.get_frame()
    if frame is None:
        raise HTTPException(
            status_code=503,
            detail=f"Camera {camera_id} not ready or no frame available"
        )
    return frame


@router.post("/persons")
async def create_person(request: CreatePersonRequest) -> PersonResponse:
    """Create a new person for enrollment."""
    _check_db_enabled()

    try:
        repo = get_person_repository()
        person_id = await repo.create_person(
            name=request.name,
            is_known=request.is_known,
            auto_created=False,
            metadata=request.metadata,
        )
        person = await repo.get_person(person_id)
        return PersonResponse(
            id=str(person["id"]),
            name=person["name"],
            is_known=person["is_known"],
            auto_created=person["auto_created"],
            created_at=person["created_at"].isoformat(),
            last_seen_at=person["last_seen_at"].isoformat() if person["last_seen_at"] else None,
        )
    except Exception as e:
        logger.error("Failed to create person: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/persons")
async def list_persons(include_unknown: bool = Query(default=True)):
    """List all registered persons."""
    _check_db_enabled()

    try:
        repo = get_person_repository()
        persons = await repo.list_persons(include_unknown=include_unknown)
        return {
            "count": len(persons),
            "persons": [
                {
                    "id": str(p["id"]),
                    "name": p["name"],
                    "is_known": p["is_known"],
                    "auto_created": p["auto_created"],
                    "created_at": p["created_at"].isoformat(),
                    "last_seen_at": p["last_seen_at"].isoformat() if p["last_seen_at"] else None,
                }
                for p in persons
            ],
        }
    except Exception as e:
        logger.error("Failed to list persons: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/persons/{person_id}")
async def get_person(person_id: str) -> PersonResponse:
    """Get person details by ID."""
    _check_db_enabled()

    try:
        repo = get_person_repository()
        person_uuid = UUID(person_id)
        person = await repo.get_person(person_uuid)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        return PersonResponse(
            id=str(person["id"]),
            name=person["name"],
            is_known=person["is_known"],
            auto_created=person["auto_created"],
            created_at=person["created_at"].isoformat(),
            last_seen_at=person["last_seen_at"].isoformat() if person["last_seen_at"] else None,
        )
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid person ID")
    except Exception as e:
        logger.error("Failed to get person: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/persons/{person_id}")
async def delete_person(person_id: str):
    """Delete a person and all their embeddings."""
    _check_db_enabled()

    try:
        repo = get_person_repository()
        person_uuid = UUID(person_id)
        person = await repo.get_person(person_uuid)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        deleted = await repo.delete_person(person_uuid)
        return {
            "success": deleted,
            "message": f"Deleted person {person['name']}" if deleted else "Delete failed",
        }
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid person ID")
    except Exception as e:
        logger.error("Failed to delete person: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/persons/{person_id}")
async def update_person(person_id: str, request: UpdatePersonRequest):
    """Update a person's details."""
    _check_db_enabled()

    try:
        repo = get_person_repository()
        person_uuid = UUID(person_id)
        person = await repo.get_person(person_uuid)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        updated = await repo.update_person(
            person_id=person_uuid,
            name=request.name,
            is_known=request.is_known,
            metadata=request.metadata,
        )
        if updated:
            person = await repo.get_person(person_uuid)
            return PersonResponse(
                id=str(person["id"]),
                name=person["name"],
                is_known=person["is_known"],
                auto_created=person["auto_created"],
                created_at=person["created_at"].isoformat(),
                last_seen_at=person["last_seen_at"].isoformat() if person["last_seen_at"] else None,
            )
        return {"success": False, "message": "No changes made"}
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid person ID")
    except Exception as e:
        logger.error("Failed to update person: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/persons/{person_id}/embeddings")
async def get_person_embeddings(person_id: str):
    """Get embedding counts for a person."""
    _check_db_enabled()

    try:
        repo = get_person_repository()
        person_uuid = UUID(person_id)
        person = await repo.get_person(person_uuid)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        counts = await repo.get_person_embedding_counts(person_uuid)
        return {
            "person_id": person_id,
            "person_name": person["name"],
            "face_embeddings": counts["face_embeddings"],
            "gait_embeddings": counts["gait_embeddings"],
        }
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid person ID")
    except Exception as e:
        logger.error("Failed to get embeddings: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events")
async def get_recognition_events(
    person_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Get recent recognition events."""
    _check_db_enabled()

    try:
        repo = get_person_repository()
        person_uuid = UUID(person_id) if person_id else None
        events = await repo.get_recent_recognition_events(
            person_id=person_uuid,
            limit=limit,
        )
        return {
            "count": len(events),
            "events": [
                {
                    "id": str(e["id"]),
                    "person_id": str(e["person_id"]) if e["person_id"] else None,
                    "person_name": e.get("person_name"),
                    "recognition_type": e["recognition_type"],
                    "confidence": e["confidence"],
                    "matched": e["matched"],
                    "camera_source": e.get("camera_source"),
                    "created_at": e["created_at"].isoformat(),
                }
                for e in events
            ],
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid person ID")
    except Exception as e:
        logger.error("Failed to get events: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Face Enrollment and Identification
# =============================================================================


@router.post("/enroll/face")
async def enroll_face(request: EnrollFaceRequest):
    """Enroll a face for a person using camera capture."""
    _check_db_enabled()
    _check_recognition_enabled()

    try:
        person_uuid = UUID(request.person_id)
        repo = get_person_repository()
        person = await repo.get_person(person_uuid)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        frame = await _get_camera_frame(request.camera_id)
        face_service = get_face_service()
        embedding_id = await face_service.enroll_face(
            frame=frame,
            person_id=person_uuid,
            source=request.source,
            save_image=True,
        )
        if not embedding_id:
            raise HTTPException(status_code=400, detail="No face detected in frame")

        return {
            "success": True,
            "person_id": request.person_id,
            "person_name": person["name"],
            "embedding_id": str(embedding_id),
            "camera_id": request.camera_id,
        }
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid person ID")
    except Exception as e:
        logger.error("Failed to enroll face: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/identify/face")
async def identify_face(request: IdentifyRequest):
    """Identify a person by their face using camera capture."""
    _check_db_enabled()
    _check_recognition_enabled()

    try:
        frame = await _get_camera_frame(request.camera_id)
        face_service = get_face_service()

        threshold = request.threshold
        if threshold is None:
            threshold = settings.recognition.face_threshold

        auto_enroll = request.auto_enroll_unknown
        if auto_enroll is None:
            auto_enroll = settings.recognition.auto_enroll_unknown

        result = await face_service.recognize_face(
            frame=frame,
            threshold=threshold,
            camera_source=request.camera_id,
            auto_enroll_unknown=auto_enroll,
            use_averaged=settings.recognition.use_averaged,
        )
        if not result:
            return {"matched": False, "message": "No face detected"}

        return {
            "matched": result.get("matched", False),
            "person_id": str(result["person_id"]) if result.get("person_id") else None,
            "person_name": result.get("name"),
            "similarity": result.get("similarity"),
            "is_known": result.get("is_known"),
            "auto_enrolled": result.get("auto_enrolled", False),
            "camera_id": request.camera_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to identify face: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Gait Enrollment and Identification
# =============================================================================

# Module-level state for gait enrollment session
_gait_enrollment_state: dict = {
    "active": False,
    "person_id": None,
    "camera_id": None,
}


@router.post("/enroll/gait/start")
async def start_gait_enrollment(request: StartGaitEnrollRequest):
    """Start gait enrollment for a person."""
    _check_db_enabled()
    _check_recognition_enabled()

    try:
        person_uuid = UUID(request.person_id)
        repo = get_person_repository()
        person = await repo.get_person(person_uuid)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        gait_service = get_gait_service()
        gait_service.clear_buffer()

        _gait_enrollment_state["active"] = True
        _gait_enrollment_state["person_id"] = request.person_id
        _gait_enrollment_state["camera_id"] = request.camera_id

        return {
            "success": True,
            "message": "Gait enrollment started",
            "person_id": request.person_id,
            "person_name": person["name"],
            "camera_id": request.camera_id,
            "frames_needed": settings.recognition.gait_sequence_length,
        }
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid person ID")
    except Exception as e:
        logger.error("Failed to start gait enrollment: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enroll/gait/frame")
async def add_gait_frame(camera_id: str = Query(default="webcam_office")):
    """Capture a frame and add pose to gait buffer."""
    _check_recognition_enabled()

    if not _gait_enrollment_state["active"]:
        raise HTTPException(status_code=400, detail="No gait enrollment in progress")

    try:
        frame = await _get_camera_frame(camera_id)
        gait_service = get_gait_service()
        pose = gait_service.extract_pose(frame)
        if pose is None:
            return {
                "success": False,
                "message": "No pose detected in frame",
                "buffer_length": len(gait_service._pose_buffer),
            }

        buffer_length = gait_service.add_pose_to_buffer(pose)
        is_full = gait_service.is_buffer_full()

        return {
            "success": True,
            "buffer_length": buffer_length,
            "frames_needed": gait_service.sequence_length,
            "is_full": is_full,
            "progress": buffer_length / gait_service.sequence_length,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add gait frame: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enroll/gait/complete")
async def complete_gait_enrollment(
    walking_direction: Optional[str] = Query(default=None),
):
    """Complete gait enrollment using collected frames."""
    _check_db_enabled()
    _check_recognition_enabled()

    if not _gait_enrollment_state["active"]:
        raise HTTPException(status_code=400, detail="No gait enrollment in progress")

    try:
        person_id = _gait_enrollment_state["person_id"]
        person_uuid = UUID(person_id)

        gait_service = get_gait_service()
        if not gait_service.is_buffer_full():
            return {
                "success": False,
                "message": "Buffer not full",
                "buffer_length": len(gait_service._pose_buffer),
                "frames_needed": gait_service.sequence_length,
            }

        embedding_id = await gait_service.enroll_gait(
            person_id=person_uuid,
            walking_direction=walking_direction,
            source="enrollment",
        )

        _gait_enrollment_state["active"] = False
        _gait_enrollment_state["person_id"] = None
        _gait_enrollment_state["camera_id"] = None

        if not embedding_id:
            return {"success": False, "message": "Failed to compute gait embedding"}

        return {
            "success": True,
            "person_id": person_id,
            "embedding_id": str(embedding_id),
            "walking_direction": walking_direction,
        }
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid person ID")
    except Exception as e:
        logger.error("Failed to complete gait enrollment: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enroll/gait/status")
async def get_gait_enrollment_status():
    """Get current gait enrollment status."""
    gait_service = get_gait_service()
    buffer_length = len(gait_service._pose_buffer)
    sequence_length = gait_service.sequence_length

    return {
        "active": _gait_enrollment_state["active"],
        "person_id": _gait_enrollment_state["person_id"],
        "camera_id": _gait_enrollment_state["camera_id"],
        "buffer_length": buffer_length,
        "frames_needed": sequence_length,
        "is_full": buffer_length >= sequence_length,
        "progress": buffer_length / sequence_length if sequence_length > 0 else 0.0,
    }


@router.post("/enroll/gait/cancel")
async def cancel_gait_enrollment():
    """Cancel ongoing gait enrollment."""
    gait_service = get_gait_service()
    gait_service.clear_buffer()

    was_active = _gait_enrollment_state["active"]
    _gait_enrollment_state["active"] = False
    _gait_enrollment_state["person_id"] = None
    _gait_enrollment_state["camera_id"] = None

    return {
        "success": True,
        "was_active": was_active,
        "message": "Gait enrollment cancelled" if was_active else "No enrollment was active",
    }


@router.post("/identify/gait/start")
async def start_gait_identification():
    """Start gait identification (clears buffer)."""
    _check_recognition_enabled()

    gait_service = get_gait_service()
    gait_service.clear_buffer()

    return {
        "success": True,
        "message": "Gait identification started",
        "frames_needed": gait_service.sequence_length,
    }


@router.post("/identify/gait/frame")
async def add_gait_identify_frame(camera_id: str = Query(default="webcam_office")):
    """Add a frame for gait identification."""
    _check_recognition_enabled()

    try:
        frame = await _get_camera_frame(camera_id)
        gait_service = get_gait_service()
        pose = gait_service.extract_pose(frame)
        if pose is None:
            return {
                "success": False,
                "message": "No pose detected in frame",
                "buffer_length": len(gait_service._pose_buffer),
            }

        buffer_length = gait_service.add_pose_to_buffer(pose)
        is_full = gait_service.is_buffer_full()

        return {
            "success": True,
            "buffer_length": buffer_length,
            "frames_needed": gait_service.sequence_length,
            "is_full": is_full,
            "progress": buffer_length / gait_service.sequence_length,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add gait identify frame: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/identify/gait/match")
async def match_gait(request: GaitIdentifyRequest):
    """Match collected gait sequence against enrolled gaits."""
    _check_db_enabled()
    _check_recognition_enabled()

    try:
        gait_service = get_gait_service()
        if not gait_service.is_buffer_full():
            return {
                "matched": False,
                "message": "Buffer not full",
                "buffer_length": len(gait_service._pose_buffer),
                "frames_needed": gait_service.sequence_length,
            }

        threshold = request.threshold
        if threshold is None:
            threshold = settings.recognition.gait_threshold

        result = await gait_service.recognize_gait(
            threshold=threshold,
            camera_source=request.camera_id,
            use_averaged=settings.recognition.use_averaged,
            auto_enroll_unknown=False,
        )
        if not result:
            return {"matched": False, "message": "No matching gait found"}

        return {
            "matched": result.get("matched", False),
            "person_id": str(result["person_id"]) if result.get("person_id") else None,
            "person_name": result.get("name"),
            "similarity": result.get("similarity"),
            "is_known": result.get("is_known"),
            "camera_id": request.camera_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to match gait: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Combined Identification
# =============================================================================


@router.post("/identify/combined")
async def identify_combined(request: CombinedIdentifyRequest):
    """Combined face + gait identification."""
    _check_db_enabled()
    _check_recognition_enabled()

    try:
        frame = await _get_camera_frame(request.camera_id)

        face_threshold = request.face_threshold
        if face_threshold is None:
            face_threshold = settings.recognition.face_threshold

        gait_threshold = request.gait_threshold
        if gait_threshold is None:
            gait_threshold = settings.recognition.gait_threshold

        face_service = get_face_service()
        face_result = await face_service.recognize_face(
            frame=frame,
            threshold=face_threshold,
            camera_source=request.camera_id,
            auto_enroll_unknown=False,
            use_averaged=settings.recognition.use_averaged,
        )

        gait_service = get_gait_service()
        gait_result = None
        if gait_service.is_buffer_full():
            gait_result = await gait_service.recognize_gait(
                threshold=gait_threshold,
                camera_source=request.camera_id,
                use_averaged=settings.recognition.use_averaged,
                auto_enroll_unknown=False,
            )

        face_matched = face_result is not None and face_result.get("matched", False)
        gait_matched = gait_result is not None and gait_result.get("matched", False)

        if face_matched and gait_matched:
            if face_result["person_id"] == gait_result["person_id"]:
                combined_similarity = (
                    face_result["similarity"] + gait_result["similarity"]
                ) / 2
                return {
                    "matched": True,
                    "method": "face+gait",
                    "person_id": str(face_result["person_id"]),
                    "person_name": face_result.get("name"),
                    "face_similarity": face_result["similarity"],
                    "gait_similarity": gait_result["similarity"],
                    "combined_similarity": combined_similarity,
                    "is_known": face_result.get("is_known"),
                }

        if face_matched:
            return {
                "matched": True,
                "method": "face",
                "person_id": str(face_result["person_id"]),
                "person_name": face_result.get("name"),
                "face_similarity": face_result["similarity"],
                "gait_similarity": None,
                "combined_similarity": face_result["similarity"],
                "is_known": face_result.get("is_known"),
            }

        if gait_matched:
            return {
                "matched": True,
                "method": "gait",
                "person_id": str(gait_result["person_id"]),
                "person_name": gait_result.get("name"),
                "face_similarity": None,
                "gait_similarity": gait_result["similarity"],
                "combined_similarity": gait_result["similarity"],
                "is_known": gait_result.get("is_known"),
            }

        return {
            "matched": False,
            "method": None,
            "message": "No match found",
            "face_detected": face_result is not None,
            "gait_buffer_full": gait_service.is_buffer_full(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed combined identification: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
