"""
Person recognition API endpoints.

Provides face and gait enrollment/recognition via webcam.
"""

import logging
from typing import Any, Optional
from uuid import UUID

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..config import settings
from ..storage import db_settings
from ..storage.database import get_db_pool
from ..services.recognition import (
    get_face_service,
    get_gait_service,
    get_person_repository,
)

logger = logging.getLogger("atlas.api.recognition")

router = APIRouter(prefix="/recognition", tags=["recognition"])


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


class EnrollFaceRequest(BaseModel):
    person_id: str
    device: int = 0
    source: str = "enrollment"


class IdentifyRequest(BaseModel):
    device: int = 0
    threshold: Optional[float] = None
    auto_enroll_unknown: Optional[bool] = None
    use_averaged: Optional[bool] = None
    camera_source: Optional[str] = None


def _check_db_enabled():
    """Check if database is enabled and initialized."""
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


def _capture_frame(device: int = 0) -> np.ndarray:
    """Capture a single frame from webcam."""
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise HTTPException(status_code=503, detail=f"Cannot open webcam {device}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Warm up camera - discard initial frames
    for _ in range(30):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise HTTPException(status_code=503, detail="Failed to capture frame")

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
        raise HTTPException(status_code=400, detail="Invalid person ID format")
    except Exception as e:
        logger.error("Failed to get person: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class UpdatePersonRequest(BaseModel):
    name: Optional[str] = None
    is_known: Optional[bool] = None
    metadata: Optional[dict[str, Any]] = None


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
            return {
                "success": True,
                "person": {
                    "id": str(person["id"]),
                    "name": person["name"],
                    "is_known": person["is_known"],
                },
            }
        return {"success": False, "message": "No changes made"}
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid person ID format")
    except Exception as e:
        logger.error("Failed to update person: %s", e)
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
        raise HTTPException(status_code=400, detail="Invalid person ID format")
    except Exception as e:
        logger.error("Failed to delete person: %s", e)
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
        raise HTTPException(status_code=400, detail="Invalid person ID format")
    except Exception as e:
        logger.error("Failed to get embedding counts: %s", e)
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
        raise HTTPException(status_code=400, detail="Invalid person ID format")
    except Exception as e:
        logger.error("Failed to get recognition events: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enroll/face")
async def enroll_face(request: EnrollFaceRequest):
    """
    Enroll a face for a person using webcam capture.

    Captures frame from webcam, detects face, extracts embedding,
    and stores it for future recognition.
    """
    _check_db_enabled()

    try:
        repo = get_person_repository()
        face_service = get_face_service()

        person_uuid = UUID(request.person_id)
        person = await repo.get_person(person_uuid)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        frame = _capture_frame(request.device)
        embedding_id = await face_service.enroll_face(
            frame=frame,
            person_id=person_uuid,
            source=request.source,
            save_image=True,
        )

        if embedding_id is None:
            raise HTTPException(
                status_code=400,
                detail="No face detected in frame. Please ensure face is visible."
            )

        return {
            "success": True,
            "person_id": str(person_uuid),
            "person_name": person["name"],
            "embedding_id": str(embedding_id),
            "message": f"Face enrolled for {person['name']}",
        }
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid person ID format")
    except Exception as e:
        logger.error("Failed to enroll face: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/identify/face")
async def identify_face(request: IdentifyRequest):
    """
    Identify a person by their face using webcam capture.

    Captures frame, detects face, matches against enrolled faces.
    If auto_enroll_unknown is True, creates profile for unknown faces.
    """
    _check_db_enabled()

    # Use config defaults when not specified
    cfg = settings.recognition
    threshold = request.threshold if request.threshold is not None else cfg.face_threshold
    auto_enroll = request.auto_enroll_unknown if request.auto_enroll_unknown is not None else cfg.auto_enroll_unknown
    use_averaged = request.use_averaged if request.use_averaged is not None else cfg.use_averaged

    try:
        face_service = get_face_service()
        frame = _capture_frame(request.device)

        result = await face_service.recognize_face(
            frame=frame,
            threshold=threshold,
            camera_source=request.camera_source,
            auto_enroll_unknown=auto_enroll,
            use_averaged=use_averaged,
        )

        if result is None:
            return {
                "matched": False,
                "message": "No face detected in frame",
            }

        return {
            "matched": result["matched"],
            "person_id": str(result["person_id"]),
            "person_name": result["name"],
            "similarity": result["similarity"],
            "is_known": result["is_known"],
            "auto_enrolled": result.get("auto_enrolled", False),
        }
    except Exception as e:
        logger.error("Failed to identify face: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# Gait enrollment state (person_id being enrolled)
_gait_enrollment_person_id: Optional[UUID] = None


class StartGaitEnrollRequest(BaseModel):
    person_id: str
    device: int = 0
    walking_direction: Optional[str] = None


@router.post("/enroll/gait/start")
async def start_gait_enrollment(request: StartGaitEnrollRequest):
    """
    Start gait enrollment for a person.

    Clears the pose buffer and sets up for collecting frames.
    Call /enroll/gait/frame repeatedly while person walks.
    Call /enroll/gait/complete when buffer is full.
    """
    global _gait_enrollment_person_id
    _check_db_enabled()

    try:
        repo = get_person_repository()
        gait_service = get_gait_service()

        person_uuid = UUID(request.person_id)
        person = await repo.get_person(person_uuid)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        gait_service.clear_buffer()
        _gait_enrollment_person_id = person_uuid

        return {
            "success": True,
            "person_id": str(person_uuid),
            "person_name": person["name"],
            "buffer_size": 0,
            "required_frames": gait_service.sequence_length,
            "message": "Gait enrollment started. Call /enroll/gait/frame to add frames.",
        }
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid person ID format")
    except Exception as e:
        logger.error("Failed to start gait enrollment: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enroll/gait/frame")
async def add_gait_frame(device: int = Query(default=0)):
    """
    Capture a frame and add pose to gait buffer.

    Call repeatedly while person walks across camera view.
    Returns current buffer status.
    """
    if _gait_enrollment_person_id is None:
        raise HTTPException(
            status_code=400,
            detail="No gait enrollment in progress. Call /enroll/gait/start first."
        )

    try:
        gait_service = get_gait_service()
        frame = _capture_frame(device)

        pose = gait_service.extract_pose(frame)
        if pose is None:
            return {
                "success": False,
                "pose_detected": False,
                "buffer_size": len(gait_service._pose_buffer),
                "required_frames": gait_service.sequence_length,
                "is_ready": gait_service.is_buffer_full(),
                "message": "No pose detected. Ensure full body is visible.",
            }

        buffer_size = gait_service.add_pose_to_buffer(pose)
        is_ready = gait_service.is_buffer_full()

        return {
            "success": True,
            "pose_detected": True,
            "buffer_size": buffer_size,
            "required_frames": gait_service.sequence_length,
            "is_ready": is_ready,
            "message": "Ready to complete enrollment" if is_ready else f"Need {gait_service.sequence_length - buffer_size} more frames",
        }
    except Exception as e:
        logger.error("Failed to add gait frame: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enroll/gait/complete")
async def complete_gait_enrollment(walking_direction: Optional[str] = Query(default=None)):
    """
    Complete gait enrollment using collected frames.

    Only call when buffer is full (is_ready=True from /enroll/gait/frame).
    """
    global _gait_enrollment_person_id
    _check_db_enabled()

    if _gait_enrollment_person_id is None:
        raise HTTPException(
            status_code=400,
            detail="No gait enrollment in progress. Call /enroll/gait/start first."
        )

    try:
        gait_service = get_gait_service()
        repo = get_person_repository()

        if not gait_service.is_buffer_full():
            return {
                "success": False,
                "buffer_size": len(gait_service._pose_buffer),
                "required_frames": gait_service.sequence_length,
                "message": f"Buffer not full. Need {gait_service.sequence_length - len(gait_service._pose_buffer)} more frames.",
            }

        person = await repo.get_person(_gait_enrollment_person_id)
        embedding_id = await gait_service.enroll_gait(
            person_id=_gait_enrollment_person_id,
            walking_direction=walking_direction,
            source="enrollment",
        )

        person_id = _gait_enrollment_person_id
        _gait_enrollment_person_id = None

        return {
            "success": True,
            "person_id": str(person_id),
            "person_name": person["name"] if person else "unknown",
            "embedding_id": str(embedding_id),
            "message": f"Gait enrolled for {person['name'] if person else 'unknown'}",
        }
    except Exception as e:
        logger.error("Failed to complete gait enrollment: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enroll/gait/status")
async def get_gait_enrollment_status():
    """Get current gait enrollment status."""
    gait_service = get_gait_service()

    return {
        "enrollment_active": _gait_enrollment_person_id is not None,
        "person_id": str(_gait_enrollment_person_id) if _gait_enrollment_person_id else None,
        "buffer_size": len(gait_service._pose_buffer),
        "required_frames": gait_service.sequence_length,
        "is_ready": gait_service.is_buffer_full(),
    }


@router.post("/enroll/gait/cancel")
async def cancel_gait_enrollment():
    """Cancel ongoing gait enrollment."""
    global _gait_enrollment_person_id

    gait_service = get_gait_service()
    gait_service.clear_buffer()
    _gait_enrollment_person_id = None

    return {
        "success": True,
        "message": "Gait enrollment cancelled",
    }


class GaitIdentifyRequest(BaseModel):
    device: int = 0
    threshold: Optional[float] = None
    use_averaged: Optional[bool] = None
    auto_enroll_unknown: Optional[bool] = None
    camera_source: Optional[str] = None


@router.post("/identify/gait/start")
async def start_gait_identification():
    """Start gait identification (clears buffer for new sequence)."""
    gait_service = get_gait_service()
    gait_service.clear_buffer()

    return {
        "success": True,
        "buffer_size": 0,
        "required_frames": gait_service.sequence_length,
        "message": "Gait identification started. Call /identify/gait/frame to add frames.",
    }


@router.post("/identify/gait/frame")
async def add_gait_identify_frame(device: int = Query(default=0)):
    """Add a frame for gait identification."""
    try:
        gait_service = get_gait_service()
        frame = _capture_frame(device)

        pose = gait_service.extract_pose(frame)
        if pose is None:
            return {
                "success": False,
                "pose_detected": False,
                "buffer_size": len(gait_service._pose_buffer),
                "required_frames": gait_service.sequence_length,
                "is_ready": gait_service.is_buffer_full(),
            }

        buffer_size = gait_service.add_pose_to_buffer(pose)

        return {
            "success": True,
            "pose_detected": True,
            "buffer_size": buffer_size,
            "required_frames": gait_service.sequence_length,
            "is_ready": gait_service.is_buffer_full(),
        }
    except Exception as e:
        logger.error("Failed to add gait identify frame: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/identify/gait/match")
async def match_gait(request: GaitIdentifyRequest):
    """
    Match collected gait sequence against enrolled gaits.

    Only call when buffer is full (is_ready=True from /identify/gait/frame).
    """
    _check_db_enabled()

    # Use config defaults when not specified
    cfg = settings.recognition
    threshold = request.threshold if request.threshold is not None else cfg.gait_threshold
    use_averaged = request.use_averaged if request.use_averaged is not None else cfg.use_averaged
    auto_enroll = request.auto_enroll_unknown if request.auto_enroll_unknown is not None else cfg.auto_enroll_unknown

    try:
        gait_service = get_gait_service()

        if not gait_service.is_buffer_full():
            return {
                "success": False,
                "matched": False,
                "buffer_size": len(gait_service._pose_buffer),
                "required_frames": gait_service.sequence_length,
                "message": f"Buffer not full. Need {gait_service.sequence_length - len(gait_service._pose_buffer)} more frames.",
            }

        result = await gait_service.recognize_gait(
            threshold=threshold,
            camera_source=request.camera_source,
            use_averaged=use_averaged,
            auto_enroll_unknown=auto_enroll,
        )

        if result is None:
            return {
                "success": True,
                "matched": False,
                "message": "No matching gait found",
            }

        return {
            "success": True,
            "matched": result["matched"],
            "person_id": str(result["person_id"]),
            "person_name": result["name"],
            "similarity": result["similarity"],
            "is_known": result["is_known"],
            "auto_enrolled": result.get("auto_enrolled", False),
        }
    except Exception as e:
        logger.error("Failed to match gait: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class CombinedIdentifyRequest(BaseModel):
    device: int = 0
    face_threshold: Optional[float] = None
    gait_threshold: Optional[float] = None
    use_averaged: Optional[bool] = None
    camera_source: Optional[str] = None


@router.post("/identify/combined")
async def identify_combined(request: CombinedIdentifyRequest):
    """
    Combined face + gait identification.

    Runs face recognition on current frame and gait recognition
    using buffered poses. Returns individual and combined scores.

    Call /identify/gait/start and /identify/gait/frame first to
    fill the gait buffer before calling this endpoint.
    """
    _check_db_enabled()

    cfg = settings.recognition
    face_threshold = request.face_threshold if request.face_threshold is not None else cfg.face_threshold
    gait_threshold = request.gait_threshold if request.gait_threshold is not None else cfg.gait_threshold
    use_averaged = request.use_averaged if request.use_averaged is not None else cfg.use_averaged

    try:
        face_service = get_face_service()
        gait_service = get_gait_service()

        # Face recognition
        frame = _capture_frame(request.device)
        face_result = await face_service.recognize_face(
            frame=frame,
            threshold=face_threshold,
            camera_source=request.camera_source,
            auto_enroll_unknown=False,
            use_averaged=use_averaged,
        )

        # Gait recognition (uses existing buffer)
        gait_result = None
        gait_buffer_status = {
            "buffer_size": len(gait_service._pose_buffer),
            "required_frames": gait_service.sequence_length,
            "is_ready": gait_service.is_buffer_full(),
        }

        if gait_service.is_buffer_full():
            gait_result = await gait_service.recognize_gait(
                threshold=gait_threshold,
                camera_source=request.camera_source,
                use_averaged=use_averaged,
                auto_enroll_unknown=False,
            )

        # Build response
        response = {
            "face": None,
            "gait": None,
            "gait_buffer": gait_buffer_status,
            "combined": None,
        }

        if face_result:
            response["face"] = {
                "matched": face_result["matched"],
                "person_id": str(face_result["person_id"]),
                "person_name": face_result["name"],
                "similarity": face_result["similarity"],
                "is_known": face_result["is_known"],
            }

        if gait_result:
            response["gait"] = {
                "matched": gait_result["matched"],
                "person_id": str(gait_result["person_id"]),
                "person_name": gait_result["name"],
                "similarity": gait_result["similarity"],
                "is_known": gait_result["is_known"],
            }

        # Combined score if both matched
        if face_result and gait_result:
            face_matched = face_result.get("matched", False)
            gait_matched = gait_result.get("matched", False)

            if face_matched and gait_matched:
                same_person = face_result["person_id"] == gait_result["person_id"]
                combined_similarity = (face_result["similarity"] + gait_result["similarity"]) / 2

                response["combined"] = {
                    "matched": same_person,
                    "person_id": str(face_result["person_id"]) if same_person else None,
                    "person_name": face_result["name"] if same_person else None,
                    "similarity": combined_similarity if same_person else 0.0,
                    "face_similarity": face_result["similarity"],
                    "gait_similarity": gait_result["similarity"],
                    "modalities_agree": same_person,
                    "confidence": "high" if same_person and combined_similarity > 0.8 else "medium" if same_person else "low",
                }

                if not same_person:
                    response["combined"]["warning"] = "Face and gait matched different people"
                    response["combined"]["face_person"] = face_result["name"]
                    response["combined"]["gait_person"] = gait_result["name"]

        return response

    except Exception as e:
        logger.error("Failed combined identification: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
