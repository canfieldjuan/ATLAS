"""
Person recognition API endpoints.

Provides face and gait enrollment/recognition.
"""

import logging
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

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


class PersonResponse(BaseModel):
    id: str
    name: str
    is_known: bool
    auto_created: bool
    created_at: str
    last_seen_at: Optional[str]


def _check_db_enabled():
    """Check if database is enabled."""
    if not db_settings.enabled:
        raise HTTPException(status_code=503, detail="Database disabled")
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not initialized")


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
