"""Observability API for the Reasoning Agent."""

import json
import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from ..config import settings
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.reasoning")

router = APIRouter(prefix="/reasoning", tags=["reasoning"])


@router.get("/events")
async def list_events(
    limit: int = Query(default=50, le=200),
    event_type: Optional[str] = Query(default=None),
    unprocessed_only: bool = Query(default=False),
):
    """List recent events with processing status."""
    if not settings.reasoning.enabled:
        raise HTTPException(status_code=503, detail="Reasoning agent disabled")

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not available")

    conditions = ["1=1"]
    params = []
    idx = 1

    if event_type:
        conditions.append(f"event_type LIKE ${idx}")
        params.append(f"{event_type}%")
        idx += 1

    if unprocessed_only:
        conditions.append("processed_at IS NULL")

    where = " AND ".join(conditions)
    params.append(limit)

    rows = await pool.fetch(
        f"""
        SELECT id, event_type, source, entity_type, entity_id,
               payload, created_at, processed_at, processing_result
        FROM atlas_events
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT ${idx}
        """,
        *params,
    )

    return {
        "events": [
            {
                "id": str(r["id"]),
                "event_type": r["event_type"],
                "source": r["source"],
                "entity_type": r["entity_type"],
                "entity_id": r["entity_id"],
                "payload": r["payload"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "processed_at": r["processed_at"].isoformat() if r["processed_at"] else None,
                "processing_result": r["processing_result"],
            }
            for r in rows
        ],
        "count": len(rows),
    }


@router.get("/locks")
async def list_locks():
    """List active entity locks."""
    if not settings.reasoning.enabled:
        raise HTTPException(status_code=503, detail="Reasoning agent disabled")

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not available")

    rows = await pool.fetch(
        """
        SELECT id, entity_type, entity_id, holder, session_id,
               acquired_at, heartbeat_at
        FROM entity_locks
        WHERE released_at IS NULL
        ORDER BY acquired_at DESC
        """
    )

    return {
        "locks": [
            {
                "id": str(r["id"]),
                "entity_type": r["entity_type"],
                "entity_id": r["entity_id"],
                "holder": r["holder"],
                "session_id": r["session_id"],
                "acquired_at": r["acquired_at"].isoformat() if r["acquired_at"] else None,
                "heartbeat_at": r["heartbeat_at"].isoformat() if r["heartbeat_at"] else None,
            }
            for r in rows
        ],
        "count": len(rows),
    }


@router.get("/queue")
async def list_queue():
    """List pending queued decisions."""
    if not settings.reasoning.enabled:
        raise HTTPException(status_code=503, detail="Reasoning agent disabled")

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not available")

    rows = await pool.fetch(
        """
        SELECT rq.id, rq.event_id, rq.entity_type, rq.entity_id, rq.queued_at,
               ae.event_type, ae.source
        FROM reasoning_queue rq
        JOIN atlas_events ae ON ae.id = rq.event_id
        WHERE rq.drained_at IS NULL
        ORDER BY rq.queued_at DESC
        LIMIT 100
        """
    )

    return {
        "queued": [
            {
                "id": str(r["id"]),
                "event_id": str(r["event_id"]),
                "entity_type": r["entity_type"],
                "entity_id": r["entity_id"],
                "queued_at": r["queued_at"].isoformat() if r["queued_at"] else None,
                "event_type": r["event_type"],
                "source": r["source"],
            }
            for r in rows
        ],
        "count": len(rows),
    }


@router.post("/process/{event_id}")
async def process_event(event_id: UUID):
    """Manually trigger processing of a specific event."""
    if not settings.reasoning.enabled:
        raise HTTPException(status_code=503, detail="Reasoning agent disabled")

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not available")

    row = await pool.fetchrow(
        "SELECT * FROM atlas_events WHERE id = $1", event_id
    )
    if not row:
        raise HTTPException(status_code=404, detail="Event not found")

    from ..reasoning.events import AtlasEvent
    from ..reasoning.agent import get_reasoning_agent

    event = AtlasEvent.from_row(dict(row))
    agent = get_reasoning_agent()
    result = await agent.process_event(event)

    # Record result
    await pool.execute(
        """
        UPDATE atlas_events
        SET processed_at = NOW(), processing_result = $1::jsonb
        WHERE id = $2
        """,
        json.dumps(result, default=str),
        event_id,
    )

    return {"event_id": str(event_id), "result": result}
