"""
REST API for presence / occupancy tracking.

Exposes current presence state and historical transitions.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query

logger = logging.getLogger("atlas.api.presence")

router = APIRouter(prefix="/presence", tags=["presence"])


@router.get("/status")
async def get_presence_status():
    """Get current occupancy state."""
    from ..autonomous.presence import get_presence_tracker

    tracker = get_presence_tracker()
    return tracker.state.to_dict()


@router.get("/history")
async def get_presence_history(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    transition: Optional[str] = Query(default=None, pattern="^(arrival|departure)$"),
):
    """Query presence transition history from the database."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"count": 0, "total": 0, "events": []}

    # Build query
    conditions = []
    params: list = []
    idx = 1

    if transition:
        conditions.append(f"transition = ${idx}")
        params.append(transition)
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    # Total count
    total_row = await pool.fetchrow(
        f"SELECT count(*) AS cnt FROM presence_events {where}",
        *params,
    )
    total = total_row["cnt"] if total_row else 0

    # Fetch page
    rows = await pool.fetch(
        f"""
        SELECT id, transition, occupancy_state, occupants,
               person_name, source_id, created_at
        FROM presence_events
        {where}
        ORDER BY created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *params, limit, offset,
    )

    events = []
    for row in rows:
        events.append({
            "id": str(row["id"]),
            "transition": row["transition"],
            "occupancy_state": row["occupancy_state"],
            "occupants": row["occupants"] or [],
            "person_name": row["person_name"],
            "source_id": row["source_id"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        })

    return {
        "count": len(events),
        "total": total,
        "events": events,
    }
