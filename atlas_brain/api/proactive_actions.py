"""
REST API for proactive action management.

CRUD operations for proactive actions extracted from conversations.
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger("atlas.api.proactive_actions")

router = APIRouter(prefix="/actions", tags=["actions"])


def _affected_rows(result: str) -> int:
    """Parse affected row count from asyncpg execute result (e.g. 'UPDATE 1')."""
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError, AttributeError):
        return -1


@router.get("/")
async def list_actions(
    status: str = Query(default="pending", description="Filter by status: pending, done, dismissed, or all"),
    limit: int = Query(default=50, ge=1, le=500),
):
    """List proactive actions with optional status filter."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"count": 0, "actions": []}

    if status == "all":
        rows = await pool.fetch(
            """
            SELECT id, action_text, action_type, status, source_time,
                   session_id, created_at, resolved_at
            FROM proactive_actions
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT id, action_text, action_type, status, source_time,
                   session_id, created_at, resolved_at
            FROM proactive_actions
            WHERE status = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            status, limit,
        )

    actions = []
    for row in rows:
        actions.append({
            "id": str(row["id"]),
            "action_text": row["action_text"],
            "action_type": row["action_type"],
            "status": row["status"],
            "source_time": row["source_time"].isoformat() if row["source_time"] else None,
            "session_id": str(row["session_id"]) if row["session_id"] else None,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "resolved_at": row["resolved_at"].isoformat() if row["resolved_at"] else None,
        })

    return {"count": len(actions), "actions": actions}


@router.post("/{action_id}/done")
async def mark_action_done(action_id: UUID):
    """Mark a proactive action as done."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    result = await pool.execute(
        """
        UPDATE proactive_actions
        SET status = 'done', resolved_at = $1
        WHERE id = $2 AND status = 'pending'
        """,
        now, action_id,
    )

    if _affected_rows(result) == 0:
        raise HTTPException(404, "Action not found or already resolved")

    row = await pool.fetchrow(
        "SELECT * FROM proactive_actions WHERE id = $1", action_id,
    )
    if not row:
        raise HTTPException(404, "Action not found")

    return _row_to_dict(row)


@router.post("/{action_id}/dismiss")
async def dismiss_action(action_id: UUID):
    """Dismiss a proactive action."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    result = await pool.execute(
        """
        UPDATE proactive_actions
        SET status = 'dismissed', resolved_at = $1
        WHERE id = $2 AND status = 'pending'
        """,
        now, action_id,
    )

    if _affected_rows(result) == 0:
        raise HTTPException(404, "Action not found or already resolved")

    row = await pool.fetchrow(
        "SELECT * FROM proactive_actions WHERE id = $1", action_id,
    )
    if not row:
        raise HTTPException(404, "Action not found")

    return _row_to_dict(row)


@router.delete("/{action_id}")
async def delete_action(action_id: UUID):
    """Hard delete a proactive action."""
    from ..storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    result = await pool.execute(
        "DELETE FROM proactive_actions WHERE id = $1", action_id,
    )

    if _affected_rows(result) == 0:
        raise HTTPException(404, "Action not found")

    return {"deleted": True, "action_id": str(action_id)}


def _row_to_dict(row) -> dict:
    """Convert a DB row to a response dict."""
    return {
        "id": str(row["id"]),
        "action_text": row["action_text"],
        "action_type": row["action_type"],
        "status": row["status"],
        "source_time": row["source_time"].isoformat() if row["source_time"] else None,
        "session_id": str(row["session_id"]) if row["session_id"] else None,
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "resolved_at": row["resolved_at"].isoformat() if row["resolved_at"] else None,
    }
