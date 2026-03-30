"""Pipeline visibility API: events, attempts, quarantines, reviews.

Exposes the visibility data model for the Pipeline Review UI.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.pipeline_visibility")
router = APIRouter(prefix="/pipeline/visibility", tags=["pipeline-visibility"])


# ---------------------------------------------------------------------------
# Summary strip
# ---------------------------------------------------------------------------

@router.get("/summary")
async def get_visibility_summary(hours: int = Query(24, ge=1, le=168)):
    """Top-level counts for the Pipeline Review summary strip."""
    pool = get_db_pool()
    row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE status = 'open' AND actionable) AS open_actionable,
            COUNT(*) FILTER (WHERE status = 'open') AS open_total
        FROM pipeline_visibility_reviews
        WHERE last_seen_at >= NOW() - make_interval(hours => $1)
        """,
        hours,
    )
    events = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE severity IN ('error', 'critical')) AS failures,
            COUNT(*) FILTER (WHERE event_type LIKE '%quarantine%' OR reason_code LIKE '%quarantine%') AS quarantines,
            COUNT(*) FILTER (WHERE decision = 'rejected') AS rejections
        FROM pipeline_visibility_events
        WHERE occurred_at >= NOW() - make_interval(hours => $1)
        """,
        hours,
    )
    return {
        "period_hours": hours,
        "open_actionable": row["open_actionable"] if row else 0,
        "open_total": row["open_total"] if row else 0,
        "failures_period": events["failures"] if events else 0,
        "quarantines_period": events["quarantines"] if events else 0,
        "rejections_period": events["rejections"] if events else 0,
    }


# ---------------------------------------------------------------------------
# Queue (actionable open items)
# ---------------------------------------------------------------------------

@router.get("/queue")
async def get_visibility_queue(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    stage: str | None = None,
    severity: str | None = None,
):
    """Unresolved actionable items for operator triage."""
    pool = get_db_pool()
    conditions = ["r.status = 'open'"]
    params: list[Any] = []
    idx = 1

    if stage:
        idx += 1
        conditions.append(f"e.stage = ${idx}")
        params.append(stage)
    if severity:
        idx += 1
        conditions.append(f"e.severity = ${idx}")
        params.append(severity)

    where = " AND ".join(conditions)
    rows = await pool.fetch(
        f"""
        SELECT r.id, r.fingerprint, r.status, r.occurrence_count,
               r.first_seen_at, r.last_seen_at,
               e.stage, e.event_type, e.severity, e.entity_type, e.entity_id,
               e.artifact_type, e.reason_code, e.rule_code, e.summary,
               e.run_id, e.actionable
        FROM pipeline_visibility_reviews r
        JOIN pipeline_visibility_events e ON e.id = r.latest_event_id
        WHERE {where}
        ORDER BY
            CASE e.severity
                WHEN 'critical' THEN 0
                WHEN 'error' THEN 1
                WHEN 'warning' THEN 2
                ELSE 3
            END,
            r.last_seen_at DESC
        LIMIT ${idx + 1} OFFSET ${idx + 2}
        """,
        *params, limit, offset,
    )
    return {
        "items": [dict(r) for r in rows],
        "limit": limit,
        "offset": offset,
    }


# ---------------------------------------------------------------------------
# Events (filterable history)
# ---------------------------------------------------------------------------

@router.get("/events")
async def get_visibility_events(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    stage: str | None = None,
    event_type: str | None = None,
    severity: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    reason_code: str | None = None,
    rule_code: str | None = None,
    hours: int | None = None,
):
    """Filterable event history."""
    pool = get_db_pool()
    conditions: list[str] = []
    params: list[Any] = []
    idx = 0

    for field, value in [
        ("stage", stage), ("event_type", event_type),
        ("severity", severity), ("entity_type", entity_type),
        ("entity_id", entity_id), ("reason_code", reason_code),
        ("rule_code", rule_code),
    ]:
        if value:
            idx += 1
            conditions.append(f"{field} = ${idx}")
            params.append(value)
    if hours:
        idx += 1
        conditions.append(f"occurred_at >= NOW() - make_interval(hours => ${idx})")
        params.append(hours)

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    rows = await pool.fetch(
        f"""
        SELECT id, occurred_at, run_id, stage, event_type, severity,
               actionable, entity_type, entity_id, artifact_type,
               reason_code, rule_code, decision, summary, detail,
               fingerprint
        FROM pipeline_visibility_events
        {where}
        ORDER BY occurred_at DESC
        LIMIT ${idx + 1} OFFSET ${idx + 2}
        """,
        *params, limit, offset,
    )
    return {
        "events": [
            {**dict(r), "detail": r["detail"] if isinstance(r["detail"], dict) else {}}
            for r in rows
        ],
        "limit": limit,
        "offset": offset,
    }


# ---------------------------------------------------------------------------
# Artifact attempts
# ---------------------------------------------------------------------------

@router.get("/attempts")
async def get_artifact_attempts(
    artifact_type: str | None = None,
    status: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    hours: int | None = None,
):
    """Artifact generation attempt history."""
    pool = get_db_pool()
    conditions: list[str] = []
    params: list[Any] = []
    idx = 0

    if artifact_type:
        idx += 1
        conditions.append(f"artifact_type = ${idx}")
        params.append(artifact_type)
    if status:
        idx += 1
        conditions.append(f"status = ${idx}")
        params.append(status)
    if hours:
        idx += 1
        conditions.append(f"created_at >= NOW() - make_interval(hours => ${idx})")
        params.append(hours)

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    rows = await pool.fetch(
        f"""
        SELECT id, artifact_type, artifact_id, run_id, attempt_no,
               stage, status, score, threshold,
               blocker_count, warning_count,
               blocking_issues, warnings,
               failure_step, error_message,
               started_at, completed_at
        FROM artifact_attempts
        {where}
        ORDER BY created_at DESC
        LIMIT ${idx + 1} OFFSET ${idx + 2}
        """,
        *params, limit, offset,
    )
    return {"attempts": [dict(r) for r in rows], "limit": limit, "offset": offset}


# ---------------------------------------------------------------------------
# Enrichment quarantines
# ---------------------------------------------------------------------------

@router.get("/quarantines")
async def get_enrichment_quarantines(
    reason_code: str | None = None,
    vendor_name: str | None = None,
    unreleased_only: bool = True,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Enrichment quarantine decisions."""
    pool = get_db_pool()
    conditions: list[str] = []
    params: list[Any] = []
    idx = 0

    if reason_code:
        idx += 1
        conditions.append(f"reason_code = ${idx}")
        params.append(reason_code)
    if vendor_name:
        idx += 1
        conditions.append(f"vendor_name = ${idx}")
        params.append(vendor_name)
    if unreleased_only:
        conditions.append("released_at IS NULL")

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    rows = await pool.fetch(
        f"""
        SELECT id, review_id, vendor_name, source, reason_code,
               severity, actionable, summary, evidence,
               quarantined_at, released_at, released_by
        FROM enrichment_quarantines
        {where}
        ORDER BY quarantined_at DESC
        LIMIT ${idx + 1} OFFSET ${idx + 2}
        """,
        *params, limit, offset,
    )
    return {"quarantines": [dict(r) for r in rows], "limit": limit, "offset": offset}


# ---------------------------------------------------------------------------
# Review actions
# ---------------------------------------------------------------------------

@router.post("/reviews/{review_id}/resolve")
async def resolve_review(
    review_id: UUID,
    action: str = Query(..., pattern="^(acknowledged|resolved|ignored|accepted_risk)$"),
    note: str | None = None,
):
    """Resolve or acknowledge a visibility review item."""
    pool = get_db_pool()
    result = await pool.execute(
        """
        UPDATE pipeline_visibility_reviews
        SET status = $2,
            resolution_note = $3,
            resolved_at = CASE WHEN $2 IN ('resolved', 'ignored', 'accepted_risk') THEN NOW() ELSE resolved_at END,
            resolved_by = 'operator',
            updated_at = NOW()
        WHERE id = $1
        """,
        str(review_id), action, note,
    )
    if result == "UPDATE 0":
        raise HTTPException(404, "Review not found")
    return {"status": "ok", "review_id": str(review_id), "action": action}
