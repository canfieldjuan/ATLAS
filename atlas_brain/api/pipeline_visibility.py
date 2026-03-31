"""Pipeline visibility API: events, attempts, quarantines, reviews.

Exposes the visibility data model for the Pipeline Review UI.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from ..autonomous.visibility import emit_event, record_review_action
from ..auth.dependencies import AuthUser, require_auth
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.pipeline_visibility")
router = APIRouter(prefix="/pipeline/visibility", tags=["pipeline-visibility"])


def _serialize_row(row) -> dict[str, Any]:
    """Convert asyncpg Record to JSON-safe dict."""
    out: dict[str, Any] = {}
    for k, v in dict(row).items():
        if isinstance(v, UUID):
            out[k] = str(v)
        elif isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, date):
            out[k] = v.isoformat()
        elif isinstance(v, str) and k in (
            "detail", "evidence", "blocking_issues", "warnings",
        ):
            try:
                out[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                out[k] = v
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Summary strip
# ---------------------------------------------------------------------------

@router.get("/summary")
async def get_visibility_summary(
    hours: int = Query(24, ge=1, le=720),
    _user: AuthUser = Depends(require_auth),
):
    """Top-level counts for the Pipeline Review summary strip."""
    pool = get_db_pool()
    row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE r.status = 'open' AND e.actionable) AS open_actionable,
            COUNT(*) FILTER (WHERE r.status = 'open') AS open_total
        FROM pipeline_visibility_reviews r
        LEFT JOIN pipeline_visibility_events e ON e.id = r.latest_event_id
        WHERE r.last_seen_at >= NOW() - make_interval(hours => $1)
        """,
        hours,
    )
    events = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE severity IN ('error', 'critical')) AS failures,
            COUNT(*) FILTER (WHERE event_type LIKE '%quarantine%' OR reason_code LIKE '%quarantine%') AS quarantines,
            COUNT(*) FILTER (WHERE decision = 'rejected') AS rejections,
            COUNT(*) FILTER (WHERE event_type = 'validation_retry_rejected') AS recovered_validation_retries
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
        "recovered_validation_retries_period": (
            events["recovered_validation_retries"] if events else 0
        ),
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
    _user: AuthUser = Depends(require_auth),
):
    """Unresolved actionable items for operator triage."""
    pool = get_db_pool()
    conditions = ["r.status = 'open'", "e.actionable = TRUE"]
    params: list[Any] = []
    idx = 0

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
        "items": [_serialize_row(r) for r in rows],
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
    _user: AuthUser = Depends(require_auth),
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
            _serialize_row(r)
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
    _user: AuthUser = Depends(require_auth),
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
    return {"attempts": [_serialize_row(r) for r in rows], "limit": limit, "offset": offset}


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
    _user: AuthUser = Depends(require_auth),
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
    return {"quarantines": [_serialize_row(r) for r in rows], "limit": limit, "offset": offset}


# ---------------------------------------------------------------------------
# Synthesis validation results
# ---------------------------------------------------------------------------

@router.get("/synthesis-validation")
async def get_synthesis_validation_results(
    vendor_name: str | None = None,
    rule_code: str | None = None,
    severity: str | None = None,
    passed: bool | None = None,
    run_id: str | None = None,
    retry_only: bool = False,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _user: AuthUser = Depends(require_auth),
):
    """Normalized per-rule synthesis validation history."""
    pool = get_db_pool()
    conditions: list[str] = []
    params: list[Any] = []
    idx = 0

    for field, value in [
        ("vendor_name", vendor_name),
        ("rule_code", rule_code),
        ("severity", severity),
        ("run_id", run_id),
    ]:
        if value:
            idx += 1
            conditions.append(f"{field} = ${idx}")
            params.append(value)
    if passed is not None:
        idx += 1
        conditions.append(f"passed = ${idx}")
        params.append(passed)
    if retry_only:
        conditions.append(
            """
            EXISTS (
                SELECT 1
                FROM artifact_attempts a_rejected
                WHERE a_rejected.run_id = synthesis_validation_results.run_id
                  AND a_rejected.artifact_type = 'reasoning_synthesis'
                  AND a_rejected.artifact_id = synthesis_validation_results.vendor_name
                  AND a_rejected.stage = 'validation'
                  AND a_rejected.status = 'rejected'
                  AND a_rejected.attempt_no = synthesis_validation_results.attempt_no
            )
            AND EXISTS (
                SELECT 1
                FROM artifact_attempts a_succeeded
                WHERE a_succeeded.run_id = synthesis_validation_results.run_id
                  AND a_succeeded.artifact_type = 'reasoning_synthesis'
                  AND a_succeeded.artifact_id = synthesis_validation_results.vendor_name
                  AND a_succeeded.status = 'succeeded'
                  AND a_succeeded.attempt_no > synthesis_validation_results.attempt_no
            )
            """
        )

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    rows = await pool.fetch(
        f"""
        SELECT id, vendor_name, as_of_date, analysis_window_days, schema_version,
               run_id, attempt_no, rule_code, severity, passed, summary,
               field_path, detail, created_at
        FROM synthesis_validation_results
        {where}
        ORDER BY created_at DESC
        LIMIT ${idx + 1} OFFSET ${idx + 2}
        """,
        *params,
        limit,
        offset,
    )
    return {"results": [_serialize_row(r) for r in rows], "limit": limit, "offset": offset}


# ---------------------------------------------------------------------------
# Dedup decisions
# ---------------------------------------------------------------------------

@router.get("/dedup-decisions")
async def get_dedup_decisions(
    stage: str | None = None,
    entity_type: str | None = None,
    reason_code: str | None = None,
    run_id: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _user: AuthUser = Depends(require_auth),
):
    """First-class dedup/discard decisions."""
    pool = get_db_pool()
    conditions: list[str] = []
    params: list[Any] = []
    idx = 0

    for field, value in [
        ("stage", stage),
        ("entity_type", entity_type),
        ("reason_code", reason_code),
        ("run_id", run_id),
    ]:
        if value:
            idx += 1
            conditions.append(f"{field} = ${idx}")
            params.append(value)

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    rows = await pool.fetch(
        f"""
        SELECT id, run_id, stage, entity_type, survivor_entity_id,
               discarded_entity_id, reason_code, comparison_metrics,
               actor_type, actor_id, decided_at
        FROM dedup_decisions
        {where}
        ORDER BY decided_at DESC
        LIMIT ${idx + 1} OFFSET ${idx + 2}
        """,
        *params,
        limit,
        offset,
    )
    return {"decisions": [_serialize_row(r) for r in rows], "limit": limit, "offset": offset}


# ---------------------------------------------------------------------------
# Review action history
# ---------------------------------------------------------------------------

@router.get("/review-actions")
async def get_review_actions(
    review_id: str | None = None,
    target_entity_type: str | None = None,
    target_entity_id: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _user: AuthUser = Depends(require_auth),
):
    """Immutable operator action history for visibility reviews."""
    pool = get_db_pool()
    conditions: list[str] = []
    params: list[Any] = []
    idx = 0

    for field, value in [
        ("review_id", review_id),
        ("target_entity_type", target_entity_type),
        ("target_entity_id", target_entity_id),
    ]:
        if value:
            idx += 1
            cast = "::uuid" if field == "review_id" else ""
            conditions.append(f"{field} = ${idx}{cast}")
            params.append(value)

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    rows = await pool.fetch(
        f"""
        SELECT id, review_id, fingerprint, target_entity_type, target_entity_id,
               action, note, actor_id, actor_type, created_at
        FROM pipeline_review_actions
        {where}
        ORDER BY created_at DESC
        LIMIT ${idx + 1} OFFSET ${idx + 2}
        """,
        *params,
        limit,
        offset,
    )
    return {"actions": [_serialize_row(r) for r in rows], "limit": limit, "offset": offset}


# ---------------------------------------------------------------------------
# Review actions
# ---------------------------------------------------------------------------

@router.post("/reviews/{review_id}/resolve")
async def resolve_review(
    review_id: UUID,
    action: str = Query(..., pattern="^(acknowledge|acknowledged|resolve|resolved|ignore|ignored|accept_risk|accepted_risk)$"),
    note: str | None = None,
    user: AuthUser = Depends(require_auth),
):
    """Resolve or acknowledge a visibility review item."""
    pool = get_db_pool()
    action_map = {
        "acknowledge": "acknowledged",
        "acknowledged": "acknowledged",
        "resolve": "resolved",
        "resolved": "resolved",
        "ignore": "ignored",
        "ignored": "ignored",
        "accept_risk": "accepted_risk",
        "accepted_risk": "accepted_risk",
    }
    mapped_action = action_map[action]
    row = await pool.fetchrow(
        """
        SELECT r.id, r.fingerprint, e.entity_type, e.entity_id
        FROM pipeline_visibility_reviews r
        LEFT JOIN pipeline_visibility_events e ON e.id = r.latest_event_id
        WHERE r.id = $1
        """,
        str(review_id),
    )
    if not row:
        raise HTTPException(404, "Review not found")
    await record_review_action(
        pool,
        review_id=str(review_id),
        fingerprint=row["fingerprint"],
        target_entity_type=row["entity_type"] or "unknown",
        target_entity_id=row["entity_id"] or "unknown",
        action=mapped_action,
        note=note,
        actor_id=user.user_id,
    )
    await emit_event(
        pool,
        stage="pipeline_review",
        event_type="review_action",
        entity_type=row["entity_type"] or "unknown",
        entity_id=row["entity_id"] or "unknown",
        summary=f"Operator marked review {mapped_action}",
        severity="info",
        actionable=False,
        decision=mapped_action,
        detail={"review_id": str(review_id), "note": note or "", "actor_id": user.user_id},
        update_review_state=False,
    )
    result = await pool.execute(
        """
        UPDATE pipeline_visibility_reviews
        SET status = $2,
            resolution_note = $3,
            resolved_at = CASE WHEN $2 IN ('resolved', 'ignored', 'accepted_risk') THEN NOW() ELSE resolved_at END,
            resolved_by = $4,
            updated_at = NOW()
        WHERE id = $1
        """,
        str(review_id), mapped_action, note, user.user_id,
    )
    if result == "UPDATE 0":
        raise HTTPException(404, "Review not found")
    return {"status": "ok", "review_id": str(review_id), "action": mapped_action}
