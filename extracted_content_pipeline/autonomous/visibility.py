"""Standalone visibility helpers for copied campaign tasks."""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from ..pipelines.notify import get_pipeline_notification_sink


logger = logging.getLogger("extracted_content_pipeline.autonomous.visibility")


async def _notify(event_type: str, payload: dict[str, Any]) -> None:
    sink = get_pipeline_notification_sink()
    if sink is None:
        return
    try:
        await sink.emit(event_type, payload)
    except Exception:
        logger.debug("Failed to emit visibility notification", exc_info=True)


async def emit_event(
    pool: Any,
    *,
    stage: str,
    event_type: str,
    entity_type: str,
    entity_id: str,
    summary: str,
    severity: str = "info",
    actionable: bool = False,
    artifact_type: str | None = None,
    reason_code: str | None = None,
    rule_code: str | None = None,
    decision: str | None = None,
    run_id: str | None = None,
    detail: dict[str, Any] | None = None,
    source_table: str | None = None,
    source_id: str | None = None,
    update_review_state: bool = True,
) -> str | None:
    """Best-effort visibility event writer with notification-sink fallback."""
    del update_review_state
    event_id = str(uuid4())
    payload = {
        "id": event_id,
        "stage": stage,
        "event_type": event_type,
        "severity": severity,
        "actionable": actionable,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "artifact_type": artifact_type,
        "reason_code": reason_code,
        "rule_code": rule_code,
        "decision": decision,
        "summary": summary,
        "detail": dict(detail or {}),
        "run_id": run_id,
        "source_table": source_table,
        "source_id": source_id,
    }
    try:
        if hasattr(pool, "execute"):
            await pool.execute(
                """
                INSERT INTO pipeline_visibility_events
                    (id, run_id, stage, event_type, severity, actionable,
                     entity_type, entity_id, artifact_type, reason_code,
                     rule_code, decision, summary, detail, source_table, source_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                        $14::jsonb, $15, $16)
                """,
                event_id,
                run_id,
                stage,
                event_type,
                severity,
                actionable,
                entity_type,
                entity_id,
                artifact_type,
                reason_code,
                rule_code,
                decision,
                summary,
                json.dumps(detail or {}, default=str),
                source_table,
                source_id,
            )
    except Exception:
        logger.debug("Failed to persist visibility event", exc_info=True)
    await _notify("pipeline_visibility_event", payload)
    return event_id


async def record_attempt(
    pool: Any,
    *,
    artifact_type: str,
    artifact_id: str | None = None,
    run_id: str | None = None,
    attempt_no: int = 1,
    stage: str,
    status: str,
    score: int | None = None,
    threshold: int | None = None,
    blocker_count: int = 0,
    warning_count: int = 0,
    blocking_issues: list[str] | None = None,
    warnings: list[str] | None = None,
    feedback_summary: str | None = None,
    failure_step: str | None = None,
    error_message: str | None = None,
) -> str | None:
    """Best-effort artifact attempt recorder."""
    attempt_id = str(uuid4())
    payload = {
        "id": attempt_id,
        "artifact_type": artifact_type,
        "artifact_id": artifact_id,
        "run_id": run_id,
        "attempt_no": attempt_no,
        "stage": stage,
        "status": status,
        "score": score,
        "threshold": threshold,
        "blocker_count": blocker_count,
        "warning_count": warning_count,
        "blocking_issues": list(blocking_issues or []),
        "warnings": list(warnings or []),
        "feedback_summary": feedback_summary,
        "failure_step": failure_step,
        "error_message": error_message,
    }
    try:
        if hasattr(pool, "execute"):
            await pool.execute(
                """
                INSERT INTO artifact_attempts
                    (id, artifact_type, artifact_id, run_id, attempt_no, stage,
                     status, score, threshold, blocker_count, warning_count,
                     blocking_issues, warnings, feedback_summary, failure_step,
                     error_message)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                        $12::jsonb, $13::jsonb, $14, $15, $16)
                """,
                attempt_id,
                artifact_type,
                artifact_id,
                run_id,
                attempt_no,
                stage,
                status,
                score,
                threshold,
                blocker_count,
                warning_count,
                json.dumps(blocking_issues or [], default=str),
                json.dumps(warnings or [], default=str),
                feedback_summary,
                failure_step,
                error_message,
            )
    except Exception:
        logger.debug("Failed to persist artifact attempt", exc_info=True)
    await _notify("artifact_attempt", payload)
    return attempt_id


__all__ = ["emit_event", "record_attempt"]
