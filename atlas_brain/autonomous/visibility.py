"""Pipeline visibility: emit events, record artifact attempts, quarantines.

Thin helpers that pipeline stages call to record operational outcomes.
All writes are best-effort (never raise, never block the pipeline).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

logger = logging.getLogger("atlas.visibility")


# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------

def _fingerprint(
    stage: str,
    event_type: str,
    entity_type: str,
    entity_id: str,
    reason_code: str | None = None,
    rule_code: str | None = None,
) -> str:
    """Deterministic fingerprint for grouping repeated issues."""
    parts = [stage, event_type, entity_type, entity_id]
    if reason_code:
        parts.append(reason_code)
    if rule_code:
        parts.append(rule_code)
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Event emitter
# ---------------------------------------------------------------------------

async def emit_event(
    pool,
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
) -> str | None:
    """Emit an immutable visibility event. Returns event ID or None on failure."""
    fp = _fingerprint(stage, event_type, entity_type, entity_id, reason_code, rule_code)
    event_id = str(uuid4())
    try:
        await pool.execute(
            """
            INSERT INTO pipeline_visibility_events
                (id, occurred_at, run_id, stage, event_type, severity, actionable,
                 entity_type, entity_id, artifact_type, reason_code, rule_code,
                 decision, summary, detail, fingerprint, source_table, source_id)
            VALUES ($1, NOW(), $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                    $14::jsonb, $15, $16, $17)
            """,
            event_id, run_id, stage, event_type, severity, actionable,
            entity_type, entity_id, artifact_type, reason_code, rule_code,
            decision, summary,
            json.dumps(detail or {}, default=str),
            fp, source_table, source_id,
        )
        # Upsert review state
        await pool.execute(
            """
            INSERT INTO pipeline_visibility_reviews
                (fingerprint, status, latest_event_id, occurrence_count,
                 first_seen_at, last_seen_at)
            VALUES ($1, 'open', $2, 1, NOW(), NOW())
            ON CONFLICT (fingerprint) DO UPDATE SET
                latest_event_id = EXCLUDED.latest_event_id,
                occurrence_count = pipeline_visibility_reviews.occurrence_count + 1,
                last_seen_at = NOW(),
                updated_at = NOW(),
                status = CASE
                    WHEN pipeline_visibility_reviews.status IN ('resolved', 'ignored')
                    THEN 'open'
                    ELSE pipeline_visibility_reviews.status
                END
            """,
            fp, event_id,
        )
        return event_id
    except Exception:
        logger.debug("Failed to emit visibility event: %s/%s", stage, event_type, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Artifact attempt recorder
# ---------------------------------------------------------------------------

async def record_attempt(
    pool,
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
    """Record an artifact generation attempt. Returns attempt ID or None."""
    attempt_id = str(uuid4())
    try:
        await pool.execute(
            """
            INSERT INTO artifact_attempts
                (id, artifact_type, artifact_id, run_id, attempt_no, stage, status,
                 score, threshold, blocker_count, warning_count,
                 blocking_issues, warnings,
                 feedback_summary, failure_step, error_message,
                 completed_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                    $12::jsonb, $13::jsonb, $14, $15, $16,
                    CASE WHEN $7 != 'pending' THEN NOW() ELSE NULL END)
            """,
            attempt_id, artifact_type, artifact_id, run_id, attempt_no,
            stage, status, score, threshold, blocker_count, warning_count,
            json.dumps(blocking_issues or [], default=str),
            json.dumps(warnings or [], default=str),
            feedback_summary, failure_step, error_message,
        )
        return attempt_id
    except Exception:
        logger.debug("Failed to record artifact attempt: %s/%s", artifact_type, stage, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Enrichment quarantine recorder
# ---------------------------------------------------------------------------

async def record_quarantine(
    pool,
    *,
    review_id: str | None = None,
    vendor_name: str | None = None,
    source: str | None = None,
    reason_code: str,
    severity: str = "warning",
    actionable: bool = False,
    summary: str | None = None,
    evidence: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> str | None:
    """Record an enrichment quarantine decision. Returns quarantine ID or None."""
    qid = str(uuid4())
    try:
        await pool.execute(
            """
            INSERT INTO enrichment_quarantines
                (id, review_id, vendor_name, source, reason_code, severity,
                 actionable, summary, evidence, run_id)
            VALUES ($1, $2::uuid, $3, $4, $5, $6, $7, $8, $9::jsonb, $10)
            """,
            qid, review_id, vendor_name, source, reason_code, severity,
            actionable, summary or reason_code,
            json.dumps(evidence or {}, default=str),
            run_id,
        )
        return qid
    except Exception:
        logger.debug("Failed to record quarantine: %s/%s", vendor_name, reason_code, exc_info=True)
        return None
