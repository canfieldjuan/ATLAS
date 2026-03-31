"""Read-model helpers for reasoning retry churn calibration."""

from __future__ import annotations

from typing import Any


async def summarize_reasoning_retry_churn(
    pool,
    *,
    hours: int = 24,
    top_n: int = 10,
    queue_limit: int = 20,
) -> dict[str, Any]:
    summary_row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE event_type = 'validation_retry_rejected') AS recovered_retries,
            COUNT(*) FILTER (WHERE event_type = 'validation_retry_escalated') AS escalations,
            COUNT(*) FILTER (
                WHERE event_type = 'validation_retry_escalated'
                  AND reason_code = 'repeated_validation_retry'
            ) AS repeated_rule_escalations,
            COUNT(*) FILTER (
                WHERE event_type = 'validation_retry_escalated'
                  AND reason_code = 'costly_validation_retry'
            ) AS costly_retry_escalations,
            COUNT(DISTINCT entity_id) FILTER (
                WHERE event_type = 'validation_retry_rejected'
            ) AS affected_vendors,
            COALESCE(
                SUM(COALESCE(NULLIF(detail->>'tokens_used', ''), '0')::int) FILTER (
                    WHERE event_type = 'validation_retry_rejected'
                ),
                0
            ) AS retry_tokens
        FROM pipeline_visibility_events
        WHERE stage = 'synthesis'
          AND occurred_at >= NOW() - make_interval(hours => $1)
        """,
        hours,
    )

    top_rules = await pool.fetch(
        """
        SELECT
            svr.rule_code,
            COUNT(*) AS retry_findings,
            COUNT(DISTINCT svr.vendor_name) AS vendor_count,
            MAX(svr.created_at) AS last_seen_at
        FROM synthesis_validation_results svr
        WHERE svr.created_at >= NOW() - make_interval(hours => $1)
          AND EXISTS (
                SELECT 1
                FROM artifact_attempts a_rejected
                WHERE a_rejected.run_id = svr.run_id
                  AND a_rejected.artifact_type = 'reasoning_synthesis'
                  AND a_rejected.artifact_id = svr.vendor_name
                  AND a_rejected.stage = 'validation'
                  AND a_rejected.status = 'rejected'
                  AND a_rejected.attempt_no = svr.attempt_no
          )
          AND EXISTS (
                SELECT 1
                FROM artifact_attempts a_succeeded
                WHERE a_succeeded.run_id = svr.run_id
                  AND a_succeeded.artifact_type = 'reasoning_synthesis'
                  AND a_succeeded.artifact_id = svr.vendor_name
                  AND a_succeeded.status = 'succeeded'
                  AND a_succeeded.attempt_no > svr.attempt_no
          )
        GROUP BY svr.rule_code
        ORDER BY retry_findings DESC, vendor_count DESC, svr.rule_code ASC
        LIMIT $2
        """,
        hours,
        top_n,
    )

    top_vendors = await pool.fetch(
        """
        SELECT
            entity_id AS vendor_name,
            COUNT(*) AS retry_count,
            COALESCE(
                SUM(COALESCE(NULLIF(detail->>'tokens_used', ''), '0')::int),
                0
            ) AS retry_tokens,
            MAX(occurred_at) AS last_seen_at
        FROM pipeline_visibility_events
        WHERE stage = 'synthesis'
          AND event_type = 'validation_retry_rejected'
          AND occurred_at >= NOW() - make_interval(hours => $1)
        GROUP BY entity_id
        ORDER BY retry_count DESC, retry_tokens DESC, entity_id ASC
        LIMIT $2
        """,
        hours,
        top_n,
    )

    open_queue = await pool.fetch(
        """
        SELECT
            r.id::text AS review_id,
            r.status,
            r.occurrence_count,
            r.last_seen_at,
            e.entity_id AS vendor_name,
            e.reason_code,
            e.rule_code,
            e.summary,
            e.detail
        FROM pipeline_visibility_reviews r
        JOIN pipeline_visibility_events e ON e.id = r.latest_event_id
        WHERE r.status = 'open'
          AND e.stage = 'synthesis'
          AND e.event_type = 'validation_retry_escalated'
        ORDER BY r.last_seen_at DESC
        LIMIT $1
        """,
        queue_limit,
    )

    summary = dict(summary_row or {})
    return {
        "hours": hours,
        "summary": {
            "recovered_retries": int(summary.get("recovered_retries", 0) or 0),
            "escalations": int(summary.get("escalations", 0) or 0),
            "repeated_rule_escalations": int(summary.get("repeated_rule_escalations", 0) or 0),
            "costly_retry_escalations": int(summary.get("costly_retry_escalations", 0) or 0),
            "affected_vendors": int(summary.get("affected_vendors", 0) or 0),
            "retry_tokens": int(summary.get("retry_tokens", 0) or 0),
        },
        "top_rules": [dict(row) for row in top_rules],
        "top_vendors": [dict(row) for row in top_vendors],
        "open_retry_escalations": [dict(row) for row in open_queue],
    }
