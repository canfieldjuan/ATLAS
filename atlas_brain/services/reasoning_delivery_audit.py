"""Read-model helpers for reasoning delivery health and provenance coverage."""

from __future__ import annotations

from typing import Any


async def summarize_reasoning_delivery_health(
    pool,
    *,
    days: int = 7,
    top_n: int = 10,
) -> dict[str, Any]:
    vendor_summary_row = await pool.fetchrow(
        """
        WITH latest_vendor AS (
            SELECT DISTINCT ON (vendor_name)
                vendor_name,
                as_of_date,
                created_at,
                tokens_used,
                synthesis
            FROM b2b_reasoning_synthesis
            WHERE created_at >= NOW() - make_interval(days => $1)
            ORDER BY vendor_name, created_at DESC
        )
        SELECT
            COUNT(*) AS vendor_rows,
            MAX(created_at) AS latest_created_at,
            COALESCE(AVG(tokens_used), 0) AS avg_tokens_used,
            COUNT(*) FILTER (
                WHERE jsonb_path_exists(synthesis, '$.reference_ids.metric_ids[*]')
            ) AS metric_ref_rows,
            COUNT(*) FILTER (
                WHERE jsonb_path_exists(synthesis, '$.reference_ids.witness_ids[*]')
            ) AS witness_ref_rows
        FROM latest_vendor
        """,
        days,
    )

    cross_vendor_summary = await pool.fetch(
        """
        SELECT
            analysis_type,
            COUNT(*) AS row_count,
            MAX(created_at) AS latest_created_at,
            COALESCE(AVG(tokens_used), 0) AS avg_tokens_used,
            COUNT(*) FILTER (
                WHERE jsonb_path_exists(synthesis, '$.reference_ids.metric_ids[*]')
            ) AS metric_ref_rows,
            COUNT(*) FILTER (
                WHERE jsonb_path_exists(synthesis, '$.reference_ids.witness_ids[*]')
            ) AS witness_ref_rows
        FROM b2b_cross_vendor_reasoning_synthesis
        WHERE created_at >= NOW() - make_interval(days => $1)
        GROUP BY analysis_type
        ORDER BY row_count DESC, analysis_type ASC
        """,
        days,
    )

    validation_rules = await pool.fetch(
        """
        SELECT
            rule_code,
            severity,
            COUNT(*) AS finding_count,
            COUNT(DISTINCT vendor_name) AS vendor_count,
            MAX(created_at) AS latest_created_at
        FROM synthesis_validation_results
        WHERE created_at >= NOW() - make_interval(days => $1)
          AND passed = false
        GROUP BY rule_code, severity
        ORDER BY finding_count DESC, vendor_count DESC, rule_code ASC
        LIMIT $2
        """,
        days,
        top_n,
    )

    latest_attempt_summary = await pool.fetch(
        """
        WITH latest_attempt AS (
            SELECT DISTINCT ON (artifact_type, artifact_id)
                artifact_type,
                artifact_id,
                status,
                stage,
                created_at
            FROM artifact_attempts
            WHERE artifact_type IN ('reasoning_synthesis', 'cross_vendor_reasoning')
              AND created_at >= NOW() - make_interval(days => $1)
            ORDER BY artifact_type, artifact_id, created_at DESC
        )
        SELECT
            artifact_type,
            status,
            COUNT(*) AS row_count
        FROM latest_attempt
        GROUP BY artifact_type, status
        ORDER BY artifact_type ASC, row_count DESC, status ASC
        """,
        days,
    )

    downstream_coverage = await pool.fetch(
        """
        SELECT
            report_type,
            COUNT(*) AS row_count,
            MAX(created_at) AS latest_created_at,
            COUNT(*) FILTER (
                WHERE jsonb_path_exists(intelligence_data, '$.**.reference_ids')
                   OR jsonb_path_exists(intelligence_data, '$.**.reasoning_reference_ids')
            ) AS rows_with_reference_ids
        FROM b2b_intelligence
        WHERE report_type IN (
            'battle_card',
            'challenger_brief',
            'accounts_in_motion',
            'weekly_churn_feed',
            'vendor_scorecard',
            'displacement_report',
            'category_overview'
        )
          AND created_at >= NOW() - make_interval(days => $1)
        GROUP BY report_type
        ORDER BY row_count DESC, report_type ASC
        """,
        days,
    )

    vendor_summary = dict(vendor_summary_row or {})
    return {
        "days": days,
        "vendor_synthesis": {
            "rows": int(vendor_summary.get("vendor_rows", 0) or 0),
            "latest_created_at": vendor_summary.get("latest_created_at"),
            "avg_tokens_used": float(vendor_summary.get("avg_tokens_used", 0) or 0),
            "rows_with_metric_refs": int(vendor_summary.get("metric_ref_rows", 0) or 0),
            "rows_with_witness_refs": int(vendor_summary.get("witness_ref_rows", 0) or 0),
        },
        "cross_vendor_synthesis": [dict(row) for row in cross_vendor_summary],
        "top_validation_findings": [dict(row) for row in validation_rules],
        "latest_attempt_statuses": [dict(row) for row in latest_attempt_summary],
        "downstream_reference_coverage": [dict(row) for row in downstream_coverage],
    }
