"""Read-model helpers for recurring report-subscription delivery health."""

from __future__ import annotations

from typing import Any


def _normalize_account_ids(account_ids: list[str] | tuple[str, ...] | set[str] | None) -> list[str] | None:
    if not account_ids:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    for value in account_ids:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized or None


async def summarize_report_subscription_delivery_health(
    pool,
    *,
    days: int = 7,
    top_n: int = 10,
    account_ids: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, Any]:
    scoped_account_ids = _normalize_account_ids(account_ids)

    summary_row = await pool.fetchrow(
        """
        WITH scoped AS (
            SELECT *
            FROM b2b_report_subscription_delivery_log
            WHERE delivered_at >= NOW() - make_interval(days => $1)
              AND ($2::text[] IS NULL OR account_id::text = ANY($2::text[]))
        )
        SELECT
            COUNT(*) AS total_attempts,
            COUNT(*) FILTER (WHERE delivery_mode = 'live') AS live_attempts,
            COUNT(*) FILTER (WHERE delivery_mode = 'dry_run') AS dry_run_attempts,
            COUNT(*) FILTER (WHERE status = 'sent') AS sent_attempts,
            COUNT(*) FILTER (WHERE status = 'partial') AS partial_attempts,
            COUNT(*) FILTER (WHERE status = 'skipped') AS skipped_attempts,
            COUNT(*) FILTER (WHERE status = 'failed') AS failed_attempts,
            COUNT(*) FILTER (WHERE status = 'processing') AS processing_attempts,
            COUNT(*) FILTER (
                WHERE status = 'skipped'
                  AND freshness_state = 'blocked'
            ) AS blocked_core_incomplete_attempts,
            COUNT(*) FILTER (
                WHERE summary ILIKE 'Skipped delivery because the eligible report package has not materially changed%'
            ) AS unchanged_skip_attempts,
            COUNT(*) FILTER (
                WHERE summary ILIKE 'Skipped delivery because every recipient on the subscription is currently suppressed.%'
            ) AS fully_suppressed_attempts,
            COALESCE(SUM((
                SELECT COUNT(*)
                FROM regexp_split_to_table(COALESCE(scoped.error, ''), '; ') AS part
                WHERE part ILIKE '%suppressed%'
            )), 0) AS suppressed_recipient_failures,
            MAX(delivered_at) AS latest_delivered_at
        FROM scoped
        """,
        days,
        scoped_account_ids,
    )

    status_breakdown = await pool.fetch(
        """
        SELECT delivery_mode, status, COUNT(*) AS attempt_count
        FROM b2b_report_subscription_delivery_log
        WHERE delivered_at >= NOW() - make_interval(days => $1)
          AND ($2::text[] IS NULL OR account_id::text = ANY($2::text[]))
        GROUP BY delivery_mode, status
        ORDER BY delivery_mode ASC, status ASC
        """,
        days,
        scoped_account_ids,
    )

    top_accounts = await pool.fetch(
        """
        SELECT
            l.account_id::text AS account_id,
            sa.name AS account_name,
            COUNT(*) AS attempt_count,
            COUNT(*) FILTER (WHERE l.delivery_mode = 'live') AS live_attempt_count,
            COUNT(*) FILTER (WHERE l.status = 'failed') AS failed_attempt_count,
            COUNT(*) FILTER (WHERE l.status = 'partial') AS partial_attempt_count,
            COUNT(*) FILTER (
                WHERE l.status = 'skipped'
                  AND l.freshness_state = 'blocked'
            ) AS blocked_core_incomplete_attempt_count,
            COUNT(*) FILTER (
                WHERE l.summary ILIKE 'Skipped delivery because the eligible report package has not materially changed%'
            ) AS unchanged_skip_attempt_count,
            MAX(l.delivered_at) AS latest_delivered_at
        FROM b2b_report_subscription_delivery_log l
        JOIN saas_accounts sa ON sa.id = l.account_id
        WHERE l.delivered_at >= NOW() - make_interval(days => $1)
          AND ($2::text[] IS NULL OR l.account_id::text = ANY($2::text[]))
        GROUP BY l.account_id, sa.name
        ORDER BY failed_attempt_count DESC, attempt_count DESC, sa.name ASC
        LIMIT $3
        """,
        days,
        scoped_account_ids,
        top_n,
    )

    recent_attempts = await pool.fetch(
        """
        SELECT
            l.id,
            l.subscription_id,
            l.account_id::text AS account_id,
            sa.name AS account_name,
            l.scheduled_for,
            l.delivered_at,
            l.delivery_mode,
            l.status,
            l.freshness_state,
            l.scope_type,
            l.scope_key,
            COALESCE(array_length(l.delivered_report_ids, 1), 0) AS report_count,
            COALESCE(array_length(l.recipient_emails, 1), 0) AS recipient_count,
            l.summary,
            l.error
        FROM b2b_report_subscription_delivery_log l
        JOIN saas_accounts sa ON sa.id = l.account_id
        WHERE l.delivered_at >= NOW() - make_interval(days => $1)
          AND ($2::text[] IS NULL OR l.account_id::text = ANY($2::text[]))
        ORDER BY l.delivered_at DESC, l.id DESC
        LIMIT $3
        """,
        days,
        scoped_account_ids,
        top_n,
    )

    due_summary = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE enabled = TRUE) AS enabled_subscriptions,
            COUNT(*) FILTER (
                WHERE enabled = TRUE
                  AND next_delivery_at IS NOT NULL
                  AND next_delivery_at <= NOW()
            ) AS due_now
        FROM b2b_report_subscriptions
        WHERE $1::text[] IS NULL OR account_id::text = ANY($1::text[])
        """,
        scoped_account_ids,
    )

    summary = dict(summary_row or {})
    due = dict(due_summary or {})
    return {
        "days": int(days),
        "filters": {
            "account_ids": list(scoped_account_ids or []),
            "top_n": int(top_n),
        },
        "delivery_summary": {
            "total_attempts": int(summary.get("total_attempts", 0) or 0),
            "live_attempts": int(summary.get("live_attempts", 0) or 0),
            "dry_run_attempts": int(summary.get("dry_run_attempts", 0) or 0),
            "sent_attempts": int(summary.get("sent_attempts", 0) or 0),
            "partial_attempts": int(summary.get("partial_attempts", 0) or 0),
            "skipped_attempts": int(summary.get("skipped_attempts", 0) or 0),
            "failed_attempts": int(summary.get("failed_attempts", 0) or 0),
            "processing_attempts": int(summary.get("processing_attempts", 0) or 0),
            "blocked_core_incomplete_attempts": int(summary.get("blocked_core_incomplete_attempts", 0) or 0),
            "unchanged_skip_attempts": int(summary.get("unchanged_skip_attempts", 0) or 0),
            "fully_suppressed_attempts": int(summary.get("fully_suppressed_attempts", 0) or 0),
            "suppressed_recipient_failures": int(summary.get("suppressed_recipient_failures", 0) or 0),
            "latest_delivered_at": summary.get("latest_delivered_at"),
        },
        "active_subscription_summary": {
            "enabled_subscriptions": int(due.get("enabled_subscriptions", 0) or 0),
            "due_now": int(due.get("due_now", 0) or 0),
        },
        "status_breakdown": [dict(row) for row in status_breakdown],
        "top_accounts": [dict(row) for row in top_accounts],
        "recent_attempts": [dict(row) for row in recent_attempts],
    }
