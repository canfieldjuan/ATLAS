"""Helpers for backfilling truthful visibility fields on legacy blog rows."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

QUALITY_GATE_STAGE = "quality_gate"
QUALITY_GATE_REJECTION_CODE = "quality_gate_rejection"
DEFAULT_BACKFILL_STATUSES = ("draft", "rejected", "failed")


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = [value]
        else:
            value = decoded
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                items.append(text)
        return items
    text = str(value or "").strip()
    return [text] if text else []


def _issue_summary(value: Any) -> str | None:
    issues = _as_list(value)
    if not issues:
        return None
    return ", ".join(issues[:3])


def derive_blog_visibility_patch(
    row: Mapping[str, Any],
    *,
    default_threshold: int,
) -> dict[str, Any]:
    """Compute conservative blog visibility field backfills from attempt history."""
    patch: dict[str, Any] = {}

    attempt_run_id = row.get("attempt_run_id")
    attempt_no = row.get("attempt_attempt_no")
    attempt_failure_step = row.get("attempt_failure_step")
    attempt_error_message = row.get("attempt_error_message")

    quality_attempt_no = row.get("quality_attempt_no")
    quality_score = row.get("quality_score_attempt")
    quality_threshold = row.get("quality_threshold_attempt")
    quality_blocker_count = row.get("quality_blocker_count")
    quality_warning_count = row.get("quality_warning_count")
    quality_completed_at = row.get("quality_completed_at")
    quality_status = str(row.get("quality_attempt_status") or "")
    quality_failure_step = str(row.get("quality_failure_step") or "")
    quality_summary = _issue_summary(row.get("quality_blocking_issues"))

    if row.get("latest_run_id") is None and attempt_run_id:
        patch["latest_run_id"] = str(attempt_run_id)
    if row.get("latest_attempt_no") is None and attempt_no is not None:
        patch["latest_attempt_no"] = int(attempt_no)

    derived_failure_step = attempt_failure_step or (
        QUALITY_GATE_STAGE
        if str(row.get("status") or "") == "rejected" and quality_attempt_no is not None
        else None
    )
    if row.get("latest_failure_step") is None and derived_failure_step:
        patch["latest_failure_step"] = str(derived_failure_step)

    derived_error_summary = attempt_error_message or quality_summary
    if row.get("latest_error_summary") is None and derived_error_summary:
        patch["latest_error_summary"] = str(derived_error_summary)

    if (
        row.get("latest_error_code") is None
        and str(row.get("status") or "") == "rejected"
        and quality_attempt_no is not None
        and (quality_status == "rejected" or quality_failure_step == QUALITY_GATE_STAGE)
    ):
        patch["latest_error_code"] = QUALITY_GATE_REJECTION_CODE

    if row.get("quality_score") is None and quality_score is not None:
        patch["quality_score"] = int(quality_score)

    if row.get("quality_threshold") is None and quality_attempt_no is not None:
        patch["quality_threshold"] = int(quality_threshold or default_threshold)

    if quality_attempt_no is not None and quality_blocker_count is not None:
        blocker_count = int(quality_blocker_count)
        if blocker_count != int(row.get("blocker_count") or 0):
            patch["blocker_count"] = blocker_count

    if quality_attempt_no is not None and quality_warning_count is not None:
        warning_count = int(quality_warning_count)
        if warning_count != int(row.get("warning_count") or 0):
            patch["warning_count"] = warning_count

    if str(row.get("status") or "") == "rejected":
        if row.get("rejection_reason") is None and quality_summary:
            patch["rejection_reason"] = quality_summary
        if row.get("rejected_at") is None and isinstance(quality_completed_at, datetime):
            patch["rejected_at"] = quality_completed_at

    return patch


async def plan_blog_visibility_backfill(
    pool,
    *,
    limit: int | None = None,
    statuses: Sequence[str] = DEFAULT_BACKFILL_STATUSES,
    default_threshold: int,
) -> dict[str, Any]:
    rows = await pool.fetch(
        """
        SELECT
            b.id::text AS id,
            b.slug,
            b.status,
            b.latest_run_id,
            b.latest_attempt_no,
            b.latest_failure_step,
            b.latest_error_code,
            b.latest_error_summary,
            b.quality_score,
            b.quality_threshold,
            b.blocker_count,
            b.warning_count,
            b.rejected_at,
            b.rejection_reason,
            la.run_id AS attempt_run_id,
            la.attempt_no AS attempt_attempt_no,
            la.failure_step AS attempt_failure_step,
            la.error_message AS attempt_error_message,
            lq.attempt_no AS quality_attempt_no,
            lq.status AS quality_attempt_status,
            lq.failure_step AS quality_failure_step,
            lq.score AS quality_score_attempt,
            lq.threshold AS quality_threshold_attempt,
            lq.blocker_count AS quality_blocker_count,
            lq.warning_count AS quality_warning_count,
            lq.blocking_issues AS quality_blocking_issues,
            lq.completed_at AS quality_completed_at
        FROM blog_posts b
        LEFT JOIN LATERAL (
            SELECT run_id, attempt_no, failure_step, error_message
            FROM artifact_attempts a
            WHERE a.artifact_type = 'blog_post'
              AND a.artifact_id = b.slug
            ORDER BY COALESCE(a.completed_at, a.created_at) DESC,
                     a.created_at DESC,
                     a.attempt_no DESC
            LIMIT 1
        ) la ON TRUE
        LEFT JOIN LATERAL (
            SELECT attempt_no, status, failure_step, score, threshold,
                   blocker_count, warning_count, blocking_issues, completed_at
            FROM artifact_attempts a
            WHERE a.artifact_type = 'blog_post'
              AND a.artifact_id = b.slug
              AND a.stage = 'quality_gate'
            ORDER BY COALESCE(a.completed_at, a.created_at) DESC,
                     a.created_at DESC,
                     a.attempt_no DESC
            LIMIT 1
        ) lq ON TRUE
        WHERE b.status = ANY($1::text[])
          AND (
                b.latest_run_id IS NULL
             OR b.latest_attempt_no IS NULL
             OR b.latest_failure_step IS NULL
             OR b.latest_error_summary IS NULL
             OR b.quality_score IS NULL
             OR b.quality_threshold IS NULL
             OR (b.status = 'rejected' AND (b.rejected_at IS NULL OR b.rejection_reason IS NULL))
          )
        ORDER BY b.created_at DESC
        LIMIT $2
        """,
        list(statuses),
        int(limit or 1000000),
    )

    items: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        patch = derive_blog_visibility_patch(row, default_threshold=default_threshold)
        items.append(
            {
                "id": str(row["id"]),
                "slug": str(row["slug"] or ""),
                "status": str(row["status"] or ""),
                "patch": patch,
            }
        )
    return {
        "scanned": len(items),
        "changed": sum(1 for item in items if item["patch"]),
        "items": items,
    }


async def apply_blog_visibility_backfill(
    pool,
    *,
    limit: int | None = None,
    statuses: Sequence[str] = DEFAULT_BACKFILL_STATUSES,
    default_threshold: int,
) -> dict[str, Any]:
    plan = await plan_blog_visibility_backfill(
        pool,
        limit=limit,
        statuses=statuses,
        default_threshold=default_threshold,
    )
    applied = 0
    for item in plan["items"]:
        patch = item["patch"]
        if not patch:
            continue
        columns = list(patch.keys())
        assignments = [f"{column} = ${idx}" for idx, column in enumerate(columns, start=2)]
        await pool.execute(
            f"""
            UPDATE blog_posts
            SET {", ".join(assignments)}
            WHERE id = $1::uuid
            """,
            item["id"],
            *[patch[column] for column in columns],
        )
        applied += 1
    return {
        "scanned": plan["scanned"],
        "changed": plan["changed"],
        "applied": applied,
        "items": plan["items"],
    }
