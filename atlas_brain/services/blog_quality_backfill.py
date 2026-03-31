"""Helpers for backfilling canonical blog quality audits on recent rows."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .blog_quality import (
    blog_quality_projection,
    blog_quality_revalidation,
    blog_row_content,
    blog_row_to_blueprint,
    latest_blog_quality_audit,
)

DEFAULT_BACKFILL_STATUSES = (
    "draft",
    "rejected",
    "failed",
    "published",
    "archived",
)


def _latest_blog_audit_needs_backfill(data_context: Any) -> bool:
    audit = latest_blog_quality_audit(data_context)
    if not audit:
        return True
    failure_explanation = audit.get("failure_explanation")
    if not isinstance(failure_explanation, dict):
        return True
    required_keys = (
        "boundary",
        "primary_blocker",
        "cause_type",
        "missing_inputs",
        "context_sources",
    )
    return any(key not in failure_explanation for key in required_keys)


def derive_blog_quality_patch(row: Mapping[str, Any]) -> dict[str, Any]:
    if not _latest_blog_audit_needs_backfill(row.get("data_context")):
        return {}

    content = blog_row_content(row)
    if not str(content.get("content") or "").strip():
        return {}

    blueprint = blog_row_to_blueprint(row)
    result = blog_quality_revalidation(
        blueprint=blueprint,
        content=content,
        boundary="backfill",
    )
    audit = result["audit"]
    projection = blog_quality_projection(audit, boundary="backfill")
    return {
        "data_context": result["data_context"],
        "score": projection["score"],
        "threshold": projection["threshold"],
        "blocker_count": projection["blocker_count"],
        "warning_count": projection["warning_count"],
        "failure_step": projection["failure_step"],
        "error_code": projection["error_code"],
        "error_summary": projection["error_summary"],
    }


async def plan_blog_quality_backfill(
    pool,
    *,
    days: int,
    limit: int | None = None,
    statuses: Sequence[str] = DEFAULT_BACKFILL_STATUSES,
) -> dict[str, Any]:
    rows = await pool.fetch(
        """
        SELECT
            id::text AS id,
            slug,
            title,
            description,
            topic_type,
            tags,
            content,
            charts,
            data_context,
            cta,
            seo_title,
            seo_description,
            target_keyword,
            secondary_keywords,
            faq,
            status,
            created_at
        FROM blog_posts
        WHERE created_at >= NOW() - ($1::int * INTERVAL '1 day')
          AND status = ANY($2::text[])
        ORDER BY created_at DESC
        LIMIT $3
        """,
        int(days),
        list(statuses),
        int(limit or 1000000),
    )

    items: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        patch = derive_blog_quality_patch(row)
        items.append(
            {
                "id": str(row["id"]),
                "slug": str(row.get("slug") or ""),
                "status": str(row.get("status") or ""),
                "patch": patch,
            }
        )

    return {
        "scanned": len(items),
        "changed": sum(1 for item in items if item["patch"]),
        "items": items,
    }


async def apply_blog_quality_backfill(
    pool,
    *,
    days: int,
    limit: int | None = None,
    statuses: Sequence[str] = DEFAULT_BACKFILL_STATUSES,
) -> dict[str, Any]:
    plan = await plan_blog_quality_backfill(
        pool,
        days=days,
        limit=limit,
        statuses=statuses,
    )

    applied = 0
    for item in plan["items"]:
        patch = item["patch"]
        if not patch:
            continue
        await pool.execute(
            """
            UPDATE blog_posts
            SET data_context = $2::jsonb,
                quality_score = $3,
                quality_threshold = $4,
                blocker_count = $5,
                warning_count = $6,
                latest_failure_step = $7,
                latest_error_code = $8,
                latest_error_summary = $9
            WHERE id = $1::uuid
            """,
            item["id"],
            json.dumps(patch["data_context"], default=str),
            patch["score"],
            patch["threshold"],
            patch["blocker_count"],
            patch["warning_count"],
            patch["failure_step"],
            patch["error_code"],
            patch["error_summary"],
        )
        applied += 1

    return {
        "scanned": plan["scanned"],
        "changed": plan["changed"],
        "applied": applied,
        "items": plan["items"],
    }
