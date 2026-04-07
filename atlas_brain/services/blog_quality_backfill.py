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
_BLOG_BACKFILL_REPAIR_POLICY_VERSION = "v10"


def _latest_blog_audit_needs_backfill(
    data_context: Any,
    *,
    status: str = "",
    content_present: bool = False,
) -> bool:
    audit = latest_blog_quality_audit(data_context)
    if not audit:
        return True
    required_audit_keys = (
        "word_count",
        "min_words_required",
        "target_words",
    )
    if any(key not in audit for key in required_audit_keys):
        return True
    failure_explanation = audit.get("failure_explanation")
    if not isinstance(failure_explanation, dict):
        return True
    required_failure_keys = (
        "boundary",
        "primary_blocker",
        "cause_type",
        "missing_inputs",
        "context_sources",
    )
    if any(key not in failure_explanation for key in required_failure_keys):
        return True
    if (
        str(status or "") in {"draft", "rejected", "failed"}
        and content_present
        and str(audit.get("boundary") or "") == "backfill"
        and str(audit.get("repair_policy_version") or "") != _BLOG_BACKFILL_REPAIR_POLICY_VERSION
    ):
        return True
    return False


def derive_blog_quality_patch(row: Mapping[str, Any]) -> dict[str, Any]:
    content = blog_row_content(row)
    if not _latest_blog_audit_needs_backfill(
        row.get("data_context"),
        status=str(row.get("status") or ""),
        content_present=bool(str(content.get("content") or "").strip()),
    ):
        return {}
    if not str(content.get("content") or "").strip():
        return {}

    from ..autonomous.tasks.b2b_blog_post_generation import (
        _finalize_blog_generation_content,
        _run_blog_quality_pass,
    )

    blueprint = blog_row_to_blueprint(row)
    resolved_content, resolved_report = _run_blog_quality_pass(blueprint, content)
    finalized_content = _finalize_blog_generation_content(resolved_content) or resolved_content
    result = blog_quality_revalidation(
        blueprint=blueprint,
        content=finalized_content,
        boundary="backfill",
        report=resolved_report,
    )
    latest_audit = result["data_context"].get("latest_quality_audit")
    if isinstance(latest_audit, dict):
        latest_audit["repair_policy_version"] = _BLOG_BACKFILL_REPAIR_POLICY_VERSION
    audit = result["audit"]
    projection = blog_quality_projection(audit, boundary="backfill")
    patch = {
        "data_context": result["data_context"],
        "score": projection["score"],
        "threshold": projection["threshold"],
        "blocker_count": projection["blocker_count"],
        "warning_count": projection["warning_count"],
        "failure_step": projection["failure_step"],
        "error_code": projection["error_code"],
        "error_summary": projection["error_summary"],
        "status": (
            "draft"
            if str(row.get("status") or "") == "rejected" and str(audit.get("status") or "") == "pass"
            else str(row.get("status") or "")
        ),
        "rejection_reason": projection["rejection_reason"],
    }
    if finalized_content != content:
        patch["resolved_content"] = finalized_content
    return patch


async def derive_blog_quality_patch_with_recovery(
    pool,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    patch = derive_blog_quality_patch(row)
    if patch:
        return patch

    if str(row.get("status") or "") != "rejected":
        return {}
    if str((row.get("content") or "")).strip():
        return {}

    batch_row = await pool.fetchrow(
        """
        SELECT response_text
        FROM anthropic_message_batch_items
        WHERE artifact_type = 'blog_post'
          AND artifact_id = $1
          AND response_text IS NOT NULL
          AND response_text <> ''
        ORDER BY created_at DESC
        LIMIT 1
        """,
        str(row.get("slug") or ""),
    )
    if not batch_row:
        return {}

    from ..autonomous.tasks.b2b_blog_post_generation import (
        _finalize_blog_generation_content,
        _parse_blog_generation_response_text,
        _run_blog_quality_pass,
    )

    blueprint = blog_row_to_blueprint(row)
    parsed = _parse_blog_generation_response_text(
        blueprint,
        str(batch_row.get("response_text") or ""),
        request_envelope={},
        provider_name="anthropic_batch_recovery",
        model_name="",
    )
    recovered_content = _finalize_blog_generation_content(parsed)
    if not isinstance(recovered_content, dict):
        return {}
    if not str(recovered_content.get("content") or "").strip():
        return {}
    recovered_content, resolved_report = _run_blog_quality_pass(
        blueprint,
        recovered_content,
    )
    finalized_content = _finalize_blog_generation_content(recovered_content) or recovered_content

    result = blog_quality_revalidation(
        blueprint=blueprint,
        content=finalized_content,
        boundary="backfill",
        report=resolved_report,
    )
    latest_audit = result["data_context"].get("latest_quality_audit")
    if isinstance(latest_audit, dict):
        latest_audit["repair_policy_version"] = _BLOG_BACKFILL_REPAIR_POLICY_VERSION
    audit = result["audit"]
    projection = blog_quality_projection(audit, boundary="backfill")
    recovered_status = "draft" if str(audit.get("status") or "") == "pass" else "rejected"
    return {
        "data_context": result["data_context"],
        "score": projection["score"],
        "threshold": projection["threshold"],
        "blocker_count": projection["blocker_count"],
        "warning_count": projection["warning_count"],
        "failure_step": projection["failure_step"],
        "error_code": projection["error_code"],
        "error_summary": projection["error_summary"],
        "status": recovered_status,
        "rejection_reason": projection["rejection_reason"],
        "resolved_content": finalized_content,
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
        patch = await derive_blog_quality_patch_with_recovery(pool, row)
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
        resolved_content = patch.get("resolved_content")
        if isinstance(resolved_content, dict):
            await pool.execute(
                """
                UPDATE blog_posts
                SET title = $2,
                    description = $3,
                    content = $4,
                    seo_title = $5,
                    seo_description = $6,
                    target_keyword = $7,
                    secondary_keywords = $8::jsonb,
                    faq = $9::jsonb,
                    status = $10,
                    data_context = $11::jsonb,
                    quality_score = $12,
                    quality_threshold = $13,
                    blocker_count = $14,
                    warning_count = $15,
                    latest_failure_step = $16,
                    latest_error_code = $17,
                    latest_error_summary = $18,
                    rejection_reason = CASE WHEN $10 = 'rejected' THEN $19 ELSE NULL END,
                    rejected_at = CASE WHEN $10 = 'rejected' THEN COALESCE(rejected_at, NOW()) ELSE NULL END,
                    rejection_count = CASE WHEN $10 = 'rejected' THEN COALESCE(rejection_count, 0) ELSE 0 END
                WHERE id = $1::uuid
                """,
                item["id"],
                str(resolved_content.get("title") or ""),
                str(resolved_content.get("description") or ""),
                str(resolved_content.get("content") or ""),
                str(resolved_content.get("seo_title") or ""),
                str(resolved_content.get("seo_description") or ""),
                str(resolved_content.get("target_keyword") or ""),
                json.dumps(resolved_content.get("secondary_keywords", []), default=str),
                json.dumps(resolved_content.get("faq", []), default=str),
                str(patch.get("status") or "rejected"),
                json.dumps(patch["data_context"], default=str),
                patch["score"],
                patch["threshold"],
                patch["blocker_count"],
                patch["warning_count"],
                patch["failure_step"],
                patch["error_code"],
                patch["error_summary"],
                patch.get("rejection_reason"),
            )
        else:
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
                    latest_error_summary = $9,
                    status = $10,
                    rejection_reason = CASE WHEN $10 = 'rejected' THEN $11 ELSE NULL END,
                    rejected_at = CASE WHEN $10 = 'rejected' THEN COALESCE(rejected_at, NOW()) ELSE NULL END,
                    rejection_count = CASE WHEN $10 = 'rejected' THEN COALESCE(rejection_count, 0) ELSE 0 END
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
                str(patch.get("status") or ""),
                patch.get("rejection_reason"),
            )
        applied += 1

    return {
        "scanned": plan["scanned"],
        "changed": plan["changed"],
        "applied": applied,
        "items": plan["items"],
    }
