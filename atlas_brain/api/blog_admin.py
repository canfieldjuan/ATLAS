"""
Admin REST endpoints for blog post draft review, editing, and publishing.

Drafts are generated nightly by blog_post_generation and stored in the
blog_posts table. This router exposes list/get/patch/publish operations.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

import markdown as _md

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..auth.dependencies import AuthUser, require_auth
from ..autonomous.visibility import emit_event, record_attempt
from ..services.blog_quality import (
    blog_failure_explanation,
    blog_first_pass_failure_explanation,
    blog_quality_projection,
    blog_quality_revalidation,
    latest_blog_first_pass_quality_audit,
    blog_row_content,
    blog_row_to_blueprint,
)
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.blog_admin")

router = APIRouter(prefix="/admin/blog", tags=["blog-admin"])


# -- schemas ------------------------------------------------------

class BlogDraftSummary(BaseModel):
    id: str
    slug: str
    title: str
    topic_type: str
    status: str
    llm_model: Optional[str] = None
    created_at: str
    published_at: Optional[str] = None
    rejected_at: Optional[str] = None
    rejection_reason: Optional[str] = None
    quality_score: Optional[int] = None
    quality_threshold: Optional[int] = None
    blocker_count: int = 0
    warning_count: int = 0
    latest_failure_step: Optional[str] = None
    latest_error_code: Optional[str] = None
    latest_error_summary: Optional[str] = None
    unresolved_issue_count: int = 0
    failure_explanation: Optional[dict] = None
    first_pass_failure_explanation: Optional[dict] = None


class BlogDraftDetail(BaseModel):
    id: str
    slug: str
    title: str
    description: Optional[str] = None
    topic_type: str
    tags: list
    content: str
    charts: list
    data_context: Optional[dict] = None
    status: str
    reviewer_notes: Optional[str] = None
    llm_model: Optional[str] = None
    source_report_date: Optional[str] = None
    created_at: str
    published_at: Optional[str] = None
    rejected_at: Optional[str] = None
    rejection_reason: Optional[str] = None
    quality_score: Optional[int] = None
    quality_threshold: Optional[int] = None
    blocker_count: int = 0
    warning_count: int = 0
    latest_failure_step: Optional[str] = None
    latest_error_code: Optional[str] = None
    latest_error_summary: Optional[str] = None
    unresolved_issue_count: int = 0
    failure_explanation: Optional[dict] = None
    first_pass_failure_explanation: Optional[dict] = None
    first_pass_quality_audit: Optional[dict] = None
    seo_title: Optional[str] = None
    seo_description: Optional[str] = None
    target_keyword: Optional[str] = None
    secondary_keywords: Optional[list] = None
    faq: Optional[list] = None
    related_slugs: Optional[list] = None
    cta: Optional[dict] = None


class BlogDraftPatch(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    charts: Optional[list] = None
    tags: Optional[list] = None
    status: Optional[str] = None
    reviewer_notes: Optional[str] = None
    seo_title: Optional[str] = None
    seo_description: Optional[str] = None
    target_keyword: Optional[str] = None
    secondary_keywords: Optional[list] = None
    faq: Optional[list] = None
    related_slugs: Optional[list] = None


class ManualGenerateRequest(BaseModel):
    vendor_name: str
    topic_type: str
    vendor_b: Optional[str] = None
    category: Optional[str] = None
    skip_sufficiency_check: bool = False
    force_retry_blocked_slug: bool = False


# -- helpers ------------------------------------------------------

def _safe_json(val):
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val
    return val


def _row_to_summary(row) -> dict:
    data_context = row.get("data_context")
    return {
        "id": str(row["id"]),
        "slug": row["slug"],
        "title": row["title"],
        "topic_type": row["topic_type"],
        "status": row["status"],
        "llm_model": row.get("llm_model"),
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "published_at": row["published_at"].isoformat() if row.get("published_at") else None,
        "rejected_at": row["rejected_at"].isoformat() if row.get("rejected_at") else None,
        "rejection_reason": row.get("rejection_reason"),
        "quality_score": row.get("quality_score"),
        "quality_threshold": row.get("quality_threshold"),
        "blocker_count": row.get("blocker_count") or 0,
        "warning_count": row.get("warning_count") or 0,
        "latest_failure_step": row.get("latest_failure_step"),
        "latest_error_code": row.get("latest_error_code"),
        "latest_error_summary": row.get("latest_error_summary"),
        "unresolved_issue_count": row.get("unresolved_issue_count") or 0,
        "failure_explanation": blog_failure_explanation(data_context),
        "first_pass_failure_explanation": blog_first_pass_failure_explanation(data_context),
    }


_md_converter = _md.Markdown(extensions=["tables", "fenced_code", "toc"])


def _render_md(text: str) -> str:
    """Convert markdown to HTML. Resets converter state between calls."""
    _md_converter.reset()
    return _md_converter.convert(text)


def _row_to_detail(row) -> dict:
    raw_content = row["content"] or ""
    data_context = _safe_json(row.get("data_context"))
    return {
        "id": str(row["id"]),
        "slug": row["slug"],
        "title": row["title"],
        "description": row.get("description"),
        "topic_type": row["topic_type"],
        "tags": _safe_json(row.get("tags", [])),
        "content": _render_md(raw_content),
        "charts": _safe_json(row.get("charts", [])),
        "data_context": data_context,
        "status": row["status"],
        "reviewer_notes": row.get("reviewer_notes"),
        "llm_model": row.get("llm_model"),
        "source_report_date": str(row["source_report_date"]) if row.get("source_report_date") else None,
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "published_at": row["published_at"].isoformat() if row.get("published_at") else None,
        "rejected_at": row["rejected_at"].isoformat() if row.get("rejected_at") else None,
        "rejection_reason": row.get("rejection_reason"),
        "quality_score": row.get("quality_score"),
        "quality_threshold": row.get("quality_threshold"),
        "blocker_count": row.get("blocker_count") or 0,
        "warning_count": row.get("warning_count") or 0,
        "latest_failure_step": row.get("latest_failure_step"),
        "latest_error_code": row.get("latest_error_code"),
        "latest_error_summary": row.get("latest_error_summary"),
        "unresolved_issue_count": row.get("unresolved_issue_count") or 0,
        "failure_explanation": blog_failure_explanation(data_context),
        "first_pass_failure_explanation": blog_first_pass_failure_explanation(data_context),
        "first_pass_quality_audit": latest_blog_first_pass_quality_audit(data_context),
        "seo_title": row.get("seo_title"),
        "seo_description": row.get("seo_description"),
        "target_keyword": row.get("target_keyword"),
        "secondary_keywords": _safe_json(row.get("secondary_keywords", [])),
        "faq": _safe_json(row.get("faq", [])),
        "related_slugs": _safe_json(row.get("related_slugs", [])),
        "cta": _safe_json(row.get("cta")) if isinstance(_safe_json(row.get("cta")), dict) else None,
    }


def _stored_row_to_blueprint(row):
    return blog_row_to_blueprint(row)


def _publish_revalidation_result(row) -> dict:
    blueprint = _stored_row_to_blueprint(row)
    return blog_quality_revalidation(
        blueprint=blueprint,
        content=blog_row_content(row),
        boundary="publish",
    )


def _publish_revalidation_report(row) -> dict:
    return _publish_revalidation_result(row)["audit"]


# -- endpoints ----------------------------------------------------

@router.get("/drafts")
async def list_drafts(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    _user: AuthUser = Depends(require_auth),
):
    """List blog post drafts, optionally filtered by status."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not ready")

    if status:
        rows = await pool.fetch(
            """
            SELECT id, slug, title, topic_type, status, llm_model, created_at, published_at,
                   rejected_at, rejection_reason, quality_score, quality_threshold,
                   blocker_count, warning_count, latest_failure_step,
                   latest_error_code, latest_error_summary, data_context,
                   (
                     SELECT COUNT(*)
                     FROM pipeline_visibility_reviews r
                     JOIN pipeline_visibility_events e ON e.id = r.latest_event_id
                     WHERE r.status = 'open'
                       AND e.entity_type = 'blog_post'
                       AND e.entity_id = blog_posts.slug
                   ) AS unresolved_issue_count
            FROM blog_posts
            WHERE status = $1
            ORDER BY created_at DESC LIMIT $2
            """,
            status, limit,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT id, slug, title, topic_type, status, llm_model, created_at, published_at,
                   rejected_at, rejection_reason, quality_score, quality_threshold,
                   blocker_count, warning_count, latest_failure_step,
                   latest_error_code, latest_error_summary, data_context,
                   (
                     SELECT COUNT(*)
                     FROM pipeline_visibility_reviews r
                     JOIN pipeline_visibility_events e ON e.id = r.latest_event_id
                     WHERE r.status = 'open'
                       AND e.entity_type = 'blog_post'
                       AND e.entity_id = blog_posts.slug
                   ) AS unresolved_issue_count
            FROM blog_posts
            ORDER BY created_at DESC LIMIT $1
            """,
            limit,
    )
    return [_row_to_summary(r) for r in rows]


@router.get("/drafts/summary")
async def draft_summary(_user: AuthUser = Depends(require_auth)):
    """Roll up current blog draft quality and status counts."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not ready")

    status_rows = await pool.fetch(
        """
        SELECT status, COUNT(*) AS count
        FROM blog_posts
        GROUP BY status
        ORDER BY count DESC, status ASC
        """
    )
    quality_row = await pool.fetchrow(
        """
        WITH blog_state AS (
            SELECT
                b.*,
                (
                    SELECT COUNT(*)
                    FROM pipeline_visibility_reviews r
                    JOIN pipeline_visibility_events e ON e.id = r.latest_event_id
                    WHERE r.status = 'open'
                      AND e.entity_type = 'blog_post'
                      AND e.entity_id = b.slug
                ) AS unresolved_issue_count
            FROM blog_posts b
        )
        SELECT
            COUNT(*) FILTER (
                WHERE COALESCE(blocker_count, 0) = 0
                  AND COALESCE(warning_count, 0) = 0
                  AND quality_score IS NOT NULL
            ) AS clean,
            COUNT(*) FILTER (
                WHERE COALESCE(blocker_count, 0) = 0
                  AND COALESCE(warning_count, 0) > 0
            ) AS warning_only,
            COUNT(*) FILTER (
                WHERE COALESCE(blocker_count, 0) > 0
            ) AS failing,
            COUNT(*) FILTER (
                WHERE COALESCE(unresolved_issue_count, 0) > 0
            ) AS unresolved,
            COALESCE(SUM(COALESCE(blocker_count, 0)), 0) AS blocker_total,
            COALESCE(SUM(COALESCE(warning_count, 0)), 0) AS warning_total
        FROM blog_state
        """
    )
    blocker_rows = await pool.fetch(
        """
        SELECT latest_error_summary AS reason, COUNT(*) AS count
        FROM blog_posts
        WHERE COALESCE(blocker_count, 0) > 0
          AND COALESCE(latest_error_summary, '') <> ''
        GROUP BY latest_error_summary
        ORDER BY count DESC, latest_error_summary ASC
        LIMIT 5
        """
    )
    failure_step_rows = await pool.fetch(
        """
        SELECT COALESCE(latest_failure_step, 'missing') AS step, COUNT(*) AS count
        FROM blog_posts
        GROUP BY step
        ORDER BY count DESC, step ASC
        LIMIT 10
        """
    )

    quality = dict(quality_row or {})
    return {
        "by_status": {
            str(item["status"] or "unknown"): int(item["count"] or 0)
            for item in status_rows
        },
        "quality": {
            "clean": int(quality.get("clean") or 0),
            "warning_only": int(quality.get("warning_only") or 0),
            "failing": int(quality.get("failing") or 0),
            "unresolved": int(quality.get("unresolved") or 0),
            "blocker_total": int(quality.get("blocker_total") or 0),
            "warning_total": int(quality.get("warning_total") or 0),
            "by_failure_step": [
                {"step": str(item["step"] or "missing"), "count": int(item["count"] or 0)}
                for item in failure_step_rows
            ],
            "top_blockers": [
                {"reason": str(item["reason"] or ""), "count": int(item["count"] or 0)}
                for item in blocker_rows
            ],
        },
    }


@router.get("/quality-trends")
async def blog_quality_trends(
    days: int = Query(14, ge=1, le=90),
    top_n: int = Query(5, ge=1, le=20),
    _user: AuthUser = Depends(require_auth),
):
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not ready")

    trend_rows = await pool.fetch(
        """
        WITH blocker_rows AS (
            SELECT
                DATE_TRUNC('day', created_at)::date::text AS day,
                blocker.reason AS reason
            FROM blog_posts
            JOIN LATERAL jsonb_array_elements_text(
                COALESCE(data_context->'latest_quality_audit'->'blocking_issues', '[]'::jsonb)
            ) AS blocker(reason) ON TRUE
            WHERE created_at >= NOW() - ($1::int * INTERVAL '1 day')
        ),
        top_reasons AS (
            SELECT reason, COUNT(*) AS total
            FROM blocker_rows
            GROUP BY reason
            ORDER BY total DESC, reason ASC
            LIMIT $2
        )
        SELECT blocker_rows.day, blocker_rows.reason, COUNT(*) AS cnt
        FROM blocker_rows
        JOIN top_reasons USING (reason)
        GROUP BY blocker_rows.day, blocker_rows.reason, top_reasons.total
        ORDER BY blocker_rows.day ASC, top_reasons.total DESC, blocker_rows.reason ASC
        """,
        days,
        top_n,
    )
    daily_rows = await pool.fetch(
        """
        SELECT
            DATE_TRUNC('day', created_at)::date::text AS day,
            COALESCE(SUM(
                jsonb_array_length(
                    COALESCE(data_context->'latest_quality_audit'->'blocking_issues', '[]'::jsonb)
                )
            ), 0) AS blocker_total
        FROM blog_posts
        WHERE created_at >= NOW() - ($1::int * INTERVAL '1 day')
        GROUP BY DATE_TRUNC('day', created_at)::date::text
        ORDER BY day ASC
        """,
        days,
    )
    reason_totals: dict[str, int] = {}
    for row in trend_rows:
        reason = str(row["reason"] or "")
        reason_totals[reason] = reason_totals.get(reason, 0) + int(row["cnt"] or 0)

    top_blockers = sorted(
        ({"reason": reason, "count": count} for reason, count in reason_totals.items()),
        key=lambda item: (-item["count"], item["reason"]),
    )
    return {
        "days": int(days),
        "top_n": int(top_n),
        "top_blockers": top_blockers,
        "series": [
            {
                "day": str(row["day"] or ""),
                "reason": str(row["reason"] or ""),
                "count": int(row["cnt"] or 0),
            }
            for row in trend_rows
        ],
        "totals_by_day": [
            {
                "day": str(row["day"] or ""),
                "blocker_total": int(row["blocker_total"] or 0),
            }
            for row in daily_rows
        ],
    }


@router.get("/quality-diagnostics")
async def blog_quality_diagnostics(
    days: int = Query(14, ge=1, le=90),
    top_n: int = Query(10, ge=1, le=50),
    _user: AuthUser = Depends(require_auth),
):
    from ..autonomous.tasks.b2b_blog_post_generation import _blog_slug_block_reason

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not ready")

    where = (
        "WHERE created_at >= NOW() - ($1::int * INTERVAL '1 day') "
        "AND COALESCE(data_context->'latest_quality_audit'->>'status', '') = 'fail'"
    )
    status_rows = await pool.fetch(
        f"""
        SELECT COALESCE(NULLIF(status, ''), 'missing') AS status_value,
               COUNT(*) AS cnt
        FROM blog_posts
        {where}
        GROUP BY 1
        ORDER BY cnt DESC, 1 ASC
        """,
        days,
    )
    boundary_rows = await pool.fetch(
        f"""
        SELECT COALESCE(data_context->'latest_quality_audit'->>'boundary', 'missing') AS boundary,
               COUNT(*) AS cnt
        FROM blog_posts
        {where}
        GROUP BY 1
        ORDER BY cnt DESC, 1 ASC
        LIMIT $2
        """,
        days,
        top_n,
    )
    cause_rows = await pool.fetch(
        f"""
        SELECT COALESCE(
                   data_context->'latest_quality_audit'->'failure_explanation'->>'cause_type',
                   'unknown'
               ) AS cause_type,
               COUNT(*) AS cnt
        FROM blog_posts
        {where}
        GROUP BY 1
        ORDER BY cnt DESC, 1 ASC
        LIMIT $2
        """,
        days,
        top_n,
    )
    blocker_rows = await pool.fetch(
        f"""
        SELECT COALESCE(
                   data_context->'latest_quality_audit'->'failure_explanation'->>'primary_blocker',
                   data_context->'latest_quality_audit'->>'primary_blocker',
                   data_context->'latest_quality_audit'->'blocking_issues'->>0,
                   'unknown'
               ) AS reason,
               COUNT(*) AS cnt
        FROM blog_posts
        {where}
        GROUP BY 1
        ORDER BY cnt DESC, 1 ASC
        LIMIT $2
        """,
        days,
        top_n,
    )
    missing_input_rows = await pool.fetch(
        f"""
        SELECT missing_input.input AS input, COUNT(*) AS cnt
        FROM blog_posts
        JOIN LATERAL jsonb_array_elements_text(
            COALESCE(
                data_context->'latest_quality_audit'->'failure_explanation'->'missing_inputs',
                '[]'::jsonb
            )
        ) AS missing_input(input) ON TRUE
        {where}
        GROUP BY 1
        ORDER BY cnt DESC, 1 ASC
        LIMIT $2
        """,
        days,
        top_n,
    )
    topic_rows = await pool.fetch(
        f"""
        SELECT COALESCE(topic_type, 'unknown') AS topic_type, COUNT(*) AS cnt
        FROM blog_posts
        {where}
        GROUP BY 1
        ORDER BY cnt DESC, 1 ASC
        LIMIT $2
        """,
        days,
        top_n,
    )
    subject_rows = await pool.fetch(
        f"""
        SELECT COALESCE(
                   NULLIF(
                       TRIM(
                           COALESCE(
                               data_context->>'vendor',
                               data_context->>'vendor_a',
                               data_context->>'vendor_name',
                               data_context->>'category',
                               ''
                           )
                       ),
                       ''
                   ),
                   'unknown'
               ) AS subject,
               COUNT(*) AS cnt
        FROM blog_posts
        {where}
        GROUP BY 1
        ORDER BY cnt DESC, 1 ASC
        LIMIT $2
        """,
        days,
        top_n,
    )
    blocked_slug_rows = await pool.fetch(
        """
        SELECT slug, status, COALESCE(rejection_count, 0) AS rejection_count, rejected_at
        FROM blog_posts
        WHERE status = 'rejected'
        ORDER BY COALESCE(rejected_at, created_at) DESC, slug ASC
        """
    )

    by_status = [
        {"status": str(row["status_value"] or "missing"), "count": int(row["cnt"] or 0)}
        for row in status_rows
    ]
    active_failure_count = sum(item["count"] for item in by_status if item["status"] == "draft")
    rejected_failure_count = sum(item["count"] for item in by_status if item["status"] == "rejected")
    blocked_slugs: list[dict[str, object]] = []
    for row in blocked_slug_rows:
        reason = _blog_slug_block_reason(row)
        if reason not in {"retry_limit", "rejection_cooldown"}:
            continue
        blocked_slugs.append(
            {
                "slug": str(row["slug"] or ""),
                "reason": str(reason),
                "rejection_count": int(row["rejection_count"] or 0),
            }
        )
    retry_limit_blocked_count = sum(1 for item in blocked_slugs if item["reason"] == "retry_limit")
    cooldown_blocked_count = sum(
        1 for item in blocked_slugs if item["reason"] == "rejection_cooldown"
    )
    blocked_slugs.sort(
        key=lambda item: (
            0 if item["reason"] == "retry_limit" else 1,
            -int(item["rejection_count"] or 0),
            str(item["slug"] or ""),
        )
    )

    return {
        "days": int(days),
        "top_n": int(top_n),
        "active_failure_count": int(active_failure_count),
        "rejected_failure_count": int(rejected_failure_count),
        "current_blocked_slug_count": len(blocked_slugs),
        "retry_limit_blocked_slug_count": int(retry_limit_blocked_count),
        "cooldown_blocked_slug_count": int(cooldown_blocked_count),
        "by_status": by_status,
        "by_boundary": [
            {"boundary": str(row["boundary"] or "missing"), "count": int(row["cnt"] or 0)}
            for row in boundary_rows
        ],
        "by_cause_type": [
            {"cause_type": str(row["cause_type"] or "unknown"), "count": int(row["cnt"] or 0)}
            for row in cause_rows
        ],
        "top_primary_blockers": [
            {"reason": str(row["reason"] or "unknown"), "count": int(row["cnt"] or 0)}
            for row in blocker_rows
        ],
        "top_missing_inputs": [
            {"input": str(row["input"] or ""), "count": int(row["cnt"] or 0)}
            for row in missing_input_rows
        ],
        "by_topic_type": [
            {"topic_type": str(row["topic_type"] or "unknown"), "count": int(row["cnt"] or 0)}
            for row in topic_rows
        ],
        "top_subjects": [
            {"subject": str(row["subject"] or "unknown"), "count": int(row["cnt"] or 0)}
            for row in subject_rows
        ],
        "top_blocked_slugs": blocked_slugs[:top_n],
    }


@router.get("/drafts/{draft_id}")
async def get_draft(
    draft_id: UUID,
    _user: AuthUser = Depends(require_auth),
):
    """Get a single blog draft with full content and charts."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not ready")

    row = await pool.fetchrow(
        """
        SELECT blog_posts.*,
               (
                 SELECT COUNT(*)
                 FROM pipeline_visibility_reviews r
                 JOIN pipeline_visibility_events e ON e.id = r.latest_event_id
                 WHERE r.status = 'open'
                   AND e.entity_type = 'blog_post'
                   AND e.entity_id = blog_posts.slug
               ) AS unresolved_issue_count
        FROM blog_posts
        WHERE id = $1
        """,
        draft_id,
    )
    if not row:
        raise HTTPException(404, "Draft not found")
    return _row_to_detail(row)


@router.get("/drafts/{draft_id}/evidence")
async def get_draft_evidence(
    draft_id: UUID,
    limit: int = Query(20, ge=1, le=100),
    _user: AuthUser = Depends(require_auth),
):
    """Return matching b2b_reviews that back the claims in a blog draft."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not ready")

    row = await pool.fetchrow(
        "SELECT data_context, source_report_date FROM blog_posts WHERE id = $1",
        draft_id,
    )
    if not row:
        raise HTTPException(404, "Draft not found")

    ctx = _safe_json(row.get("data_context") or {})
    vendor_name = ctx.get("vendor_name") or ctx.get("vendor_a") or ctx.get("vendor")
    if not vendor_name:
        return {"reviews": [], "count": 0}

    # Build query based on available context
    # Columns: summary (not headline), review_text (not full_text),
    # enrichment JSONB contains urgency_score and pain_category,
    # source (not source_site), reviewed_at (not review_date)
    report_date = row.get("source_report_date")
    # APPROVED-ENRICHMENT-READ: pain_category, urgency_score, reviewer_context.industry
    # Reason: blog review quality aggregation
    if report_date:
        reviews = await pool.fetch(
            """
            SELECT id, vendor_name, reviewer_company, summary, review_text,
                   enrichment->>'pain_category' AS pain_category,
                   (enrichment->>'urgency_score')::numeric AS urgency_score,
                   source, reviewed_at, reviewer_title, company_size_raw,
                   COALESCE(reviewer_industry, enrichment->'reviewer_context'->>'industry') AS industry
            FROM b2b_reviews
            WHERE LOWER(vendor_name) = LOWER($1)
              AND enrichment_status = 'enriched'
              AND reviewed_at >= ($2::date - INTERVAL '90 days')
            ORDER BY (enrichment->>'urgency_score')::numeric DESC NULLS LAST
            LIMIT $3
            """,
            vendor_name, report_date, limit,
        )
    else:
        # APPROVED-ENRICHMENT-READ: pain_category, urgency_score, reviewer_context.industry
        # Reason: blog review quality aggregation
        reviews = await pool.fetch(
            """
            SELECT id, vendor_name, reviewer_company, summary, review_text,
                   enrichment->>'pain_category' AS pain_category,
                   (enrichment->>'urgency_score')::numeric AS urgency_score,
                   source, reviewed_at, reviewer_title, company_size_raw,
                   COALESCE(reviewer_industry, enrichment->'reviewer_context'->>'industry') AS industry
            FROM b2b_reviews
            WHERE LOWER(vendor_name) = LOWER($1)
              AND enrichment_status = 'enriched'
            ORDER BY (enrichment->>'urgency_score')::numeric DESC NULLS LAST
            LIMIT $2
            """,
            vendor_name, limit,
        )

    result = []
    for r in reviews:
        pain = r.get("pain_category")
        result.append({
            "id": str(r["id"]),
            "vendor_name": r["vendor_name"],
            "reviewer_company": r.get("reviewer_company"),
            "reviewer_title": r.get("reviewer_title"),
            "company_size": r.get("company_size_raw"),
            "industry": r.get("industry"),
            "headline": r.get("summary"),
            "full_text": r.get("review_text"),
            "pain_categories": [pain] if pain else [],
            "urgency_score": float(r["urgency_score"]) if r.get("urgency_score") is not None else None,
            "source_site": r.get("source"),
            "review_date": r["reviewed_at"].isoformat() if r.get("reviewed_at") else None,
        })

    return {"reviews": result, "count": len(result)}


@router.patch("/drafts/{draft_id}")
async def update_draft(
    draft_id: UUID,
    patch: BlogDraftPatch,
    _user: AuthUser = Depends(require_auth),
):
    """Edit draft title, content, charts, tags, status, or reviewer notes."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not ready")

    existing = await pool.fetchrow("SELECT id FROM blog_posts WHERE id = $1", draft_id)
    if not existing:
        raise HTTPException(404, "Draft not found")

    updates = []
    args = []
    idx = 1

    for field_name, column in [
        ("title", "title"),
        ("description", "description"),
        ("content", "content"),
        ("status", "status"),
        ("reviewer_notes", "reviewer_notes"),
        ("seo_title", "seo_title"),
        ("seo_description", "seo_description"),
        ("target_keyword", "target_keyword"),
    ]:
        val = getattr(patch, field_name, None)
        if val is not None:
            updates.append(f"{column} = ${idx}")
            args.append(val)
            idx += 1

    for json_field, json_column in [
        ("charts", "charts"),
        ("tags", "tags"),
        ("secondary_keywords", "secondary_keywords"),
        ("faq", "faq"),
        ("related_slugs", "related_slugs"),
    ]:
        val = getattr(patch, json_field, None)
        if val is not None:
            updates.append(f"{json_column} = ${idx}")
            args.append(json.dumps(val))
            idx += 1

    if not updates:
        raise HTTPException(400, "No fields to update")

    args.append(draft_id)
    query = f"UPDATE blog_posts SET {', '.join(updates)} WHERE id = ${idx} RETURNING id"
    await pool.fetchrow(query, *args)
    return {"ok": True, "id": str(draft_id)}


@router.post("/drafts/{draft_id}/publish")
async def publish_draft(
    draft_id: UUID,
    _user: AuthUser = Depends(require_auth),
):
    """Publish a draft: set status=published, set published_at, optionally write TS file."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not ready")

    row = await pool.fetchrow("SELECT * FROM blog_posts WHERE id = $1", draft_id)
    if not row:
        raise HTTPException(404, "Draft not found")

    if row["status"] == "published":
        raise HTTPException(400, "Already published")

    from ..config import settings

    if settings.b2b_churn.blog_publish_revalidate_enabled:
        result = _publish_revalidation_result(row)
        report = result["audit"]
        projection = blog_quality_projection(report, boundary="publish")
        if report.get("status") != "pass":
            blocking_issues = list(report.get("blocking_issues") or [])
            warnings = list(report.get("warnings") or [])
            summary = projection["error_summary"] or "publish_revalidation_failed"
            await pool.execute(
                """
                UPDATE blog_posts
                SET data_context = $1::jsonb,
                    latest_failure_step = $2,
                    latest_error_code = $3,
                    latest_error_summary = $4,
                    quality_score = $5,
                    quality_threshold = $6,
                    blocker_count = $7,
                    warning_count = $8
                WHERE id = $9
                """,
                json.dumps(result["data_context"], default=str),
                projection["failure_step"],
                projection["error_code"],
                summary,
                projection["score"],
                projection["threshold"],
                projection["blocker_count"],
                projection["warning_count"],
                draft_id,
            )
            await record_attempt(
                pool,
                artifact_type="blog_post",
                artifact_id=row["slug"],
                attempt_no=1,
                stage="publish_validation",
                status="rejected",
                score=report.get("score"),
                threshold=report.get("threshold"),
                blocker_count=len(blocking_issues),
                warning_count=len(warnings),
                blocking_issues=blocking_issues,
                warnings=warnings,
                failure_step="publish_validation",
                error_message=summary,
            )
            await emit_event(
                pool,
                stage="blog",
                event_type="publish_revalidation_blocked",
                entity_type="blog_post",
                entity_id=row["slug"],
                summary=f"Blog publish blocked for {row['slug']}",
                severity="warning",
                actionable=True,
                artifact_type="blog_post",
                reason_code=(blocking_issues[0] if blocking_issues else "publish_revalidation_failed")[:80],
                decision="blocked",
                detail=report,
            )
            raise HTTPException(409, summary)
    else:
        result = _publish_revalidation_result(row)
        report = result["audit"]
        projection = blog_quality_projection(report, boundary="publish")

    now = datetime.now(timezone.utc)
    await pool.execute(
        """
        UPDATE blog_posts
        SET data_context = $1::jsonb,
            status = 'published',
            published_at = $2,
            rejected_at = NULL,
            rejection_reason = NULL,
            latest_failure_step = $3,
            latest_error_code = $4,
            latest_error_summary = $5,
            quality_score = $6,
            quality_threshold = $7,
            blocker_count = $8,
            warning_count = $9
        WHERE id = $10
        """,
        json.dumps(result["data_context"], default=str),
        now,
        projection["failure_step"],
        projection["error_code"],
        projection["error_summary"],
        projection["score"],
        projection["threshold"],
        projection["blocker_count"],
        projection["warning_count"],
        draft_id,
    )
    await record_attempt(
        pool,
        artifact_type="blog_post",
        artifact_id=row["slug"],
        attempt_no=1,
        stage="publish_validation",
        status="succeeded",
    )

    # Optionally write the TS file to the blog content directory
    topic_type = row.get("topic_type", "")
    b2b_types = (
        "vendor_alternative", "vendor_showdown", "churn_report", "migration_guide",
        "vendor_deep_dive", "market_landscape", "pricing_reality_check",
        "switching_story", "pain_point_roundup", "best_fit_guide",
    )
    if topic_type in b2b_types:
        ui_path = settings.b2b_churn.blog_post_ui_path
    else:
        ui_path = settings.external_data.blog_post_ui_path
    ts_path = None
    deployed = None
    if topic_type in b2b_types:
        deploy_cfg = settings.b2b_churn
    else:
        deploy_cfg = settings.external_data
    if ui_path and os.path.isdir(ui_path):
        ts_path = _write_blog_ts_file(row, ui_path, now)
        if ts_path:
            try:
                from ..autonomous.tasks._blog_deploy import auto_deploy_blog
                deployed = await auto_deploy_blog(
                    ui_path,
                    row["slug"],
                    enabled=deploy_cfg.blog_auto_deploy_enabled,
                    branch=deploy_cfg.blog_auto_deploy_branch,
                    hook_url=deploy_cfg.blog_auto_deploy_hook_url,
                )
            except Exception:
                logger.warning("Blog auto-deploy failed", exc_info=True)
            
            # Fire the Vercel CLI directly from the UI root directory
            import subprocess
            ui_root = ui_path.split("/src/")[0]
            try:
                # Log that we are attempting the CLI approach
                logger.info(f"Triggering direct Vercel CLI deployment in {ui_root}")
                subprocess.Popen(
                    ["vercel", "--prod", "--yes"], 
                    cwd=ui_root
                )
                if not deployed:
                    deployed = {}
                deployed["vercel_cli"] = "triggered"
            except Exception as e:
                logger.error(f"Failed to trigger Vercel CLI deployment: {e}")

    return {
        "ok": True,
        "id": str(draft_id),
        "slug": row["slug"],
        "published_at": now.isoformat(),
        "ts_file": ts_path,
        "deploy": deployed,
    }


@router.post("/generate")
async def generate_post(
    req: ManualGenerateRequest,
    _user: AuthUser = Depends(require_auth),
):
    """Manually trigger blog post generation for a specific vendor + topic type."""
    from ..autonomous.tasks.b2b_blog_post_generation import (
        _KNOWN_TOPIC_TYPES,
        _check_data_sufficiency,
        _check_blueprint_sufficiency,
        _gather_data,
        _load_pool_layers_for_blog,
        _build_blueprint,
        _generate_content_async,
        _enforce_blog_quality_async,
        _assemble_and_store,
        _get_blog_slug_block_reason,
        _persist_first_pass_blog_quality,
        _upsert_blog_post_state,
        build_manual_topic_ctx,
    )
    from ..config import settings

    if req.topic_type not in _KNOWN_TOPIC_TYPES:
        raise HTTPException(
            422,
            f"Unknown topic_type '{req.topic_type}'. Must be one of: {sorted(_KNOWN_TOPIC_TYPES)}",
        )

    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not ready")

    cfg = settings.b2b_churn
    run_id = str(uuid4())

    # Get LLM
    from ..pipelines.llm import get_pipeline_llm
    llm = get_pipeline_llm(
        workload="synthesis",
        try_openrouter=True,
        auto_activate_ollama=False,
        openrouter_model=cfg.blog_post_openrouter_model,
    )
    if llm is None:
        from ..services import llm_registry
        llm = llm_registry.get_active()
    if llm is None:
        raise HTTPException(503, "No LLM available for blog post generation")

    try:
        topic_ctx = await build_manual_topic_ctx(
            pool, req.vendor_name, req.topic_type,
            vendor_b=req.vendor_b, category=req.category,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))

    data = await _gather_data(pool, req.topic_type, topic_ctx)
    await _load_pool_layers_for_blog(pool, req.topic_type, topic_ctx, data)

    sufficiency = _check_data_sufficiency(req.topic_type, data)
    if not sufficiency["sufficient"] and not req.skip_sufficiency_check:
        raise HTTPException(422, {
            "error": "Insufficient data",
            "reason": sufficiency["reason"],
            "hint": "Set skip_sufficiency_check=true to override",
        })

    blueprint = _build_blueprint(req.topic_type, topic_ctx, data)
    blueprint_sufficiency = _check_blueprint_sufficiency(blueprint)
    if not blueprint_sufficiency["sufficient"] and not req.skip_sufficiency_check:
        raise HTTPException(422, {
            "error": "Insufficient blueprint coverage",
            "reason": blueprint_sufficiency["reason"],
            "hint": "Set skip_sufficiency_check=true to override",
        })
    if not req.force_retry_blocked_slug:
        block_reason = await _get_blog_slug_block_reason(pool, blueprint.slug)
        if block_reason:
            raise HTTPException(
                409,
                {
                    "error": "Blog slug is currently blocked from regeneration",
                    "slug": blueprint.slug,
                    "block_reason": block_reason,
                    "requires_force_retry": True,
                    "retry_limit_reached": block_reason == "retry_limit",
                    "cooldown_active": block_reason == "rejection_cooldown",
                    "max_rejection_retries": int(
                        getattr(cfg, "blog_post_max_rejection_retries", 0) or 0
                    ),
                    "rejection_cooldown_hours": int(
                        getattr(cfg, "blog_post_rejection_cooldown_hours", 0) or 0
                    ),
                },
            )
    content = await _generate_content_async(
        llm,
        blueprint,
        cfg.blog_post_max_tokens,
        run_id=run_id,
    )
    if content is None:
        await _upsert_blog_post_state(
            pool,
            blueprint,
            llm,
            status="failed",
            run_id=run_id,
            attempt_no=1,
            failure_step="llm_call",
            error_code="llm_returned_none",
            error_summary="LLM returned None",
        )
        await record_attempt(
            pool,
            artifact_type="blog_post",
            artifact_id=blueprint.slug,
            run_id=run_id,
            attempt_no=1,
            stage="generation",
            status="failed",
            failure_step="llm_call",
            error_message="LLM returned None",
        )
        await emit_event(
            pool,
            stage="blog",
            event_type="generation_failure",
            entity_type="blog_post",
            entity_id=blueprint.slug,
            summary=f"LLM failed for {blueprint.slug} ({req.topic_type})",
            severity="error",
            actionable=True,
            artifact_type="blog_post",
            run_id=run_id,
            reason_code="llm_returned_none",
        )
        raise HTTPException(500, "LLM content generation failed")

    content, quality_report = await _enforce_blog_quality_async(
        llm,
        blueprint,
        content,
        cfg.blog_post_max_tokens,
        run_id=run_id,
    )
    first_pass_audit = _persist_first_pass_blog_quality(blueprint, quality_report)
    if quality_report.get("_retry_requested"):
        await record_attempt(
            pool,
            artifact_type="blog_post",
            artifact_id=blueprint.slug,
            run_id=run_id,
            attempt_no=1,
            stage="quality_gate_first_pass",
            status="retry_requested",
            score=first_pass_audit.get("score"),
            threshold=first_pass_audit.get("threshold"),
            blocker_count=len(first_pass_audit.get("blocking_issues", [])),
            warning_count=len(first_pass_audit.get("warnings", [])),
            blocking_issues=first_pass_audit.get("blocking_issues"),
            warnings=first_pass_audit.get("warnings"),
            failure_step="quality_gate_first_pass",
            error_message=", ".join(first_pass_audit.get("blocking_issues", [])[:3]) or None,
        )
    if content is None:
        rejected_content = quality_report.get("_rejected_content") if isinstance(quality_report, dict) else None
        audit = blog_quality_revalidation(
            blueprint=blueprint,
            content=rejected_content if isinstance(rejected_content, dict) else None,
            boundary="manual_generate",
            report=quality_report,
        )["audit"]
        await _upsert_blog_post_state(
            pool,
            blueprint,
            llm,
            status="rejected",
            run_id=run_id,
            attempt_no=1,
            score=audit.get("score"),
            threshold=audit.get("threshold"),
            blocker_count=len(audit.get("blocking_issues", [])),
            warning_count=len(audit.get("warnings", [])),
            failure_step="quality_gate",
            error_code="quality_gate_rejection",
            error_summary=", ".join(audit.get("blocking_issues", [])[:3]),
            rejection_reason=", ".join(audit.get("blocking_issues", [])[:3]),
            content=rejected_content if isinstance(rejected_content, dict) else None,
        )
        await record_attempt(
            pool,
            artifact_type="blog_post",
            artifact_id=blueprint.slug,
            run_id=run_id,
            attempt_no=1,
            stage="quality_gate",
            status="rejected",
            score=audit.get("score"),
            threshold=audit.get("threshold"),
            blocker_count=len(audit.get("blocking_issues", [])),
            warning_count=len(audit.get("warnings", [])),
            blocking_issues=audit.get("blocking_issues"),
            warnings=audit.get("warnings"),
            failure_step="quality_gate",
            error_message=", ".join(audit.get("blocking_issues", [])[:3]) or None,
        )
        await emit_event(
            pool,
            stage="blog",
            event_type="quality_gate_rejection",
            entity_type="blog_post",
            entity_id=blueprint.slug,
            summary=f"Quality gate rejected {blueprint.slug} ({req.topic_type})",
            severity="warning",
            actionable=True,
            artifact_type="blog_post",
            run_id=run_id,
            reason_code=(
                audit.get("blocking_issues", ["unknown"])[0][:80]
                if audit.get("blocking_issues")
                else "unknown"
            ),
            decision="rejected",
            detail=audit,
        )
        raise HTTPException(
            422,
            {
                "error": "Generated content failed quality gate",
                "blocking_issues": audit.get("blocking_issues", []),
                "warnings": audit.get("warnings", []),
                "failure_explanation": audit.get("failure_explanation"),
            },
        )
    result = blog_quality_revalidation(
        blueprint=blueprint,
        content=content,
        boundary="manual_generate",
        report=quality_report,
    )
    blueprint.data_context = result["data_context"]

    post_id = await _assemble_and_store(
        pool,
        blueprint,
        content,
        llm,
        run_id=run_id,
        attempt_no=1,
    )
    if not post_id:
        raise HTTPException(409, f"Slug '{blueprint.slug}' already published")

    return {
        "ok": True,
        "post_id": post_id,
        "slug": blueprint.slug,
        "topic_type": req.topic_type,
        "charts": len(blueprint.charts),
        "sufficiency": sufficiency,
        "first_pass_failure_explanation": first_pass_audit.get("failure_explanation"),
    }


def _write_blog_ts_file(row, ui_path: str, published_at: datetime) -> str | None:
    """Write a TypeScript blog post file and update the index."""
    from pathlib import Path
    from ..autonomous.tasks._blog_ts import build_post_ts, update_blog_index

    slug = row["slug"]
    charts = _safe_json(row.get("charts", []))
    tags = _safe_json(row.get("tags", []))
    date_str = published_at.strftime("%Y-%m-%d")

    data_context = _safe_json(row.get("data_context", {}))

    faq = _safe_json(row.get("faq", []))
    secondary_keywords = _safe_json(row.get("secondary_keywords", []))
    related_slugs = _safe_json(row.get("related_slugs", []))
    cta = _safe_json(row.get("cta"))

    var_name, ts_content = build_post_ts(
        slug=slug,
        title=row["title"],
        description=row.get("description", ""),
        date_str=date_str,
        author="Churn Signals Team",
        tags=tags,
        topic_type=row.get("topic_type", ""),
        charts_json=charts,
        content=row["content"],
        data_context=data_context,
        seo_title=row.get("seo_title", ""),
        seo_description=row.get("seo_description", ""),
        target_keyword=row.get("target_keyword", ""),
        secondary_keywords=secondary_keywords,
        faq=faq,
        related_slugs=related_slugs,
        cta=cta if isinstance(cta, dict) else None,
    )

    file_path = os.path.join(ui_path, f"{slug}.ts")
    try:
        with open(file_path, "w") as f:
            f.write(ts_content)

        blog_dir = Path(ui_path)
        update_blog_index(blog_dir / "index.ts", slug, var_name)

        logger.info("Wrote blog TS file: %s", file_path)
        return file_path
    except Exception:
        logger.exception("Failed to write blog TS file: %s", file_path)
        return None
