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
from uuid import UUID

import markdown as _md

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..auth.dependencies import AuthUser, require_auth
from ..autonomous.visibility import emit_event, record_attempt
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
    }


_md_converter = _md.Markdown(extensions=["tables", "fenced_code", "toc"])


def _render_md(text: str) -> str:
    """Convert markdown to HTML. Resets converter state between calls."""
    _md_converter.reset()
    return _md_converter.convert(text)


def _row_to_detail(row) -> dict:
    raw_content = row["content"] or ""
    return {
        "id": str(row["id"]),
        "slug": row["slug"],
        "title": row["title"],
        "description": row.get("description"),
        "topic_type": row["topic_type"],
        "tags": _safe_json(row.get("tags", [])),
        "content": _render_md(raw_content),
        "charts": _safe_json(row.get("charts", [])),
        "data_context": _safe_json(row.get("data_context")),
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
        "seo_title": row.get("seo_title"),
        "seo_description": row.get("seo_description"),
        "target_keyword": row.get("target_keyword"),
        "secondary_keywords": _safe_json(row.get("secondary_keywords", [])),
        "faq": _safe_json(row.get("faq", [])),
        "related_slugs": _safe_json(row.get("related_slugs", [])),
        "cta": _safe_json(row.get("cta")) if isinstance(_safe_json(row.get("cta")), dict) else None,
    }


def _stored_row_to_blueprint(row):
    from ..autonomous.tasks.b2b_blog_post_generation import ChartSpec, PostBlueprint

    raw_charts = _safe_json(row.get("charts", []))
    charts: list[ChartSpec] = []
    for idx, chart in enumerate(raw_charts if isinstance(raw_charts, list) else []):
        if not isinstance(chart, dict):
            continue
        charts.append(
            ChartSpec(
                chart_id=str(chart.get("chart_id") or f"chart_{idx + 1}"),
                chart_type=str(chart.get("chart_type") or "bar"),
                title=str(chart.get("title") or f"Chart {idx + 1}"),
                data=chart.get("data") if isinstance(chart.get("data"), list) else [],
                config=chart.get("config") if isinstance(chart.get("config"), dict) else {},
            )
        )

    data_context = _safe_json(row.get("data_context"))
    if not isinstance(data_context, dict):
        data_context = {}

    quotable_phrases = data_context.get("quotable_phrases")
    if not isinstance(quotable_phrases, list):
        quotable_phrases = []

    return PostBlueprint(
        topic_type=str(row.get("topic_type") or ""),
        slug=str(row.get("slug") or ""),
        suggested_title=str(row.get("title") or ""),
        tags=_safe_json(row.get("tags", [])) if isinstance(_safe_json(row.get("tags", [])), list) else [],
        data_context=data_context,
        sections=[],
        charts=charts,
        quotable_phrases=quotable_phrases,
        cta=_safe_json(row.get("cta")) if isinstance(_safe_json(row.get("cta")), dict) else None,
    )


def _publish_revalidation_report(row) -> dict:
    from ..autonomous.tasks.b2b_blog_post_generation import (
        _apply_blog_quality_gate,
        _with_unresolved_critical_warnings,
    )

    blueprint = _stored_row_to_blueprint(row)
    _, report = _apply_blog_quality_gate(
        blueprint,
        {
            "title": row.get("title") or blueprint.suggested_title,
            "description": row.get("description") or "",
            "content": row.get("content") or "",
        },
    )
    return _with_unresolved_critical_warnings(report)


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
                   latest_error_code, latest_error_summary,
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
                   latest_error_code, latest_error_summary,
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
        report = _publish_revalidation_report(row)
        if report.get("status") != "pass":
            blocking_issues = list(report.get("blocking_issues") or [])
            warnings = list(report.get("warnings") or [])
            summary = ", ".join(blocking_issues[:3]) or "publish_revalidation_failed"
            await pool.execute(
                """
                UPDATE blog_posts
                SET latest_failure_step = $1,
                    latest_error_code = $2,
                    latest_error_summary = $3,
                    quality_score = $4,
                    quality_threshold = $5,
                    blocker_count = $6,
                    warning_count = $7
                WHERE id = $8
                """,
                "publish_validation",
                "publish_revalidation_failed",
                summary,
                report.get("score"),
                report.get("threshold"),
                len(blocking_issues),
                len(warnings),
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

    now = datetime.now(timezone.utc)
    await pool.execute(
        """
        UPDATE blog_posts
        SET status = 'published',
            published_at = $1,
            rejected_at = NULL,
            rejection_reason = NULL,
            latest_failure_step = NULL,
            latest_error_code = NULL,
            latest_error_summary = NULL
        WHERE id = $2
        """,
        now, draft_id,
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
        _gather_data,
        _build_blueprint,
        _generate_content,
        _enforce_blog_quality,
        _assemble_and_store,
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

    sufficiency = _check_data_sufficiency(req.topic_type, data)
    if not sufficiency["sufficient"] and not req.skip_sufficiency_check:
        raise HTTPException(422, {
            "error": "Insufficient data",
            "reason": sufficiency["reason"],
            "hint": "Set skip_sufficiency_check=true to override",
        })

    blueprint = _build_blueprint(req.topic_type, topic_ctx, data)
    content = _generate_content(llm, blueprint, cfg.blog_post_max_tokens)
    if content is None:
        raise HTTPException(500, "LLM content generation failed")

    content, quality_report = _enforce_blog_quality(
        llm,
        blueprint,
        content,
        cfg.blog_post_max_tokens,
    )
    if content is None:
        raise HTTPException(
            422,
            {
                "error": "Generated content failed quality gate",
                "blocking_issues": quality_report.get("blocking_issues", []),
                "warnings": quality_report.get("warnings", []),
            },
        )
    blueprint.data_context["generation_quality"] = quality_report

    post_id = await _assemble_and_store(pool, blueprint, content, llm)
    if not post_id:
        raise HTTPException(409, f"Slug '{blueprint.slug}' already published")

    return {
        "ok": True,
        "post_id": post_id,
        "slug": blueprint.slug,
        "topic_type": req.topic_type,
        "charts": len(blueprint.charts),
        "sufficiency": sufficiency,
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
