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

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..auth.dependencies import AuthUser, require_auth
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


class BlogDraftPatch(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    charts: Optional[list] = None
    tags: Optional[list] = None
    status: Optional[str] = None
    reviewer_notes: Optional[str] = None


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
    }


def _row_to_detail(row) -> dict:
    return {
        "id": str(row["id"]),
        "slug": row["slug"],
        "title": row["title"],
        "description": row.get("description"),
        "topic_type": row["topic_type"],
        "tags": _safe_json(row.get("tags", [])),
        "content": row["content"],
        "charts": _safe_json(row.get("charts", [])),
        "data_context": _safe_json(row.get("data_context")),
        "status": row["status"],
        "reviewer_notes": row.get("reviewer_notes"),
        "llm_model": row.get("llm_model"),
        "source_report_date": str(row["source_report_date"]) if row.get("source_report_date") else None,
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "published_at": row["published_at"].isoformat() if row.get("published_at") else None,
    }


# -- endpoints ----------------------------------------------------

@router.get("/drafts")
async def list_drafts(
    status: Optional[str] = Query(None, description="Filter by status (draft, published, archived)"),
    limit: int = Query(50, ge=1, le=200),
    _user: AuthUser = Depends(require_auth),
):
    """List blog post drafts, optionally filtered by status."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not ready")

    if status:
        rows = await pool.fetch(
            "SELECT id, slug, title, topic_type, status, llm_model, created_at, published_at "
            "FROM blog_posts WHERE status = $1 ORDER BY created_at DESC LIMIT $2",
            status, limit,
        )
    else:
        rows = await pool.fetch(
            "SELECT id, slug, title, topic_type, status, llm_model, created_at, published_at "
            "FROM blog_posts ORDER BY created_at DESC LIMIT $1",
            limit,
        )
    return [_row_to_summary(r) for r in rows]


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
        "SELECT * FROM blog_posts WHERE id = $1", draft_id,
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
                   source, reviewed_at
            FROM b2b_reviews
            WHERE LOWER(vendor_name) = LOWER($1)
              AND enrichment_status = 'complete'
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
                   source, reviewed_at
            FROM b2b_reviews
            WHERE LOWER(vendor_name) = LOWER($1)
              AND enrichment_status = 'complete'
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
    ]:
        val = getattr(patch, field_name, None)
        if val is not None:
            updates.append(f"{column} = ${idx}")
            args.append(val)
            idx += 1

    if patch.charts is not None:
        updates.append(f"charts = ${idx}")
        args.append(json.dumps(patch.charts))
        idx += 1

    if patch.tags is not None:
        updates.append(f"tags = ${idx}")
        args.append(json.dumps(patch.tags))
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

    now = datetime.now(timezone.utc)
    await pool.execute(
        "UPDATE blog_posts SET status = 'published', published_at = $1 WHERE id = $2",
        now, draft_id,
    )

    # Optionally write the TS file to the blog content directory
    from ..config import settings
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


def _write_blog_ts_file(row, ui_path: str, published_at: datetime) -> str | None:
    """Write a TypeScript blog post file and update the index."""
    from pathlib import Path
    from ..autonomous.tasks._blog_ts import build_post_ts, update_blog_index

    slug = row["slug"]
    charts = _safe_json(row.get("charts", []))
    tags = _safe_json(row.get("tags", []))
    date_str = published_at.strftime("%Y-%m-%d")

    data_context = _safe_json(row.get("data_context", {}))

    var_name, ts_content = build_post_ts(
        slug=slug,
        title=row["title"],
        description=row.get("description", ""),
        date_str=date_str,
        author="Atlas Intelligence",
        tags=tags,
        topic_type=row.get("topic_type", ""),
        charts_json=charts,
        content=row["content"],
        data_context=data_context,
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
