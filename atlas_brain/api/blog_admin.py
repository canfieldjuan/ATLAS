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
    ui_path = settings.external_data.blog_post_ui_path
    ts_path = None
    if ui_path and os.path.isdir(ui_path):
        ts_path = _write_blog_ts_file(row, ui_path, now)

    return {
        "ok": True,
        "id": str(draft_id),
        "slug": row["slug"],
        "published_at": now.isoformat(),
        "ts_file": ts_path,
    }


def _write_blog_ts_file(row, ui_path: str, published_at: datetime) -> str | None:
    """Write a TypeScript blog post file and update the index."""
    slug = row["slug"]
    charts = _safe_json(row.get("charts", []))
    tags = _safe_json(row.get("tags", []))
    content = row["content"]
    date_str = published_at.strftime("%Y-%m-%d")

    # Escape backticks and dollar signs in content for template literals
    escaped_content = content.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

    # Build charts TS literal
    charts_ts = json.dumps(charts, indent=2, default=str) if charts else "[]"
    tags_ts = json.dumps(tags)

    # Variable name from slug
    var_name = slug.replace("-", "_").replace(".", "_")

    ts_content = f"""import type {{ BlogPost }} from './index'

const {var_name}: BlogPost = {{
  slug: '{slug}',
  title: {json.dumps(row['title'])},
  description: {json.dumps(row.get('description', ''))},
  date: '{date_str}',
  author: 'Atlas Intelligence',
  tags: {tags_ts},
  topic_type: {json.dumps(row['topic_type'])},
  charts: {charts_ts},
  content: `{escaped_content}`,
}}

export default {var_name}
"""

    file_path = os.path.join(ui_path, f"{slug}.ts")
    try:
        with open(file_path, "w") as f:
            f.write(ts_content)

        # Update index.ts to include the new post
        index_path = os.path.join(ui_path, "index.ts")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index_content = f.read()

            import_line = f"import {var_name} from './{slug}'"
            if import_line not in index_content:
                # Add import after the last existing import line
                last_import = index_content.rfind("\nimport ")
                if last_import >= 0:
                    end_of_line = index_content.index("\n", last_import + 1)
                    index_content = (
                        index_content[: end_of_line + 1]
                        + import_line
                        + "\n"
                        + index_content[end_of_line + 1:]
                    )

                # Add to POSTS array (only if not already present)
                posts_section = index_content.split("POSTS")
                if len(posts_section) > 1 and var_name not in posts_section[1]:
                    index_content = index_content.replace(
                        "].sort(",
                        f"  {var_name},\n].sort(",
                    )

                with open(index_path, "w") as f:
                    f.write(index_content)

        logger.info("Wrote blog TS file: %s", file_path)
        return file_path
    except Exception:
        logger.exception("Failed to write blog TS file: %s", file_path)
        return None
