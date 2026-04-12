"""
Public REST endpoints for published blog posts.

No authentication required. Used by the Next.js frontend at build time
(ISR/SSG) to fetch published B2B blog content.
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.params import Param

from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.blog_public")

router = APIRouter(prefix="/blog", tags=["blog-public"])


def _unwrap_param_default(value: object | None) -> object | None:
    if isinstance(value, Param):
        return value.default
    return value


def _clean_optional_text(value: object | None) -> str | None:
    value = _unwrap_param_default(value)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_required_text(value: object | None, field_name: str) -> str:
    text = _clean_optional_text(value)
    if text is None:
        raise HTTPException(422, f"{field_name} is required")
    return text


def _clean_int_query(value: object | None, *, default: int) -> int:
    value = _unwrap_param_default(value)
    if value is None:
        return default
    return int(value)


def _safe_json(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return raw
    return raw


@router.get("/published")
async def list_published_posts(
    topic_type: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """List published blog posts for the public frontend."""
    topic_type = _clean_optional_text(topic_type)
    limit = _clean_int_query(limit, default=50)
    offset = _clean_int_query(offset, default=0)
    pool = get_db_pool()

    filters = ["status = 'published'"]
    params: list[Any] = []
    idx = 1

    if topic_type:
        params.append(topic_type)
        filters.append(f"topic_type = ${idx}")
        idx += 1

    params.append(limit)
    params.append(offset)

    where = " AND ".join(filters)
    rows = await pool.fetch(
        f"""
        SELECT id, slug, title, description, topic_type, tags,
               content, charts, data_context,
               seo_title, seo_description, target_keyword,
               secondary_keywords, faq, related_slugs,
               llm_model, source_report_date,
               published_at, created_at
        FROM blog_posts
        WHERE {where}
        ORDER BY published_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *params,
    )

    total = await pool.fetchval(
        f"SELECT count(*) FROM blog_posts WHERE {where}",
        *(params[:-2]),
    )

    posts = []
    for row in rows:
        posts.append({
            "id": str(row["id"]),
            "slug": row["slug"],
            "title": row["title"],
            "description": row["description"],
            "topic_type": row["topic_type"],
            "tags": _safe_json(row.get("tags", [])),
            "content": row["content"],
            "charts": _safe_json(row.get("charts", [])),
            "data_context": _safe_json(row.get("data_context", {})),
            "seo_title": row.get("seo_title"),
            "seo_description": row.get("seo_description"),
            "target_keyword": row.get("target_keyword"),
            "secondary_keywords": _safe_json(row.get("secondary_keywords", [])),
            "faq": _safe_json(row.get("faq", [])),
            "related_slugs": _safe_json(row.get("related_slugs", [])),
            "date": str(row.get("published_at") or row.get("created_at") or ""),
            "author": "Churn Signals",
        })

    return {"posts": posts, "total": total}


@router.get("/published/{slug}")
async def get_published_post(slug: str) -> dict:
    """Get a single published blog post by slug."""
    slug = _clean_required_text(slug, "slug")
    pool = get_db_pool()
    row = await pool.fetchrow(
        """
        SELECT id, slug, title, description, topic_type, tags,
               content, charts, data_context,
               seo_title, seo_description, target_keyword,
               secondary_keywords, faq, related_slugs,
               llm_model, source_report_date,
               published_at, created_at
        FROM blog_posts
        WHERE slug = $1 AND status = 'published'
        """,
        slug,
    )
    if not row:
        return {"post": None}

    return {
        "post": {
            "id": str(row["id"]),
            "slug": row["slug"],
            "title": row["title"],
            "description": row["description"],
            "topic_type": row["topic_type"],
            "tags": _safe_json(row.get("tags", [])),
            "content": row["content"],
            "charts": _safe_json(row.get("charts", [])),
            "data_context": _safe_json(row.get("data_context", {})),
            "seo_title": row.get("seo_title"),
            "seo_description": row.get("seo_description"),
            "target_keyword": row.get("target_keyword"),
            "secondary_keywords": _safe_json(row.get("secondary_keywords", [])),
            "faq": _safe_json(row.get("faq", [])),
            "related_slugs": _safe_json(row.get("related_slugs", [])),
            "date": str(row.get("published_at") or row.get("created_at") or ""),
            "author": "Churn Signals",
        },
    }
