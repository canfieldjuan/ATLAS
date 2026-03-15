"""B2B Churn MCP -- content tools."""
import json
import uuid as _uuid
from typing import Optional

from ._shared import _is_uuid, _safe_json, get_pool, logger
from .server import mcp


@mcp.tool()
async def list_blog_posts(
    status: Optional[str] = None,
    topic_type: Optional[str] = None,
    limit: int = 20,
) -> str:
    """
    List generated B2B blog posts.

    status: Filter by status (draft, published)
    topic_type: Filter by topic type (vendor_alternative, vendor_showdown,
                churn_report, migration_guide, vendor_deep_dive,
                market_landscape, pricing_reality_check, switching_story,
                pain_point_roundup, best_fit_guide)
    limit: Maximum results (default 20, cap 50)
    """
    limit = max(1, min(limit, 50))
    try:
        pool = get_pool()
        conditions = []
        params = []
        idx = 1

        if status:
            conditions.append(f"status = ${idx}")
            params.append(status)
            idx += 1

        if topic_type:
            conditions.append(f"topic_type = ${idx}")
            params.append(topic_type)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await pool.fetch(
            f"""
            SELECT id, slug, title, description, topic_type, tags,
                   status, llm_model, created_at, published_at, cta
            FROM blog_posts
            {where}
            ORDER BY created_at DESC
            LIMIT ${idx}
            """,
            *params,
        )

        posts = [
            {
                "id": str(r["id"]),
                "slug": r["slug"],
                "title": r["title"],
                "description": r["description"],
                "topic_type": r["topic_type"],
                "tags": _safe_json(r["tags"]),
                "status": r["status"],
                "llm_model": r["llm_model"],
                "created_at": r["created_at"],
                "published_at": r["published_at"],
                "cta": _safe_json(r["cta"]),
            }
            for r in rows
        ]

        return json.dumps({"posts": posts, "count": len(posts)}, default=str)
    except Exception:
        logger.exception("list_blog_posts error")
        return json.dumps({"error": "Internal error", "posts": [], "count": 0})


@mcp.tool()
async def get_blog_post(
    post_id: Optional[str] = None,
    slug: Optional[str] = None,
) -> str:
    """
    Fetch a full blog post by UUID or slug.

    post_id: UUID of the blog post (optional if slug provided)
    slug: URL slug of the blog post (optional if post_id provided)
    """
    if not post_id and not slug:
        return json.dumps({"success": False, "error": "Provide either post_id or slug"})

    try:
        pool = get_pool()

        if post_id and _is_uuid(post_id):
            row = await pool.fetchrow(
                "SELECT * FROM blog_posts WHERE id = $1",
                _uuid.UUID(post_id),
            )
        elif slug:
            row = await pool.fetchrow(
                "SELECT * FROM blog_posts WHERE slug = $1",
                slug.strip(),
            )
        else:
            return json.dumps({"success": False, "error": "Invalid post_id (must be UUID) or provide slug"})

        if not row:
            return json.dumps({"success": False, "error": "Blog post not found"})

        post = {
            "id": str(row["id"]),
            "slug": row["slug"],
            "title": row["title"],
            "description": row["description"],
            "topic_type": row["topic_type"],
            "tags": _safe_json(row["tags"]),
            "content": row["content"],
            "charts": _safe_json(row["charts"]),
            "data_context": _safe_json(row["data_context"]),
            "status": row["status"],
            "reviewer_notes": row["reviewer_notes"],
            "llm_model": row["llm_model"],
            "created_at": row["created_at"],
            "published_at": row["published_at"],
            "cta": _safe_json(row["cta"]),
        }

        return json.dumps({"success": True, "post": post}, default=str)
    except Exception:
        logger.exception("get_blog_post error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def list_affiliate_partners(
    category: Optional[str] = None,
    enabled_only: bool = True,
    limit: int = 20,
) -> str:
    """
    List affiliate partner configurations.

    category: Filter by product category (partial match, case-insensitive)
    enabled_only: Only show enabled partners (default true)
    limit: Maximum results (default 20, cap 50)
    """
    limit = max(1, min(limit, 50))
    try:
        pool = get_pool()
        conditions = []
        params = []
        idx = 1

        if enabled_only:
            conditions.append("enabled = true")

        if category:
            conditions.append(f"category ILIKE '%' || ${idx} || '%'")
            params.append(category)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await pool.fetch(
            f"""
            SELECT id, name, product_name, product_aliases, category,
                   affiliate_url, commission_type, commission_value,
                   enabled, created_at
            FROM affiliate_partners
            {where}
            ORDER BY name ASC
            LIMIT ${idx}
            """,
            *params,
        )

        partners = [
            {
                "id": str(r["id"]),
                "name": r["name"],
                "product_name": r["product_name"],
                "product_aliases": list(r["product_aliases"]) if r["product_aliases"] else [],
                "category": r["category"],
                "affiliate_url": r["affiliate_url"],
                "commission_type": r["commission_type"],
                "commission_value": r["commission_value"],
                "enabled": r["enabled"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

        return json.dumps({"partners": partners, "count": len(partners)}, default=str)
    except Exception:
        logger.exception("list_affiliate_partners error")
        return json.dumps({"error": "Internal error", "partners": [], "count": 0})
