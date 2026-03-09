"""
Shared blog post matching for campaign pipelines.

Fetches published blog posts relevant to a vendor/category/brand,
returning full URLs for injection into campaign selling context.

Pipeline isolation ensures B2B and consumer blog posts never cross-match.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("atlas.autonomous.tasks._blog_matching")

# B2B blog topic types (b2b_blog_post_generation.py)
_B2B_TOPIC_TYPES = (
    "vendor_alternative",
    "vendor_showdown",
    "churn_report",
    "migration_guide",
    "vendor_deep_dive",
    "market_landscape",
    "pricing_reality_check",
    "switching_story",
    "pain_point_roundup",
    "best_fit_guide",
)

# Consumer blog topic types (blog_post_generation.py)
_CONSUMER_TOPIC_TYPES = (
    "brand_showdown",
    "complaint_roundup",
    "migration_report",
    "safety_spotlight",
)

_PIPELINE_TOPIC_TYPES = {
    "b2b": _B2B_TOPIC_TYPES,
    "consumer": _CONSUMER_TOPIC_TYPES,
}


async def fetch_relevant_blog_posts(
    pool,
    *,
    pipeline: str,
    vendor_name: str | None = None,
    category: str | None = None,
    brand_names: list[str] | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Fetch published blog posts relevant to a campaign target.

    Args:
        pool: asyncpg connection pool.
        pipeline: ``"b2b"`` or ``"consumer"`` -- restricts to that pipeline's
            topic types so B2B posts never appear in consumer emails and vice
            versa.
        vendor_name: Vendor/company name to match in title/slug.
        category: Product category to match in ``tags->>0``.
        brand_names: Additional brand names to match in title/slug.
        limit: Max posts to return (default 3).

    Returns:
        List of ``{title, url, topic_type}`` sorted by relevance then recency.
        Empty list on no matches or errors -- campaigns proceed without links.
    """
    topic_types = _PIPELINE_TOPIC_TYPES.get(pipeline)
    if not topic_types:
        logger.warning("Unknown pipeline %r, returning empty blog list", pipeline)
        return []

    # Resolve base URL from config
    from ...config import settings

    if pipeline == "b2b":
        base_url = settings.b2b_churn.blog_base_url.rstrip("/")
    else:
        base_url = settings.external_data.blog_base_url.rstrip("/")

    # Build search terms for name matching
    search_terms: list[str] = []
    if vendor_name:
        search_terms.append(vendor_name.lower())
    if brand_names:
        search_terms.extend(b.lower() for b in brand_names if b)

    try:
        rows = await pool.fetch(
            """
            SELECT title, slug, topic_type, tags
            FROM blog_posts
            WHERE status = 'published'
              AND topic_type = ANY($1::text[])
            ORDER BY published_at DESC
            LIMIT 50
            """,
            list(topic_types),
        )
    except Exception:
        logger.exception("Failed to fetch blog posts for matching")
        return []

    if not rows:
        return []

    # Score each post
    scored: list[tuple[int, str, dict]] = []
    cat_lower = (category or "").lower()

    for row in rows:
        score = 0
        slug = (row["slug"] or "").lower()
        title = (row["title"] or "").lower()
        tags = row["tags"] or []
        if isinstance(tags, list):
            tags_text = " ".join(str(tag).lower() for tag in tags if tag)
        else:
            tags_text = str(tags).lower()

        # Category match across tags, slug, and title.
        if cat_lower and (cat_lower in tags_text or cat_lower in slug or cat_lower in title):
            score += 2

        # Name match: vendor or brand name appears in tags, slug, or title.
        for term in search_terms:
            if term in tags_text or term in slug or term in title:
                score += 3
                break  # one name match is enough

        if score > 0:
            scored.append((score, row["slug"], {
                "title": row["title"],
                "url": f"{base_url}/blog/{row['slug']}",
                "topic_type": row["topic_type"],
            }))

    # Sort by score desc, then by position in original query (recency)
    scored.sort(key=lambda t: -t[0])
    return [item for _, _, item in scored[:limit]]
