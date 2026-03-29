"""
Shared blog post matching for campaign pipelines.

Fetches published blog posts relevant to a vendor/category/brand,
returning full URLs for injection into campaign selling context.

Pipeline isolation ensures B2B and consumer blog posts never cross-match.
"""

from __future__ import annotations

import json
import logging
import re
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


_WORD_RE = re.compile(r"[a-z0-9]+")

# Maps role keyword fragments → {topic_type: score_boost}.
# Higher boost = stronger preference for that topic type for this role.
_ROLE_TOPIC_BOOSTS: list[tuple[tuple[str, ...], dict[str, int]]] = [
    # Finance, procurement, budget holders → pricing content resonates most
    (("cfo", "finance", "controller", "procurement", "budget", "treasurer", "accounting"),
     {"pricing_reality_check": 3, "churn_report": 1}),
    # Technical buyers → migration feasibility is the key concern
    (("cto", "engineer", "developer", "architect", "devops", "platform", "infrastructure", "technical"),
     {"migration_guide": 3, "vendor_deep_dive": 1}),
    # Operations/IT leadership → churn and market context
    (("operations", "coo", "director of ops", "it director", "it manager"),
     {"churn_report": 2, "market_landscape": 1}),
    # Product/marketing → market landscape and best-fit content
    (("cmo", "marketing", "product", "growth", "demand gen"),
     {"market_landscape": 2, "best_fit_guide": 2}),
    # Sales/BD → competitive comparison assets
    (("sales", "business development", "bd", "revenue", "account executive", "ae"),
     {"vendor_showdown": 2, "vendor_alternative": 2}),
    # C-suite generalists → deep dives and landscape
    (("ceo", "president", "founder", "owner", "vp", "vice president", "svp", "evp"),
     {"vendor_deep_dive": 2, "market_landscape": 2}),
]


def _role_topic_boosts(contact_role: str | None) -> dict[str, int]:
    """Return ``{topic_type: boost}`` for a contact role string."""
    if not contact_role:
        return {}
    role_lower = contact_role.lower()
    boosts: dict[str, int] = {}
    for keywords, topic_boosts in _ROLE_TOPIC_BOOSTS:
        if any(kw in role_lower for kw in keywords):
            for topic, pts in topic_boosts.items():
                boosts[topic] = max(boosts.get(topic, 0), pts)
    return boosts


def _tokenize_text(value: str) -> list[str]:
    return _WORD_RE.findall((value or "").lower())


def _contains_term_as_tokens(text: str, term: str) -> bool:
    """Token-aware contains check to avoid substring false positives."""
    hay = _tokenize_text(text)
    needle = _tokenize_text(term)
    if not hay or not needle:
        return False
    if len(needle) == 1:
        return needle[0] in hay
    for idx in range(0, len(hay) - len(needle) + 1):
        if hay[idx:idx + len(needle)] == needle:
            return True
    return False


def _normalized_post_dedupe_key(
    row: dict[str, Any],
    data_context: dict[str, Any],
    topic_ctx: dict[str, Any],
) -> str:
    """Build a stable dedupe key so draft/published variants do not both surface."""
    topic_type = str(row.get("topic_type") or "").lower()
    if topic_type == "vendor_showdown":
        vendor_a = str(topic_ctx.get("vendor_a") or data_context.get("vendor_a") or "").strip().lower()
        vendor_b = str(topic_ctx.get("vendor_b") or data_context.get("vendor_b") or "").strip().lower()
        if vendor_a and vendor_b:
            ordered = "::".join(sorted((vendor_a, vendor_b)))
            return f"{topic_type}:{ordered}"
    slug = str(row.get("slug") or "").strip().lower()
    slug = re.sub(r"-draft$", "", slug)
    if slug:
        return f"{topic_type}:{slug}"
    return f"{topic_type}:{str(row.get('title') or '').strip().lower()}"


async def fetch_relevant_blog_posts(
    pool,
    *,
    pipeline: str,
    vendor_name: str | None = None,
    category: str | None = None,
    brand_names: list[str] | None = None,
    pain_categories: list[str] | None = None,
    alternative_vendors: list[str] | None = None,
    contact_role: str | None = None,
    include_drafts: bool = False,
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
        pain_categories: Pain categories to match in data_context JSONB
            (e.g. ``["pricing", "support"]``).  Highest-weight signal.
        alternative_vendors: Alternative vendor names the target is
            evaluating.  Matches against ``data_context.vendor_b`` in
            showdown posts.
        contact_role: Contact's job title/role.  Boosts topic types that
            resonate with the buyer persona (e.g. CFO → pricing_reality_check).
        include_drafts: When true, allow ``draft`` posts as a fallback when
            no published matches exist.
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
        statuses = ["published", "draft"] if include_drafts else ["published"]
        rows = await pool.fetch(
            """
            SELECT title, slug, topic_type, tags, data_context, status
            FROM blog_posts
            WHERE status = ANY($2::text[])
              AND topic_type = ANY($1::text[])
            ORDER BY COALESCE(published_at, created_at) DESC
            LIMIT 50
            """,
            list(topic_types),
            statuses,
        )
    except Exception:
        logger.exception("Failed to fetch blog posts for matching")
        return []

    if not rows:
        return []

    # Prepare matching inputs
    cat_lower = (category or "").lower()
    pain_lower = [p.lower() for p in (pain_categories or []) if p]
    alt_lower = [a.lower() for a in (alternative_vendors or []) if a]
    role_boosts = _role_topic_boosts(contact_role)

    # Score each post
    allowed_statuses = set(statuses)
    scored_rows: list[tuple[int, int, int, str, dict[str, Any]]] = []

    for idx, row in enumerate(rows):
        score = 0
        slug = (row["slug"] or "").lower()
        title = (row["title"] or "").lower()
        status = str(row.get("status") or "published").lower()
        if status not in allowed_statuses:
            continue
        tags = row["tags"] or []
        if isinstance(tags, list):
            tags_text = " ".join(str(tag).lower() for tag in tags if tag)
        else:
            tags_text = str(tags).lower()

        # Parse data_context for deep matching
        dc = row.get("data_context")
        if isinstance(dc, str):
            try:
                dc = json.loads(dc)
            except Exception:
                dc = {}
        if not isinstance(dc, dict):
            dc = {}
        tc = dc.get("topic_ctx") if isinstance(dc.get("topic_ctx"), dict) else {}

        # Category match across tags, slug, and title (+2)
        if cat_lower and (cat_lower in tags_text or cat_lower in slug or cat_lower in title):
            score += 2

        # Name match: vendor or brand name appears in tags, slug, or title (+3)
        for term in search_terms:
            if term in tags_text or term in slug or term in title:
                score += 3
                break  # one name match is enough

        # Pain category match: post covers the same pain the campaign targets.
        # Check topic_type, tags, title, slug, and pain-specific data_context
        # fields -- NOT the entire data_context blob (avoids false positives
        # from vendor names or categories containing pain keywords).
        # Scoring: +4 first matched pain, +2 second, +1 each additional --
        # rewards breadth while keeping top-priority pain as the dominant signal.
        if pain_lower:
            pain_haystack_parts = [title, slug, tags_text]
            # Extract pain-specific fields from data_context
            for pain_key in ("pain_distribution", "pain_breakdown",
                             "pain_categories", "top_pain"):
                pv = dc.get(pain_key) or tc.get(pain_key)
                if pv is not None:
                    pain_haystack_parts.append(str(pv).lower())
            # topic_type itself is a pain signal (pricing_reality_check)
            pain_haystack_parts.append(row["topic_type"] or "")
            pain_haystack = " ".join(pain_haystack_parts)
            pain_match_count = 0
            for pain in pain_lower:
                if len(pain) >= 3 and pain in pain_haystack:
                    score += 4 if pain_match_count == 0 else (2 if pain_match_count == 1 else 1)
                    pain_match_count += 1

        # Alternative vendor match: post compares with the vendor being evaluated (+5)
        if alt_lower:
            # Check vendor_b in data_context (showdown posts store the second vendor here)
            vendor_b = str(tc.get("vendor_b") or dc.get("vendor_b") or "").lower()
            alt_haystack = " ".join(filter(None, [title, slug, tags_text]))
            for alt in alt_lower:
                if len(alt) < 3:
                    continue  # skip very short names to avoid substring false positives
                if alt == vendor_b or _contains_term_as_tokens(alt_haystack, alt):
                    score += 5
                    break

        # Role-based topic boost: e.g. CFO contact → pricing_reality_check +3
        if role_boosts:
            topic_type = str(row.get("topic_type") or "").lower()
            score += role_boosts.get(topic_type, 0)

        if score > 0:
            status_rank = 1 if status == "published" else 0
            dedupe_key = _normalized_post_dedupe_key(row, dc, tc)
            scored_rows.append((score, status_rank, -idx, dedupe_key, {
                "title": row["title"],
                "url": f"{base_url}/blog/{row['slug']}",
                "topic_type": row["topic_type"],
            }))

    # Sort by relevance first, then prefer published over draft, then recency.
    scored_rows.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    results: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for _, _, _, dedupe_key, item in scored_rows:
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        results.append(item)
        if len(results) >= limit:
            break
    return results
