"""
Centralized product comparison normalization for competitive flow analysis.

Extracts, normalizes, and aggregates product_comparisons JSONB data from
product_reviews into clean competitive flow edges with proper direction
semantics and brand-aware collapse.

Used by: subcategory_intelligence, amazon_seller_campaign_generation,
competitive_intelligence, blog_post_generation, consumer_dashboard.
"""

import logging
from typing import Any

logger = logging.getLogger("atlas.pipelines.comparisons")

# Directions that indicate adjacency / complementary use, NOT competition
ADJACENCY_DIRECTIONS = {
    "used_with", "relied_on", "tested_with", "compatible_with",
}

# Noise terms that should never appear as brand names in competitive flows
COMPETITOR_NOISE = {
    "amazon", "generic", "no name", "other", "unknown", "n/a", "none",
    "previous model", "old one", "old version", "newer model", "last one",
    "competitor", "different brand", "store brand", "off brand",
    "microsoft vista", "windows", "mac os", "other brand",
    "another product", "cheap one", "other one",
    "2 button version", "button", "cable", "adapter", "charger",
    "brand", "another", "cheap", "remote", "device", "product",
    "unit", "model", "version", "itunes", "youtube", "forum",
    "ebay", "walmart",
}


async def load_known_brands(pool) -> dict[str, str]:
    """Fetch known brands from product_metadata. Returns {lowercase: original_casing}."""
    rows = await pool.fetch(
        "SELECT DISTINCT brand FROM product_metadata "
        "WHERE brand IS NOT NULL AND brand != ''"
    )
    return {
        r["brand"].lower(): r["brand"]
        for r in rows
        if r["brand"].lower() not in COMPETITOR_NOISE
    }


def normalize_brand(raw: str, known_brands: dict[str, str]) -> str | None:
    """Normalize a raw product/brand name. Returns None if it should be filtered."""
    raw = raw.strip()
    if not raw:
        return None

    lowered = raw.lower()

    if lowered in COMPETITOR_NOISE:
        return None
    if raw.isdigit():
        return None

    words = raw.split()
    first_lower = words[0].lower()

    # Filter short names unless they match a known brand (e.g. "LG")
    if len(raw) < 3 and first_lower not in known_brands:
        return None

    # Brand-aware collapse: if first word matches a known brand, normalize
    # to just the brand name (drop model number / spec details)
    if first_lower in known_brands:
        return known_brands[first_lower]

    return raw.title()


def normalize_competitive_flows(
    raw_flows: list[dict[str, Any]],
    known_brands: dict[str, str],
) -> list[dict[str, Any]]:
    """Normalize and aggregate raw competitive flow entries.

    Each entry in raw_flows should have:
        - product_name: the mentioned entity (competitor / other product)
        - direction: comparison direction (default "compared")
        - reviewed_brand: the brand of the product being reviewed (pm.brand)
        - count: pre-aggregated count (default 1)

    Returns list of dicts with from_brand, to_brand, direction, is_directional, count.
    """
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}

    for entry in raw_flows:
        direction = entry.get("direction") or "compared"

        # Skip adjacency relations
        if direction in ADJACENCY_DIRECTIONS:
            continue

        product_name = (entry.get("product_name") or "").strip()
        reviewed_brand = (entry.get("reviewed_brand") or "").strip()

        # Normalize the mentioned entity
        normalized_other = normalize_brand(product_name, known_brands)
        if not normalized_other:
            continue

        # Normalize the reviewed brand too
        normalized_reviewed = normalize_brand(reviewed_brand, known_brands) if reviewed_brand else None
        if not normalized_reviewed:
            continue

        # Resolve from/to based on direction
        if direction == "switched_from":
            from_brand = normalized_other
            to_brand = normalized_reviewed
            is_directional = True
        elif direction == "switched_to":
            from_brand = normalized_reviewed
            to_brand = normalized_other
            is_directional = True
        else:
            # compared, avoided, considered, recommended -- non-directional
            from_brand = normalized_reviewed
            to_brand = normalized_other
            is_directional = False

        # Skip self-references
        if from_brand.lower() == to_brand.lower():
            continue

        count = entry.get("count", 1)
        key = (from_brand, to_brand, direction)
        if key in merged:
            merged[key]["count"] += count
        else:
            merged[key] = {
                "from_brand": from_brand,
                "to_brand": to_brand,
                "direction": direction,
                "is_directional": is_directional,
                "count": count,
            }

    return sorted(merged.values(), key=lambda x: x["count"], reverse=True)


async def fetch_competitive_flows(
    pool,
    *,
    where_clause: str = "TRUE",
    params: list | None = None,
    min_mentions: int = 2,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Shared query + normalization for competitive flow extraction.

    Runs the standard product_comparisons JSONB unnesting query with caller's
    WHERE clause, then normalizes through the full pipeline.

    Args:
        pool: asyncpg connection pool
        where_clause: SQL WHERE fragment (use $1, $2 etc for params)
        params: Query parameters for where_clause
        min_mentions: Minimum mention count to include (applied post-normalization)
        limit: Max number of flows to return
    """
    if params is None:
        params = []

    rows = await pool.fetch(
        f"""
        SELECT
            comp->>'product_name' AS product_name,
            COALESCE(comp->>'direction', 'compared') AS direction,
            pm.brand AS reviewed_brand,
            COUNT(*) AS count
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin,
             jsonb_array_elements(
                 CASE jsonb_typeof(pr.deep_extraction->'product_comparisons')
                      WHEN 'array' THEN pr.deep_extraction->'product_comparisons'
                      ELSE '[]'::jsonb
                 END
             ) AS comp
        WHERE {where_clause}
          AND pr.deep_enrichment_status = 'enriched'
          AND pr.deep_extraction->'product_comparisons' IS NOT NULL
        GROUP BY comp->>'product_name', comp->>'direction', pm.brand
        ORDER BY COUNT(*) DESC
        """,
        *params,
    )

    raw_flows = [
        {
            "product_name": r["product_name"] or "",
            "direction": r["direction"],
            "reviewed_brand": r["reviewed_brand"] or "",
            "count": r["count"],
        }
        for r in rows
    ]

    known_brands = await load_known_brands(pool)
    normalized = normalize_competitive_flows(raw_flows, known_brands)

    # Apply min_mentions filter and limit
    filtered = [f for f in normalized if f["count"] >= min_mentions]
    return filtered[:limit]
