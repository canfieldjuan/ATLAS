"""
Centralized product comparison normalization for competitive flow analysis.

Extracts, normalizes, and aggregates product_comparisons JSONB data from
product_reviews into clean competitive flow edges with proper direction
semantics and brand-aware collapse.

Used by: subcategory_intelligence, amazon_seller_campaign_generation,
competitive_intelligence, blog_post_generation, consumer_dashboard.
"""

import re
import logging
from typing import Any

from ..config import settings

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


def _comparison_cfg():
    return settings.comparison_normalization


def _parse_csv_config(raw: str) -> list[str]:
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _contains_suspicious_term(value: str, terms: list[str]) -> bool:
    for term in terms:
        if not term:
            continue
        if " " in term:
            if term in value:
                return True
            continue
        if len(term) >= 4 and term in value:
            return True
        pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
        if re.search(pattern, value):
            return True
    return False


def is_trusted_known_brand(raw: str, product_count: int | None = None) -> bool:
    """Return True when a canonical brand value is safe to trust."""
    cleaned = " ".join(raw.strip().split())
    if not cleaned:
        return False

    lowered = cleaned.lower()
    cfg = _comparison_cfg()
    invalid_known_brands = _parse_csv_config(cfg.invalid_known_brands)
    suspicious_singleton_terms = _parse_csv_config(cfg.suspicious_singleton_terms)

    if lowered in COMPETITOR_NOISE:
        return False
    if lowered in invalid_known_brands:
        return False
    if cleaned.isdigit():
        return False
    if not any(ch.isalpha() for ch in cleaned):
        return False
    if len(cleaned) > cfg.known_brand_max_length:
        return False
    if len(cleaned.split()) > cfg.known_brand_max_words:
        return False

    if product_count is not None and product_count <= cfg.suspicious_singleton_max_products:
        if any(ch in cleaned for ch in cfg.suspicious_singleton_chars):
            return False
        if any(ch.isdigit() for ch in cleaned) and " " in cleaned:
            return False
        if _contains_suspicious_term(lowered, suspicious_singleton_terms):
            return False

    return True


async def load_known_brands(pool) -> dict[str, str]:
    """Fetch known brands from product_metadata + consumer brand registry.

    Returns {lowercase: canonical_casing}.  Registry aliases override
    product_metadata casing, so ``"kitchen aid"`` maps to ``"KitchenAid"``.
    """
    rows = await pool.fetch(
        "SELECT brand, COUNT(*) AS product_count FROM product_metadata "
        "WHERE brand IS NOT NULL AND brand != '' GROUP BY brand"
    )
    known = {
        r["brand"].lower(): r["brand"]
        for r in rows
        if is_trusted_known_brand(r["brand"], r["product_count"])
    }

    # Overlay consumer brand registry canonical names + aliases
    try:
        reg_rows = await pool.fetch(
            "SELECT canonical_name, aliases FROM consumer_brand_registry"
        )
        for r in reg_rows:
            canonical = r["canonical_name"]
            known[canonical.lower()] = canonical
            aliases = r["aliases"]
            if isinstance(aliases, list):
                for alias in aliases:
                    if isinstance(alias, str) and alias.strip():
                        known[alias.lower()] = canonical
    except Exception:
        pass  # Table may not exist during migration rollout

    return known


def normalize_canonical_brand(raw: str, known_brands: dict[str, str]) -> str | None:
    """Normalize a trusted canonical brand from product_metadata."""
    cleaned = " ".join(raw.strip().split())
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered in known_brands:
        return known_brands[lowered]

    words = cleaned.split()
    for n in range(min(len(words), 4), 0, -1):
        prefix = " ".join(words[:n]).lower()
        if prefix in known_brands:
            return known_brands[prefix]

    return None


def sanitize_metadata_brand(raw: str) -> str:
    """Sanitize a metadata-derived canonical brand before DB insertion."""
    cleaned = " ".join(raw.strip().split())
    if not cleaned:
        return ""
    if not is_trusted_known_brand(cleaned):
        return ""
    return cleaned


def normalize_brand(raw: str, known_brands: dict[str, str]) -> str | None:
    """Normalize a raw product/brand name. Returns None if it should be filtered."""
    raw = " ".join(raw.strip().split())
    if not raw:
        return None

    lowered = raw.lower()
    cfg = _comparison_cfg()
    invalid_known_brands = _parse_csv_config(cfg.invalid_known_brands)

    if lowered in COMPETITOR_NOISE:
        return None
    if lowered in invalid_known_brands:
        return None
    if raw.isdigit():
        return None

    words = raw.split()

    # Filter short names unless they match a known brand (e.g. "LG")
    if len(raw) < 3 and lowered not in known_brands:
        return None

    # Brand-aware collapse: try longest prefix first so multi-word brands
    # like "Hamilton Beach" match before single-word "Hamilton" would.
    # Cap at 4 words — no real brand is longer than that.
    for n in range(min(len(words), 4), 0, -1):
        prefix = " ".join(words[:n]).lower()
        if prefix in known_brands:
            return known_brands[prefix]

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
        normalized_reviewed = normalize_canonical_brand(reviewed_brand, known_brands) if reviewed_brand else None
        if not normalized_reviewed and reviewed_brand and not known_brands:
            normalized_reviewed = normalize_brand(reviewed_brand, known_brands)
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
