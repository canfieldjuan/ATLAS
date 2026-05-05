"""Refresh Amazon seller category intelligence snapshots from review data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from typing import Any


JsonDict = dict[str, Any]

_DEFAULT_ADJACENCY_DIRECTIONS = (
    "used_with",
    "relied_on",
    "tested_with",
    "compatible_with",
)


@dataclass(frozen=True)
class CategoryIntelligenceRefreshResult:
    refreshed: int = 0
    skipped: int = 0
    categories: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "refreshed": self.refreshed,
            "skipped": self.skipped,
            "categories": list(self.categories),
        }


@dataclass(frozen=True)
class CategoryIntelligenceLimits:
    min_brand_reviews: int = 5
    brand_row_limit: int = 20
    brand_health_limit: int = 10
    pain_point_limit: int = 10
    feature_gap_min_mentions: int = 2
    feature_gap_limit: int = 15
    competitive_flow_min_mentions: int = 2
    competitive_flow_limit: int = 15
    safety_signal_limit: int = 10
    manufacturing_insight_limit: int = 10
    affected_asin_limit: int = 5
    root_cause_limit: int = 10
    default_severity: str = "medium"
    rising_health_threshold: float = 70
    falling_health_threshold: float = 40
    safety_penalty_multiplier: float = 10
    health_score_scale: float = 100
    health_score_precision: int = 1
    competitive_adjacency_directions: tuple[str, ...] = _DEFAULT_ADJACENCY_DIRECTIONS


async def refresh_seller_category_intelligence(
    pool: Any,
    *,
    categories: Sequence[str] = (),
    min_reviews: int = 50,
    limit: int = 20,
    reviews_table: str = "product_reviews",
    metadata_table: str = "product_metadata",
    snapshots_table: str = "category_intelligence_snapshots",
    intelligence_limits: CategoryIntelligenceLimits | None = None,
) -> CategoryIntelligenceRefreshResult:
    """Refresh broad Amazon seller category intelligence snapshots."""

    normalized_min_reviews = _normalize_limit(min_reviews, "min_reviews")
    normalized_limit = _normalize_limit(limit, "limit")
    category_names = _clean_sequence(categories)
    if not category_names:
        category_names = await _discover_categories(
            pool,
            min_reviews=normalized_min_reviews,
            limit=normalized_limit,
            reviews_table=reviews_table,
        )
    known_brands = (
        await _fetch_known_brands(pool, _identifier(metadata_table))
        if category_names
        else {}
    )
    refreshed: list[str] = []
    skipped = 0
    for category in category_names:
        snapshot = await aggregate_seller_category_intelligence(
            pool,
            category,
            min_reviews=normalized_min_reviews,
            reviews_table=reviews_table,
            metadata_table=metadata_table,
            intelligence_limits=intelligence_limits,
            known_brands=known_brands,
        )
        if not snapshot:
            skipped += 1
            continue
        await save_seller_category_intelligence_snapshot(
            pool,
            snapshot,
            snapshots_table=snapshots_table,
        )
        refreshed.append(category)
    return CategoryIntelligenceRefreshResult(
        refreshed=len(refreshed),
        skipped=skipped,
        categories=tuple(refreshed),
    )


async def aggregate_seller_category_intelligence(
    pool: Any,
    category: str,
    *,
    min_reviews: int = 50,
    reviews_table: str = "product_reviews",
    metadata_table: str = "product_metadata",
    intelligence_limits: CategoryIntelligenceLimits | None = None,
    known_brands: Mapping[str, str] | None = None,
) -> JsonDict | None:
    """Build one seller category snapshot from product review rows."""

    clean_category = _clean_required(category, "category")
    reviews = _identifier(reviews_table)
    metadata = _identifier(metadata_table)
    stats = await pool.fetchrow(
        f"""
        SELECT
            COUNT(*) AS total_reviews,
            COUNT(DISTINCT pr.asin) AS total_products,
            COUNT(DISTINCT pm.brand) AS total_brands
          FROM {reviews} pr
          LEFT JOIN {metadata} pm ON pm.asin = pr.asin
         WHERE pr.source_category = $1
        """,
        clean_category,
    )
    if not stats or int(stats.get("total_reviews") or 0) < min_reviews:
        return None

    limits = intelligence_limits or CategoryIntelligenceLimits()
    brand_lookup = (
        dict(known_brands)
        if known_brands is not None
        else await _fetch_known_brands(pool, metadata)
    )
    brand_rows = await _fetch_brand_rows(pool, clean_category, reviews, metadata, limits)
    return {
        "category": clean_category,
        "category_stats": {
            "total_reviews": int(stats.get("total_reviews") or 0),
            "total_brands": int(stats.get("total_brands") or 0),
            "total_products": int(stats.get("total_products") or 0),
            "date_range": "all available data",
        },
        "top_pain_points": await _fetch_top_pain_points(
            pool, clean_category, reviews, metadata, limits
        ),
        "feature_gaps": await _fetch_feature_gaps(
            pool, clean_category, reviews, metadata, limits
        ),
        "competitive_flows": await _fetch_competitive_flows(
            pool, clean_category, reviews, metadata, brand_lookup, limits
        ),
        "brand_health": _brand_health(brand_rows, limits),
        "safety_signals": await _fetch_safety_signals(
            pool, clean_category, reviews, metadata, limits
        ),
        "manufacturing_insights": await _fetch_manufacturing_insights(
            pool, clean_category, reviews, limits
        ),
        "top_root_causes": await _fetch_top_root_causes(
            pool, clean_category, reviews, limits
        ),
    }


async def save_seller_category_intelligence_snapshot(
    pool: Any,
    snapshot: Mapping[str, Any],
    *,
    snapshots_table: str = "category_intelligence_snapshots",
) -> None:
    stats = snapshot.get("category_stats") if isinstance(snapshot, Mapping) else {}
    if not isinstance(stats, Mapping):
        stats = {}
    await pool.execute(
        f"""
        INSERT INTO {_identifier(snapshots_table)} (
            category, total_reviews, total_brands, total_products,
            top_pain_points, feature_gaps, competitive_flows,
            brand_health, safety_signals, manufacturing_insights,
            top_root_causes
        ) VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb,
                  $8::jsonb, $9::jsonb, $10::jsonb, $11::jsonb)
        ON CONFLICT (category, COALESCE(subcategory, ''), snapshot_date) DO UPDATE SET
            total_reviews = EXCLUDED.total_reviews,
            total_brands = EXCLUDED.total_brands,
            total_products = EXCLUDED.total_products,
            top_pain_points = EXCLUDED.top_pain_points,
            feature_gaps = EXCLUDED.feature_gaps,
            competitive_flows = EXCLUDED.competitive_flows,
            brand_health = EXCLUDED.brand_health,
            safety_signals = EXCLUDED.safety_signals,
            manufacturing_insights = EXCLUDED.manufacturing_insights,
            top_root_causes = EXCLUDED.top_root_causes
        """,
        _clean_required(snapshot.get("category"), "category"),
        int(stats.get("total_reviews") or 0),
        int(stats.get("total_brands") or 0),
        int(stats.get("total_products") or 0),
        _jsonb(snapshot.get("top_pain_points") or []),
        _jsonb(snapshot.get("feature_gaps") or []),
        _jsonb(snapshot.get("competitive_flows") or []),
        _jsonb(snapshot.get("brand_health") or []),
        _jsonb(snapshot.get("safety_signals") or []),
        _jsonb(snapshot.get("manufacturing_insights") or []),
        _jsonb(snapshot.get("top_root_causes") or []),
    )


async def _discover_categories(
    pool: Any,
    *,
    min_reviews: int,
    limit: int,
    reviews_table: str,
) -> tuple[str, ...]:
    rows = await pool.fetch(
        f"""
        SELECT source_category, COUNT(*) AS review_count
          FROM {_identifier(reviews_table)}
         WHERE source_category IS NOT NULL AND source_category != ''
         GROUP BY source_category
        HAVING COUNT(*) >= $1
         ORDER BY COUNT(*) DESC
         LIMIT $2
        """,
        min_reviews,
        limit,
    )
    return tuple(
        text
        for text in (_clean(_row_dict(row).get("source_category")) for row in rows)
        if text
    )


async def _fetch_brand_rows(
    pool: Any,
    category: str,
    reviews: str,
    metadata: str,
    limits: CategoryIntelligenceLimits,
) -> Sequence[JsonDict]:
    rows = await pool.fetch(
        f"""
        SELECT
            pm.brand,
            COUNT(*) AS total_reviews,
            COUNT(*) FILTER (WHERE pr.would_repurchase IS TRUE) AS repurchase_yes,
            COUNT(*) FILTER (WHERE pr.would_repurchase IS FALSE) AS repurchase_no,
            COUNT(*) FILTER (
                WHERE pr.deep_extraction->'safety_flag'->>'flagged' = 'true'
            ) AS safety_count,
            COUNT(*) FILTER (
                WHERE pr.deep_extraction IS NOT NULL
                  AND pr.deep_extraction != '{{}}'::jsonb
            ) AS deep_count
          FROM {reviews} pr
          JOIN {metadata} pm ON pm.asin = pr.asin
         WHERE pr.source_category = $1
           AND pm.brand IS NOT NULL AND pm.brand != ''
         GROUP BY pm.brand
        HAVING COUNT(*) >= $2
         ORDER BY COUNT(*) DESC
         LIMIT $3
        """,
        category,
        limits.min_brand_reviews,
        limits.brand_row_limit,
    )
    return tuple(_row_dict(row) for row in rows)


async def _fetch_top_pain_points(
    pool: Any,
    category: str,
    reviews: str,
    metadata: str,
    limits: CategoryIntelligenceLimits,
) -> list[JsonDict]:
    rows = await pool.fetch(
        f"""
        SELECT pr.root_cause AS complaint,
               COUNT(*) AS count,
               MAX(pr.severity) AS severity,
               COUNT(DISTINCT pm.brand) AS affected_brands
          FROM {reviews} pr
          LEFT JOIN {metadata} pm ON pm.asin = pr.asin
         WHERE pr.source_category = $1
           AND pr.enrichment_status = 'enriched'
           AND pr.root_cause IS NOT NULL AND pr.root_cause != ''
           AND pr.rating <= 3
         GROUP BY pr.root_cause
         ORDER BY COUNT(*) DESC
         LIMIT $2
        """,
        category,
        limits.pain_point_limit,
    )
    return [
        {
            "complaint": row.get("complaint"),
            "count": row.get("count"),
            "severity": row.get("severity") or limits.default_severity,
            "affected_brands": row.get("affected_brands") or 0,
        }
        for row in (_row_dict(item) for item in rows)
    ]


async def _fetch_feature_gaps(
    pool: Any,
    category: str,
    reviews: str,
    metadata: str,
    limits: CategoryIntelligenceLimits,
) -> list[JsonDict]:
    rows = await pool.fetch(
        f"""
        SELECT req AS request,
               COUNT(*) AS count,
               COUNT(DISTINCT brand) AS brand_count,
               ROUND(AVG(rating), 1) AS avg_rating
          FROM (
            SELECT pr.asin, pr.rating, pm.brand,
                   CASE jsonb_typeof(elem)
                        WHEN 'string' THEN elem #>> '{{}}'
                        WHEN 'object' THEN elem ->> 'request'
                        ELSE elem #>> '{{}}'
                   END AS req
              FROM {reviews} pr
              LEFT JOIN {metadata} pm ON pm.asin = pr.asin,
                   jsonb_array_elements(
                       CASE jsonb_typeof(pr.deep_extraction->'feature_requests')
                            WHEN 'array' THEN pr.deep_extraction->'feature_requests'
                            ELSE '[]'::jsonb
                       END
                   ) AS elem
             WHERE pr.source_category = $1
               AND pr.deep_enrichment_status = 'enriched'
               AND pr.deep_extraction->'feature_requests' IS NOT NULL
          ) sub
         WHERE req IS NOT NULL AND req != '' AND req != 'null'
         GROUP BY req
        HAVING COUNT(*) >= $2
         ORDER BY COUNT(*) DESC
         LIMIT $3
        """,
        category,
        limits.feature_gap_min_mentions,
        limits.feature_gap_limit,
    )
    return [
        {
            "request": row.get("request"),
            "count": row.get("count"),
            "brand_count": row.get("brand_count"),
            "avg_rating": float(row.get("avg_rating") or 0),
        }
        for row in (_row_dict(item) for item in rows)
    ]


async def _fetch_known_brands(pool: Any, metadata: str) -> dict[str, str]:
    rows = await pool.fetch(
        f"""
        SELECT brand
          FROM {metadata}
         WHERE brand IS NOT NULL AND BTRIM(brand) != ''
         GROUP BY brand
        """
    )
    return {
        _brand_key(row.get("brand")): str(row.get("brand")).strip()
        for row in (_row_dict(item) for item in rows)
        if _brand_key(row.get("brand"))
    }


def _normalize_known_brand(value: Any, known_brands: Mapping[str, str]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    exact = known_brands.get(_brand_key(text))
    if exact:
        return exact
    words = text.split()
    for size in range(len(words), 0, -1):
        candidate = " ".join(words[:size])
        matched = known_brands.get(_brand_key(candidate))
        if matched:
            return matched
    return " ".join(words).title()


def _brand_key(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


async def _fetch_competitive_flows(
    pool: Any,
    category: str,
    reviews: str,
    metadata: str,
    known_brands: Mapping[str, str],
    limits: CategoryIntelligenceLimits,
) -> list[JsonDict]:
    rows = await pool.fetch(
        f"""
        SELECT pm.brand AS reviewed_brand,
               comp ->> 'product_name' AS compared_product,
               COALESCE(NULLIF(BTRIM(comp ->> 'direction'), ''), 'compared') AS direction,
               COUNT(*) AS count
          FROM {reviews} pr
          JOIN {metadata} pm ON pm.asin = pr.asin,
               jsonb_array_elements(
                   CASE jsonb_typeof(pr.deep_extraction->'product_comparisons')
                        WHEN 'array' THEN pr.deep_extraction->'product_comparisons'
                        ELSE '[]'::jsonb
                   END
               ) AS comp
         WHERE pr.source_category = $1
           AND pr.deep_enrichment_status = 'enriched'
           AND pr.deep_extraction->'product_comparisons' IS NOT NULL
         GROUP BY pm.brand, comp ->> 'product_name', comp ->> 'direction'
         ORDER BY COUNT(*) DESC
        """,
        category,
    )
    adjacency = {item.strip().lower() for item in limits.competitive_adjacency_directions}
    merged: dict[tuple[str, str, str], JsonDict] = {}
    for row in (_row_dict(item) for item in rows):
        direction = str(row.get("direction") or "compared").strip().lower()
        if direction in adjacency:
            continue
        reviewed_brand = _normalize_known_brand(row.get("reviewed_brand"), known_brands)
        compared_brand = _normalize_known_brand(row.get("compared_product"), known_brands)
        if not reviewed_brand or not compared_brand:
            continue
        if reviewed_brand.lower() == compared_brand.lower():
            continue
        if direction == "switched_from":
            from_brand, to_brand = compared_brand, reviewed_brand
        else:
            from_brand, to_brand = reviewed_brand, compared_brand
        key = (from_brand, to_brand, direction)
        entry = merged.setdefault(
            key,
            {
                "from_brand": from_brand,
                "to_brand": to_brand,
                "direction": direction,
                "count": 0,
            },
        )
        entry["count"] += int(row.get("count") or 0)
    filtered = [
        item
        for item in merged.values()
        if int(item.get("count") or 0) >= limits.competitive_flow_min_mentions
    ]
    return sorted(filtered, key=lambda item: int(item.get("count") or 0), reverse=True)[
        : limits.competitive_flow_limit
    ]


def _brand_health(
    rows: Sequence[Mapping[str, Any]],
    limits: CategoryIntelligenceLimits,
) -> list[JsonDict]:
    brand_health: list[JsonDict] = []
    for row in rows[: limits.brand_health_limit]:
        yes = int(row.get("repurchase_yes") or 0)
        no = int(row.get("repurchase_no") or 0)
        total = yes + no
        repurchase_rate = yes / total if total > 0 else 0.5
        deep = int(row.get("deep_count") or 0)
        safety = int(row.get("safety_count") or 0)
        safety_rate = (
            max(0, 1.0 - (safety / deep) * limits.safety_penalty_multiplier)
            if deep >= limits.min_brand_reviews
            else 1.0
        )
        health_score = round(
            (repurchase_rate + safety_rate) / 2 * limits.health_score_scale,
            limits.health_score_precision,
        )
        brand_health.append({
            "brand": row.get("brand"),
            "health_score": health_score,
            "trend": (
                "rising"
                if health_score >= limits.rising_health_threshold
                else (
                    "falling"
                    if health_score < limits.falling_health_threshold
                    else "stable"
                )
            ),
            "review_count": row.get("total_reviews") or 0,
        })
    return brand_health


async def _fetch_safety_signals(
    pool: Any,
    category: str,
    reviews: str,
    metadata: str,
    limits: CategoryIntelligenceLimits,
) -> list[JsonDict]:
    rows = await pool.fetch(
        f"""
        SELECT
            COALESCE(pm.brand, pr.asin) AS brand,
            pr.deep_extraction->'safety_flag'->>'category' AS category,
            pr.deep_extraction->'safety_flag'->>'description' AS description,
            COUNT(*) AS flagged_count
          FROM {reviews} pr
          LEFT JOIN {metadata} pm ON pm.asin = pr.asin
         WHERE pr.source_category = $1
           AND pr.deep_enrichment_status = 'enriched'
           AND pr.deep_extraction->'safety_flag'->>'flagged' = 'true'
         GROUP BY COALESCE(pm.brand, pr.asin),
                  pr.deep_extraction->'safety_flag'->>'category',
                  pr.deep_extraction->'safety_flag'->>'description'
         ORDER BY COUNT(*) DESC
         LIMIT $2
        """,
        category,
        limits.safety_signal_limit,
    )
    return [
        {
            "brand": row.get("brand"),
            "category": row.get("category") or "",
            "description": row.get("description") or "",
            "flagged_count": row.get("flagged_count"),
        }
        for row in (_row_dict(item) for item in rows)
    ]


async def _fetch_manufacturing_insights(
    pool: Any,
    category: str,
    reviews: str,
    limits: CategoryIntelligenceLimits,
) -> list[JsonDict]:
    rows = await pool.fetch(
        f"""
        SELECT manufacturing_suggestion AS suggestion,
               COUNT(*) AS count,
               ARRAY_AGG(DISTINCT asin) AS affected_asins
          FROM {reviews}
         WHERE source_category = $1
           AND enrichment_status = 'enriched'
           AND actionable_for_manufacturing = TRUE
           AND manufacturing_suggestion IS NOT NULL
           AND manufacturing_suggestion != ''
         GROUP BY manufacturing_suggestion
         ORDER BY COUNT(*) DESC
         LIMIT $2
        """,
        category,
        limits.manufacturing_insight_limit,
    )
    return [
        {
            "suggestion": row.get("suggestion"),
            "count": row.get("count"),
            "affected_asins": list(row.get("affected_asins") or [])[
                : limits.affected_asin_limit
            ],
        }
        for row in (_row_dict(item) for item in rows)
    ]


async def _fetch_top_root_causes(
    pool: Any,
    category: str,
    reviews: str,
    limits: CategoryIntelligenceLimits,
) -> list[JsonDict]:
    rows = await pool.fetch(
        f"""
        SELECT root_cause AS cause, COUNT(*) AS count
          FROM {reviews}
         WHERE source_category = $1
           AND enrichment_status = 'enriched'
           AND root_cause IS NOT NULL AND root_cause != ''
         GROUP BY root_cause
         ORDER BY COUNT(*) DESC
         LIMIT $2
        """,
        category,
        limits.root_cause_limit,
    )
    return [
        {"cause": row.get("cause"), "count": row.get("count")}
        for row in (_row_dict(item) for item in rows)
    ]


def _jsonb(value: Any) -> str:
    return json.dumps(value if value is not None else {}, default=str, separators=(",", ":"))


def _clean_sequence(values: Sequence[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    items: list[str] = []
    for item in (_clean(value) for value in values):
        if not item or item in seen:
            continue
        seen.add(item)
        items.append(item)
    return tuple(items)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _clean_required(value: Any, field_name: str) -> str:
    text = _clean(value)
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _normalize_limit(value: Any, field_name: str) -> int:
    limit = int(value)
    if limit < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return limit


def _row_dict(row: Mapping[str, Any] | Any) -> JsonDict:
    if isinstance(row, Mapping):
        return dict(row)
    try:
        return dict(row)
    except (TypeError, ValueError):
        return {}


def _identifier(value: str) -> str:
    parts = str(value or "").strip().split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid SQL identifier: {value!r}")
    for part in parts:
        if not all(char.isalnum() or char == "_" for char in part):
            raise ValueError(f"invalid SQL identifier: {value!r}")
    return ".".join(f'"{part}"' for part in parts)


__all__ = [
    "CategoryIntelligenceLimits",
    "CategoryIntelligenceRefreshResult",
    "aggregate_seller_category_intelligence",
    "refresh_seller_category_intelligence",
    "save_seller_category_intelligence_snapshot",
]
