"""Weekly ecosystem (category-level) analysis for Tier 3 + Tier 4 cache.

Runs all-category analysis via EcosystemAnalyzer and caches results
in reasoning_semantic_cache as two distinct tiers per category:

    T4 (market_dynamics): market structure, HHI, displacement intensity
    T3 (category_pattern): archetype distribution, pain patterns, vendor positions
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.tasks.ecosystem_analysis")


def _split_tier_evidence(evidence: dict[str, Any]) -> tuple[dict, dict]:
    """Split full ecosystem evidence into T4 (market dynamics) and T3 (category patterns)."""
    t4 = {
        "category": evidence.get("category", ""),
        "vendor_count": evidence.get("vendor_count", 0),
        "total_reviews": evidence.get("total_reviews", 0),
        "hhi": evidence.get("hhi", 0),
        "displacement_intensity": evidence.get("displacement_intensity", 0),
        "market_structure": evidence.get("market_structure", ""),
    }
    t3 = {
        "category": evidence.get("category", ""),
        "vendor_count": evidence.get("vendor_count", 0),
        "avg_churn_density": evidence.get("avg_churn_density", 0),
        "avg_urgency": evidence.get("avg_urgency", 0),
        "avg_positive_pct": evidence.get("avg_positive_pct", 0),
        "dominant_archetype": evidence.get("dominant_archetype", ""),
        "archetype_distribution": evidence.get("archetype_distribution", {}),
        "category_pains": evidence.get("category_pains", {}),
        "vendor_positions": evidence.get("vendor_positions", []),
        "top_flows": evidence.get("top_flows", []),
    }
    return t4, t3


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Analyze all B2B product categories and cache ecosystem evidence."""
    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    from atlas_brain.reasoning.ecosystem import EcosystemAnalyzer

    analyzer = EcosystemAnalyzer(pool)
    results = await analyzer.analyze_all_categories()

    if not results:
        return {"_skip_synthesis": "No categories with data for ecosystem analysis"}

    cached_t4 = 0
    cached_t3 = 0
    now = datetime.now(timezone.utc)

    for category, eco in results.items():
        evidence_dict = EcosystemAnalyzer.to_evidence_dict(eco)
        t4_evidence, t3_evidence = _split_tier_evidence(evidence_dict)
        cat_key = category.lower().replace(" ", "_")

        # T4: Market Dynamics (quarterly scope)
        try:
            await pool.execute(
                """
                INSERT INTO reasoning_semantic_cache (
                    vendor_name, product_category, pattern_sig,
                    pattern_class, conclusion_type, confidence,
                    conclusion, created_at, last_validated_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $8
                )
                ON CONFLICT (pattern_sig) DO UPDATE SET
                    conclusion = EXCLUDED.conclusion,
                    confidence = EXCLUDED.confidence,
                    last_validated_at = EXCLUDED.last_validated_at,
                    invalidated_at = NULL
                """,
                "__ecosystem__",
                category,
                f"t4:market:{cat_key}",
                eco.health.market_structure or "stable",
                "market_dynamics",
                0.8,
                json.dumps(t4_evidence, default=str),
                now,
            )
            cached_t4 += 1
        except Exception:
            logger.warning("Failed to cache T4 for %s", category, exc_info=True)

        # T3: Category Patterns (monthly scope)
        try:
            await pool.execute(
                """
                INSERT INTO reasoning_semantic_cache (
                    vendor_name, product_category, pattern_sig,
                    pattern_class, conclusion_type, confidence,
                    conclusion, created_at, last_validated_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $8
                )
                ON CONFLICT (pattern_sig) DO UPDATE SET
                    conclusion = EXCLUDED.conclusion,
                    confidence = EXCLUDED.confidence,
                    last_validated_at = EXCLUDED.last_validated_at,
                    invalidated_at = NULL
                """,
                "__ecosystem__",
                category,
                f"t3:category:{cat_key}",
                eco.health.dominant_archetype or "mixed",
                "category_pattern",
                0.75,
                json.dumps(t3_evidence, default=str),
                now,
            )
            cached_t3 += 1
        except Exception:
            logger.warning("Failed to cache T3 for %s", category, exc_info=True)

    logger.info(
        "Ecosystem analysis complete: %d categories analyzed, %d T4 + %d T3 cached",
        len(results), cached_t4, cached_t3,
    )

    return {
        "_skip_synthesis": "Ecosystem analysis complete",
        "categories_analyzed": len(results),
        "t4_cached": cached_t4,
        "t3_cached": cached_t3,
        "market_structures": {
            cat: eco.health.market_structure for cat, eco in results.items()
        },
    }
