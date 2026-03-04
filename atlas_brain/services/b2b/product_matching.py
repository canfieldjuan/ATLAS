"""
B2B Product Matching Engine: score product profiles against a churning
company's pain points to recommend alternatives.

Pure function, no LLM calls. Called from campaign generation and MCP tools.
"""

import json
import logging
from typing import Any

logger = logging.getLogger("atlas.services.b2b.product_matching")

# Scoring weights (sum = 100)
_WEIGHT_PAIN_ALIGNMENT = 40
_WEIGHT_DISPLACEMENT = 20
_WEIGHT_SIZE_FIT = 15
_WEIGHT_SATISFACTION_DELTA = 15
_WEIGHT_RECOMMEND_RATE = 10

# Severity multipliers for pain alignment scoring
_SEVERITY_WEIGHTS = {
    "primary": 1.0,
    "secondary": 0.6,
    "mentioned": 0.3,
}


def _safe_json(val) -> Any:
    """Safely parse a JSON value."""
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass
    return val if val is not None else {}


def _estimate_size_bucket(company_size: int | None) -> str | None:
    """Map a numeric company size to a profile bucket."""
    if company_size is None:
        return None
    if company_size <= 50:
        return "1-50"
    elif company_size <= 200:
        return "51-200"
    elif company_size <= 1000:
        return "201-1000"
    else:
        return "1000+"


def _score_pain_alignment(
    pain_addressed: dict[str, float],
    pain_categories: list[dict],
) -> float:
    """Score 0-1: how well the candidate addresses the company's pain.

    pain_categories: [{"category": "pricing", "severity": "primary"}, ...]
    pain_addressed: {"pricing": 0.85, ...}
    """
    if not pain_categories:
        return 0.5  # neutral when no pain data

    total_weight = 0.0
    weighted_score = 0.0

    for pain in pain_categories:
        cat = pain.get("category", "")
        severity = pain.get("severity", "mentioned")
        weight = _SEVERITY_WEIGHTS.get(severity, 0.3)

        score = pain_addressed.get(cat, 0.5)  # neutral default
        weighted_score += score * weight
        total_weight += weight

    return weighted_score / total_weight if total_weight > 0 else 0.5


def _score_displacement(
    commonly_switched_from: list[dict],
    churning_from: str,
) -> float:
    """Score 0-1: whether this candidate has evidence of people switching from the churned vendor."""
    if not commonly_switched_from:
        return 0.0

    churning_lower = churning_from.lower()
    for entry in commonly_switched_from:
        vendor = (entry.get("vendor") or "").lower()
        if vendor == churning_lower:
            # Scale by count (more evidence = higher score, cap at 1.0)
            count = entry.get("count", 1)
            return min(1.0, count / 10.0)

    return 0.0


def _score_size_fit(
    typical_company_size: dict[str, float],
    company_size: int | None,
) -> float:
    """Score 0-1: how well the candidate fits this company's size."""
    bucket = _estimate_size_bucket(company_size)
    if bucket is None or not typical_company_size:
        return 0.5  # neutral when no data

    return typical_company_size.get(bucket, 0.0)


def _score_satisfaction_delta(
    candidate_avg_rating: float | None,
    churned_vendor_avg_rating: float | None,
) -> float:
    """Score 0-1: satisfaction improvement from switching.

    Maps rating delta (-4 to +4) to 0-1 range.
    """
    if candidate_avg_rating is None or churned_vendor_avg_rating is None:
        return 0.5

    delta = candidate_avg_rating - churned_vendor_avg_rating
    # Map [-4, +4] to [0, 1] with midpoint at 0.5
    return max(0.0, min(1.0, 0.5 + delta / 8.0))


def _build_reasoning(
    pain_categories: list[dict],
    pain_addressed: dict[str, float],
    switched_from: list[dict],
    churning_from: str,
) -> str:
    """Build a human-readable reasoning string for this match."""
    parts = []

    # Pain alignment
    addressed_pains = []
    for pain in pain_categories:
        cat = pain.get("category", "")
        score = pain_addressed.get(cat, 0)
        if score >= 0.7:
            addressed_pains.append(cat.replace("_", " "))
    if addressed_pains:
        parts.append(f"Addresses pain: {', '.join(addressed_pains[:3])}")

    # Displacement evidence
    churning_lower = churning_from.lower()
    for entry in switched_from:
        if (entry.get("vendor") or "").lower() == churning_lower:
            count = entry.get("count", 0)
            reason = entry.get("top_reason", "")
            msg = f"{count} companies switched from {churning_from}"
            if reason:
                msg += f" (reason: {reason})"
            parts.append(msg)
            break

    if not parts:
        parts.append("Alternative in same category")

    return "; ".join(parts)


async def match_products(
    churning_from: str,
    pain_categories: list[dict],
    company_size: int | None,
    industry: str | None,
    pool,
    limit: int = 3,
) -> list[dict]:
    """Score all product profiles against this company's pain and return top matches.

    Args:
        churning_from: vendor name the company is leaving
        pain_categories: [{"category": "pricing", "severity": "primary"}, ...]
        company_size: number of employees (or None)
        industry: company's industry (or None)
        pool: asyncpg pool
        limit: max results to return

    Returns:
        List of dicts with keys: vendor_name, score, reasoning, profile_summary,
        pain_alignment, displacement_evidence, avg_rating, recommend_rate
    """
    # Fetch all profiles (typically tens, not thousands)
    rows = await pool.fetch(
        """
        SELECT vendor_name, product_category,
               strengths, weaknesses, pain_addressed,
               total_reviews_analyzed, avg_rating, recommend_rate, avg_urgency,
               primary_use_cases, typical_company_size, typical_industries,
               top_integrations, commonly_compared_to, commonly_switched_from,
               profile_summary
        FROM b2b_product_profiles
        ORDER BY total_reviews_analyzed DESC
        """
    )

    if not rows:
        return []

    # Get churned vendor's avg_rating for satisfaction delta
    churned_rating = None
    churning_lower = churning_from.lower()
    for r in rows:
        if r["vendor_name"].lower() == churning_lower:
            churned_rating = float(r["avg_rating"]) if r["avg_rating"] is not None else None
            break

    candidates = []
    for r in rows:
        # Never recommend the vendor they're leaving
        if r["vendor_name"].lower() == churning_lower:
            continue

        pain_addressed = _safe_json(r["pain_addressed"])
        if not isinstance(pain_addressed, dict):
            pain_addressed = {}

        commonly_switched_from = _safe_json(r["commonly_switched_from"])
        if not isinstance(commonly_switched_from, list):
            commonly_switched_from = []

        typical_company_size = _safe_json(r["typical_company_size"])
        if not isinstance(typical_company_size, dict):
            typical_company_size = {}

        candidate_rating = float(r["avg_rating"]) if r["avg_rating"] is not None else None
        recommend = float(r["recommend_rate"]) if r["recommend_rate"] is not None else None

        # Compute individual scores (all 0-1)
        pain_score = _score_pain_alignment(pain_addressed, pain_categories)
        displacement_score = _score_displacement(commonly_switched_from, churning_from)
        size_score = _score_size_fit(typical_company_size, company_size)
        satisfaction_score = _score_satisfaction_delta(candidate_rating, churned_rating)
        recommend_score = recommend if recommend is not None else 0.5

        # Weighted composite (0-100)
        total = (
            pain_score * _WEIGHT_PAIN_ALIGNMENT
            + displacement_score * _WEIGHT_DISPLACEMENT
            + size_score * _WEIGHT_SIZE_FIT
            + satisfaction_score * _WEIGHT_SATISFACTION_DELTA
            + recommend_score * _WEIGHT_RECOMMEND_RATE
        )

        scores = {
            "pain_alignment": round(pain_score, 3),
            "displacement": round(displacement_score, 3),
            "size_fit": round(size_score, 3),
            "satisfaction_delta": round(satisfaction_score, 3),
            "recommend_rate": round(recommend_score, 3),
        }

        reasoning = _build_reasoning(
            pain_categories,
            pain_addressed,
            commonly_switched_from,
            churning_from,
        )

        candidates.append({
            "vendor_name": r["vendor_name"],
            "product_category": r["product_category"],
            "score": round(total, 1),
            "scores": scores,
            "reasoning": reasoning,
            "profile_summary": r["profile_summary"],
            "avg_rating": candidate_rating,
            "recommend_rate": recommend,
            "total_reviews": r["total_reviews_analyzed"],
        })

    # Sort by composite score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:limit]
