"""Planner helpers for competitive-set scoped synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from ..storage.models import CompetitiveSet


def _norm_vendor(name: str) -> str:
    return str(name or "").strip().lower()


@dataclass
class CompetitiveSetPlan:
    """Concrete vendor and cross-vendor work derived from a competitive set."""

    competitive_set_id: str
    focal_vendor_name: str
    vendor_names: list[str] = field(default_factory=list)
    pairwise_pairs: list[list[str]] = field(default_factory=list)
    category_names: list[str] = field(default_factory=list)
    asymmetry_pairs: list[list[str]] = field(default_factory=list)
    vendor_synthesis_enabled: bool = True
    pairwise_enabled: bool = True
    category_council_enabled: bool = False
    asymmetry_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        vendor_job_count = len(self.vendor_names) if self.vendor_synthesis_enabled else 0
        pairwise_count = len(self.pairwise_pairs) if self.pairwise_enabled else 0
        category_count = len(self.category_names) if self.category_council_enabled else 0
        asymmetry_count = len(self.asymmetry_pairs) if self.asymmetry_enabled else 0
        return {
            "competitive_set_id": self.competitive_set_id,
            "focal_vendor_name": self.focal_vendor_name,
            "vendor_names": self.vendor_names,
            "pairwise_pairs": self.pairwise_pairs,
            "category_names": self.category_names,
            "asymmetry_pairs": self.asymmetry_pairs,
            "vendor_synthesis_enabled": self.vendor_synthesis_enabled,
            "pairwise_enabled": self.pairwise_enabled,
            "category_council_enabled": self.category_council_enabled,
            "asymmetry_enabled": self.asymmetry_enabled,
            "vendor_job_count": vendor_job_count,
            "pairwise_job_count": pairwise_count,
            "category_job_count": category_count,
            "asymmetry_job_count": asymmetry_count,
            "estimated_total_jobs": vendor_job_count + pairwise_count + category_count + asymmetry_count,
        }


def build_competitive_set_plan(
    competitive_set: CompetitiveSet,
    *,
    category_by_vendor: dict[str, str] | None = None,
) -> CompetitiveSetPlan:
    """Expand a competitive set into explicit vendor and cross-vendor jobs."""
    focal = str(competitive_set.focal_vendor_name or "").strip()
    if not focal:
        raise ValueError("Competitive set missing focal_vendor_name")

    competitors: list[str] = []
    seen: set[str] = {_norm_vendor(focal)}
    for raw_name in competitive_set.competitor_vendor_names:
        name = str(raw_name or "").strip()
        if not name:
            continue
        key = _norm_vendor(name)
        if key in seen:
            continue
        seen.add(key)
        competitors.append(name)

    vendor_names = [focal, *competitors]
    pairwise_pairs = [[focal, competitor] for competitor in competitors]

    category_names: list[str] = []
    if competitive_set.category_council_enabled and category_by_vendor:
        focal_category = str(category_by_vendor.get(_norm_vendor(focal)) or "").strip()
        if focal_category:
            matching_vendors = [
                vendor_name
                for vendor_name in vendor_names
                if _norm_vendor(category_by_vendor.get(_norm_vendor(vendor_name)) or "") == _norm_vendor(focal_category)
            ]
            if len(matching_vendors) >= 3:
                category_names.append(focal_category)

    asymmetry_pairs = pairwise_pairs[:] if competitive_set.asymmetry_enabled else []

    return CompetitiveSetPlan(
        competitive_set_id=str(competitive_set.id),
        focal_vendor_name=focal,
        vendor_names=vendor_names,
        pairwise_pairs=pairwise_pairs if competitive_set.pairwise_enabled else [],
        category_names=category_names,
        asymmetry_pairs=asymmetry_pairs,
        vendor_synthesis_enabled=competitive_set.vendor_synthesis_enabled,
        pairwise_enabled=competitive_set.pairwise_enabled,
        category_council_enabled=competitive_set.category_council_enabled,
        asymmetry_enabled=competitive_set.asymmetry_enabled,
    )


def plan_to_synthesis_metadata(plan: CompetitiveSetPlan) -> dict[str, Any]:
    """Convert a competitive-set plan into task metadata for scoped synthesis."""
    payload = plan.to_dict()
    payload.update({
        "scope_type": "competitive_set",
        "scope_id": plan.competitive_set_id,
        "scope_vendor_names": plan.vendor_names,
        "scope_pairwise_pairs": plan.pairwise_pairs,
        "scope_category_names": plan.category_names,
        "scope_asymmetry_pairs": plan.asymmetry_pairs,
    })
    return payload


def _rounded_usd(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


async def estimate_competitive_set_plan(
    pool,
    plan: CompetitiveSetPlan,
) -> dict[str, Any]:
    """Estimate token and cost upper bounds for a competitive-set run.

    Estimates use recent persisted synthesis history for scoped vendors and
    recent LLM usage aggregates for cost-per-token fallback. This is an upper
    bound for `force=false`; actual spend may be lower when vendor hashes reuse.
    """
    lookback_days = int(settings.b2b_churn.competitive_set_preview_lookback_days)

    vendor_rows = await pool.fetch(
        """
        WITH latest AS (
            SELECT DISTINCT ON (vendor_name)
                   vendor_name,
                   tokens_used
            FROM b2b_reasoning_synthesis
            WHERE schema_version = 'v2'
              AND vendor_name = ANY($1::text[])
            ORDER BY vendor_name, created_at DESC
        )
        SELECT vendor_name, tokens_used
        FROM latest
        """,
        plan.vendor_names,
    )
    vendor_tokens_by_name = {
        str(row["vendor_name"]): int(row["tokens_used"] or 0)
        for row in vendor_rows
    }

    usage_rows = await pool.fetch(
        """
        SELECT span_name,
               AVG(input_tokens + output_tokens)::float AS avg_total_tokens,
               AVG(cost_usd)::float AS avg_cost_usd,
               COUNT(*)::int AS sample_count
        FROM llm_usage
        WHERE span_name IN (
                'task.b2b_reasoning_synthesis',
                'task.b2b_reasoning_synthesis.cross_vendor'
            )
          AND created_at >= NOW() - make_interval(days => $1)
        GROUP BY span_name
        """,
        lookback_days,
    )
    usage_by_span = {
        str(row["span_name"]): {
            "avg_total_tokens": float(row["avg_total_tokens"] or 0.0),
            "avg_cost_usd": float(row["avg_cost_usd"] or 0.0),
            "sample_count": int(row["sample_count"] or 0),
        }
        for row in usage_rows
    }

    cross_rows = await pool.fetch(
        """
        SELECT analysis_type,
               AVG(tokens_used)::float AS avg_tokens_used,
               COUNT(*)::int AS sample_count
        FROM b2b_cross_vendor_reasoning_synthesis
        WHERE created_at >= NOW() - make_interval(days => $1)
        GROUP BY analysis_type
        """,
        lookback_days,
    )
    cross_tokens_by_type = {
        str(row["analysis_type"]): {
            "avg_tokens_used": float(row["avg_tokens_used"] or 0.0),
            "sample_count": int(row["sample_count"] or 0),
        }
        for row in cross_rows
    }

    vendor_usage = usage_by_span.get("task.b2b_reasoning_synthesis", {})
    cross_usage = usage_by_span.get("task.b2b_reasoning_synthesis.cross_vendor", {})
    vendor_fallback_tokens = float(vendor_usage.get("avg_total_tokens") or 0.0)
    cross_fallback_tokens = float(cross_usage.get("avg_total_tokens") or 0.0)
    vendor_cost_per_token = (
        float(vendor_usage.get("avg_cost_usd") or 0.0) / vendor_fallback_tokens
        if vendor_fallback_tokens > 0
        else None
    )
    cross_cost_per_token = (
        float(cross_usage.get("avg_cost_usd") or 0.0) / cross_fallback_tokens
        if cross_fallback_tokens > 0
        else None
    )

    vendor_jobs_with_history = 0
    vendor_jobs_using_fallback = 0
    estimated_vendor_tokens = 0.0
    if plan.vendor_synthesis_enabled:
        for vendor_name in plan.vendor_names:
            known_tokens = vendor_tokens_by_name.get(vendor_name)
            if known_tokens is not None and known_tokens > 0:
                vendor_jobs_with_history += 1
                estimated_vendor_tokens += float(known_tokens)
            else:
                vendor_jobs_using_fallback += 1
                estimated_vendor_tokens += vendor_fallback_tokens

    def _cross_type_tokens(analysis_type: str) -> tuple[float, bool]:
        stats = cross_tokens_by_type.get(analysis_type) or {}
        avg_tokens = float(stats.get("avg_tokens_used") or 0.0)
        if avg_tokens > 0:
            return avg_tokens, True
        return cross_fallback_tokens, False

    cross_jobs_with_history = 0
    cross_jobs_using_fallback = 0
    estimated_pairwise_tokens = 0.0
    estimated_category_tokens = 0.0
    estimated_asymmetry_tokens = 0.0

    if plan.pairwise_enabled:
        avg_tokens, has_history = _cross_type_tokens("pairwise_battle")
        estimated_pairwise_tokens = float(len(plan.pairwise_pairs)) * avg_tokens
        if has_history:
            cross_jobs_with_history += len(plan.pairwise_pairs)
        else:
            cross_jobs_using_fallback += len(plan.pairwise_pairs)
    if plan.category_council_enabled:
        avg_tokens, has_history = _cross_type_tokens("category_council")
        estimated_category_tokens = float(len(plan.category_names)) * avg_tokens
        if has_history:
            cross_jobs_with_history += len(plan.category_names)
        else:
            cross_jobs_using_fallback += len(plan.category_names)
    if plan.asymmetry_enabled:
        avg_tokens, has_history = _cross_type_tokens("resource_asymmetry")
        estimated_asymmetry_tokens = float(len(plan.asymmetry_pairs)) * avg_tokens
        if has_history:
            cross_jobs_with_history += len(plan.asymmetry_pairs)
        else:
            cross_jobs_using_fallback += len(plan.asymmetry_pairs)

    estimated_cross_tokens = (
        estimated_pairwise_tokens
        + estimated_category_tokens
        + estimated_asymmetry_tokens
    )
    estimated_total_tokens = estimated_vendor_tokens + estimated_cross_tokens
    estimated_vendor_cost = (
        estimated_vendor_tokens * vendor_cost_per_token
        if vendor_cost_per_token is not None
        else None
    )
    estimated_cross_cost = (
        estimated_cross_tokens * cross_cost_per_token
        if cross_cost_per_token is not None
        else None
    )
    estimated_total_cost = (
        (estimated_vendor_cost or 0.0) + (estimated_cross_cost or 0.0)
        if estimated_vendor_cost is not None or estimated_cross_cost is not None
        else None
    )

    return {
        "lookback_days": lookback_days,
        "vendor_jobs_planned": len(plan.vendor_names) if plan.vendor_synthesis_enabled else 0,
        "pairwise_jobs_planned": len(plan.pairwise_pairs) if plan.pairwise_enabled else 0,
        "category_jobs_planned": len(plan.category_names) if plan.category_council_enabled else 0,
        "asymmetry_jobs_planned": len(plan.asymmetry_pairs) if plan.asymmetry_enabled else 0,
        "estimated_vendor_tokens": int(round(estimated_vendor_tokens)),
        "estimated_cross_vendor_tokens": int(round(estimated_cross_tokens)),
        "estimated_total_tokens": int(round(estimated_total_tokens)),
        "estimated_vendor_cost_usd": _rounded_usd(estimated_vendor_cost),
        "estimated_cross_vendor_cost_usd": _rounded_usd(estimated_cross_cost),
        "estimated_total_cost_usd": _rounded_usd(estimated_total_cost),
        "vendor_jobs_with_history": vendor_jobs_with_history,
        "vendor_jobs_using_fallback": vendor_jobs_using_fallback,
        "cross_vendor_jobs_with_history": cross_jobs_with_history,
        "cross_vendor_jobs_using_fallback": cross_jobs_using_fallback,
        "recent_vendor_sample_count": int(vendor_usage.get("sample_count") or 0),
        "recent_cross_vendor_sample_count": int(cross_usage.get("sample_count") or 0),
        "note": (
            "Upper-bound estimate for a non-forced run. Actual spend may be lower "
            "when unchanged vendors hash-reuse existing synthesis."
        ),
    }


async def load_vendor_category_map(pool, vendor_names: list[str]) -> dict[str, str]:
    """Load product categories for a vendor subset."""
    vendor_names = [str(name or "").strip() for name in vendor_names if str(name or "").strip()]
    if not vendor_names:
        return {}
    rows = await pool.fetch(
        """
        WITH requested AS (
            SELECT UNNEST($1::text[]) AS vendor_name
        ),
        profile_match AS (
            SELECT DISTINCT ON (r.vendor_name)
                   r.vendor_name AS requested_vendor,
                   p.product_category
            FROM requested r
            LEFT JOIN b2b_product_profiles p
              ON LOWER(p.vendor_name) = LOWER(r.vendor_name)
            ORDER BY r.vendor_name, p.product_category NULLS LAST
        ),
        signal_match AS (
            SELECT DISTINCT ON (r.vendor_name)
                   r.vendor_name AS requested_vendor,
                   s.product_category
            FROM requested r
            LEFT JOIN b2b_churn_signals s
              ON LOWER(s.vendor_name) = LOWER(r.vendor_name)
            ORDER BY r.vendor_name, s.total_reviews DESC NULLS LAST
        )
        SELECT r.vendor_name,
               COALESCE(pm.product_category, sm.product_category, '') AS product_category
        FROM requested r
        LEFT JOIN profile_match pm ON pm.requested_vendor = r.vendor_name
        LEFT JOIN signal_match sm ON sm.requested_vendor = r.vendor_name
        """,
        vendor_names,
    )
    return {
        _norm_vendor(row["vendor_name"]): str(row["product_category"] or "").strip()
        for row in rows
        if str(row["product_category"] or "").strip()
    }
