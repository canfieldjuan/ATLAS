"""Planner helpers for competitive-set scoped synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
