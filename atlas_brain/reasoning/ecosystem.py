"""Ecosystem Pattern Recognition (WS5).

Category-level and market-level intelligence aggregated from per-vendor
analysis. Computes category health metrics, classifies market structure,
and generates "State of Category" evidence for Tier 4 caching.

Metrics:
    - HHI (Herfindahl-Hirschman Index): market concentration
    - Category churn velocity: weighted average churn across vendors
    - Displacement intensity: how much switching is happening
    - Pain convergence: are all vendors suffering the same complaints?
    - Market structure: consolidating / fragmenting / displacing / stable

All outputs are pure data -- they feed into the stratified reasoner as
Tier 4 (Market Dynamics) evidence cached quarterly.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atlas.reasoning.ecosystem")


@dataclass
class CategoryHealth:
    """Health metrics for a product category."""

    category: str
    vendor_count: int = 0
    total_reviews: int = 0
    avg_churn_density: float = 0.0
    avg_urgency: float = 0.0
    avg_positive_pct: float = 0.0
    hhi: float = 0.0  # Herfindahl-Hirschman Index (0-10000)
    displacement_intensity: float = 0.0  # displacement edges per vendor
    pain_convergence: float = 0.0  # how similar pain profiles are (0-1)
    dominant_archetype: str = ""
    market_structure: str = ""  # consolidating, fragmenting, displacing, stable


@dataclass
class CategoryVendorSlice:
    """A vendor's metrics within a category context."""

    vendor_name: str
    review_share: float = 0.0  # % of category reviews
    churn_density: float = 0.0
    avg_urgency: float = 0.0
    displacement_out: int = 0  # losing customers to competitors
    displacement_in: int = 0  # gaining customers from competitors
    net_displacement: int = 0  # in - out (positive = gaining)
    archetype: str = ""


@dataclass
class EcosystemEvidence:
    """Complete ecosystem analysis for Tier 4 caching."""

    category: str
    health: CategoryHealth
    vendor_slices: list[CategoryVendorSlice] = field(default_factory=list)
    pain_distribution: dict[str, int] = field(default_factory=dict)
    top_displacement_flows: list[dict[str, Any]] = field(default_factory=list)
    archetype_distribution: dict[str, int] = field(default_factory=dict)


class EcosystemAnalyzer:
    """Computes category-level intelligence from vendor data."""

    def __init__(self, pool: Any):
        self._pool = pool

    async def analyze_category(self, category: str) -> EcosystemEvidence:
        """Full ecosystem analysis for a product category."""
        vendors = await self._load_category_vendors(category)

        if not vendors:
            return EcosystemEvidence(
                category=category,
                health=CategoryHealth(category=category),
            )

        health = self._compute_health(category, vendors)
        slices = self._compute_vendor_slices(vendors)
        pain_dist = await self._compute_pain_distribution(category)
        top_flows = await self._load_top_displacements(category)

        # Classify market structure
        health.market_structure = self._classify_market(health, slices)

        # Dominant archetype from archetype scoring
        archetype_dist = await self._compute_archetype_distribution(category)
        if archetype_dist:
            health.dominant_archetype = max(archetype_dist, key=archetype_dist.get)

        return EcosystemEvidence(
            category=category,
            health=health,
            vendor_slices=slices,
            pain_distribution=pain_dist,
            top_displacement_flows=top_flows,
            archetype_distribution=archetype_dist,
        )

    async def analyze_all_categories(self) -> dict[str, EcosystemEvidence]:
        """Analyze all categories with sufficient data."""
        categories = await self._pool.fetch("""
            SELECT DISTINCT product_category FROM b2b_churn_signals
            WHERE product_category IS NOT NULL AND product_category != ''
        """)
        results = {}
        for row in categories:
            cat = row["product_category"]
            results[cat] = await self.analyze_category(cat)
        return results

    # ------------------------------------------------------------------
    # Health metrics
    # ------------------------------------------------------------------

    def _compute_health(
        self, category: str, vendors: list[dict],
    ) -> CategoryHealth:
        """Compute aggregate health metrics for a category."""
        n = len(vendors)
        total_reviews = sum(v.get("total_reviews", 0) or 0 for v in vendors)

        # Weighted averages
        avg_churn = _safe_avg([v.get("churn_density", 0) for v in vendors])
        avg_urgency = _safe_avg([v.get("avg_urgency", 0) for v in vendors])
        avg_pos = _safe_avg([v.get("positive_review_pct", 0) for v in vendors])

        # HHI: sum of squared market shares (by review count)
        hhi = 0.0
        if total_reviews > 0:
            for v in vendors:
                share = ((v.get("total_reviews", 0) or 0) / total_reviews) * 100
                hhi += share * share

        # Displacement intensity
        total_disp = sum(
            (v.get("displacement_edge_count", 0) or 0) for v in vendors
        )
        disp_intensity = total_disp / n if n > 0 else 0.0

        return CategoryHealth(
            category=category,
            vendor_count=n,
            total_reviews=total_reviews,
            avg_churn_density=round(avg_churn, 2),
            avg_urgency=round(avg_urgency, 2),
            avg_positive_pct=round(avg_pos, 2),
            hhi=round(hhi, 1),
            displacement_intensity=round(disp_intensity, 2),
        )

    def _compute_vendor_slices(self, vendors: list[dict]) -> list[CategoryVendorSlice]:
        """Compute per-vendor category context."""
        total_reviews = sum(v.get("total_reviews", 0) or 0 for v in vendors)
        slices = []
        for v in vendors:
            reviews = v.get("total_reviews", 0) or 0
            share = (reviews / total_reviews * 100) if total_reviews > 0 else 0

            disp_out = v.get("displacement_out", 0) or 0
            disp_in = v.get("displacement_in", 0) or 0

            slices.append(CategoryVendorSlice(
                vendor_name=v["vendor_name"],
                review_share=round(share, 1),
                churn_density=float(v.get("churn_density", 0) or 0),
                avg_urgency=float(v.get("avg_urgency", 0) or 0),
                displacement_out=disp_out,
                displacement_in=disp_in,
                net_displacement=disp_in - disp_out,
            ))

        slices.sort(key=lambda s: s.review_share, reverse=True)
        return slices

    # ------------------------------------------------------------------
    # Market structure classification
    # ------------------------------------------------------------------

    def _classify_market(
        self, health: CategoryHealth, slices: list[CategoryVendorSlice],
    ) -> str:
        """Classify market structure from health metrics.

        - consolidating: high HHI (>2500) or top vendor >40% share, low displacement
        - fragmenting: low HHI (<1500), many vendors, moderate displacement
        - displacing: high displacement intensity, clear winners/losers
        - stable: low churn, low displacement, balanced shares
        """
        if health.vendor_count < 2:
            return "insufficient_data"

        # Check for active displacement
        if health.displacement_intensity > 2.0:
            # Are there clear winners?
            if slices:
                max_net = max(s.net_displacement for s in slices)
                min_net = min(s.net_displacement for s in slices)
                if max_net > 2 and min_net < -2:
                    return "displacing"

        # Check concentration
        if health.hhi > 2500:
            return "consolidating"
        if slices and slices[0].review_share > 40:
            return "consolidating"

        # Check fragmentation
        if health.hhi < 1500 and health.vendor_count > 5:
            return "fragmenting"

        # Check stability
        if health.avg_churn_density < 30 and health.displacement_intensity < 1.0:
            return "stable"

        # Default: check churn vs displacement balance
        if health.avg_churn_density > 40:
            return "displacing" if health.displacement_intensity > 1.5 else "fragmenting"

        return "stable"

    # ------------------------------------------------------------------
    # Pain convergence
    # ------------------------------------------------------------------

    async def _compute_pain_distribution(self, category: str) -> dict[str, int]:
        """Aggregate pain point distribution across a category."""
        rows = await self._pool.fetch("""
            SELECT pp.pain_category, SUM(pp.mention_count) AS total_mentions
            FROM b2b_vendor_pain_points pp
            JOIN b2b_churn_signals cs
                ON LOWER(pp.vendor_name) = LOWER(cs.vendor_name)
            WHERE LOWER(cs.product_category) = LOWER($1)
            GROUP BY pp.pain_category
            ORDER BY total_mentions DESC
            LIMIT 15
        """, category)
        return {r["pain_category"]: r["total_mentions"] for r in rows}

    async def _compute_archetype_distribution(self, category: str) -> dict[str, int]:
        """Count cached archetype classifications for a category."""
        rows = await self._pool.fetch("""
            SELECT pattern_class, COUNT(*) AS cnt
            FROM reasoning_semantic_cache
            WHERE LOWER(product_category) = LOWER($1)
              AND invalidated_at IS NULL
            GROUP BY pattern_class
            ORDER BY cnt DESC
        """, category)
        return {r["pattern_class"]: r["cnt"] for r in rows}

    # ------------------------------------------------------------------
    # Displacement flows
    # ------------------------------------------------------------------

    async def _load_top_displacements(self, category: str) -> list[dict[str, Any]]:
        """Load top displacement flows within a category."""
        rows = await self._pool.fetch("""
            SELECT de.from_vendor, de.to_vendor,
                   de.mention_count, de.primary_driver, de.signal_strength
            FROM b2b_displacement_edges de
            JOIN b2b_churn_signals cs
                ON LOWER(de.from_vendor) = LOWER(cs.vendor_name)
            WHERE LOWER(cs.product_category) = LOWER($1)
            ORDER BY de.mention_count DESC
            LIMIT 10
        """, category)
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _load_category_vendors(self, category: str) -> list[dict]:
        """Load vendor data for a category, including displacement counts."""
        rows = await self._pool.fetch("""
            SELECT cs.vendor_name, cs.total_reviews,
                   cs.avg_urgency_score AS avg_urgency,
                   cs.confidence_score,
                   snap.churn_density, snap.positive_review_pct,
                   snap.displacement_edge_count,
                   COALESCE(d_out.cnt, 0) AS displacement_out,
                   COALESCE(d_in.cnt, 0) AS displacement_in
            FROM b2b_churn_signals cs
            LEFT JOIN (
                SELECT DISTINCT ON (vendor_name) *
                FROM b2b_vendor_snapshots
                ORDER BY vendor_name, snapshot_date DESC
            ) snap ON LOWER(cs.vendor_name) = LOWER(snap.vendor_name)
            LEFT JOIN (
                SELECT from_vendor, SUM(mention_count) AS cnt
                FROM b2b_displacement_edges
                GROUP BY from_vendor
            ) d_out ON LOWER(cs.vendor_name) = LOWER(d_out.from_vendor)
            LEFT JOIN (
                SELECT to_vendor, SUM(mention_count) AS cnt
                FROM b2b_displacement_edges
                GROUP BY to_vendor
            ) d_in ON LOWER(cs.vendor_name) = LOWER(d_in.to_vendor)
            WHERE LOWER(cs.product_category) = LOWER($1)
        """, category)
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Serialization for Tier 4 caching
    # ------------------------------------------------------------------

    @staticmethod
    def to_evidence_dict(eco: EcosystemEvidence) -> dict[str, Any]:
        """Convert ecosystem evidence to a dict for the stratified reasoner."""
        h = eco.health
        evidence: dict[str, Any] = {
            "category": eco.category,
            "vendor_count": h.vendor_count,
            "total_reviews": h.total_reviews,
            "avg_churn_density": h.avg_churn_density,
            "avg_urgency": h.avg_urgency,
            "avg_positive_pct": h.avg_positive_pct,
            "hhi": h.hhi,
            "displacement_intensity": h.displacement_intensity,
            "market_structure": h.market_structure,
            "dominant_archetype": h.dominant_archetype,
        }

        if eco.vendor_slices:
            evidence["vendor_positions"] = [
                {
                    "vendor": s.vendor_name,
                    "review_share": s.review_share,
                    "churn_density": s.churn_density,
                    "net_displacement": s.net_displacement,
                }
                for s in eco.vendor_slices[:10]
            ]

        if eco.pain_distribution:
            evidence["category_pains"] = eco.pain_distribution

        if eco.top_displacement_flows:
            evidence["top_flows"] = [
                {
                    "from": f["from_vendor"],
                    "to": f["to_vendor"],
                    "mentions": f["mention_count"],
                    "driver": f.get("primary_driver", ""),
                }
                for f in eco.top_displacement_flows[:5]
            ]

        if eco.archetype_distribution:
            evidence["archetype_distribution"] = eco.archetype_distribution

        return evidence


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _safe_avg(values: list) -> float:
    """Average that handles None/zero values."""
    clean = [float(v) for v in values if v is not None]
    return sum(clean) / len(clean) if clean else 0.0
