"""Falsification Watcher (WS0F).

Each cached conclusion stores "what would prove this wrong" as a list of
falsification conditions (free-text descriptions). The watcher runs as a
nightly autonomous task, checking conditions against fresh data.

If any condition is triggered, the cache entry is invalidated, forcing
re-reasoning on the next query for that vendor/pattern.

Example falsification conditions:
    - "Competitor drops price below Vendor X"
    - "Positive review trend reversal (3+ consecutive weeks improving)"
    - "Vendor X releases SSO feature"
    - "Support response time improves to < 4h average"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("atlas.reasoning.falsification")


@dataclass
class FalsificationResult:
    """Result of checking one cache entry's falsification conditions."""

    pattern_sig: str
    vendor_name: str | None
    total_conditions: int
    triggered_conditions: list[str]
    invalidated: bool


class FalsificationWatcher:
    """Checks cached conclusions against new evidence for invalidation."""

    def __init__(self, pool: Any, cache: Any):
        """
        *pool*: DatabasePool
        *cache*: SemanticCache instance (for invalidation calls)
        """
        self._pool = pool
        self._cache = cache

    async def run_nightly_check(self) -> list[FalsificationResult]:
        """Check all active cache entries. Returns results for entries checked.

        This is the main entry point, called by the autonomous scheduler.
        """
        # Load all active entries with falsification conditions
        rows = await self._pool.fetch(
            """
            SELECT pattern_sig, vendor_name, product_category,
                   conclusion, falsification_conditions, evidence_hash
            FROM reasoning_semantic_cache
            WHERE invalidated_at IS NULL
              AND falsification_conditions IS NOT NULL
              AND falsification_conditions != '[]'::jsonb
            ORDER BY last_validated_at ASC
            """
        )

        if not rows:
            logger.info("Falsification check: no entries with conditions to check")
            return []

        logger.info("Falsification check: examining %d cache entries", len(rows))
        results = []

        for row in rows:
            result = await self._check_entry(row)
            results.append(result)

        invalidated_count = sum(1 for r in results if r.invalidated)
        logger.info(
            "Falsification check complete: %d/%d entries invalidated",
            invalidated_count, len(results),
        )
        return results

    async def _check_entry(self, row) -> FalsificationResult:
        """Check a single cache entry against current data."""
        pattern_sig = row["pattern_sig"]
        vendor_name = row["vendor_name"]
        conditions = row["falsification_conditions"]

        if isinstance(conditions, str):
            try:
                conditions = json.loads(conditions)
            except (json.JSONDecodeError, TypeError):
                conditions = []
        if not isinstance(conditions, list):
            conditions = []

        if not conditions:
            return FalsificationResult(
                pattern_sig=pattern_sig,
                vendor_name=vendor_name,
                total_conditions=0,
                triggered_conditions=[],
                invalidated=False,
            )

        # Fetch fresh evidence for this vendor
        fresh = await self._fetch_fresh_signals(vendor_name)

        # Check each condition against fresh data
        triggered = []
        for condition in conditions:
            if not isinstance(condition, str):
                continue
            if self._check_condition(condition, fresh):
                triggered.append(condition)

        # Invalidate if any condition triggered
        invalidated = False
        if triggered:
            await self._cache.invalidate(
                pattern_sig,
                reason=f"falsification: {'; '.join(triggered[:3])}",
            )
            invalidated = True
            logger.info(
                "FALSIFIED %s (%s): %d conditions triggered",
                pattern_sig, vendor_name, len(triggered),
            )

        return FalsificationResult(
            pattern_sig=pattern_sig,
            vendor_name=vendor_name,
            total_conditions=len(conditions),
            triggered_conditions=triggered,
            invalidated=invalidated,
        )

    def _check_condition(self, condition: str, fresh_signals: dict[str, Any]) -> bool:
        """Evaluate a single falsification condition against fresh data.

        Uses keyword-based heuristic matching. In Phase 3+, this could be
        upgraded to LLM-based evaluation for complex conditions.
        """
        cond_lower = condition.lower()

        # Pattern: positive review trend / improvement
        if any(kw in cond_lower for kw in ["positive trend", "improving", "improvement"]):
            positive_pct = fresh_signals.get("positive_review_pct", 0)
            prev_positive_pct = fresh_signals.get("prev_positive_review_pct")
            if prev_positive_pct is not None and positive_pct > prev_positive_pct + 5:
                return True

        # Pattern: competitor price drop / cheaper alternative
        if any(kw in cond_lower for kw in ["price drop", "cheaper", "free tier", "drops price"]):
            competitor_price_changes = fresh_signals.get("competitor_price_changes", [])
            if any(c.get("direction") == "decrease" for c in competitor_price_changes):
                return True

        # Pattern: feature release / launches feature
        if any(kw in cond_lower for kw in ["releases", "launches", "ships", "adds feature"]):
            recent_features = fresh_signals.get("recent_feature_releases", [])
            for feature in recent_features:
                # Check if the feature mentioned in condition was released
                feature_name = feature.get("name", "").lower()
                for word in cond_lower.split():
                    if len(word) > 3 and word in feature_name:
                        return True

        # Pattern: support improvement
        if any(kw in cond_lower for kw in ["support", "response time", "csat"]):
            support_trend = fresh_signals.get("support_trend")
            if support_trend == "improving":
                return True

        # Pattern: urgency/churn decrease
        if any(kw in cond_lower for kw in ["urgency decreas", "churn decreas", "stabiliz"]):
            urgency = fresh_signals.get("avg_urgency", 10)
            prev_urgency = fresh_signals.get("prev_avg_urgency")
            if prev_urgency is not None and urgency < prev_urgency - 1.0:
                return True

        # Pattern: review volume drop (indicates issue resolved)
        if any(kw in cond_lower for kw in ["complaint", "negative review", "review volume"]):
            neg_count = fresh_signals.get("negative_review_count_7d", 0)
            prev_neg = fresh_signals.get("prev_negative_review_count_7d")
            if prev_neg is not None and neg_count < prev_neg * 0.5:
                return True

        return False

    async def _fetch_fresh_signals(self, vendor_name: str | None) -> dict[str, Any]:
        """Fetch the latest signals for a vendor from existing tables."""
        if not vendor_name:
            return {}

        signals: dict[str, Any] = {}

        try:
            # Latest snapshot
            row = await self._pool.fetchrow(
                """
                SELECT * FROM b2b_vendor_snapshots
                WHERE vendor_name = $1
                ORDER BY snapshot_date DESC LIMIT 1
                """,
                vendor_name,
            )
            if row:
                signals["avg_urgency"] = row.get("avg_urgency")
                signals["positive_review_pct"] = row.get("positive_review_pct")
                signals["churn_density"] = row.get("churn_density")

            # Previous snapshot (for delta comparison)
            prev = await self._pool.fetchrow(
                """
                SELECT * FROM b2b_vendor_snapshots
                WHERE vendor_name = $1
                ORDER BY snapshot_date DESC LIMIT 1 OFFSET 1
                """,
                vendor_name,
            )
            if prev:
                signals["prev_avg_urgency"] = prev.get("avg_urgency")
                signals["prev_positive_review_pct"] = prev.get("positive_review_pct")

            # Recent negative review count (7 days)
            neg_count = await self._pool.fetchval(
                """
                SELECT COUNT(*) FROM b2b_reviews
                WHERE vendor_name = $1
                  AND enriched_at >= NOW() - INTERVAL '7 days'
                  AND overall_sentiment = 'negative'
                """,
                vendor_name,
            )
            signals["negative_review_count_7d"] = neg_count or 0

            # Change events for support trend
            support_events = await self._pool.fetch(
                """
                SELECT event_type, direction FROM b2b_change_events
                WHERE vendor_name = $1
                  AND event_type LIKE '%support%'
                  AND detected_at >= NOW() - INTERVAL '14 days'
                ORDER BY detected_at DESC LIMIT 5
                """,
                vendor_name,
            )
            if support_events:
                improving = sum(1 for e in support_events if e.get("direction") == "improving")
                signals["support_trend"] = "improving" if improving > len(support_events) / 2 else "stable"

        except Exception:
            logger.debug("Failed to fetch fresh signals for %s", vendor_name, exc_info=True)

        return signals
