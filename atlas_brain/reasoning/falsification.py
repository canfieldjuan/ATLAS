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

        # Evaluate all conditions via LLM
        str_conditions = [c for c in conditions if isinstance(c, str) and c.strip()]
        triggered = await self._check_conditions_llm(
            str_conditions, fresh, vendor_name or "unknown",
        )

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

    async def _check_conditions_llm(
        self,
        conditions: list[str],
        fresh_signals: dict[str, Any],
        vendor_name: str,
    ) -> list[str]:
        """Evaluate falsification conditions against fresh data using local LLM.

        Returns the list of triggered condition strings.
        """
        if not conditions:
            return []

        from ..pipelines.llm import get_pipeline_llm
        llm = get_pipeline_llm(workload="vllm")
        if llm is None:
            llm = get_pipeline_llm(workload="synthesis")
        if llm is None:
            logger.warning("No LLM available for falsification evaluation")
            return []

        signal_summary = json.dumps(fresh_signals, indent=2, default=str)
        conditions_text = "\n".join(
            f"{i+1}. {c}" for i, c in enumerate(conditions)
        )

        prompt = (
            f"Vendor: {vendor_name}\n\n"
            f"Current signals:\n{signal_summary}\n\n"
            f"Falsification conditions to evaluate:\n{conditions_text}\n\n"
            "For each condition, determine if the current signal data "
            "provides evidence that the condition has been met.\n\n"
            "Return ONLY a JSON array of integers representing the "
            "1-based indices of conditions that ARE triggered by the data. "
            "If none are triggered, return an empty array [].\n"
            "Example: [2, 5] means conditions 2 and 5 are triggered.\n"
            "Return ONLY the JSON array, no explanation."
        )

        from ..services.protocols import Message
        import asyncio

        messages = [
            Message(
                role="system",
                content=(
                    "You are a data analyst evaluating whether observed metrics "
                    "satisfy specific falsification conditions. Be conservative: "
                    "only mark a condition as triggered if the data clearly "
                    "supports it. Ambiguous or insufficient data means NOT triggered."
                ),
            ),
            Message(role="user", content=prompt),
        ]

        try:
            result = await asyncio.to_thread(
                llm.chat,
                messages=messages,
                max_tokens=128,
                temperature=0.0,
            )
            text = (result.get("response") or "").strip() if isinstance(result, dict) else ""
            if not text:
                return []

            # Clean markdown fences
            import re as _re
            text = _re.sub(r"^```\w*\n?", "", text)
            text = _re.sub(r"\n?```$", "", text)
            text = text.strip()

            triggered_indices = json.loads(text)
            if not isinstance(triggered_indices, list):
                return []

            triggered = []
            for idx in triggered_indices:
                if isinstance(idx, int) and 1 <= idx <= len(conditions):
                    triggered.append(conditions[idx - 1])
            return triggered

        except Exception:
            logger.debug(
                "LLM falsification evaluation failed for %s",
                vendor_name, exc_info=True,
            )
            return []

    async def _fetch_fresh_signals(self, vendor_name: str | None) -> dict[str, Any]:
        """Fetch the latest signals for a vendor from existing tables."""
        if not vendor_name:
            return {}

        signals: dict[str, Any] = {}

        try:
            # Latest snapshot
            row = await self._pool.fetchrow(
                """
                SELECT avg_urgency, positive_review_pct, churn_density,
                       pressure_score, total_reviews, recommend_ratio,
                       archetype, archetype_confidence
                FROM b2b_vendor_snapshots
                WHERE vendor_name = $1
                ORDER BY snapshot_date DESC LIMIT 1
                """,
                vendor_name,
            )
            if row:
                for col in ("avg_urgency", "positive_review_pct", "churn_density",
                            "pressure_score", "total_reviews", "recommend_ratio",
                            "archetype", "archetype_confidence"):
                    if row.get(col) is not None:
                        signals[col] = float(row[col]) if isinstance(row[col], (int, float)) else row[col]

            # Previous snapshot (for delta comparison)
            prev = await self._pool.fetchrow(
                """
                SELECT avg_urgency, positive_review_pct, churn_density,
                       pressure_score, recommend_ratio
                FROM b2b_vendor_snapshots
                WHERE vendor_name = $1
                ORDER BY snapshot_date DESC LIMIT 1 OFFSET 1
                """,
                vendor_name,
            )
            if prev:
                for col in ("avg_urgency", "positive_review_pct", "churn_density",
                            "pressure_score", "recommend_ratio"):
                    if prev.get(col) is not None:
                        signals[f"prev_{col}"] = float(prev[col]) if isinstance(prev[col], (int, float)) else prev[col]

            # Churn signal enrichment (price, DM rate, competitors)
            sig_row = await self._pool.fetchrow(
                """
                SELECT price_complaint_rate, decision_maker_churn_rate,
                       archetype, archetype_confidence
                FROM b2b_churn_signals
                WHERE vendor_name = $1 AND archetype IS NOT NULL
                ORDER BY last_computed_at DESC LIMIT 1
                """,
                vendor_name,
            )
            if sig_row:
                if sig_row.get("price_complaint_rate") is not None:
                    signals["price_complaint_rate"] = float(sig_row["price_complaint_rate"])
                if sig_row.get("decision_maker_churn_rate") is not None:
                    signals["dm_churn_rate"] = float(sig_row["decision_maker_churn_rate"])
                if sig_row.get("archetype"):
                    signals["current_archetype"] = sig_row["archetype"]
                    signals["archetype_confidence"] = float(sig_row["archetype_confidence"] or 0)

            # Recent negative review count (7 days)
            neg_count = await self._pool.fetchval(
                """
                SELECT COUNT(*) FROM b2b_reviews
                WHERE vendor_name = $1
                  AND duplicate_of_review_id IS NULL
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
                  AND created_at >= NOW() - INTERVAL '14 days'
                ORDER BY created_at DESC LIMIT 5
                """,
                vendor_name,
            )
            if support_events:
                improving = sum(1 for e in support_events if e.get("direction") == "improving")
                signals["support_trend"] = "improving" if improving > len(support_events) / 2 else "stable"

        except Exception:
            logger.debug("Failed to fetch fresh signals for %s", vendor_name, exc_info=True)

        return signals
