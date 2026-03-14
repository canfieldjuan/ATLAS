"""Metacognitive Monitor (WS0E).

Tracks reasoning system health: cache hit rates, conclusion type distribution,
surprise detection, and exploration budget enforcement.

Surprise detection ("boredom" algorithm):
    - Maintains rolling distribution of conclusion types (archetypes)
    - If new conclusion is in bottom 5% of distribution -> escalate to full Reason
    - 10-15% of queries get random full-reasoning regardless (drift detection)

Exploration budget:
    - Prevents the system from getting stuck in local optima
    - Random sample of "solved" cases get full re-reasoning
    - Results compared to cached conclusions for cache quality scoring
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("atlas.reasoning.metacognition")

EXPLORATION_RATE = 0.12     # 12% of queries get forced full-reasoning
SURPRISE_PERCENTILE = 0.05  # bottom 5% of distribution = surprise


@dataclass
class MetacognitiveState:
    """In-memory state for the current reasoning session."""

    total_queries: int = 0
    recall_hits: int = 0
    reconstitute_hits: int = 0
    full_reasons: int = 0
    surprise_escalations: int = 0
    exploration_samples: int = 0
    tokens_spent: int = 0
    tokens_saved: int = 0
    conclusion_types: Counter = field(default_factory=Counter)


class MetacognitiveMonitor:
    """Monitors reasoning quality and decides when to force re-reasoning."""

    def __init__(self, pool: Any):
        self._pool = pool
        self._state = MetacognitiveState()
        self._distribution_cache: dict[str, float] | None = None
        self._distribution_loaded_at: datetime | None = None

    # ------------------------------------------------------------------
    # Decision helpers (called by StratifiedReasoner before recall)
    # ------------------------------------------------------------------

    def should_force_exploration(self) -> bool:
        """Random exploration: force full-reason to detect cache drift."""
        if random.random() < EXPLORATION_RATE:
            self._state.exploration_samples += 1
            logger.debug("Exploration sample triggered (rate=%.0f%%)", EXPLORATION_RATE * 100)
            return True
        return False

    async def is_surprise(self, conclusion_type: str) -> bool:
        """Check if this conclusion type is rare enough to be 'surprising'.

        Uses the rolling distribution from the DB. If the type is in the
        bottom 5% of observed types, it's a surprise -> force full reasoning.
        """
        dist = await self._get_distribution()
        if not dist:
            return False  # no history yet, can't detect surprise

        total = sum(dist.values())
        if total < 20:
            return False  # not enough data for statistical significance

        type_count = dist.get(conclusion_type, 0)
        type_freq = type_count / total

        if type_freq <= SURPRISE_PERCENTILE:
            self._state.surprise_escalations += 1
            logger.info(
                "Surprise detected: '%s' is %.1f%% of distribution (threshold=%.1f%%)",
                conclusion_type, type_freq * 100, SURPRISE_PERCENTILE * 100,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, mode: str, tokens_used: int, conclusion_type: str = "") -> None:
        """Record a reasoning outcome in the in-memory state."""
        self._state.total_queries += 1
        self._state.tokens_spent += tokens_used

        if mode == "recall":
            self._state.recall_hits += 1
            self._state.tokens_saved += 2000  # avg tokens for a full reason
        elif mode == "reconstitute":
            self._state.reconstitute_hits += 1
            self._state.tokens_saved += 1400  # saves ~70% vs full
        elif mode == "reason":
            self._state.full_reasons += 1

        if conclusion_type:
            self._state.conclusion_types[conclusion_type] += 1

    async def flush(self) -> None:
        """Persist accumulated state to the DB. Call periodically or at shutdown."""
        if self._state.total_queries == 0:
            return

        import json as _json

        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today.replace(hour=23, minute=59, second=59)

        # Merge distribution additively: read existing, sum in Python, write back
        new_dist = dict(self._state.conclusion_types)
        try:
            row = await self._pool.fetchrow(
                """SELECT conclusion_type_distribution FROM reasoning_metacognition
                   WHERE period_start = $1""",
                today,
            )
            if row and row["conclusion_type_distribution"]:
                existing = row["conclusion_type_distribution"]
                if isinstance(existing, str):
                    existing = _json.loads(existing)
                if isinstance(existing, dict):
                    for k, v in existing.items():
                        new_dist[k] = new_dist.get(k, 0) + (int(v) if isinstance(v, (int, float)) else 0)
        except Exception:
            pass  # proceed with just the new distribution

        dist_json = _json.dumps(new_dist)

        try:
            await self._pool.execute(
                """
                INSERT INTO reasoning_metacognition (
                    period_start, period_end, total_queries,
                    recall_hits, reconstitute_hits, full_reasons,
                    surprise_escalations, exploration_samples,
                    total_tokens_saved, total_tokens_spent,
                    conclusion_type_distribution
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
                ON CONFLICT (period_start) DO UPDATE SET
                    total_queries = reasoning_metacognition.total_queries + EXCLUDED.total_queries,
                    recall_hits = reasoning_metacognition.recall_hits + EXCLUDED.recall_hits,
                    reconstitute_hits = reasoning_metacognition.reconstitute_hits + EXCLUDED.reconstitute_hits,
                    full_reasons = reasoning_metacognition.full_reasons + EXCLUDED.full_reasons,
                    surprise_escalations = reasoning_metacognition.surprise_escalations + EXCLUDED.surprise_escalations,
                    exploration_samples = reasoning_metacognition.exploration_samples + EXCLUDED.exploration_samples,
                    total_tokens_saved = reasoning_metacognition.total_tokens_saved + EXCLUDED.total_tokens_saved,
                    total_tokens_spent = reasoning_metacognition.total_tokens_spent + EXCLUDED.total_tokens_spent,
                    conclusion_type_distribution = EXCLUDED.conclusion_type_distribution
                """,
                today, tomorrow,
                self._state.total_queries,
                self._state.recall_hits,
                self._state.reconstitute_hits,
                self._state.full_reasons,
                self._state.surprise_escalations,
                self._state.exploration_samples,
                self._state.tokens_saved,
                self._state.tokens_spent,
                dist_json,
            )
            logger.info(
                "Flushed metacognition: %d queries (%d recall, %d reconstitute, %d reason, %d surprise, %d explore)",
                self._state.total_queries, self._state.recall_hits,
                self._state.reconstitute_hits, self._state.full_reasons,
                self._state.surprise_escalations, self._state.exploration_samples,
            )
            self._state = MetacognitiveState()  # reset after flush
        except Exception:
            logger.warning("Failed to flush metacognition", exc_info=True)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(self, days: int = 7) -> dict[str, Any]:
        """Aggregate stats over the last N days."""
        row = await self._pool.fetchrow(
            """
            SELECT
                SUM(total_queries) AS total_queries,
                SUM(recall_hits) AS recall_hits,
                SUM(reconstitute_hits) AS reconstitute_hits,
                SUM(full_reasons) AS full_reasons,
                SUM(surprise_escalations) AS surprise_escalations,
                SUM(exploration_samples) AS exploration_samples,
                SUM(total_tokens_saved) AS tokens_saved,
                SUM(total_tokens_spent) AS tokens_spent
            FROM reasoning_metacognition
            WHERE period_start >= NOW() - ($1 || ' days')::INTERVAL
            """,
            str(days),
        )
        if row is None or row["total_queries"] is None:
            return {"total_queries": 0, "period_days": days}

        total = row["total_queries"]
        return {
            "period_days": days,
            "total_queries": total,
            "recall_rate": (row["recall_hits"] / total * 100) if total else 0,
            "reconstitute_rate": (row["reconstitute_hits"] / total * 100) if total else 0,
            "full_reason_rate": (row["full_reasons"] / total * 100) if total else 0,
            "surprise_rate": (row["surprise_escalations"] / total * 100) if total else 0,
            "exploration_rate": (row["exploration_samples"] / total * 100) if total else 0,
            "tokens_saved": row["tokens_saved"] or 0,
            "tokens_spent": row["tokens_spent"] or 0,
            "efficiency": (
                (row["tokens_saved"] / (row["tokens_saved"] + row["tokens_spent"]) * 100)
                if (row["tokens_saved"] or 0) + (row["tokens_spent"] or 0) > 0
                else 0
            ),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get_distribution(self) -> dict[str, float]:
        """Load conclusion type distribution from DB (cached 5 min)."""
        now = datetime.now(timezone.utc)
        if (
            self._distribution_cache is not None
            and self._distribution_loaded_at is not None
            and (now - self._distribution_loaded_at).total_seconds() < 300
        ):
            return self._distribution_cache

        try:
            rows = await self._pool.fetch(
                """
                SELECT conclusion_type_distribution
                FROM reasoning_metacognition
                WHERE period_start >= NOW() - INTERVAL '30 days'
                  AND conclusion_type_distribution IS NOT NULL
                ORDER BY period_start DESC
                LIMIT 30
                """
            )
            merged: Counter = Counter()
            for row in rows:
                dist = row["conclusion_type_distribution"]
                if isinstance(dist, dict):
                    for k, v in dist.items():
                        merged[k] += int(v) if isinstance(v, (int, float)) else 1

            self._distribution_cache = dict(merged)
            self._distribution_loaded_at = now
            return self._distribution_cache
        except Exception:
            logger.debug("Failed to load distribution", exc_info=True)
            return {}
