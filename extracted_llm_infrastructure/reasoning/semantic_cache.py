"""Semantic memory cache for the stratified reasoning engine.

Stores generalised reasoning conclusions in Postgres with confidence decay.
Each entry represents a cached pattern conclusion (e.g. "pricing_shock for
Vendor X") that can be recalled without re-running the LLM.

Confidence decays exponentially since last validation:
    effective = confidence * 2^(-(days_since_validated / decay_half_life_days))
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("atlas.reasoning.semantic_cache")


@dataclass
class CacheEntry:
    """A single cached reasoning conclusion."""

    pattern_sig: str
    pattern_class: str
    conclusion: dict[str, Any]
    confidence: float
    reasoning_steps: list[dict[str, Any]] = field(default_factory=list)
    boundary_conditions: dict[str, Any] = field(default_factory=dict)
    falsification_conditions: list[str] = field(default_factory=list)
    uncertainty_sources: list[str] = field(default_factory=list)
    vendor_name: str | None = None
    product_category: str | None = None
    decay_half_life_days: int = 90
    conclusion_type: str | None = None
    evidence_hash: str | None = None
    created_at: datetime | None = None
    last_validated_at: datetime | None = None
    validation_count: int = 1
    effective_confidence: float | None = None


def compute_evidence_hash(evidence: dict[str, Any]) -> str:
    """SHA-256 of deterministically serialised evidence dict."""
    raw = json.dumps(evidence, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _apply_decay(confidence: float, last_validated: datetime, half_life_days: int) -> float:
    """Return effective confidence after exponential decay."""
    now = datetime.now(timezone.utc)
    if last_validated.tzinfo is None:
        last_validated = last_validated.replace(tzinfo=timezone.utc)
    days = (now - last_validated).total_seconds() / 86400.0
    if days <= 0 or half_life_days <= 0:
        return confidence
    return confidence * math.pow(2, -(days / half_life_days))


class SemanticCache:
    """Postgres-backed semantic memory for cached reasoning conclusions."""

    # Entries with effective confidence below this are treated as stale
    STALE_THRESHOLD = 0.5

    def __init__(self, pool: Any):
        """*pool*: atlas_brain.storage.database.DatabasePool instance."""
        self._pool = pool

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def lookup(self, pattern_sig: str) -> CacheEntry | None:
        """Recall: fetch active entry if confidence is fresh enough.

        Returns None on miss or if effective_confidence < STALE_THRESHOLD.
        """
        row = await self._pool.fetchrow(
            """
            SELECT * FROM reasoning_semantic_cache
            WHERE pattern_sig = $1 AND invalidated_at IS NULL
            """,
            pattern_sig,
        )
        if row is None:
            return None

        entry = self._row_to_entry(row)
        eff = _apply_decay(entry.confidence, entry.last_validated_at, entry.decay_half_life_days)
        entry.effective_confidence = eff

        if eff < self.STALE_THRESHOLD:
            logger.debug("Cache hit for %s but stale (eff=%.3f)", pattern_sig, eff)
            return None

        logger.debug("Cache hit for %s (eff=%.3f)", pattern_sig, eff)
        return entry

    async def store(self, entry: CacheEntry) -> None:
        """Upsert a reasoning conclusion into the cache."""
        await self._pool.execute(
            """
            INSERT INTO reasoning_semantic_cache (
                pattern_sig, pattern_class, vendor_name, product_category,
                conclusion, confidence, reasoning_steps, boundary_conditions,
                falsification_conditions, uncertainty_sources,
                decay_half_life_days, conclusion_type, evidence_hash,
                last_validated_at, validation_count
            ) VALUES (
                $1, $2, $3, $4, $5::jsonb, $6, $7::jsonb, $8::jsonb,
                $9::jsonb, $10, $11, $12, $13, NOW(), 1
            )
            ON CONFLICT (pattern_sig) DO UPDATE SET
                pattern_class = EXCLUDED.pattern_class,
                vendor_name = EXCLUDED.vendor_name,
                product_category = EXCLUDED.product_category,
                conclusion = EXCLUDED.conclusion,
                confidence = EXCLUDED.confidence,
                reasoning_steps = EXCLUDED.reasoning_steps,
                boundary_conditions = EXCLUDED.boundary_conditions,
                falsification_conditions = EXCLUDED.falsification_conditions,
                uncertainty_sources = EXCLUDED.uncertainty_sources,
                decay_half_life_days = EXCLUDED.decay_half_life_days,
                conclusion_type = EXCLUDED.conclusion_type,
                evidence_hash = EXCLUDED.evidence_hash,
                last_validated_at = NOW(),
                validation_count = reasoning_semantic_cache.validation_count + 1,
                invalidated_at = NULL
            """,
            entry.pattern_sig,
            entry.pattern_class,
            entry.vendor_name,
            entry.product_category,
            json.dumps(entry.conclusion, default=str),
            entry.confidence,
            json.dumps(entry.reasoning_steps, default=str),
            json.dumps(entry.boundary_conditions, default=str),
            json.dumps(entry.falsification_conditions, default=str),
            entry.uncertainty_sources,
            entry.decay_half_life_days,
            entry.conclusion_type,
            entry.evidence_hash,
        )
        logger.info("Stored cache entry: %s (conf=%.2f)", entry.pattern_sig, entry.confidence)

    async def validate(self, pattern_sig: str, new_confidence: float | None = None) -> None:
        """Refresh an existing entry (bumps last_validated_at + count)."""
        if new_confidence is not None:
            await self._pool.execute(
                """
                UPDATE reasoning_semantic_cache
                SET last_validated_at = NOW(),
                    validation_count = validation_count + 1,
                    confidence = $2
                WHERE pattern_sig = $1 AND invalidated_at IS NULL
                """,
                pattern_sig,
                new_confidence,
            )
        else:
            await self._pool.execute(
                """
                UPDATE reasoning_semantic_cache
                SET last_validated_at = NOW(),
                    validation_count = validation_count + 1
                WHERE pattern_sig = $1 AND invalidated_at IS NULL
                """,
                pattern_sig,
            )

    async def invalidate(self, pattern_sig: str, reason: str = "") -> None:
        """Soft-delete by setting invalidated_at."""
        await self._pool.execute(
            """
            UPDATE reasoning_semantic_cache
            SET invalidated_at = NOW()
            WHERE pattern_sig = $1 AND invalidated_at IS NULL
            """,
            pattern_sig,
        )
        logger.info("Invalidated cache entry: %s (reason: %s)", pattern_sig, reason)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def lookup_by_class(
        self, pattern_class: str, vendor_name: str | None = None, limit: int = 20
    ) -> list[CacheEntry]:
        """Find all active entries for a pattern class (or by vendor if class is empty)."""
        if vendor_name and not pattern_class:
            # Search by vendor only (used by reconstitute to find any prior entry)
            rows = await self._pool.fetch(
                """
                SELECT * FROM reasoning_semantic_cache
                WHERE vendor_name = $1 AND invalidated_at IS NULL
                ORDER BY last_validated_at DESC
                LIMIT $2
                """,
                vendor_name,
                limit,
            )
        elif vendor_name:
            rows = await self._pool.fetch(
                """
                SELECT * FROM reasoning_semantic_cache
                WHERE pattern_class = $1 AND vendor_name = $2 AND invalidated_at IS NULL
                ORDER BY confidence DESC
                LIMIT $3
                """,
                pattern_class,
                vendor_name,
                limit,
            )
        else:
            rows = await self._pool.fetch(
                """
                SELECT * FROM reasoning_semantic_cache
                WHERE pattern_class = $1 AND invalidated_at IS NULL
                ORDER BY confidence DESC
                LIMIT $2
                """,
                pattern_class,
                limit,
            )
        entries = []
        for row in rows:
            e = self._row_to_entry(row)
            e.effective_confidence = _apply_decay(e.confidence, e.last_validated_at, e.decay_half_life_days)
            entries.append(e)
        return entries

    async def lookup_for_tier(
        self,
        conclusion_type: str,
        product_category: str | None = None,
        vendor_name: str | None = None,
        limit: int = 5,
    ) -> list[CacheEntry]:
        """Find active entries by conclusion_type, with optional category/vendor filter.

        Used by tier inheritance to find T4 ecosystem or T2 archetype entries.
        """
        if product_category and not vendor_name:
            rows = await self._pool.fetch(
                """
                SELECT * FROM reasoning_semantic_cache
                WHERE conclusion_type = $1 AND product_category = $2
                  AND invalidated_at IS NULL
                ORDER BY last_validated_at DESC
                LIMIT $3
                """,
                conclusion_type,
                product_category,
                limit,
            )
        elif vendor_name:
            rows = await self._pool.fetch(
                """
                SELECT * FROM reasoning_semantic_cache
                WHERE vendor_name = $1 AND invalidated_at IS NULL
                ORDER BY last_validated_at DESC
                LIMIT $2
                """,
                vendor_name,
                limit,
            )
        else:
            rows = await self._pool.fetch(
                """
                SELECT * FROM reasoning_semantic_cache
                WHERE conclusion_type = $1 AND invalidated_at IS NULL
                ORDER BY last_validated_at DESC
                LIMIT $2
                """,
                conclusion_type,
                limit,
            )
        entries = []
        for row in rows:
            e = self._row_to_entry(row)
            e.effective_confidence = _apply_decay(e.confidence, e.last_validated_at, e.decay_half_life_days)
            if e.effective_confidence >= self.STALE_THRESHOLD:
                entries.append(e)
        return entries

    async def get_cache_stats(self) -> dict[str, Any]:
        """Aggregate stats for metacognition tracking."""
        row = await self._pool.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE invalidated_at IS NULL) AS active,
                COUNT(*) FILTER (WHERE invalidated_at IS NOT NULL) AS invalidated,
                AVG(confidence) FILTER (WHERE invalidated_at IS NULL) AS avg_confidence,
                AVG(validation_count) FILTER (WHERE invalidated_at IS NULL) AS avg_validations
            FROM reasoning_semantic_cache
            """
        )
        if row is None:
            return {"active": 0, "invalidated": 0, "avg_confidence": 0, "avg_validations": 0}
        return dict(row)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row) -> CacheEntry:
        """Convert an asyncpg Record to a CacheEntry."""
        falsification = row["falsification_conditions"]
        if isinstance(falsification, str):
            falsification = json.loads(falsification)
        if isinstance(falsification, dict):
            falsification = list(falsification.values()) if falsification else []

        return CacheEntry(
            pattern_sig=row["pattern_sig"],
            pattern_class=row["pattern_class"],
            vendor_name=row["vendor_name"],
            product_category=row["product_category"],
            conclusion=row["conclusion"] if isinstance(row["conclusion"], dict) else json.loads(row["conclusion"]),
            confidence=row["confidence"],
            reasoning_steps=row["reasoning_steps"] if isinstance(row["reasoning_steps"], list) else json.loads(row["reasoning_steps"]),
            boundary_conditions=row["boundary_conditions"] if isinstance(row["boundary_conditions"], dict) else json.loads(row["boundary_conditions"]),
            falsification_conditions=falsification if isinstance(falsification, list) else [],
            uncertainty_sources=list(row["uncertainty_sources"]) if row["uncertainty_sources"] else [],
            decay_half_life_days=row["decay_half_life_days"],
            conclusion_type=row["conclusion_type"],
            evidence_hash=row["evidence_hash"],
            created_at=row["created_at"],
            last_validated_at=row["last_validated_at"],
            validation_count=row["validation_count"],
        )
