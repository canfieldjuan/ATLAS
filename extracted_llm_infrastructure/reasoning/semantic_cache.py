"""Postgres-backed semantic cache storage for the stratified reasoning engine.

PR-C2 (PR 4 from the reasoning boundary audit) split this module: the
*pure* primitives (``CacheEntry``, ``compute_evidence_hash``,
``apply_decay``, ``row_to_cache_entry``, ``STALE_THRESHOLD``) now live
in ``extracted_reasoning_core.semantic_cache_keys``. This module owns
the Postgres-specific storage class (``SemanticCache``) and the
asyncpg-shaped pool Protocol it consumes. ``CacheEntry`` /
``compute_evidence_hash`` / ``STALE_THRESHOLD`` are re-exported here
so existing callers keep working without changing imports.

Confidence decays exponentially since last validation:
    effective = confidence * 2^(-(days_since_validated / decay_half_life_days))

The decay is computed by ``apply_decay`` from the core module; this
adapter just calls it on every row read out of Postgres.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from extracted_reasoning_core.semantic_cache_keys import (
    CacheEntry,
    STALE_THRESHOLD as _CORE_STALE_THRESHOLD,
    apply_decay as _apply_decay,
    compute_evidence_hash,
    row_to_cache_entry,
)

# Re-export as a module-level name so callers doing
# ``from <this module> import STALE_THRESHOLD`` keep working. The
# atlas-side and LLM-infra-mirror copies of this file are validated
# byte-for-byte, so the wording is module-path-agnostic; the canonical
# constant lives in ``extracted_reasoning_core.semantic_cache_keys``.
# Aliased on import to avoid the ``STALE_THRESHOLD = STALE_THRESHOLD``
# self-reference shadowing that made the prior shape look like a no-op.
STALE_THRESHOLD = _CORE_STALE_THRESHOLD

logger = logging.getLogger("atlas.reasoning.semantic_cache")


class SemanticCachePool(Protocol):
    """Async pool contract that ``SemanticCache`` requires.

    Any object exposing these three coroutine methods works -- atlas's
    ``DatabasePool`` wrapper, an asyncpg ``Pool`` directly, or a test
    fake that emulates the same shape. The Protocol declares only the
    surface this module touches; the SQL itself remains
    Postgres-specific (``::jsonb``, ``ON CONFLICT``, ``NOW()``,
    ``INTERVAL``) so the pool is expected to speak Postgres dialect.

    ``fetchrow`` and ``fetch`` return row mappings (the production
    binding returns ``asyncpg.Record`` instances that subscript like
    dicts; ``row_to_cache_entry`` reads them via ``row[key]`` and
    accepts anything that supports the ``Mapping`` protocol).
    """

    async def fetchrow(self, query: str, *args: Any) -> Any: ...
    async def fetch(self, query: str, *args: Any) -> Any: ...
    async def execute(self, query: str, *args: Any) -> Any: ...


class SemanticCache:
    """Postgres-backed semantic memory for cached reasoning conclusions.

    Implements the ``SemanticCacheStore`` port declared in
    ``extracted_reasoning_core.ports``: ``lookup`` / ``store`` /
    ``validate`` / ``invalidate`` plus the read-side helpers
    ``lookup_by_class`` / ``lookup_for_tier`` / ``get_cache_stats``
    that callers reach for directly.
    """

    # Re-exported as a class attribute for backward-compat with callers
    # that read ``SemanticCache.STALE_THRESHOLD``. The module-level
    # constant in core's ``semantic_cache_keys`` is the canonical home.
    STALE_THRESHOLD = _CORE_STALE_THRESHOLD

    # Sentinel account UUID for atlas's internal pipeline (PR-D3).
    # Atlas's existing reasoning calls write to the cache without
    # knowing about accounts; the sentinel marks those rows so
    # customer cache hits cannot leak across tenants.
    SENTINEL_ACCOUNT_ID = "00000000-0000-0000-0000-000000000000"

    def __init__(
        self,
        pool: SemanticCachePool,
        *,
        account_id: str = SENTINEL_ACCOUNT_ID,
    ):
        """*pool*: any object exposing the ``SemanticCachePool``
        contract (``fetchrow`` / ``fetch`` / ``execute`` coroutine
        methods that speak Postgres dialect). The atlas
        ``DatabasePool`` wrapper and a raw asyncpg ``Pool`` both
        satisfy this Protocol.

        *account_id*: scopes every read/write to the given account
        (PR-D3). Defaults to the SENTINEL so atlas's existing
        instantiations keep working unmodified. Customer-facing
        callers (PR-D4 LLM Gateway router) construct a new instance
        per request with the requesting account's UUID, so cross-
        tenant cache hits are impossible -- the (pattern_sig,
        account_id) UNIQUE constraint guarantees isolation at the
        storage layer.
        """
        self._pool = pool
        self._account_id = account_id

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
            WHERE pattern_sig = $1 AND account_id = $2
              AND invalidated_at IS NULL
            """,
            pattern_sig,
            self._account_id,
        )
        if row is None:
            return None

        entry = row_to_cache_entry(row)
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
                pattern_sig, account_id, pattern_class, vendor_name,
                product_category, conclusion, confidence, reasoning_steps,
                boundary_conditions, falsification_conditions,
                uncertainty_sources, decay_half_life_days, conclusion_type,
                evidence_hash, last_validated_at, validation_count
            ) VALUES (
                $1, $2, $3, $4, $5, $6::jsonb, $7, $8::jsonb, $9::jsonb,
                $10::jsonb, $11, $12, $13, $14, NOW(), 1
            )
            ON CONFLICT (pattern_sig, account_id) DO UPDATE SET
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
            self._account_id,
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
                    confidence = $3
                WHERE pattern_sig = $1 AND account_id = $2
                  AND invalidated_at IS NULL
                """,
                pattern_sig,
                self._account_id,
                new_confidence,
            )
        else:
            await self._pool.execute(
                """
                UPDATE reasoning_semantic_cache
                SET last_validated_at = NOW(),
                    validation_count = validation_count + 1
                WHERE pattern_sig = $1 AND account_id = $2
                  AND invalidated_at IS NULL
                """,
                pattern_sig,
                self._account_id,
            )

    async def invalidate(self, pattern_sig: str, reason: str = "") -> None:
        """Soft-delete by setting invalidated_at."""
        await self._pool.execute(
            """
            UPDATE reasoning_semantic_cache
            SET invalidated_at = NOW()
            WHERE pattern_sig = $1 AND account_id = $2
              AND invalidated_at IS NULL
            """,
            pattern_sig,
            self._account_id,
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
                WHERE vendor_name = $1 AND account_id = $2
                  AND invalidated_at IS NULL
                ORDER BY last_validated_at DESC
                LIMIT $3
                """,
                vendor_name,
                self._account_id,
                limit,
            )
        elif vendor_name:
            rows = await self._pool.fetch(
                """
                SELECT * FROM reasoning_semantic_cache
                WHERE pattern_class = $1 AND vendor_name = $2
                  AND account_id = $3 AND invalidated_at IS NULL
                ORDER BY confidence DESC
                LIMIT $4
                """,
                pattern_class,
                vendor_name,
                self._account_id,
                limit,
            )
        else:
            rows = await self._pool.fetch(
                """
                SELECT * FROM reasoning_semantic_cache
                WHERE pattern_class = $1 AND account_id = $2
                  AND invalidated_at IS NULL
                ORDER BY confidence DESC
                LIMIT $3
                """,
                pattern_class,
                self._account_id,
                limit,
            )
        entries = []
        for row in rows:
            e = row_to_cache_entry(row)
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
                  AND account_id = $3 AND invalidated_at IS NULL
                ORDER BY last_validated_at DESC
                LIMIT $4
                """,
                conclusion_type,
                product_category,
                self._account_id,
                limit,
            )
        elif vendor_name:
            rows = await self._pool.fetch(
                """
                SELECT * FROM reasoning_semantic_cache
                WHERE vendor_name = $1 AND account_id = $2
                  AND invalidated_at IS NULL
                ORDER BY last_validated_at DESC
                LIMIT $3
                """,
                vendor_name,
                self._account_id,
                limit,
            )
        else:
            rows = await self._pool.fetch(
                """
                SELECT * FROM reasoning_semantic_cache
                WHERE conclusion_type = $1 AND account_id = $2
                  AND invalidated_at IS NULL
                ORDER BY last_validated_at DESC
                LIMIT $3
                """,
                conclusion_type,
                self._account_id,
                limit,
            )
        entries = []
        for row in rows:
            e = row_to_cache_entry(row)
            e.effective_confidence = _apply_decay(e.confidence, e.last_validated_at, e.decay_half_life_days)
            if e.effective_confidence >= self.STALE_THRESHOLD:
                entries.append(e)
        return entries

    async def get_cache_stats(self) -> dict[str, Any]:
        """Aggregate stats for metacognition tracking.

        Scoped to ``self._account_id`` so tenant A's stats never
        include tenant B's rows.
        """
        row = await self._pool.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE invalidated_at IS NULL) AS active,
                COUNT(*) FILTER (WHERE invalidated_at IS NOT NULL) AS invalidated,
                AVG(confidence) FILTER (WHERE invalidated_at IS NULL) AS avg_confidence,
                AVG(validation_count) FILTER (WHERE invalidated_at IS NULL) AS avg_validations
            FROM reasoning_semantic_cache
            WHERE account_id = $1
            """,
            self._account_id,
        )
        if row is None:
            return {"active": 0, "invalidated": 0, "avg_confidence": 0, "avg_validations": 0}
        return dict(row)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    #
    # Row-to-entry coercion now lives in
    # ``extracted_reasoning_core.semantic_cache_keys.row_to_cache_entry``
    # (PR-C2). The static method that used to live here has been
    # removed; callers below import the function directly.
