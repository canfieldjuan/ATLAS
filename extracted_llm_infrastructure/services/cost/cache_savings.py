"""Cache-savings persistence for the cost-closure surface.

Owned by the extracted LLM-infrastructure package -- not synced from
atlas_brain. Closes the "cache hits in memory only" telemetry gap by
writing one row per cache hit with the input/output tokens and cost
that the underlying LLM call would have incurred without the cache.
Rolls up for dashboards via ``daily_cache_savings``.

Public API:

    record_cache_hit(pool, *, cache_key, namespace, provider, model,
        would_have_been_input_tokens, would_have_been_output_tokens,
        would_have_been_cost_usd, attribution=None, metadata=None) -> None

    daily_cache_savings(pool, *, date_range, attribution_key=None)
        -> CacheSavingsRollup

    CacheSavingsRollup -- TypedDict with totals + per-namespace and
        per-attribution-dimension breakdowns.

Schema: ``storage/migrations/259_llm_cache_savings.sql`` (owned, not
back-ported to atlas_brain). One row per cache hit. ``attribution`` is
free-form JSONB; consumers index dimensions they care about with
PostgreSQL generated columns or partial indexes.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from decimal import Decimal
from typing import Any, Mapping, TypedDict

logger = logging.getLogger(__name__)


class CacheSavingsRollup(TypedDict):
    """Aggregated cache-savings over a date range.

    All cost values are ``Decimal`` so callers can sum/format without
    floating-point drift. Token counts are plain ``int``.
    """

    total_saved_usd: Decimal
    total_saved_input_tokens: int
    total_saved_output_tokens: int
    hit_count: int
    by_namespace: Mapping[str, Decimal]
    by_attribution_dim: Mapping[str, Mapping[str, Decimal]]


async def record_cache_hit(
    pool: Any,
    *,
    cache_key: str,
    namespace: str,
    provider: str,
    model: str,
    would_have_been_input_tokens: int,
    would_have_been_output_tokens: int,
    would_have_been_cost_usd: Decimal | float | int,
    attribution: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Persist one cache-savings row.

    The ``would_have_been_*`` fields are the cost the LLM call would
    have incurred without the cache. Together they enable
    "$ saved by cache last month" rollups.

    ``attribution`` is free-form: keys are strings, values are anything
    JSON-serializable. The schema column is JSONB so callers can add
    dimensions (numeric IDs, booleans, nested objects) without
    schema migrations.

    No-op when ``pool`` is ``None`` so callers can opt out without
    changing call sites. Failures (including invalid Decimal input or
    non-JSON-serializable attribution/metadata) are logged with the
    full stack trace + identifying fields and swallowed -- cache
    telemetry must not block the cache hit itself.
    """
    if pool is None:
        return

    try:
        cost_value = Decimal(str(would_have_been_cost_usd))
        attribution_json = json.dumps(dict(attribution) if attribution else {})
        metadata_json = json.dumps(dict(metadata) if metadata else {})
        await pool.execute(
            """
            INSERT INTO llm_cache_savings
                (cache_key, namespace, provider, model,
                 saved_input_tokens, saved_output_tokens, saved_cost_usd,
                 attribution, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb)
            """,
            cache_key,
            namespace,
            provider,
            model,
            int(would_have_been_input_tokens),
            int(would_have_been_output_tokens),
            cost_value,
            attribution_json,
            metadata_json,
        )
    except Exception:  # pragma: no cover -- defensive log path
        logger.exception(
            "cache savings record failed",
            extra={
                "cache_key": cache_key,
                "namespace": namespace,
                "provider": provider,
                "model": model,
            },
        )


async def daily_cache_savings(
    pool: Any,
    *,
    date_range: tuple[date, date],
    attribution_key: str | None = None,
) -> CacheSavingsRollup:
    """Aggregate ``llm_cache_savings`` over ``[start, end)``.

    ``date_range`` is ``(start, end)`` -- inclusive on start, exclusive
    on end so consecutive ranges do not double-count.

    ``attribution_key``, when set, includes a per-dimension-value
    rollup of saved-cost in ``by_attribution_dim[attribution_key]``.
    Useful for "saved per customer" dashboards without forcing every
    attribution dimension into the response.
    """
    start, end = date_range

    # Cast date params to TIMESTAMPTZ explicitly so the comparison is
    # not subject to the DB session timezone. ``date`` -> ``timestamp``
    # implicit casts in Postgres use the session timezone, which can
    # shift day boundaries depending on the connection's TZ. Casting to
    # TIMESTAMPTZ at midnight UTC anchors the range to UTC days.
    summary_rows = await pool.fetch(
        """
        SELECT
            COALESCE(SUM(saved_cost_usd), 0) AS total_cost,
            COALESCE(SUM(saved_input_tokens), 0)::BIGINT AS total_input,
            COALESCE(SUM(saved_output_tokens), 0)::BIGINT AS total_output,
            COUNT(*)::BIGINT AS hit_count
        FROM llm_cache_savings
        WHERE hit_at >= ($1::DATE)::TIMESTAMPTZ
          AND hit_at < ($2::DATE)::TIMESTAMPTZ
        """,
        start,
        end,
    )
    summary = summary_rows[0] if summary_rows else None

    namespace_rows = await pool.fetch(
        """
        SELECT namespace, COALESCE(SUM(saved_cost_usd), 0) AS cost
        FROM llm_cache_savings
        WHERE hit_at >= ($1::DATE)::TIMESTAMPTZ
          AND hit_at < ($2::DATE)::TIMESTAMPTZ
        GROUP BY namespace
        ORDER BY cost DESC
        """,
        start,
        end,
    )
    by_namespace = {
        row["namespace"]: Decimal(str(row["cost"])) for row in namespace_rows
    }

    by_attribution_dim: dict[str, Mapping[str, Decimal]] = {}
    if attribution_key:
        attribution_rows = await pool.fetch(
            """
            SELECT
                attribution ->> $3 AS dim_value,
                COALESCE(SUM(saved_cost_usd), 0) AS cost
            FROM llm_cache_savings
            WHERE hit_at >= ($1::DATE)::TIMESTAMPTZ
              AND hit_at < ($2::DATE)::TIMESTAMPTZ
              AND attribution ? $3
            GROUP BY dim_value
            ORDER BY cost DESC
            """,
            start,
            end,
            attribution_key,
        )
        by_attribution_dim[attribution_key] = {
            row["dim_value"]: Decimal(str(row["cost"]))
            for row in attribution_rows
            if row["dim_value"] is not None
        }

    if summary is None:
        return CacheSavingsRollup(
            total_saved_usd=Decimal(0),
            total_saved_input_tokens=0,
            total_saved_output_tokens=0,
            hit_count=0,
            by_namespace=by_namespace,
            by_attribution_dim=by_attribution_dim,
        )

    return CacheSavingsRollup(
        total_saved_usd=Decimal(str(summary["total_cost"])),
        total_saved_input_tokens=int(summary["total_input"] or 0),
        total_saved_output_tokens=int(summary["total_output"] or 0),
        hit_count=int(summary["hit_count"] or 0),
        by_namespace=by_namespace,
        by_attribution_dim=by_attribution_dim,
    )


__all__ = [
    "CacheSavingsRollup",
    "record_cache_hit",
    "daily_cache_savings",
]
