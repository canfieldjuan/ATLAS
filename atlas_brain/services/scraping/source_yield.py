"""Utilities for disabling scrape targets that repeatedly produce no yield."""

from __future__ import annotations

import json
from typing import Any


def _normalize_source(value: str) -> str:
    source = str(value or "").strip().lower()
    if not source:
        raise ValueError("source is required")
    return source


def _normalize_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


async def select_low_yield_targets(
    pool,
    *,
    source: str,
    lookback_runs: int,
    min_runs: int,
    max_inserted_total: int,
    enabled_only: bool = True,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Return targets whose recent runs inserted at most max_inserted_total reviews."""
    source_norm = _normalize_source(source)
    lookback = max(1, int(lookback_runs))
    min_observed = max(1, int(min_runs))
    max_inserted = max(0, int(max_inserted_total))
    max_rows = max(1, int(limit))

    rows = await pool.fetch(
        """
        WITH recent_logs AS (
            SELECT
                t.id AS target_id,
                t.source,
                t.vendor_name,
                t.product_slug,
                t.product_category,
                t.enabled,
                t.metadata,
                l.status,
                l.reviews_inserted,
                l.started_at,
                ROW_NUMBER() OVER (
                    PARTITION BY t.id
                    ORDER BY l.started_at DESC, l.id DESC
                ) AS rn
            FROM b2b_scrape_targets t
            LEFT JOIN b2b_scrape_log l ON l.target_id = t.id
            WHERE t.source = $1
              AND ($2::boolean = false OR t.enabled = true)
        ),
        aggregated AS (
            SELECT
                target_id,
                source,
                vendor_name,
                product_slug,
                product_category,
                enabled,
                metadata,
                COUNT(*) FILTER (
                    WHERE rn <= $3 AND status IS NOT NULL
                ) AS runs_observed,
                COALESCE(SUM(COALESCE(reviews_inserted, 0)) FILTER (
                    WHERE rn <= $3
                ), 0) AS inserted_sum,
                MAX(started_at) FILTER (WHERE rn <= $3) AS last_run_at,
                ARRAY_AGG(status ORDER BY rn) FILTER (
                    WHERE rn <= $3 AND status IS NOT NULL
                ) AS statuses
            FROM recent_logs
            GROUP BY
                target_id, source, vendor_name, product_slug,
                product_category, enabled, metadata
        )
        SELECT
            target_id,
            source,
            vendor_name,
            product_slug,
            product_category,
            enabled,
            metadata,
            runs_observed,
            inserted_sum,
            last_run_at,
            statuses
        FROM aggregated
        WHERE runs_observed >= $4
          AND inserted_sum <= $5
        ORDER BY
            inserted_sum ASC,
            runs_observed DESC,
            last_run_at DESC NULLS LAST,
            vendor_name ASC
        LIMIT $6
        """,
        source_norm,
        enabled_only,
        lookback,
        min_observed,
        max_inserted,
        max_rows,
    )

    candidates: list[dict[str, Any]] = []
    for row in rows:
        statuses = row["statuses"] or []
        item = {
            "target_id": str(row["target_id"]),
            "source": row["source"],
            "vendor_name": row["vendor_name"],
            "product_slug": row["product_slug"],
            "product_category": row["product_category"],
            "enabled": bool(row["enabled"]),
            "metadata": _normalize_metadata(row["metadata"]),
            "runs_observed": int(row["runs_observed"] or 0),
            "inserted_sum": int(row["inserted_sum"] or 0),
            "last_run_at": row["last_run_at"],
            "statuses": [str(status) for status in statuses if status],
        }
        candidates.append(item)
    return candidates


async def apply_low_yield_policy(
    pool,
    *,
    candidates: list[dict[str, Any]],
    policy_name: str,
) -> int:
    """Disable selected scrape targets and stamp policy metadata."""
    disabled = 0
    for item in candidates:
        result = await pool.execute(
            """
            UPDATE b2b_scrape_targets
            SET enabled = false,
                metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object(
                    'disabled_by_policy', true,
                    'disabled_policy', $2,
                    'disabled_policy_at', NOW()::text,
                    'disabled_policy_runs_observed', $3,
                    'disabled_policy_inserted_sum', $4
                ),
                updated_at = NOW()
            WHERE id = $1::uuid
              AND enabled = true
            """,
            str(item["target_id"]),
            policy_name,
            int(item.get("runs_observed") or 0),
            int(item.get("inserted_sum") or 0),
        )
        if isinstance(result, str) and result.endswith("1"):
            disabled += 1
    return disabled


async def prune_low_yield_targets(
    pool,
    *,
    source: str,
    lookback_runs: int,
    min_runs: int,
    max_inserted_total: int,
    max_disable_per_run: int,
    dry_run: bool = True,
    enabled_only: bool = True,
    policy_name: str | None = None,
) -> dict[str, Any]:
    """Select and optionally disable low-yield targets for a source."""
    source_norm = _normalize_source(source)
    lookback = max(1, int(lookback_runs))
    min_observed = max(1, int(min_runs))
    max_inserted = max(0, int(max_inserted_total))
    max_disable = max(1, int(max_disable_per_run))
    selected_policy = str(policy_name or f"{source_norm}_low_yield_last{lookback}runs")

    candidates = await select_low_yield_targets(
        pool,
        source=source_norm,
        lookback_runs=lookback,
        min_runs=min_observed,
        max_inserted_total=max_inserted,
        enabled_only=enabled_only,
        limit=max_disable,
    )
    disabled = len(candidates)
    if not dry_run and candidates:
        disabled = await apply_low_yield_policy(
            pool,
            candidates=candidates,
            policy_name=selected_policy,
        )

    return {
        "dry_run": bool(dry_run),
        "source": source_norm,
        "policy_name": selected_policy,
        "lookback_runs": lookback,
        "min_runs": min_observed,
        "max_inserted_total": max_inserted,
        "max_disable_per_run": max_disable,
        "requested": len(candidates),
        "disabled": disabled,
        "targets": candidates,
    }
