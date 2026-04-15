"""Utilities for disabling scrape targets that repeatedly produce no yield."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from .sources import parse_source_allowlist


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


def _effective_min_runs(source: str, requested_min_runs: int) -> int:
    source_norm = _normalize_source(source)
    high_yield = set(
        parse_source_allowlist(
            getattr(settings.b2b_scrape, "high_yield_priority_sources", "")
        )
    )
    context_rich = set(
        parse_source_allowlist(
            getattr(settings.b2b_scrape, "context_rich_priority_sources", "")
        )
    )
    if source_norm in high_yield:
        return max(
            int(requested_min_runs),
            int(
                getattr(
                    settings.b2b_scrape,
                    "source_low_yield_pruning_high_yield_min_runs_floor",
                    requested_min_runs,
                )
            ),
        )
    if source_norm in context_rich:
        return max(
            int(requested_min_runs),
            int(
                getattr(
                    settings.b2b_scrape,
                    "source_low_yield_pruning_context_rich_min_runs_floor",
                    requested_min_runs,
                )
            ),
        )
    return int(requested_min_runs)


def _source_yield_tier(source: str) -> str:
    source_norm = _normalize_source(source)
    high_yield = set(
        parse_source_allowlist(
            getattr(settings.b2b_scrape, "high_yield_priority_sources", "")
        )
    )
    context_rich = set(
        parse_source_allowlist(
            getattr(settings.b2b_scrape, "context_rich_priority_sources", "")
        )
    )
    if source_norm in high_yield:
        return "high_yield"
    if source_norm in context_rich:
        return "context_rich"
    return "standard"


def _policy_summary(
    *,
    source: str,
    source_tier: str,
    lookback_runs: int,
    requested_min_runs: int,
    effective_min_runs: int,
    max_inserted_total: int,
) -> str:
    tier_label = source_tier.replace("_", " ")
    if effective_min_runs > requested_min_runs:
        return (
            f"{source} is a {tier_label} source, so the pruning floor rose from "
            f"{requested_min_runs} to {effective_min_runs} runs across the last "
            f"{lookback_runs} runs, with at most {max_inserted_total} inserted reviews."
        )
    return (
        f"{source} uses the standard pruning floor of {effective_min_runs} runs across "
        f"the last {lookback_runs} runs, with at most {max_inserted_total} inserted reviews."
    )


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
    min_observed = max(1, _effective_min_runs(source_norm, int(min_runs)))
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


async def apply_disable_policy(
    pool,
    *,
    candidates: list[dict[str, Any]],
    policy_name: str,
    metadata_builder,
) -> int:
    """Disable selected scrape targets with a caller-supplied metadata payload."""
    disabled = 0
    for item in candidates:
        metadata_payload = metadata_builder(item)
        result = await pool.execute(
            """
            UPDATE b2b_scrape_targets
            SET enabled = false,
                metadata = COALESCE(metadata, '{}'::jsonb) || $2::jsonb,
                updated_at = NOW()
            WHERE id = $1::uuid
              AND enabled = true
            """,
            str(item["target_id"]),
            json.dumps(metadata_payload, default=str),
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
    requested_min_runs = max(1, int(min_runs))
    min_observed = max(1, _effective_min_runs(source_norm, requested_min_runs))
    max_inserted = max(0, int(max_inserted_total))
    max_disable = max(1, int(max_disable_per_run))
    selected_policy = str(policy_name or f"{source_norm}_low_yield_last{lookback}runs")
    source_tier = _source_yield_tier(source_norm)
    policy_summary = _policy_summary(
        source=source_norm,
        source_tier=source_tier,
        lookback_runs=lookback,
        requested_min_runs=requested_min_runs,
        effective_min_runs=min_observed,
        max_inserted_total=max_inserted,
    )

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
        "source_tier": source_tier,
        "policy_name": selected_policy,
        "lookback_runs": lookback,
        "min_runs": min_observed,
        "requested_min_runs": requested_min_runs,
        "min_runs_floor_applied": min_observed > requested_min_runs,
        "max_inserted_total": max_inserted,
        "max_disable_per_run": max_disable,
        "policy_context": {
            "source_tier": source_tier,
            "requested_min_runs": requested_min_runs,
            "effective_min_runs": min_observed,
            "min_runs_floor_applied": min_observed > requested_min_runs,
            "lookback_runs": lookback,
            "max_inserted_total": max_inserted,
            "max_disable_per_run": max_disable,
        },
        "policy_summary": policy_summary,
        "requested": len(candidates),
        "disabled": disabled,
        "targets": candidates,
    }


async def disable_persistently_blocked_targets(
    pool,
    *,
    min_blocked_age_hours: int,
    max_disable_per_run: int,
    dry_run: bool = True,
    sources: list[str] | None = None,
    vendors: list[str] | None = None,
) -> dict[str, Any]:
    """Select and optionally disable enabled targets that remain blocked past a cooldown."""
    min_age = max(1, int(min_blocked_age_hours))
    max_disable = max(1, int(max_disable_per_run))
    source_filters = {_normalize_source(item) for item in (sources or []) if str(item).strip()}
    vendor_filters = {str(item).strip().lower() for item in (vendors or []) if str(item).strip()}
    rows = await pool.fetch(
        """
        SELECT id AS target_id,
               source,
               vendor_name,
               product_name,
               product_slug,
               product_category,
               priority,
               max_pages,
               scrape_mode,
               last_scraped_at,
               last_scrape_status,
               metadata
        FROM b2b_scrape_targets
        WHERE enabled = true
          AND last_scrape_status = 'blocked'
          AND last_scraped_at IS NOT NULL
          AND last_scraped_at <= NOW() - make_interval(hours => $1::int)
        ORDER BY priority DESC, last_scraped_at ASC, vendor_name ASC, id ASC
        LIMIT $2::int
        """,
        min_age,
        max_disable,
    )
    candidates: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        source_norm = _normalize_source(item.get("source"))
        vendor_norm = str(item.get("vendor_name") or "").strip().lower()
        if source_filters and source_norm not in source_filters:
            continue
        if vendor_filters and vendor_norm not in vendor_filters:
            continue
        item["source_tier"] = _source_yield_tier(source_norm)
        candidates.append(item)

    policy_name = "persistently_blocked_target"
    disabled = len(candidates)
    if not dry_run and candidates:
        def _build_metadata(item: dict[str, Any]) -> dict[str, Any]:
            return {
                "disabled_by_policy": True,
                "disabled_policy": policy_name,
                "disabled_policy_at": datetime.now(timezone.utc).isoformat(),
                "disabled_policy_last_scrape_status": str(item.get("last_scrape_status") or ""),
                "disabled_policy_last_scraped_at": str(item.get("last_scraped_at") or ""),
                "disabled_policy_min_blocked_age_hours": min_age,
            }

        disabled = await apply_disable_policy(
            pool,
            candidates=candidates,
            policy_name=policy_name,
            metadata_builder=_build_metadata,
        )

    return {
        "dry_run": bool(dry_run),
        "policy_name": policy_name,
        "min_blocked_age_hours": min_age,
        "max_disable_per_run": max_disable,
        "requested": len(candidates),
        "disabled": disabled,
        "targets": candidates,
    }
