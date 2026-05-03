"""Provider billing sync for cost reconciliation."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from ..config import settings
from .llm.openrouter import _resolve_openrouter_api_key

logger = logging.getLogger("atlas.services.provider_cost_sync")

_OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"
_ANTHROPIC_COST_REPORT_URL = "https://api.anthropic.com/v1/organizations/cost_report"
_ANTHROPIC_VERSION = "2023-06-01"


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _resolve_openrouter_management_key() -> str:
    override = str(getattr(settings.provider_cost, "openrouter_api_key", "") or "").strip()
    if override:
        return override
    return _resolve_openrouter_api_key()


def _resolve_anthropic_admin_key() -> str:
    override = str(getattr(settings.provider_cost, "anthropic_admin_api_key", "") or "").strip()
    if override:
        return override
    return str(os.environ.get("ANTHROPIC_ADMIN_API_KEY", "") or "").strip()


async def _insert_openrouter_snapshot(
    pool,
    *,
    snapshot_at: datetime,
    total_usage_usd: float,
    total_credits_usd: float | None,
    raw_payload: dict[str, Any],
) -> None:
    await pool.execute(
        """
        INSERT INTO llm_provider_usage_snapshots (
            provider,
            snapshot_at,
            total_usage_usd,
            total_credits_usd,
            raw_payload
        )
        VALUES ($1, $2, $3, $4, $5::jsonb)
        """,
        "openrouter",
        snapshot_at,
        total_usage_usd,
        total_credits_usd,
        json.dumps(raw_payload),
    )


async def _upsert_anthropic_daily_cost(
    pool,
    *,
    cost_date: str,
    provider_cost_usd: float,
    currency: str,
    raw_payload: dict[str, Any],
) -> None:
    await pool.execute(
        """
        INSERT INTO llm_provider_daily_costs (
            provider,
            cost_date,
            provider_cost_usd,
            currency,
            source_kind,
            raw_payload,
            imported_at
        )
        VALUES ($1, $2::date, $3, $4, $5, $6::jsonb, NOW())
        ON CONFLICT (provider, cost_date, source_kind)
        DO UPDATE SET
            provider_cost_usd = EXCLUDED.provider_cost_usd,
            currency = EXCLUDED.currency,
            raw_payload = EXCLUDED.raw_payload,
            imported_at = NOW()
        """,
        "anthropic",
        cost_date,
        provider_cost_usd,
        currency or "USD",
        "anthropic_cost_report",
        json.dumps(raw_payload),
    )


async def _cleanup_old_rows(pool) -> None:
    now = datetime.now(timezone.utc)
    snapshot_cutoff = now - timedelta(days=int(settings.provider_cost.snapshot_retention_days))
    daily_cutoff = (now - timedelta(days=int(settings.provider_cost.daily_retention_days))).date()
    await pool.execute(
        """
        DELETE FROM llm_provider_usage_snapshots
        WHERE snapshot_at < $1
        """,
        snapshot_cutoff,
    )
    await pool.execute(
        """
        DELETE FROM llm_provider_daily_costs
        WHERE cost_date < $1::date
        """,
        daily_cutoff,
    )


async def _fetch_openrouter_credits_snapshot(
    client: httpx.AsyncClient,
    *,
    api_key: str,
) -> dict[str, Any]:
    response = await client.get(
        _OPENROUTER_CREDITS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else {}
    if not isinstance(data, dict):
        data = {}
    return {
        "provider": "openrouter",
        "snapshot_at": datetime.now(timezone.utc),
        "total_usage_usd": _safe_float(data.get("total_usage")),
        "total_credits_usd": _safe_float(data.get("total_credits")) if data.get("total_credits") is not None else None,
        "raw_payload": payload if isinstance(payload, dict) else {"raw": payload},
    }


async def _fetch_anthropic_daily_costs(
    client: httpx.AsyncClient,
    *,
    admin_api_key: str,
    lookback_days: int,
) -> list[dict[str, Any]]:
    ending_at = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    starting_at = (ending_at - timedelta(days=lookback_days)).replace(hour=0)
    rows: dict[str, dict[str, Any]] = {}
    page: str | None = None

    while True:
        params: dict[str, Any] = {
            "starting_at": starting_at.isoformat().replace("+00:00", "Z"),
            "ending_at": ending_at.isoformat().replace("+00:00", "Z"),
            "bucket_width": "1d",
            "limit": min(max(lookback_days, 1), 31),
        }
        if page:
            params["page"] = page
        response = await client.get(
            _ANTHROPIC_COST_REPORT_URL,
            params=params,
            headers={
                "x-api-key": admin_api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else []
        if not isinstance(data, list):
            data = []
        for bucket in data:
            if not isinstance(bucket, dict):
                continue
            bucket_start = str(bucket.get("starting_at") or "").strip()
            if not bucket_start:
                continue
            cost_date = bucket_start[:10]
            results = bucket.get("results")
            total_cost = 0.0
            currency = "USD"
            if isinstance(results, list):
                for entry in results:
                    if not isinstance(entry, dict):
                        continue
                    total_cost += _safe_float(entry.get("amount"))
                    currency = str(entry.get("currency") or currency or "USD")
            rows[cost_date] = {
                "cost_date": cost_date,
                "provider_cost_usd": round(total_cost, 6),
                "currency": currency or "USD",
                "raw_payload": bucket,
            }
        has_more = bool(payload.get("has_more")) if isinstance(payload, dict) else False
        next_page = payload.get("next_page") if isinstance(payload, dict) else None
        if not has_more or not next_page:
            break
        page = str(next_page)

    return [rows[key] for key in sorted(rows.keys())]


async def sync_provider_costs(*, pool, client: httpx.AsyncClient | None = None) -> dict[str, Any]:
    """Sync provider billing totals into local reconciliation tables."""
    cfg = settings.provider_cost
    if not bool(cfg.enabled):
        return {"_skip_synthesis": "Provider cost sync disabled"}

    timeout = float(getattr(cfg, "sync_timeout_seconds", 20) or 20)
    summary: dict[str, Any] = {
        "openrouter_snapshot_written": False,
        "anthropic_daily_rows_upserted": 0,
        "providers_synced": [],
        "errors": [],
    }

    async def _run(active_client: httpx.AsyncClient) -> None:
        if bool(cfg.openrouter_enabled):
            api_key = _resolve_openrouter_management_key()
            if api_key:
                try:
                    snapshot = await _fetch_openrouter_credits_snapshot(active_client, api_key=api_key)
                    await _insert_openrouter_snapshot(
                        pool,
                        snapshot_at=snapshot["snapshot_at"],
                        total_usage_usd=snapshot["total_usage_usd"],
                        total_credits_usd=snapshot["total_credits_usd"],
                        raw_payload=snapshot["raw_payload"],
                    )
                    summary["openrouter_snapshot_written"] = True
                    summary["providers_synced"].append("openrouter")
                except Exception as exc:
                    logger.warning("provider_cost_sync.openrouter_failed: %s", exc)
                    summary["errors"].append(f"openrouter:{type(exc).__name__}")
            else:
                summary["errors"].append("openrouter:missing_api_key")

        if bool(cfg.anthropic_enabled):
            admin_api_key = _resolve_anthropic_admin_key()
            if admin_api_key:
                try:
                    rows = await _fetch_anthropic_daily_costs(
                        active_client,
                        admin_api_key=admin_api_key,
                        lookback_days=int(getattr(cfg, "anthropic_lookback_days", 7) or 7),
                    )
                    for row in rows:
                        await _upsert_anthropic_daily_cost(
                            pool,
                            cost_date=row["cost_date"],
                            provider_cost_usd=row["provider_cost_usd"],
                            currency=row["currency"],
                            raw_payload=row["raw_payload"],
                        )
                    if rows:
                        summary["providers_synced"].append("anthropic")
                    summary["anthropic_daily_rows_upserted"] = len(rows)
                except Exception as exc:
                    logger.warning("provider_cost_sync.anthropic_failed: %s", exc)
                    summary["errors"].append(f"anthropic:{type(exc).__name__}")
            else:
                summary["errors"].append("anthropic:missing_admin_api_key")

    if client is not None:
        await _run(client)
    else:
        async with httpx.AsyncClient(timeout=timeout) as active_client:
            await _run(active_client)

    await _cleanup_old_rows(pool)

    if not summary["providers_synced"] and not summary["openrouter_snapshot_written"] and not summary["anthropic_daily_rows_upserted"]:
        summary["_skip_synthesis"] = "No provider cost data synced"
    return summary
