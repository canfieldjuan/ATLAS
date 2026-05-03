"""OpenAI Costs API billing fetcher.

Owned by the extracted LLM-infrastructure package -- not synced from
atlas_brain. Adds a third provider to the cost-closure surface
alongside OpenRouter (credits snapshots) and Anthropic (admin daily
cost reports). OpenAI exposes an Organization Costs endpoint that
returns daily usage broken down by line items; this module fetches
those rows and upserts them into ``llm_provider_daily_costs`` so the
drift report can reconcile against locally-tracked spend.

Public API:

    sync_openai_daily_costs(*, pool, client=None,
        days_back=DEFAULT_DAYS_BACK,
        api_key=None) -> dict[str, Any]

    fetch_openai_daily_costs(*, client, api_key,
        days_back=DEFAULT_DAYS_BACK) -> list[OpenAIDailyCost]

    OpenAIDailyCost -- frozen dataclass per (cost_date,
    provider_cost_usd, raw_payload).

Why this is owned, not synced from atlas_brain
----------------------------------------------
``provider_cost_sync.py`` currently covers OpenRouter + Anthropic.
Adding OpenAI to that file would mean modifying a synced file, which
breaks the Phase 1 byte-for-byte contract. This module is the
extraction-owned sibling: it composes with the existing fetchers via
the same target table (``llm_provider_daily_costs``) but lives in
``services/cost/`` so it can evolve independently. A future Phase 3
unification can pull it into a single orchestrator behind a
``ProviderBillingPort`` Protocol.

Settings
--------
The OpenAI admin API key is resolved in priority order:

1. Explicit ``api_key=`` argument.
2. ``settings.provider_cost.openai_admin_api_key`` (if the standalone
   ``ProviderCostSubConfig`` has the field; falls through if not).
3. ``OPENAI_ADMIN_API_KEY`` environment variable.

This mirrors the resolution pattern in ``provider_cost_sync.py`` for
OpenRouter and Anthropic, including the ``getattr(settings, ..., "")``
fallback so a missing settings field does not raise.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_DAYS_BACK = 7
_OPENAI_COSTS_URL = "https://api.openai.com/v1/organization/costs"
_DEFAULT_TIMEOUT_SECONDS = 30.0
_PROVIDER_NAME = "openai"
_SOURCE_KIND = "openai_organization_costs"


@dataclass(frozen=True)
class OpenAIDailyCost:
    """One day of invoiced spend from the OpenAI Costs API.

    ``raw_payload`` preserves the original line-item shape so future
    breakdowns (per-model, per-line-item) can be derived without a
    second API call.
    """

    cost_date: date
    provider_cost_usd: Decimal
    currency: str = "USD"
    raw_payload: dict[str, Any] = field(default_factory=dict)


def _resolve_openai_admin_key(explicit: str | None = None) -> str:
    if explicit:
        return explicit.strip()
    try:
        from ...config import settings  # noqa: WPS433 -- lazy
        override = str(
            getattr(settings.provider_cost, "openai_admin_api_key", "") or ""
        ).strip()
        if override:
            return override
    except Exception:
        # Settings may not be configured (standalone with missing
        # provider_cost section). Fall through to env.
        pass
    return (os.environ.get("OPENAI_ADMIN_API_KEY") or "").strip()


def _safe_decimal(value: Any) -> Decimal:
    if value is None:
        return Decimal(0)
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal(0)


def _parse_cost_rows(payload: dict[str, Any]) -> list[OpenAIDailyCost]:
    """Translate one page of the OpenAI Costs API response into rows.

    The Costs API returns ``data: [{start_time, end_time, results: [
    {amount: {value, currency}, ...}, ...]}]``. We sum ``amount.value``
    per bucket so we have one ``OpenAIDailyCost`` per day.
    """
    rows: list[OpenAIDailyCost] = []
    for bucket in payload.get("data") or []:
        if not isinstance(bucket, dict):
            continue
        start_time = bucket.get("start_time")
        if start_time is None:
            continue
        # OpenAI returns Unix epoch seconds (UTC) for bucket start.
        try:
            cost_date = datetime.fromtimestamp(
                int(start_time), tz=timezone.utc
            ).date()
        except Exception:
            continue
        bucket_total = Decimal(0)
        currency = "USD"
        for result in bucket.get("results") or []:
            if not isinstance(result, dict):
                continue
            amount = result.get("amount") or {}
            bucket_total += _safe_decimal(amount.get("value"))
            currency = str(amount.get("currency") or currency)
        rows.append(
            OpenAIDailyCost(
                cost_date=cost_date,
                provider_cost_usd=bucket_total,
                currency=currency,
                raw_payload=bucket,
            )
        )
    return rows


async def fetch_openai_daily_costs(
    *,
    client: httpx.AsyncClient,
    api_key: str,
    days_back: int = DEFAULT_DAYS_BACK,
) -> list[OpenAIDailyCost]:
    """Fetch the last ``days_back`` days of OpenAI org costs.

    Returns an empty list if ``api_key`` is empty so callers do not
    need to gate the call on key presence; the orchestrator
    ``sync_openai_daily_costs`` does its own enabled check up front
    and only calls this when the key resolves.
    """
    if not api_key:
        return []
    if days_back <= 0:
        return []

    end = datetime.now(tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = end - timedelta(days=int(days_back))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    params = {
        "start_time": int(start.timestamp()),
        "end_time": int(end.timestamp()),
        "bucket_width": "1d",
    }
    try:
        response = await client.get(
            _OPENAI_COSTS_URL,
            headers=headers,
            params=params,
            timeout=_DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json() if response.content else {}
    except Exception:
        logger.exception("OpenAI Costs API fetch failed")
        return []

    if not isinstance(payload, dict):
        return []
    return _parse_cost_rows(payload)


async def _upsert_openai_daily_cost(
    pool: Any,
    *,
    row: OpenAIDailyCost,
) -> None:
    """Insert / update one row into ``llm_provider_daily_costs``.

    Mirrors the Anthropic upsert path in ``provider_cost_sync.py``:
    same target table, same primary key (provider, cost_date,
    source_kind). One row per day per source kind so re-runs are
    idempotent.
    """
    try:
        import json as _json
        await pool.execute(
            """
            INSERT INTO llm_provider_daily_costs
                (provider, cost_date, provider_cost_usd, currency,
                 source_kind, raw_payload, imported_at)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW())
            ON CONFLICT (provider, cost_date, source_kind)
            DO UPDATE SET
                provider_cost_usd = EXCLUDED.provider_cost_usd,
                currency = EXCLUDED.currency,
                raw_payload = EXCLUDED.raw_payload,
                imported_at = NOW()
            """,
            _PROVIDER_NAME,
            row.cost_date,
            row.provider_cost_usd,
            row.currency,
            _SOURCE_KIND,
            _json.dumps(row.raw_payload, default=str),
        )
    except Exception:
        logger.exception(
            "OpenAI daily-cost upsert failed",
            extra={"cost_date": str(row.cost_date)},
        )


async def sync_openai_daily_costs(
    *,
    pool: Any,
    client: httpx.AsyncClient | None = None,
    days_back: int = DEFAULT_DAYS_BACK,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Fetch and persist OpenAI daily costs for the last ``days_back``
    days.

    Returns a small summary dict so callers (scheduled tasks, ad-hoc
    scripts) can log meaningful telemetry: how many rows were upserted,
    whether the call was skipped, and what the resolved key state was.

    No-ops cleanly if:

    - ``pool`` is None (caller opted out).
    - The OpenAI admin key cannot be resolved.
    """
    resolved_key = _resolve_openai_admin_key(api_key)
    if not resolved_key:
        return {"provider": _PROVIDER_NAME, "skipped": "no_api_key", "rows": 0}
    if pool is None:
        return {"provider": _PROVIDER_NAME, "skipped": "no_pool", "rows": 0}

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient()
    try:
        rows = await fetch_openai_daily_costs(
            client=client,
            api_key=resolved_key,
            days_back=days_back,
        )
        for row in rows:
            await _upsert_openai_daily_cost(pool, row=row)
    finally:
        if owns_client and client is not None:
            await client.aclose()

    return {
        "provider": _PROVIDER_NAME,
        "skipped": None,
        "rows": len(rows),
    }


__all__ = [
    "DEFAULT_DAYS_BACK",
    "OpenAIDailyCost",
    "fetch_openai_daily_costs",
    "sync_openai_daily_costs",
]
