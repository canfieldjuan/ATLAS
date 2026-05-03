"""Drift report: local-vs-invoiced cost reconciliation.

Owned by the extracted LLM-infrastructure package -- not synced from
atlas_brain. The differentiated cost-closure wedge: every observability
tool logs token usage; few reconcile that log against the provider's
own billing API. This module computes the per-day delta between
``llm_usage`` (locally-tracked spend from the FTL tracer) and
``llm_provider_daily_costs`` (invoiced spend from the provider's
admin/billing API, populated by ``provider_cost_sync``).

Public API:

    compute_drift(pool, *, provider, date_range) -> list[DriftRow]

    DriftRow -- per-day delta with explanatory chips.

A non-empty ``explained_by`` list is a hint, not a verdict. The chips
exist so an operator looking at a $87 / $400 daily delta can see at a
glance whether it's likely a stale-pricing issue, a missing-row issue,
or genuinely outside the noise floor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)

# Threshold for the "high_drift" chip. Drift below this absolute
# percentage is considered noise (rounding, retry counting, fp drift)
# and not worth flagging.
_HIGH_DRIFT_PCT_THRESHOLD = 0.20  # 20%


@dataclass(frozen=True)
class DriftRow:
    """One per (provider, cost_date) within the requested range.

    Drift is reported as ``invoiced_usd - local_usd``. Positive delta
    means the invoice is larger than locally-tracked spend (the common
    case: stale local pricing tables, retries not deduped, hidden fees,
    multi-modal billing). Negative delta means local thinks we spent
    more than the invoice says (less common: cancelled requests still
    counted locally, batch refunds applied later).
    """

    provider: str
    cost_date: date
    local_usd: Decimal
    invoiced_usd: Decimal
    delta_usd: Decimal
    delta_pct: float
    explained_by: list[str] = field(default_factory=list)


def _classify(local: Decimal, invoiced: Decimal, delta: Decimal) -> list[str]:
    """Heuristic chips that hint at the cause of drift.

    Multiple chips can apply. Order is significance-first: structural
    issues (missing rows) before magnitude (high_drift) before
    direction (stale_pricing).
    """
    chips: list[str] = []
    zero = Decimal(0)

    if local == zero and invoiced > zero:
        chips.append("missing_local_rows")
    elif invoiced == zero and local > zero:
        chips.append("missing_invoice")

    if invoiced > zero:
        pct = float(delta) / float(invoiced) if invoiced else 0.0
        if abs(pct) >= _HIGH_DRIFT_PCT_THRESHOLD:
            chips.append("high_drift")
        if delta > zero and "missing_local_rows" not in chips:
            # Invoice higher than local AND not the trivial empty case.
            # Most common cause: local pricing table is older than
            # the provider's invoice (batch fees, multi-modal, etc.).
            chips.append("stale_pricing")

    return chips


async def compute_drift(
    pool: Any,
    *,
    provider: str,
    date_range: tuple[date, date],
) -> list[DriftRow]:
    """Compute per-day drift for one provider over ``[start, end)``.

    Joins (FULL OUTER) the local-usage daily sum with the invoiced
    daily-cost row so days with one side missing still show up.

    ``date_range`` is half-open: inclusive start, exclusive end.
    """
    start, end = date_range

    rows = await pool.fetch(
        """
        WITH local_daily AS (
            SELECT
                DATE(created_at AT TIME ZONE 'UTC') AS cost_date,
                COALESCE(SUM(cost_usd), 0)::NUMERIC(12, 6) AS local_usd
            FROM llm_usage
            WHERE created_at >= ($2::DATE)::TIMESTAMPTZ
              AND created_at < ($3::DATE)::TIMESTAMPTZ
              AND model_provider = $1
            GROUP BY cost_date
        ),
        invoiced_daily AS (
            SELECT
                cost_date,
                SUM(provider_cost_usd)::NUMERIC(12, 6) AS invoiced_usd
            FROM llm_provider_daily_costs
            WHERE provider = $1
              AND cost_date >= $2::DATE
              AND cost_date < $3::DATE
            GROUP BY cost_date
        )
        SELECT
            COALESCE(local_daily.cost_date, invoiced_daily.cost_date) AS cost_date,
            COALESCE(local_daily.local_usd, 0)::NUMERIC(12, 6) AS local_usd,
            COALESCE(invoiced_daily.invoiced_usd, 0)::NUMERIC(12, 6) AS invoiced_usd
        FROM local_daily
        FULL OUTER JOIN invoiced_daily USING (cost_date)
        ORDER BY cost_date ASC
        """,
        provider,
        start,
        end,
    )

    drift_rows: list[DriftRow] = []
    for row in rows:
        local = Decimal(str(row["local_usd"]))
        invoiced = Decimal(str(row["invoiced_usd"]))
        delta = invoiced - local
        pct = float(delta) / float(invoiced) if invoiced else 0.0
        drift_rows.append(
            DriftRow(
                provider=provider,
                cost_date=row["cost_date"],
                local_usd=local,
                invoiced_usd=invoiced,
                delta_usd=delta,
                delta_pct=pct,
                explained_by=_classify(local, invoiced, delta),
            )
        )
    return drift_rows


__all__ = [
    "DriftRow",
    "compute_drift",
]
