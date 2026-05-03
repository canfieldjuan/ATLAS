"""Runtime budget gate for LLM call sites.

Owned by the extracted LLM-infrastructure package -- not synced from
atlas_brain. Adds a hard-cap kill-switch that callers consult before
making an LLM call. Caps are configurable per (daily total) and per
(attribution dimension, value) so a runaway customer can be capped
without affecting other customers.

Public API:

    BudgetGate(pool, *,
        daily_cap_usd: Decimal | None = None,
        per_attribution_caps: Mapping[str, Mapping[str, Decimal]] | None = None,
    )

    async gate.check_before_call(
        *, estimated_cost_usd, attribution=None,
    ) -> BudgetDecision

    BudgetDecision -- frozen dataclass: allowed, reason,
        consumed_usd, cap_usd.

Reads ``llm_usage`` (the FTL-tracer-populated cost log) to compute
current spend. Compares projected spend (current + estimated) against
caps. ``allowed=False`` returns means the caller MUST NOT make the
call.

Soft policy: if no caps are configured (both args None / empty), every
call is allowed. The caller decides whether to gate or not.

Wiring status: as of PR-A4b this module ships the gate machinery
only. No production LLM call site consults ``check_before_call`` yet.
Wiring it into ``pipelines/llm.py`` and ``services/b2b/anthropic_batch.py``
is a follow-up PR; until then these caps cannot block real traffic.

Concurrency note: this is a read-only check against ``llm_usage``,
which is written *after* the LLM call completes. Concurrent requests
near the limit can all read the same ``consumed`` total, all pass
the gate, and overshoot. The gate is best-effort throttling, not a
strict barrier. A strict barrier would need a write-ahead reservation
table (out of scope for the first cut).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# Attribution keys that ``llm_usage`` materializes into dedicated
# top-level columns (with indexes). When the caller passes one of these
# keys, the query reads the column directly instead of doing a
# ``metadata ->> key`` JSON extraction. Two reasons:
#   1. The column has an index; ``metadata ->>`` does not -- on the
#      hot path we don't want a full-table JSON scan per LLM call.
#   2. Many traces store business attribution under
#      ``metadata['business']``, never at the metadata top level.
#      ``llm_usage`` backfills these into the column at insert time
#      (see migrations 253, 257). Reading ``metadata ->>`` would miss
#      those rows; reading the column catches both shapes.
_PROMOTED_ATTRIBUTION_COLUMNS: frozenset[str] = frozenset(
    {
        "model_provider",
        "span_name",
        "vendor_name",
        "run_id",
        "source_name",
        "event_type",
        "entity_type",
        "entity_id",
    }
)


@dataclass(frozen=True)
class BudgetDecision:
    """Result of a pre-call budget check.

    ``reason`` is None when allowed; otherwise a stable string code
    (``daily_cap_exceeded`` or ``attribution_cap_exceeded:<key>=<value>``)
    so callers / dashboards can categorize denials without parsing
    free-form text.

    ``consumed_usd`` is current spend BEFORE the projected call;
    ``cap_usd`` is the cap that bound the decision (whichever cap was
    closest to being breached). Both are present so the caller can
    surface a useful "you have $X of $Y left today" message.
    """

    allowed: bool
    reason: str | None
    consumed_usd: Decimal
    cap_usd: Decimal


class BudgetGate:
    """Async budget check against ``llm_usage`` daily aggregates.

    Today's spend is recomputed on every ``check_before_call`` from the
    ``llm_usage`` table -- intentionally simple. For low-volume gates
    this is fine; for high-volume gates a future revision can layer
    a 60s in-process cache on top without changing the public API.
    """

    def __init__(
        self,
        pool: Any,
        *,
        daily_cap_usd: Decimal | None = None,
        per_attribution_caps: Mapping[str, Mapping[str, Decimal]] | None = None,
    ) -> None:
        self._pool = pool
        self._daily_cap_usd = daily_cap_usd
        # Defensive copy so the caller can't mutate caps after construction.
        self._per_attribution_caps: dict[str, dict[str, Decimal]] = {}
        if per_attribution_caps:
            for key, dims in per_attribution_caps.items():
                self._per_attribution_caps[key] = {
                    value: Decimal(str(cap)) for value, cap in dims.items()
                }

    @staticmethod
    def _utc_today() -> date:
        return datetime.now(tz=timezone.utc).date()

    async def _daily_total(self, today: date) -> Decimal:
        if self._pool is None:
            return Decimal(0)
        rows = await self._pool.fetch(
            """
            SELECT COALESCE(SUM(cost_usd), 0)::NUMERIC(12, 6) AS total
            FROM llm_usage
            WHERE created_at >= ($1::DATE) AT TIME ZONE 'UTC'
              AND created_at < (($1::DATE) + INTERVAL '1 day') AT TIME ZONE 'UTC'
            """,
            today,
        )
        if not rows:
            return Decimal(0)
        return Decimal(str(rows[0]["total"]))

    async def _attribution_total(
        self,
        today: date,
        attribution_key: str,
        attribution_value: str,
    ) -> Decimal:
        if self._pool is None:
            return Decimal(0)
        # When the key is one of the indexed top-level columns, read the
        # column directly (uses the index, also catches rows where the
        # source value lives under ``metadata['business'][key]``
        # instead of ``metadata[key]`` -- see migrations 253, 257).
        # Otherwise fall back to the generic JSONB filter.
        if attribution_key in _PROMOTED_ATTRIBUTION_COLUMNS:
            # Column name is whitelisted, safe to interpolate.
            sql = (
                "SELECT COALESCE(SUM(cost_usd), 0)::NUMERIC(12, 6) AS total "
                "FROM llm_usage "
                "WHERE created_at >= ($1::DATE) AT TIME ZONE 'UTC' "
                "  AND created_at < (($1::DATE) + INTERVAL '1 day') AT TIME ZONE 'UTC' "
                f"  AND {attribution_key} = $2"
            )
            rows = await self._pool.fetch(sql, today, attribution_value)
        else:
            rows = await self._pool.fetch(
                """
                SELECT COALESCE(SUM(cost_usd), 0)::NUMERIC(12, 6) AS total
                FROM llm_usage
                WHERE created_at >= ($1::DATE) AT TIME ZONE 'UTC'
                  AND created_at < (($1::DATE) + INTERVAL '1 day') AT TIME ZONE 'UTC'
                  AND metadata ->> $2 = $3
                """,
                today,
                attribution_key,
                attribution_value,
            )
        if not rows:
            return Decimal(0)
        return Decimal(str(rows[0]["total"]))

    async def check_before_call(
        self,
        *,
        estimated_cost_usd: Decimal | float | int,
        attribution: Mapping[str, str] | None = None,
    ) -> BudgetDecision:
        """Return ``BudgetDecision(allowed=...)`` for the projected call.

        Order of checks: daily cap first (cheaper query, denies fastest),
        then per-attribution caps (one query per (dim, value) the caller
        actually has and that has a cap configured).

        With zero caps configured: returns
        ``BudgetDecision(allowed=True, reason=None, consumed_usd=0,
        cap_usd=0)``.
        """
        try:
            estimate = Decimal(str(estimated_cost_usd))
        except Exception:
            logger.exception("budget gate estimate coercion failed")
            # Fail open on bad input so a malformed estimate does not
            # silently block all calls. The caller is responsible for
            # passing sane values.
            return BudgetDecision(
                allowed=True,
                reason=None,
                consumed_usd=Decimal(0),
                cap_usd=Decimal(0),
            )
        if estimate < 0:
            # A negative estimate would *subtract* from the consumed
            # total and could trick the gate into allowing an
            # over-budget call. Treat as malformed input -> fail-open
            # the same way as a non-numeric estimate, but log so the
            # caller can fix the bug.
            logger.warning(
                "budget gate received negative estimate %s; treating as 0",
                estimate,
            )
            estimate = Decimal(0)

        today = self._utc_today()

        # Daily cap
        if self._daily_cap_usd is not None:
            daily_cap = Decimal(str(self._daily_cap_usd))
            consumed = await self._daily_total(today)
            if consumed + estimate > daily_cap:
                return BudgetDecision(
                    allowed=False,
                    reason="daily_cap_exceeded",
                    consumed_usd=consumed,
                    cap_usd=daily_cap,
                )

        # Per-attribution caps
        if attribution and self._per_attribution_caps:
            for key, value in attribution.items():
                if key not in self._per_attribution_caps:
                    continue
                cap = self._per_attribution_caps[key].get(str(value))
                if cap is None:
                    continue
                consumed = await self._attribution_total(today, key, str(value))
                if consumed + estimate > cap:
                    return BudgetDecision(
                        allowed=False,
                        reason=f"attribution_cap_exceeded:{key}={value}",
                        consumed_usd=consumed,
                        cap_usd=cap,
                    )

        # No cap was breached.
        return BudgetDecision(
            allowed=True,
            reason=None,
            consumed_usd=Decimal(0),
            cap_usd=Decimal(0),
        )


__all__ = [
    "BudgetDecision",
    "BudgetGate",
]
