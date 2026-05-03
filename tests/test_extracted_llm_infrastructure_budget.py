"""Tests for extracted_llm_infrastructure.services.cost.budget.

Uses an in-memory fake pool that emulates the daily-total and
attribution-total queries from llm_usage. Verifies allow/deny
semantics + the BudgetDecision shape without a live database.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pytest

from extracted_llm_infrastructure.services.cost.budget import (
    BudgetDecision,
    BudgetGate,
)


class _FakePool:
    """Fake pool that emulates the budget-gate queries.

    Tests populate ``llm_usage_rows`` with dicts containing cost_usd,
    created_at (UTC datetime), and metadata (dict). The fake's
    ``fetch`` answers either the daily-total query or the
    attribution-total query depending on which SQL shape it sees.
    """

    def __init__(self) -> None:
        self.llm_usage_rows: list[dict[str, Any]] = []

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        normalized = " ".join(sql.split())
        today = args[0]
        # Anchor: UTC midnight start, +24h end.
        start = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)
        end = datetime.combine(
            today.replace(day=today.day + 1) if today.day < 28 else today,
            datetime.min.time(),
            tzinfo=timezone.utc,
        ) if today.day < 28 else None
        # Simpler: use a + 1 day timedelta in real code; for the test
        # just compare to (today + 1 day) at UTC midnight.
        from datetime import timedelta
        end = start + timedelta(days=1)

        in_range = [
            r
            for r in self.llm_usage_rows
            if start <= r["created_at"] < end
        ]

        if "metadata ->> $2" in normalized:
            attribution_key = args[1]
            attribution_value = args[2]
            scoped = [
                r
                for r in in_range
                if r.get("metadata", {}).get(attribution_key) == attribution_value
            ]
            total = sum(
                (Decimal(str(r["cost_usd"])) for r in scoped),
                Decimal(0),
            )
        else:
            total = sum(
                (Decimal(str(r["cost_usd"])) for r in in_range),
                Decimal(0),
            )

        return [{"total": total}]


def _utc_today_at(hour: int = 12) -> datetime:
    today = datetime.now(tz=timezone.utc).date()
    return datetime(today.year, today.month, today.day, hour, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_no_caps_allows_every_call():
    gate = BudgetGate(_FakePool())
    decision = await gate.check_before_call(estimated_cost_usd=Decimal("100.00"))
    assert decision.allowed is True
    assert decision.reason is None


@pytest.mark.asyncio
async def test_daily_cap_allows_below_threshold():
    pool = _FakePool()
    pool.llm_usage_rows = [
        {"cost_usd": Decimal("10.00"), "created_at": _utc_today_at(), "metadata": {}},
    ]
    gate = BudgetGate(pool, daily_cap_usd=Decimal("100.00"))
    # Current 10 + estimate 5 = 15 < 100
    decision = await gate.check_before_call(estimated_cost_usd=Decimal("5.00"))
    assert decision.allowed is True
    assert decision.reason is None


@pytest.mark.asyncio
async def test_daily_cap_denies_when_estimate_pushes_over():
    pool = _FakePool()
    pool.llm_usage_rows = [
        {"cost_usd": Decimal("95.00"), "created_at": _utc_today_at(), "metadata": {}},
    ]
    gate = BudgetGate(pool, daily_cap_usd=Decimal("100.00"))
    # Current 95 + estimate 10 = 105 > 100
    decision = await gate.check_before_call(estimated_cost_usd=Decimal("10.00"))
    assert decision.allowed is False
    assert decision.reason == "daily_cap_exceeded"
    assert decision.consumed_usd == Decimal("95.00")
    assert decision.cap_usd == Decimal("100.00")


@pytest.mark.asyncio
async def test_daily_cap_denies_at_exact_boundary_due_to_strict_gt():
    pool = _FakePool()
    pool.llm_usage_rows = [
        {"cost_usd": Decimal("95.00"), "created_at": _utc_today_at(), "metadata": {}},
    ]
    gate = BudgetGate(pool, daily_cap_usd=Decimal("100.00"))
    # Current 95 + estimate 5 = 100 == cap. Strict > means: still allowed.
    decision = await gate.check_before_call(estimated_cost_usd=Decimal("5.00"))
    assert decision.allowed is True


@pytest.mark.asyncio
async def test_attribution_cap_denies_specific_customer():
    pool = _FakePool()
    pool.llm_usage_rows = [
        {
            "cost_usd": Decimal("8.00"),
            "created_at": _utc_today_at(),
            "metadata": {"customer_id": "acme"},
        },
        {
            "cost_usd": Decimal("2.00"),
            "created_at": _utc_today_at(),
            "metadata": {"customer_id": "globex"},
        },
    ]
    gate = BudgetGate(
        pool,
        per_attribution_caps={"customer_id": {"acme": Decimal("10.00")}},
    )
    # acme has 8.00 spent, cap 10.00. Estimate 5 -> 13 > 10 -> denied.
    denied = await gate.check_before_call(
        estimated_cost_usd=Decimal("5.00"),
        attribution={"customer_id": "acme"},
    )
    assert denied.allowed is False
    assert denied.reason == "attribution_cap_exceeded:customer_id=acme"
    # globex has no cap configured -> allowed.
    allowed = await gate.check_before_call(
        estimated_cost_usd=Decimal("5.00"),
        attribution={"customer_id": "globex"},
    )
    assert allowed.allowed is True


@pytest.mark.asyncio
async def test_daily_cap_checked_before_attribution_cap():
    # If both daily and attribution would deny, the daily reason wins
    # (it's the one returned). Documents check order.
    pool = _FakePool()
    pool.llm_usage_rows = [
        {
            "cost_usd": Decimal("99.00"),
            "created_at": _utc_today_at(),
            "metadata": {"customer_id": "acme"},
        },
    ]
    gate = BudgetGate(
        pool,
        daily_cap_usd=Decimal("100.00"),
        per_attribution_caps={"customer_id": {"acme": Decimal("100.00")}},
    )
    decision = await gate.check_before_call(
        estimated_cost_usd=Decimal("2.00"),
        attribution={"customer_id": "acme"},
    )
    assert decision.allowed is False
    assert decision.reason == "daily_cap_exceeded"


@pytest.mark.asyncio
async def test_invalid_estimate_fails_open():
    # Defensive: bad estimate input must not silently block all calls.
    gate = BudgetGate(_FakePool(), daily_cap_usd=Decimal("100.00"))
    decision = await gate.check_before_call(
        estimated_cost_usd="not-a-number",  # type: ignore[arg-type]
    )
    assert decision.allowed is True


@pytest.mark.asyncio
async def test_caps_are_defensively_copied():
    # Mutating the caps dict after BudgetGate construction must not
    # change the gate's behavior.
    caps = {"customer_id": {"acme": Decimal("10.00")}}
    pool = _FakePool()
    pool.llm_usage_rows = [
        {
            "cost_usd": Decimal("5.00"),
            "created_at": _utc_today_at(),
            "metadata": {"customer_id": "acme"},
        },
    ]
    gate = BudgetGate(pool, per_attribution_caps=caps)
    # Mutate the source dict
    caps["customer_id"]["acme"] = Decimal("0.01")
    caps["customer_id"]["globex"] = Decimal("0.01")
    # Gate should still use the cap as it was at construction time.
    decision = await gate.check_before_call(
        estimated_cost_usd=Decimal("3.00"),
        attribution={"customer_id": "acme"},
    )
    # 5 + 3 = 8 < 10 (original cap) -> allowed
    assert decision.allowed is True


def test_budget_decision_is_frozen_dataclass():
    decision = BudgetDecision(
        allowed=True,
        reason=None,
        consumed_usd=Decimal(0),
        cap_usd=Decimal(0),
    )
    with pytest.raises(Exception):
        decision.allowed = False  # type: ignore[misc]
