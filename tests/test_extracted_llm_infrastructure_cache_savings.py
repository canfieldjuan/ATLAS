"""Tests for extracted_llm_infrastructure.services.cost.cache_savings.

Uses an in-memory fake pool that mimics asyncpg's ``execute`` and
``fetch`` semantics for the queries this module issues. Verifies the
public API contract without requiring a live database.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import pytest

from extracted_llm_infrastructure.services.cost.cache_savings import (
    CacheSavingsRollup,
    daily_cache_savings,
    record_cache_hit,
)


class _FakePool:
    """Minimal asyncpg-compatible pool stub for unit tests.

    Honors the ``hit_at >= start AND hit_at < end`` filter on every
    fetch query so the date-range semantics of ``daily_cache_savings``
    can actually be exercised. Set ``next_hit_at`` before
    ``record_cache_hit`` to control the row's timestamp; otherwise
    rows default to a fixed in-window timestamp so older tests that
    don't care about time keep passing.
    """

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.next_hit_at: datetime | None = None

    async def execute(self, sql: str, *args: Any) -> None:
        self.execute_calls.append((sql, args))
        normalized = " ".join(sql.split())
        if normalized.upper().startswith("INSERT INTO LLM_CACHE_SAVINGS"):
            self.rows.append(
                {
                    "cache_key": args[0],
                    "namespace": args[1],
                    "provider": args[2],
                    "model": args[3],
                    "saved_input_tokens": args[4],
                    "saved_output_tokens": args[5],
                    "saved_cost_usd": args[6],
                    "attribution": json.loads(args[7]),
                    "metadata": json.loads(args[8]),
                    "hit_at": self.next_hit_at
                    or datetime(2026, 5, 3, 12, tzinfo=timezone.utc),
                }
            )

    @staticmethod
    def _within_range(row: dict[str, Any], start: date, end: date) -> bool:
        # Mirror the SQL semantics (date cast to TIMESTAMPTZ at UTC midnight,
        # half-open interval): hit_at >= start_utc AND hit_at < end_utc.
        boundary_start = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
        boundary_end = datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc)
        return boundary_start <= row["hit_at"] < boundary_end

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((sql, args))
        normalized = " ".join(sql.split())
        start, end = args[0], args[1]
        scoped = [row for row in self.rows if self._within_range(row, start, end)]
        if "GROUP BY namespace" in normalized:
            buckets: dict[str, Decimal] = {}
            for row in scoped:
                buckets[row["namespace"]] = (
                    buckets.get(row["namespace"], Decimal(0))
                    + Decimal(str(row["saved_cost_usd"]))
                )
            return [
                {"namespace": ns, "cost": cost}
                for ns, cost in sorted(
                    buckets.items(), key=lambda kv: kv[1], reverse=True
                )
            ]
        if "GROUP BY dim_value" in normalized:
            attribution_key = args[2]
            buckets = {}
            for row in scoped:
                value = row["attribution"].get(attribution_key)
                if value is None:
                    continue
                buckets[value] = (
                    buckets.get(value, Decimal(0))
                    + Decimal(str(row["saved_cost_usd"]))
                )
            return [
                {"dim_value": v, "cost": cost}
                for v, cost in sorted(
                    buckets.items(), key=lambda kv: kv[1], reverse=True
                )
            ]
        # Summary query
        total_cost = sum(
            (Decimal(str(row["saved_cost_usd"])) for row in scoped),
            Decimal(0),
        )
        total_input = sum(int(row["saved_input_tokens"]) for row in scoped)
        total_output = sum(int(row["saved_output_tokens"]) for row in scoped)
        return [
            {
                "total_cost": total_cost,
                "total_input": total_input,
                "total_output": total_output,
                "hit_count": len(scoped),
            }
        ]


@pytest.mark.asyncio
async def test_record_cache_hit_writes_row():
    pool = _FakePool()
    await record_cache_hit(
        pool,
        cache_key="abc123",
        namespace="b2b_enrichment.tier1",
        provider="anthropic",
        model="claude-haiku-4-5",
        would_have_been_input_tokens=1000,
        would_have_been_output_tokens=500,
        would_have_been_cost_usd=Decimal("0.0123"),
        attribution={"customer_id": "acme"},
        metadata={"reason": "enrichment retry"},
    )
    assert len(pool.rows) == 1
    row = pool.rows[0]
    assert row["cache_key"] == "abc123"
    assert row["namespace"] == "b2b_enrichment.tier1"
    assert row["provider"] == "anthropic"
    assert row["model"] == "claude-haiku-4-5"
    assert row["saved_input_tokens"] == 1000
    assert row["saved_output_tokens"] == 500
    assert row["saved_cost_usd"] == Decimal("0.0123")
    assert row["attribution"] == {"customer_id": "acme"}
    assert row["metadata"] == {"reason": "enrichment retry"}


@pytest.mark.asyncio
async def test_record_cache_hit_no_op_on_none_pool():
    # Defensive: callers should be able to opt out via feature flag
    # without sprinkling None checks at every call site.
    await record_cache_hit(
        None,
        cache_key="abc",
        namespace="ns",
        provider="anthropic",
        model="claude",
        would_have_been_input_tokens=0,
        would_have_been_output_tokens=0,
        would_have_been_cost_usd=Decimal(0),
    )


@pytest.mark.asyncio
async def test_record_cache_hit_coerces_cost_to_decimal():
    # Float and int inputs should round-trip cleanly via Decimal.
    pool = _FakePool()
    await record_cache_hit(
        pool,
        cache_key="k",
        namespace="ns",
        provider="anthropic",
        model="claude",
        would_have_been_input_tokens=1,
        would_have_been_output_tokens=2,
        would_have_been_cost_usd=0.05,
    )
    assert pool.rows[0]["saved_cost_usd"] == Decimal("0.05")


@pytest.mark.asyncio
async def test_daily_cache_savings_empty_range():
    pool = _FakePool()
    rollup = await daily_cache_savings(
        pool,
        date_range=(date(2026, 5, 1), date(2026, 5, 4)),
    )
    assert rollup["hit_count"] == 0
    assert rollup["total_saved_usd"] == Decimal(0)
    assert rollup["total_saved_input_tokens"] == 0
    assert rollup["total_saved_output_tokens"] == 0
    assert dict(rollup["by_namespace"]) == {}
    assert dict(rollup["by_attribution_dim"]) == {}


@pytest.mark.asyncio
async def test_daily_cache_savings_aggregates_multiple_hits():
    pool = _FakePool()
    for i in range(3):
        await record_cache_hit(
            pool,
            cache_key=f"key-{i}",
            namespace="ns_a",
            provider="anthropic",
            model="claude",
            would_have_been_input_tokens=1000,
            would_have_been_output_tokens=500,
            would_have_been_cost_usd=Decimal("0.01"),
            attribution={"customer_id": "acme"},
        )
    await record_cache_hit(
        pool,
        cache_key="key-other",
        namespace="ns_b",
        provider="anthropic",
        model="claude",
        would_have_been_input_tokens=2000,
        would_have_been_output_tokens=1000,
        would_have_been_cost_usd=Decimal("0.02"),
        attribution={"customer_id": "globex"},
    )
    rollup = await daily_cache_savings(
        pool,
        date_range=(date(2026, 5, 1), date(2026, 5, 10)),
    )
    assert rollup["hit_count"] == 4
    assert rollup["total_saved_usd"] == Decimal("0.05")
    assert rollup["total_saved_input_tokens"] == 5000
    assert rollup["total_saved_output_tokens"] == 2500
    assert dict(rollup["by_namespace"]) == {
        "ns_a": Decimal("0.03"),
        "ns_b": Decimal("0.02"),
    }


@pytest.mark.asyncio
async def test_daily_cache_savings_attribution_dim_rollup():
    pool = _FakePool()
    await record_cache_hit(
        pool,
        cache_key="k1",
        namespace="ns",
        provider="anthropic",
        model="claude",
        would_have_been_input_tokens=1000,
        would_have_been_output_tokens=500,
        would_have_been_cost_usd=Decimal("0.05"),
        attribution={"customer_id": "acme"},
    )
    await record_cache_hit(
        pool,
        cache_key="k2",
        namespace="ns",
        provider="anthropic",
        model="claude",
        would_have_been_input_tokens=1000,
        would_have_been_output_tokens=500,
        would_have_been_cost_usd=Decimal("0.03"),
        attribution={"customer_id": "globex"},
    )
    rollup = await daily_cache_savings(
        pool,
        date_range=(date(2026, 5, 1), date(2026, 5, 10)),
        attribution_key="customer_id",
    )
    by_customer = dict(rollup["by_attribution_dim"]["customer_id"])
    assert by_customer == {"acme": Decimal("0.05"), "globex": Decimal("0.03")}


@pytest.mark.asyncio
async def test_daily_cache_savings_skips_attribution_when_no_key():
    # When attribution_key is None, by_attribution_dim should be empty
    # even if rows have attribution data.
    pool = _FakePool()
    await record_cache_hit(
        pool,
        cache_key="k",
        namespace="ns",
        provider="anthropic",
        model="claude",
        would_have_been_input_tokens=10,
        would_have_been_output_tokens=5,
        would_have_been_cost_usd=Decimal("0.01"),
        attribution={"customer_id": "acme"},
    )
    rollup = await daily_cache_savings(
        pool,
        date_range=(date(2026, 5, 1), date(2026, 5, 10)),
    )
    assert dict(rollup["by_attribution_dim"]) == {}


@pytest.mark.asyncio
async def test_daily_cache_savings_respects_date_range_boundaries():
    # Two hits inside the window, one before, one after. Only the two
    # in-window hits should appear in the rollup.
    pool = _FakePool()
    pool.next_hit_at = datetime(2026, 4, 30, 23, 0, tzinfo=timezone.utc)  # before
    await record_cache_hit(
        pool, cache_key="before", namespace="ns", provider="anthropic",
        model="claude", would_have_been_input_tokens=100,
        would_have_been_output_tokens=50, would_have_been_cost_usd=Decimal("0.99"),
    )
    pool.next_hit_at = datetime(2026, 5, 1, 0, 1, tzinfo=timezone.utc)  # in
    await record_cache_hit(
        pool, cache_key="in1", namespace="ns", provider="anthropic",
        model="claude", would_have_been_input_tokens=100,
        would_have_been_output_tokens=50, would_have_been_cost_usd=Decimal("0.10"),
    )
    pool.next_hit_at = datetime(2026, 5, 3, 12, tzinfo=timezone.utc)  # in
    await record_cache_hit(
        pool, cache_key="in2", namespace="ns", provider="anthropic",
        model="claude", would_have_been_input_tokens=100,
        would_have_been_output_tokens=50, would_have_been_cost_usd=Decimal("0.20"),
    )
    pool.next_hit_at = datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)  # at end (excluded)
    await record_cache_hit(
        pool, cache_key="at_end", namespace="ns", provider="anthropic",
        model="claude", would_have_been_input_tokens=100,
        would_have_been_output_tokens=50, would_have_been_cost_usd=Decimal("0.99"),
    )
    rollup = await daily_cache_savings(
        pool, date_range=(date(2026, 5, 1), date(2026, 5, 4)),
    )
    assert rollup["hit_count"] == 2
    assert rollup["total_saved_usd"] == Decimal("0.30")


@pytest.mark.asyncio
async def test_record_cache_hit_attribution_accepts_non_string_values():
    # Free-form JSONB: callers should be able to attach numeric IDs,
    # booleans, and nested dicts without coercing to strings.
    pool = _FakePool()
    await record_cache_hit(
        pool,
        cache_key="k",
        namespace="ns",
        provider="anthropic",
        model="claude",
        would_have_been_input_tokens=10,
        would_have_been_output_tokens=5,
        would_have_been_cost_usd=Decimal("0.01"),
        attribution={
            "tenant_id": 12345,
            "is_premium": True,
            "tags": ["alpha", "beta"],
        },
    )
    assert pool.rows[0]["attribution"] == {
        "tenant_id": 12345,
        "is_premium": True,
        "tags": ["alpha", "beta"],
    }


@pytest.mark.asyncio
async def test_record_cache_hit_swallows_invalid_decimal():
    # Non-Decimal-coercible cost must not raise out of the helper.
    pool = _FakePool()
    await record_cache_hit(
        pool,
        cache_key="k",
        namespace="ns",
        provider="anthropic",
        model="claude",
        would_have_been_input_tokens=10,
        would_have_been_output_tokens=5,
        would_have_been_cost_usd="not-a-number",  # type: ignore[arg-type]
    )
    assert pool.rows == []  # no insert reached the pool


@pytest.mark.asyncio
async def test_record_cache_hit_swallows_unserializable_attribution():
    # Non-JSON-serializable attribution must not raise out of the helper.
    pool = _FakePool()
    class _NotSerializable:
        pass
    await record_cache_hit(
        pool,
        cache_key="k",
        namespace="ns",
        provider="anthropic",
        model="claude",
        would_have_been_input_tokens=10,
        would_have_been_output_tokens=5,
        would_have_been_cost_usd=Decimal("0.01"),
        attribution={"obj": _NotSerializable()},  # type: ignore[dict-item]
    )
    assert pool.rows == []  # no insert reached the pool


def test_cache_savings_rollup_is_typed_dict():
    rollup: CacheSavingsRollup = {
        "total_saved_usd": Decimal(0),
        "total_saved_input_tokens": 0,
        "total_saved_output_tokens": 0,
        "hit_count": 0,
        "by_namespace": {},
        "by_attribution_dim": {},
    }
    # Round-trip dict access works (TypedDict is a dict at runtime)
    assert rollup["hit_count"] == 0
