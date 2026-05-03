"""Tests for extracted_llm_infrastructure.services.cost.openai_billing.

Uses an in-memory fake httpx client + fake pool to verify the
fetch + parse + upsert path without hitting the real OpenAI Costs
API or a live database.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import pytest

from extracted_llm_infrastructure.services.cost.openai_billing import (
    DEFAULT_DAYS_BACK,
    OpenAIDailyCost,
    _parse_cost_rows,
    _resolve_openai_admin_key,
    fetch_openai_daily_costs,
    sync_openai_daily_costs,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status
        self.content = json.dumps(payload).encode() if payload is not None else b""

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    """Minimal httpx-shaped async client. Tests stuff a payload to return."""

    def __init__(self, payload: dict[str, Any] | None = None, status: int = 200) -> None:
        self.payload = payload or {}
        self.status = status
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    async def get(
        self, url: str, *, headers: dict[str, Any], params: dict[str, Any], timeout: float
    ) -> _FakeResponse:
        self.calls.append((url, dict(headers), dict(params)))
        return _FakeResponse(self.payload, status=self.status)

    async def aclose(self) -> None:
        pass


class _FakePool:
    """Fake pool that records upserts."""

    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []

    async def execute(self, sql: str, *args: Any) -> None:
        normalized = " ".join(sql.split())
        if "INSERT INTO LLM_PROVIDER_DAILY_COSTS" in normalized.upper():
            self.upserts.append(
                {
                    "provider": args[0],
                    "cost_date": args[1],
                    "provider_cost_usd": args[2],
                    "currency": args[3],
                    "source_kind": args[4],
                    "raw_payload": json.loads(args[5]),
                }
            )


def _bucket(start_dt: datetime, value: float, currency: str = "USD") -> dict[str, Any]:
    return {
        "start_time": int(start_dt.timestamp()),
        "end_time": int(start_dt.timestamp()) + 86400,
        "results": [{"amount": {"value": value, "currency": currency}}],
    }


# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------


def test_parse_cost_rows_sums_results_in_a_bucket():
    payload = {
        "data": [
            {
                "start_time": int(datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp()),
                "end_time": int(datetime(2026, 5, 2, tzinfo=timezone.utc).timestamp()),
                "results": [
                    {"amount": {"value": 1.50, "currency": "USD"}},
                    {"amount": {"value": 2.50, "currency": "USD"}},
                ],
            }
        ]
    }
    rows = _parse_cost_rows(payload)
    assert len(rows) == 1
    assert rows[0].cost_date == date(2026, 5, 1)
    assert rows[0].provider_cost_usd == Decimal("4.00")
    assert rows[0].currency == "USD"


def test_parse_cost_rows_empty_when_no_data():
    assert _parse_cost_rows({}) == []
    assert _parse_cost_rows({"data": []}) == []


def test_parse_cost_rows_skips_malformed_buckets():
    payload = {
        "data": [
            "not-a-dict",
            {"start_time": None, "results": []},
            {
                "start_time": int(datetime(2026, 5, 2, tzinfo=timezone.utc).timestamp()),
                "results": [{"amount": {"value": 1.00, "currency": "USD"}}],
            },
        ]
    }
    rows = _parse_cost_rows(payload)
    assert len(rows) == 1
    assert rows[0].cost_date == date(2026, 5, 2)


def test_parse_cost_rows_handles_missing_amount_value():
    # A bucket with no usable amount should round-trip to Decimal(0).
    payload = {
        "data": [
            {
                "start_time": int(datetime(2026, 5, 3, tzinfo=timezone.utc).timestamp()),
                "results": [{"amount": {}}],
            }
        ]
    }
    rows = _parse_cost_rows(payload)
    assert rows[0].provider_cost_usd == Decimal(0)


# ---------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------


def test_resolve_openai_admin_key_explicit_wins():
    assert _resolve_openai_admin_key("sk-explicit") == "sk-explicit"


def test_resolve_openai_admin_key_env_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_ADMIN_API_KEY", "sk-env")
    assert _resolve_openai_admin_key(None) == "sk-env"


def test_resolve_openai_admin_key_returns_empty_when_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_ADMIN_API_KEY", raising=False)
    assert _resolve_openai_admin_key(None) == ""


# ---------------------------------------------------------------------
# fetch_openai_daily_costs
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_returns_empty_when_api_key_blank():
    client = _FakeClient()
    rows = await fetch_openai_daily_costs(client=client, api_key="", days_back=7)
    assert rows == []
    assert client.calls == []  # no HTTP attempted


@pytest.mark.asyncio
async def test_fetch_returns_empty_when_days_back_zero():
    client = _FakeClient()
    rows = await fetch_openai_daily_costs(client=client, api_key="sk", days_back=0)
    assert rows == []
    assert client.calls == []


@pytest.mark.asyncio
async def test_fetch_calls_costs_endpoint_with_bearer_and_bucket_params():
    payload = {"data": [_bucket(datetime(2026, 5, 1, tzinfo=timezone.utc), 1.23)]}
    client = _FakeClient(payload=payload)
    rows = await fetch_openai_daily_costs(client=client, api_key="sk-test", days_back=3)
    assert len(rows) == 1
    assert rows[0].provider_cost_usd == Decimal("1.23")
    # Verify the request shape
    url, headers, params = client.calls[0]
    assert url.endswith("/v1/organization/costs")
    assert headers["Authorization"] == "Bearer sk-test"
    assert params["bucket_width"] == "1d"
    assert isinstance(params["start_time"], int)
    assert isinstance(params["end_time"], int)


@pytest.mark.asyncio
async def test_fetch_swallows_http_errors():
    client = _FakeClient(payload={}, status=500)
    rows = await fetch_openai_daily_costs(client=client, api_key="sk", days_back=3)
    assert rows == []


# ---------------------------------------------------------------------
# sync_openai_daily_costs
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_skips_when_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_ADMIN_API_KEY", raising=False)
    pool = _FakePool()
    summary = await sync_openai_daily_costs(pool=pool, client=_FakeClient())
    assert summary["skipped"] == "no_api_key"
    assert summary["rows"] == 0
    assert pool.upserts == []


@pytest.mark.asyncio
async def test_sync_skips_when_pool_is_none():
    summary = await sync_openai_daily_costs(
        pool=None, client=_FakeClient(), api_key="sk-test"
    )
    assert summary["skipped"] == "no_pool"


@pytest.mark.asyncio
async def test_sync_upserts_one_row_per_bucket():
    payload = {
        "data": [
            _bucket(datetime(2026, 5, 1, tzinfo=timezone.utc), 1.00),
            _bucket(datetime(2026, 5, 2, tzinfo=timezone.utc), 2.50),
        ]
    }
    pool = _FakePool()
    client = _FakeClient(payload=payload)
    summary = await sync_openai_daily_costs(
        pool=pool,
        client=client,
        days_back=3,
        api_key="sk-test",
    )
    assert summary["skipped"] is None
    assert summary["rows"] == 2
    assert len(pool.upserts) == 2
    assert pool.upserts[0]["provider"] == "openai"
    assert pool.upserts[0]["source_kind"] == "openai_organization_costs"
    assert pool.upserts[0]["cost_date"] == date(2026, 5, 1)
    assert pool.upserts[0]["provider_cost_usd"] == Decimal("1.00")
    assert pool.upserts[1]["cost_date"] == date(2026, 5, 2)
    assert pool.upserts[1]["provider_cost_usd"] == Decimal("2.50")


@pytest.mark.asyncio
async def test_sync_swallows_upsert_failure_per_row():
    # If one upsert raises, the sync should still report progress for
    # the rest. Implementation logs and swallows; we just verify no
    # raise out of sync_openai_daily_costs.
    payload = {
        "data": [_bucket(datetime(2026, 5, 1, tzinfo=timezone.utc), 1.00)]
    }
    class _BrokenPool:
        async def execute(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("db down")
    summary = await sync_openai_daily_costs(
        pool=_BrokenPool(),
        client=_FakeClient(payload=payload),
        api_key="sk-test",
    )
    # rows reflects fetched count; failed upserts are logged not counted out
    assert summary["rows"] == 1


def test_openai_daily_cost_is_frozen_dataclass():
    row = OpenAIDailyCost(
        cost_date=date(2026, 5, 1),
        provider_cost_usd=Decimal("1.00"),
    )
    assert row.currency == "USD"
    assert row.raw_payload == {}
    with pytest.raises(Exception):
        row.cost_date = date(2026, 5, 2)  # type: ignore[misc]


def test_default_days_back_constant_exists():
    assert isinstance(DEFAULT_DAYS_BACK, int)
    assert DEFAULT_DAYS_BACK > 0
