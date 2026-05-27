from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from extracted_content_pipeline.api.control_surfaces import _required_scope_account_id
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.content_ops_usage_summary import (
    summarize_content_ops_llm_usage,
)


class _UsageSummaryPool:
    def __init__(self) -> None:
        self.fetchrow_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetchrow(self, query: str, *args: object) -> dict[str, object]:
        self.fetchrow_calls.append((query, args))
        return {
            "total_cost_usd": Decimal("0"),
            "input_tokens": 0,
            "billable_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "cache_savings_usd": Decimal("0"),
            "total_calls": 0,
            "failed_calls": 0,
            "cache_hit_calls": 0,
            "avg_duration_ms": 0,
            "latest_call_at": None,
        }

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        self.fetch_calls.append((query, args))
        return []


@pytest.mark.asyncio
async def test_content_ops_usage_summary_filters_by_account_id_in_sql() -> None:
    pool = _UsageSummaryPool()

    payload = await summarize_content_ops_llm_usage(
        pool,
        days=14,
        account_id="acct-123",
        asset_type="landing_page",
        run_id="run-1",
        request_id="req-1",
    )

    query, args = pool.fetchrow_calls[0]
    assert "metadata ->> 'account_id' = $2" in query
    assert "metadata ->> 'asset_type' = $3" in query
    assert "(run_id = $4 OR metadata ->> 'run_id' = $4)" in query
    assert "metadata ->> 'request_id' = $5" in query
    assert args == (14, "acct-123", "landing_page", "run-1", "req-1")
    assert pool.fetch_calls[0][1] == args
    assert pool.fetch_calls[1][1] == args
    assert pool.fetch_calls[2][1] == args
    assert "metadata ->> 'cache_mode'" in pool.fetch_calls[2][0]
    assert "GROUP BY cache_mode, cache_reason, cache_result, cache_store_result" in (
        pool.fetch_calls[2][0]
    )
    assert payload["filters"] == {
        "account_id": "acct-123",
        "asset_type": "landing_page",
        "run_id": "run-1",
        "request_id": "req-1",
    }


def test_content_ops_tenant_usage_requires_scope_account_id() -> None:
    assert _required_scope_account_id(TenantScope(account_id="acct-123")) == "acct-123"

    with pytest.raises(Exception) as exc:
        _required_scope_account_id(TenantScope())

    assert getattr(exc.value, "status_code", None) == 400
    assert getattr(exc.value, "detail", None) == "account_id is required"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_content_ops_usage_summary_contract_against_postgres() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("EXTRACTED_DATABASE_URL or DATABASE_URL is required")

    root = Path(__file__).resolve().parents[1]
    request_id = f"usage-summary-{uuid4().hex}"
    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=2)

    try:
        await pool.execute((root / "atlas_brain/storage/migrations/127_llm_usage.sql").read_text())
        await pool.execute((root / "atlas_brain/storage/migrations/252_llm_usage_cache_breakdown.sql").read_text())
        await pool.execute((root / "atlas_brain/storage/migrations/253_llm_usage_vendor_and_run_id.sql").read_text())
        await pool.executemany(
            """
            INSERT INTO llm_usage (
                span_name, operation_type, model_name, model_provider,
                input_tokens, output_tokens, total_tokens, billable_input_tokens,
                cached_tokens, cache_write_tokens, cost_usd, duration_ms, status,
                metadata, run_id
            )
            VALUES (
                $1, 'llm_call', $2, $3,
                $4, $5, $6, $7,
                $8, $9, $10, $11, $12,
                $13::jsonb, $14
            )
            """,
            [
                (
                    "content_ops.llm.complete",
                    "anthropic/claude-haiku-4-5",
                    "openrouter",
                    100,
                    40,
                    140,
                    90,
                    10,
                    5,
                    Decimal("0.100000"),
                    120,
                    "completed",
                    json.dumps({
                        "product": "content_ops",
                        "asset_type": "blog_post",
                        "request_id": request_id,
                        "cache_mode": "exact",
                        "cache_reason": "eligible",
                        "cache_result": "hit",
                        "cache_savings_usd": 0.0123,
                    }),
                    "run-a",
                ),
                (
                    "content_ops.llm.complete",
                    "anthropic/claude-haiku-4-5",
                    "openrouter",
                    30,
                    10,
                    40,
                    30,
                    0,
                    0,
                    Decimal("0.050000"),
                    80,
                    "failed",
                    json.dumps({
                        "product": "content_ops",
                        "asset_type": "blog_post",
                        "request_id": request_id,
                        "cache_mode": "no_store",
                        "cache_reason": "exact_cache_disabled",
                        "cache_savings_usd": "not-a-number",
                    }),
                    "run-a",
                ),
                (
                    "other.span",
                    "anthropic/claude-haiku-4-5",
                    "openrouter",
                    999,
                    999,
                    1998,
                    999,
                    0,
                    0,
                    Decimal("9.990000"),
                    10,
                    "completed",
                    json.dumps({
                        "product": "other",
                        "asset_type": "blog_post",
                        "request_id": request_id,
                    }),
                    "run-a",
                ),
            ],
        )

        payload = await summarize_content_ops_llm_usage(
            pool,
            days=1,
            asset_type="blog_post",
            run_id="run-a",
            request_id=request_id,
        )

        assert payload["summary"]["total_cost_usd"] == 0.15
        assert payload["summary"]["total_calls"] == 2
        assert payload["summary"]["failed_calls"] == 1
        assert payload["summary"]["input_tokens"] == 130
        assert payload["summary"]["output_tokens"] == 50
        assert payload["summary"]["total_tokens"] == 180
        assert payload["summary"]["billable_input_tokens"] == 120
        assert payload["summary"]["cached_tokens"] == 10
        assert payload["summary"]["cache_write_tokens"] == 5
        assert payload["summary"]["total_cache_savings_usd"] == 0.0123
        assert payload["summary"]["cache_hit_calls"] == 1
        assert payload["by_model"] == [
            {
                "provider": "openrouter",
                "model": "anthropic/claude-haiku-4-5",
                "cost_usd": 0.15,
                "cache_savings_usd": 0.0123,
                "calls": 2,
                "input_tokens": 130,
                "output_tokens": 50,
            }
        ]
        assert payload["by_asset_type"] == [
            {
                "asset_type": "blog_post",
                "cost_usd": 0.15,
                "cache_savings_usd": 0.0123,
                "calls": 2,
                "input_tokens": 130,
                "output_tokens": 50,
            }
        ]
        assert payload["by_cache_status"] == [
            {
                "cache_mode": "exact",
                "cache_reason": "eligible",
                "cache_result": "hit",
                "cache_store_result": "unknown",
                "cost_usd": 0.1,
                "cache_savings_usd": 0.0123,
                "calls": 1,
                "input_tokens": 100,
                "output_tokens": 40,
            },
            {
                "cache_mode": "no_store",
                "cache_reason": "exact_cache_disabled",
                "cache_result": "unknown",
                "cache_store_result": "unknown",
                "cost_usd": 0.05,
                "cache_savings_usd": 0.0,
                "calls": 1,
                "input_tokens": 30,
                "output_tokens": 10,
            },
        ]
    finally:
        await pool.execute(
            "DELETE FROM llm_usage WHERE metadata ->> 'request_id' = $1",
            request_id,
        )
        await pool.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_content_ops_usage_summary_isolates_accounts_against_postgres() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    database_url = os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("EXTRACTED_DATABASE_URL or DATABASE_URL is required")

    root = Path(__file__).resolve().parents[1]
    request_id = f"usage-summary-isolation-{uuid4().hex}"
    account_a = f"acct-a-{uuid4().hex}"
    account_b = f"acct-b-{uuid4().hex}"
    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=2)

    try:
        await pool.execute((root / "atlas_brain/storage/migrations/127_llm_usage.sql").read_text())
        await pool.execute((root / "atlas_brain/storage/migrations/252_llm_usage_cache_breakdown.sql").read_text())
        await pool.execute((root / "atlas_brain/storage/migrations/253_llm_usage_vendor_and_run_id.sql").read_text())
        await pool.executemany(
            """
            INSERT INTO llm_usage (
                span_name, operation_type, model_name, model_provider,
                input_tokens, output_tokens, total_tokens, billable_input_tokens,
                cached_tokens, cache_write_tokens, cost_usd, duration_ms, status,
                metadata, run_id
            )
            VALUES (
                $1, 'llm_call', $2, $3,
                $4, $5, $6, $7,
                $8, $9, $10, $11, $12,
                $13::jsonb, $14
            )
            """,
            [
                (
                    "content_ops.llm.complete",
                    "anthropic/claude-haiku-4-5",
                    "openrouter",
                    100,
                    40,
                    140,
                    90,
                    10,
                    5,
                    Decimal("0.100000"),
                    120,
                    "completed",
                    json.dumps({
                        "product": "content_ops",
                        "account_id": account_a,
                        "asset_type": "blog_post",
                        "request_id": request_id,
                        "cache_mode": "exact",
                        "cache_reason": "eligible",
                        "cache_result": "miss",
                        "cache_store_result": "stored",
                        "cache_savings_usd": 0.01,
                    }),
                    "run-a",
                ),
                (
                    "content_ops.llm.complete",
                    "anthropic/claude-haiku-4-5",
                    "openrouter",
                    200,
                    80,
                    280,
                    180,
                    20,
                    10,
                    Decimal("0.200000"),
                    240,
                    "completed",
                    json.dumps({
                        "product": "content_ops",
                        "account_id": account_a,
                        "asset_type": "landing_page",
                        "request_id": request_id,
                        "cache_mode": "exact",
                        "cache_reason": "eligible",
                        "cache_result": "hit",
                        "cache_savings_usd": 0.02,
                    }),
                    "run-a",
                ),
                (
                    "content_ops.llm.complete",
                    "anthropic/claude-haiku-4-5",
                    "openrouter",
                    500,
                    100,
                    600,
                    500,
                    0,
                    0,
                    Decimal("0.500000"),
                    300,
                    "completed",
                    json.dumps({
                        "product": "content_ops",
                        "account_id": account_b,
                        "asset_type": "blog_post",
                        "request_id": request_id,
                        "cache_mode": "no_store",
                        "cache_reason": "customer_data_no_store",
                        "cache_savings_usd": 0.05,
                    }),
                    "run-b",
                ),
            ],
        )

        account_a_payload = await summarize_content_ops_llm_usage(
            pool,
            days=1,
            account_id=account_a,
            request_id=request_id,
        )
        account_b_payload = await summarize_content_ops_llm_usage(
            pool,
            days=1,
            account_id=account_b,
            request_id=request_id,
        )

        assert account_a_payload["summary"]["total_cost_usd"] == 0.3
        assert account_a_payload["summary"]["total_cache_savings_usd"] == 0.03
        assert account_a_payload["summary"]["total_calls"] == 2
        assert account_a_payload["summary"]["input_tokens"] == 300
        assert account_a_payload["summary"]["output_tokens"] == 120
        assert account_a_payload["filters"]["account_id"] == account_a
        assert {row["asset_type"] for row in account_a_payload["by_asset_type"]} == {
            "blog_post",
            "landing_page",
        }
        assert {row["cache_result"] for row in account_a_payload["by_cache_status"]} == {
            "hit",
            "miss",
        }

        assert account_b_payload["summary"]["total_cost_usd"] == 0.5
        assert account_b_payload["summary"]["total_cache_savings_usd"] == 0.05
        assert account_b_payload["summary"]["total_calls"] == 1
        assert account_b_payload["summary"]["input_tokens"] == 500
        assert account_b_payload["summary"]["output_tokens"] == 100
        assert account_b_payload["filters"]["account_id"] == account_b
        assert account_b_payload["by_asset_type"] == [
            {
                "asset_type": "blog_post",
                "cost_usd": 0.5,
                "cache_savings_usd": 0.05,
                "calls": 1,
                "input_tokens": 500,
                "output_tokens": 100,
            }
        ]
        assert account_b_payload["by_cache_status"] == [
            {
                "cache_mode": "no_store",
                "cache_reason": "customer_data_no_store",
                "cache_result": "unknown",
                "cache_store_result": "unknown",
                "cost_usd": 0.5,
                "cache_savings_usd": 0.05,
                "calls": 1,
                "input_tokens": 500,
                "output_tokens": 100,
            }
        ]
    finally:
        await pool.execute(
            "DELETE FROM llm_usage WHERE metadata ->> 'request_id' = $1",
            request_id,
        )
        await pool.close()
