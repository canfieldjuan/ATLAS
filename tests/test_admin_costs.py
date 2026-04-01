from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from atlas_brain.api.admin_costs import router


class _FakePool:
    def __init__(self):
        self.is_initialized = True
        self.last_fetch_query = ""
        self.last_fetch_args = ()

    async def fetchrow(self, query, *args):
        if "today_cost" in query:
            return {"today_cost": Decimal("4.25"), "today_calls": 11}
        return {
            "total_cost": Decimal("12.5"),
            "total_input": 12000,
            "total_billable_input": 8000,
            "total_cached_tokens": 3000,
            "total_cache_write_tokens": 1000,
            "total_output": 2100,
            "total_tokens": 14100,
            "total_calls": 17,
            "cache_hit_calls": 9,
            "cache_write_calls": 3,
            "avg_duration_ms": 420.2,
            "avg_tps": 33.7,
        }

    async def fetch(self, query, *args):
        self.last_fetch_query = query
        self.last_fetch_args = args
        if "GROUP BY span_name, operation_type, model_name, model_provider" in query:
            return [{
                "span_name": "task.b2b_blog_post_generation",
                "operation_type": "llm_call",
                "model_name": "anthropic/claude-sonnet-4-6",
                "model_provider": "openrouter",
                "cost": Decimal("1.234"),
                "input_tokens": 10000,
                "billable_input_tokens": 4200,
                "cached_tokens": 5000,
                "cache_write_tokens": 800,
                "output_tokens": 1800,
                "total_tokens": 11800,
                "calls": 4,
                "cache_hit_calls": 3,
                "cache_write_calls": 1,
                "avg_duration_ms": 811.5,
                "latest_created_at": datetime(2026, 3, 31, 22, 0, tzinfo=timezone.utc),
            }]
        return [{
            "id": uuid4(),
            "span_name": "task.b2b_blog_post_generation",
            "operation_type": "llm_call",
            "model_name": "anthropic/claude-sonnet-4-6",
            "model_provider": "openrouter",
            "input_tokens": 10000,
            "billable_input_tokens": 4200,
            "cached_tokens": 5000,
            "cache_write_tokens": 800,
            "output_tokens": 1800,
            "total_tokens": 11800,
            "cost_usd": Decimal("0.0483"),
            "duration_ms": 922,
            "ttft_ms": 210,
            "inference_time_ms": 640,
            "queue_time_ms": 12,
            "tokens_per_second": 28.1,
            "status": "completed",
            "api_endpoint": "https://openrouter.ai/api/v1/chat/completions",
            "provider_request_id": "req_123",
            "metadata": '{"vendor_name":"HubSpot"}',
            "created_at": datetime(2026, 3, 31, 22, 1, tzinfo=timezone.utc),
        }]


def _client(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    pool = _FakePool()
    monkeypatch.setattr("atlas_brain.api.admin_costs.get_db_pool", lambda: pool)
    return TestClient(app), pool


def test_cost_summary_includes_cache_metrics(monkeypatch):
    client, _ = _client(monkeypatch)
    with client:
        res = client.get("/admin/costs/summary?days=30")
    assert res.status_code == 200
    body = res.json()
    assert body["total_billable_input_tokens"] == 8000
    assert body["total_cached_tokens"] == 3000
    assert body["total_cache_write_tokens"] == 1000
    assert body["cache_hit_calls"] == 9
    assert body["cache_write_calls"] == 3


def test_cost_by_operation_exposes_cache_rollups(monkeypatch):
    client, _ = _client(monkeypatch)
    with client:
        res = client.get("/admin/costs/by-operation?days=7&limit=25")
    assert res.status_code == 200
    row = res.json()["operations"][0]
    assert row["span_name"] == "task.b2b_blog_post_generation"
    assert row["billable_input_tokens"] == 4200
    assert row["cached_tokens"] == 5000
    assert row["cache_write_tokens"] == 800
    assert row["cache_hit_calls"] == 3


def test_recent_calls_returns_granular_cache_fields(monkeypatch):
    client, pool = _client(monkeypatch)
    with client:
        res = client.get(
            "/admin/costs/recent?days=7&provider=openrouter&status=completed&cache_only=true&limit=10"
        )
    assert res.status_code == 200
    row = res.json()["calls"][0]
    assert row["detail"] == "HubSpot"
    assert row["billable_input_tokens"] == 4200
    assert row["cached_tokens"] == 5000
    assert row["cache_write_tokens"] == 800
    assert row["cache_hit"] is True
    assert row["cache_write"] is True
    assert "model_provider = $2" in pool.last_fetch_query
    assert "(cached_tokens > 0 OR cache_write_tokens > 0)" in pool.last_fetch_query
