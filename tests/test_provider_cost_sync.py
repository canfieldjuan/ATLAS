from __future__ import annotations

import pytest

from atlas_brain.config import settings
from atlas_brain.services import provider_cost_sync as mod


class _FakePool:
    def __init__(self):
        self.execute_calls: list[tuple[str, tuple]] = []

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http_{self.status_code}")

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict | None, dict | None]] = []

    async def get(self, url, *, params=None, headers=None):
        self.calls.append((url, params, headers))
        if not self.responses:
            raise AssertionError("Unexpected HTTP request")
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_sync_provider_costs_writes_openrouter_snapshot(monkeypatch):
    monkeypatch.setattr(settings.provider_cost, "enabled", True, raising=False)
    monkeypatch.setattr(settings.provider_cost, "openrouter_enabled", True, raising=False)
    monkeypatch.setattr(settings.provider_cost, "anthropic_enabled", False, raising=False)
    monkeypatch.setattr(settings.provider_cost, "openrouter_api_key", "or-admin-key", raising=False)
    pool = _FakePool()
    client = _FakeClient(
        [
            _FakeResponse(
                {
                    "data": {
                        "total_credits": 100.5,
                        "total_usage": 25.75,
                    }
                }
            )
        ]
    )

    result = await mod.sync_provider_costs(pool=pool, client=client)

    assert result["openrouter_snapshot_written"] is True
    assert result["anthropic_daily_rows_upserted"] == 0
    assert result["providers_synced"] == ["openrouter"]
    assert any("INSERT INTO llm_provider_usage_snapshots" in query for query, _ in pool.execute_calls)
    assert any("DELETE FROM llm_provider_usage_snapshots" in query for query, _ in pool.execute_calls)


@pytest.mark.asyncio
async def test_sync_provider_costs_upserts_anthropic_daily_rows(monkeypatch):
    monkeypatch.setattr(settings.provider_cost, "enabled", True, raising=False)
    monkeypatch.setattr(settings.provider_cost, "openrouter_enabled", False, raising=False)
    monkeypatch.setattr(settings.provider_cost, "anthropic_enabled", True, raising=False)
    monkeypatch.setattr(settings.provider_cost, "anthropic_admin_api_key", "ant-admin-key", raising=False)
    monkeypatch.setattr(settings.provider_cost, "anthropic_lookback_days", 3, raising=False)
    pool = _FakePool()
    client = _FakeClient(
        [
            _FakeResponse(
                {
                    "data": [
                        {
                            "starting_at": "2026-04-01T00:00:00Z",
                            "ending_at": "2026-04-02T00:00:00Z",
                            "results": [
                                {"amount": 1.25, "currency": "USD"},
                                {"amount": 0.75, "currency": "USD"},
                            ],
                        }
                    ],
                    "has_more": False,
                    "next_page": None,
                }
            )
        ]
    )

    result = await mod.sync_provider_costs(pool=pool, client=client)

    assert result["openrouter_snapshot_written"] is False
    assert result["anthropic_daily_rows_upserted"] == 1
    assert result["providers_synced"] == ["anthropic"]
    assert any("INSERT INTO llm_provider_daily_costs" in query for query, _ in pool.execute_calls)
