import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from atlas_brain.autonomous.tasks import b2b_enrichment
from atlas_brain.storage.models import ScheduledTask


def _task() -> ScheduledTask:
    return ScheduledTask(
        id=uuid4(),
        name="b2b_enrichment",
        task_type="builtin",
        schedule_type="interval",
        interval_seconds=300,
        enabled=True,
        metadata={"builtin_handler": "b2b_enrichment"},
    )


class _Pool:
    def __init__(self, batches):
        self.is_initialized = True
        self.fetch = AsyncMock(side_effect=batches)
        self.fetchval = AsyncMock(return_value=0)
        self.execute = AsyncMock(return_value="UPDATE 0")


@pytest.mark.asyncio
async def test_run_limits_rounds_and_reports_orphan_recovery(monkeypatch):
    rows = [{"id": uuid4(), "enrichment_attempts": 0}]
    pool = _Pool([rows, rows])

    monkeypatch.setattr(b2b_enrichment, "get_db_pool", lambda: pool)
    monkeypatch.setattr(
        b2b_enrichment,
        "_recover_orphaned_enriching",
        AsyncMock(return_value=5),
    )
    monkeypatch.setattr(
        b2b_enrichment,
        "_queue_version_upgrades",
        AsyncMock(return_value=2),
    )
    monkeypatch.setattr(
        b2b_enrichment,
        "_enrich_rows",
        AsyncMock(return_value={"enriched": 3, "failed": 1, "no_signal": 2}),
    )

    cfg = b2b_enrichment.settings.b2b_churn
    monkeypatch.setattr(cfg, "enabled", True)
    monkeypatch.setattr(cfg, "enrichment_max_per_batch", 10)
    monkeypatch.setattr(cfg, "enrichment_max_attempts", 3)
    monkeypatch.setattr(cfg, "enrichment_max_rounds_per_run", 1)

    result = await b2b_enrichment.run(_task())

    assert result["rounds"] == 1
    assert result["orphaned_requeued"] == 5
    assert result["version_upgrade_requeued"] == 2
    assert result["enriched"] == 3
    assert pool.fetch.await_count == 1


@pytest.mark.asyncio
async def test_recover_orphaned_enriching_parses_update_count():
    pool = SimpleNamespace(execute=AsyncMock(return_value="UPDATE 7"))

    count = await b2b_enrichment._recover_orphaned_enriching(pool, 3)

    assert count == 7
    query = pool.execute.await_args.args[0]
    assert "WHERE enrichment_status = 'enriching'" in query


@pytest.mark.asyncio
async def test_enrich_rows_uses_configured_concurrency(monkeypatch):
    active = 0
    max_seen = 0

    async def _fake_enrich_single(pool, row, max_attempts, local_only, max_tokens, truncate_length):
        nonlocal active, max_seen
        active += 1
        max_seen = max(max_seen, active)
        await asyncio.sleep(0.01)
        active -= 1
        return True

    monkeypatch.setattr(b2b_enrichment, "_enrich_single", _fake_enrich_single)

    rows = [{"id": uuid4(), "enrichment_attempts": 0} for _ in range(5)]
    cfg = SimpleNamespace(
        enrichment_max_attempts=3,
        enrichment_concurrency=2,
        enrichment_local_only=False,
        enrichment_max_tokens=2048,
        review_truncate_length=3000,
    )
    pool = SimpleNamespace(fetchval=AsyncMock(return_value=0))

    result = await b2b_enrichment._enrich_rows(rows, cfg, pool)

    assert result["enriched"] == 5
    assert max_seen == 2
