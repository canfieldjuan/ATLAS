from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import atlas_brain.autonomous.tasks.content_ops_deflection_delta_automation as mod
from extracted_content_pipeline.deflection_report_access import DeflectionDeltaBatchSummary


def _set_delta_settings(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    values = {
        "enabled": True,
        "cron_expression": "15 8 1 * *",
        "account_limit": 11,
        "reports_per_account": 9,
    }
    values.update(overrides)
    for name, value in values.items():
        monkeypatch.setattr(mod.settings.deflection_delta, name, value, raising=False)


def test_deflection_delta_automation_task_is_registered_builtin() -> None:
    from atlas_brain.autonomous.tasks import _BUILTIN_TASKS

    assert (
        "content_ops_deflection_delta_automation",
        "run",
        "content_ops_deflection_delta_automation",
    ) in _BUILTIN_TASKS


def test_deflection_delta_automation_default_task_seed_is_disabled_monthly_cron() -> None:
    from atlas_brain.autonomous.scheduler import TaskScheduler

    seed = next(
        (
            task
            for task in TaskScheduler._DEFAULT_TASKS
            if task.get("name") == "content_ops_deflection_delta_automation"
        ),
        None,
    )

    assert seed is not None
    assert seed["task_type"] == "builtin"
    assert seed["schedule_type"] == "cron"
    assert seed["cron_expression"] is None
    assert seed["enabled"] is False
    assert seed["timeout_seconds"] == 300
    assert seed["metadata"]["builtin_handler"] == "content_ops_deflection_delta_automation"
    assert seed["metadata"]["notify_tags"] == "chart_with_upwards_trend,content_ops,deflection"


def test_deflection_delta_automation_registers_on_runner() -> None:
    class _Recorder:
        def __init__(self) -> None:
            self.registered: dict[str, object] = {}

        def register_builtin(self, name: str, handler: object) -> None:
            self.registered[name] = handler

    from atlas_brain.autonomous.tasks import register_builtin_tasks

    runner = _Recorder()
    register_builtin_tasks(runner)

    handler = runner.registered["content_ops_deflection_delta_automation"]
    assert callable(handler)
    assert getattr(handler, "__name__", "") == "run"
    assert handler.__module__.endswith("content_ops_deflection_delta_automation")


@pytest.mark.asyncio
async def test_run_skips_when_delta_automation_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_delta_settings(monkeypatch, enabled=False)
    get_pool = AsyncMock()
    monkeypatch.setattr(mod, "get_db_pool", get_pool)

    result = await mod.run(SimpleNamespace(metadata={}))

    assert result == {"_skip_synthesis": "Deflection delta automation disabled"}
    get_pool.assert_not_called()


@pytest.mark.asyncio
async def test_run_skips_when_db_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_delta_settings(monkeypatch)
    pool = SimpleNamespace(is_initialized=False)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)

    result = await mod.run(SimpleNamespace(metadata={}))

    assert result == {"_skip_synthesis": "DB not ready"}


@pytest.mark.asyncio
async def test_run_delegates_to_batch_worker_with_typed_and_metadata_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_delta_settings(monkeypatch, account_limit=5, reports_per_account=4)
    pool = SimpleNamespace(is_initialized=True)
    calls: list[dict[str, Any]] = []

    async def _compute(store: Any, *, account_limit: int, reports_per_account: int) -> Any:
        calls.append(
            {
                "store_type": type(store).__name__,
                "store_pool": getattr(store, "pool", None),
                "account_limit": account_limit,
                "reports_per_account": reports_per_account,
            }
        )
        return DeflectionDeltaBatchSummary(
            accounts_scanned=2,
            reports_scanned=3,
            deltas_saved=1,
            skipped_no_delta=2,
            failed=0,
        )

    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "compute_and_save_recent_deflection_deltas", _compute)

    result = await mod.run(
        SimpleNamespace(metadata={"account_limit": "7", "reports_per_account": "6"})
    )

    assert calls == [
        {
            "store_type": "PostgresDeflectionReportArtifactStore",
            "store_pool": pool,
            "account_limit": 7,
            "reports_per_account": 6,
        }
    ]
    assert result == {
        "_skip_synthesis": "Deflection delta automation complete",
        "accounts_scanned": 2,
        "reports_scanned": 3,
        "deltas_saved": 1,
        "skipped_no_delta": 2,
        "failed": 0,
    }


@pytest.mark.asyncio
async def test_scheduler_seed_uses_typed_delta_cron_and_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atlas_brain.autonomous.scheduler import TaskScheduler
    from atlas_brain.storage.models import ScheduledTask

    _set_delta_settings(monkeypatch, enabled=True, cron_expression="45 9 1 * *")
    scheduler = TaskScheduler()
    scheduler._DEFAULT_TASKS = [
        {
            "name": "content_ops_deflection_delta_automation",
            "task_type": "builtin",
            "schedule_type": "cron",
            "cron_expression": None,
            "timeout_seconds": 300,
            "enabled": False,
            "metadata": {"builtin_handler": "content_ops_deflection_delta_automation"},
        },
    ]
    repo = AsyncMock()
    repo.get_by_name = AsyncMock(return_value=None)
    repo.create = AsyncMock(
        return_value=ScheduledTask(
            id=uuid4(),
            name="content_ops_deflection_delta_automation",
            task_type="builtin",
            schedule_type="cron",
            cron_expression="45 9 1 * *",
            timeout_seconds=300,
            enabled=True,
            metadata={"builtin_handler": "content_ops_deflection_delta_automation"},
        )
    )
    monkeypatch.setattr(
        "atlas_brain.storage.repositories.scheduled_task.get_scheduled_task_repo",
        lambda: repo,
    )
    monkeypatch.setattr(scheduler, "register_and_schedule", AsyncMock())
    monkeypatch.setattr("atlas_brain.pipelines.get_pipeline_interval_overrides", lambda: {})
    monkeypatch.setattr("atlas_brain.pipelines.get_pipeline_default_tasks", lambda: [])

    await scheduler._ensure_default_tasks()

    create_kwargs = repo.create.await_args.kwargs
    assert create_kwargs["enabled"] is True
    assert create_kwargs["cron_expression"] == "45 9 1 * *"
    assert create_kwargs["metadata"]["enabled_config_key"] == "settings.deflection_delta.enabled"
    assert create_kwargs["metadata"]["enabled_config_value"] is True
    assert scheduler.register_and_schedule.await_args.args[0].name == (
        "content_ops_deflection_delta_automation"
    )
