"""
Unit tests for TaskScheduler trigger building and retry logic.

Tests _build_trigger and _maybe_schedule_retry directly (no DB).
Tests _execute_task with mocked DB repo and headless runner.
"""

from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import HTTPException

from atlas_brain.api.autonomous import disable_task, enable_task, run_task_now
from atlas_brain.autonomous.scheduler import TaskScheduler
from atlas_brain.storage.models import ScheduledTask, TaskExecution
from atlas_brain.storage.repositories.scheduled_task import ScheduledTaskRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scheduler() -> TaskScheduler:
    return TaskScheduler()


def _task(
    schedule_type: str = "cron",
    cron: str | None = "0 8 * * *",
    interval: int | None = None,
    run_at: datetime | None = None,
    max_retries: int = 0,
    retry_delay: int = 60,
    timeout: int = 120,
    enabled: bool = True,
    ) -> ScheduledTask:
    return ScheduledTask(
        id=uuid4(),
        name="test_task",
        task_type="agent_prompt",
        schedule_type=schedule_type,
        cron_expression=cron,
        interval_seconds=interval,
        run_at=run_at,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay,
        timeout_seconds=timeout,
        enabled=enabled,
        prompt="Do something",
    )


class _FakeExecPool:
    def __init__(self) -> None:
        self.is_initialized = True
        self.calls: list[tuple] = []

    async def execute(self, *args):
        self.calls.append(args)


# ---------------------------------------------------------------------------
# Trigger building (pure, no mocks)
# ---------------------------------------------------------------------------

class TestBuildTrigger:
    """Verify _build_trigger produces correct APScheduler trigger types."""

    def test_cron_schedule_returns_cron_trigger(self):
        s = _scheduler()
        task = _task(schedule_type="cron", cron="0 8 * * *")
        trigger = s._build_trigger(task)
        assert isinstance(trigger, CronTrigger)

    def test_cron_day_of_week_uses_standard_crontab_semantics(self):
        s = _scheduler()
        task = _task(schedule_type="cron", cron="0 9 * * 1")
        trigger = s._build_trigger(task)

        start = datetime(2026, 3, 22, 8, 0, tzinfo=ZoneInfo("America/Chicago"))
        next_fire = trigger.get_next_fire_time(None, start)

        assert next_fire is not None
        assert next_fire.date().isoformat() == "2026-03-23"
        assert next_fire.weekday() == 0

    def test_cron_day_range_maps_weekdays_correctly(self):
        s = _scheduler()
        task = _task(schedule_type="cron", cron="0 23 * * 1-5")
        trigger = s._build_trigger(task)

        start = datetime(2026, 3, 22, 20, 0, tzinfo=ZoneInfo("America/Chicago"))
        next_fire = trigger.get_next_fire_time(None, start)

        assert next_fire is not None
        assert next_fire.date().isoformat() == "2026-03-23"
        assert next_fire.weekday() == 0

    def test_interval_schedule_returns_interval_trigger(self):
        s = _scheduler()
        task = _task(schedule_type="interval", cron=None, interval=3600)
        trigger = s._build_trigger(task)
        assert isinstance(trigger, IntervalTrigger)

    def test_once_schedule_returns_date_trigger(self):
        s = _scheduler()
        run_at = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
        task = _task(schedule_type="once", cron=None, run_at=run_at)
        trigger = s._build_trigger(task)
        assert isinstance(trigger, DateTrigger)

    def test_unknown_schedule_returns_none(self):
        s = _scheduler()
        task = _task(schedule_type="unknown_type", cron=None)
        trigger = s._build_trigger(task)
        assert trigger is None

    def test_cron_without_expression_returns_none(self):
        s = _scheduler()
        task = _task(schedule_type="cron", cron=None)
        trigger = s._build_trigger(task)
        assert trigger is None

    def test_interval_without_seconds_returns_none(self):
        s = _scheduler()
        task = _task(schedule_type="interval", cron=None, interval=None)
        trigger = s._build_trigger(task)
        assert trigger is None

    def test_once_without_run_at_returns_none(self):
        s = _scheduler()
        task = _task(schedule_type="once", cron=None, run_at=None)
        trigger = s._build_trigger(task)
        assert trigger is None


class TestExecutionMetadataRepository:
    @pytest.mark.asyncio
    @patch("atlas_brain.storage.repositories.scheduled_task.get_db_pool")
    async def test_update_execution_metadata_uses_jsonb_merge(self, mock_pool_fn):
        repo = ScheduledTaskRepository()
        pool = _FakeExecPool()
        mock_pool_fn.return_value = pool
        exec_id = uuid4()

        await repo.update_execution_metadata(exec_id, {"stage": "llm_overlay", "cards_built": 15})

        assert len(pool.calls) == 1
        query, called_exec_id, metadata_json = pool.calls[0]
        assert "metadata = COALESCE(metadata, '{}'::jsonb) || $2::jsonb" in query
        assert called_exec_id == exec_id
        assert '"stage": "llm_overlay"' in metadata_json


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    """Verify _maybe_schedule_retry scheduling behavior."""

    def test_retry_scheduled_when_retries_remaining(self):
        s = _scheduler()
        s._running = True
        s._scheduler = MagicMock()

        task = _task(max_retries=3, retry_delay=30)
        s._maybe_schedule_retry(task, current_retry_count=1)

        s._scheduler.add_job.assert_called_once()
        call_kwargs = s._scheduler.add_job.call_args
        assert "retry_2" in call_kwargs.kwargs.get("id", "") or "retry_2" in str(call_kwargs)

    def test_no_retry_when_max_retries_zero(self):
        s = _scheduler()
        s._running = True
        s._scheduler = MagicMock()

        task = _task(max_retries=0)
        s._maybe_schedule_retry(task, current_retry_count=0)

        s._scheduler.add_job.assert_not_called()

    def test_no_retry_when_retries_exhausted(self):
        s = _scheduler()
        s._running = True
        s._scheduler = MagicMock()

        task = _task(max_retries=2)
        s._maybe_schedule_retry(task, current_retry_count=2)

        s._scheduler.add_job.assert_not_called()

    def test_retry_uses_task_delay(self):
        s = _scheduler()
        s._running = True
        s._scheduler = MagicMock()

        task = _task(max_retries=3, retry_delay=120)
        s._maybe_schedule_retry(task, current_retry_count=0)

        # Verify the trigger uses approximately the right delay
        call_args = s._scheduler.add_job.call_args
        trigger = call_args.kwargs.get("trigger") or call_args[1].get("trigger")
        assert isinstance(trigger, DateTrigger)

    def test_no_retry_when_scheduler_not_running(self):
        s = _scheduler()
        s._running = False
        s._scheduler = MagicMock()

        task = _task(max_retries=3)
        s._maybe_schedule_retry(task, current_retry_count=0)

        s._scheduler.add_job.assert_not_called()


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------

class TestExecuteTask:
    """Verify _execute_task with mocked dependencies."""

    @pytest.mark.asyncio
    @patch("atlas_brain.autonomous.runner.get_headless_runner")
    @patch("atlas_brain.storage.repositories.scheduled_task.get_scheduled_task_repo")
    async def test_skips_disabled_task(self, mock_repo_fn, mock_runner_fn):
        s = _scheduler()
        s._semaphore = __import__("asyncio").Semaphore(2)

        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = _task(enabled=False)
        mock_repo_fn.return_value = mock_repo

        task_id = uuid4()
        await s._execute_task(task_id)

        mock_runner_fn.return_value.run.assert_not_called()

    @pytest.mark.asyncio
    @patch("atlas_brain.autonomous.runner.get_headless_runner")
    @patch("atlas_brain.storage.repositories.scheduled_task.get_scheduled_task_repo")
    async def test_skips_missing_task(self, mock_repo_fn, mock_runner_fn):
        s = _scheduler()
        s._semaphore = __import__("asyncio").Semaphore(2)

        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = None
        mock_repo_fn.return_value = mock_repo

        await s._execute_task(uuid4())

        mock_runner_fn.return_value.run.assert_not_called()

    @pytest.mark.asyncio
    @patch.object(TaskScheduler, "_check_consecutive_failures", new_callable=AsyncMock)
    @patch.object(TaskScheduler, "_get_next_run_time", return_value=None)
    @patch("atlas_brain.autonomous.runner.get_headless_runner")
    @patch("atlas_brain.storage.repositories.scheduled_task.get_scheduled_task_repo")
    async def test_records_execution_and_calls_runner(
        self, mock_repo_fn, mock_runner_fn, _next_run, _check_fail
    ):
        s = _scheduler()
        s._semaphore = __import__("asyncio").Semaphore(2)

        task = _task(enabled=True)
        exec_id = uuid4()

        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = task
        mock_repo.record_execution.return_value = exec_id
        mock_repo_fn.return_value = mock_repo

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.response_text = "done"
        mock_result.error = None
        mock_runner = AsyncMock()
        mock_runner.run.return_value = mock_result
        mock_runner_fn.return_value = mock_runner

        await s._execute_task(task.id)

        mock_repo.record_execution.assert_awaited_once()
        mock_runner.run.assert_awaited_once_with(task)
        mock_repo.complete_execution.assert_awaited_once()
        status_arg = mock_repo.complete_execution.call_args[0][1]
        assert status_arg == "completed"
        assert task.metadata["_execution_id"] == str(exec_id)


class TestRunNow:
    @pytest.mark.asyncio
    @patch("atlas_brain.storage.repositories.scheduled_task.get_scheduled_task_repo")
    async def test_returns_existing_running_execution_without_spawning(self, mock_repo_fn):
        s = _scheduler()
        task = _task(enabled=True)
        running_exec = TaskExecution(id=uuid4(), task_id=task.id, status="running")

        mock_repo = AsyncMock()
        mock_repo.get_running_execution.return_value = running_exec
        mock_repo_fn.return_value = mock_repo

        with patch("asyncio.create_task") as mock_create_task:
            result = await s.run_now(task)

        assert result["already_running"] is True
        assert result["execution_id"] == str(running_exec.id)
        mock_repo.record_execution.assert_not_awaited()
        mock_create_task.assert_not_called()

    @pytest.mark.asyncio
    @patch("atlas_brain.storage.repositories.scheduled_task.get_scheduled_task_repo")
    async def test_records_execution_when_not_running(self, mock_repo_fn):
        s = _scheduler()
        task = _task(enabled=True)
        exec_id = uuid4()

        mock_repo = AsyncMock()
        mock_repo.get_running_execution.return_value = None
        mock_repo.record_execution.return_value = exec_id
        mock_repo_fn.return_value = mock_repo

        dummy_task = MagicMock()
        def _fake_create_task(coro, **_kwargs):
            coro.close()
            return dummy_task

        with patch("asyncio.create_task", side_effect=_fake_create_task) as mock_create_task:
            result = await s.run_now(task)

        assert result["execution_id"] == str(exec_id)
        assert result["status"] == "running"
        assert str(task.id) in s._manual_running_task_ids
        mock_repo.record_execution.assert_awaited_once()
        mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(TaskScheduler, "_check_consecutive_failures", new_callable=AsyncMock)
    @patch.object(TaskScheduler, "_get_next_run_time", return_value=None)
    @patch("atlas_brain.autonomous.runner.get_headless_runner")
    @patch("atlas_brain.storage.repositories.scheduled_task.get_scheduled_task_repo")
    async def test_manual_background_run_injects_execution_id(
        self, mock_repo_fn, mock_runner_fn, _next_run, _check_fail
    ):
        s = _scheduler()
        s._semaphore = __import__("asyncio").Semaphore(2)
        task = _task(enabled=True)
        exec_id = uuid4()

        mock_repo = AsyncMock()
        mock_repo_fn.return_value = mock_repo

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.response_text = "done"
        mock_result.error = None
        mock_runner = AsyncMock()
        mock_runner.run.return_value = mock_result
        mock_runner_fn.return_value = mock_runner

        await s._run_task_background(task, exec_id)

        mock_runner.run.assert_awaited_once_with(task)
        assert task.metadata["_execution_id"] == str(exec_id)
        mock_repo.complete_execution.assert_awaited_once()


class TestScheduleSync:
    @pytest.mark.asyncio
    @patch("atlas_brain.storage.repositories.scheduled_task.get_scheduled_task_repo")
    async def test_register_and_schedule_persists_next_run(self, mock_repo_fn):
        s = _scheduler()
        s._running = True
        s._scheduler = MagicMock()

        next_run = datetime(2026, 3, 23, 20, 0, tzinfo=timezone.utc)
        s._scheduler.get_job.return_value = MagicMock(next_run_time=next_run)

        task = _task(schedule_type="interval", cron=None, interval=300, enabled=True)
        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = None
        mock_repo_fn.return_value = mock_repo

        result = await s.register_and_schedule(task)

        s._scheduler.add_job.assert_called_once()
        mock_repo.update_next_run.assert_awaited_once_with(task.id, next_run)
        assert result.next_run_at == next_run

    @pytest.mark.asyncio
    @patch("atlas_brain.storage.repositories.scheduled_task.get_scheduled_task_repo")
    async def test_register_and_schedule_clears_next_run_when_disabled(self, mock_repo_fn):
        s = _scheduler()
        s._running = True
        s._scheduler = MagicMock()

        task = _task(enabled=False)
        task.next_run_at = datetime(2026, 3, 23, 20, 0, tzinfo=timezone.utc)
        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = None
        mock_repo_fn.return_value = mock_repo

        result = await s.register_and_schedule(task)

        s._scheduler.add_job.assert_not_called()
        s._scheduler.remove_job.assert_called_once_with(str(task.id))
        mock_repo.update_next_run.assert_awaited_once_with(task.id, None)
        assert result.next_run_at is None


class TestAutonomousApiRunNow:
    @pytest.mark.asyncio
    @patch("atlas_brain.autonomous.scheduler.get_task_scheduler")
    @patch("atlas_brain.storage.repositories.scheduled_task.get_scheduled_task_repo")
    async def test_api_returns_409_when_task_already_running(self, mock_repo_fn, mock_sched_fn):
        task = _task(enabled=True)

        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = task
        mock_repo_fn.return_value = mock_repo

        mock_scheduler = AsyncMock()
        mock_scheduler.run_now.return_value = {
            "execution_id": str(uuid4()),
            "status": "running",
            "message": f"Task '{task.name}' is already running.",
            "already_running": True,
        }
        mock_sched_fn.return_value = mock_scheduler

        with pytest.raises(HTTPException) as exc:
            await run_task_now(task.id)

        assert exc.value.status_code == 409
        assert "already running" in exc.value.detail["message"]

    @pytest.mark.asyncio
    @patch.object(TaskScheduler, "_check_consecutive_failures", new_callable=AsyncMock)
    @patch.object(TaskScheduler, "_maybe_schedule_retry")
    @patch.object(TaskScheduler, "_get_next_run_time", return_value=None)
    @patch("atlas_brain.autonomous.runner.get_headless_runner")
    @patch("atlas_brain.storage.repositories.scheduled_task.get_scheduled_task_repo")
    async def test_failed_result_triggers_retry(
        self, mock_repo_fn, mock_runner_fn, _next_run, mock_retry, _check_fail
    ):
        s = _scheduler()
        s._semaphore = __import__("asyncio").Semaphore(2)

        task = _task(enabled=True, max_retries=2)

        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = task
        mock_repo.record_execution.return_value = uuid4()
        mock_repo_fn.return_value = mock_repo

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.response_text = ""
        mock_result.error = "something failed"
        mock_runner = AsyncMock()
        mock_runner.run.return_value = mock_result
        mock_runner_fn.return_value = mock_runner

        await s._execute_task(task.id)

        mock_retry.assert_called_once_with(task, 0)

    @pytest.mark.asyncio
    @patch.object(TaskScheduler, "_check_consecutive_failures", new_callable=AsyncMock)
    @patch.object(TaskScheduler, "_get_next_run_time", return_value=None)
    @patch("atlas_brain.autonomous.runner.get_headless_runner")
    @patch("atlas_brain.storage.repositories.scheduled_task.get_scheduled_task_repo")
    async def test_timeout_records_timeout_status(
        self, mock_repo_fn, mock_runner_fn, _next_run, _check_fail
    ):
        import asyncio

        s = _scheduler()
        s._semaphore = asyncio.Semaphore(2)

        task = _task(enabled=True, timeout=1)

        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = task
        mock_repo.record_execution.return_value = uuid4()
        mock_repo_fn.return_value = mock_repo

        async def slow_run(_task):
            await asyncio.sleep(10)

        mock_runner = AsyncMock()
        mock_runner.run.side_effect = slow_run
        mock_runner_fn.return_value = mock_runner

        await s._execute_task(task.id)

        mock_repo.complete_execution.assert_awaited_once()
        status_arg = mock_repo.complete_execution.call_args[0][1]
        assert status_arg == "timeout"


class TestAutonomousApiScheduling:
    @pytest.mark.asyncio
    @patch("atlas_brain.autonomous.scheduler.get_task_scheduler")
    @patch("atlas_brain.storage.repositories.scheduled_task.get_scheduled_task_repo")
    async def test_enable_task_returns_rescheduled_next_run(self, mock_repo_fn, mock_sched_fn):
        task = _task(enabled=True, schedule_type="interval", cron=None, interval=300)
        next_run = datetime(2026, 3, 23, 20, 0, tzinfo=timezone.utc)
        task.next_run_at = next_run

        mock_repo = AsyncMock()
        mock_repo.update.return_value = _task(enabled=True, schedule_type="interval", cron=None, interval=300)
        mock_repo_fn.return_value = mock_repo

        mock_scheduler = AsyncMock()
        mock_scheduler.register_and_schedule.return_value = task
        mock_sched_fn.return_value = mock_scheduler

        result = await enable_task(task.id)

        assert result["enabled"] is True
        assert result["next_run_at"] == next_run.isoformat()

    @pytest.mark.asyncio
    @patch("atlas_brain.autonomous.scheduler.get_task_scheduler")
    @patch("atlas_brain.storage.repositories.scheduled_task.get_scheduled_task_repo")
    async def test_disable_task_clears_next_run(self, mock_repo_fn, mock_sched_fn):
        task = _task(enabled=False)
        task.next_run_at = None

        mock_repo = AsyncMock()
        mock_repo.update.return_value = _task(enabled=False)
        mock_repo_fn.return_value = mock_repo

        mock_scheduler = AsyncMock()
        mock_scheduler.register_and_schedule.return_value = task
        mock_sched_fn.return_value = mock_scheduler

        result = await disable_task(task.id)

        assert result["enabled"] is False
        assert result["next_run_at"] is None


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    """Verify initial scheduler state."""

    def test_not_running_initially(self):
        s = _scheduler()
        assert not s.is_running

    def test_scheduled_count_zero_when_not_running(self):
        s = _scheduler()
        assert s.scheduled_count == 0

    @pytest.mark.asyncio
    @patch("atlas_brain.storage.repositories.scheduled_task.get_scheduled_task_repo")
    async def test_ensure_default_tasks_continues_when_seeded_task_registration_fails(self, mock_repo_fn, monkeypatch):
        s = _scheduler()
        s._DEFAULT_TASKS = [
            {
                "name": "seed_one",
                "task_type": "builtin",
                "schedule_type": "interval",
                "interval_seconds": 60,
                "enabled": True,
                "metadata": {},
            },
            {
                "name": "seed_two",
                "task_type": "builtin",
                "schedule_type": "interval",
                "interval_seconds": 120,
                "enabled": True,
                "metadata": {},
            },
        ]

        created_one = ScheduledTask(
            id=uuid4(),
            name="seed_one",
            task_type="builtin",
            schedule_type="interval",
            interval_seconds=60,
            enabled=True,
        )
        created_two = ScheduledTask(
            id=uuid4(),
            name="seed_two",
            task_type="builtin",
            schedule_type="interval",
            interval_seconds=120,
            enabled=True,
        )

        repo = AsyncMock()
        repo.get_by_name = AsyncMock(side_effect=[None, None])
        repo.create = AsyncMock(side_effect=[created_one, created_two])
        mock_repo_fn.return_value = repo

        async def _register_side_effect(task):
            if task.name == "seed_one":
                raise RuntimeError("registration failed")
            return task

        monkeypatch.setattr(s, "register_and_schedule", AsyncMock(side_effect=_register_side_effect))

        await s._ensure_default_tasks()

        assert repo.create.await_count == 2
        assert s.register_and_schedule.await_count == 2
