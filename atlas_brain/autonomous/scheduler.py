"""
Task scheduler for autonomous agent execution.

Uses APScheduler as the timing engine with PostgreSQL as the
source of truth for task definitions and execution history.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from ..storage.models import ScheduledTask
from .config import autonomous_config

logger = logging.getLogger("atlas.autonomous.scheduler")


class TaskScheduler:
    """
    Wraps APScheduler AsyncIOScheduler for autonomous task execution.

    APScheduler handles timing; PostgreSQL is the source of truth
    for task definitions and execution history.
    """

    def __init__(self):
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._background_tasks: set[asyncio.Task] = set()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def scheduled_count(self) -> int:
        """Derive count from APScheduler's actual job list."""
        if self._scheduler and self._running:
            return len(self._scheduler.get_jobs())
        return 0

    async def start(self) -> None:
        """Create the scheduler, start it, and load tasks from DB."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._semaphore = asyncio.Semaphore(autonomous_config.max_concurrent_tasks)

        self._scheduler = AsyncIOScheduler(
            job_defaults={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": 300,
            }
        )

        self._scheduler.start()
        self._running = True

        await self._load_tasks_from_db()

        # Seed default builtin tasks if they do not exist yet
        await self._ensure_default_tasks()

        # Sync configurable intervals from settings to DB
        await self._sync_configurable_intervals()

        # Mark orphaned running executions from previous crashes
        await self._cleanup_orphaned_executions()

        logger.info("TaskScheduler started with %d tasks", self.scheduled_count)

    async def stop(self) -> None:
        """Shutdown the scheduler gracefully."""
        if self._scheduler and self._running:
            # Use wait=False to avoid blocking the async event loop.
            # In-flight tasks are already wrapped in asyncio.wait_for()
            # with their own timeouts, so they'll complete or timeout
            # on their own.
            self._scheduler.shutdown(wait=False)
            self._running = False
            logger.info("TaskScheduler stopped")

    async def _load_tasks_from_db(self) -> None:
        """Load enabled tasks from the database and register them."""
        try:
            from ..storage.repositories.scheduled_task import get_scheduled_task_repo

            repo = get_scheduled_task_repo()
            tasks = await repo.get_enabled()

            for task in tasks:
                try:
                    self._register_task(task)
                except Exception as e:
                    logger.error("Failed to register task '%s': %s", task.name, e)

        except Exception as e:
            logger.error("Failed to load tasks from DB: %s", e)

    async def _cleanup_orphaned_executions(self) -> None:
        """Mark executions stuck in 'running' status as failed (from prior crashes)."""
        try:
            from ..storage.database import get_db_pool

            pool = get_db_pool()
            if not pool.is_initialized:
                return

            result = await pool.execute(
                """
                UPDATE task_executions
                SET status = 'failed',
                    completed_at = $1,
                    error = 'Server restarted while task was running'
                WHERE status = 'running'
                """,
                datetime.now(timezone.utc),
            )

            # Parse count from "UPDATE N"
            count = 0
            if result:
                try:
                    count = int(result.split()[-1])
                except (ValueError, IndexError):
                    pass

            if count > 0:
                logger.warning("Marked %d orphaned task executions as failed", count)

        except Exception as e:
            logger.error("Failed to cleanup orphaned executions: %s", e)

    # Default builtin task definitions seeded on first start.
    _DEFAULT_TASKS = [
        {
            "name": "nightly_memory_sync",
            "description": "Nightly batch sync of conversations to GraphRAG and purge of old PostgreSQL messages",
            "task_type": "builtin",
            "schedule_type": "cron",
            "cron_expression": "0 3 * * *",
            "timeout_seconds": 300,
            "metadata": {"builtin_handler": "nightly_memory_sync"},
        },
        {
            "name": "email_graph_sync",
            "description": "Daily extraction of graph-worthy facts from emails to knowledge graph",
            "task_type": "builtin",
            "schedule_type": "cron",
            "cron_expression": "0 1 * * *",
            "timeout_seconds": 1800,  # 30 min: ~80s/message * 20 max emails + overhead
            "metadata": {"builtin_handler": "email_graph_sync"},
        },
        {
            "name": "cleanup_old_executions",
            "description": "Purge old task execution records, presence events, and resolved proactive actions",
            "task_type": "builtin",
            "schedule_type": "cron",
            "cron_expression": "30 3 * * *",
            "timeout_seconds": 120,
            "metadata": {"builtin_handler": "cleanup_old_executions"},
        },
        # Phase 2: digest tasks with LLM synthesis
        {
            "name": "device_health_check",
            "description": "Daily device health scan",
            "task_type": "builtin",
            "schedule_type": "cron",
            "cron_expression": "0 6 * * *",
            "timeout_seconds": 60,
            "metadata": {
                "builtin_handler": "device_health_check",
                "synthesis_skill": "digest/device_health",
            },
        },
        {
            "name": "proactive_actions",
            "description": "Extract actionable items from recent conversations",
            "task_type": "builtin",
            "schedule_type": "cron",
            "cron_expression": "30 6 * * *",
            "timeout_seconds": 60,
            "metadata": {
                "builtin_handler": "proactive_actions",
                "synthesis_skill": "digest/proactive_actions",
            },
        },
        {
            "name": "morning_briefing",
            "description": "Daily morning briefing: calendar, weather, security, devices, pending actions",
            "task_type": "builtin",
            "schedule_type": "cron",
            "cron_expression": "0 7 * * *",
            "timeout_seconds": 120,
            "metadata": {
                "builtin_handler": "morning_briefing",
                "synthesis_skill": "digest/morning_briefing",
                "synthesis_llm": "email_triage",
                "calendar_hours": 12,
                "security_hours": 8,
            },
        },
        {
            "name": "gmail_digest",
            "description": "Daily email digest with triage",
            "task_type": "builtin",
            "schedule_type": "cron",
            "cron_expression": "5 7 * * *",
            "timeout_seconds": 60,
            "metadata": {
                "builtin_handler": "gmail_digest",
                "synthesis_skill": "digest/email_triage",
                "synthesis_llm": "email_triage",
                "notify_tags": "email,inbox",
            },
        },
        {
            "name": "email_draft",
            "description": "Generate reply drafts for action-required emails",
            "task_type": "builtin",
            "schedule_type": "interval",
            "interval_seconds": None,  # resolved from settings.email_draft.schedule_interval_seconds
            "timeout_seconds": 120,
            "metadata": {"builtin_handler": "email_draft"},
        },
        {
            "name": "email_intake",
            "description": "Near-real-time email polling with CRM cross-reference",
            "task_type": "builtin",
            "schedule_type": "interval",
            "interval_seconds": None,  # resolved from settings.email_intake.interval_seconds
            "timeout_seconds": 120,
            "metadata": {"builtin_handler": "email_intake"},
        },
        {
            "name": "security_summary",
            "description": "Periodic security event and alert aggregation",
            "task_type": "builtin",
            "schedule_type": "interval",
            "interval_seconds": 21600,
            "timeout_seconds": 60,
            "metadata": {
                "builtin_handler": "security_summary",
                "synthesis_skill": "digest/security_summary",
                "hours": 6,
            },
        },
        # Hook-triggered: runs on presence departure, no cron schedule
        # Note: departure_check is called internally by departure_auto_fix,
        # it does not need its own _DEFAULT_TASKS entry.
        {
            "name": "departure_auto_fix",
            "description": "Auto-fix lights, locks, covers when house goes empty",
            "task_type": "builtin",
            "schedule_type": "cron",
            "cron_expression": None,
            "timeout_seconds": 30,
            "metadata": {
                "builtin_handler": "departure_auto_fix",
                "trigger_rules": ["presence_departure"],
                "synthesis_skill": "digest/departure_check",
            },
        },
        # Proactive: calendar reminders when someone is home
        {
            "name": "calendar_reminder",
            "description": "Check upcoming calendar events and notify when someone is home",
            "task_type": "builtin",
            "schedule_type": "interval",
            "interval_seconds": 300,
            "timeout_seconds": 30,
            "metadata": {
                "builtin_handler": "calendar_reminder",
                "lead_minutes": 30,
                "min_minutes": 15,
            },
        },
        # Proactive: nudge about aging pending action items
        {
            "name": "action_escalation",
            "description": "Nudge about aging pending action items when someone is home",
            "task_type": "builtin",
            "schedule_type": "interval",
            "interval_seconds": 14400,
            "timeout_seconds": 30,
            "metadata": {
                "builtin_handler": "action_escalation",
                "synthesis_skill": "digest/action_escalation",
            },
        },
        # Nightly: learn temporal patterns from presence and session data
        {
            "name": "pattern_learning",
            "description": "Learn temporal patterns from presence and session data",
            "task_type": "builtin",
            "schedule_type": "cron",
            "cron_expression": "0 2 * * *",
            "timeout_seconds": 60,
            "metadata": {
                "builtin_handler": "pattern_learning",
                "lookback_days": 30,
            },
        },
        # Nightly: auto-tune user preferences from conversation patterns
        {
            "name": "preference_learning",
            "description": "Auto-tune user response style and expertise level from conversation patterns",
            "task_type": "builtin",
            "schedule_type": "cron",
            "cron_expression": "30 2 * * *",
            "timeout_seconds": 60,
            "metadata": {
                "builtin_handler": "preference_learning",
                "lookback_days": 7,
                "min_turns": 20,
            },
        },
        # Proactive: detect unusual presence patterns and device states
        {
            "name": "anomaly_detection",
            "description": "Detect unusual presence patterns and device states",
            "task_type": "builtin",
            "schedule_type": "interval",
            "interval_seconds": 900,
            "timeout_seconds": 30,
            "metadata": {
                "builtin_handler": "anomaly_detection",
                "deviation_threshold": 2.0,
                "min_samples": 5,
            },
        },
        # Model swap: keep qwen3:14b loaded during the day, free VRAM at night for graphiti.
        # Both tasks no-op when ATLAS_LLM__MODEL_SWAP_ENABLED=false (default).
        {
            "name": "model_swap_day",
            "description": "Unload night model (qwen3:32b), pre-warm day model (qwen3:14b) at 7:30 AM",
            "task_type": "builtin",
            "schedule_type": "cron",
            "cron_expression": "30 7 * * *",
            "timeout_seconds": 300,  # load_model httpx timeout is 120s + list/unload overhead
            "metadata": {"builtin_handler": "model_swap_day"},
        },
        {
            "name": "model_swap_night",
            "description": "Unload day model (qwen3:14b) at midnight to free VRAM for graphiti-wrapper",
            "task_type": "builtin",
            "schedule_type": "cron",
            "cron_expression": "0 0 * * *",
            "timeout_seconds": 60,
            "metadata": {"builtin_handler": "model_swap_night"},
        },
        {
            "name": "email_backfill",
            "description": "Scan Sent Mail and populate CRM contacts from recipients",
            "task_type": "builtin",
            "schedule_type": "interval",
            "interval_seconds": 86400,
            "timeout_seconds": 1800,
            "enabled": False,
            "metadata": {
                "builtin_handler": "email_backfill",
                "max_days": 730,
                "batch_size": 10,
                "mailbox": "[Gmail]/Sent Mail",
            },
        },
        {
            "name": "email_auto_approve",
            "description": "Auto-approve and send eligible email drafts after delay window",
            "task_type": "builtin",
            "schedule_type": "interval",
            "interval_seconds": 120,
            "timeout_seconds": 60,
            "metadata": {"builtin_handler": "email_auto_approve"},
        },
        {
            "name": "email_stale_check",
            "description": "Detect stale drafts, unactioned emails, and unanswered estimates",
            "task_type": "builtin",
            "schedule_type": "interval",
            "interval_seconds": None,  # resolved from config
            "timeout_seconds": 120,
            "metadata": {"builtin_handler": "email_stale_check"},
        },
        {
            "name": "invoice_overdue_check",
            "description": "Daily scan for overdue invoices, mark and notify",
            "task_type": "builtin",
            "schedule_type": "cron",
            "cron_expression": "0 9 * * *",
            "timeout_seconds": 60,
            "metadata": {
                "builtin_handler": "invoice_overdue_check",
                "synthesis_skill": "invoicing/overdue_notification",
            },
        },
        {
            "name": "invoice_payment_reminders",
            "description": "Daily payment reminders for overdue invoices",
            "task_type": "builtin",
            "schedule_type": "cron",
            "cron_expression": "0 10 * * *",
            "timeout_seconds": 120,
            "metadata": {
                "builtin_handler": "invoice_payment_reminders",
                "synthesis_skill": "invoicing/payment_reminder",
            },
        },
        {
            "name": "monthly_invoice_generation",
            "description": "Generate and send monthly invoices from calendar cleaning events",
            "task_type": "builtin",
            "schedule_type": "cron",
            "cron_expression": "0 8 1 * *",
            "timeout_seconds": 300,
            "metadata": {
                "builtin_handler": "monthly_invoice_generation",
                "notify_priority": "high",
            },
        },
        {
            "name": "reasoning_tick",
            "description": "Polling safety net for reasoning agent missed events",
            "task_type": "builtin",
            "schedule_type": "interval",
            "interval_seconds": 300,
            "timeout_seconds": 120,
            "metadata": {
                "builtin_handler": "reasoning_tick",
            },
        },
        {
            "name": "reasoning_reflection",
            "description": "Proactive cross-domain pattern detection and recommendations",
            "task_type": "builtin",
            "schedule_type": "cron",
            "cron_expression": "0 9,13,17,21 * * *",
            "timeout_seconds": 300,
            "metadata": {
                "builtin_handler": "reasoning_reflection",
            },
        },
    ]

    async def _ensure_default_tasks(self) -> None:
        """Seed default builtin tasks if they do not already exist.

        Idempotent: checks by name before inserting. Newly created tasks
        are immediately registered with APScheduler.
        """
        try:
            from ..config import settings
            from ..storage.repositories.scheduled_task import get_scheduled_task_repo

            repo = get_scheduled_task_repo()

            # Resolve configurable intervals at runtime
            _interval_overrides = {
                "email_draft": settings.email_draft.schedule_interval_seconds,
                "email_intake": settings.email_intake.interval_seconds,
                "email_stale_check": settings.email_stale_check.interval_seconds,
            }

            for task_def in self._DEFAULT_TASKS:
                # Apply runtime interval override if the definition left it as None
                if task_def.get("interval_seconds") is None and task_def["name"] in _interval_overrides:
                    task_def = {**task_def, "interval_seconds": _interval_overrides[task_def["name"]]}
                existing = await repo.get_by_name(task_def["name"])
                if existing is not None:
                    # Merge any new metadata keys from the definition into the existing row.
                    # This propagates code-side metadata changes (e.g. synthesis_llm) to
                    # already-seeded tasks without overwriting user customisations.
                    default_meta = task_def.get("metadata") or {}
                    existing_meta = existing.metadata or {}
                    new_keys = {k: v for k, v in default_meta.items() if k not in existing_meta}
                    if new_keys:
                        merged = {**existing_meta, **new_keys}
                        from ..storage.database import get_db_pool
                        pool = get_db_pool()
                        if not pool.is_initialized:
                            logger.warning(
                                "DB not available; skipping metadata merge for task '%s'",
                                task_def["name"],
                            )
                        else:
                            await pool.execute(
                                "UPDATE scheduled_tasks SET metadata = $1::jsonb WHERE id = $2",
                                json.dumps(merged), existing.id,
                            )
                            logger.info(
                                "Merged new metadata keys into task '%s': %s",
                                task_def["name"], list(new_keys),
                            )
                    continue

                task = await repo.create(
                    name=task_def["name"],
                    description=task_def.get("description"),
                    task_type=task_def["task_type"],
                    schedule_type=task_def["schedule_type"],
                    cron_expression=task_def.get("cron_expression"),
                    interval_seconds=task_def.get("interval_seconds"),
                    timeout_seconds=task_def.get("timeout_seconds", 120),
                    enabled=task_def.get("enabled", True),
                    metadata=task_def.get("metadata"),
                )
                if task.enabled:
                    self._register_task(task)
                schedule_info = (
                    task.cron_expression
                    or (f"every {task.interval_seconds}s" if task.interval_seconds else task.schedule_type)
                )
                logger.info(
                    "Seeded default task '%s' (%s, enabled=%s)",
                    task.name,
                    schedule_info,
                    task.enabled,
                )

        except Exception as e:
            logger.warning("Failed to seed default tasks: %s", e)

    async def _sync_configurable_intervals(self) -> None:
        """Update DB task schedules that differ from current config values.

        Allows env-var tuning to take effect on restart without manual DB edits.
        Handles both interval_seconds (e.g. ATLAS_EMAIL_DRAFT_SCHEDULE_INTERVAL_SECONDS)
        and cron_expression (e.g. ATLAS_LLM__MODEL_SWAP_DAY_CRON).
        """
        try:
            from ..config import settings
            from ..storage.repositories.scheduled_task import get_scheduled_task_repo

            repo = get_scheduled_task_repo()

            # Interval-based overrides
            interval_overrides = {
                "email_draft": settings.email_draft.schedule_interval_seconds,
                "email_intake": settings.email_intake.interval_seconds,
                "email_stale_check": settings.email_stale_check.interval_seconds,
            }

            for task_name, desired_interval in interval_overrides.items():
                task = await repo.get_by_name(task_name)
                if task is None or task.interval_seconds == desired_interval:
                    continue

                old = task.interval_seconds
                updated = await repo.update(task.id, interval_seconds=desired_interval)
                if updated:
                    self._register_task(updated)
                    logger.info(
                        "Updated '%s' interval: %ss -> %ss (from config)",
                        task_name, old, desired_interval,
                    )

            # Cron-based overrides (model swap times + email_graph_sync migration 6AM->1AM)
            cron_overrides = {
                "model_swap_day": settings.llm.model_swap_day_cron,
                "model_swap_night": settings.llm.model_swap_night_cron,
                "email_graph_sync": "0 1 * * *",
                "reasoning_reflection": settings.reasoning.reflection_cron,
            }

            for task_name, desired_cron in cron_overrides.items():
                task = await repo.get_by_name(task_name)
                if task is None or task.cron_expression == desired_cron:
                    continue

                old = task.cron_expression
                updated = await repo.update(task.id, cron_expression=desired_cron)
                if updated:
                    self._register_task(updated)
                    logger.info(
                        "Updated '%s' cron: '%s' -> '%s' (from config)",
                        task_name, old, desired_cron,
                    )

        except Exception as e:
            logger.warning("Failed to sync configurable intervals: %s", e)

    def _register_task(self, task: ScheduledTask) -> None:
        """Register a task with APScheduler."""
        trigger = self._build_trigger(task)
        if trigger is None:
            # Hook-only tasks intentionally have no schedule; don't warn.
            is_hook_only = bool((task.metadata or {}).get("trigger_rules"))
            log = logger.debug if is_hook_only else logger.warning
            log(
                "Could not build trigger for task '%s' (schedule_type=%s)",
                task.name, task.schedule_type,
            )
            return

        self._scheduler.add_job(
            self._execute_task,
            trigger=trigger,
            id=str(task.id),
            name=task.name,
            replace_existing=True,
            args=[task.id],
        )
        logger.info("Registered task '%s' (%s)", task.name, task.schedule_type)

    def unregister_task(self, task_id: str) -> None:
        """Remove a task from APScheduler."""
        if self._scheduler:
            try:
                self._scheduler.remove_job(task_id)
                logger.info("Unregistered task %s", task_id)
            except Exception:
                logger.debug("Task %s not found in scheduler", task_id)

    def _build_trigger(self, task: ScheduledTask):
        """Build an APScheduler trigger from task schedule config."""
        default_tz = autonomous_config.default_timezone
        if task.schedule_type == "cron" and task.cron_expression:
            return CronTrigger.from_crontab(
                task.cron_expression,
                timezone=task.timezone or default_tz,
            )
        elif task.schedule_type == "interval" and task.interval_seconds:
            return IntervalTrigger(seconds=task.interval_seconds)
        elif task.schedule_type == "once" and task.run_at:
            return DateTrigger(
                run_date=task.run_at,
                timezone=task.timezone or default_tz,
            )
        return None

    async def _execute_task(self, task_id, retry_count: int = 0) -> None:
        """Execute a scheduled task with concurrency control and error handling."""
        from ..storage.repositories.scheduled_task import get_scheduled_task_repo
        from .runner import get_headless_runner

        repo = get_scheduled_task_repo()

        async with self._semaphore:
            task = await repo.get_by_id(task_id)
            if not task or not task.enabled:
                logger.debug("Skipping disabled/missing task %s", task_id)
                return

            exec_id = await repo.record_execution(task_id, retry_count=retry_count)
            start_time = time.monotonic()
            now = datetime.now(timezone.utc)

            logger.info("Executing task '%s' (exec_id=%s, retry=%d)", task.name, exec_id, retry_count)

            try:
                timeout = task.timeout_seconds or autonomous_config.task_timeout_seconds
                runner = get_headless_runner()
                result = await asyncio.wait_for(
                    runner.run(task),
                    timeout=timeout,
                )

                duration_ms = int((time.monotonic() - start_time) * 1000)
                status = "completed" if result.success else "failed"

                await repo.complete_execution(
                    exec_id, status,
                    result_text=result.response_text,
                    duration_ms=duration_ms,
                    error=result.error,
                )

                next_run = self._get_next_run_time(task)
                await repo.update_last_run(task_id, now, next_run)

                logger.info(
                    "Task '%s' %s in %dms",
                    task.name, status, duration_ms,
                )

                try:
                    from ..events.broadcaster import broadcast_system_event
                    evt_level = "error" if status == "failed" else "info"
                    await broadcast_system_event(
                        "task", evt_level,
                        "Task '%s' %s (%dms)" % (task.name, status, duration_ms),
                    )
                except Exception:
                    pass

                if status == "failed":
                    self._maybe_schedule_retry(task, retry_count)
                    await self._check_consecutive_failures(task.id)

            except asyncio.TimeoutError:
                duration_ms = int((time.monotonic() - start_time) * 1000)
                await repo.complete_execution(
                    exec_id, "timeout",
                    duration_ms=duration_ms,
                    error=f"Task timed out after {task.timeout_seconds}s",
                )
                next_run = self._get_next_run_time(task)
                await repo.update_last_run(task_id, now, next_run)
                logger.warning("Task '%s' timed out after %ds", task.name, task.timeout_seconds)
                await self._check_consecutive_failures(task.id)

            except Exception as e:
                duration_ms = int((time.monotonic() - start_time) * 1000)
                await repo.complete_execution(
                    exec_id, "failed",
                    duration_ms=duration_ms,
                    error=str(e),
                )
                next_run = self._get_next_run_time(task)
                await repo.update_last_run(task_id, now, next_run)
                logger.error("Task '%s' failed: %s", task.name, e)
                self._maybe_schedule_retry(task, retry_count)
                await self._check_consecutive_failures(task.id)

    async def _check_consecutive_failures(self, task_id) -> None:
        """Auto-disable a task after N consecutive primary-run failures."""
        threshold = autonomous_config.auto_disable_after_failures
        if threshold <= 0:
            return

        from ..storage.repositories.scheduled_task import get_scheduled_task_repo
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return

        try:
            rows = await pool.fetch(
                """
                SELECT status FROM task_executions
                WHERE task_id = $1 AND retry_count = 0
                ORDER BY started_at DESC
                LIMIT $2
                """,
                task_id, threshold,
            )

            if len(rows) < threshold:
                return

            if all(r["status"] in ("failed", "timeout") for r in rows):
                repo = get_scheduled_task_repo()
                await repo.update(task_id, enabled=False)
                self.unregister_task(str(task_id))
                logger.warning(
                    "Auto-disabled task %s after %d consecutive failures",
                    task_id, threshold,
                )
        except Exception as e:
            logger.error("Failed to check consecutive failures for %s: %s", task_id, e)

    def _maybe_schedule_retry(self, task: ScheduledTask, current_retry_count: int) -> None:
        """Schedule a retry if the task has retries remaining."""
        if not self._scheduler or not self._running:
            return

        next_retry = current_retry_count + 1
        if task.max_retries <= 0 or next_retry > task.max_retries:
            return

        delay = task.retry_delay_seconds or 60
        run_date = datetime.now(timezone.utc) + timedelta(seconds=delay)
        job_id = f"{task.id}_retry_{next_retry}"

        self._scheduler.add_job(
            self._execute_task,
            trigger=DateTrigger(run_date=run_date),
            id=job_id,
            name=f"{task.name}_retry_{next_retry}",
            replace_existing=True,
            args=[task.id, next_retry],
        )
        logger.info(
            "Scheduled retry %d/%d for task '%s' in %ds (job_id=%s)",
            next_retry, task.max_retries, task.name, delay, job_id,
        )

    def _get_next_run_time(self, task: ScheduledTask) -> Optional[datetime]:
        """Get next run time from APScheduler job."""
        if not self._scheduler:
            return None
        try:
            job = self._scheduler.get_job(str(task.id))
            if job and job.next_run_time:
                return job.next_run_time
        except Exception:
            pass
        return None

    async def register_and_schedule(self, task: ScheduledTask) -> None:
        """Register a new task (called when task is created via API)."""
        if self._running and task.enabled:
            self._register_task(task)

    async def run_now(self, task: ScheduledTask) -> dict:
        """Execute a task immediately (manual trigger). Returns immediately with execution ID."""
        from ..storage.repositories.scheduled_task import get_scheduled_task_repo

        repo = get_scheduled_task_repo()
        exec_id = await repo.record_execution(task.id, metadata={"trigger": "manual"})

        bg_task = asyncio.create_task(
            self._run_task_background(task, exec_id),
            name=f"manual_{task.name}",
        )
        self._background_tasks.add(bg_task)
        bg_task.add_done_callback(self._background_tasks.discard)

        return {
            "execution_id": str(exec_id),
            "status": "running",
            "message": "Task started. Poll GET /{task_id}/executions for status.",
        }

    async def _run_task_background(self, task: ScheduledTask, exec_id) -> None:
        """Background execution for manual run_now triggers."""
        from ..storage.repositories.scheduled_task import get_scheduled_task_repo
        from .runner import get_headless_runner

        repo = get_scheduled_task_repo()
        start_time = time.monotonic()
        now = datetime.now(timezone.utc)

        try:
            async with self._semaphore:
                timeout = task.timeout_seconds or autonomous_config.task_timeout_seconds
                runner = get_headless_runner()
                result = await asyncio.wait_for(
                    runner.run(task),
                    timeout=timeout,
                )

                duration_ms = int((time.monotonic() - start_time) * 1000)
                status = "completed" if result.success else "failed"

                await repo.complete_execution(
                    exec_id, status,
                    result_text=result.response_text,
                    duration_ms=duration_ms,
                    error=result.error,
                )
                await repo.update_last_run(task.id, now, self._get_next_run_time(task))

                logger.info("Manual task '%s' %s in %dms", task.name, status, duration_ms)

                try:
                    from ..events.broadcaster import broadcast_system_event
                    evt_level = "error" if status == "failed" else "info"
                    await broadcast_system_event(
                        "task", evt_level,
                        "Task '%s' %s (%dms)" % (task.name, status, duration_ms),
                    )
                except Exception:
                    pass

                if status == "failed":
                    self._maybe_schedule_retry(task, 0)
                    await self._check_consecutive_failures(task.id)

        except asyncio.TimeoutError:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            await repo.complete_execution(
                exec_id, "timeout",
                duration_ms=duration_ms,
                error=f"Task timed out after {task.timeout_seconds}s",
            )
            await repo.update_last_run(task.id, now, self._get_next_run_time(task))
            logger.warning("Manual task '%s' timed out after %ds", task.name, task.timeout_seconds)
            await self._check_consecutive_failures(task.id)

        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            await repo.complete_execution(
                exec_id, "failed",
                duration_ms=duration_ms,
                error=str(e),
            )
            await repo.update_last_run(task.id, now, self._get_next_run_time(task))
            logger.error("Manual task '%s' failed: %s", task.name, e)
            self._maybe_schedule_retry(task, 0)
            await self._check_consecutive_failures(task.id)


_task_scheduler: Optional[TaskScheduler] = None


def get_task_scheduler() -> TaskScheduler:
    """Get the global task scheduler."""
    global _task_scheduler
    if _task_scheduler is None:
        _task_scheduler = TaskScheduler()
    return _task_scheduler
