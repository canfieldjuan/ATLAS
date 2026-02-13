"""
Hook manager for alert-driven autonomous task execution.

Listens for AlertManager callbacks and triggers hook tasks
when alert rules match.
"""

import asyncio
import copy
import logging
import time
from typing import Optional

from ..alerts.events import AlertEvent
from ..alerts.rules import AlertRule
from ..storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.hooks")


class HookManager:
    """
    Manages alert-to-task hook mappings.

    When an alert fires, HookManager checks if any hook tasks
    are registered for that alert rule and runs them via HeadlessRunner.
    """

    def __init__(self):
        self._rule_to_tasks: dict[str, list[str]] = {}  # rule_name -> [task_name, ...]
        self._last_execution: dict[tuple[str, str], float] = {}  # (task_name, rule_name) -> monotonic time
        self._loaded = False

    @property
    def hook_count(self) -> int:
        return sum(len(tasks) for tasks in self._rule_to_tasks.values())

    async def load_hooks_from_db(self) -> None:
        """Load hook task mappings from the database."""
        try:
            from ..storage.repositories.scheduled_task import get_scheduled_task_repo

            repo = get_scheduled_task_repo()
            tasks = await repo.get_enabled()

            self._rule_to_tasks.clear()

            for task in tasks:
                if task.task_type != "hook":
                    continue

                trigger_rules = (task.metadata or {}).get("trigger_rules", [])
                if not trigger_rules:
                    logger.warning("Hook task '%s' has no trigger_rules in metadata", task.name)
                    continue

                for rule_name in trigger_rules:
                    if rule_name not in self._rule_to_tasks:
                        self._rule_to_tasks[rule_name] = []
                    self._rule_to_tasks[rule_name].append(task.name)

            self._loaded = True
            logger.info(
                "Loaded %d hook mappings across %d rules",
                self.hook_count, len(self._rule_to_tasks),
            )

        except Exception as e:
            logger.error("Failed to load hooks from DB: %s", e)

    def _is_in_cooldown(self, task_name: str, rule_name: str, cooldown: int) -> bool:
        """Check if a (task, rule) pair is still within its cooldown window."""
        if cooldown <= 0:
            return False
        key = (task_name, rule_name)
        last = self._last_execution.get(key)
        if last is None:
            return False
        return (time.monotonic() - last) < cooldown

    def _record_execution_time(self, task_name: str, rule_name: str) -> None:
        """Record the monotonic time of a successful hook execution."""
        self._last_execution[(task_name, rule_name)] = time.monotonic()

    async def on_alert(
        self,
        message: str,
        rule: AlertRule,
        event: AlertEvent,
    ) -> None:
        """
        AlertManager callback. Triggers hook tasks for matching rules.

        Matches the AlertCallback signature:
            Callable[[str, AlertRule, AlertEvent], Awaitable[None]]
        """
        if not self._loaded:
            return

        task_names = self._rule_to_tasks.get(rule.name, [])
        if not task_names:
            return

        logger.info(
            "Alert '%s' matched %d hook task(s): %s",
            rule.name, len(task_names), task_names,
        )

        from ..storage.repositories.scheduled_task import get_scheduled_task_repo
        from .runner import get_headless_runner
        from .config import autonomous_config

        repo = get_scheduled_task_repo()
        runner = get_headless_runner()
        cooldown = autonomous_config.hook_cooldown_seconds

        for task_name in task_names:
            try:
                # Skip if within cooldown window
                if self._is_in_cooldown(task_name, rule.name, cooldown):
                    logger.info(
                        "Hook task '%s' skipped (cooldown %ds, triggered by %s)",
                        task_name, cooldown, rule.name,
                    )
                    continue

                task = await repo.get_by_name(task_name)
                if not task or not task.enabled:
                    continue

                # Build hook-specific task with alert context injected
                hook_task = self._inject_alert_context(task, message, rule, event)

                exec_id = await repo.record_execution(
                    task.id,
                    metadata={
                        "trigger": "hook",
                        "rule_name": rule.name,
                        "event_type": event.event_type,
                    },
                )

                start_time = time.monotonic()
                timeout = task.timeout_seconds or autonomous_config.task_timeout_seconds

                try:
                    result = await asyncio.wait_for(
                        runner.run(hook_task),
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

                    # Record execution time for cooldown (not on timeout — allow re-trigger)
                    self._record_execution_time(task_name, rule.name)

                    logger.info(
                        "Hook task '%s' %s in %dms (triggered by %s)",
                        task_name, status, duration_ms, rule.name,
                    )

                except asyncio.TimeoutError:
                    duration_ms = int((time.monotonic() - start_time) * 1000)
                    await repo.complete_execution(
                        exec_id, "timeout",
                        duration_ms=duration_ms,
                        error=f"Hook timed out after {timeout}s",
                    )
                    logger.warning(
                        "Hook task '%s' timed out after %ds (triggered by %s)",
                        task_name, timeout, rule.name,
                    )

            except Exception as e:
                logger.error("Hook task '%s' failed: %s", task_name, e)

    async def on_alert_batch(self, batch: list) -> None:
        """
        Process a batch of deduplicated events from EventQueue.

        Groups events by rule and picks the highest-count event per rule,
        then dispatches a single hook execution per rule per batch.

        Args:
            batch: List of QueuedEvent objects from EventQueue
        """
        if not self._loaded or not batch:
            return

        # Group by rule name, keep highest-count event per rule
        rule_events: dict[str, object] = {}  # rule_name -> QueuedEvent
        for qe in batch:
            rn = qe.rule.name
            if rn not in rule_events or qe.count > rule_events[rn].count:
                rule_events[rn] = qe

        logger.info(
            "Processing batch of %d events across %d rules",
            len(batch), len(rule_events),
        )

        from ..storage.repositories.scheduled_task import get_scheduled_task_repo
        from .runner import get_headless_runner
        from .config import autonomous_config

        repo = get_scheduled_task_repo()
        runner = get_headless_runner()
        cooldown = autonomous_config.hook_cooldown_seconds

        for rule_name, qe in rule_events.items():
            task_names = self._rule_to_tasks.get(rule_name, [])
            if not task_names:
                continue

            for task_name in task_names:
                try:
                    if self._is_in_cooldown(task_name, rule_name, cooldown):
                        logger.debug(
                            "Hook task '%s' skipped in batch (cooldown, rule=%s)",
                            task_name, rule_name,
                        )
                        continue

                    task = await repo.get_by_name(task_name)
                    if not task or not task.enabled:
                        continue

                    # Inject batch context into the alert message
                    batch_msg = qe.message
                    if qe.count > 1:
                        elapsed = (qe.last_seen - qe.first_seen).total_seconds()
                        batch_msg = (
                            f"{qe.message} "
                            f"({qe.count} events in {elapsed:.0f}s)"
                        )

                    hook_task = self._inject_alert_context(
                        task, batch_msg, qe.rule, qe.event,
                    )

                    exec_id = await repo.record_execution(
                        task.id,
                        metadata={
                            "trigger": "hook_batch",
                            "rule_name": rule_name,
                            "event_type": qe.event.event_type,
                            "batch_count": qe.count,
                        },
                    )

                    start_time = time.monotonic()
                    timeout = task.timeout_seconds or autonomous_config.task_timeout_seconds

                    try:
                        result = await asyncio.wait_for(
                            runner.run(hook_task),
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
                        self._record_execution_time(task_name, rule_name)

                        logger.info(
                            "Hook task '%s' %s in %dms (batch, rule=%s, events=%d)",
                            task_name, status, duration_ms, rule_name, qe.count,
                        )

                    except asyncio.TimeoutError:
                        duration_ms = int((time.monotonic() - start_time) * 1000)
                        await repo.complete_execution(
                            exec_id, "timeout",
                            duration_ms=duration_ms,
                            error=f"Hook timed out after {timeout}s",
                        )
                        logger.warning(
                            "Hook task '%s' timed out in batch after %ds (rule=%s)",
                            task_name, timeout, rule_name,
                        )

                except Exception as e:
                    logger.error("Batch hook task '%s' failed: %s", task_name, e)

    def _inject_alert_context(
        self,
        task: ScheduledTask,
        message: str,
        rule: AlertRule,
        event: AlertEvent,
    ) -> ScheduledTask:
        """Create a copy of the task with alert context injected into the prompt."""
        hook_task = copy.deepcopy(task)

        alert_context = (
            f"\n\n[Alert Context]\n"
            f"Rule: {rule.name}\n"
            f"Event type: {event.event_type}\n"
            f"Source: {event.source_id}\n"
            f"Message: {message}\n"
            f"Timestamp: {event.timestamp.isoformat()}\n"
        )

        if event.metadata:
            alert_context += f"Event data: {event.metadata}\n"

        hook_task.prompt = (task.prompt or "") + alert_context
        return hook_task


_hook_manager: Optional[HookManager] = None


def get_hook_manager() -> HookManager:
    """Get the global hook manager."""
    global _hook_manager
    if _hook_manager is None:
        _hook_manager = HookManager()
    return _hook_manager
