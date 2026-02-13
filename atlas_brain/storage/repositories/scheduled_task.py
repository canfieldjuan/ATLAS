"""
Scheduled task repository for persistence and retrieval.

Provides CRUD operations for scheduled tasks and execution history
stored in PostgreSQL.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from ..database import get_db_pool
from ..exceptions import DatabaseUnavailableError, DatabaseOperationError
from ..models import ScheduledTask, TaskExecution

logger = logging.getLogger("atlas.storage.scheduled_task")


class ScheduledTaskRepository:
    _default_timezone: str = "America/Chicago"

    def __init__(self):
        try:
            from ..config import settings
            self._default_timezone = settings.autonomous.default_timezone
        except Exception:
            pass  # Use class default if config not available
    """
    Repository for scheduled task storage and retrieval.

    Handles persistence of autonomous task definitions and
    their execution history.
    """

    async def create(
        self,
        name: str,
        task_type: str,
        schedule_type: str,
        description: Optional[str] = None,
        prompt: Optional[str] = None,
        agent_type: str = "atlas",
        cron_expression: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        run_at: Optional[datetime] = None,
        timezone_str: Optional[str] = None,
        enabled: bool = True,
        max_retries: int = 0,
        retry_delay_seconds: int = 60,
        timeout_seconds: int = 120,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ScheduledTask:
        """Create a new scheduled task."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("create scheduled task")

        task_id = uuid4()
        now = datetime.now(timezone.utc)
        metadata_json = json.dumps(metadata or {})

        try:
            row = await pool.fetchrow(
                """
                INSERT INTO scheduled_tasks (
                    id, name, description, task_type, prompt, agent_type,
                    schedule_type, cron_expression, interval_seconds, run_at,
                    timezone, enabled, max_retries, retry_delay_seconds,
                    timeout_seconds, metadata, created_at, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16::jsonb, $17, $18)
                RETURNING id, created_at, updated_at
                """,
                task_id, name, description, task_type, prompt, agent_type,
                schedule_type, cron_expression, interval_seconds, run_at,
                timezone_str or self._default_timezone, enabled, max_retries, retry_delay_seconds,
                timeout_seconds, metadata_json, now, now,
            )

            if row:
                logger.info("Created scheduled task %s: %s (%s)", task_id, name, task_type)
                return ScheduledTask(
                    id=row["id"],
                    name=name,
                    description=description,
                    task_type=task_type,
                    prompt=prompt,
                    agent_type=agent_type,
                    schedule_type=schedule_type,
                    cron_expression=cron_expression,
                    interval_seconds=interval_seconds,
                    run_at=run_at,
                    timezone=timezone_str or self._default_timezone,
                    enabled=enabled,
                    max_retries=max_retries,
                    retry_delay_seconds=retry_delay_seconds,
                    timeout_seconds=timeout_seconds,
                    metadata=metadata or {},
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )

            raise DatabaseOperationError("create scheduled task", Exception("No row returned"))

        except (DatabaseUnavailableError, DatabaseOperationError):
            raise
        except Exception as e:
            logger.error("Failed to create scheduled task: %s", e)
            raise DatabaseOperationError("create scheduled task", e)

    async def get_by_id(self, task_id: UUID) -> Optional[ScheduledTask]:
        """Get a scheduled task by ID."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("get scheduled task by id")

        try:
            row = await pool.fetchrow(
                """
                SELECT id, name, description, task_type, prompt, agent_type,
                       schedule_type, cron_expression, interval_seconds, run_at,
                       timezone, enabled, max_retries, retry_delay_seconds,
                       timeout_seconds, metadata, created_at, updated_at,
                       last_run_at, next_run_at
                FROM scheduled_tasks
                WHERE id = $1
                """,
                task_id,
            )

            if row:
                return self._row_to_task(row)
            return None

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get scheduled task by id", e)

    async def get_by_name(self, name: str) -> Optional[ScheduledTask]:
        """Get a scheduled task by name."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("get scheduled task by name")

        try:
            row = await pool.fetchrow(
                """
                SELECT id, name, description, task_type, prompt, agent_type,
                       schedule_type, cron_expression, interval_seconds, run_at,
                       timezone, enabled, max_retries, retry_delay_seconds,
                       timeout_seconds, metadata, created_at, updated_at,
                       last_run_at, next_run_at
                FROM scheduled_tasks
                WHERE name = $1
                """,
                name,
            )

            if row:
                return self._row_to_task(row)
            return None

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get scheduled task by name", e)

    async def get_enabled(self) -> list[ScheduledTask]:
        """Get all enabled scheduled tasks."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("get enabled scheduled tasks")

        try:
            rows = await pool.fetch(
                """
                SELECT id, name, description, task_type, prompt, agent_type,
                       schedule_type, cron_expression, interval_seconds, run_at,
                       timezone, enabled, max_retries, retry_delay_seconds,
                       timeout_seconds, metadata, created_at, updated_at,
                       last_run_at, next_run_at
                FROM scheduled_tasks
                WHERE enabled = TRUE
                ORDER BY created_at ASC
                """
            )

            return [self._row_to_task(row) for row in rows]

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get enabled scheduled tasks", e)

    async def list_all(self, include_disabled: bool = False) -> list[ScheduledTask]:
        """List all scheduled tasks."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("list scheduled tasks")

        try:
            if include_disabled:
                rows = await pool.fetch(
                    """
                    SELECT id, name, description, task_type, prompt, agent_type,
                           schedule_type, cron_expression, interval_seconds, run_at,
                           timezone, enabled, max_retries, retry_delay_seconds,
                           timeout_seconds, metadata, created_at, updated_at,
                           last_run_at, next_run_at
                    FROM scheduled_tasks
                    ORDER BY created_at ASC
                    """
                )
            else:
                rows = await pool.fetch(
                    """
                    SELECT id, name, description, task_type, prompt, agent_type,
                           schedule_type, cron_expression, interval_seconds, run_at,
                           timezone, enabled, max_retries, retry_delay_seconds,
                           timeout_seconds, metadata, created_at, updated_at,
                           last_run_at, next_run_at
                    FROM scheduled_tasks
                    WHERE enabled = TRUE
                    ORDER BY created_at ASC
                    """
                )

            return [self._row_to_task(row) for row in rows]

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("list scheduled tasks", e)

    async def update(self, task_id: UUID, **fields) -> Optional[ScheduledTask]:
        """Update a scheduled task's fields."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("update scheduled task")

        allowed_fields = {
            "name", "description", "task_type", "prompt", "agent_type",
            "schedule_type", "cron_expression", "interval_seconds", "run_at",
            "timezone", "enabled", "max_retries", "retry_delay_seconds",
            "timeout_seconds", "metadata",
        }

        update_fields = {k: v for k, v in fields.items() if k in allowed_fields}
        if not update_fields:
            return await self.get_by_id(task_id)

        try:
            set_clauses = []
            params = []
            param_idx = 1

            for field_name, value in update_fields.items():
                if field_name == "metadata":
                    set_clauses.append(f"metadata = ${param_idx}::jsonb")
                    params.append(json.dumps(value) if value is not None else None)
                else:
                    set_clauses.append(f"{field_name} = ${param_idx}")
                    params.append(value)
                param_idx += 1

            set_clauses.append(f"updated_at = ${param_idx}")
            params.append(datetime.now(timezone.utc))
            param_idx += 1

            params.append(task_id)

            query = f"""
                UPDATE scheduled_tasks
                SET {', '.join(set_clauses)}
                WHERE id = ${param_idx}
            """

            await pool.execute(query, *params)
            logger.info("Updated scheduled task %s", task_id)
            return await self.get_by_id(task_id)

        except (DatabaseUnavailableError, DatabaseOperationError):
            raise
        except Exception as e:
            logger.error("Failed to update scheduled task: %s", e)
            raise DatabaseOperationError("update scheduled task", e)

    async def delete(self, task_id: UUID) -> bool:
        """Delete a scheduled task."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("delete scheduled task")

        try:
            result = await pool.execute(
                "DELETE FROM scheduled_tasks WHERE id = $1",
                task_id,
            )

            success = self._parse_row_count(result) > 0
            if success:
                logger.info("Deleted scheduled task: %s", task_id)
            return success

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("delete scheduled task", e)

    async def update_last_run(
        self,
        task_id: UUID,
        last_run_at: datetime,
        next_run_at: Optional[datetime] = None,
    ) -> None:
        """Update last run and next run timestamps."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("update task last run")

        try:
            await pool.execute(
                """
                UPDATE scheduled_tasks
                SET last_run_at = $2, next_run_at = $3, updated_at = $4
                WHERE id = $1
                """,
                task_id, last_run_at, next_run_at, datetime.now(timezone.utc),
            )
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            logger.error("Failed to update task last run: %s", e)
            raise DatabaseOperationError("update task last run", e)

    async def record_execution(
        self,
        task_id: UUID,
        status: str = "running",
        metadata: Optional[dict[str, Any]] = None,
        retry_count: int = 0,
    ) -> UUID:
        """Record the start of a task execution. Returns execution ID."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("record task execution")

        exec_id = uuid4()
        metadata_json = json.dumps(metadata or {})

        try:
            await pool.execute(
                """
                INSERT INTO task_executions (id, task_id, status, started_at, retry_count, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                """,
                exec_id, task_id, status, datetime.now(timezone.utc), retry_count, metadata_json,
            )
            return exec_id

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            logger.error("Failed to record task execution: %s", e)
            raise DatabaseOperationError("record task execution", e)

    async def complete_execution(
        self,
        exec_id: UUID,
        status: str,
        result_text: Optional[str] = None,
        duration_ms: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Complete a task execution record."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("complete task execution")

        try:
            await pool.execute(
                """
                UPDATE task_executions
                SET status = $2, completed_at = $3, duration_ms = $4,
                    result_text = $5, error = $6
                WHERE id = $1
                """,
                exec_id, status, datetime.now(timezone.utc),
                duration_ms, result_text, error,
            )
        except DatabaseUnavailableError:
            raise
        except Exception as e:
            logger.error("Failed to complete task execution: %s", e)
            raise DatabaseOperationError("complete task execution", e)

    async def get_executions(
        self,
        task_id: UUID,
        limit: int = 20,
    ) -> list[TaskExecution]:
        """Get execution history for a task."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("get task executions")

        try:
            rows = await pool.fetch(
                """
                SELECT id, task_id, status, started_at, completed_at,
                       duration_ms, result_text, error, retry_count, metadata
                FROM task_executions
                WHERE task_id = $1
                ORDER BY started_at DESC
                LIMIT $2
                """,
                task_id, limit,
            )

            return [self._row_to_execution(row) for row in rows]

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("get task executions", e)

    async def cleanup_old_executions(self, older_than_days: int = 30) -> int:
        """Delete execution records older than N days."""
        pool = get_db_pool()

        if not pool.is_initialized:
            raise DatabaseUnavailableError("cleanup old executions")

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)

            result = await pool.execute(
                """
                DELETE FROM task_executions
                WHERE completed_at < $1
                  AND status IN ('completed', 'failed', 'timeout')
                """,
                cutoff,
            )

            count = self._parse_row_count(result)
            if count > 0:
                logger.info("Cleaned up %d old task executions", count)
            return count

        except DatabaseUnavailableError:
            raise
        except Exception as e:
            raise DatabaseOperationError("cleanup old executions", e)

    def _parse_row_count(self, result: str) -> int:
        """Parse row count from PostgreSQL command result."""
        if not result:
            return 0
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    def _row_to_task(self, row) -> ScheduledTask:
        """Convert a database row to a ScheduledTask object."""
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        elif metadata is None:
            metadata = {}

        return ScheduledTask(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            task_type=row["task_type"],
            prompt=row["prompt"],
            agent_type=row["agent_type"] or "atlas",
            schedule_type=row["schedule_type"],
            cron_expression=row["cron_expression"],
            interval_seconds=row["interval_seconds"],
            run_at=row["run_at"],
            timezone=row["timezone"] or self._default_timezone,
            enabled=row["enabled"],
            max_retries=row["max_retries"] or 0,
            retry_delay_seconds=row["retry_delay_seconds"] or 60,
            timeout_seconds=row["timeout_seconds"] or 120,
            metadata=metadata,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_run_at=row["last_run_at"],
            next_run_at=row["next_run_at"],
        )

    def _row_to_execution(self, row) -> TaskExecution:
        """Convert a database row to a TaskExecution object."""
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        elif metadata is None:
            metadata = {}

        return TaskExecution(
            id=row["id"],
            task_id=row["task_id"],
            status=row["status"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            duration_ms=row["duration_ms"],
            result_text=row["result_text"],
            error=row["error"],
            retry_count=row["retry_count"] or 0,
            metadata=metadata,
        )


_scheduled_task_repo: Optional[ScheduledTaskRepository] = None


def get_scheduled_task_repo() -> ScheduledTaskRepository:
    """Get the global scheduled task repository."""
    global _scheduled_task_repo
    if _scheduled_task_repo is None:
        _scheduled_task_repo = ScheduledTaskRepository()
    return _scheduled_task_repo
