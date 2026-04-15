"""
Integration tests for default builtin task seeding.

Verifies that the scheduled_tasks table can be seeded with
nightly_memory_sync and cleanup_old_executions builtin tasks,
matching the definitions in TaskScheduler._DEFAULT_TASKS.

These tests exercise the repository layer directly to avoid
importing apscheduler (not in the test virtualenv).
"""

import json
from uuid import UUID

import pytest
import pytest_asyncio


# Mirror of TaskScheduler._DEFAULT_TASKS -- kept in sync with scheduler.py
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
        "name": "cleanup_old_executions",
        "description": "Purge old task execution records, presence events, and resolved proactive actions",
        "task_type": "builtin",
        "schedule_type": "cron",
        "cron_expression": "30 3 * * *",
        "timeout_seconds": 120,
        "metadata": {"builtin_handler": "cleanup_old_executions"},
    },
    {
        "name": "b2b_watchlist_alert_delivery",
        "description": "Evaluate due saved-view watchlist alerts and deliver scheduled email digests",
        "task_type": "builtin",
        "schedule_type": "interval",
        "interval_seconds": 3600,
        "timeout_seconds": 300,
        "enabled": False,
        "metadata": {
            "builtin_handler": "b2b_watchlist_alert_delivery",
            "notify_tags": "bell,email,b2b",
        },
    },
    {
        "name": "b2b_parser_upgrade_maintenance",
        "description": "Drain parser-version maintenance backlog for healthy structured scrape sources",
        "task_type": "builtin",
        "schedule_type": "interval",
        "interval_seconds": 21600,
        "timeout_seconds": 5400,
        "enabled": True,
        "metadata": {"builtin_handler": "b2b_parser_upgrade_maintenance"},
    },
]


async def _seed_defaults(repo):
    """Replicate _ensure_default_tasks logic without importing the scheduler."""
    created = []
    for task_def in _DEFAULT_TASKS:
        existing = await repo.get_by_name(task_def["name"])
        if existing is not None:
            continue
        task = await repo.create(
            name=task_def["name"],
            description=task_def.get("description"),
            task_type=task_def["task_type"],
            schedule_type=task_def["schedule_type"],
            cron_expression=task_def.get("cron_expression"),
            interval_seconds=task_def.get("interval_seconds"),
            timeout_seconds=task_def.get("timeout_seconds", 120),
            metadata=task_def.get("metadata"),
            enabled=task_def.get("enabled", True),
        )
        created.append(task)
    return created


@pytest.mark.integration
class TestDefaultTaskSeeding:
    """Test that default builtin tasks are seeded correctly."""

    @pytest_asyncio.fixture(autouse=True)
    async def _cleanup_seeded_tasks(self, db_pool):
        """Remove seeded tasks before and after each test."""
        names = (
            "nightly_memory_sync",
            "cleanup_old_executions",
            "b2b_watchlist_alert_delivery",
            "b2b_parser_upgrade_maintenance",
        )
        for name in names:
            await db_pool.execute(
                "DELETE FROM scheduled_tasks WHERE name = $1", name
            )
        yield
        for name in names:
            await db_pool.execute(
                "DELETE FROM scheduled_tasks WHERE name = $1", name
            )

    @pytest_asyncio.fixture
    async def task_repo(self, db_pool):
        """Get ScheduledTaskRepository with fresh state."""
        from atlas_brain.storage.repositories.scheduled_task import (
            get_scheduled_task_repo,
        )
        import atlas_brain.storage.repositories.scheduled_task as st_module

        st_module._scheduled_task_repo = None
        return get_scheduled_task_repo()

    @pytest.mark.asyncio
    async def test_seeds_nightly_memory_sync(self, db_pool, task_repo):
        """nightly_memory_sync row is created with correct attributes."""
        await _seed_defaults(task_repo)

        row = await db_pool.fetchrow(
            "SELECT * FROM scheduled_tasks WHERE name = $1",
            "nightly_memory_sync",
        )

        assert row is not None
        assert row["task_type"] == "builtin"
        assert row["schedule_type"] == "cron"
        assert row["cron_expression"] == "0 3 * * *"
        assert row["enabled"] is True

    @pytest.mark.asyncio
    async def test_seeds_b2b_parser_upgrade_maintenance(self, db_pool, task_repo):
        """b2b_parser_upgrade_maintenance row is created with correct attributes."""
        await _seed_defaults(task_repo)

        row = await db_pool.fetchrow(
            "SELECT * FROM scheduled_tasks WHERE name = $1",
            "b2b_parser_upgrade_maintenance",
        )

        assert row is not None
        assert row["task_type"] == "builtin"
        assert row["schedule_type"] == "interval"
        assert row["interval_seconds"] == 21600
        assert row["timeout_seconds"] == 5400
        assert row["enabled"] is True
        meta = row["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        assert meta["builtin_handler"] == "b2b_parser_upgrade_maintenance"

    @pytest.mark.asyncio
    async def test_seeds_cleanup_old_executions(self, db_pool, task_repo):
        """cleanup_old_executions row is created with correct attributes."""
        await _seed_defaults(task_repo)

        row = await db_pool.fetchrow(
            "SELECT * FROM scheduled_tasks WHERE name = $1",
            "cleanup_old_executions",
        )

        assert row is not None
        assert row["task_type"] == "builtin"
        assert row["schedule_type"] == "cron"
        assert row["cron_expression"] == "30 3 * * *"
        assert row["enabled"] is True
        assert row["timeout_seconds"] == 120
        meta = row["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        assert meta["builtin_handler"] == "cleanup_old_executions"

    @pytest.mark.asyncio
    async def test_idempotent_does_not_duplicate(self, db_pool, task_repo):
        """Calling seed twice does not create duplicates."""
        await _seed_defaults(task_repo)
        await _seed_defaults(task_repo)

        rows = await db_pool.fetch(
            "SELECT id FROM scheduled_tasks WHERE name IN ($1, $2)",
            "nightly_memory_sync",
            "cleanup_old_executions",
        )

        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_skips_existing_task(self, db_pool, task_repo):
        """If a task already exists (e.g. user-modified), it is not overwritten."""
        # Pre-create nightly_memory_sync with a custom cron
        await db_pool.execute(
            """INSERT INTO scheduled_tasks
               (name, task_type, schedule_type, cron_expression,
                timeout_seconds, metadata, enabled)
               VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)""",
            "nightly_memory_sync",
            "builtin",
            "cron",
            "0 4 * * *",
            600,
            '{"builtin_handler": "nightly_memory_sync"}',
            True,
        )

        await _seed_defaults(task_repo)

        row = await db_pool.fetchrow(
            "SELECT cron_expression, timeout_seconds FROM scheduled_tasks WHERE name = $1",
            "nightly_memory_sync",
        )

        # Original values preserved, not overwritten by defaults
        assert row["cron_expression"] == "0 4 * * *"
        assert row["timeout_seconds"] == 600

    @pytest.mark.asyncio
    async def test_both_tasks_have_valid_uuids(self, db_pool, task_repo):
        """Seeded tasks have proper UUID primary keys."""
        await _seed_defaults(task_repo)

        rows = await db_pool.fetch(
            "SELECT id FROM scheduled_tasks WHERE name IN ($1, $2, $3)",
            "nightly_memory_sync",
            "cleanup_old_executions",
            "b2b_watchlist_alert_delivery",
        )

        for row in rows:
            assert isinstance(row["id"], UUID)

    @pytest.mark.asyncio
    async def test_seeds_watchlist_alert_delivery_with_interval_defaults(self, db_pool, task_repo):
        """watchlist alert delivery is seeded as a disabled interval task with default metadata."""
        await _seed_defaults(task_repo)

        row = await db_pool.fetchrow(
            "SELECT * FROM scheduled_tasks WHERE name = $1",
            "b2b_watchlist_alert_delivery",
        )

        assert row is not None
        assert row["task_type"] == "builtin"
        assert row["schedule_type"] == "interval"
        assert row["interval_seconds"] == 3600
        assert row["enabled"] is False
        assert row["timeout_seconds"] == 300
        meta = row["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        assert meta["builtin_handler"] == "b2b_watchlist_alert_delivery"
        assert meta["notify_tags"] == "bell,email,b2b"
