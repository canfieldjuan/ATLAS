from __future__ import annotations

from typing import Any
from uuid import UUID


class ScheduledTaskRepository:
    async def update_execution_metadata(
        self,
        exec_id: UUID,
        metadata: dict[str, Any],
    ) -> None:
        return None


_scheduled_task_repo: ScheduledTaskRepository | None = None


def get_scheduled_task_repo() -> ScheduledTaskRepository:
    global _scheduled_task_repo
    if _scheduled_task_repo is None:
        _scheduled_task_repo = ScheduledTaskRepository()
    return _scheduled_task_repo
