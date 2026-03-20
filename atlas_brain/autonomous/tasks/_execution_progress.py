"""Shared helpers for reporting live autonomous task execution progress."""

from typing import Any
from uuid import UUID

from ...storage.models import ScheduledTask


def _task_execution_id(task: ScheduledTask) -> UUID | None:
    """Return the current execution id injected by the scheduler, if present."""
    raw = (task.metadata or {}).get("_execution_id")
    if not raw:
        return None
    try:
        return UUID(str(raw))
    except (TypeError, ValueError, AttributeError):
        return None


async def _update_execution_progress(
    task: ScheduledTask,
    *,
    stage: str,
    progress_current: int | None = None,
    progress_total: int | None = None,
    progress_message: str | None = None,
    **counters: Any,
) -> None:
    """Merge progress metadata into the current task execution row."""
    exec_id = _task_execution_id(task)
    if exec_id is None:
        return
    metadata: dict[str, Any] = {"stage": stage}
    if progress_current is not None:
        metadata["progress_current"] = int(progress_current)
    if progress_total is not None:
        metadata["progress_total"] = int(progress_total)
    if progress_message:
        metadata["progress_message"] = str(progress_message)
    for key, value in counters.items():
        if value is not None:
            metadata[key] = value
    try:
        from ...storage.repositories.scheduled_task import get_scheduled_task_repo
        await get_scheduled_task_repo().update_execution_metadata(exec_id, metadata)
    except Exception:
        return
