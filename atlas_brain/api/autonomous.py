"""
REST API for autonomous task scheduler.

CRUD operations for scheduled tasks plus manual trigger and status endpoints.
"""

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("atlas.api.autonomous")

router = APIRouter(prefix="/autonomous", tags=["autonomous"])


# ── Request / Response models ─────────────────────────────────────────

class TaskCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    task_type: str = Field(..., pattern="^(agent_prompt|builtin|hook)$")
    prompt: Optional[str] = None
    agent_type: str = Field(default="atlas", max_length=32)
    schedule_type: str = Field(..., pattern="^(cron|interval|once)$")
    cron_expression: Optional[str] = Field(default=None, max_length=128)
    interval_seconds: Optional[int] = Field(default=None, ge=60, le=86400)
    run_at: Optional[datetime] = None
    timezone: Optional[str] = Field(default=None, max_length=64, description="Timezone (defaults to ATLAS_AUTONOMOUS_DEFAULT_TIMEZONE)")
    enabled: bool = True
    max_retries: int = Field(default=0, ge=0, le=10)
    retry_delay_seconds: int = Field(default=60, ge=10, le=3600)
    timeout_seconds: int = Field(default=120, ge=10, le=600)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None
    task_type: Optional[str] = Field(default=None, pattern="^(agent_prompt|builtin|hook)$")
    prompt: Optional[str] = None
    agent_type: Optional[str] = Field(default=None, max_length=32)
    schedule_type: Optional[str] = Field(default=None, pattern="^(cron|interval|once)$")
    cron_expression: Optional[str] = Field(default=None, max_length=128)
    interval_seconds: Optional[int] = Field(default=None, ge=60, le=86400)
    run_at: Optional[datetime] = None
    timezone: Optional[str] = Field(default=None, max_length=64)
    enabled: Optional[bool] = None
    max_retries: Optional[int] = Field(default=None, ge=0, le=10)
    retry_delay_seconds: Optional[int] = Field(default=None, ge=10, le=3600)
    timeout_seconds: Optional[int] = Field(default=None, ge=10, le=600)
    metadata: Optional[dict[str, Any]] = None


# ── Status endpoint (MUST be before /{task_id}) ──────────────────────

@router.get("/status/summary")
async def get_status_summary():
    """Get scheduler status summary."""
    from ..autonomous.scheduler import get_task_scheduler
    from ..autonomous.hooks import get_hook_manager

    scheduler = get_task_scheduler()
    hook_manager = get_hook_manager()

    return {
        "running": scheduler.is_running,
        "scheduled_count": scheduler.scheduled_count,
        "hook_count": hook_manager.hook_count,
    }


# ── CRUD endpoints ───────────────────────────────────────────────────

@router.get("/")
async def list_tasks(include_disabled: bool = False):
    """List all scheduled tasks."""
    from ..storage.repositories.scheduled_task import get_scheduled_task_repo

    repo = get_scheduled_task_repo()
    tasks = await repo.list_all(include_disabled=include_disabled)
    return {"tasks": [t.to_dict() for t in tasks]}


@router.post("/", status_code=201)
async def create_task(req: TaskCreateRequest):
    """Create a new scheduled task."""
    from ..storage.repositories.scheduled_task import get_scheduled_task_repo
    from ..autonomous.scheduler import get_task_scheduler

    # Validate schedule config
    if req.schedule_type == "cron" and not req.cron_expression:
        raise HTTPException(400, "cron_expression required for cron schedule")
    if req.schedule_type == "interval" and not req.interval_seconds:
        raise HTTPException(400, "interval_seconds required for interval schedule")
    if req.schedule_type == "once" and not req.run_at:
        raise HTTPException(400, "run_at required for once schedule")

    # Validate task_type config
    if req.task_type in ("agent_prompt", "hook") and not req.prompt:
        raise HTTPException(400, "prompt required for agent_prompt/hook tasks")
    if req.task_type == "builtin" and not req.metadata.get("builtin_handler"):
        raise HTTPException(400, "metadata.builtin_handler required for builtin tasks")

    repo = get_scheduled_task_repo()

    # Check for duplicate name
    existing = await repo.get_by_name(req.name)
    if existing:
        raise HTTPException(409, f"Task with name '{req.name}' already exists")

    task = await repo.create(
        name=req.name,
        description=req.description,
        task_type=req.task_type,
        prompt=req.prompt,
        agent_type=req.agent_type,
        schedule_type=req.schedule_type,
        cron_expression=req.cron_expression,
        interval_seconds=req.interval_seconds,
        run_at=req.run_at,
        timezone_str=req.timezone,
        enabled=req.enabled,
        max_retries=req.max_retries,
        retry_delay_seconds=req.retry_delay_seconds,
        timeout_seconds=req.timeout_seconds,
        metadata=req.metadata,
    )

    # Register with scheduler if enabled
    scheduler = get_task_scheduler()
    await scheduler.register_and_schedule(task)

    # Reload hooks if this is a hook task
    if task.task_type == "hook":
        from ..autonomous.hooks import get_hook_manager
        await get_hook_manager().load_hooks_from_db()

    return task.to_dict()


@router.get("/{task_id}")
async def get_task(task_id: UUID):
    """Get a scheduled task by ID."""
    from ..storage.repositories.scheduled_task import get_scheduled_task_repo

    repo = get_scheduled_task_repo()
    task = await repo.get_by_id(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task.to_dict()


@router.put("/{task_id}")
async def update_task(task_id: UUID, req: TaskUpdateRequest):
    """Update a scheduled task."""
    from ..storage.repositories.scheduled_task import get_scheduled_task_repo
    from ..autonomous.scheduler import get_task_scheduler

    repo = get_scheduled_task_repo()

    existing = await repo.get_by_id(task_id)
    if not existing:
        raise HTTPException(404, "Task not found")

    update_fields = req.model_dump(exclude_unset=True)
    if not update_fields:
        return existing.to_dict()

    task = await repo.update(task_id, **update_fields)

    # Re-register with scheduler
    scheduler = get_task_scheduler()
    scheduler.unregister_task(str(task_id))
    await scheduler.register_and_schedule(task)

    # Reload hooks if this is a hook task
    if task.task_type == "hook":
        from ..autonomous.hooks import get_hook_manager
        await get_hook_manager().load_hooks_from_db()

    return task.to_dict()


@router.delete("/{task_id}")
async def delete_task(task_id: UUID):
    """Delete a scheduled task."""
    from ..storage.repositories.scheduled_task import get_scheduled_task_repo
    from ..autonomous.scheduler import get_task_scheduler

    repo = get_scheduled_task_repo()

    existing = await repo.get_by_id(task_id)
    if not existing:
        raise HTTPException(404, "Task not found")

    # Unschedule first
    scheduler = get_task_scheduler()
    scheduler.unregister_task(str(task_id))

    await repo.delete(task_id)

    # Reload hooks if this was a hook task
    if existing.task_type == "hook":
        from ..autonomous.hooks import get_hook_manager
        await get_hook_manager().load_hooks_from_db()

    return {"deleted": True, "task_id": str(task_id)}


# ── Action endpoints ─────────────────────────────────────────────────

@router.post("/{task_id}/run", status_code=202)
async def run_task_now(task_id: UUID):
    """Execute a task immediately (manual trigger). Returns 202 with execution ID.

    Poll GET /{task_id}/executions to check completion status.
    """
    from ..storage.repositories.scheduled_task import get_scheduled_task_repo
    from ..autonomous.scheduler import get_task_scheduler

    repo = get_scheduled_task_repo()
    task = await repo.get_by_id(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    scheduler = get_task_scheduler()
    result = await scheduler.run_now(task)
    return result


@router.post("/{task_id}/enable")
async def enable_task(task_id: UUID):
    """Enable a task and schedule it."""
    from ..storage.repositories.scheduled_task import get_scheduled_task_repo
    from ..autonomous.scheduler import get_task_scheduler

    repo = get_scheduled_task_repo()
    task = await repo.update(task_id, enabled=True)
    if not task:
        raise HTTPException(404, "Task not found")

    scheduler = get_task_scheduler()
    await scheduler.register_and_schedule(task)

    # Reload hooks if this is a hook task
    if task.task_type == "hook":
        from ..autonomous.hooks import get_hook_manager
        await get_hook_manager().load_hooks_from_db()

    return task.to_dict()


@router.post("/{task_id}/disable")
async def disable_task(task_id: UUID):
    """Disable a task and unschedule it."""
    from ..storage.repositories.scheduled_task import get_scheduled_task_repo
    from ..autonomous.scheduler import get_task_scheduler

    repo = get_scheduled_task_repo()
    task = await repo.update(task_id, enabled=False)
    if not task:
        raise HTTPException(404, "Task not found")

    scheduler = get_task_scheduler()
    scheduler.unregister_task(str(task_id))

    # Reload hooks if this is a hook task
    if task.task_type == "hook":
        from ..autonomous.hooks import get_hook_manager
        await get_hook_manager().load_hooks_from_db()

    return task.to_dict()


@router.get("/{task_id}/executions")
async def get_executions(task_id: UUID, limit: int = 20):
    """Get execution history for a task."""
    from ..storage.repositories.scheduled_task import get_scheduled_task_repo

    repo = get_scheduled_task_repo()

    task = await repo.get_by_id(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    executions = await repo.get_executions(task_id, limit=min(limit, 100))
    return {
        "task_id": str(task_id),
        "task_name": task.name,
        "executions": [e.to_dict() for e in executions],
    }
