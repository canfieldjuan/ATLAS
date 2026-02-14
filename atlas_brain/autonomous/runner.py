"""
Headless agent runner for autonomous task execution.

Wraps process_with_fallback() to run agent prompts and
builtin tasks without a voice/WebSocket frontend.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable, Optional

from ..agents.interface import process_with_fallback
from ..agents.protocols import AgentResult
from ..storage.models import ScheduledTask
from .config import autonomous_config

logger = logging.getLogger("atlas.autonomous.runner")


class HeadlessRunner:
    """
    Runs scheduled tasks headlessly using the agent pipeline.

    Dispatches on task_type:
    - agent_prompt / hook: calls process_with_fallback()
    - builtin: calls a registered Python handler
    """

    def __init__(self):
        self._builtin_handlers: dict[str, Callable[..., Awaitable[Any]]] = {}
        self._register_default_builtins()

    def _register_default_builtins(self) -> None:
        """Register default builtin task handlers."""
        self._builtin_handlers["nightly_memory_sync"] = self._run_nightly_sync
        self._builtin_handlers["cleanup_old_executions"] = self._run_cleanup_executions
        # Phase 2 builtin tasks
        from .tasks import register_builtin_tasks
        register_builtin_tasks(self)

    def register_builtin(
        self,
        name: str,
        handler: Callable[..., Awaitable[Any]],
    ) -> None:
        """Register a builtin task handler."""
        self._builtin_handlers[name] = handler
        logger.info("Registered builtin handler: %s", name)

    async def run(self, task: ScheduledTask) -> AgentResult:
        """
        Execute a task and return the result.

        Args:
            task: The scheduled task to execute

        Returns:
            AgentResult with success/failure and response text
        """
        if task.task_type in ("agent_prompt", "hook"):
            return await self._run_agent_prompt(task)
        elif task.task_type == "builtin":
            return await self._run_builtin(task)
        else:
            return AgentResult(
                success=False,
                error=f"Unknown task type: {task.task_type}",
                response_text=f"Cannot execute unknown task type: {task.task_type}",
            )

    async def _run_agent_prompt(self, task: ScheduledTask) -> AgentResult:
        """Run a task via the agent pipeline."""
        prompt = task.prompt
        if not prompt:
            return AgentResult(
                success=False,
                error="No prompt configured for task",
                response_text="Task has no prompt to execute.",
            )

        session_id = self._build_session_id(task)

        logger.info(
            "Running agent task '%s' (type=%s, agent=%s, session=%s)",
            task.name, task.task_type, task.agent_type, session_id,
        )

        return await process_with_fallback(
            input_text=prompt,
            agent_type=task.agent_type or autonomous_config.default_agent_type,
            session_id=session_id,
            input_type="autonomous",
            runtime_context={
                "source": "autonomous_scheduler",
                "task_name": task.name,
                "task_type": task.task_type,
            },
        )

    async def _run_builtin(self, task: ScheduledTask) -> AgentResult:
        """Run a builtin task handler."""
        handler_name = (task.metadata or {}).get("builtin_handler")
        if not handler_name:
            return AgentResult(
                success=False,
                error="No builtin_handler specified in task metadata",
                response_text="Builtin task missing handler name.",
            )

        handler = self._builtin_handlers.get(handler_name)
        if not handler:
            available = ", ".join(sorted(self._builtin_handlers.keys()))
            return AgentResult(
                success=False,
                error=f"Unknown builtin handler: {handler_name}. Available: {available}",
                response_text=f"Unknown builtin handler: {handler_name}",
            )

        logger.info("Running builtin task '%s' (handler=%s)", task.name, handler_name)

        try:
            result = await handler(task)

            # Post-processing: LLM synthesis via skill
            skill_name = (task.metadata or {}).get("synthesis_skill")
            if skill_name and isinstance(result, dict):
                synthesized = await self._synthesize_with_skill(
                    result, skill_name, task.name,
                )
                if synthesized:
                    return AgentResult(
                        success=True,
                        response_text=synthesized,
                        metadata={"raw_result": result, "synthesis_skill": skill_name},
                    )

            return AgentResult(
                success=True,
                response_text=str(result) if result else "Builtin task completed.",
            )
        except Exception as e:
            logger.error("Builtin handler '%s' failed: %s", handler_name, e)
            return AgentResult(
                success=False,
                error=str(e),
                response_text=f"Builtin task failed: {e}",
            )

    async def _synthesize_with_skill(
        self,
        raw_result: dict,
        skill_name: str,
        task_name: str,
    ) -> Optional[str]:
        """
        Synthesize a raw builtin result into natural language using a skill.

        Returns the synthesized text, or None if synthesis is unavailable/fails.
        """
        if not autonomous_config.synthesis_enabled:
            return None

        # Lazy imports (same pattern as email.py:763-766)
        from ..skills import get_skill_registry
        from ..services import llm_registry
        from ..services.protocols import Message
        from ..orchestration import cuda_lock

        skill = get_skill_registry().get(skill_name)
        if skill is None:
            logger.warning(
                "Synthesis skill '%s' not found for task '%s'",
                skill_name, task_name,
            )
            return None

        llm = llm_registry.get_active()
        if llm is None:
            logger.warning("No active LLM for synthesis of task '%s'", task_name)
            return None

        messages = [
            Message(role="system", content=skill.content),
            Message(
                role="user",
                content=json.dumps(raw_result, indent=2, default=str),
            ),
        ]

        try:
            async with cuda_lock:
                result = llm.chat(
                    messages=messages,
                    max_tokens=autonomous_config.synthesis_max_tokens,
                    temperature=autonomous_config.synthesis_temperature,
                )

            text = result.get("response", "").strip()
            if not text:
                logger.warning("LLM returned empty synthesis for task '%s'", task_name)
                return None

            # Strip <think> tags (Qwen3 models)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            logger.info(
                "Synthesized task '%s' with skill '%s' (%d chars)",
                task_name, skill_name, len(text),
            )
            return text

        except Exception:
            logger.exception("Synthesis failed for task '%s'", task_name)
            return None

    def _build_session_id(self, task: ScheduledTask) -> str:
        """Build a deterministic session ID for a task."""
        prefix = autonomous_config.default_session_prefix
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", task.name)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"{prefix}_{safe_name}_{date_str}"

    async def _run_nightly_sync(self, task: ScheduledTask) -> dict:
        """Builtin: Run nightly memory sync."""
        from ..jobs.nightly_memory_sync import run_nightly_sync

        purge_days = (task.metadata or {}).get("purge_days")
        return await run_nightly_sync(purge_days=purge_days)

    async def _run_cleanup_executions(self, task: ScheduledTask) -> dict:
        """Builtin: Clean up old task execution records and phase 3 tables."""
        from ..storage.repositories.scheduled_task import get_scheduled_task_repo
        from ..storage.database import get_db_pool

        retention_days = (task.metadata or {}).get(
            "retention_days",
            autonomous_config.task_history_retention_days,
        )
        repo = get_scheduled_task_repo()
        count = await repo.cleanup_old_executions(older_than_days=retention_days)

        result = {"cleaned_up": count, "retention_days": retention_days}

        # Clean up presence_events and resolved proactive_actions
        pool = get_db_pool()
        if pool.is_initialized:
            try:
                presence_result = await pool.execute(
                    """
                    DELETE FROM presence_events
                    WHERE created_at < CURRENT_TIMESTAMP - make_interval(days => $1)
                    """,
                    retention_days,
                )
                result["presence_events_cleaned"] = presence_result
            except Exception as e:
                logger.warning("Presence events cleanup failed: %s", e)

            try:
                actions_result = await pool.execute(
                    """
                    DELETE FROM proactive_actions
                    WHERE status IN ('done', 'dismissed')
                      AND resolved_at < CURRENT_TIMESTAMP - make_interval(days => $1)
                    """,
                    retention_days,
                )
                result["proactive_actions_cleaned"] = actions_result
            except Exception as e:
                logger.warning("Proactive actions cleanup failed: %s", e)

        return result


_headless_runner: Optional[HeadlessRunner] = None


def get_headless_runner() -> HeadlessRunner:
    """Get the global headless runner."""
    global _headless_runner
    if _headless_runner is None:
        _headless_runner = HeadlessRunner()
    return _headless_runner
