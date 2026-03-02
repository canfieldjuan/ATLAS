"""
Voice-triggered digest tool.

Calls existing autonomous task handlers directly and synthesizes
the result via LLM + skill, returning a one-shot spoken summary.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .base import ToolParameter, ToolResult

logger = logging.getLogger("atlas.tools.digest")


@dataclass
class _SubTaskStub:
    """Minimal stand-in for ScheduledTask when calling handlers internally."""
    metadata: dict[str, Any] = field(default_factory=dict)


# handler_name -> skill path
_DIGEST_TYPES: dict[str, tuple[str, str]] = {
    "morning_briefing":  ("morning_briefing",    "digest/morning_briefing"),
    "security_summary":  ("security_summary",    "digest/security_summary"),
    "device_health":     ("device_health_check", "digest/device_health"),
    "email_digest":      ("gmail_digest",        "digest/email_triage"),
}


class RunDigestTool:
    """Run an autonomous digest on demand and return synthesized text."""

    @property
    def name(self) -> str:
        return "run_digest"

    @property
    def description(self) -> str:
        return (
            "Run a digest (morning briefing, security summary, device health, "
            "or email digest) and return a spoken summary."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="digest_type",
                param_type="string",
                description=(
                    "Type of digest: morning_briefing, security_summary, "
                    "device_health, or email_digest"
                ),
                required=False,
                default="morning_briefing",
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["digest", "briefing", "morning briefing", "daily summary"]

    @property
    def category(self) -> str:
        return "utility"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        digest_type = params.get("digest_type", "morning_briefing")
        entry = _DIGEST_TYPES.get(digest_type)
        if entry is None:
            valid = ", ".join(sorted(_DIGEST_TYPES.keys()))
            return ToolResult(
                success=False,
                error="INVALID_DIGEST_TYPE",
                message=f"Unknown digest type '{digest_type}'. Valid: {valid}",
            )

        handler_name, skill_name = entry

        # Import and call the handler
        try:
            from ..autonomous.runner import get_headless_runner
            runner = get_headless_runner()
            handler = runner._builtin_handlers.get(handler_name)
            if handler is None:
                return ToolResult(
                    success=False,
                    error="HANDLER_NOT_FOUND",
                    message=f"Handler '{handler_name}' not registered.",
                )

            raw_result = await handler(_SubTaskStub())
        except Exception as e:
            logger.exception("Digest handler '%s' failed", handler_name)
            return ToolResult(
                success=False,
                error="HANDLER_ERROR",
                message="Digest failed.",
            )

        if not isinstance(raw_result, dict):
            return ToolResult(
                success=True,
                data={"raw": str(raw_result)},
                message=str(raw_result) if raw_result else "Digest completed.",
            )

        # Synthesize via LLM + skill (same pattern as runner._synthesize_with_skill)
        try:
            from ..skills import get_skill_registry
            from ..services import llm_registry
            from ..services.protocols import Message
            from ..autonomous.config import autonomous_config

            skill = get_skill_registry().get(skill_name)
            if skill is None:
                logger.warning("Digest skill '%s' not found, returning raw summary", skill_name)
                summary = raw_result.get("summary", json.dumps(raw_result, default=str))
                return ToolResult(success=True, data=raw_result, message=summary)

            llm = llm_registry.get_active()
            if llm is None:
                summary = raw_result.get("summary", json.dumps(raw_result, default=str))
                return ToolResult(success=True, data=raw_result, message=summary)

            messages = [
                Message(role="system", content=skill.content),
                Message(role="user", content=json.dumps(raw_result, indent=2, default=str)),
            ]

            result = llm.chat(
                messages=messages,
                max_tokens=autonomous_config.synthesis_max_tokens,
                temperature=autonomous_config.synthesis_temperature,
            )

            text = result.get("response", "").strip()
            # Strip <think> tags (Qwen3 models)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            if not text:
                text = raw_result.get("summary", "Digest completed.")

            return ToolResult(success=True, data=raw_result, message=text)

        except Exception as e:
            logger.exception("Digest synthesis failed for '%s'", digest_type)
            summary = raw_result.get("summary", "Digest completed.")
            return ToolResult(success=True, data=raw_result, message=summary)


# Module-level instance
digest_tool = RunDigestTool()
