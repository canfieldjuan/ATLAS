"""
GPT-OSS LLM Service for tool calling.

Uses gpt-oss:20b via Ollama for multi-turn tool calling.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from ..config import settings

logger = logging.getLogger("atlas.pipecat.llm")

# Defaults from config, can be overridden at instantiation
DEFAULT_OLLAMA_URL = settings.llm.ollama_url
DEFAULT_TOOL_MODEL = "gpt-oss:20b"  # Specialized for tool calling
MAX_TOOL_ITERATIONS = 5

# Priority tools for LLM tool calling (reduces model confusion)
PRIORITY_TOOL_NAMES = [
    "get_time", "get_weather", "get_calendar", "get_location",
    "set_reminder", "list_reminders", "send_notification",
]


@dataclass
class ToolCallResult:
    """Result from tool calling LLM."""
    response: str
    tools_executed: list[str] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    total_latency_ms: float = 0
    turns: int = 0


class GptOssToolService:
    """
    GPT-OSS 20B service for multi-turn tool calling.

    Handles complex queries that need LLM reasoning with tools.
    """

    def __init__(
        self,
        model: str = DEFAULT_TOOL_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        max_iterations: int = MAX_TOOL_ITERATIONS,
    ):
        self._model = model
        self._ollama_url = ollama_url
        self._max_iterations = max_iterations
        self._tool_registry = None
        self._tools_schema = None

    async def _get_tool_registry(self):
        """Lazy load tool registry."""
        if self._tool_registry is None:
            from ..tools import tool_registry
            self._tool_registry = tool_registry
        return self._tool_registry

    async def _get_tools_schema(self) -> list[dict]:
        """Get tool schemas in OpenAI format."""
        if self._tools_schema is None:
            registry = await self._get_tool_registry()
            self._tools_schema = registry.get_tool_schemas_filtered(PRIORITY_TOOL_NAMES)
            logger.info("Loaded %d tool schemas", len(self._tools_schema))
        return self._tools_schema

    async def _call_ollama(self, messages: list, tools: list) -> dict:
        """Call Ollama chat API with tools."""
        payload = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "keep_alive": "30m",
        }

        async with httpx.AsyncClient(timeout=120) as client:
            start = time.time()
            response = await client.post(
                f"{self._ollama_url}/api/chat",
                json=payload,
            )
            latency = (time.time() - start) * 1000

        result = response.json()
        return {
            "message": result.get("message", {}),
            "latency_ms": latency,
        }

    async def process_with_tools(
        self,
        query: str,
        system_prompt: Optional[str] = None,
    ) -> ToolCallResult:
        """
        Process a query with multi-turn tool calling.

        Args:
            query: User query
            system_prompt: Optional system prompt

        Returns:
            ToolCallResult with response and tool execution info
        """
        tools = await self._get_tools_schema()
        registry = await self._get_tool_registry()

        if system_prompt is None:
            system_prompt = (
                "You are Atlas, a helpful voice assistant. "
                "Use the available tools to help the user. "
                "Be concise - your response will be spoken aloud."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        total_start = time.time()
        tools_executed = []
        tool_results = {}
        turns = 0
        final_response = ""

        for iteration in range(self._max_iterations):
            turns += 1
            result = await self._call_ollama(messages, tools)
            msg = result["message"]

            tool_calls = msg.get("tool_calls", [])
            content = msg.get("content", "")

            if not tool_calls:
                # No more tool calls - we have the final response
                final_response = content
                logger.info(
                    "LLM final response after %d turns: '%s'",
                    turns, content[:100]
                )
                break

            # Process tool calls
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                tool_args = func.get("arguments", {})

                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}

                logger.info(
                    "LLM tool call: %s(%s) [%.0fms]",
                    tool_name, tool_args, result["latency_ms"]
                )

                # Execute tool
                tool_result = await registry.execute(tool_name, tool_args)
                tools_executed.append(tool_name)
                tool_results[tool_name] = tool_result.message

                # Add to conversation
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tc],
                })

                messages.append({
                    "role": "tool",
                    "content": json.dumps({
                        "name": tool_name,
                        "success": tool_result.success,
                        "message": tool_result.message,
                        "data": tool_result.data,
                    }),
                })

        total_latency = (time.time() - total_start) * 1000

        return ToolCallResult(
            response=final_response,
            tools_executed=tools_executed,
            tool_results=tool_results,
            total_latency_ms=total_latency,
            turns=turns,
        )


# Module-level instance
_service: Optional[GptOssToolService] = None


def get_gptoss_service() -> GptOssToolService:
    """Get or create the GPT-OSS service."""
    global _service
    if _service is None:
        _service = GptOssToolService()
    return _service


async def process_complex_query(query: str) -> ToolCallResult:
    """Convenience function to process a complex query."""
    service = get_gptoss_service()
    return await service.process_with_tools(query)
