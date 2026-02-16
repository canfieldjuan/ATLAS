"""
Tool execution service for LLM tool calling.

Handles the tool calling loop:
1. Call LLM with tool schemas
2. Parse tool calls from response
3. Execute tools via registry
4. Format tool results
5. Call LLM again for final response
"""

import asyncio
import json
import logging
import re
from typing import Any

from ..tools import tool_registry
from ..tools.base import ToolResult
from .protocols import Message

logger = logging.getLogger("atlas.services.tool_executor")

MAX_TOOL_ITERATIONS = 3

# Priority tools for LLM tool calling (reduces model confusion)
PRIORITY_TOOL_NAMES = [
    "get_time", "get_weather", "get_calendar", "get_location",
    "set_reminder", "list_reminders", "send_notification",
    "send_email", "check_availability", "book_appointment",
    "cancel_appointment", "reschedule_appointment",
]

# Pattern to match text-based tool calls
# Format 1: <function=tool_name>json_args</function>
TEXT_TOOL_PATTERN = re.compile(
    r"<function=(\w+)>(.*?)</function>",
    re.DOTALL
)

# Pattern to extract parameter tags
PARAM_PATTERN = re.compile(
    r"<parameter=(\w+)>\s*(.*?)\s*</parameter>",
    re.DOTALL
)


def _strip_tool_xml(text: str) -> str:
    """Remove tool call XML artifacts and LLM thinking tags from a response."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?tool_call>", "", text)
    text = TEXT_TOOL_PATTERN.sub("", text)
    return text.strip()


def parse_text_tool_calls(content: str) -> list[dict]:
    """Parse text-based tool calls from LLM response."""
    tool_calls = []
    for match in TEXT_TOOL_PATTERN.finditer(content):
        tool_name = match.group(1)
        inner_content = match.group(2).strip()
        args = {}

        # Try to parse as JSON first
        if inner_content and not inner_content.startswith("<"):
            try:
                args = json.loads(inner_content)
            except json.JSONDecodeError:
                pass

        # Try to parse parameter tags
        if not args:
            for param_match in PARAM_PATTERN.finditer(inner_content):
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()
                args[param_name] = param_value

        tool_calls.append({
            "function": {
                "name": tool_name,
                "arguments": args,
            }
        })
    return tool_calls


async def execute_with_tools(
    llm,
    messages: list[Message],
    max_tokens: int = 256,
    temperature: float = 0.7,
    target_tool: str | None = None,
    tools_override: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Execute LLM query with tool calling loop.

    Args:
        llm: LLM service with chat_with_tools method
        messages: Initial message list
        max_tokens: Max tokens for LLM response
        temperature: LLM temperature
        target_tool: If provided, only include this specific tool (improves reliability)
        tools_override: Pre-built tool schemas to use instead of registry lookup

    Returns:
        Dict with response, tools_executed, and tool_results
    """
    # Check if LLM supports tool calling
    if not hasattr(llm, "chat_with_tools"):
        logger.warning("LLM does not support tool calling, using regular chat")
        result = llm.chat(messages=messages, max_tokens=max_tokens, temperature=temperature)
        return {
            "response": _strip_tool_xml(result.get("response", "")),
            "tools_executed": [],
            "tool_results": {},
        }

    # If caller provided pre-built tool schemas, use them directly
    if tools_override is not None:
        tools = tools_override
        logger.info("Tool executor: using %d override tools", len(tools))
    elif target_tool:
        # Use registry to resolve alias to actual tool name
        tool_name = tool_registry.resolve_alias(target_tool)

        if not tool_name:
            # Fallback: try adding common prefixes
            known_prefixes = ("get_", "set_", "list_", "check_", "book_", "cancel_", "reschedule_")
            for prefix in known_prefixes:
                candidate = f"{prefix}{target_tool}"
                if tool_registry.get(candidate):
                    tool_name = candidate
                    break

        if not tool_name:
            # Last resort: use as-is
            tool_name = target_tool

        tools = tool_registry.get_tool_schemas_filtered([tool_name])
        if tools:
            logger.info("Tool executor: using target tool '%s'", tool_name)
        else:
            logger.warning("Target tool '%s' not found, falling back to priority tools", tool_name)
            target_tool = None  # Fall through to priority tools

    if tools_override is None and not target_tool:
        # Use registry to get priority tool schemas
        tools = tool_registry.get_tool_schemas_filtered(PRIORITY_TOOL_NAMES)
        if not tools:
            tools = tool_registry.get_tool_schemas()
        logger.info("Tool executor: %d tools available", len(tools))

    logger.info("Tool names: %s", [t.get("function", {}).get("name") for t in tools])

    current_messages = list(messages)
    tool_results = {}
    last_response = ""

    for iteration in range(MAX_TOOL_ITERATIONS):
        logger.info("Tool calling iteration %d", iteration + 1)

        result = llm.chat_with_tools(
            messages=current_messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        last_response = result.get("response", "")
        tool_calls = result.get("tool_calls", [])
        logger.info("LLM response: len=%d, tool_calls=%d, response='%s'",
                   len(last_response), len(tool_calls), last_response)

        # If no structured tool calls, try parsing text-based calls
        if not tool_calls and last_response:
            tool_calls = parse_text_tool_calls(last_response)
            if tool_calls:
                logger.info("Parsed %d text-based tool call(s)", len(tool_calls))

        if not tool_calls:
            # If LLM returned empty but we have tool results, use tool result as response
            final_response = _strip_tool_xml(last_response)
            if not final_response and tool_results:
                # Use the last tool's result as the response
                final_response = list(tool_results.values())[-1]
                logger.info("LLM returned empty, using tool result as response: %s", final_response)
            elif not final_response:
                logger.warning("LLM returned empty response with no tool results")
                final_response = "Sorry, I wasn't able to process that."
            else:
                logger.info("No tool calls from LLM, returning response directly")
            return {
                "response": final_response,
                "tools_executed": list(tool_results.keys()),
                "tool_results": tool_results,
            }

        logger.info("LLM requested %d tool call(s)", len(tool_calls))

        # Add assistant message with tool calls
        current_messages.append(Message(
            role="assistant",
            content=last_response or "",
            tool_calls=tool_calls,
        ))

        # Process each tool call
        for call in tool_calls:
            func = call.get("function", {})
            tool_name = func.get("name", "")
            args = func.get("arguments", {})

            # Parse arguments if string
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse tool args: %s", args)
                    args = {}

            logger.info("Executing tool: %s with args: %s", tool_name, args)

            # Execute tool with per-tool timeout
            from ..config import settings
            tool_timeout = settings.voice.tool_execution_timeout
            try:
                tool_result = await asyncio.wait_for(
                    tool_registry.execute(tool_name, args),
                    timeout=tool_timeout,
                )
            except asyncio.TimeoutError:
                logger.error("Tool %s timed out after %.1fs", tool_name, tool_timeout)
                tool_result = ToolResult(
                    success=False,
                    error="TOOL_TIMEOUT",
                    message=f"Tool {tool_name} timed out after {tool_timeout:.0f}s",
                )
            tool_results[tool_name] = tool_result.message

            # Add tool result to messages
            result_content = json.dumps({
                "name": tool_name,
                "success": tool_result.success,
                "message": tool_result.message,
                "data": tool_result.data,
            })
            current_messages.append(Message(
                role="tool",
                content=result_content,
            ))

            logger.info(
                "Tool %s result: success=%s, message=%s",
                tool_name,
                tool_result.success,
                tool_result.message[:50] if tool_result.message else "",
            )

    # Max iterations reached -- strip any leftover tool XML from response
    logger.warning("Max tool iterations (%d) reached", MAX_TOOL_ITERATIONS)
    final_response = _strip_tool_xml(last_response)
    if not final_response and tool_results:
        final_response = list(tool_results.values())[-1]
        logger.info("Using tool result as fallback response: %s", final_response)
    return {
        "response": final_response,
        "tools_executed": list(tool_results.keys()),
        "tool_results": tool_results,
    }
