"""
Tool execution service for LLM tool calling.

Handles the tool calling loop:
1. Call LLM with tool schemas
2. Parse tool calls from response
3. Execute tools via registry
4. Format tool results
5. Call LLM again for final response
"""

import json
import logging
import re
from typing import Any

from ..tools import tool_registry
from .protocols import Message

logger = logging.getLogger("atlas.services.tool_executor")

MAX_TOOL_ITERATIONS = 3

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
) -> dict[str, Any]:
    """
    Execute LLM query with tool calling loop.

    Args:
        llm: LLM service with chat_with_tools method
        messages: Initial message list
        max_tokens: Max tokens for LLM response
        temperature: LLM temperature
        target_tool: If provided, only include this specific tool (improves reliability)

    Returns:
        Dict with response, tools_executed, and tool_results
    """
    # Check if LLM supports tool calling
    if not hasattr(llm, "chat_with_tools"):
        logger.warning("LLM does not support tool calling, using regular chat")
        result = llm.chat(messages=messages, max_tokens=max_tokens, temperature=temperature)
        return {
            "response": result.get("response", ""),
            "tools_executed": [],
            "tool_results": {},
        }

    # Map common intent names to tool names
    tool_name_map = {
        "time": "get_time",
        "weather": "get_weather",
        "calendar": "get_calendar",
        "location": "get_location",
        "reminder": "set_reminder",
        "reminders": "list_reminders",
        "availability": "check_availability",
        "appointment": "book_appointment",
        "booking": "book_appointment",
        "schedule": "book_appointment",
    }

    # If a specific target tool was identified, use only that tool
    # This dramatically improves reliability (100% vs ~33% with multiple tools)
    if target_tool:
        # Normalize tool name (intent uses "time", registry uses "get_time")
        tool_name = tool_name_map.get(target_tool, target_tool)
        # Only add prefix if tool name doesn't have a known prefix
        known_prefixes = ("get_", "set_", "list_", "check_", "book_", "cancel_", "reschedule_")
        if not any(tool_name.startswith(p) for p in known_prefixes):
            tool_name = f"get_{target_tool}"

        tools = tool_registry.get_tool_schemas_filtered([tool_name])
        if tools:
            logger.info("Tool executor: using target tool '%s'", tool_name)
        else:
            logger.warning("Target tool '%s' not found, falling back to priority tools", tool_name)
            target_tool = None  # Fall through to priority tools

    if not target_tool:
        # Filter to commonly used tools to avoid overwhelming the model
        priority_tools = [
            "get_time", "get_weather", "get_calendar", "get_location",
            "set_reminder", "list_reminders", "send_notification",
            "check_availability", "book_appointment",
        ]
        all_tools = tool_registry.get_tool_schemas()
        tools = [t for t in all_tools if t.get("function", {}).get("name") in priority_tools]
        if not tools:
            tools = all_tools
        logger.info("Tool executor: %d tools available (filtered from %d)", len(tools), len(all_tools))

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
        logger.info("LLM response: len=%d, tool_calls=%d, response_preview='%s'",
                   len(last_response), len(tool_calls), last_response[:100] if last_response else "")

        # If no structured tool calls, try parsing text-based calls
        if not tool_calls and last_response:
            tool_calls = parse_text_tool_calls(last_response)
            if tool_calls:
                logger.info("Parsed %d text-based tool call(s)", len(tool_calls))

        if not tool_calls:
            logger.info("No tool calls from LLM, returning response directly")
            return {
                "response": last_response,
                "tools_executed": list(tool_results.keys()),
                "tool_results": tool_results,
            }

        logger.info("LLM requested %d tool call(s)", len(tool_calls))

        # Add assistant message with tool calls
        current_messages.append(Message(
            role="assistant",
            content=last_response or "",
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

            # Execute tool
            tool_result = await tool_registry.execute(tool_name, args)
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

    # Max iterations reached
    logger.warning("Max tool iterations (%d) reached", MAX_TOOL_ITERATIONS)
    return {
        "response": last_response,
        "tools_executed": list(tool_results.keys()),
        "tool_results": tool_results,
    }
