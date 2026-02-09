#!/usr/bin/env python3
"""
Test gpt-oss:20b 5-turn tool calling.

Scenario: Plan a day
1. Check calendar
2. Get weather
3. Set reminder for first meeting
4. Get current time
5. Set another reminder
"""

import asyncio
import json
import logging
import time

import httpx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("test")

OLLAMA_URL = "http://localhost:11434"
MODEL = "gpt-oss:20b"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time and date",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Timezone"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_calendar",
            "description": "Get calendar events",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD)"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Reminder message"},
                    "time": {"type": "string", "description": "When to remind"},
                },
                "required": ["message"],
            },
        },
    },
]

# Simulated tool responses
TOOL_RESPONSES = {
    "get_calendar": json.dumps({
        "events": [
            {"title": "Team standup", "time": "9:00 AM"},
            {"title": "Lunch with Sarah", "time": "12:30 PM"},
            {"title": "Project review", "time": "3:00 PM"},
        ]
    }),
    "get_weather": json.dumps({
        "location": "Dallas",
        "temperature": "72F",
        "condition": "Partly cloudy",
        "humidity": "45%",
    }),
    "get_time": json.dumps({
        "time": "8:15 AM",
        "date": "January 18, 2026",
        "day": "Sunday",
    }),
    "set_reminder": json.dumps({
        "success": True,
        "message": "Reminder set successfully",
    }),
}


async def call_ollama(messages: list) -> dict:
    """Call Ollama chat API."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
        "keep_alive": "30m",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        start = time.time()
        response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        latency = (time.time() - start) * 1000

    return {
        "message": response.json().get("message", {}),
        "latency_ms": latency,
    }


async def test_5_turn():
    """Test 5-turn tool calling conversation."""
    logger.info("=" * 60)
    logger.info("5-Turn Tool Calling Test")
    logger.info("=" * 60)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use tools to help the user plan their day. Call tools one at a time."
        },
        {
            "role": "user",
            "content": "Help me plan my day. Check my calendar, get the weather in Dallas, tell me the time, and set reminders for my meetings."
        },
    ]

    total_start = time.time()
    turn = 0
    tools_called = []

    while turn < 10:  # Max 10 turns to prevent infinite loop
        turn += 1
        result = await call_ollama(messages)
        msg = result["message"]
        latency = result["latency_ms"]

        tool_calls = msg.get("tool_calls", [])
        content = msg.get("content", "")

        if tool_calls:
            # Process tool calls
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                tool_args = func.get("arguments", {})

                logger.info(
                    "Turn %d: %s(%s) [%.0fms]",
                    turn, tool_name, tool_args, latency
                )
                tools_called.append(tool_name)

                # Add assistant message
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tc],
                })

                # Add tool response
                tool_response = TOOL_RESPONSES.get(tool_name, '{"error": "Unknown tool"}')
                messages.append({
                    "role": "tool",
                    "content": tool_response,
                })
        else:
            # No more tool calls - final response
            logger.info("Turn %d: Final response [%.0fms]", turn, latency)
            logger.info("Response: %s", content[:300] if content else "(empty)")
            break

    total_time = (time.time() - total_start) * 1000

    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info("Total turns: %d", turn)
    logger.info("Tools called: %s", tools_called)
    logger.info("Total time: %.0fms", total_time)

    # Check if we got the expected tools
    expected = {"get_calendar", "get_weather", "get_time", "set_reminder"}
    called_set = set(tools_called)

    if expected.issubset(called_set):
        logger.info("Result: PASS - All expected tools called")
        return True
    else:
        missing = expected - called_set
        logger.info("Result: PARTIAL - Missing tools: %s", missing)
        return False


async def main():
    asyncio.run(test_5_turn())


if __name__ == "__main__":
    asyncio.run(test_5_turn())
