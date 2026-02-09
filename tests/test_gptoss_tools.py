#!/usr/bin/env python3
"""
Test gpt-oss:20b tool calling capabilities.

Tests:
1. Single tool call (get_time)
2. Multi-turn tool call (calendar + reminder)
3. Tool call with parameters (weather for location)
"""

import asyncio
import json
import logging
import time

import httpx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("test.gptoss")

OLLAMA_URL = "http://localhost:11434"
MODEL = "gpt-oss:20b"

# Tool definitions in OpenAI format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time and date",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone (e.g., America/Chicago)",
                    }
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
                    "location": {
                        "type": "string",
                        "description": "City name (e.g., Dallas, TX)",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_calendar",
            "description": "Get calendar events for today",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date to check (YYYY-MM-DD)",
                    }
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
                    "message": {
                        "type": "string",
                        "description": "Reminder message",
                    },
                    "time": {
                        "type": "string",
                        "description": "When to remind (e.g., 'in 10 minutes', '3pm')",
                    },
                },
                "required": ["message"],
            },
        },
    },
]


async def call_ollama(messages: list, tools: list = None) -> dict:
    """Call Ollama chat API with optional tools."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",
    }
    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=120) as client:
        start = time.time()
        response = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
        )
        latency = (time.time() - start) * 1000

    result = response.json()
    return {
        "message": result.get("message", {}),
        "latency_ms": latency,
    }


async def test_single_tool():
    """Test single tool call."""
    logger.info("=" * 60)
    logger.info("Test 1: Single Tool Call (get_time)")
    logger.info("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": "What time is it?"},
    ]

    result = await call_ollama(messages, TOOLS)
    msg = result["message"]

    logger.info("Latency: %.0fms", result["latency_ms"])
    logger.info("Response role: %s", msg.get("role"))
    logger.info("Content: %s", msg.get("content", "")[:100])

    tool_calls = msg.get("tool_calls", [])
    if tool_calls:
        logger.info("Tool calls: %d", len(tool_calls))
        for tc in tool_calls:
            func = tc.get("function", {})
            logger.info("  - %s(%s)", func.get("name"), func.get("arguments"))
        return True
    else:
        logger.warning("No tool calls detected")
        return False


async def test_tool_with_params():
    """Test tool call with parameters."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Tool Call with Parameters (get_weather)")
    logger.info("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": "What's the weather in Dallas?"},
    ]

    result = await call_ollama(messages, TOOLS)
    msg = result["message"]

    logger.info("Latency: %.0fms", result["latency_ms"])

    tool_calls = msg.get("tool_calls", [])
    if tool_calls:
        for tc in tool_calls:
            func = tc.get("function", {})
            logger.info("Tool: %s", func.get("name"))
            logger.info("Args: %s", func.get("arguments"))

            # Check if location parameter was extracted
            args = func.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if "location" in args:
                logger.info("Location extracted: %s", args["location"])
                return True
    return False


async def test_multi_turn():
    """Test multi-turn tool calling (simulated)."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Multi-turn Tool Call")
    logger.info("=" * 60)

    # First turn - ask about calendar
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": "Check my calendar and remind me about my first meeting."},
    ]

    result = await call_ollama(messages, TOOLS)
    msg = result["message"]

    logger.info("Turn 1 Latency: %.0fms", result["latency_ms"])

    tool_calls = msg.get("tool_calls", [])
    if tool_calls:
        logger.info("Turn 1 Tool calls:")
        for tc in tool_calls:
            func = tc.get("function", {})
            logger.info("  - %s(%s)", func.get("name"), func.get("arguments"))

        # Simulate tool response and continue
        messages.append(msg)
        messages.append({
            "role": "tool",
            "content": json.dumps({
                "events": [
                    {"title": "Team standup", "time": "9:00 AM"},
                    {"title": "Project review", "time": "2:00 PM"},
                ]
            }),
        })

        # Second turn - should set reminder
        result2 = await call_ollama(messages, TOOLS)
        msg2 = result2["message"]

        logger.info("Turn 2 Latency: %.0fms", result2["latency_ms"])

        tool_calls2 = msg2.get("tool_calls", [])
        if tool_calls2:
            logger.info("Turn 2 Tool calls:")
            for tc in tool_calls2:
                func = tc.get("function", {})
                logger.info("  - %s(%s)", func.get("name"), func.get("arguments"))
            return True
        else:
            logger.info("Turn 2 Content: %s", msg2.get("content", "")[:200])

    return False


async def test_no_tool_needed():
    """Test that model doesn't call tools unnecessarily."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: No Tool Needed (conversation)")
    logger.info("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": "Tell me a joke."},
    ]

    result = await call_ollama(messages, TOOLS)
    msg = result["message"]

    logger.info("Latency: %.0fms", result["latency_ms"])

    tool_calls = msg.get("tool_calls", [])
    if not tool_calls:
        logger.info("Correctly did NOT call tools")
        logger.info("Response: %s", msg.get("content", "")[:200])
        return True
    else:
        logger.warning("Incorrectly called tools for conversation")
        return False


async def main():
    """Run all tests."""
    logger.info("GPT-OSS 20B Tool Calling Tests")
    logger.info("=" * 60)

    # Check if model is available
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            if not any(MODEL in m for m in models):
                logger.error("Model %s not found. Available: %s", MODEL, models)
                return
    except Exception as e:
        logger.error("Cannot connect to Ollama: %s", e)
        return

    results = {}

    # Run tests
    results["single_tool"] = await test_single_tool()
    results["tool_with_params"] = await test_tool_with_params()
    results["multi_turn"] = await test_multi_turn()
    results["no_tool_needed"] = await test_no_tool_needed()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info("  %s: %s", test, status)

    passed = sum(1 for v in results.values() if v)
    logger.info("\nTotal: %d/%d passed", passed, len(results))


if __name__ == "__main__":
    asyncio.run(main())
