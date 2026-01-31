"""Test presence workflow with mock tools."""

import asyncio
import os
import sys

# Use mock tools for testing
os.environ["USE_REAL_TOOLS"] = "false"
sys.path.insert(0, "/home/juan-canfield/Desktop/Atlas-LangGraph-Agents-ToolUse")


async def main():
    from atlas_brain.agents.graphs.presence import run_presence_workflow

    print("=" * 70)
    print("PRESENCE WORKFLOW TEST (Mock Tools)")
    print("=" * 70)

    test_cases = [
        # Lights control
        ("turn on the lights", "lights_control"),
        ("turn off the lights", "lights_control"),
        ("lights on", "lights_control"),
        ("toggle the lights", "lights_control"),
        ("dim the lights", "lights_control"),
        ("brighten the lights", "lights_control"),

        # Media control
        ("turn on the tv", "media_control"),
        ("turn off the television", "media_control"),
        ("pause", "media_control"),
        ("play", "media_control"),
        ("stop the video", "media_control"),

        # Scene control
        ("make it cozy", "scene_set"),
        ("movie mode", "scene_set"),
        ("set it to bright", "scene_set"),
        ("dim mode", "scene_set"),
        ("I am going to watch a movie", "scene_set"),

        # Location
        ("where am i", "where_am_i"),
        ("what room am i in", "where_am_i"),
        ("my current room", "where_am_i"),
    ]

    passed = 0
    failed = 0

    for text, expected_intent in test_cases:
        result = await run_presence_workflow(text)
        actual_intent = result.get("intent")
        status = "PASS" if actual_intent == expected_intent else "FAIL"

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"\n[{status}] '{text}'")
        print(f"       Expected: {expected_intent}")
        print(f"       Got:      {actual_intent}")
        print(f"       Response: {result.get('response', '')[:70]}...")
        print(f"       Time: {result.get('total_ms', 0):.1f}ms")

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)}")
    print("=" * 70)

    # Detailed response tests
    print("\n\nDetailed Response Tests:")
    print("-" * 50)

    # Test lights
    result = await run_presence_workflow("turn on the lights")
    print(f"\nLights On:")
    print(f"  Response: {result.get('response')}")
    print(f"  Room: {result.get('room_name')}")
    print(f"  Devices: {result.get('devices_controlled')}")

    # Test scene
    result = await run_presence_workflow("movie mode")
    print(f"\nMovie Mode:")
    print(f"  Response: {result.get('response')}")
    print(f"  Executed: {result.get('action_executed')}")

    # Test location
    result = await run_presence_workflow("where am i")
    print(f"\nWhere Am I:")
    print(f"  Response: {result.get('response')}")
    print(f"  Room: {result.get('room_name')}")
    print(f"  Confidence: {result.get('presence_confidence')}")


if __name__ == "__main__":
    asyncio.run(main())
