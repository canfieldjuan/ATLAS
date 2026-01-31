"""Test security workflow with mock tools."""

import asyncio
import os
import sys

# Use mock tools for testing
os.environ["USE_REAL_TOOLS"] = "false"
sys.path.insert(0, "/home/juan-canfield/Desktop/Atlas-LangGraph-Agents-ToolUse")


async def main():
    from atlas_brain.agents.graphs.security import run_security_workflow

    print("=" * 70)
    print("SECURITY WORKFLOW TEST (Mock Tools)")
    print("=" * 70)

    test_cases = [
        # Camera tests
        ("list cameras", "camera_list"),
        ("show all cameras", "camera_list"),
        ("status of the front door camera", "camera_status"),
        ("is the backyard camera online", "camera_status"),
        ("start recording on the front camera", "camera_record_start"),
        ("stop recording on the garage camera", "camera_record_stop"),
        ("pan the front door camera left", "camera_ptz"),

        # Detection tests
        ("any current detections", "detection_current"),
        ("what are the cameras seeing", "detection_current"),
        ("show me recent detections", "detection_query"),
        ("is there anyone at the front door", "detection_person_location"),
        ("check if someone is at the garage", "detection_person_location"),
        ("any motion events", "detection_motion"),
        ("was there any motion in the last hour", "detection_motion"),

        # Zone tests
        ("list security zones", "zone_list"),
        ("show all zones", "zone_list"),
        ("status of the perimeter zone", "zone_status"),
        ("arm the perimeter zone", "zone_arm"),
        ("disarm the interior zone", "zone_disarm"),
    ]

    passed = 0
    failed = 0

    for text, expected_intent in test_cases:
        result = await run_security_workflow(text)
        actual_intent = result.get("intent")
        status = "PASS" if actual_intent == expected_intent else "FAIL"

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"\n[{status}] '{text}'")
        print(f"       Expected: {expected_intent}")
        print(f"       Got:      {actual_intent}")
        print(f"       Response: {result.get('response', '')[:80]}...")
        print(f"       Time: {result.get('total_ms', 0):.1f}ms")

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)}")
    print("=" * 70)

    # Additional detailed test
    print("\n\nDetailed Response Tests:")
    print("-" * 50)

    # Test camera list
    result = await run_security_workflow("list all cameras")
    print(f"\nCamera List:")
    print(f"  Response: {result.get('response')}")
    print(f"  Count: {result.get('camera_count')}")

    # Test zone list
    result = await run_security_workflow("what are the security zones")
    print(f"\nZone List:")
    print(f"  Response: {result.get('response')}")
    print(f"  Count: {result.get('zone_count')}")

    # Test arm zone
    result = await run_security_workflow("arm the perimeter")
    print(f"\nArm Zone:")
    print(f"  Response: {result.get('response')}")
    print(f"  Armed: {result.get('zone_armed')}")

    # Test person detection
    result = await run_security_workflow("is anyone at the front door")
    print(f"\nPerson Detection:")
    print(f"  Response: {result.get('response')}")
    print(f"  Person Found: {result.get('person_found')}")


if __name__ == "__main__":
    asyncio.run(main())
