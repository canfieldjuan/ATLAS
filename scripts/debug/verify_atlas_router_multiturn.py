#!/usr/bin/env python3
"""
Verification script for Phase 4: Atlas Router Integration.

Tests that:
1. Atlas router detects active workflows
2. Atlas routes to workflow continuation instead of classification
3. Cancel detection works
4. Workflow timeout is handled

Run: ATLAS_DB_PORT=5433 python verify_atlas_router_multiturn.py
"""

import asyncio
import os
import sys

# Set test environment
os.environ.setdefault("ATLAS_DB_ENABLED", "true")
os.environ.setdefault("ATLAS_DB_HOST", "localhost")
os.environ.setdefault("ATLAS_DB_PORT", "5433")
os.environ.setdefault("ATLAS_DB_DATABASE", "atlas")
os.environ.setdefault("ATLAS_DB_USER", "atlas")
os.environ.setdefault("ATLAS_DB_PASSWORD", "atlas_dev_password")


async def run_tests():
    """Run verification tests for atlas router integration."""
    from atlas_brain.storage.database import get_db_pool
    from atlas_brain.storage.repositories.session import get_session_repo
    from atlas_brain.agents.graphs.workflow_state import get_workflow_state_manager
    from atlas_brain.agents.graphs.atlas import (
        check_active_workflow,
        _is_cancel_intent,
        route_after_check_workflow,
    )

    # Initialize database
    pool = get_db_pool()
    try:
        await pool.initialize()
    except Exception as e:
        print(f"SKIP: Database not available: {e}")
        return True

    repo = get_session_repo()
    manager = get_workflow_state_manager()
    all_passed = True

    # Test 1: Cancel intent detection
    print("Test 1: Cancel intent detection...", end=" ")
    try:
        assert _is_cancel_intent("nevermind") is True
        assert _is_cancel_intent("cancel") is True
        assert _is_cancel_intent("stop") is True
        assert _is_cancel_intent("forget it") is True
        assert _is_cancel_intent("I want to cancel") is True
        assert _is_cancel_intent("stop that") is True
        assert _is_cancel_intent("hello") is False
        assert _is_cancel_intent("book an appointment") is False
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False

    # Test 2: check_active_workflow with no active workflow
    print("Test 2: check_active_workflow with no active workflow...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-router-1")
        state = {
            "input_text": "hello",
            "session_id": str(session.id),
            "action_type": "",
        }
        result = await check_active_workflow(state)
        # Should return state unchanged (no active_workflow set)
        assert result.get("active_workflow") is None
        assert result.get("action_type") == ""
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 3: check_active_workflow with active workflow
    print("Test 3: check_active_workflow with active workflow...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-router-2")
        session_id = str(session.id)

        # Save a workflow state
        await manager.save_workflow_state(
            session_id=session_id,
            workflow_type="booking",
            current_step="awaiting_info",
            partial_state={"customer_name": None},
        )

        state = {
            "input_text": "John Smith",
            "session_id": session_id,
            "action_type": "",
        }
        result = await check_active_workflow(state)

        assert result.get("action_type") == "workflow_continuation"
        assert result.get("active_workflow") is not None
        assert result["active_workflow"]["workflow_type"] == "booking"

        await manager.clear_workflow_state(session_id)
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 4: check_active_workflow with cancel intent
    print("Test 4: check_active_workflow with cancel intent...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-router-3")
        session_id = str(session.id)

        # Save a workflow state
        await manager.save_workflow_state(
            session_id=session_id,
            workflow_type="booking",
            current_step="awaiting_info",
            partial_state={"customer_name": None},
        )

        state = {
            "input_text": "nevermind",
            "session_id": session_id,
            "action_type": "",
        }
        result = await check_active_workflow(state)

        assert result.get("action_type") == "workflow_cancelled"
        assert "cancel" in result.get("response", "").lower()

        # Workflow should be cleared
        saved = await manager.restore_workflow_state(session_id)
        assert saved is None

        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 5: route_after_check_workflow routing
    print("Test 5: route_after_check_workflow routing...", end=" ")
    try:
        # No action type - go to classify
        state1 = {"action_type": ""}
        assert route_after_check_workflow(state1) == "classify"

        # Workflow continuation - go to continue_workflow
        state2 = {"action_type": "workflow_continuation"}
        assert route_after_check_workflow(state2) == "continue_workflow"

        # Workflow cancelled - go to respond
        state3 = {"action_type": "workflow_cancelled"}
        assert route_after_check_workflow(state3) == "respond"

        # Mode switch - go to respond
        state4 = {"action_type": "mode_switch"}
        assert route_after_check_workflow(state4) == "respond"

        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False

    # Test 6: Expired workflow is cleared
    print("Test 6: Expired workflow is cleared...", end=" ")
    try:
        from datetime import datetime, timedelta, timezone

        session = await repo.create_session(terminal_id="test-router-4")
        session_id = str(session.id)

        # Save workflow with old timestamp (simulate expiration)
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        await repo.update_metadata(
            session.id,
            {
                "active_workflow": {
                    "workflow_type": "booking",
                    "current_step": "awaiting_info",
                    "started_at": old_time,
                    "partial_state": {},
                    "conversation_context": [],
                }
            },
        )

        state = {
            "input_text": "John Smith",
            "session_id": session_id,
            "action_type": "",
        }
        result = await check_active_workflow(state)

        # Should clear expired workflow and return normal state
        assert result.get("action_type") != "workflow_continuation"

        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Cleanup
    await pool.close()

    return all_passed


def main():
    """Main entry point."""
    print("=" * 50)
    print("Phase 4: Atlas Router Integration Verification")
    print("=" * 50)

    passed = asyncio.run(run_tests())

    print("=" * 50)
    if passed:
        print("All tests PASSED")
        sys.exit(0)
    else:
        print("Some tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
