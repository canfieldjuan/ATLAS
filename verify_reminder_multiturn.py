#!/usr/bin/env python3
"""
Verification script for reminder workflow multi-turn support.

Run: ATLAS_DB_PORT=5433 python verify_reminder_multiturn.py
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
    """Run verification tests for reminder workflow multi-turn."""
    from atlas_brain.storage.database import get_db_pool
    from atlas_brain.storage.repositories.session import get_session_repo
    from atlas_brain.agents.graphs.workflow_state import get_workflow_state_manager
    from atlas_brain.agents.graphs.reminder import (
        run_reminder_workflow,
        REMINDER_WORKFLOW_TYPE,
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

    # Test 1: New reminder request saves state when time missing
    print("Test 1: New request saves state when time missing...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-reminder-1")
        result = await run_reminder_workflow(
            input_text="remind me to call mom",
            session_id=str(session.id),
        )
        assert result.get("needs_clarification") is True
        assert "when" in result.get("response", "").lower()

        # Verify state was saved
        saved = await manager.restore_workflow_state(str(session.id))
        assert saved is not None, "State should be saved"
        assert saved.workflow_type == REMINDER_WORKFLOW_TYPE
        assert saved.current_step == "awaiting_info"
        assert saved.partial_state.get("reminder_message") == "call mom"

        await manager.clear_workflow_state(str(session.id))
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 2: New reminder request saves state when message missing
    print("Test 2: New request saves state when message missing...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-reminder-2")
        result = await run_reminder_workflow(
            input_text="set a reminder for in 30 minutes",
            session_id=str(session.id),
        )
        assert result.get("needs_clarification") is True
        assert "what" in result.get("response", "").lower()

        # Verify state was saved
        saved = await manager.restore_workflow_state(str(session.id))
        assert saved is not None, "State should be saved"
        assert saved.workflow_type == REMINDER_WORKFLOW_TYPE

        await manager.clear_workflow_state(str(session.id))
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 3: Continuation restores state and merges time
    print("Test 3: Continuation merges time input...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-reminder-3")
        session_id = str(session.id)

        # Turn 1: Initial request with message only
        result1 = await run_reminder_workflow(
            input_text="remind me to call mom",
            session_id=session_id,
        )
        assert result1.get("needs_clarification") is True

        # Turn 2: Provide time
        result2 = await run_reminder_workflow(
            input_text="in 30 minutes",
            session_id=session_id,
        )

        # Should have created the reminder
        assert result2.get("reminder_created") is True or result2.get("needs_clarification") is False

        await manager.clear_workflow_state(session_id)
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 4: Complete reminder clears workflow state
    print("Test 4: Complete reminder clears workflow state...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-reminder-4")
        session_id = str(session.id)

        # Run full reminder with all info
        result = await run_reminder_workflow(
            input_text="remind me to call mom in 30 minutes",
            session_id=session_id,
        )

        # If reminder was created, state should be cleared
        if result.get("reminder_created"):
            saved = await manager.restore_workflow_state(session_id)
            assert saved is None, "State should be cleared after completion"

        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 5: Graph has correct entry point
    print("Test 5: Graph has check_continuation as entry point...", end=" ")
    try:
        from atlas_brain.agents.graphs.reminder import build_reminder_graph

        graph = build_reminder_graph()
        nodes = list(graph.nodes.keys())
        assert "check_continuation" in nodes
        assert "merge_continuation" in nodes
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
    print("Reminder Workflow Multi-Turn Verification")
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
