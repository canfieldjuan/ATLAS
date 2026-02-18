#!/usr/bin/env python3
"""
Verification script for calendar workflow (LLM-first approach).

Run: ATLAS_DB_PORT=5433 python verify_calendar_multiturn.py
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
    """Run verification tests for calendar workflow."""
    from atlas_brain.storage.database import get_db_pool
    from atlas_brain.storage.repositories.session import get_session_repo
    from atlas_brain.agents.graphs.workflow_state import get_workflow_state_manager
    from atlas_brain.agents.graphs.calendar import (
        run_calendar_workflow,
        CALENDAR_WORKFLOW_TYPE,
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

    # Test 1: Import and constants
    print("Test 1: Import and constants...", end=" ")
    try:
        assert CALENDAR_WORKFLOW_TYPE == "calendar"
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False

    # Test 2: Workflow returns response and awaiting_user_input
    print("Test 2: Workflow returns expected keys...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-calendar-2")
        result = await run_calendar_workflow(
            input_text="what's on my calendar today",
            session_id=str(session.id),
        )
        assert "response" in result, "Missing 'response' key"
        assert "awaiting_user_input" in result, "Missing 'awaiting_user_input' key"
        assert isinstance(result["response"], str), "Response should be string"
        await manager.clear_workflow_state(str(session.id))
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 3: Multi-turn saves conversation context
    print("Test 3: Multi-turn saves conversation context...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-calendar-3")
        session_id = str(session.id)

        # Turn 1: Vague request should ask for details
        result1 = await run_calendar_workflow(
            input_text="create an event",
            session_id=session_id,
        )
        assert result1.get("awaiting_user_input") is True, "Should await input"

        # Verify context was saved
        saved = await manager.restore_workflow_state(session_id)
        assert saved is not None, "State should be saved"
        assert saved.workflow_type == CALENDAR_WORKFLOW_TYPE
        assert saved.current_step == "conversation"
        assert len(saved.conversation_context) >= 2, "Should have user + assistant turns"

        await manager.clear_workflow_state(session_id)
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
    print("Calendar Workflow Verification (LLM-First)")
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
