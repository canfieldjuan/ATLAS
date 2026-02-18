#!/usr/bin/env python3
"""
Verification script for multi-turn booking workflow.

Run: ATLAS_DB_PORT=5433 python verify_booking_multiturn.py
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
    """Run verification tests for multi-turn booking workflow."""
    from atlas_brain.storage.database import get_db_pool
    from atlas_brain.storage.repositories.session import get_session_repo
    from atlas_brain.agents.graphs.booking import run_booking_workflow
    from atlas_brain.agents.graphs.workflow_state import get_workflow_state_manager

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

    # Test 1: New booking request saves state and awaits input
    print("Test 1: New request saves state and awaits input...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-booking-1")
        result = await run_booking_workflow(
            input_text="I want to schedule an appointment",
            session_id=str(session.id),
        )
        assert result.get("awaiting_user_input") is True, "Should await user input"
        assert result.get("response"), "Should have a response"

        # Verify state was saved
        saved = await manager.restore_workflow_state(str(session.id))
        assert saved is not None, "State should be saved"
        assert saved.workflow_type == "booking"
        assert saved.current_step == "conversation"

        await manager.clear_workflow_state(str(session.id))
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 2: Continuation restores context and responds
    print("Test 2: Continuation restores context and responds...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-booking-2")
        session_id = str(session.id)

        # Turn 1: Initial request
        result1 = await run_booking_workflow(
            input_text="book an appointment",
            session_id=session_id,
        )
        assert result1.get("awaiting_user_input") is True

        # Turn 2: Provide name
        result2 = await run_booking_workflow(
            input_text="It's for John Smith",
            session_id=session_id,
        )

        # LLM should respond (either ask for more info or proceed)
        assert result2.get("response") is not None, "Should have a response"

        await manager.clear_workflow_state(session_id)
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 3: Multi-turn flow preserves conversation context
    print("Test 3: Multi-turn flow preserves context...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-booking-3")
        session_id = str(session.id)

        # Turn 1: Initial request
        r1 = await run_booking_workflow("schedule appointment", session_id=session_id)
        assert r1.get("awaiting_user_input") is True

        # Turn 2: Provide name
        r2 = await run_booking_workflow("John Smith", session_id=session_id)
        assert r2.get("response"), "Should respond after name"

        # Turn 3: Provide more info
        r3 = await run_booking_workflow("555-123-4567", session_id=session_id)
        assert r3.get("response"), "Should respond after phone"

        # Verify context grows with each turn
        saved = await manager.restore_workflow_state(session_id)
        if saved:
            assert len(saved.conversation_context) >= 4, (
                f"Should have at least 4 context turns (3 user + responses), "
                f"got {len(saved.conversation_context)}"
            )

        await manager.clear_workflow_state(session_id)
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 4: Workflow state expires after timeout
    print("Test 4: Workflow state expiration...", end=" ")
    try:
        from atlas_brain.agents.graphs.workflow_state import ActiveWorkflowState
        from datetime import datetime, timedelta, timezone

        # Create expired state
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
        expired = ActiveWorkflowState(
            workflow_type="booking",
            current_step="conversation",
            started_at=old_time,
            partial_state={"speaker_id": "test"},
        )
        assert expired.is_expired(10) is True
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 5: Booking completion clears state
    print("Test 5: Booking completion clears state...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-booking-5")
        session_id = str(session.id)

        # Run booking with all info — LLM may complete in one turn
        result = await run_booking_workflow(
            input_text="Book appointment for John Smith phone 555-1234 tomorrow at 2pm",
            session_id=session_id,
        )

        # If booking completed (awaiting_user_input=False), state should be cleared
        if not result.get("awaiting_user_input", True):
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

    # Cleanup
    await pool.close()

    return all_passed


def main():
    """Main entry point."""
    print("=" * 50)
    print("Multi-Turn Booking Workflow Verification")
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
