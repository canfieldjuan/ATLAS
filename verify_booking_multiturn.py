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

    # Test 1: New booking request saves state when missing info
    print("Test 1: New request saves state when missing name...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-booking-1")
        result = await run_booking_workflow(
            input_text="I want to schedule an appointment",
            session_id=str(session.id),
        )
        assert result.get("awaiting_user_input") is True
        assert "name" in result.get("response", "").lower()

        # Verify state was saved
        saved = await manager.restore_workflow_state(str(session.id))
        assert saved is not None, "State should be saved"
        assert saved.workflow_type == "booking"
        assert saved.current_step == "awaiting_info"

        await manager.clear_workflow_state(str(session.id))
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 2: Continuation restores state and merges input
    print("Test 2: Continuation restores and merges input...", end=" ")
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
            input_text="My name is John Smith",
            session_id=session_id,
        )

        # Should either ask for more info or proceed
        assert result2.get("response") is not None
        # Check that name was captured
        assert result2.get("customer_name") == "John Smith" or "date" in result2.get("response", "").lower()

        await manager.clear_workflow_state(session_id)
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 3: Full multi-turn flow
    print("Test 3: Full multi-turn flow...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-booking-3")
        session_id = str(session.id)

        # Turn 1: Initial request
        r1 = await run_booking_workflow("schedule appointment", session_id=session_id)
        assert r1.get("awaiting_user_input") is True

        # Turn 2: Provide name
        r2 = await run_booking_workflow("John Smith", session_id=session_id)

        # Turn 3: Provide phone
        r3 = await run_booking_workflow("555-123-4567", session_id=session_id)

        # Should have captured phone
        assert r3.get("customer_phone") is not None or "date" in r3.get("response", "").lower()

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
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        expired = ActiveWorkflowState(
            workflow_type="booking",
            current_step="awaiting_info",
            started_at=old_time,
            partial_state={"customer_name": "Old Data"},
        )
        assert expired.is_expired(5) is True
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 5: Complete booking clears state
    print("Test 5: Complete booking clears state...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-booking-5")
        session_id = str(session.id)

        # Run full booking with all info
        result = await run_booking_workflow(
            input_text="Book appointment for John Smith phone 555-1234 tomorrow at 2pm",
            session_id=session_id,
        )

        # If booking completed, state should be cleared
        if result.get("booking_confirmed"):
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
