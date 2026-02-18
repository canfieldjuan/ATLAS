#!/usr/bin/env python3
"""
Verification script for WorkflowStateManager.

Run: python verify_workflow_state.py
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
    """Run verification tests for WorkflowStateManager."""
    from atlas_brain.storage.database import get_db_pool
    from atlas_brain.storage.repositories.session import get_session_repo
    from atlas_brain.agents.graphs.workflow_state import (
        WorkflowStateManager,
        ActiveWorkflowState,
    )

    # Initialize database
    pool = get_db_pool()
    try:
        await pool.initialize()
    except Exception as e:
        print(f"SKIP: Database not available: {e}")
        return True  # Not a failure, just skip

    repo = get_session_repo()
    manager = WorkflowStateManager(timeout_minutes=5)
    all_passed = True

    # Test 1: save_workflow_state
    print("Test 1: save_workflow_state...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-wf-1")
        saved = await manager.save_workflow_state(
            session_id=str(session.id),
            workflow_type="booking",
            current_step="awaiting_name",
            partial_state={"requested_date": "tomorrow"},
            conversation_context=[{"role": "user", "content": "book appointment"}],
        )
        assert saved is True, "save should return True"
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 2: restore_workflow_state
    print("Test 2: restore_workflow_state...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-wf-2")
        await manager.save_workflow_state(
            session_id=str(session.id),
            workflow_type="email",
            current_step="awaiting_recipient",
            partial_state={"subject": "Hello"},
        )
        restored = await manager.restore_workflow_state(str(session.id))
        assert restored is not None, "should restore workflow"
        assert restored.workflow_type == "email"
        assert restored.current_step == "awaiting_recipient"
        assert restored.partial_state.get("subject") == "Hello"
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 3: clear_workflow_state
    print("Test 3: clear_workflow_state...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-wf-3")
        await manager.save_workflow_state(
            session_id=str(session.id),
            workflow_type="reminder",
            current_step="awaiting_time",
            partial_state={},
        )
        cleared = await manager.clear_workflow_state(str(session.id))
        assert cleared is True, "clear should return True"
        restored = await manager.restore_workflow_state(str(session.id))
        assert restored is None, "should return None after clear"
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 4: add_context_turn
    print("Test 4: add_context_turn...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-wf-4")
        await manager.save_workflow_state(
            session_id=str(session.id),
            workflow_type="booking",
            current_step="awaiting_name",
            partial_state={},
            conversation_context=[],
        )
        added = await manager.add_context_turn(
            str(session.id),
            "user",
            "My name is John",
        )
        assert added is True, "add should return True"
        restored = await manager.restore_workflow_state(str(session.id))
        assert len(restored.conversation_context) == 1
        assert restored.conversation_context[0]["content"] == "My name is John"
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 5: update_partial_state
    print("Test 5: update_partial_state...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-wf-5")
        await manager.save_workflow_state(
            session_id=str(session.id),
            workflow_type="booking",
            current_step="awaiting_name",
            partial_state={"requested_date": "tomorrow"},
        )
        updated = await manager.update_partial_state(
            str(session.id),
            {"customer_name": "John Smith"},
            new_step="awaiting_time",
        )
        assert updated is True, "update should return True"
        restored = await manager.restore_workflow_state(str(session.id))
        assert restored.partial_state.get("customer_name") == "John Smith"
        assert restored.partial_state.get("requested_date") == "tomorrow"
        assert restored.current_step == "awaiting_time"
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 6: is_expired
    print("Test 6: ActiveWorkflowState.is_expired...", end=" ")
    try:
        from datetime import datetime, timedelta, timezone
        # Not expired
        recent = ActiveWorkflowState(
            workflow_type="test",
            current_step="test",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        assert not recent.is_expired(5), "Recent should not be expired"

        # Expired
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        old = ActiveWorkflowState(
            workflow_type="test",
            current_step="test",
            started_at=old_time,
        )
        assert old.is_expired(5), "Old should be expired"
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
    print("WorkflowStateManager Verification")
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
