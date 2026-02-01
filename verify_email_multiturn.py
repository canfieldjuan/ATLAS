#!/usr/bin/env python3
"""
Verification script for email workflow multi-turn support.

Run: ATLAS_DB_PORT=5433 python verify_email_multiturn.py
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
    """Run verification tests for email workflow multi-turn."""
    from atlas_brain.storage.database import get_db_pool
    from atlas_brain.storage.repositories.session import get_session_repo
    from atlas_brain.agents.graphs.workflow_state import get_workflow_state_manager
    from atlas_brain.agents.graphs.email import (
        run_email_workflow,
        EMAIL_WORKFLOW_TYPE,
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

    # Test 1: Email request saves state when fields missing
    print("Test 1: Email request saves state when fields missing...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-email-1")
        result = await run_email_workflow(
            input_text="send an email",
            session_id=str(session.id),
        )
        assert result.get("needs_clarification") is True

        # Verify state was saved
        saved = await manager.restore_workflow_state(str(session.id))
        assert saved is not None, "State should be saved"
        assert saved.workflow_type == EMAIL_WORKFLOW_TYPE
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

    # Test 2: Estimate email saves state when fields missing
    print("Test 2: Estimate email saves state when fields missing...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-email-2")
        result = await run_email_workflow(
            input_text="send an estimate email",
            session_id=str(session.id),
        )
        assert result.get("needs_clarification") is True

        # Verify state was saved
        saved = await manager.restore_workflow_state(str(session.id))
        assert saved is not None, "State should be saved"
        assert saved.workflow_type == EMAIL_WORKFLOW_TYPE
        assert saved.partial_state.get("intent") == "send_estimate"

        await manager.clear_workflow_state(str(session.id))
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 3: Graph has check_continuation as entry point
    print("Test 3: Graph has check_continuation as entry point...", end=" ")
    try:
        from atlas_brain.agents.graphs.email import build_email_graph

        graph = build_email_graph()
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

    # Test 4: Continuation is detected
    print("Test 4: Continuation is detected...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-email-4")
        session_id = str(session.id)

        # Save a workflow state
        await manager.save_workflow_state(
            session_id=session_id,
            workflow_type=EMAIL_WORKFLOW_TYPE,
            current_step="awaiting_info",
            partial_state={
                "intent": "send_email",
                "to_address": None,
                "subject": "Test Subject",
                "body": None,
            },
        )

        from atlas_brain.agents.graphs.email import check_continuation

        state = {
            "input_text": "test@example.com",
            "session_id": session_id,
        }
        result = await check_continuation(state)

        assert result.get("is_continuation") is True
        assert result.get("restored_from_step") == "awaiting_info"

        await manager.clear_workflow_state(session_id)
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 5: Routing functions work correctly
    print("Test 5: Routing functions work correctly...", end=" ")
    try:
        from atlas_brain.agents.graphs.email import (
            route_after_check_continuation,
            route_after_merge,
        )

        # No continuation - go to classify
        state1 = {"is_continuation": False}
        assert route_after_check_continuation(state1) == "classify_intent"

        # Continuation - go to merge
        state2 = {"is_continuation": True}
        assert route_after_check_continuation(state2) == "merge_continuation"

        # After merge with estimate intent
        state3 = {"intent": "send_estimate"}
        assert route_after_merge(state3) == "extract_context"

        # After merge with generic email intent
        state4 = {"intent": "send_email"}
        assert route_after_merge(state4) == "generate_draft"

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
    print("Email Workflow Multi-Turn Verification")
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
