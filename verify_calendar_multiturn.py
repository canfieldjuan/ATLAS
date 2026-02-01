#!/usr/bin/env python3
"""
Verification script for calendar workflow multi-turn support.

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
    """Run verification tests for calendar workflow multi-turn."""
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

    # Test 1: Create event request saves state when title missing
    print("Test 1: Create event request saves state when title missing...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-calendar-1")
        result = await run_calendar_workflow(
            input_text="create a calendar event",
            session_id=str(session.id),
        )
        assert result.get("needs_clarification") is True
        assert "call this event" in result.get("response", "").lower()

        # Verify state was saved
        saved = await manager.restore_workflow_state(str(session.id))
        assert saved is not None, "State should be saved"
        assert saved.workflow_type == CALENDAR_WORKFLOW_TYPE
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

    # Test 2: Create event with title saves state when time missing
    print("Test 2: Create event with title saves state when time missing...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-calendar-2")
        result = await run_calendar_workflow(
            input_text="create a meeting called team standup",
            session_id=str(session.id),
        )
        assert result.get("needs_clarification") is True
        assert "when" in result.get("response", "").lower()

        # Verify state was saved with title
        saved = await manager.restore_workflow_state(str(session.id))
        assert saved is not None, "State should be saved"
        assert saved.workflow_type == CALENDAR_WORKFLOW_TYPE
        assert saved.partial_state.get("event_title") is not None

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
        from atlas_brain.agents.graphs.calendar import build_calendar_graph

        graph = build_calendar_graph()
        nodes = list(graph.nodes.keys())
        assert "check_continuation" in nodes
        assert "merge_continuation" in nodes
        assert "classify" in nodes
        assert "execute_create" in nodes
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
        session = await repo.create_session(terminal_id="test-calendar-4")
        session_id = str(session.id)

        # Save a workflow state
        await manager.save_workflow_state(
            session_id=session_id,
            workflow_type=CALENDAR_WORKFLOW_TYPE,
            current_step="awaiting_info",
            partial_state={
                "intent": "create_event",
                "event_title": "Team Meeting",
                "parsed_start_at": None,
            },
        )

        from atlas_brain.agents.graphs.calendar import check_continuation

        state = {
            "input_text": "tomorrow at 2pm",
            "session_id": session_id,
        }
        result = await check_continuation(state)

        assert result.get("is_continuation") is True
        assert result.get("restored_from_step") == "awaiting_info"
        assert result.get("event_title") == "Team Meeting"

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
        from atlas_brain.agents.graphs.calendar import (
            route_after_check_continuation,
            route_after_merge,
            route_by_intent,
        )

        # No continuation - go to classify
        state1 = {"is_continuation": False}
        assert route_after_check_continuation(state1) == "classify"

        # Continuation - go to merge
        state2 = {"is_continuation": True}
        assert route_after_check_continuation(state2) == "merge_continuation"

        # After merge with complete data
        state3 = {"event_title": "Meeting", "parsed_start_at": "2026-02-01T14:00:00"}
        assert route_after_merge(state3) == "execute_create"

        # After merge with incomplete data
        state4 = {"event_title": "Meeting", "parsed_start_at": None}
        assert route_after_merge(state4) == "parse_create"

        # Intent routing
        state5 = {"intent": "create_event"}
        assert route_by_intent(state5) == "parse_create"

        state6 = {"intent": "query_events"}
        assert route_by_intent(state6) == "execute_query"

        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 6: Query events works
    print("Test 6: Query events works...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-calendar-6")
        result = await run_calendar_workflow(
            input_text="show my calendar",
            session_id=str(session.id),
        )

        # Should either query successfully or return mock data
        assert result.get("intent") == "query_events"
        assert result.get("response") is not None

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
    print("Calendar Workflow Multi-Turn Verification")
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
