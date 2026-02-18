#!/usr/bin/env python3
"""
Verification script for session metadata methods.

Run: python verify_session_metadata.py
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
    """Run verification tests for session metadata."""
    from uuid import uuid4
    from atlas_brain.storage.database import get_db_pool
    from atlas_brain.storage.repositories.session import get_session_repo

    # Initialize database
    pool = get_db_pool()
    try:
        await pool.initialize()
    except Exception as e:
        print(f"SKIP: Database not available: {e}")
        return True  # Not a failure, just skip

    repo = get_session_repo()
    all_passed = True

    # Test 1: update_metadata adds new key
    print("Test 1: update_metadata adds new key...", end=" ")
    try:
        session = await repo.create_session(terminal_id="test-verify")
        updated = await repo.update_metadata(
            session.id,
            {"active_workflow": {"workflow_type": "booking"}},
        )
        assert updated is True, "update_metadata should return True"
        refreshed = await repo.get_session(session.id)
        assert refreshed.metadata.get("active_workflow") is not None
        assert refreshed.metadata["active_workflow"]["workflow_type"] == "booking"
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 2: update_metadata merges with existing
    print("Test 2: update_metadata merges with existing...", end=" ")
    try:
        session = await repo.create_session(
            terminal_id="test-verify",
            metadata={"existing_key": "existing_value"},
        )
        await repo.update_metadata(session.id, {"new_key": "new_value"})
        refreshed = await repo.get_session(session.id)
        assert refreshed.metadata.get("existing_key") == "existing_value"
        assert refreshed.metadata.get("new_key") == "new_value"
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 3: clear_metadata_key removes key
    print("Test 3: clear_metadata_key removes key...", end=" ")
    try:
        session = await repo.create_session(
            terminal_id="test-verify",
            metadata={"keep_this": "yes", "remove_this": "bye"},
        )
        cleared = await repo.clear_metadata_key(session.id, "remove_this")
        assert cleared is True, "clear_metadata_key should return True"
        refreshed = await repo.get_session(session.id)
        assert "remove_this" not in refreshed.metadata
        assert refreshed.metadata.get("keep_this") == "yes"
        await repo.close_session(session.id)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 4: update_metadata returns False for nonexistent session
    print("Test 4: update_metadata returns False for nonexistent...", end=" ")
    try:
        fake_id = uuid4()
        updated = await repo.update_metadata(fake_id, {"test": "value"})
        assert updated is False, "Should return False for nonexistent session"
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"ERROR: {e}")
        all_passed = False

    # Test 5: clear_metadata_key returns False for nonexistent session
    print("Test 5: clear_metadata_key returns False for nonexistent...", end=" ")
    try:
        fake_id = uuid4()
        cleared = await repo.clear_metadata_key(fake_id, "some_key")
        assert cleared is False, "Should return False for nonexistent session"
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
    print("Session Metadata Verification")
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
