"""
End-to-end tests for session management.

Tests daily sessions, multi-terminal continuity, and session lifecycle.
"""

from datetime import date, datetime
from uuid import uuid4

import pytest
import pytest_asyncio


@pytest.mark.integration
class TestDailySessions:
    """Test daily session creation and management."""

    @pytest.mark.asyncio
    async def test_create_session_anonymous(self, session_repo):
        """Anonymous sessions can be created without user_id."""
        session = await session_repo.create_session(
            user_id=None,
            terminal_id="test-terminal",
        )

        assert session is not None
        assert session.id is not None
        assert session.user_id is None
        assert session.terminal_id == "test-terminal"
        assert session.is_active is True
        assert session.session_date == date.today()

        # Cleanup
        await session_repo.close_session(session.id)

    @pytest.mark.asyncio
    async def test_create_session_with_user(self, db_pool, session_repo):
        """Sessions can be created with a user_id."""
        # Create user first
        user_id = uuid4()
        await db_pool.execute(
            "INSERT INTO users (id, name, created_at) VALUES ($1, 'Test User', NOW())",
            user_id,
        )

        session = await session_repo.create_session(
            user_id=user_id,
            terminal_id="office",
        )

        assert session.user_id == user_id
        assert session.terminal_id == "office"
        assert session.session_date == date.today()

        # Cleanup
        await session_repo.close_session(session.id)
        await db_pool.execute("DELETE FROM users WHERE id = $1", user_id)

    @pytest.mark.asyncio
    async def test_get_todays_session(self, db_pool, session_repo):
        """get_todays_session_for_user returns only today's session."""
        user_id = uuid4()
        await db_pool.execute(
            "INSERT INTO users (id, name, created_at) VALUES ($1, 'Test User', NOW())",
            user_id,
        )

        # Create today's session
        session = await session_repo.create_session(
            user_id=user_id,
            terminal_id="living-room",
        )

        # Should find today's session
        found = await session_repo.get_todays_session_for_user(user_id)
        assert found is not None
        assert found.id == session.id
        assert found.session_date == date.today()

        # Cleanup
        await session_repo.close_session(session.id)
        await db_pool.execute("DELETE FROM users WHERE id = $1", user_id)

    @pytest.mark.asyncio
    async def test_get_or_create_returns_existing(self, db_pool, session_repo):
        """get_or_create_session returns existing session for today."""
        user_id = uuid4()
        await db_pool.execute(
            "INSERT INTO users (id, name, created_at) VALUES ($1, 'Test User', NOW())",
            user_id,
        )

        # First call creates session
        session1 = await session_repo.get_or_create_session(
            user_id=user_id,
            terminal_id="kitchen",
        )

        # Second call returns same session
        session2 = await session_repo.get_or_create_session(
            user_id=user_id,
            terminal_id="kitchen",
        )

        assert session1.id == session2.id

        # Cleanup
        await session_repo.close_session(session1.id)
        await db_pool.execute("DELETE FROM users WHERE id = $1", user_id)


@pytest.mark.integration
class TestMultiTerminalContinuity:
    """Test session continuity across terminals."""

    @pytest.mark.asyncio
    async def test_terminal_update_on_move(self, db_pool, session_repo):
        """Terminal ID updates when user moves to new location."""
        user_id = uuid4()
        await db_pool.execute(
            "INSERT INTO users (id, name, created_at) VALUES ($1, 'Test User', NOW())",
            user_id,
        )

        # Start in office
        session1 = await session_repo.get_or_create_session(
            user_id=user_id,
            terminal_id="office",
        )
        assert session1.terminal_id == "office"

        # Move to living room - same session, updated terminal
        session2 = await session_repo.get_or_create_session(
            user_id=user_id,
            terminal_id="living-room",
        )

        assert session2.id == session1.id  # Same session
        assert session2.terminal_id == "living-room"  # Updated terminal

        # Cleanup
        await session_repo.close_session(session1.id)
        await db_pool.execute("DELETE FROM users WHERE id = $1", user_id)

    @pytest.mark.asyncio
    async def test_session_follows_user(self, db_pool, session_repo, conversation_repo):
        """Conversation history follows user across terminals."""
        user_id = uuid4()
        await db_pool.execute(
            "INSERT INTO users (id, name, created_at) VALUES ($1, 'Test User', NOW())",
            user_id,
        )

        # Start conversation in office
        session = await session_repo.get_or_create_session(
            user_id=user_id,
            terminal_id="office",
        )

        # Add some conversation
        await conversation_repo.add_turn(
            session_id=session.id,
            role="user",
            content="What's the weather?",
            turn_type="conversation",
        )
        await conversation_repo.add_turn(
            session_id=session.id,
            role="assistant",
            content="It's sunny today!",
            turn_type="conversation",
        )

        # Move to car terminal
        session_car = await session_repo.get_or_create_session(
            user_id=user_id,
            terminal_id="car",
        )

        # Same session
        assert session_car.id == session.id

        # History should be available
        history = await conversation_repo.get_history(session_car.id, limit=10)
        assert len(history) == 2
        assert history[0].content == "What's the weather?"

        # Cleanup
        await db_pool.execute(
            "DELETE FROM conversation_turns WHERE session_id = $1",
            session.id,
        )
        await session_repo.close_session(session.id)
        await db_pool.execute("DELETE FROM users WHERE id = $1", user_id)


@pytest.mark.integration
class TestSessionLifecycle:
    """Test session close and cleanup."""

    @pytest.mark.asyncio
    async def test_close_session(self, session_repo):
        """Closing session marks it as inactive."""
        session = await session_repo.create_session(
            terminal_id="test",
        )

        assert session.is_active is True

        await session_repo.close_session(session.id)

        # Refetch and check
        closed = await session_repo.get_session(session.id)
        assert closed.is_active is False

    @pytest.mark.asyncio
    async def test_close_user_sessions(self, db_pool, session_repo):
        """close_user_sessions closes all active sessions for user."""
        user_id = uuid4()
        await db_pool.execute(
            "INSERT INTO users (id, name, created_at) VALUES ($1, 'Test User', NOW())",
            user_id,
        )

        # Create multiple sessions (shouldn't happen normally, but test the cleanup)
        session1 = await session_repo.create_session(user_id=user_id, terminal_id="a")
        session2 = await session_repo.create_session(user_id=user_id, terminal_id="b")

        closed_count = await session_repo.close_user_sessions(user_id)
        assert closed_count == 2

        s1 = await session_repo.get_session(session1.id)
        s2 = await session_repo.get_session(session2.id)
        assert s1.is_active is False
        assert s2.is_active is False

        # Cleanup
        await db_pool.execute("DELETE FROM users WHERE id = $1", user_id)

    @pytest.mark.asyncio
    async def test_list_active_sessions(self, session_repo):
        """list_active_sessions returns only active sessions."""
        session = await session_repo.create_session(terminal_id="test")

        active = await session_repo.list_active_sessions(limit=100)
        session_ids = [s.id for s in active]
        assert session.id in session_ids

        await session_repo.close_session(session.id)

        # Should no longer appear
        active = await session_repo.list_active_sessions(limit=100)
        session_ids = [s.id for s in active]
        assert session.id not in session_ids


@pytest.mark.integration
class TestSessionMetadata:
    """Test session metadata operations."""

    @pytest.mark.asyncio
    async def test_update_metadata_new_key(self, session_repo):
        """update_metadata adds new keys to empty metadata."""
        session = await session_repo.create_session(terminal_id="test")

        updated = await session_repo.update_metadata(
            session.id,
            {"active_workflow": {"workflow_type": "booking", "step": "parse"}},
        )
        assert updated is True

        # Verify the metadata was saved
        refreshed = await session_repo.get_session(session.id)
        assert refreshed.metadata.get("active_workflow") is not None
        assert refreshed.metadata["active_workflow"]["workflow_type"] == "booking"

        # Cleanup
        await session_repo.close_session(session.id)

    @pytest.mark.asyncio
    async def test_update_metadata_merge(self, session_repo):
        """update_metadata merges with existing metadata."""
        session = await session_repo.create_session(
            terminal_id="test",
            metadata={"existing_key": "existing_value"},
        )

        # Add new key without overwriting existing
        await session_repo.update_metadata(
            session.id,
            {"new_key": "new_value"},
        )

        refreshed = await session_repo.get_session(session.id)
        assert refreshed.metadata.get("existing_key") == "existing_value"
        assert refreshed.metadata.get("new_key") == "new_value"

        # Cleanup
        await session_repo.close_session(session.id)

    @pytest.mark.asyncio
    async def test_update_metadata_overwrite_key(self, session_repo):
        """update_metadata overwrites existing keys."""
        session = await session_repo.create_session(
            terminal_id="test",
            metadata={"step": "parse"},
        )

        await session_repo.update_metadata(session.id, {"step": "confirm"})

        refreshed = await session_repo.get_session(session.id)
        assert refreshed.metadata.get("step") == "confirm"

        # Cleanup
        await session_repo.close_session(session.id)

    @pytest.mark.asyncio
    async def test_clear_metadata_key(self, session_repo):
        """clear_metadata_key removes specific key."""
        session = await session_repo.create_session(
            terminal_id="test",
            metadata={"keep_this": "yes", "remove_this": "bye"},
        )

        cleared = await session_repo.clear_metadata_key(session.id, "remove_this")
        assert cleared is True

        refreshed = await session_repo.get_session(session.id)
        assert "remove_this" not in refreshed.metadata
        assert refreshed.metadata.get("keep_this") == "yes"

        # Cleanup
        await session_repo.close_session(session.id)

    @pytest.mark.asyncio
    async def test_update_metadata_nonexistent_session(self, session_repo):
        """update_metadata returns False for nonexistent session."""
        fake_id = uuid4()
        updated = await session_repo.update_metadata(fake_id, {"test": "value"})
        assert updated is False

    @pytest.mark.asyncio
    async def test_clear_metadata_key_nonexistent(self, session_repo):
        """clear_metadata_key returns False for nonexistent session."""
        fake_id = uuid4()
        cleared = await session_repo.clear_metadata_key(fake_id, "some_key")
        assert cleared is False