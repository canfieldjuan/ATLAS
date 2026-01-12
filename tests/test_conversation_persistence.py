"""
End-to-end tests for conversation persistence.

Tests turn storage, history retrieval, and turn_type separation.
"""

from uuid import uuid4

import pytest
import pytest_asyncio


@pytest.mark.integration
class TestConversationTurns:
    """Test basic conversation turn operations."""

    @pytest.mark.asyncio
    async def test_add_turn_user(self, conversation_repo, test_session):
        """User turns are stored correctly."""
        turn_id = await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="Hello, Atlas!",
            turn_type="conversation",
        )

        assert turn_id is not None

        # Verify by fetching history
        history = await conversation_repo.get_history(test_session, limit=1)
        assert len(history) == 1
        assert history[0].role == "user"
        assert history[0].content == "Hello, Atlas!"
        assert history[0].turn_type == "conversation"

    @pytest.mark.asyncio
    async def test_add_turn_assistant(self, conversation_repo, test_session):
        """Assistant turns are stored correctly."""
        await conversation_repo.add_turn(
            session_id=test_session,
            role="assistant",
            content="Hello! How can I help?",
            turn_type="conversation",
        )

        history = await conversation_repo.get_history(test_session, limit=1)
        assert len(history) == 1
        assert history[0].role == "assistant"

    @pytest.mark.asyncio
    async def test_add_turn_with_speaker(self, conversation_repo, test_session):
        """Speaker ID is stored with turns."""
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="Turn on the lights",
            speaker_id="Juan",
            turn_type="command",
        )

        history = await conversation_repo.get_history(test_session, limit=1)
        assert history[0].speaker_id == "Juan"

    @pytest.mark.asyncio
    async def test_add_turn_with_intent(self, conversation_repo, test_session):
        """Intent is stored with turns."""
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="Turn off the TV",
            intent="turn_off",
            turn_type="command",
        )

        history = await conversation_repo.get_history(test_session, limit=1)
        assert history[0].intent == "turn_off"


@pytest.mark.integration
class TestTurnTypeSeparation:
    """Test that commands and conversations are tracked separately."""

    @pytest.mark.asyncio
    async def test_conversation_and_command_turns(self, conversation_repo, test_session):
        """Both turn types are stored correctly."""
        # Add a conversation
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="What time is it?",
            turn_type="conversation",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="assistant",
            content="It's 3pm.",
            turn_type="conversation",
        )

        # Add a command
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="Turn on the lights",
            intent="turn_on",
            turn_type="command",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="assistant",
            content="Done.",
            turn_type="command",
        )

        # All turns
        all_turns = await conversation_repo.get_history(test_session, limit=10)
        assert len(all_turns) == 4

    @pytest.mark.asyncio
    async def test_filter_conversations_only(self, conversation_repo, test_session):
        """Filtering by turn_type=conversation excludes commands."""
        # Add mixed turns
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="Tell me a joke",
            turn_type="conversation",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="Turn off the TV",
            turn_type="command",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="What's the weather?",
            turn_type="conversation",
        )

        # Filter conversations only
        conversations = await conversation_repo.get_history(
            test_session,
            limit=10,
            turn_type="conversation",
        )

        assert len(conversations) == 2
        assert all(t.turn_type == "conversation" for t in conversations)
        contents = [t.content for t in conversations]
        assert "Tell me a joke" in contents
        assert "What's the weather?" in contents
        assert "Turn off the TV" not in contents

    @pytest.mark.asyncio
    async def test_filter_commands_only(self, conversation_repo, test_session):
        """Filtering by turn_type=command excludes conversations."""
        # Add mixed turns
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="How are you?",
            turn_type="conversation",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="Turn on the lights",
            turn_type="command",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="Switch off the fan",
            turn_type="command",
        )

        # Filter commands only
        commands = await conversation_repo.get_history(
            test_session,
            limit=10,
            turn_type="command",
        )

        assert len(commands) == 2
        assert all(t.turn_type == "command" for t in commands)

    @pytest.mark.asyncio
    async def test_llm_context_excludes_commands(self, conversation_repo, test_session):
        """
        LLM context loading should only include conversations, not commands.

        This is the key behavior: when we load history for LLM context,
        device commands should be excluded to keep the context relevant.
        """
        # Simulate a realistic session
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="Hey Atlas, I'm planning a dinner party tonight",
            turn_type="conversation",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="assistant",
            content="That sounds fun! How many guests are you expecting?",
            turn_type="conversation",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="Turn on the living room lights",
            turn_type="command",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="assistant",
            content="Done.",
            turn_type="command",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="Set the lights to 50%",
            turn_type="command",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="assistant",
            content="Lights set to 50%.",
            turn_type="command",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="About 6 people",
            turn_type="conversation",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="assistant",
            content="Perfect! Do you need any recipe suggestions?",
            turn_type="conversation",
        )

        # Load context for LLM (conversations only)
        context_history = await conversation_repo.get_history(
            test_session,
            limit=10,
            turn_type="conversation",
        )

        # Should only have the 4 conversation turns
        assert len(context_history) == 4

        # Verify the conversation flow makes sense without commands
        contents = [t.content for t in context_history]
        assert "planning a dinner party" in contents[0]
        assert "How many guests" in contents[1]
        assert "About 6 people" in contents[2]
        assert "recipe suggestions" in contents[3]

        # Commands should NOT be in context
        for turn in context_history:
            assert "lights" not in turn.content.lower()


@pytest.mark.integration
class TestHistoryOrdering:
    """Test that history is returned in correct order."""

    @pytest.mark.asyncio
    async def test_history_oldest_first(self, conversation_repo, test_session):
        """History should be returned oldest first for context building."""
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="First message",
            turn_type="conversation",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="Second message",
            turn_type="conversation",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="Third message",
            turn_type="conversation",
        )

        history = await conversation_repo.get_history(test_session, limit=10)

        assert len(history) == 3
        assert history[0].content == "First message"
        assert history[1].content == "Second message"
        assert history[2].content == "Third message"

    @pytest.mark.asyncio
    async def test_history_limit(self, conversation_repo, test_session):
        """History limit returns most recent N turns."""
        # Add 5 turns
        for i in range(5):
            await conversation_repo.add_turn(
                session_id=test_session,
                role="user",
                content=f"Message {i+1}",
                turn_type="conversation",
            )

        # Get only last 3
        history = await conversation_repo.get_history(test_session, limit=3)

        assert len(history) == 3
        # Should be messages 3, 4, 5 (oldest first of the last 3)
        assert history[0].content == "Message 3"
        assert history[1].content == "Message 4"
        assert history[2].content == "Message 5"


@pytest.mark.integration
class TestConversationCount:
    """Test turn counting."""

    @pytest.mark.asyncio
    async def test_count_turns(self, conversation_repo, test_session):
        """count_turns returns correct count."""
        # Initially empty
        count = await conversation_repo.count_turns(test_session)
        assert count == 0

        # Add turns
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="Hello",
            turn_type="conversation",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="assistant",
            content="Hi!",
            turn_type="conversation",
        )

        count = await conversation_repo.count_turns(test_session)
        assert count == 2

    @pytest.mark.asyncio
    async def test_delete_session_history(self, conversation_repo, test_session):
        """delete_session_history removes all turns."""
        # Add turns
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="Message 1",
            turn_type="conversation",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="Message 2",
            turn_type="conversation",
        )

        # Delete
        deleted = await conversation_repo.delete_session_history(test_session)
        assert deleted == 2

        # Verify empty
        history = await conversation_repo.get_history(test_session, limit=10)
        assert len(history) == 0
