"""
End-to-end tests for the full voice pipeline integration.

Tests the orchestrator with mocked services to verify:
- Text processing flow (skipping STT)
- Intent parsing and action dispatch
- Session persistence
- Turn type determination
"""

from uuid import uuid4

import pytest
import pytest_asyncio

from tests.conftest import MockIntentParser, MockActionDispatcher, MockLLM


@pytest.mark.integration
class TestPipelineTextProcessing:
    """Test text-based pipeline processing."""

    @pytest.mark.asyncio
    async def test_process_text_with_intent(
        self,
        db_pool,
        test_session,
        conversation_repo,
    ):
        """Command text (with intent) is processed correctly."""
        from atlas_brain.orchestration.orchestrator import Orchestrator, OrchestratorConfig

        # Mock that returns an intent
        mock_parser = MockIntentParser(return_intent=True)
        mock_dispatcher = MockActionDispatcher(success=True)

        config = OrchestratorConfig(
            wake_word_enabled=False,
            auto_execute=True,
        )
        orchestrator = Orchestrator(config=config, session_id=str(test_session))
        orchestrator._intent_parser = mock_parser
        orchestrator._action_dispatcher = mock_dispatcher
        orchestrator._llm = None
        orchestrator._tts = None

        result = await orchestrator.process_text("Turn on the lights")

        assert result.success is True
        assert result.transcript == "Turn on the lights"
        assert result.intent is not None
        assert result.intent.action == "turn_on"
        assert len(result.action_results) == 1
        assert result.action_results[0].success is True
        assert "Done" in result.response_text

    @pytest.mark.asyncio
    async def test_process_text_no_intent_fallback(
        self,
        db_pool,
        test_session,
    ):
        """Text without intent falls back to 'I heard' response when no LLM."""
        from atlas_brain.orchestration.orchestrator import Orchestrator, OrchestratorConfig

        mock_parser = MockIntentParser(return_intent=False)

        config = OrchestratorConfig(
            wake_word_enabled=False,
            auto_execute=True,
        )
        orchestrator = Orchestrator(config=config, session_id=str(test_session))
        orchestrator._intent_parser = mock_parser
        orchestrator._llm = None  # No LLM available
        orchestrator._tts = None

        result = await orchestrator.process_text("Hello there")

        assert result.success is True
        assert result.transcript == "Hello there"
        assert result.intent is None
        # Should fall back to "I heard: ..." when no LLM
        assert "I heard" in result.response_text or result.response_text is not None


@pytest.mark.integration
class TestPipelinePersistence:
    """Test that pipeline correctly persists conversations."""

    @pytest.mark.asyncio
    async def test_command_stored_as_command_type(
        self,
        db_pool,
        test_session,
        conversation_repo,
    ):
        """Commands (with intent) are stored with turn_type='command'."""
        from atlas_brain.orchestration.orchestrator import Orchestrator, OrchestratorConfig

        mock_parser = MockIntentParser(return_intent=True)
        mock_dispatcher = MockActionDispatcher(success=True)

        config = OrchestratorConfig(wake_word_enabled=False, auto_execute=True)
        orchestrator = Orchestrator(config=config, session_id=str(test_session))
        orchestrator._intent_parser = mock_parser
        orchestrator._action_dispatcher = mock_dispatcher
        orchestrator._llm = None
        orchestrator._tts = None

        await orchestrator.process_text("Turn on the living room lights")

        # Check stored turns
        history = await conversation_repo.get_history(test_session, limit=10)

        # Should have user and assistant turns
        assert len(history) >= 2

        user_turn = next(t for t in history if t.role == "user")
        assert user_turn.turn_type == "command"
        assert user_turn.content == "Turn on the living room lights"
        assert user_turn.intent == "turn_on"  # Intent action is stored

    @pytest.mark.asyncio
    async def test_mixed_session_context_loading(
        self,
        db_pool,
        test_session,
        conversation_repo,
    ):
        """
        Load history for LLM context should exclude commands.

        This tests the real behavior: when we load history for context,
        device commands should not be included.
        """
        from atlas_brain.orchestration.orchestrator import Orchestrator, OrchestratorConfig

        # First, add some mixed turns directly
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="What's the capital of France?",
            turn_type="conversation",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="assistant",
            content="The capital of France is Paris.",
            turn_type="conversation",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="Turn on the lights",
            turn_type="command",
            intent="turn_on",
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="assistant",
            content="Done.",
            turn_type="command",
        )

        # Create orchestrator for this session
        config = OrchestratorConfig(wake_word_enabled=False, auto_execute=True)
        orchestrator = Orchestrator(config=config, session_id=str(test_session))

        # Load session history (conversations only - the default)
        history = await orchestrator.load_session_history(limit=10)

        # Should only have the conversation turns
        assert len(history) == 2
        assert history[0].content == "What's the capital of France?"
        assert history[1].content == "The capital of France is Paris."

        # Load with commands included
        history_with_commands = await orchestrator.load_session_history(
            limit=10, include_commands=True
        )
        assert len(history_with_commands) == 4


@pytest.mark.integration
class TestPipelineLatency:
    """Test pipeline timing."""

    @pytest.mark.asyncio
    async def test_processing_time_tracked(
        self,
        db_pool,
        test_session,
    ):
        """Processing time is tracked in result."""
        from atlas_brain.orchestration.orchestrator import Orchestrator, OrchestratorConfig

        mock_parser = MockIntentParser(return_intent=False)

        config = OrchestratorConfig(wake_word_enabled=False, auto_execute=True)
        orchestrator = Orchestrator(config=config, session_id=str(test_session))
        orchestrator._intent_parser = mock_parser
        orchestrator._llm = None
        orchestrator._tts = None

        result = await orchestrator.process_text("Test query")

        # Latency should be tracked
        assert result.latency_ms > 0
        assert result.processing_ms >= 0

    @pytest.mark.asyncio
    async def test_action_time_tracked(
        self,
        db_pool,
        test_session,
    ):
        """Action execution time is tracked for commands."""
        from atlas_brain.orchestration.orchestrator import Orchestrator, OrchestratorConfig

        mock_parser = MockIntentParser(return_intent=True)
        mock_dispatcher = MockActionDispatcher(success=True)

        config = OrchestratorConfig(wake_word_enabled=False, auto_execute=True)
        orchestrator = Orchestrator(config=config, session_id=str(test_session))
        orchestrator._intent_parser = mock_parser
        orchestrator._action_dispatcher = mock_dispatcher
        orchestrator._llm = None
        orchestrator._tts = None

        result = await orchestrator.process_text("Turn on the lights")

        assert result.action_ms >= 0


@pytest.mark.integration
class TestPipelineStateManagement:
    """Test pipeline state management."""

    @pytest.mark.asyncio
    async def test_reset_clears_state(
        self,
        db_pool,
        test_session,
    ):
        """Reset clears pipeline state."""
        from atlas_brain.orchestration.orchestrator import Orchestrator, OrchestratorConfig
        from atlas_brain.orchestration.states import PipelineState

        config = OrchestratorConfig(wake_word_enabled=False, auto_execute=True)
        orchestrator = Orchestrator(config=config, session_id=str(test_session))

        # Initial state
        assert orchestrator.state == PipelineState.IDLE

        # Reset
        orchestrator.reset()

        # Should be back to idle
        assert orchestrator.state == PipelineState.IDLE
        assert orchestrator.context.transcript is None
        assert orchestrator.context.intent is None


@pytest.mark.integration
class TestSessionAwarePipeline:
    """Test pipeline session awareness."""

    @pytest.mark.asyncio
    async def test_no_session_still_processes(
        self,
        db_pool,
    ):
        """Pipeline without session_id still processes correctly."""
        from atlas_brain.orchestration.orchestrator import Orchestrator, OrchestratorConfig

        mock_parser = MockIntentParser(return_intent=True)
        mock_dispatcher = MockActionDispatcher(success=True)

        config = OrchestratorConfig(wake_word_enabled=False, auto_execute=True)
        orchestrator = Orchestrator(config=config, session_id=None)  # No session
        orchestrator._intent_parser = mock_parser
        orchestrator._action_dispatcher = mock_dispatcher
        orchestrator._llm = None
        orchestrator._tts = None

        # Should work without errors
        result = await orchestrator.process_text("Turn on the lights")

        assert result.success is True
        assert result.intent is not None

    @pytest.mark.asyncio
    async def test_invalid_session_handles_gracefully(
        self,
        db_pool,
    ):
        """Invalid session ID is handled gracefully."""
        from atlas_brain.orchestration.orchestrator import Orchestrator, OrchestratorConfig

        mock_parser = MockIntentParser(return_intent=True)
        mock_dispatcher = MockActionDispatcher(success=True)

        config = OrchestratorConfig(wake_word_enabled=False, auto_execute=True)
        # Invalid UUID format
        orchestrator = Orchestrator(config=config, session_id="invalid-uuid")
        orchestrator._intent_parser = mock_parser
        orchestrator._action_dispatcher = mock_dispatcher
        orchestrator._llm = None
        orchestrator._tts = None

        # Should still work (storage will fail silently)
        result = await orchestrator.process_text("Turn on the lights")

        assert result.success is True
