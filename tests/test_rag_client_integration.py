"""
Integration tests for RAG client consumers.

Verifies that all services using the unified RAGClient still work correctly:
1. retrieve_memory node (atlas.py) -> search() -> SearchSource[] in state
2. gather_context pre_fetched_sources path (service.py) -> track_sources
3. gather_context voice path (service.py) -> enhance_prompt
4. store_conversation (service.py) -> add_messages
5. nightly sync (nightly_memory_sync.py) -> send_messages
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas_brain.memory.rag_client import (
    EnhancedPromptResult,
    RAGClient,
    SearchResult,
    SearchSource,
)


def _make_source(name: str = "fact-1", fact: str = "The sky is blue") -> SearchSource:
    return SearchSource(
        uuid="aaaaaaaa-0000-0000-0000-000000000001",
        name=name,
        fact=fact,
        confidence=0.9,
    )


def _make_memory_service():
    """Build MemoryService with mocked deps."""
    from atlas_brain.memory.service import MemoryService

    with (
        patch("atlas_brain.memory.service.get_rag_client") as mock_rag,
        patch("atlas_brain.memory.service.get_token_estimator") as mock_te,
        patch("atlas_brain.memory.service.get_feedback_service") as mock_fb,
    ):
        estimator = MagicMock()
        estimator.optimize_context.side_effect = lambda ctx: (
            ctx,
            SimpleNamespace(total=0, to_dict=lambda: {}),
            False,
        )
        mock_te.return_value = estimator

        rag = MagicMock()
        rag.enhance_prompt = AsyncMock(
            return_value=EnhancedPromptResult(
                prompt="test",
                context_used=True,
                sources=[_make_source()],
            )
        )
        rag.add_messages = AsyncMock(return_value=True)
        mock_rag.return_value = rag

        fb = MagicMock()
        fb.track_sources = AsyncMock(return_value=MagicMock(usage_ids=["u1"]))
        mock_fb.return_value = fb

        svc = MemoryService()

    return svc


# ===================================================================
# 1. retrieve_memory -> search() -> SearchSource[] in state
# ===================================================================

class TestRetrieveMemorySearch:
    """retrieve_memory calls rag_client.search() and stores facts in state."""

    @pytest.mark.asyncio
    async def test_search_returns_sources_in_state(self):
        from atlas_brain.agents.graphs.atlas import retrieve_memory

        state = {
            "input_text": "What did we talk about yesterday?",
            "action_type": "conversation",
            "confidence": 0.95,
        }

        sources = [_make_source("mem-1", "We discussed the project")]

        mock_settings = MagicMock()
        mock_settings.memory.enabled = True
        mock_settings.memory.retrieve_context = True
        mock_settings.memory.context_results = 5
        mock_settings.memory.context_timeout = 5.0
        mock_settings.intent_router.conversation_confidence_threshold = 0.7

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = SimpleNamespace(
            use_rag=True, category="knowledge", reason="test", confidence=1.0,
        )

        mock_client = MagicMock()
        mock_client.search_with_traversal = AsyncMock(
            return_value=SearchResult(facts=sources),
        )

        with (
            patch("atlas_brain.config.settings", mock_settings),
            patch(
                "atlas_brain.memory.query_classifier.get_query_classifier",
                MagicMock(return_value=mock_classifier),
            ),
            patch(
                "atlas_brain.memory.rag_client.get_rag_client",
                MagicMock(return_value=mock_client),
            ),
        ):
            cmd = await retrieve_memory(state)

        assert cmd.update["retrieved_sources"] == sources
        assert cmd.update["retrieved_sources"][0].fact == "We discussed the project"
        assert cmd.update["memory_ms"] > 0
        mock_client.search_with_traversal.assert_called_once()

    @pytest.mark.asyncio
    async def test_classifier_skip_sets_empty_sources(self):
        """When classifier says skip RAG, retrieved_sources should be []."""
        from atlas_brain.agents.graphs.atlas import retrieve_memory

        state = {
            "input_text": "Turn on the lights",
            "action_type": "conversation",
            "confidence": 0.3,
        }

        mock_settings = MagicMock()
        mock_settings.memory.enabled = True
        mock_settings.memory.retrieve_context = True
        mock_settings.intent_router.conversation_confidence_threshold = 0.7

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = SimpleNamespace(
            use_rag=False, category="command", reason="device", confidence=0.9,
        )

        with (
            patch("atlas_brain.config.settings", mock_settings),
            patch(
                "atlas_brain.memory.query_classifier.get_query_classifier",
                MagicMock(return_value=mock_classifier),
            ),
        ):
            cmd = await retrieve_memory(state)

        assert cmd.update["retrieved_sources"] == []
        assert cmd.goto == "parse"  # low confidence, goes to parse


# ===================================================================
# 2. gather_context pre_fetched_sources -> track_sources
# ===================================================================

class TestGatherContextPreFetched:
    """Pre-fetched sources from retrieve_memory flow through gather_context."""

    @pytest.mark.asyncio
    async def test_sources_tracked_for_feedback(self):
        svc = _make_memory_service()
        sources = [_make_source("tracked-fact")]

        with (
            patch("atlas_brain.memory.service.db_settings") as mock_db,
            patch("atlas_brain.memory.service.settings") as mock_settings,
        ):
            mock_db.enabled = False
            mock_settings.memory.enabled = True

            ctx = await svc.gather_context(
                query="test query",
                session_id="sess-123",
                include_rag=True,
                pre_fetched_sources=sources,
                include_history=False,
                include_physical=False,
            )

        assert ctx.rag_context_used is True
        assert ctx.rag_result is not None
        assert len(ctx.rag_result.sources) == 1
        assert ctx.rag_result.sources[0].name == "tracked-fact"
        # track_sources must have been called
        svc._feedback_service.track_sources.assert_called_once_with(
            session_id="sess-123",
            query="test query",
            sources=sources,
        )
        assert ctx.feedback_context is not None

    @pytest.mark.asyncio
    async def test_empty_sources_skip_tracking(self):
        """Empty pre_fetched_sources=[] should not call track_sources."""
        svc = _make_memory_service()

        with (
            patch("atlas_brain.memory.service.db_settings") as mock_db,
            patch("atlas_brain.memory.service.settings") as mock_settings,
        ):
            mock_db.enabled = False
            mock_settings.memory.enabled = True

            ctx = await svc.gather_context(
                query="hello",
                include_rag=True,
                pre_fetched_sources=[],
                include_history=False,
                include_physical=False,
            )

        assert ctx.rag_context_used is False
        assert ctx.rag_result is not None
        assert ctx.rag_result.sources == []
        svc._feedback_service.track_sources.assert_not_called()


# ===================================================================
# 3. gather_context voice path -> enhance_prompt
# ===================================================================

class TestGatherContextVoicePath:
    """Voice streaming path: pre_fetched_sources=None triggers enhance_prompt."""

    @pytest.mark.asyncio
    async def test_enhance_prompt_called_when_no_prefetch(self):
        svc = _make_memory_service()

        with (
            patch("atlas_brain.memory.service.db_settings") as mock_db,
            patch("atlas_brain.memory.service.settings") as mock_settings,
        ):
            mock_db.enabled = False
            mock_settings.memory.enabled = True
            mock_settings.memory.retrieve_context = True
            mock_settings.memory.context_results = 3

            ctx = await svc.gather_context(
                query="What is the weather?",
                session_id="sess-456",
                include_rag=True,
                pre_fetched_sources=None,  # voice path
                include_history=False,
                include_physical=False,
            )

        svc._rag_client.enhance_prompt.assert_called_once_with(
            query="What is the weather?",
            max_sources=3,
        )
        assert ctx.rag_context_used is True
        assert len(ctx.rag_result.sources) == 1
        svc._feedback_service.track_sources.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhance_prompt_not_called_when_prefetch_provided(self):
        """When pre_fetched_sources is provided, enhance_prompt must not run."""
        svc = _make_memory_service()
        sources = [_make_source()]

        with (
            patch("atlas_brain.memory.service.db_settings") as mock_db,
            patch("atlas_brain.memory.service.settings") as mock_settings,
        ):
            mock_db.enabled = False
            mock_settings.memory.enabled = True

            await svc.gather_context(
                query="test",
                include_rag=True,
                pre_fetched_sources=sources,
                include_history=False,
                include_physical=False,
            )

        svc._rag_client.enhance_prompt.assert_not_called()


# ===================================================================
# 4. store_conversation -> add_messages (GraphRAG storage)
# ===================================================================

class TestStoreConversationGraphRAG:
    """store_conversation sends turns to GraphRAG via add_messages."""

    @pytest.mark.asyncio
    async def test_conversation_turns_sent_to_graphrag(self):
        svc = _make_memory_service()

        with (
            patch("atlas_brain.memory.service.db_settings") as mock_db,
            patch("atlas_brain.memory.service.settings") as mock_settings,
        ):
            mock_db.enabled = False
            mock_settings.memory.enabled = True
            mock_settings.memory.store_conversations = True

            await svc.store_conversation(
                session_id="00000000-0000-0000-0000-000000000001",
                user_content="How are you?",
                assistant_content="I'm doing well!",
                turn_type="conversation",
            )

        svc._rag_client.add_messages.assert_called_once()
        call_kwargs = svc._rag_client.add_messages.call_args
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role_type"] == "user"
        assert messages[0]["content"] == "How are you?"
        assert messages[1]["role_type"] == "assistant"
        assert messages[1]["content"] == "I'm doing well!"

    @pytest.mark.asyncio
    async def test_command_turns_not_sent_to_graphrag(self):
        """turn_type='command' should not be sent to GraphRAG."""
        svc = _make_memory_service()

        with (
            patch("atlas_brain.memory.service.db_settings") as mock_db,
            patch("atlas_brain.memory.service.settings") as mock_settings,
        ):
            mock_db.enabled = False
            mock_settings.memory.enabled = True
            mock_settings.memory.store_conversations = True

            await svc.store_conversation(
                session_id="00000000-0000-0000-0000-000000000001",
                user_content="Turn on lights",
                assistant_content="Done!",
                turn_type="command",
            )

        svc._rag_client.add_messages.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_conversations_disabled_skips_graphrag(self):
        """When store_conversations=False, add_messages is not called."""
        svc = _make_memory_service()

        with (
            patch("atlas_brain.memory.service.db_settings") as mock_db,
            patch("atlas_brain.memory.service.settings") as mock_settings,
        ):
            mock_db.enabled = False
            mock_settings.memory.enabled = True
            mock_settings.memory.store_conversations = False

            await svc.store_conversation(
                session_id="00000000-0000-0000-0000-000000000001",
                user_content="Hello",
                assistant_content="Hi!",
                turn_type="conversation",
            )

        svc._rag_client.add_messages.assert_not_called()


# ===================================================================
# 5. Nightly sync -> send_messages
# ===================================================================

class TestNightlySyncSendMessages:
    """Nightly sync uses rag_client.send_messages() for batch extraction."""

    @pytest.mark.asyncio
    async def test_send_messages_called_with_formatted_turns(self):
        mock_settings = MagicMock()
        mock_settings.memory.enabled = True
        mock_settings.memory.purge_days = 30

        with patch("atlas_brain.config.settings", mock_settings):
            from atlas_brain.jobs.nightly_memory_sync import NightlyMemorySync
            sync = NightlyMemorySync(purge_days=30)

        mock_client = MagicMock()
        mock_client.send_messages = AsyncMock(return_value={"success": True})
        sync._rag_client = mock_client
        sync._ensure_graphiti_reachable = AsyncMock(return_value=True)

        sync._load_unsynced_turns = AsyncMock(return_value={
            "session-1": [
                {"id": "t1", "role": "user", "content": "Hello", "speaker_id": "Juan", "created_at": None, "metadata": None},
                {"id": "t2", "role": "assistant", "content": "Hi there!", "speaker_id": None, "created_at": None, "metadata": None},
            ],
        })
        sync._mark_turns_synced = AsyncMock()
        sync._purge_old_messages = AsyncMock(return_value=0)

        summary = await sync.run()

        mock_client.send_messages.assert_called_once()
        messages = mock_client.send_messages.call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["content"] == "Hello"
        assert messages[0]["role_type"] == "user"
        assert messages[0]["role"] == "Juan"
        assert messages[1]["content"] == "Hi there!"
        assert messages[1]["role_type"] == "assistant"
        sync._mark_turns_synced.assert_called_once_with(["t1", "t2"])
        assert summary["sessions_processed"] == 1
        assert summary["turns_sent"] == 2

    @pytest.mark.asyncio
    async def test_send_messages_failure_skips_mark_synced(self):
        """When send_messages returns empty dict (failure), turns are not marked synced."""
        mock_settings = MagicMock()
        mock_settings.memory.enabled = True
        mock_settings.memory.purge_days = 30

        with patch("atlas_brain.config.settings", mock_settings):
            from atlas_brain.jobs.nightly_memory_sync import NightlyMemorySync
            sync = NightlyMemorySync(purge_days=30)

        mock_client = MagicMock()
        mock_client.send_messages = AsyncMock(return_value={})  # empty = failure
        sync._rag_client = mock_client
        sync._ensure_graphiti_reachable = AsyncMock(return_value=True)

        sync._load_unsynced_turns = AsyncMock(return_value={
            "session-1": [
                {"id": "t1", "role": "user", "content": "Hello", "speaker_id": None, "created_at": None, "metadata": None},
            ],
        })
        sync._mark_turns_synced = AsyncMock()
        sync._purge_old_messages = AsyncMock(return_value=0)

        summary = await sync.run()

        sync._mark_turns_synced.assert_not_called()
        assert summary["sessions_processed"] == 0

    @pytest.mark.asyncio
    async def test_turn_cap_limits_processing(self):
        """max_turns_per_run caps how many turns are processed per run."""
        mock_settings = MagicMock()
        mock_settings.memory.enabled = True
        mock_settings.memory.purge_days = 30

        with patch("atlas_brain.config.settings", mock_settings):
            from atlas_brain.jobs.nightly_memory_sync import NightlyMemorySync
            sync = NightlyMemorySync(purge_days=30, max_turns_per_run=3)

        mock_client = MagicMock()
        mock_client.send_messages = AsyncMock(return_value={"success": True})
        sync._rag_client = mock_client
        sync._ensure_graphiti_reachable = AsyncMock(return_value=True)

        # 3 sessions with 2 turns each = 6 total, but cap is 3
        sync._load_unsynced_turns = AsyncMock(return_value={
            "session-1": [
                {"id": "t1", "role": "user", "content": "Hello", "speaker_id": None, "created_at": None, "metadata": None},
                {"id": "t2", "role": "assistant", "content": "Hi!", "speaker_id": None, "created_at": None, "metadata": None},
            ],
            "session-2": [
                {"id": "t3", "role": "user", "content": "Bye", "speaker_id": None, "created_at": None, "metadata": None},
                {"id": "t4", "role": "assistant", "content": "Later!", "speaker_id": None, "created_at": None, "metadata": None},
            ],
            "session-3": [
                {"id": "t5", "role": "user", "content": "Test", "speaker_id": None, "created_at": None, "metadata": None},
                {"id": "t6", "role": "assistant", "content": "OK!", "speaker_id": None, "created_at": None, "metadata": None},
            ],
        })
        sync._mark_turns_synced = AsyncMock()
        sync._purge_old_messages = AsyncMock(return_value=0)

        summary = await sync.run()

        # Should process session-1 (2 turns) + session-2 (2 turns, budget goes to -1)
        # but session-3 should be skipped (budget exhausted)
        assert summary["turns_sent"] == 4
        assert summary["sessions_processed"] == 2
        assert summary["turns_remaining"] == 2
        assert mock_client.send_messages.call_count == 2

    @pytest.mark.asyncio
    async def test_preflight_unreachable_aborts_sync(self):
        """When Graphiti is unreachable and restart fails, sync aborts."""
        mock_settings = MagicMock()
        mock_settings.memory.enabled = True
        mock_settings.memory.purge_days = 30

        with patch("atlas_brain.config.settings", mock_settings):
            from atlas_brain.jobs.nightly_memory_sync import NightlyMemorySync
            sync = NightlyMemorySync(purge_days=30)

        sync._ensure_graphiti_reachable = AsyncMock(return_value=False)
        sync._load_unsynced_turns = AsyncMock()

        summary = await sync.run()

        assert "Graphiti unreachable" in summary["errors"][0]
        sync._load_unsynced_turns.assert_not_called()

    @pytest.mark.asyncio
    async def test_preflight_auto_recovery(self):
        """Pre-flight restarts container and retries when Graphiti is down."""
        mock_settings = MagicMock()
        mock_settings.memory.enabled = True
        mock_settings.memory.purge_days = 30
        mock_settings.memory.base_url = "http://localhost:8001"

        with patch("atlas_brain.config.settings", mock_settings):
            from atlas_brain.jobs.nightly_memory_sync import NightlyMemorySync
            sync = NightlyMemorySync(purge_days=30)

        # First ping fails, then succeeds after restart
        call_count = 0
        async def mock_ping(url):
            nonlocal call_count
            call_count += 1
            return call_count > 1  # fail first, pass after

        sync._ping_graphiti = staticmethod(mock_ping)

        with patch(
            "atlas_brain.jobs.nightly_memory_sync.subprocess.run"
        ) as mock_run, patch(
            "atlas_brain.jobs.nightly_memory_sync.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            result = await sync._ensure_graphiti_reachable()

        assert result is True
        mock_run.assert_called_once()
        assert "restart" in str(mock_run.call_args)


# ===================================================================
# 6. RAGClient method signatures (contract test)
# ===================================================================

class TestRAGClientContract:
    """Verify all expected methods exist with correct signatures."""

    def test_all_methods_present(self):
        client = RAGClient()
        for method in [
            "search", "enhance_prompt", "add_messages",
            "send_messages", "add_conversation_turn", "add_fact",
            "health_check", "close",
        ]:
            assert hasattr(client, method), f"Missing method: {method}"
            assert callable(getattr(client, method)), f"{method} not callable"

    def test_search_returns_search_result(self):
        """SearchResult has facts list attribute."""
        result = SearchResult()
        assert hasattr(result, "facts")
        assert isinstance(result.facts, list)

    def test_search_source_has_required_fields(self):
        """SearchSource has uuid, name, fact, confidence."""
        src = _make_source()
        assert src.uuid
        assert src.name
        assert src.fact
        assert isinstance(src.confidence, float)

    def test_enhanced_prompt_result_fields(self):
        """EnhancedPromptResult has prompt, context_used, sources."""
        result = EnhancedPromptResult(prompt="q", context_used=True, sources=[_make_source()])
        assert result.prompt == "q"
        assert result.context_used is True
        assert len(result.sources) == 1
