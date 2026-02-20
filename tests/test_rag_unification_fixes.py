"""
Tests for RAG unification fixes:

Fix 1: User profile `enable_rag=False` must be respected even when
       `pre_fetched_sources` is provided to gather_context().

Fix 2: On timeout in `retrieve_memory`, the Command must return
       `retrieved_sources: []` (not leave it unset), so that
       gather_context receives an explicit empty list and does not retry.
"""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas_brain.memory.rag_client import EnhancedPromptResult, SearchSource
from atlas_brain.memory.service import MemoryContext, MemoryService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeProfile:
    """Minimal user-profile mock."""
    display_name: str = "Test User"
    timezone: str = "UTC"
    response_style: str = "balanced"
    expertise_level: str = "intermediate"
    enable_rag: bool = True


def _make_source(name: str = "test-fact") -> SearchSource:
    return SearchSource(
        uuid="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        name=name,
        fact=f"The sky is blue ({name})",
        confidence=0.9,
    )


def _build_memory_service() -> MemoryService:
    """Build a MemoryService with all heavy deps neutralised."""
    with (
        patch("atlas_brain.memory.service.get_rag_client") as mock_rag,
        patch("atlas_brain.memory.service.get_token_estimator") as mock_te,
        patch("atlas_brain.memory.service.get_feedback_service") as mock_fb,
    ):
        # Token estimator: pass context through unchanged
        estimator = MagicMock()
        estimator.optimize_context.side_effect = lambda ctx: (
            ctx,
            SimpleNamespace(total=0, to_dict=lambda: {}),
            False,
        )
        mock_te.return_value = estimator

        # RAG client: should never be called in these tests
        mock_rag.return_value = MagicMock()

        # Feedback service
        fb = MagicMock()
        fb.track_sources = AsyncMock(return_value=None)
        mock_fb.return_value = fb

        svc = MemoryService()

    return svc


# ===================================================================
# Fix 1 tests -- enable_rag=False on user profile
# ===================================================================

class TestProfileDisablesRag:
    """When user profile has enable_rag=False, RAG must be skipped
    even if pre_fetched_sources are provided."""

    @pytest.mark.asyncio
    async def test_pre_fetched_sources_ignored_when_profile_disables_rag(self):
        """Core positive test for Fix 1.

        pre_fetched_sources contains real sources, but the user profile
        says enable_rag=False.  gather_context must NOT use those sources.
        """
        svc = _build_memory_service()

        # Mock _load_user_profile to return a profile with enable_rag=False
        profile = _FakeProfile(enable_rag=False)
        svc._load_user_profile = AsyncMock(return_value=profile)

        sources = [_make_source("should-be-ignored")]

        with (
            patch("atlas_brain.memory.service.db_settings") as mock_db,
            patch("atlas_brain.memory.service.settings") as mock_settings,
        ):
            mock_db.enabled = True
            mock_settings.memory.enabled = True
            mock_settings.memory.context_results = 3

            ctx = await svc.gather_context(
                query="Tell me about the weather",
                session_id=None,
                user_id="00000000-0000-0000-0000-000000000001",
                include_rag=True,
                pre_fetched_sources=sources,
                include_history=False,
                include_physical=False,
            )

        # The profile override must have taken effect
        assert ctx.rag_context_used is False, (
            "rag_context_used should be False when profile disables RAG"
        )
        assert ctx.rag_result is None, (
            "rag_result should be None when profile disables RAG"
        )

    @pytest.mark.asyncio
    async def test_fresh_rag_search_also_skipped_when_profile_disables_rag(self):
        """Even without pre_fetched_sources, a fresh RAG search must be
        skipped when profile says enable_rag=False."""
        svc = _build_memory_service()

        profile = _FakeProfile(enable_rag=False)
        svc._load_user_profile = AsyncMock(return_value=profile)

        with (
            patch("atlas_brain.memory.service.db_settings") as mock_db,
            patch("atlas_brain.memory.service.settings") as mock_settings,
        ):
            mock_db.enabled = True
            mock_settings.memory.enabled = True
            mock_settings.memory.context_results = 3

            ctx = await svc.gather_context(
                query="Tell me about the weather",
                session_id=None,
                user_id="00000000-0000-0000-0000-000000000001",
                include_rag=True,
                pre_fetched_sources=None,  # no pre-fetch
                include_history=False,
                include_physical=False,
            )

        assert ctx.rag_context_used is False
        assert ctx.rag_result is None
        # RAG client enhance_prompt must NOT have been called
        svc._rag_client.enhance_prompt.assert_not_called()


class TestProfileAllowsRag:
    """Negative test: when enable_rag is True (default), pre_fetched_sources
    SHOULD be used."""

    @pytest.mark.asyncio
    async def test_pre_fetched_sources_used_when_profile_allows_rag(self):
        """When profile.enable_rag=True (default), gather_context must
        build an EnhancedPromptResult from the supplied sources."""
        svc = _build_memory_service()

        profile = _FakeProfile(enable_rag=True)
        svc._load_user_profile = AsyncMock(return_value=profile)

        sources = [_make_source("should-be-used")]

        with (
            patch("atlas_brain.memory.service.db_settings") as mock_db,
            patch("atlas_brain.memory.service.settings") as mock_settings,
        ):
            mock_db.enabled = True
            mock_settings.memory.enabled = True
            mock_settings.memory.context_results = 3

            ctx = await svc.gather_context(
                query="Tell me about the weather",
                session_id=None,
                user_id="00000000-0000-0000-0000-000000000001",
                include_rag=True,
                pre_fetched_sources=sources,
                include_history=False,
                include_physical=False,
            )

        assert ctx.rag_context_used is True, (
            "rag_context_used should be True when profile allows RAG"
        )
        assert ctx.rag_result is not None, (
            "rag_result should be populated when profile allows RAG"
        )
        assert len(ctx.rag_result.sources) == 1
        assert ctx.rag_result.sources[0].name == "should-be-used"

    @pytest.mark.asyncio
    async def test_empty_pre_fetched_sources_does_not_trigger_fresh_search(self):
        """When pre_fetched_sources=[] (empty list, NOT None), gather_context
        should use the empty list and NOT fall through to a fresh RAG search."""
        svc = _build_memory_service()

        profile = _FakeProfile(enable_rag=True)
        svc._load_user_profile = AsyncMock(return_value=profile)

        with (
            patch("atlas_brain.memory.service.db_settings") as mock_db,
            patch("atlas_brain.memory.service.settings") as mock_settings,
        ):
            mock_db.enabled = True
            mock_settings.memory.enabled = True
            mock_settings.memory.context_results = 3

            ctx = await svc.gather_context(
                query="Hello there",
                session_id=None,
                user_id="00000000-0000-0000-0000-000000000001",
                include_rag=True,
                pre_fetched_sources=[],  # explicit empty
                include_history=False,
                include_physical=False,
            )

        # context_used should be False because list is empty
        assert ctx.rag_context_used is False
        assert ctx.rag_result is not None  # result exists, just empty
        assert ctx.rag_result.sources == []
        # Fresh RAG search must NOT have been attempted
        svc._rag_client.enhance_prompt.assert_not_called()


# ===================================================================
# Fix 2 tests -- retrieve_memory returns retrieved_sources=[] on timeout
# ===================================================================

class TestRetrieveMemoryTimeout:
    """When RAG search times out in retrieve_memory, the Command must
    include `retrieved_sources: []` so downstream never retries."""

    @pytest.mark.asyncio
    async def test_timeout_returns_empty_retrieved_sources(self):
        """Simulate asyncio.TimeoutError from client.search() and verify
        the returned Command has retrieved_sources=[]."""
        from atlas_brain.agents.graphs.atlas import retrieve_memory

        # Build a state dict that will enter the RAG search branch
        state: dict = {
            "input_text": "What did we discuss yesterday?",
            "action_type": "conversation",
            "confidence": 0.95,
        }

        # Mock settings to enable memory
        mock_settings = MagicMock()
        mock_settings.memory.enabled = True
        mock_settings.memory.retrieve_context = True
        mock_settings.memory.context_results = 3
        mock_settings.memory.context_timeout = 0.1
        mock_settings.intent_router.conversation_confidence_threshold = 0.7

        # Mock query classifier to say "use RAG"
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = SimpleNamespace(
            use_rag=True, category="knowledge", reason="test", confidence=1.0,
        )
        mock_get_classifier = MagicMock(return_value=mock_classifier)

        # Mock RAG client whose search() raises TimeoutError
        mock_client = MagicMock()
        mock_client.search = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_get_rag = MagicMock(return_value=mock_client)

        # settings is imported locally inside retrieve_memory via
        # `from ...config import settings`, so patch the canonical location
        with (
            patch("atlas_brain.config.settings", mock_settings),
            patch(
                "atlas_brain.memory.query_classifier.get_query_classifier",
                mock_get_classifier,
            ),
            patch(
                "atlas_brain.memory.rag_client.get_rag_client",
                mock_get_rag,
            ),
        ):
            cmd = await retrieve_memory(state)

        # The Command must carry an update dict
        assert hasattr(cmd, "update"), "Command should have an update dict"
        assert cmd.update is not None, "update must not be None on timeout"
        assert "retrieved_sources" in cmd.update, (
            "retrieved_sources must be set on timeout"
        )
        assert cmd.update["retrieved_sources"] == [], (
            "retrieved_sources must be an empty list on timeout"
        )
        # Should still route somewhere
        assert cmd.goto in ("parse", "respond"), (
            f"Unexpected goto={cmd.goto!r}"
        )

    @pytest.mark.asyncio
    async def test_generic_exception_also_returns_empty_sources(self):
        """Even a generic Exception (not just TimeoutError) must produce
        retrieved_sources=[]."""
        from atlas_brain.agents.graphs.atlas import retrieve_memory

        state: dict = {
            "input_text": "What did we discuss yesterday?",
            "action_type": "conversation",
            "confidence": 0.95,
        }

        mock_settings = MagicMock()
        mock_settings.memory.enabled = True
        mock_settings.memory.retrieve_context = True
        mock_settings.memory.context_results = 3
        mock_settings.memory.context_timeout = 0.1
        mock_settings.intent_router.conversation_confidence_threshold = 0.7

        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = SimpleNamespace(
            use_rag=True, category="knowledge", reason="test", confidence=1.0,
        )
        mock_get_classifier = MagicMock(return_value=mock_classifier)

        mock_client = MagicMock()
        mock_client.search = AsyncMock(
            side_effect=ConnectionError("RAG service unreachable"),
        )
        mock_get_rag = MagicMock(return_value=mock_client)

        with (
            patch("atlas_brain.config.settings", mock_settings),
            patch(
                "atlas_brain.memory.query_classifier.get_query_classifier",
                mock_get_classifier,
            ),
            patch(
                "atlas_brain.memory.rag_client.get_rag_client",
                mock_get_rag,
            ),
        ):
            cmd = await retrieve_memory(state)

        assert cmd.update is not None
        assert cmd.update["retrieved_sources"] == []
        assert cmd.goto in ("parse", "respond")
