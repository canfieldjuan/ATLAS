"""Tests for the correction feedback stash/pop/downvote cycle in FeedbackService."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from atlas_brain.memory.feedback import FeedbackContext, FeedbackService, SourceCitation

# Path to the feedback repo getter used in local imports inside FeedbackService
_REPO_PATCH_PATH = "atlas_brain.storage.repositories.feedback.get_feedback_repo"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source(uuid_str="src-001", fact="the sky is blue", confidence=0.9):
    """Create a SimpleNamespace source object mimicking a RAG result."""
    return SimpleNamespace(uuid=uuid_str, fact=fact, confidence=confidence)


def _make_session_id_str() -> str:
    """Return a valid UUID string suitable for track_sources session_id."""
    return str(uuid4())


def _make_service_db_disabled():
    """Create a FeedbackService with DB disabled (no DB calls)."""
    with patch("atlas_brain.memory.feedback.db_settings") as mock_db:
        mock_db.enabled = False
        svc = FeedbackService()
    return svc


def _make_service_db_enabled():
    """Create a FeedbackService with DB enabled."""
    with patch("atlas_brain.memory.feedback.db_settings") as mock_db:
        mock_db.enabled = True
        svc = FeedbackService()
    return svc


# ---------------------------------------------------------------------------
# TestStashPopBasics
# ---------------------------------------------------------------------------


class TestStashPopBasics:
    """Test basic stash/pop operations on the in-memory session cache."""

    def test_stash_then_pop_returns_ids(self):
        svc = _make_service_db_disabled()
        uuid1, uuid2 = uuid4(), uuid4()

        svc.stash_session_usage("s1", [uuid1, uuid2])
        result = svc.pop_session_usage("s1")

        assert result == [uuid1, uuid2]

    def test_pop_after_pop_returns_empty(self):
        """Pop consumes the stashed IDs; second pop returns empty list."""
        svc = _make_service_db_disabled()
        uuid1 = uuid4()

        svc.stash_session_usage("s1", [uuid1])
        svc.pop_session_usage("s1")  # consume
        result = svc.pop_session_usage("s1")

        assert result == []

    def test_pop_nonexistent_session_returns_empty(self):
        svc = _make_service_db_disabled()
        result = svc.pop_session_usage("nonexistent")
        assert result == []

    def test_stash_empty_list_is_noop(self):
        """Stashing an empty list should not create an entry."""
        svc = _make_service_db_disabled()
        svc.stash_session_usage("s1", [])

        # Nothing was stored, so pop returns empty
        assert svc.pop_session_usage("s1") == []
        assert "s1" not in svc._session_usage


# ---------------------------------------------------------------------------
# TestStashOverwrites
# ---------------------------------------------------------------------------


class TestStashOverwrites:
    """Test that re-stashing for the same session overwrites the previous value."""

    def test_latest_stash_wins(self):
        svc = _make_service_db_disabled()
        old_ids = [uuid4(), uuid4()]
        new_ids = [uuid4()]

        svc.stash_session_usage("s1", old_ids)
        svc.stash_session_usage("s1", new_ids)

        result = svc.pop_session_usage("s1")
        assert result == new_ids

    def test_overwrite_does_not_accumulate(self):
        """After overwrite, only the latest IDs are present, not a merge."""
        svc = _make_service_db_disabled()
        id1 = uuid4()
        id2 = uuid4()

        svc.stash_session_usage("s1", [id1])
        svc.stash_session_usage("s1", [id2])

        result = svc.pop_session_usage("s1")
        assert result == [id2]
        assert id1 not in result


# ---------------------------------------------------------------------------
# TestCacheEviction
# ---------------------------------------------------------------------------


class TestCacheEviction:
    """Test eviction when the stash exceeds _STASH_MAX_SESSIONS."""

    def test_eviction_on_exceeding_max(self):
        svc = _make_service_db_disabled()
        # Temporarily lower the limit to make the test practical
        svc._STASH_MAX_SESSIONS = 10
        svc._STASH_EVICT_COUNT = 5

        # Fill the cache to exactly the max
        for i in range(10):
            svc.stash_session_usage(f"sess-{i}", [uuid4()])

        # All 10 should be present
        assert len(svc._session_usage) == 10

        # Adding one more triggers eviction of the oldest 5
        svc.stash_session_usage("sess-10", [uuid4()])

        # After eviction: 10 + 1 - 5 = 6 remain
        assert len(svc._session_usage) == 6

        # The oldest sessions (0-4) should have been evicted
        for i in range(5):
            assert svc.pop_session_usage(f"sess-{i}") == []

        # The newer sessions should still be present
        for i in range(5, 11):
            assert len(svc.pop_session_usage(f"sess-{i}")) == 1

    def test_eviction_preserves_newest_entries(self):
        svc = _make_service_db_disabled()
        svc._STASH_MAX_SESSIONS = 5
        svc._STASH_EVICT_COUNT = 3

        newest_id = uuid4()
        for i in range(5):
            svc.stash_session_usage(f"s-{i}", [uuid4()])

        # Trigger eviction
        svc.stash_session_usage("s-newest", [newest_id])

        # The newest entry must survive
        assert svc.pop_session_usage("s-newest") == [newest_id]

    def test_eviction_with_default_limits(self):
        """Verify eviction works at the real 1000 limit (add 1001 entries)."""
        svc = _make_service_db_disabled()
        # Use the real limits
        assert svc._STASH_MAX_SESSIONS == 1000
        assert svc._STASH_EVICT_COUNT == 500

        for i in range(1001):
            svc.stash_session_usage(f"sess-{i}", [uuid4()])

        # After eviction: 1001 - 500 = 501
        assert len(svc._session_usage) == 501

        # First session should have been evicted
        assert svc.pop_session_usage("sess-0") == []

        # Last session should still be present
        assert len(svc.pop_session_usage("sess-1000")) == 1


# ---------------------------------------------------------------------------
# TestFullCorrectionLoop
# ---------------------------------------------------------------------------


class TestFullCorrectionLoop:
    """Simulate the 2-turn correction feedback loop end-to-end."""

    @pytest.mark.asyncio
    async def test_two_turn_correction_cycle(self):
        """
        Simulate:
        1. Turn N: track_sources -> stash usage_ids
        2. Turn N+1: pop -> detect correction -> record_not_helpful
        """
        svc = _make_service_db_disabled()
        uuid1, uuid2 = uuid4(), uuid4()

        # Turn N: stash usage IDs (as if gathered from track_sources)
        svc.stash_session_usage("session-a", [uuid1, uuid2])

        # Turn N+1: pre-pop before graph runs
        prev_ids = svc.pop_session_usage("session-a")
        assert prev_ids == [uuid1, uuid2]

        # Simulate correction detection -> call record_not_helpful
        # Enable DB so the method actually tries to call the repo
        svc._enabled = True

        mock_repo = MagicMock()
        mock_repo.record_feedback = AsyncMock()

        with patch(
            _REPO_PATCH_PATH,
            return_value=mock_repo,
        ):
            await svc.record_not_helpful(prev_ids, "correction")

        # Verify record_feedback was called for each usage ID
        assert mock_repo.record_feedback.await_count == 2
        mock_repo.record_feedback.assert_any_await(
            uuid1, was_helpful=False, feedback_type="correction"
        )
        mock_repo.record_feedback.assert_any_await(
            uuid2, was_helpful=False, feedback_type="correction"
        )

    @pytest.mark.asyncio
    async def test_no_correction_no_downvote(self):
        """If no correction is detected, previously popped IDs are not downvoted."""
        svc = _make_service_db_disabled()
        uuid1 = uuid4()

        svc.stash_session_usage("session-b", [uuid1])
        prev_ids = svc.pop_session_usage("session-b")

        # If no correction detected, record_not_helpful is never called.
        # Verify the IDs were consumed and nothing remains.
        assert prev_ids == [uuid1]
        assert svc.pop_session_usage("session-b") == []

    @pytest.mark.asyncio
    async def test_stash_pop_with_record_helpful(self):
        """Verify record_helpful works in the same pattern (positive feedback)."""
        svc = _make_service_db_enabled()
        uuid1 = uuid4()

        svc.stash_session_usage("session-c", [uuid1])
        prev_ids = svc.pop_session_usage("session-c")

        mock_repo = MagicMock()
        mock_repo.record_feedback = AsyncMock()

        with patch(
            _REPO_PATCH_PATH,
            return_value=mock_repo,
        ):
            await svc.record_helpful(prev_ids, "implicit")

        mock_repo.record_feedback.assert_awaited_once_with(
            uuid1, was_helpful=True, feedback_type="implicit"
        )


# ---------------------------------------------------------------------------
# TestTrackSourcesDbDisabled
# ---------------------------------------------------------------------------


class TestTrackSourcesDbDisabled:
    """Test track_sources when the database is disabled."""

    @pytest.mark.asyncio
    async def test_populates_citations_without_db(self):
        svc = _make_service_db_disabled()
        sources = [
            _make_source("src-1", "the sky is blue", 0.9),
            _make_source("src-2", "water is wet", 0.85),
        ]

        ctx = await svc.track_sources(_make_session_id_str(), "what color is the sky?", sources)

        assert isinstance(ctx, FeedbackContext)
        assert ctx.has_sources()
        assert len(ctx.sources) == 2
        assert ctx.sources[0].fact == "the sky is blue"
        assert ctx.sources[0].source_id == "src-1"
        assert ctx.sources[0].confidence == 0.9
        assert ctx.sources[1].fact == "water is wet"
        # No usage_ids because DB is disabled
        assert ctx.usage_ids == []
        # Individual citations should have no usage_id
        assert ctx.sources[0].usage_id is None
        assert ctx.sources[1].usage_id is None

    @pytest.mark.asyncio
    async def test_empty_sources_returns_empty_context(self):
        svc = _make_service_db_disabled()
        ctx = await svc.track_sources(_make_session_id_str(), "hello", [])

        assert not ctx.has_sources()
        assert ctx.sources == []
        assert ctx.usage_ids == []

    @pytest.mark.asyncio
    async def test_dict_sources_supported(self):
        """Sources can be dicts instead of objects."""
        svc = _make_service_db_disabled()
        sources = [{"uuid": "src-d1", "fact": "grass is green", "confidence": 0.7}]

        ctx = await svc.track_sources(_make_session_id_str(), "what color is grass?", sources)

        assert len(ctx.sources) == 1
        assert ctx.sources[0].fact == "grass is green"
        assert ctx.sources[0].source_id == "src-d1"

    @pytest.mark.asyncio
    async def test_none_session_id(self):
        """track_sources handles None session_id gracefully."""
        svc = _make_service_db_disabled()
        ctx = await svc.track_sources(None, "test query", [_make_source()])

        assert ctx.session_id is None
        assert len(ctx.sources) == 1


# ---------------------------------------------------------------------------
# TestTrackSourcesDbEnabled
# ---------------------------------------------------------------------------


class TestTrackSourcesDbEnabled:
    """Test track_sources when the database is enabled."""

    @pytest.mark.asyncio
    async def test_records_usage_and_returns_ids(self):
        svc = _make_service_db_enabled()

        usage_id_1 = uuid4()
        usage_id_2 = uuid4()
        mock_usage_1 = SimpleNamespace(id=usage_id_1)
        mock_usage_2 = SimpleNamespace(id=usage_id_2)

        mock_repo = MagicMock()
        mock_repo.record_source_usage = AsyncMock(
            side_effect=[mock_usage_1, mock_usage_2]
        )

        sources = [
            _make_source("src-1", "the sky is blue", 0.9),
            _make_source("src-2", "water is wet", 0.85),
        ]

        with patch(
            _REPO_PATCH_PATH,
            return_value=mock_repo,
        ):
            ctx = await svc.track_sources(
                _make_session_id_str(),
                "what color is the sky?",
                sources,
            )

        assert ctx.has_sources()
        assert len(ctx.sources) == 2
        assert len(ctx.usage_ids) == 2
        assert ctx.usage_ids == [usage_id_1, usage_id_2]

        # Each citation should have the usage_id assigned
        assert ctx.sources[0].usage_id == usage_id_1
        assert ctx.sources[1].usage_id == usage_id_2

        # Verify the repo was called correctly
        assert mock_repo.record_source_usage.await_count == 2

    @pytest.mark.asyncio
    async def test_db_error_is_swallowed(self):
        """If the repo raises an exception, track_sources logs and continues."""
        svc = _make_service_db_enabled()

        mock_repo = MagicMock()
        mock_repo.record_source_usage = AsyncMock(
            side_effect=Exception("DB connection lost")
        )

        sources = [_make_source("src-err", "some fact", 0.5)]

        with patch(
            _REPO_PATCH_PATH,
            return_value=mock_repo,
        ):
            ctx = await svc.track_sources(_make_session_id_str(), "test", sources)

        # Source should NOT appear (exception was raised before appending)
        assert len(ctx.sources) == 0
        assert len(ctx.usage_ids) == 0


# ---------------------------------------------------------------------------
# TestRecordHelpful
# ---------------------------------------------------------------------------


class TestRecordHelpful:
    """Test record_helpful behavior."""

    @pytest.mark.asyncio
    async def test_empty_ids_is_noop(self):
        svc = _make_service_db_enabled()

        mock_repo = MagicMock()
        mock_repo.record_feedback = AsyncMock()

        with patch(
            _REPO_PATCH_PATH,
            return_value=mock_repo,
        ):
            await svc.record_helpful([], "explicit")

        mock_repo.record_feedback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_db_disabled_is_noop(self):
        svc = _make_service_db_disabled()

        mock_repo = MagicMock()
        mock_repo.record_feedback = AsyncMock()

        with patch(
            _REPO_PATCH_PATH,
            return_value=mock_repo,
        ):
            await svc.record_helpful([uuid4()], "explicit")

        mock_repo.record_feedback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_calls_repo_for_each_id(self):
        svc = _make_service_db_enabled()
        id1, id2 = uuid4(), uuid4()

        mock_repo = MagicMock()
        mock_repo.record_feedback = AsyncMock()

        with patch(
            _REPO_PATCH_PATH,
            return_value=mock_repo,
        ):
            await svc.record_helpful([id1, id2], "thumbs_up")

        assert mock_repo.record_feedback.await_count == 2
        mock_repo.record_feedback.assert_any_await(
            id1, was_helpful=True, feedback_type="thumbs_up"
        )
        mock_repo.record_feedback.assert_any_await(
            id2, was_helpful=True, feedback_type="thumbs_up"
        )

    @pytest.mark.asyncio
    async def test_db_error_is_swallowed(self):
        """Exceptions from repo.record_feedback are logged, not raised."""
        svc = _make_service_db_enabled()

        mock_repo = MagicMock()
        mock_repo.record_feedback = AsyncMock(side_effect=Exception("DB down"))

        with patch(
            _REPO_PATCH_PATH,
            return_value=mock_repo,
        ):
            # Should not raise
            await svc.record_helpful([uuid4()], "explicit")


# ---------------------------------------------------------------------------
# TestRecordNotHelpful
# ---------------------------------------------------------------------------


class TestRecordNotHelpful:
    """Test record_not_helpful behavior."""

    @pytest.mark.asyncio
    async def test_empty_ids_is_noop(self):
        svc = _make_service_db_enabled()

        mock_repo = MagicMock()
        mock_repo.record_feedback = AsyncMock()

        with patch(
            _REPO_PATCH_PATH,
            return_value=mock_repo,
        ):
            await svc.record_not_helpful([], "correction")

        mock_repo.record_feedback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_db_disabled_is_noop(self):
        svc = _make_service_db_disabled()

        mock_repo = MagicMock()
        mock_repo.record_feedback = AsyncMock()

        with patch(
            _REPO_PATCH_PATH,
            return_value=mock_repo,
        ):
            await svc.record_not_helpful([uuid4()], "correction")

        mock_repo.record_feedback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_calls_repo_for_each_id(self):
        svc = _make_service_db_enabled()
        id1, id2, id3 = uuid4(), uuid4(), uuid4()

        mock_repo = MagicMock()
        mock_repo.record_feedback = AsyncMock()

        with patch(
            _REPO_PATCH_PATH,
            return_value=mock_repo,
        ):
            await svc.record_not_helpful([id1, id2, id3], "correction")

        assert mock_repo.record_feedback.await_count == 3
        mock_repo.record_feedback.assert_any_await(
            id1, was_helpful=False, feedback_type="correction"
        )
        mock_repo.record_feedback.assert_any_await(
            id2, was_helpful=False, feedback_type="correction"
        )
        mock_repo.record_feedback.assert_any_await(
            id3, was_helpful=False, feedback_type="correction"
        )

    @pytest.mark.asyncio
    async def test_db_error_is_swallowed(self):
        """Exceptions from repo.record_feedback are logged, not raised."""
        svc = _make_service_db_enabled()

        mock_repo = MagicMock()
        mock_repo.record_feedback = AsyncMock(side_effect=Exception("timeout"))

        with patch(
            _REPO_PATCH_PATH,
            return_value=mock_repo,
        ):
            # Should not raise
            await svc.record_not_helpful([uuid4()], "correction")

    @pytest.mark.asyncio
    async def test_default_feedback_type(self):
        """Default feedback_type is 'implicit'."""
        svc = _make_service_db_enabled()
        uid = uuid4()

        mock_repo = MagicMock()
        mock_repo.record_feedback = AsyncMock()

        with patch(
            _REPO_PATCH_PATH,
            return_value=mock_repo,
        ):
            await svc.record_not_helpful([uid])

        mock_repo.record_feedback.assert_awaited_once_with(
            uid, was_helpful=False, feedback_type="implicit"
        )


# ---------------------------------------------------------------------------
# TestPartialDbFailure
# ---------------------------------------------------------------------------


class TestPartialDbFailure:
    """Test that track_sources handles per-source DB errors gracefully."""

    @pytest.mark.asyncio
    async def test_first_source_fails_second_succeeds(self):
        """If source 1 raises, source 2 should still be tracked."""
        svc = _make_service_db_enabled()

        usage_id_2 = uuid4()
        mock_usage_2 = SimpleNamespace(id=usage_id_2)

        mock_repo = MagicMock()
        mock_repo.record_source_usage = AsyncMock(
            side_effect=[Exception("transient error"), mock_usage_2]
        )

        sources = [
            _make_source("src-fail", "fact A", 0.9),
            _make_source("src-ok", "fact B", 0.8),
        ]

        with patch(_REPO_PATCH_PATH, return_value=mock_repo):
            ctx = await svc.track_sources(
                _make_session_id_str(), "test query", sources,
            )

        # Source 1 failed: not in citations or usage_ids
        # Source 2 succeeded: present in both
        assert len(ctx.sources) == 1
        assert ctx.sources[0].fact == "fact B"
        assert ctx.sources[0].usage_id == usage_id_2
        assert ctx.usage_ids == [usage_id_2]


# ---------------------------------------------------------------------------
# TestFormatCitations
# ---------------------------------------------------------------------------


class TestFormatCitations:
    """Test FeedbackService.format_citations() logic."""

    def test_empty_context(self):
        svc = _make_service_db_disabled()
        ctx = FeedbackContext()
        assert svc.format_citations(ctx) == ""

    def test_basic_formatting(self):
        svc = _make_service_db_disabled()
        ctx = FeedbackContext(sources=[
            SourceCitation(source_id="s1", fact="the sky is blue", confidence=0.9),
            SourceCitation(source_id="s2", fact="water is wet", confidence=0.8),
        ])
        result = svc.format_citations(ctx)
        assert "Sources:" in result
        assert "[1] the sky is blue" in result
        assert "[2] water is wet" in result

    def test_max_citations_limit(self):
        svc = _make_service_db_disabled()
        ctx = FeedbackContext(sources=[
            SourceCitation(source_id=f"s{i}", fact=f"fact {i}", confidence=0.9)
            for i in range(5)
        ])
        result = svc.format_citations(ctx, max_citations=2)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" not in result

    def test_min_confidence_filter(self):
        svc = _make_service_db_disabled()
        ctx = FeedbackContext(sources=[
            SourceCitation(source_id="s1", fact="high conf", confidence=0.9),
            SourceCitation(source_id="s2", fact="low conf", confidence=0.1),
        ])
        result = svc.format_citations(ctx, min_confidence=0.3)
        assert "high conf" in result
        assert "low conf" not in result

    def test_all_below_confidence_returns_empty(self):
        svc = _make_service_db_disabled()
        ctx = FeedbackContext(sources=[
            SourceCitation(source_id="s1", fact="low", confidence=0.1),
        ])
        assert svc.format_citations(ctx, min_confidence=0.5) == ""

    def test_fact_truncation_at_100_chars(self):
        svc = _make_service_db_disabled()
        long_fact = "x" * 150
        ctx = FeedbackContext(sources=[
            SourceCitation(source_id="s1", fact=long_fact, confidence=0.9),
        ])
        result = svc.format_citations(ctx)
        # Truncated to 97 chars + "..."
        assert "x" * 97 + "..." in result
        assert "x" * 98 not in result


# ---------------------------------------------------------------------------
# TestFormatInlineCitations
# ---------------------------------------------------------------------------


class TestFormatInlineCitations:
    """Test FeedbackService.format_inline_citations() logic."""

    def test_empty_context(self):
        svc = _make_service_db_disabled()
        ctx = FeedbackContext()
        assert svc.format_inline_citations(ctx) == []

    def test_filters_by_confidence(self):
        svc = _make_service_db_disabled()
        ctx = FeedbackContext(sources=[
            SourceCitation(source_id="s1", fact="a", confidence=0.9),
            SourceCitation(source_id="s2", fact="b", confidence=0.3),
            SourceCitation(source_id="s3", fact="c", confidence=0.7),
        ])
        markers = svc.format_inline_citations(ctx, min_confidence=0.5)
        assert markers == ["[1]", "[3]"]


# ---------------------------------------------------------------------------
# TestGetFeedbackServiceSingleton
# ---------------------------------------------------------------------------


class TestGetFeedbackServiceSingleton:
    """Test that get_feedback_service returns the same instance."""

    def test_returns_same_instance(self):
        import atlas_brain.memory.feedback as fb_module

        # Reset global to ensure clean test
        original = fb_module._feedback_service
        fb_module._feedback_service = None

        try:
            with patch("atlas_brain.memory.feedback.db_settings") as mock_db:
                mock_db.enabled = False
                svc1 = fb_module.get_feedback_service()
                svc2 = fb_module.get_feedback_service()
        finally:
            fb_module._feedback_service = original

        assert svc1 is svc2
