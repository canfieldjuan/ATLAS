"""
Tests for entity-focused graph traversal.

Covers:
1. LLM entity extraction via intent router
2. RAGClient.get_entity_edges()
3. Parallel search + traversal in retrieve_memory node
4. RAGClient.search_with_traversal() unit tests
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas_brain.memory.query_classifier import ClassificationResult, QueryClassifier
from atlas_brain.memory.rag_client import RAGClient, SearchResult, SearchSource
from atlas_brain.services.intent_router import IntentRouteResult


# ---------------------------------------------------------------------------
# 1. LLM entity extraction via intent router
# ---------------------------------------------------------------------------


class TestLLMEntityExtraction:
    """Tests for entity_name on IntentRouteResult from LLM fallback."""

    def test_entity_name_field_exists(self):
        result = IntentRouteResult(
            action_category="conversation",
            raw_label="conversation",
            confidence=0.99,
            entity_name="Juan",
        )
        assert result.entity_name == "Juan"

    def test_entity_name_defaults_to_none(self):
        result = IntentRouteResult(
            action_category="conversation",
            raw_label="conversation",
            confidence=0.99,
        )
        assert result.entity_name is None

    def test_entity_name_passed_to_state(self):
        """entity_name from route result should flow into state update."""
        result = IntentRouteResult(
            action_category="conversation",
            raw_label="conversation",
            confidence=0.99,
            entity_name="Effingham Office Maids",
        )
        assert result.entity_name == "Effingham Office Maids"


# ---------------------------------------------------------------------------
# 2. RAGClient.get_entity_edges()
# ---------------------------------------------------------------------------


class TestGetEntityEdges:
    """Tests for RAGClient.get_entity_edges()."""

    @pytest.fixture
    def mock_settings(self):
        with patch("atlas_brain.memory.rag_client.settings") as mock:
            mock.memory.enabled = True
            mock.memory.base_url = "http://localhost:8003"
            mock.memory.timeout = 10.0
            mock.memory.group_id = "test-group"
            yield mock

    @pytest.mark.asyncio
    async def test_returns_search_result_with_facts(self, mock_settings):
        client = RAGClient(base_url="http://localhost:8003")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "edges": [
                {
                    "uuid": "edge-1",
                    "name": "lives_at",
                    "fact": "Juan lives at 123 Main St",
                    "score": 0.9,
                    "created_at": "2025-01-01T00:00:00Z",
                    "expired_at": None,
                },
                {
                    "uuid": "edge-2",
                    "name": "prefers",
                    "fact": "Juan prefers dark mode",
                    "score": 0.8,
                },
            ],
        }

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.get_entity_edges("Juan")

        assert isinstance(result, SearchResult)
        assert len(result.facts) == 2
        assert result.facts[0].uuid == "edge-1"
        assert result.facts[0].fact == "Juan lives at 123 Main St"
        assert result.facts[0].confidence == 0.9
        assert result.facts[1].uuid == "edge-2"

        # Verify URL encoding
        mock_http.get.assert_called_once()
        call_args = mock_http.get.call_args
        assert "Juan" in call_args[0][0]

        await client.close()

    @pytest.mark.asyncio
    async def test_returns_empty_when_disabled(self, mock_settings):
        mock_settings.memory.enabled = False
        client = RAGClient()
        result = await client.get_entity_edges("Juan")
        assert result.facts == []

    @pytest.mark.asyncio
    async def test_handles_connection_error(self, mock_settings):
        import httpx

        client = RAGClient(base_url="http://localhost:8003")
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.get_entity_edges("Juan")
        assert result.facts == []

        await client.close()

    @pytest.mark.asyncio
    async def test_handles_http_error(self, mock_settings):
        import httpx

        client = RAGClient(base_url="http://localhost:8003")
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_resp,
        )

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.get_entity_edges("Juan")
        assert result.facts == []

        await client.close()

    @pytest.mark.asyncio
    async def test_respects_max_edges(self, mock_settings):
        client = RAGClient(base_url="http://localhost:8003")
        edges = [{"uuid": f"e-{i}", "name": "rel", "fact": f"fact {i}"} for i in range(30)]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"edges": edges}

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.get_entity_edges("Juan", max_edges=5)
        assert len(result.facts) == 5

        await client.close()

    @pytest.mark.asyncio
    async def test_url_encodes_entity_name(self, mock_settings):
        client = RAGClient(base_url="http://localhost:8003")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"edges": []}

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.is_closed = False
        client._client = mock_http

        await client.get_entity_edges("Living Room")

        call_url = mock_http.get.call_args[0][0]
        assert "Living%20Room" in call_url

        await client.close()


# ---------------------------------------------------------------------------
# 3. Parallel retrieval in retrieve_memory
# ---------------------------------------------------------------------------


def _make_source(uuid: str, fact: str) -> SearchSource:
    return SearchSource(uuid=uuid, name="test", fact=fact, confidence=0.9)


def _patch_config():
    """Patch atlas_brain.config.settings for retrieve_memory tests."""
    return patch("atlas_brain.config.settings")


def _configure_memory_settings(mock_cfg):
    """Set standard memory settings on a mocked config."""
    mock_cfg.memory.enabled = True
    mock_cfg.memory.retrieve_context = True
    mock_cfg.memory.context_results = 5
    mock_cfg.memory.context_timeout = 5.0
    mock_cfg.intent_router.conversation_confidence_threshold = 0.8


class TestParallelRetrieval:
    """Tests for parallel search + traversal in retrieve_memory node."""

    def _make_state(self, input_text: str, action_type: str = "conversation",
                    entity_name: str = None):
        return {
            "input_text": input_text,
            "action_type": action_type,
            "confidence": 0.9,
            "session_id": "test-session",
            "runtime_context": {},
            "entity_name": entity_name,
        }

    @pytest.mark.asyncio
    async def test_entity_query_calls_search_with_traversal(self):
        """When entity_name is in state, search_with_traversal is called with it."""
        merged_facts = [
            _make_source("s1", "Juan is a developer"),
            _make_source("t1", "Juan lives in Texas"),
        ]
        mock_swt = AsyncMock(return_value=SearchResult(facts=merged_facts))

        with _patch_config() as mock_cfg:
            _configure_memory_settings(mock_cfg)

            mock_client = MagicMock()
            mock_client.search_with_traversal = mock_swt

            with patch(
                "atlas_brain.memory.rag_client.get_rag_client",
                return_value=mock_client,
            ), patch(
                "atlas_brain.memory.query_classifier.get_query_classifier",
            ) as mock_clf:
                mock_clf.return_value.classify.return_value = ClassificationResult(
                    use_rag=True,
                    category="knowledge",
                    reason="test",
                )

                from atlas_brain.agents.graphs.atlas import retrieve_memory

                state = self._make_state("What do you know about Juan?", entity_name="Juan")
                result = await retrieve_memory(state)

                mock_swt.assert_called_once()
                call_kwargs = mock_swt.call_args
                assert call_kwargs.kwargs.get("entity_name") == "Juan"

                sources = result.update.get("retrieved_sources", [])
                assert len(sources) == 2
                uuids = {s.uuid for s in sources}
                assert "s1" in uuids
                assert "t1" in uuids

    @pytest.mark.asyncio
    async def test_no_entity_passes_none(self):
        """When no entity_name, search_with_traversal is called with entity_name=None."""
        search_facts = [_make_source("s1", "Some fact")]
        mock_swt = AsyncMock(return_value=SearchResult(facts=search_facts))

        with _patch_config() as mock_cfg:
            _configure_memory_settings(mock_cfg)

            mock_client = MagicMock()
            mock_client.search_with_traversal = mock_swt

            with patch(
                "atlas_brain.memory.rag_client.get_rag_client",
                return_value=mock_client,
            ), patch(
                "atlas_brain.memory.query_classifier.get_query_classifier",
            ) as mock_clf:
                mock_clf.return_value.classify.return_value = ClassificationResult(
                    use_rag=True,
                    category="knowledge",
                    reason="test",
                )

                from atlas_brain.agents.graphs.atlas import retrieve_memory

                state = self._make_state("How does weather work?")
                result = await retrieve_memory(state)

                mock_swt.assert_called_once()
                call_kwargs = mock_swt.call_args
                assert call_kwargs.kwargs.get("entity_name") is None

    @pytest.mark.asyncio
    async def test_retrieve_memory_passes_results_through(self):
        """retrieve_memory stores search_with_traversal results in retrieved_sources."""
        facts = [
            _make_source("s1", "Juan is a developer"),
            _make_source("t2", "Juan lives in Texas"),
        ]
        mock_swt = AsyncMock(return_value=SearchResult(facts=facts))

        with _patch_config() as mock_cfg:
            _configure_memory_settings(mock_cfg)

            mock_client = MagicMock()
            mock_client.search_with_traversal = mock_swt

            with patch(
                "atlas_brain.memory.rag_client.get_rag_client",
                return_value=mock_client,
            ), patch(
                "atlas_brain.memory.query_classifier.get_query_classifier",
            ) as mock_clf:
                mock_clf.return_value.classify.return_value = ClassificationResult(
                    use_rag=True,
                    category="knowledge",
                    reason="test",
                )

                from atlas_brain.agents.graphs.atlas import retrieve_memory

                state = self._make_state("What do you know about Juan?", entity_name="Juan")
                result = await retrieve_memory(state)

                sources = result.update.get("retrieved_sources", [])
                assert len(sources) == 2

    @pytest.mark.asyncio
    async def test_search_with_traversal_failure_returns_empty(self):
        """If search_with_traversal raises, retrieve_memory returns empty sources."""
        mock_swt = AsyncMock(side_effect=Exception("search boom"))

        with _patch_config() as mock_cfg:
            _configure_memory_settings(mock_cfg)

            mock_client = MagicMock()
            mock_client.search_with_traversal = mock_swt

            with patch(
                "atlas_brain.memory.rag_client.get_rag_client",
                return_value=mock_client,
            ), patch(
                "atlas_brain.memory.query_classifier.get_query_classifier",
            ) as mock_clf:
                mock_clf.return_value.classify.return_value = ClassificationResult(
                    use_rag=True,
                    category="knowledge",
                    reason="test",
                )

                from atlas_brain.agents.graphs.atlas import retrieve_memory

                state = self._make_state("What do you know about Juan?", entity_name="Juan")
                result = await retrieve_memory(state)

                sources = result.update.get("retrieved_sources", [])
                assert sources == []


# ---------------------------------------------------------------------------
# 4. RAGClient.search_with_traversal() unit tests
# ---------------------------------------------------------------------------


class TestSearchWithTraversal:
    """Tests for the search_with_traversal method on RAGClient."""

    @pytest.mark.asyncio
    async def test_entity_runs_parallel_and_merges(self):
        """With entity_name, both search and traversal run and merge."""
        search_facts = [_make_source("s1", "Juan is a developer")]
        traversal_facts = [_make_source("t1", "Juan lives in Texas")]

        client = RAGClient(base_url="http://fake:8001")
        client.search = AsyncMock(
            return_value=SearchResult(facts=search_facts),
        )
        client.get_entity_edges = AsyncMock(
            return_value=SearchResult(facts=traversal_facts),
        )

        with patch("atlas_brain.memory.rag_client.settings") as mock_cfg:
            mock_cfg.memory.enabled = True
            mock_cfg.memory.context_timeout = 5.0

            result = await client.search_with_traversal(
                query="What do you know about Juan?",
                entity_name="Juan",
                max_facts=5,
            )

        client.search.assert_called_once()
        client.get_entity_edges.assert_called_once_with("Juan")
        assert len(result.facts) == 2
        uuids = {f.uuid for f in result.facts}
        assert "s1" in uuids
        assert "t1" in uuids

    @pytest.mark.asyncio
    async def test_no_entity_only_searches(self):
        """Without entity_name, only search runs."""
        search_facts = [_make_source("s1", "Some fact")]

        client = RAGClient(base_url="http://fake:8001")
        client.search = AsyncMock(
            return_value=SearchResult(facts=search_facts),
        )
        client.get_entity_edges = AsyncMock()

        with patch("atlas_brain.memory.rag_client.settings") as mock_cfg:
            mock_cfg.memory.enabled = True
            mock_cfg.memory.context_timeout = 5.0

            result = await client.search_with_traversal(
                query="How does weather work?",
                entity_name=None,
                max_facts=5,
            )

        client.search.assert_called_once()
        client.get_entity_edges.assert_not_called()
        assert len(result.facts) == 1

    @pytest.mark.asyncio
    async def test_deduplication_by_uuid(self):
        """Duplicate UUIDs across search and traversal are deduplicated."""
        shared = "shared-1"
        search_facts = [_make_source(shared, "Juan is a developer")]
        traversal_facts = [
            _make_source(shared, "Juan is a developer"),
            _make_source("t2", "Juan lives in Texas"),
        ]

        client = RAGClient(base_url="http://fake:8001")
        client.search = AsyncMock(
            return_value=SearchResult(facts=search_facts),
        )
        client.get_entity_edges = AsyncMock(
            return_value=SearchResult(facts=traversal_facts),
        )

        with patch("atlas_brain.memory.rag_client.settings") as mock_cfg:
            mock_cfg.memory.enabled = True
            mock_cfg.memory.context_timeout = 5.0

            result = await client.search_with_traversal(
                query="Tell me about Juan",
                entity_name="Juan",
                max_facts=5,
            )

        assert len(result.facts) == 2
        uuids = [f.uuid for f in result.facts]
        assert uuids.count(shared) == 1

    @pytest.mark.asyncio
    async def test_traversal_failure_preserves_search(self):
        """If traversal raises, search results are still returned."""
        search_facts = [_make_source("s1", "Juan is a developer")]

        client = RAGClient(base_url="http://fake:8001")
        client.search = AsyncMock(
            return_value=SearchResult(facts=search_facts),
        )
        client.get_entity_edges = AsyncMock(
            side_effect=Exception("traversal boom"),
        )

        with patch("atlas_brain.memory.rag_client.settings") as mock_cfg:
            mock_cfg.memory.enabled = True
            mock_cfg.memory.context_timeout = 5.0

            result = await client.search_with_traversal(
                query="Tell me about Juan",
                entity_name="Juan",
                max_facts=5,
            )

        assert len(result.facts) == 1
        assert result.facts[0].uuid == "s1"

    @pytest.mark.asyncio
    async def test_search_failure_preserves_traversal(self):
        """If search raises, traversal results are still returned."""
        traversal_facts = [_make_source("t1", "Juan lives in Texas")]

        client = RAGClient(base_url="http://fake:8001")
        client.search = AsyncMock(
            side_effect=Exception("search boom"),
        )
        client.get_entity_edges = AsyncMock(
            return_value=SearchResult(facts=traversal_facts),
        )

        with patch("atlas_brain.memory.rag_client.settings") as mock_cfg:
            mock_cfg.memory.enabled = True
            mock_cfg.memory.context_timeout = 5.0

            result = await client.search_with_traversal(
                query="Tell me about Juan",
                entity_name="Juan",
                max_facts=5,
            )

        assert len(result.facts) == 1
        assert result.facts[0].uuid == "t1"

    @pytest.mark.asyncio
    async def test_memory_disabled_returns_empty(self):
        """When memory is disabled, returns empty SearchResult."""
        client = RAGClient(base_url="http://fake:8001")
        client.search = AsyncMock()

        with patch("atlas_brain.memory.rag_client.settings") as mock_cfg:
            mock_cfg.memory.enabled = False

            result = await client.search_with_traversal(
                query="Tell me about Juan",
                entity_name="Juan",
            )

        client.search.assert_not_called()
        assert result.facts == []
