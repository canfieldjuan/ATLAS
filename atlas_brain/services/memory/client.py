"""
Memory client for atlas-memory (graphiti-wrapper).

Provides conversation and knowledge storage via the GraphRAG API.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import httpx

from ...config import settings as app_settings
from ...memory.query_classifier import get_query_classifier

logger = logging.getLogger("atlas.services.memory")


@dataclass
class SearchResult:
    """A search result from the knowledge graph."""

    uuid: str
    name: str
    fact: str
    score: float
    source_description: Optional[str] = None
    source_node: Optional[dict] = None
    target_node: Optional[dict] = None


@dataclass
class EnhancedSearchResult:
    """Result from enhanced search with processing metadata."""

    results: list
    skipped: bool = False
    skip_reason: Optional[str] = None
    query_expanded: bool = False
    original_query: str = ""
    search_query: str = ""


@dataclass
class EpisodeResult:
    """Result from adding an episode."""

    episode_id: str
    entities_created: int = 0
    relations_created: int = 0


class MemoryClient:
    """
    Client for atlas-memory GraphRAG service.

    Handles storing conversations and retrieving relevant context.
    """

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def config(self):
        """Get memory config from app settings."""
        return app_settings.memory

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if the memory service is available."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning("Memory service health check failed: %s", e)
            return False

    async def add_conversation_turn(
        self,
        role: str,
        content: str,
        session_id: str,
        speaker_name: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[EpisodeResult]:
        """
        Store a conversation turn in the knowledge graph.

        Args:
            role: "user" or "assistant"
            content: The message content
            session_id: Session identifier
            speaker_name: Optional speaker name
            timestamp: Optional timestamp (defaults to now)

        Returns:
            EpisodeResult if successful, None otherwise
        """
        try:
            client = await self._get_client()

            ts = timestamp or datetime.utcnow()
            name = f"{role}:{session_id[:8]}"
            source = f"atlas-voice-{role}"
            if speaker_name:
                source = f"{source}:{speaker_name}"

            payload = {
                "name": name,
                "episode_body": content,
                "source_description": source,
                "reference_time": ts.isoformat() + "Z",
                "group_id": self.config.group_id,
                "is_historical": False,
                "data_source_type": "current",
            }

            response = await client.post("/episodes", json=payload)
            response.raise_for_status()

            data = response.json()
            return EpisodeResult(
                episode_id=data.get("episode_id", ""),
                entities_created=data.get("entities_created", 0),
                relations_created=data.get("relations_created", 0),
            )

        except Exception as e:
            logger.error("Failed to add conversation turn: %s", e)
            return None

    async def search(
        self,
        query: str,
        num_results: int = 5,
        group_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Search the knowledge graph for relevant context.

        Args:
            query: Search query
            num_results: Maximum results to return
            group_id: Optional group ID filter

        Returns:
            List of search results
        """
        try:
            client = await self._get_client()

            params = {
                "query": query,
                "group_ids": group_id or self.config.group_id,
                "num_results": num_results,
            }

            response = await client.get("/search", params=params)
            response.raise_for_status()

            data = response.json()
            results = []

            for edge in data.get("edges", []):
                results.append(SearchResult(
                    uuid=edge.get("uuid", ""),
                    name=edge.get("name", ""),
                    fact=edge.get("fact", ""),
                    score=edge.get("score", 0.0),
                    source_description=edge.get("source_description"),
                    source_node=edge.get("source_node"),
                    target_node=edge.get("target_node"),
                ))

            return results

        except Exception as e:
            logger.error("Failed to search memory: %s", e)
            return []

    async def enhanced_search(
        self,
        query: str,
        num_results: int = 5,
        group_id: Optional[str] = None,
        use_expansion: bool = True,
        use_reranking: bool = True,
        use_deduplication: bool = True,
    ) -> EnhancedSearchResult:
        """
        Enhanced search with query processing pipeline.

        Uses classification, expansion, reranking, and deduplication.

        Args:
            query: Search query
            num_results: Maximum results to return
            group_id: Optional group ID filter
            use_expansion: Enable query expansion
            use_reranking: Enable heuristic reranking
            use_deduplication: Enable content deduplication

        Returns:
            EnhancedSearchResult with results and processing metadata
        """
        try:
            client = await self._get_client()

            gid = group_id or self.config.group_id
            group_ids = [gid] if gid else None

            payload = {
                "query": query,
                "group_ids": group_ids,
                "num_results": num_results,
                "use_expansion": use_expansion,
                "use_reranking": use_reranking,
                "use_deduplication": use_deduplication,
            }

            response = await client.post("/search/enhanced", json=payload)
            response.raise_for_status()

            data = response.json()
            processing = data.get("processing", {})

            results = []
            for edge in data.get("edges", []):
                results.append(SearchResult(
                    uuid=edge.get("uuid", ""),
                    name=edge.get("name", ""),
                    fact=edge.get("fact", ""),
                    score=edge.get("score", 0.0),
                    source_description=edge.get("source_description"),
                    source_node=edge.get("source_node"),
                    target_node=edge.get("target_node"),
                ))

            return EnhancedSearchResult(
                results=results,
                skipped=processing.get("skipped", False),
                skip_reason=processing.get("skip_reason"),
                query_expanded=processing.get("query_expanded", False),
                original_query=processing.get("original_query", query),
                search_query=processing.get("search_query", query),
            )

        except Exception as e:
            logger.error("Failed to enhanced search: %s", e)
            return EnhancedSearchResult(results=[], original_query=query)

    async def get_context_for_query(
        self,
        query: str,
        num_results: int = 3,
        use_enhanced: bool = False,
    ) -> str:
        """
        Get formatted context string for a query.

        Uses query classification to skip RAG for device commands
        and simple queries, improving latency.

        Args:
            query: User query
            num_results: Number of results to include
            use_enhanced: Use enhanced search (currently disabled)

        Returns:
            Formatted context string
        """
        # Classify the query first
        classifier = get_query_classifier()
        classification = classifier.classify(query)

        if not classification.use_rag:
            logger.debug(
                "Skipping RAG (category=%s): %s",
                classification.category,
                query[:50],
            )
            return ""

        # Use basic search (enhanced search endpoint has issues)
        results = await self.search(query, num_results=num_results)

        if not results:
            return ""

        context_parts = []
        for result in results:
            if result.fact:
                context_parts.append(f"- {result.fact}")

        if not context_parts:
            return ""

        return "Relevant context:\n" + "\n".join(context_parts)

    async def add_fact(
        self,
        fact: str,
        source: str = "atlas-learned",
        timestamp: Optional[datetime] = None,
    ) -> Optional[EpisodeResult]:
        """
        Store a learned fact in the knowledge graph.

        Args:
            fact: The fact to store
            source: Source description
            timestamp: Optional timestamp

        Returns:
            EpisodeResult if successful
        """
        try:
            client = await self._get_client()

            ts = timestamp or datetime.utcnow()

            payload = {
                "name": f"fact:{ts.strftime('%Y%m%d%H%M%S')}",
                "episode_body": fact,
                "source_description": source,
                "reference_time": ts.isoformat() + "Z",
                "group_id": self.config.group_id,
                "is_historical": False,
                "data_source_type": "current",
            }

            response = await client.post("/episodes", json=payload)
            response.raise_for_status()

            data = response.json()
            return EpisodeResult(
                episode_id=data.get("episode_id", ""),
                entities_created=data.get("entities_created", 0),
                relations_created=data.get("relations_created", 0),
            )

        except Exception as e:
            logger.error("Failed to add fact: %s", e)
            return None


_memory_client: Optional[MemoryClient] = None


def get_memory_client() -> MemoryClient:
    """Get or create the global memory client."""
    global _memory_client
    if _memory_client is None:
        _memory_client = MemoryClient()
    return _memory_client
