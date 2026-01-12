"""
RAG Client for graphiti-wrapper integration.

Provides async HTTP client for the GraphRAG service.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx

from ..config import settings

logger = logging.getLogger("atlas.memory.rag_client")


@dataclass
class SearchSource:
    """A source from RAG search results."""

    entity: str
    relation: str
    fact: str
    confidence: float
    source_description: str = ""


@dataclass
class EnhancedPromptResult:
    """Result from prompt enhancement."""

    prompt: str
    context_used: bool
    sources: list[SearchSource] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """Result from RAG search."""

    edges: list[dict] = field(default_factory=list)
    nodes: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class RAGClient:
    """
    Async client for graphiti-wrapper RAG service.

    Connects to the GraphRAG service for:
    - Searching the knowledge graph
    - Enhancing prompts with document context
    - Adding conversation episodes
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self._base_url = base_url or settings.memory.base_url
        self._timeout = timeout or settings.memory.timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create httpx async client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if the RAG service is healthy."""
        try:
            client = await self._get_client()
            resp = await client.get("/health")
            if resp.status_code == 200:
                data = resp.json()
                return data.get("status") == "healthy"
            return False
        except Exception as e:
            logger.debug("RAG health check failed: %s", e)
            return False

    async def search(
        self,
        query: str,
        group_id: Optional[str] = None,
        num_results: int = 5,
    ) -> SearchResult:
        """
        Search the knowledge graph for relevant facts.

        Args:
            query: Search query
            group_id: Group ID to search within
            num_results: Maximum number of results

        Returns:
            SearchResult with edges and nodes
        """
        if not settings.memory.enabled:
            return SearchResult()

        gid = group_id or settings.memory.group_id

        try:
            client = await self._get_client()
            params = {
                "query": query,
                "group_ids": gid,
                "num_results": num_results,
            }

            resp = await client.get("/search", params=params)

            if resp.status_code != 200:
                logger.warning("RAG search failed (%d): %s", resp.status_code, resp.text[:200])
                return SearchResult()

            data = resp.json()
            return SearchResult(
                edges=data.get("edges", []),
                nodes=data.get("nodes", []),
            )

        except httpx.RequestError as e:
            logger.warning("RAG search connection error: %s", e)
            return SearchResult()
        except Exception as e:
            logger.error("RAG search error: %s", e)
            return SearchResult()

    async def enhance_prompt(
        self,
        query: str,
        group_id: Optional[str] = None,
        min_confidence: float = 0.3,
        max_sources: int = 5,
        max_context_length: int = 4000,
        compress_context: bool = False,
    ) -> EnhancedPromptResult:
        """
        Enhance a user prompt with document context.

        Args:
            query: User query to enhance
            group_id: Group ID to search within
            min_confidence: Minimum confidence threshold
            max_sources: Maximum number of sources to include
            max_context_length: Maximum context length in characters
            compress_context: Enable context compression

        Returns:
            EnhancedPromptResult with enhanced prompt and sources
        """
        if not settings.memory.enabled or not settings.memory.retrieve_context:
            return EnhancedPromptResult(prompt=query, context_used=False)

        gid = group_id or settings.memory.group_id

        try:
            client = await self._get_client()
            payload = {
                "query": query,
                "group_ids": gid,
                "min_confidence": min_confidence,
                "max_sources": max_sources,
                "max_context_length": max_context_length,
                "compress_context": compress_context,
                "include_metadata": True,
            }

            resp = await client.post("/enhance", json=payload)

            if resp.status_code != 200:
                logger.warning("RAG enhance failed (%d): %s", resp.status_code, resp.text[:200])
                return EnhancedPromptResult(prompt=query, context_used=False)

            data = resp.json()

            sources = []
            for s in data.get("sources", []):
                sources.append(SearchSource(
                    entity=s.get("entity", ""),
                    relation=s.get("relation", ""),
                    fact=s.get("fact", ""),
                    confidence=s.get("confidence", 0.0),
                    source_description=s.get("source_description", ""),
                ))

            return EnhancedPromptResult(
                prompt=data.get("prompt", query),
                context_used=data.get("context_used", False),
                sources=sources,
                metadata=data.get("metadata", {}),
            )

        except httpx.RequestError as e:
            logger.warning("RAG enhance connection error: %s", e)
            return EnhancedPromptResult(prompt=query, context_used=False)
        except Exception as e:
            logger.error("RAG enhance error: %s", e)
            return EnhancedPromptResult(prompt=query, context_used=False)

    async def add_episode(
        self,
        content: str,
        source_description: str,
        group_id: Optional[str] = None,
        name: Optional[str] = None,
        reference_time: Optional[str] = None,
    ) -> Optional[str]:
        """
        Add an episode to the knowledge graph.

        Args:
            content: Episode content (e.g., conversation turn)
            source_description: Description of the source
            group_id: Group ID for the episode
            name: Optional episode name
            reference_time: ISO 8601 timestamp (defaults to now)

        Returns:
            Episode ID if successful, None otherwise
        """
        if not settings.memory.enabled or not settings.memory.store_conversations:
            return None

        gid = group_id or settings.memory.group_id
        ref_time = reference_time or datetime.utcnow().isoformat() + "Z"
        ep_name = name or f"conversation-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

        try:
            client = await self._get_client()
            payload = {
                "name": ep_name,
                "episode_body": content,
                "source_description": source_description,
                "reference_time": ref_time,
                "group_id": gid,
                "is_historical": False,
                "data_source_type": "current",
            }

            resp = await client.post("/episodes", json=payload)

            if resp.status_code != 200:
                logger.warning("RAG add_episode failed (%d): %s", resp.status_code, resp.text[:200])
                return None

            data = resp.json()
            episode_id = data.get("episode_id")
            logger.debug("Added episode %s to RAG", episode_id)
            return episode_id

        except httpx.RequestError as e:
            logger.warning("RAG add_episode connection error: %s", e)
            return None
        except Exception as e:
            logger.error("RAG add_episode error: %s", e)
            return None


# Global client instance
_rag_client: Optional[RAGClient] = None


def get_rag_client() -> RAGClient:
    """Get the global RAG client instance."""
    global _rag_client
    if _rag_client is None:
        _rag_client = RAGClient()
    return _rag_client
