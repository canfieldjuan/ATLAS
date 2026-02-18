"""
RAG Client for graphiti-wrapper integration.

Provides async HTTP client for the GraphRAG service.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

from ..config import settings

logger = logging.getLogger("atlas.memory.rag_client")


@dataclass
class SearchSource:
    """A single fact from the knowledge graph."""

    uuid: str
    name: str
    fact: str
    confidence: float = 1.0
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    created_at: Optional[str] = None
    expired_at: Optional[str] = None


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

    facts: list[SearchSource] = field(default_factory=list)


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
        reranker: str = "cross_encoder",
    ):
        self._base_url = base_url or settings.memory.base_url
        self._timeout = timeout or settings.memory.timeout
        self._reranker = reranker
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
            resp = await client.get("/healthcheck")
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
        max_facts: int = 5,
    ) -> SearchResult:
        """
        Search the knowledge graph for relevant facts.

        Args:
            query: Search query
            group_id: Group ID to search within
            max_facts: Maximum number of facts to return

        Returns:
            SearchResult with list of facts
        """
        if not settings.memory.enabled:
            return SearchResult()

        gid = group_id or settings.memory.group_id
        group_ids = [gid] if gid else None

        try:
            client = await self._get_client()
            payload = {
                "query": query,
                "group_ids": group_ids,
                "max_facts": max_facts,
                "search_methods": ["cosine_similarity"],
                "reranker": self._reranker,
            }

            resp = await client.post("/search", json=payload)

            if resp.status_code != 200:
                logger.warning("RAG search failed (%d): %s", resp.status_code, resp.text[:200])
                return SearchResult()

            data = resp.json()
            facts = []
            for f in data.get("facts", []):
                facts.append(SearchSource(
                    uuid=f.get("uuid", ""),
                    name=f.get("name", ""),
                    fact=f.get("fact", ""),
                    valid_at=f.get("valid_at"),
                    invalid_at=f.get("invalid_at"),
                    created_at=f.get("created_at"),
                    expired_at=f.get("expired_at"),
                ))
            return SearchResult(facts=facts)

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
        max_sources: int = 5,
    ) -> EnhancedPromptResult:
        """
        Enhance a user prompt with knowledge graph context.

        Searches the graph for relevant facts and returns them
        alongside the original query.

        Args:
            query: User query to enhance
            group_id: Group ID to search within
            max_sources: Maximum number of facts to include

        Returns:
            EnhancedPromptResult with original prompt and sources
        """
        if not settings.memory.enabled or not settings.memory.retrieve_context:
            return EnhancedPromptResult(prompt=query, context_used=False)

        try:
            result = await self.search(
                query=query,
                group_id=group_id,
                max_facts=max_sources,
            )

            if not result.facts:
                return EnhancedPromptResult(prompt=query, context_used=False)

            return EnhancedPromptResult(
                prompt=query,
                context_used=True,
                sources=result.facts,
            )

        except Exception as e:
            logger.warning("RAG enhance_prompt error: %s", e)
            return EnhancedPromptResult(prompt=query, context_used=False)

    async def add_messages(
        self,
        messages: list[dict],
        group_id: Optional[str] = None,
    ) -> bool:
        """
        Send conversation messages for knowledge graph extraction.

        Uses POST /messages which queues LLM-powered entity/relationship
        extraction across the conversation context.

        Args:
            messages: List of message dicts with keys:
                content (str, required),
                role_type ("user"|"assistant", required),
                role (str or None),
                source_description (str, optional)
            group_id: Group ID for the messages

        Returns:
            True if messages were accepted, False otherwise
        """
        if not settings.memory.enabled or not settings.memory.store_conversations:
            return False

        gid = group_id or settings.memory.group_id

        try:
            client = await self._get_client()
            payload = {
                "group_id": gid,
                "messages": messages,
            }

            resp = await client.post("/messages", json=payload)

            if resp.status_code not in (200, 202):
                logger.warning(
                    "RAG add_messages failed (%d): %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return False

            data = resp.json()
            logger.debug("RAG add_messages: %s", data.get("message", ""))
            return data.get("success", False)

        except httpx.RequestError as e:
            logger.warning("RAG add_messages connection error: %s", e)
            return False
        except Exception as e:
            logger.error("RAG add_messages error: %s", e)
            return False

    async def send_messages(
        self,
        messages: list[dict],
        group_id: str | None = None,
    ) -> dict:
        """
        Send a batch of conversation messages for GraphRAG extraction.

        Like add_messages() but returns the full response dict (for nightly sync).

        Args:
            messages: List of message dicts with keys:
                content, role_type ("user"|"assistant"), role (optional), timestamp (ISO8601)
            group_id: Optional group ID (defaults to config group_id)

        Returns:
            Response dict from Graphiti, or empty dict on failure
        """
        try:
            client = await self._get_client()

            payload = {
                "group_id": group_id or settings.memory.group_id,
                "messages": messages,
            }

            resp = await client.post(
                "/messages",
                json=payload,
                timeout=300.0,  # batch extraction can be slow
            )
            resp.raise_for_status()
            return resp.json()

        except Exception as e:
            logger.error("RAG send_messages error: %s", e)
            return {}

    async def add_conversation_turn(
        self,
        role: str,
        content: str,
        session_id: str,
        speaker_name: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Store a single conversation turn via /messages.

        Args:
            role: "user" or "assistant"
            content: The message content
            session_id: Session identifier
            speaker_name: Optional speaker name
            timestamp: Optional timestamp (defaults to now)

        Returns:
            True if accepted, False otherwise
        """
        ts = timestamp or datetime.now(timezone.utc)
        source = "atlas-voice-%s" % role
        if speaker_name:
            source = "%s:%s" % (source, speaker_name)

        message = {
            "content": content,
            "role_type": role,
            "role": speaker_name,
            "source_description": source,
            "timestamp": ts.isoformat() + "Z",
        }
        return await self.add_messages(
            messages=[message],
            group_id=settings.memory.group_id,
        )

    async def add_fact(
        self,
        fact: str,
        source: str = "atlas-learned",
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Store a learned fact via /messages.

        Args:
            fact: The fact to store
            source: Source description
            timestamp: Optional timestamp

        Returns:
            True if accepted, False otherwise
        """
        ts = timestamp or datetime.now(timezone.utc)
        message = {
            "content": fact,
            "role_type": "system",
            "role": None,
            "source_description": source,
            "timestamp": ts.isoformat() + "Z",
        }
        return await self.add_messages(
            messages=[message],
            group_id=settings.memory.group_id,
        )


# Global client instance
_rag_client: Optional[RAGClient] = None


def get_rag_client() -> RAGClient:
    """Get the global RAG client instance."""
    global _rag_client
    if _rag_client is None:
        _rag_client = RAGClient()
    return _rag_client
