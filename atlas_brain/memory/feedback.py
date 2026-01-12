"""
Feedback service for RAG source tracking.

Tracks which sources are retrieved, collects feedback,
and formats source citations for responses.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from ..config import settings
from ..storage import db_settings

logger = logging.getLogger("atlas.memory.feedback")


@dataclass
class SourceCitation:
    """A formatted source citation for display."""

    source_id: Optional[str]
    fact: str
    confidence: float
    usage_id: Optional[UUID] = None


@dataclass
class FeedbackContext:
    """Context for feedback tracking in a response."""

    session_id: Optional[UUID] = None
    query: str = ""
    sources: list[SourceCitation] = field(default_factory=list)
    usage_ids: list[UUID] = field(default_factory=list)

    def has_sources(self) -> bool:
        """Check if any sources were used."""
        return len(self.sources) > 0


class FeedbackService:
    """
    Service for tracking RAG source usage and collecting feedback.

    Provides:
    - Source usage tracking during context gathering
    - Feedback collection (explicit and implicit)
    - Source citation formatting for responses
    """

    def __init__(self):
        self._enabled = db_settings.enabled

    async def track_sources(
        self,
        session_id: Optional[str],
        query: str,
        sources: list[Any],
    ) -> FeedbackContext:
        """
        Track RAG sources used for a query.

        Args:
            session_id: The session ID
            query: The user query
            sources: List of RAG sources (from EnhancedPromptResult)

        Returns:
            FeedbackContext with tracked sources
        """
        context = FeedbackContext(
            session_id=UUID(session_id) if session_id else None,
            query=query,
        )

        if not sources:
            return context

        if not self._enabled:
            # Still populate citations even if DB disabled
            for source in sources:
                fact = source.fact if hasattr(source, "fact") else source.get("fact", "")
                conf = source.confidence if hasattr(source, "confidence") else source.get("confidence", 0)
                source_id = source.uuid if hasattr(source, "uuid") else source.get("uuid")
                context.sources.append(SourceCitation(
                    source_id=str(source_id) if source_id else None,
                    fact=fact,
                    confidence=conf,
                ))
            return context

        # Track each source in database
        from ..storage.repositories.feedback import get_feedback_repo
        repo = get_feedback_repo()

        for source in sources:
            try:
                fact = source.fact if hasattr(source, "fact") else source.get("fact", "")
                conf = source.confidence if hasattr(source, "confidence") else source.get("confidence", 0)
                source_id = source.uuid if hasattr(source, "uuid") else source.get("uuid")

                usage = await repo.record_source_usage(
                    session_id=context.session_id,
                    query=query,
                    source_id=str(source_id) if source_id else None,
                    source_fact=fact,
                    confidence=conf,
                )

                context.sources.append(SourceCitation(
                    source_id=str(source_id) if source_id else None,
                    fact=fact,
                    confidence=conf,
                    usage_id=usage.id,
                ))
                context.usage_ids.append(usage.id)

            except Exception as e:
                logger.warning("Failed to track source: %s", e)

        logger.debug("Tracked %d sources for query", len(context.sources))
        return context

    async def record_helpful(
        self,
        usage_ids: list[UUID],
        feedback_type: str = "implicit",
    ) -> None:
        """
        Record that sources were helpful.

        Args:
            usage_ids: List of usage IDs to mark as helpful
            feedback_type: Type of feedback (implicit, explicit, thumbs_up)
        """
        if not self._enabled or not usage_ids:
            return

        from ..storage.repositories.feedback import get_feedback_repo
        repo = get_feedback_repo()

        for usage_id in usage_ids:
            try:
                await repo.record_feedback(usage_id, was_helpful=True, feedback_type=feedback_type)
            except Exception as e:
                logger.warning("Failed to record helpful feedback: %s", e)

    async def record_not_helpful(
        self,
        usage_ids: list[UUID],
        feedback_type: str = "implicit",
    ) -> None:
        """
        Record that sources were not helpful.

        Args:
            usage_ids: List of usage IDs to mark as not helpful
            feedback_type: Type of feedback (implicit, explicit, thumbs_down)
        """
        if not self._enabled or not usage_ids:
            return

        from ..storage.repositories.feedback import get_feedback_repo
        repo = get_feedback_repo()

        for usage_id in usage_ids:
            try:
                await repo.record_feedback(usage_id, was_helpful=False, feedback_type=feedback_type)
            except Exception as e:
                logger.warning("Failed to record not helpful feedback: %s", e)

    def format_citations(
        self,
        context: FeedbackContext,
        max_citations: int = 3,
        min_confidence: float = 0.3,
    ) -> str:
        """
        Format source citations for inclusion in a response.

        Args:
            context: The feedback context with tracked sources
            max_citations: Maximum number of citations to include
            min_confidence: Minimum confidence to include a citation

        Returns:
            Formatted citation string
        """
        if not context.has_sources():
            return ""

        # Filter by confidence and limit
        citations = [
            s for s in context.sources
            if s.confidence >= min_confidence
        ][:max_citations]

        if not citations:
            return ""

        lines = ["Sources:"]
        for i, citation in enumerate(citations, 1):
            # Truncate fact if too long
            fact = citation.fact
            if len(fact) > 100:
                fact = fact[:97] + "..."
            lines.append(f"  [{i}] {fact}")

        return "\n".join(lines)

    def format_inline_citations(
        self,
        context: FeedbackContext,
        min_confidence: float = 0.5,
    ) -> list[str]:
        """
        Get inline citation markers for high-confidence sources.

        Args:
            context: The feedback context with tracked sources
            min_confidence: Minimum confidence for inline citation

        Returns:
            List of citation markers like "[1]", "[2]"
        """
        return [
            f"[{i}]"
            for i, s in enumerate(context.sources, 1)
            if s.confidence >= min_confidence
        ]


# Global service instance
_feedback_service: Optional[FeedbackService] = None


def get_feedback_service() -> FeedbackService:
    """Get the global feedback service."""
    global _feedback_service
    if _feedback_service is None:
        _feedback_service = FeedbackService()
    return _feedback_service
