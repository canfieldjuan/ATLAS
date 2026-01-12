"""
RAG feedback repository for source usage tracking.

Tracks which RAG sources are retrieved and their effectiveness.
"""

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from ..database import get_db_pool
from ..models import RAGSourceStats, RAGSourceUsage

logger = logging.getLogger("atlas.storage.feedback")


class FeedbackRepository:
    """
    Repository for RAG feedback tracking.

    Tracks source usage and aggregates effectiveness statistics.
    """

    async def record_source_usage(
        self,
        session_id: Optional[UUID],
        query: str,
        source_id: Optional[str],
        source_fact: str,
        confidence: float,
    ) -> RAGSourceUsage:
        """
        Record that a RAG source was retrieved for a query.

        Args:
            session_id: The session where the query occurred
            query: The user query that triggered retrieval
            source_id: Identifier of the source (from GraphRAG)
            source_fact: The fact/content from the source
            confidence: Confidence score of the retrieval

        Returns:
            The created RAGSourceUsage record
        """
        pool = get_db_pool()
        usage_id = uuid4()
        now = datetime.utcnow()

        await pool.execute(
            """
            INSERT INTO rag_source_usage
                (id, session_id, query, source_id, source_fact, confidence,
                 created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            usage_id,
            session_id,
            query,
            source_id,
            source_fact,
            confidence,
            now,
            now,
        )

        # Update aggregate stats
        await self._update_source_stats(source_id, confidence)

        logger.debug("Recorded source usage for query: %s", query[:50])

        return RAGSourceUsage(
            id=usage_id,
            session_id=session_id,
            query=query,
            source_id=source_id,
            source_fact=source_fact,
            confidence=confidence,
            created_at=now,
            updated_at=now,
        )

    async def record_feedback(
        self,
        usage_id: UUID,
        was_helpful: bool,
        feedback_type: Optional[str] = None,
    ) -> Optional[RAGSourceUsage]:
        """
        Record feedback on a RAG source usage.

        Args:
            usage_id: The usage record to update
            was_helpful: Whether the source was helpful
            feedback_type: Type of feedback (explicit, implicit, etc.)

        Returns:
            Updated RAGSourceUsage if found, None otherwise
        """
        pool = get_db_pool()
        now = datetime.utcnow()

        # Get the source_id for updating stats
        source_id = await pool.fetchval(
            "SELECT source_id FROM rag_source_usage WHERE id = $1",
            usage_id,
        )

        if source_id is None:
            return None

        await pool.execute(
            """
            UPDATE rag_source_usage
            SET was_helpful = $1, feedback_type = $2, updated_at = $3
            WHERE id = $4
            """,
            was_helpful,
            feedback_type,
            now,
            usage_id,
        )

        # Update aggregate stats
        await self._update_feedback_stats(source_id, was_helpful)

        logger.debug("Recorded feedback for usage %s: helpful=%s", usage_id, was_helpful)

        return await self.get_usage(usage_id)

    async def get_usage(self, usage_id: UUID) -> Optional[RAGSourceUsage]:
        """Get a specific usage record."""
        pool = get_db_pool()
        row = await pool.fetchrow(
            """
            SELECT id, session_id, query, source_id, source_fact, confidence,
                   was_helpful, feedback_type, created_at, updated_at
            FROM rag_source_usage
            WHERE id = $1
            """,
            usage_id,
        )

        if not row:
            return None

        return RAGSourceUsage(
            id=row["id"],
            session_id=row["session_id"],
            query=row["query"],
            source_id=row["source_id"],
            source_fact=row["source_fact"],
            confidence=row["confidence"],
            was_helpful=row["was_helpful"],
            feedback_type=row["feedback_type"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def get_session_sources(
        self,
        session_id: UUID,
        limit: int = 20,
    ) -> list[RAGSourceUsage]:
        """Get all sources used in a session."""
        pool = get_db_pool()
        rows = await pool.fetch(
            """
            SELECT id, session_id, query, source_id, source_fact, confidence,
                   was_helpful, feedback_type, created_at, updated_at
            FROM rag_source_usage
            WHERE session_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            session_id,
            limit,
        )

        return [
            RAGSourceUsage(
                id=row["id"],
                session_id=row["session_id"],
                query=row["query"],
                source_id=row["source_id"],
                source_fact=row["source_fact"],
                confidence=row["confidence"],
                was_helpful=row["was_helpful"],
                feedback_type=row["feedback_type"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    async def get_source_stats(self, source_id: str) -> Optional[RAGSourceStats]:
        """Get aggregate stats for a source."""
        pool = get_db_pool()
        row = await pool.fetchrow(
            """
            SELECT id, source_id, times_retrieved, times_helpful, times_not_helpful,
                   avg_confidence, last_retrieved_at, created_at, updated_at
            FROM rag_source_stats
            WHERE source_id = $1
            """,
            source_id,
        )

        if not row:
            return None

        return RAGSourceStats(
            id=row["id"],
            source_id=row["source_id"],
            times_retrieved=row["times_retrieved"],
            times_helpful=row["times_helpful"],
            times_not_helpful=row["times_not_helpful"],
            avg_confidence=row["avg_confidence"],
            last_retrieved_at=row["last_retrieved_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def get_top_sources(self, limit: int = 10) -> list[RAGSourceStats]:
        """Get top sources by helpfulness rate."""
        pool = get_db_pool()
        rows = await pool.fetch(
            """
            SELECT id, source_id, times_retrieved, times_helpful, times_not_helpful,
                   avg_confidence, last_retrieved_at, created_at, updated_at
            FROM rag_source_stats
            WHERE times_helpful + times_not_helpful > 0
            ORDER BY (times_helpful::float / (times_helpful + times_not_helpful)) DESC
            LIMIT $1
            """,
            limit,
        )

        return [
            RAGSourceStats(
                id=row["id"],
                source_id=row["source_id"],
                times_retrieved=row["times_retrieved"],
                times_helpful=row["times_helpful"],
                times_not_helpful=row["times_not_helpful"],
                avg_confidence=row["avg_confidence"],
                last_retrieved_at=row["last_retrieved_at"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    async def _update_source_stats(
        self,
        source_id: Optional[str],
        confidence: float,
    ) -> None:
        """Update aggregate stats when a source is retrieved."""
        if source_id is None:
            return

        pool = get_db_pool()
        now = datetime.utcnow()

        # Upsert stats record
        await pool.execute(
            """
            INSERT INTO rag_source_stats
                (id, source_id, times_retrieved, avg_confidence, last_retrieved_at,
                 created_at, updated_at)
            VALUES ($1, $2, 1, $3, $4, $5, $6)
            ON CONFLICT (source_id) DO UPDATE SET
                times_retrieved = rag_source_stats.times_retrieved + 1,
                avg_confidence = (
                    (rag_source_stats.avg_confidence * rag_source_stats.times_retrieved + $3)
                    / (rag_source_stats.times_retrieved + 1)
                ),
                last_retrieved_at = $4,
                updated_at = $6
            """,
            uuid4(),
            source_id,
            confidence,
            now,
            now,
            now,
        )

    async def _update_feedback_stats(
        self,
        source_id: Optional[str],
        was_helpful: bool,
    ) -> None:
        """Update aggregate stats when feedback is received."""
        if source_id is None:
            return

        pool = get_db_pool()
        now = datetime.utcnow()

        if was_helpful:
            await pool.execute(
                """
                UPDATE rag_source_stats
                SET times_helpful = times_helpful + 1, updated_at = $1
                WHERE source_id = $2
                """,
                now,
                source_id,
            )
        else:
            await pool.execute(
                """
                UPDATE rag_source_stats
                SET times_not_helpful = times_not_helpful + 1, updated_at = $1
                WHERE source_id = $2
                """,
                now,
                source_id,
            )


# Global repository instance
_feedback_repo: Optional[FeedbackRepository] = None


def get_feedback_repo() -> FeedbackRepository:
    """Get the global feedback repository."""
    global _feedback_repo
    if _feedback_repo is None:
        _feedback_repo = FeedbackRepository()
    return _feedback_repo
