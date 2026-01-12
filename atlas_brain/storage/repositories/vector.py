"""
Vector repository for semantic search operations.

Uses pgvector for efficient similarity queries.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID

from ..database import get_db_pool

logger = logging.getLogger("atlas.storage.vector")


@dataclass
class SimilarityResult:
    """Result from a similarity search."""

    id: UUID
    content: str
    similarity: float
    metadata: dict


class VectorRepository:
    """
    Repository for vector-based operations.

    Handles embedding storage and similarity queries.
    """

    async def search_similar_turns(
        self,
        embedding: str,
        limit: int = 5,
        session_id: Optional[UUID] = None,
        min_similarity: float = 0.5,
    ) -> list[SimilarityResult]:
        """
        Search for similar conversation turns.

        Args:
            embedding: pgvector format string "[0.1, 0.2, ...]"
            limit: Maximum results to return
            session_id: Optional filter by session
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of similar turns with similarity scores
        """
        pool = get_db_pool()
        if not pool.is_initialized:
            return []

        session_filter = ""
        args = [embedding, limit]

        if session_id:
            session_filter = "AND session_id = $3"
            args.append(session_id)

        query = f"""
            SELECT
                id,
                content,
                1 - (embedding <=> $1::vector) as similarity,
                metadata,
                role,
                speaker_id,
                created_at
            FROM conversation_turns
            WHERE embedding IS NOT NULL
            {session_filter}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """

        try:
            rows = await pool.fetch(query, *args)
            results = []
            for row in rows:
                sim = float(row["similarity"])
                if sim >= min_similarity:
                    results.append(SimilarityResult(
                        id=row["id"],
                        content=row["content"],
                        similarity=sim,
                        metadata={
                            "role": row["role"],
                            "speaker_id": row["speaker_id"],
                            "created_at": row["created_at"].isoformat(),
                            **(row["metadata"] or {}),
                        },
                    ))
            return results
        except Exception as e:
            logger.error("Failed to search similar turns: %s", e)
            return []

    async def search_similar_chunks(
        self,
        embedding: str,
        limit: int = 5,
        document_id: Optional[UUID] = None,
        min_similarity: float = 0.5,
    ) -> list[SimilarityResult]:
        """
        Search for similar document chunks.

        Args:
            embedding: pgvector format string
            limit: Maximum results to return
            document_id: Optional filter by document
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar chunks with similarity scores
        """
        pool = get_db_pool()
        if not pool.is_initialized:
            return []

        doc_filter = ""
        args = [embedding, limit]

        if document_id:
            doc_filter = "AND document_id = $3"
            args.append(document_id)

        query = f"""
            SELECT
                c.id,
                c.content,
                1 - (c.embedding <=> $1::vector) as similarity,
                c.metadata,
                c.chunk_index,
                d.filename
            FROM document_chunks c
            JOIN knowledge_documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
            {doc_filter}
            ORDER BY c.embedding <=> $1::vector
            LIMIT $2
        """

        try:
            rows = await pool.fetch(query, *args)
            results = []
            for row in rows:
                sim = float(row["similarity"])
                if sim >= min_similarity:
                    results.append(SimilarityResult(
                        id=row["id"],
                        content=row["content"],
                        similarity=sim,
                        metadata={
                            "chunk_index": row["chunk_index"],
                            "filename": row["filename"],
                            **(row["metadata"] or {}),
                        },
                    ))
            return results
        except Exception as e:
            logger.error("Failed to search similar chunks: %s", e)
            return []

    async def search_similar_memories(
        self,
        embedding: str,
        user_id: Optional[UUID] = None,
        memory_type: Optional[str] = None,
        limit: int = 5,
        min_similarity: float = 0.5,
    ) -> list[SimilarityResult]:
        """
        Search for similar memories.

        Args:
            embedding: pgvector format string
            user_id: Optional filter by user
            memory_type: Optional filter by type
            limit: Maximum results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar memories
        """
        pool = get_db_pool()
        if not pool.is_initialized:
            return []

        filters = []
        args = [embedding, limit]
        param_idx = 3

        if user_id:
            filters.append(f"user_id = ${param_idx}")
            args.append(user_id)
            param_idx += 1

        if memory_type:
            filters.append(f"memory_type = ${param_idx}")
            args.append(memory_type)
            param_idx += 1

        where_clause = ""
        if filters:
            where_clause = "AND " + " AND ".join(filters)

        query = f"""
            SELECT
                id,
                content,
                1 - (embedding <=> $1::vector) as similarity,
                metadata,
                memory_type,
                importance,
                created_at
            FROM memories
            WHERE embedding IS NOT NULL
            AND (expires_at IS NULL OR expires_at > NOW())
            {where_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """

        try:
            rows = await pool.fetch(query, *args)
            results = []
            for row in rows:
                sim = float(row["similarity"])
                if sim >= min_similarity:
                    results.append(SimilarityResult(
                        id=row["id"],
                        content=row["content"],
                        similarity=sim,
                        metadata={
                            "memory_type": row["memory_type"],
                            "importance": float(row["importance"]),
                            "created_at": row["created_at"].isoformat(),
                            **(row["metadata"] or {}),
                        },
                    ))
            return results
        except Exception as e:
            logger.error("Failed to search similar memories: %s", e)
            return []

    async def update_turn_embedding(
        self,
        turn_id: UUID,
        embedding: str,
    ) -> bool:
        """
        Update embedding for a conversation turn.

        Args:
            turn_id: Turn UUID
            embedding: pgvector format string

        Returns:
            True if updated successfully
        """
        pool = get_db_pool()
        if not pool.is_initialized:
            return False

        query = """
            UPDATE conversation_turns
            SET embedding = $1::vector
            WHERE id = $2
        """

        try:
            await pool.execute(query, embedding, turn_id)
            return True
        except Exception as e:
            logger.error("Failed to update turn embedding: %s", e)
            return False

    async def add_memory(
        self,
        user_id: Optional[UUID],
        memory_type: str,
        content: str,
        embedding: str,
        importance: float = 0.5,
        metadata: Optional[dict] = None,
        expires_at: Optional[datetime] = None,
    ) -> Optional[UUID]:
        """
        Add a new memory with embedding.

        Returns:
            Memory UUID if created successfully
        """
        pool = get_db_pool()
        if not pool.is_initialized:
            return None

        query = """
            INSERT INTO memories (
                user_id, memory_type, content, embedding,
                importance, metadata, expires_at
            )
            VALUES ($1, $2, $3, $4::vector, $5, $6, $7)
            RETURNING id
        """

        try:
            row = await pool.fetchrow(
                query,
                user_id,
                memory_type,
                content,
                embedding,
                importance,
                metadata or {},
                expires_at,
            )
            return row["id"] if row else None
        except Exception as e:
            logger.error("Failed to add memory: %s", e)
            return None

    async def add_document_chunk(
        self,
        document_id: UUID,
        chunk_index: int,
        content: str,
        embedding: str,
        token_count: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[UUID]:
        """
        Add a document chunk with embedding.

        Returns:
            Chunk UUID if created successfully
        """
        pool = get_db_pool()
        if not pool.is_initialized:
            return None

        query = """
            INSERT INTO document_chunks (
                document_id, chunk_index, content, embedding,
                token_count, metadata
            )
            VALUES ($1, $2, $3, $4::vector, $5, $6)
            RETURNING id
        """

        try:
            row = await pool.fetchrow(
                query,
                document_id,
                chunk_index,
                content,
                embedding,
                token_count,
                metadata or {},
            )
            return row["id"] if row else None
        except Exception as e:
            logger.error("Failed to add document chunk: %s", e)
            return None


# Global instance
_vector_repo: Optional[VectorRepository] = None


def get_vector_repository() -> VectorRepository:
    """Get or create the global vector repository."""
    global _vector_repo
    if _vector_repo is None:
        _vector_repo = VectorRepository()
    return _vector_repo
