"""
Conversation repository for persistent conversation storage.

Provides CRUD operations for conversation turns with minimal latency.
Target: <2ms for writes, <3ms for reads.
"""

import json
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from ..database import get_db_pool
from ..models import ConversationTurn

logger = logging.getLogger("atlas.storage.conversation")


class ConversationRepository:
    """
    Repository for conversation turn persistence.

    All methods are async and optimized for low latency.
    """

    async def add_turn(
        self,
        session_id: UUID,
        role: str,
        content: str,
        speaker_id: Optional[str] = None,
        intent: Optional[str] = None,
        turn_type: str = "conversation",
        metadata: Optional[dict] = None,
    ) -> UUID:
        """
        Add a conversation turn.

        Args:
            session_id: The session this turn belongs to
            role: "user" or "assistant"
            content: The message content
            speaker_id: Identified speaker name (optional)
            intent: Parsed intent (optional)
            turn_type: "conversation" or "command" (for context filtering)
            metadata: Additional metadata (optional)

        Returns:
            UUID of the created turn

        Target latency: <2ms
        """
        pool = get_db_pool()
        turn_id = uuid4()
        metadata_json = json.dumps(metadata or {})

        await pool.execute(
            """
            INSERT INTO conversation_turns
                (id, session_id, role, content, speaker_id, intent, turn_type, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            """,
            turn_id,
            session_id,
            role,
            content,
            speaker_id,
            intent,
            turn_type,
            metadata_json,
        )

        logger.debug("Added %s turn %s to session %s", turn_type, turn_id, session_id)
        return turn_id

    async def get_history(
        self,
        session_id: UUID,
        limit: int = 20,
        before: Optional[datetime] = None,
        turn_type: Optional[str] = None,
    ) -> list[ConversationTurn]:
        """
        Get conversation history for a session.

        Args:
            session_id: The session to get history for
            limit: Maximum number of turns to return
            before: Only return turns before this timestamp
            turn_type: Filter by type ("conversation" or "command"), None for all

        Returns:
            List of ConversationTurn objects, oldest first

        Target latency: <3ms for 20 turns
        """
        pool = get_db_pool()

        # Build query based on filters
        if turn_type and before:
            rows = await pool.fetch(
                """
                SELECT id, session_id, role, content, speaker_id, intent,
                       turn_type, created_at, metadata
                FROM conversation_turns
                WHERE session_id = $1 AND created_at < $2 AND turn_type = $3
                ORDER BY created_at DESC
                LIMIT $4
                """,
                session_id,
                before,
                turn_type,
                limit,
            )
        elif turn_type:
            rows = await pool.fetch(
                """
                SELECT id, session_id, role, content, speaker_id, intent,
                       turn_type, created_at, metadata
                FROM conversation_turns
                WHERE session_id = $1 AND turn_type = $2
                ORDER BY created_at DESC
                LIMIT $3
                """,
                session_id,
                turn_type,
                limit,
            )
        elif before:
            rows = await pool.fetch(
                """
                SELECT id, session_id, role, content, speaker_id, intent,
                       turn_type, created_at, metadata
                FROM conversation_turns
                WHERE session_id = $1 AND created_at < $2
                ORDER BY created_at DESC
                LIMIT $3
                """,
                session_id,
                before,
                limit,
            )
        else:
            rows = await pool.fetch(
                """
                SELECT id, session_id, role, content, speaker_id, intent,
                       turn_type, created_at, metadata
                FROM conversation_turns
                WHERE session_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                session_id,
                limit,
            )

        # Reverse to get oldest-first order
        turns = [
            ConversationTurn(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                speaker_id=row["speaker_id"],
                intent=row["intent"],
                turn_type=row["turn_type"] or "conversation",
                created_at=row["created_at"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            for row in reversed(rows)
        ]

        logger.debug("Retrieved %d turns for session %s", len(turns), session_id)
        return turns

    async def get_history_for_user(
        self,
        user_id: UUID,
        limit: int = 50,
        include_inactive_sessions: bool = True,
        turn_type: Optional[str] = None,
    ) -> list[ConversationTurn]:
        """
        Get conversation history across all sessions for a user.

        Args:
            user_id: The user to get history for
            limit: Maximum number of turns to return
            include_inactive_sessions: Include turns from closed sessions
            turn_type: Filter by type ("conversation" or "command"), None for all

        Returns:
            List of ConversationTurn objects, oldest first
        """
        pool = get_db_pool()

        # Build base query with turn_type
        type_filter = "AND ct.turn_type = $3" if turn_type else ""
        params = [user_id, limit]
        if turn_type:
            params.append(turn_type)

        if include_inactive_sessions:
            rows = await pool.fetch(
                f"""
                SELECT ct.id, ct.session_id, ct.role, ct.content, ct.speaker_id,
                       ct.intent, ct.turn_type, ct.created_at, ct.metadata
                FROM conversation_turns ct
                JOIN sessions s ON ct.session_id = s.id
                WHERE s.user_id = $1 {type_filter}
                ORDER BY ct.created_at DESC
                LIMIT $2
                """,
                *params,
            )
        else:
            active_filter = "AND s.is_active = true"
            rows = await pool.fetch(
                f"""
                SELECT ct.id, ct.session_id, ct.role, ct.content, ct.speaker_id,
                       ct.intent, ct.turn_type, ct.created_at, ct.metadata
                FROM conversation_turns ct
                JOIN sessions s ON ct.session_id = s.id
                WHERE s.user_id = $1 {active_filter} {type_filter}
                ORDER BY ct.created_at DESC
                LIMIT $2
                """,
                *params,
            )

        # Reverse to get oldest-first order
        return [
            ConversationTurn(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                speaker_id=row["speaker_id"],
                intent=row["intent"],
                turn_type=row["turn_type"] or "conversation",
                created_at=row["created_at"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            for row in reversed(rows)
        ]

    async def delete_session_history(self, session_id: UUID) -> int:
        """
        Delete all turns for a session.

        Returns:
            Number of turns deleted
        """
        pool = get_db_pool()
        result = await pool.execute(
            "DELETE FROM conversation_turns WHERE session_id = $1",
            session_id,
        )
        # Result is like "DELETE 5"
        count = int(result.split()[-1]) if result else 0
        logger.info("Deleted %d turns from session %s", count, session_id)
        return count

    async def count_turns(self, session_id: UUID) -> int:
        """Get the number of turns in a session."""
        pool = get_db_pool()
        return await pool.fetchval(
            "SELECT COUNT(*) FROM conversation_turns WHERE session_id = $1",
            session_id,
        )


# Global repository instance
_conversation_repo: Optional[ConversationRepository] = None


def get_conversation_repo() -> ConversationRepository:
    """Get the global conversation repository."""
    global _conversation_repo
    if _conversation_repo is None:
        _conversation_repo = ConversationRepository()
    return _conversation_repo
