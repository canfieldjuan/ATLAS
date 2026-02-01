"""
Session repository for managing user sessions across terminals.

Enables seamless conversation continuity when users move between locations.
Sessions are daily - a new session is created each day for each user.
"""

import json
import logging
from datetime import date, datetime
from typing import Optional
from uuid import UUID, uuid4

from ..database import get_db_pool
from ..models import Session

logger = logging.getLogger("atlas.storage.session")


class SessionRepository:
    """
    Repository for session management.

    Sessions track active conversations and enable
    multi-terminal continuity.
    """

    async def create_session(
        self,
        user_id: Optional[UUID] = None,
        terminal_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Session:
        """
        Create a new daily session.

        Args:
            user_id: Optional user this session belongs to
            terminal_id: Optional terminal/device ID
            metadata: Additional metadata

        Returns:
            The created Session
        """
        pool = get_db_pool()
        session_id = uuid4()
        now = datetime.utcnow()
        today = date.today()
        metadata_json = json.dumps(metadata or {})

        await pool.execute(
            """
            INSERT INTO sessions (id, user_id, terminal_id, started_at, last_activity_at,
                                  is_active, session_date, metadata)
            VALUES ($1, $2, $3, $4, $5, true, $6, $7::jsonb)
            """,
            session_id,
            user_id,
            terminal_id,
            now,
            now,
            today,
            metadata_json,
        )

        logger.info("Created daily session %s for user %s on %s", session_id, user_id, today)

        return Session(
            id=session_id,
            user_id=user_id,
            terminal_id=terminal_id,
            started_at=now,
            last_activity_at=now,
            is_active=True,
            session_date=today,
            metadata=metadata or {},
        )

    async def get_session(self, session_id: UUID) -> Optional[Session]:
        """Get a session by ID."""
        pool = get_db_pool()
        row = await pool.fetchrow(
            """
            SELECT id, user_id, terminal_id, started_at, last_activity_at,
                   is_active, session_date, metadata
            FROM sessions
            WHERE id = $1
            """,
            session_id,
        )

        if not row:
            return None

        return Session(
            id=row["id"],
            user_id=row["user_id"],
            terminal_id=row["terminal_id"],
            started_at=row["started_at"],
            last_activity_at=row["last_activity_at"],
            is_active=row["is_active"],
            session_date=row["session_date"] or date.today(),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    async def get_todays_session_for_user(self, user_id: UUID) -> Optional[Session]:
        """
        Get today's active session for a user.

        Sessions are daily - returns only if there's an active session from today.
        """
        pool = get_db_pool()
        today = date.today()
        row = await pool.fetchrow(
            """
            SELECT id, user_id, terminal_id, started_at, last_activity_at,
                   is_active, session_date, metadata
            FROM sessions
            WHERE user_id = $1 AND is_active = true AND session_date = $2
            LIMIT 1
            """,
            user_id,
            today,
        )

        if not row:
            return None

        return Session(
            id=row["id"],
            user_id=row["user_id"],
            terminal_id=row["terminal_id"],
            started_at=row["started_at"],
            last_activity_at=row["last_activity_at"],
            is_active=row["is_active"],
            session_date=row["session_date"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    async def get_or_create_session(
        self,
        user_id: Optional[UUID] = None,
        terminal_id: Optional[str] = None,
    ) -> Session:
        """
        Get today's session or create a new one.

        Sessions are daily - a new session is created each day.
        If the user has an active session from today, returns it.
        Otherwise, closes any old sessions and creates a new one.

        Args:
            user_id: The user ID (optional for anonymous sessions)
            terminal_id: The terminal/device ID

        Returns:
            Today's Session
        """
        # Try to find existing session for today
        if user_id:
            existing = await self.get_todays_session_for_user(user_id)
            if existing:
                # Update terminal ID if user moved locations
                if existing.terminal_id != terminal_id:
                    await self.update_terminal(existing.id, terminal_id)
                    existing.terminal_id = terminal_id
                    logger.info(
                        "User %s continued today's session %s on terminal %s",
                        user_id, existing.id, terminal_id
                    )
                return existing

            # Close any old active sessions before creating new one
            await self.close_user_sessions(user_id)

        # Create new daily session
        return await self.create_session(
            user_id=user_id,
            terminal_id=terminal_id,
        )

    async def update_terminal(self, session_id: UUID, terminal_id: str) -> None:
        """Update the terminal ID for a session (user moved locations)."""
        pool = get_db_pool()
        await pool.execute(
            """
            UPDATE sessions
            SET terminal_id = $2, last_activity_at = $3
            WHERE id = $1
            """,
            session_id,
            terminal_id,
            datetime.utcnow(),
        )
        logger.debug("Updated session %s terminal to %s", session_id, terminal_id)

    async def touch_session(self, session_id: UUID) -> None:
        """Update the last activity timestamp."""
        pool = get_db_pool()
        await pool.execute(
            "UPDATE sessions SET last_activity_at = $2 WHERE id = $1",
            session_id,
            datetime.utcnow(),
        )

    async def update_metadata(
        self,
        session_id: UUID,
        metadata_updates: dict,
    ) -> bool:
        """
        Update session metadata by merging new values.

        Uses PostgreSQL JSONB concatenation to merge updates atomically.
        Existing keys not in updates are preserved.

        Args:
            session_id: The session ID to update
            metadata_updates: Dictionary of metadata to merge

        Returns:
            True if session was found and updated, False otherwise
        """
        pool = get_db_pool()
        updates_json = json.dumps(metadata_updates)
        result = await pool.execute(
            """
            UPDATE sessions
            SET metadata = COALESCE(metadata, '{}'::jsonb) || $2::jsonb,
                last_activity_at = $3
            WHERE id = $1
            """,
            session_id,
            updates_json,
            datetime.utcnow(),
        )
        updated = result and "UPDATE 1" in result
        if updated:
            logger.debug("Updated metadata for session %s", session_id)
        return updated

    async def clear_metadata_key(self, session_id: UUID, key: str) -> bool:
        """
        Remove a specific key from session metadata.

        Args:
            session_id: The session ID to update
            key: The metadata key to remove

        Returns:
            True if session was found and updated, False otherwise
        """
        pool = get_db_pool()
        result = await pool.execute(
            """
            UPDATE sessions
            SET metadata = metadata - $2,
                last_activity_at = $3
            WHERE id = $1
            """,
            session_id,
            key,
            datetime.utcnow(),
        )
        updated = result and "UPDATE 1" in result
        if updated:
            logger.debug("Cleared metadata key '%s' for session %s", key, session_id)
        return updated

    async def close_session(self, session_id: UUID) -> None:
        """Mark a session as inactive."""
        pool = get_db_pool()
        await pool.execute(
            "UPDATE sessions SET is_active = false, last_activity_at = $2 WHERE id = $1",
            session_id,
            datetime.utcnow(),
        )
        logger.info("Closed session %s", session_id)

    async def close_user_sessions(self, user_id: UUID) -> int:
        """Close all active sessions for a user."""
        pool = get_db_pool()
        result = await pool.execute(
            """
            UPDATE sessions
            SET is_active = false, last_activity_at = $2
            WHERE user_id = $1 AND is_active = true
            """,
            user_id,
            datetime.utcnow(),
        )
        count = int(result.split()[-1]) if result else 0
        logger.info("Closed %d sessions for user %s", count, user_id)
        return count

    async def list_active_sessions(self, limit: int = 100) -> list[Session]:
        """List all active sessions."""
        pool = get_db_pool()
        rows = await pool.fetch(
            """
            SELECT id, user_id, terminal_id, started_at, last_activity_at,
                   is_active, session_date, metadata
            FROM sessions
            WHERE is_active = true
            ORDER BY last_activity_at DESC
            LIMIT $1
            """,
            limit,
        )

        return [
            Session(
                id=row["id"],
                user_id=row["user_id"],
                terminal_id=row["terminal_id"],
                started_at=row["started_at"],
                last_activity_at=row["last_activity_at"],
                is_active=row["is_active"],
                session_date=row["session_date"] or date.today(),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            for row in rows
        ]


# Global repository instance
_session_repo: Optional[SessionRepository] = None


def get_session_repo() -> SessionRepository:
    """Get the global session repository."""
    global _session_repo
    if _session_repo is None:
        _session_repo = SessionRepository()
    return _session_repo
