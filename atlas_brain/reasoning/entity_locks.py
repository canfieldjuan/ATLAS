"""Entity lock manager for sovereignty between AtlasAgent and ReasoningAgent."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from .events import AtlasEvent

logger = logging.getLogger("atlas.reasoning.entity_locks")


class EntityLockManager:
    """Manages entity-level locks in the entity_locks table.

    When AtlasAgent holds an active voice session with a contact, it
    acquires a lock. ReasoningAgent checks before acting and queues
    decisions for locked entities.
    """

    # ------------------------------------------------------------------
    # Acquire / Release
    # ------------------------------------------------------------------

    async def acquire(
        self,
        entity_type: str,
        entity_id: str,
        holder: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """Acquire a lock. Returns True if acquired, False if already held."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO entity_locks (entity_type, entity_id, holder, session_id)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (entity_type, entity_id) WHERE released_at IS NULL
            DO NOTHING
            RETURNING id
            """,
            entity_type,
            entity_id,
            holder,
            session_id,
        )
        acquired = row is not None
        if acquired:
            logger.info(
                "Lock acquired: %s/%s by %s (session=%s)",
                entity_type, entity_id, holder, session_id,
            )
        else:
            logger.debug(
                "Lock already held: %s/%s (requested by %s)",
                entity_type, entity_id, holder,
            )
        return acquired

    async def release(
        self, entity_type: str, entity_id: str, holder: str
    ) -> bool:
        """Release a lock held by *holder*. Returns True if released."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        tag = await pool.execute(
            """
            UPDATE entity_locks
            SET released_at = NOW()
            WHERE entity_type = $1
              AND entity_id = $2
              AND holder = $3
              AND released_at IS NULL
            """,
            entity_type,
            entity_id,
            holder,
        )
        released = tag == "UPDATE 1"
        if released:
            logger.info("Lock released: %s/%s by %s", entity_type, entity_id, holder)
        return released

    async def release_by_session(self, session_id: str) -> list[tuple[str, str]]:
        """Release all locks held by *session_id*. Returns (entity_type, entity_id) pairs."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        rows = await pool.fetch(
            """
            UPDATE entity_locks
            SET released_at = NOW()
            WHERE session_id = $1
              AND released_at IS NULL
            RETURNING entity_type, entity_id
            """,
            session_id,
        )
        pairs = [(r["entity_type"], r["entity_id"]) for r in rows]
        if pairs:
            logger.info("Released %d locks for session %s", len(pairs), session_id)
        return pairs

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def is_locked(
        self, entity_type: str, entity_id: str
    ) -> tuple[bool, Optional[str]]:
        """Check if an entity is locked. Returns (locked, holder)."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        row = await pool.fetchrow(
            """
            SELECT holder FROM entity_locks
            WHERE entity_type = $1
              AND entity_id = $2
              AND released_at IS NULL
            """,
            entity_type,
            entity_id,
        )
        if row:
            return True, row["holder"]
        return False, None

    # ------------------------------------------------------------------
    # Heartbeat / Expiry
    # ------------------------------------------------------------------

    async def heartbeat(
        self, entity_type: str, entity_id: str, holder: str
    ) -> bool:
        """Update heartbeat timestamp. Returns True if lock exists."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        tag = await pool.execute(
            """
            UPDATE entity_locks
            SET heartbeat_at = NOW()
            WHERE entity_type = $1
              AND entity_id = $2
              AND holder = $3
              AND released_at IS NULL
            """,
            entity_type,
            entity_id,
            holder,
        )
        return tag == "UPDATE 1"

    async def expire_stale(self, timeout_seconds: int) -> int:
        """Release locks whose heartbeat is older than *timeout_seconds*."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        rows = await pool.fetch(
            """
            UPDATE entity_locks
            SET released_at = NOW()
            WHERE released_at IS NULL
              AND heartbeat_at < NOW() - make_interval(secs => $1)
            RETURNING entity_type, entity_id, holder
            """,
            float(timeout_seconds),
        )
        if rows:
            for r in rows:
                logger.warning(
                    "Expired stale lock: %s/%s held by %s",
                    r["entity_type"], r["entity_id"], r["holder"],
                )
        return len(rows)

    # ------------------------------------------------------------------
    # Queue / Drain
    # ------------------------------------------------------------------

    async def queue_for_entity(
        self, event_id: UUID, entity_type: str, entity_id: str
    ) -> UUID:
        """Queue an event for later processing when the lock is released."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        queue_id = await pool.fetchval(
            """
            INSERT INTO reasoning_queue (event_id, entity_type, entity_id)
            VALUES ($1, $2, $3)
            RETURNING id
            """,
            event_id,
            entity_type,
            entity_id,
        )
        logger.debug(
            "Queued event %s for %s/%s (queue_id=%s)",
            event_id, entity_type, entity_id, queue_id,
        )
        return queue_id

    async def drain_queue(
        self, entity_type: str, entity_id: str
    ) -> list[AtlasEvent]:
        """Fetch and mark queued events as drained. Returns AtlasEvent list."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        rows = await pool.fetch(
            """
            WITH to_drain AS (
                SELECT rq.id AS rq_id, rq.event_id
                FROM reasoning_queue rq
                WHERE rq.entity_type = $1
                  AND rq.entity_id = $2
                  AND rq.drained_at IS NULL
                LIMIT 200
            )
            UPDATE reasoning_queue rq
            SET drained_at = NOW()
            FROM to_drain
            JOIN atlas_events ae ON ae.id = to_drain.event_id
            WHERE rq.id = to_drain.rq_id
            RETURNING ae.id, ae.event_type, ae.source, ae.entity_type,
                      ae.entity_id, ae.payload, ae.created_at
            """,
            entity_type,
            entity_id,
        )
        events = [AtlasEvent.from_row(dict(r)) for r in rows]
        if events:
            logger.info(
                "Drained %d queued events for %s/%s",
                len(events), entity_type, entity_id,
            )
        return events
