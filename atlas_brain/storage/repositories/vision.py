"""
Vision events repository for storing detection events from atlas_vision nodes.

Provides storage and retrieval of vision detection events.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID, uuid4

from ..database import get_db_pool
from ..models import VisionEventRecord

logger = logging.getLogger("atlas.storage.vision")


class VisionEventRepository:
    """
    Repository for vision event storage and retrieval.

    Handles persistence of detection events from atlas_vision nodes.
    """

    async def save_event(self, event: VisionEventRecord) -> VisionEventRecord:
        """
        Save a vision event to the database.

        Uses INSERT ... ON CONFLICT to handle duplicates gracefully.

        Args:
            event: The event to save

        Returns:
            The saved event with ID
        """
        pool = get_db_pool()

        if not pool.is_initialized:
            logger.debug("Database not initialized, skipping event save")
            return event

        metadata_json = json.dumps(event.metadata)

        try:
            row = await pool.fetchrow(
                """
                INSERT INTO vision_events (
                    id, event_id, event_type, track_id, class_name,
                    source_id, node_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    event_timestamp, received_at, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb)
                ON CONFLICT (node_id, event_id) DO NOTHING
                RETURNING id, received_at
                """,
                event.id,
                event.event_id,
                event.event_type,
                event.track_id,
                event.class_name,
                event.source_id,
                event.node_id,
                event.bbox_x1,
                event.bbox_y1,
                event.bbox_x2,
                event.bbox_y2,
                event.event_timestamp,
                event.received_at,
                metadata_json,
            )

            if row:
                event.id = row["id"]
                event.received_at = row["received_at"]
                logger.debug("Saved vision event: %s", event.event_id)
            else:
                logger.debug("Event already exists: %s", event.event_id)

            return event

        except Exception as e:
            logger.error("Failed to save vision event %s: %s", event.event_id, e)
            raise

    async def get_recent_events(
        self,
        limit: int = 100,
        source_id: Optional[str] = None,
        node_id: Optional[str] = None,
        class_name: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> list[VisionEventRecord]:
        """
        Get recent vision events with optional filters.

        Args:
            limit: Maximum number of events to return
            source_id: Filter by camera ID
            node_id: Filter by vision node ID
            class_name: Filter by detected class
            event_type: Filter by event type (new_track, track_lost, etc.)
            since: Only return events after this timestamp

        Returns:
            List of matching events, most recent first
        """
        pool = get_db_pool()

        if not pool.is_initialized:
            return []

        # Build query with optional filters
        conditions = []
        params = []
        param_idx = 1

        if source_id:
            conditions.append(f"source_id = ${param_idx}")
            params.append(source_id)
            param_idx += 1

        if node_id:
            conditions.append(f"node_id = ${param_idx}")
            params.append(node_id)
            param_idx += 1

        if class_name:
            conditions.append(f"class_name = ${param_idx}")
            params.append(class_name)
            param_idx += 1

        if event_type:
            conditions.append(f"event_type = ${param_idx}")
            params.append(event_type)
            param_idx += 1

        if since:
            conditions.append(f"event_timestamp >= ${param_idx}")
            params.append(since)
            param_idx += 1

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        params.append(limit)

        query = f"""
            SELECT id, event_id, event_type, track_id, class_name,
                   source_id, node_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                   event_timestamp, received_at, metadata
            FROM vision_events
            {where_clause}
            ORDER BY event_timestamp DESC
            LIMIT ${param_idx}
        """

        rows = await pool.fetch(query, *params)
        return [self._row_to_event(row) for row in rows]

    async def get_events_in_range(
        self,
        start_time: datetime,
        end_time: datetime,
        source_id: Optional[str] = None,
    ) -> list[VisionEventRecord]:
        """
        Get events within a time range.

        Args:
            start_time: Start of time range
            end_time: End of time range
            source_id: Optional camera filter

        Returns:
            List of events in the time range
        """
        pool = get_db_pool()

        if not pool.is_initialized:
            return []

        if source_id:
            rows = await pool.fetch(
                """
                SELECT id, event_id, event_type, track_id, class_name,
                       source_id, node_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                       event_timestamp, received_at, metadata
                FROM vision_events
                WHERE event_timestamp BETWEEN $1 AND $2
                  AND source_id = $3
                ORDER BY event_timestamp ASC
                """,
                start_time,
                end_time,
                source_id,
            )
        else:
            rows = await pool.fetch(
                """
                SELECT id, event_id, event_type, track_id, class_name,
                       source_id, node_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                       event_timestamp, received_at, metadata
                FROM vision_events
                WHERE event_timestamp BETWEEN $1 AND $2
                ORDER BY event_timestamp ASC
                """,
                start_time,
                end_time,
            )

        return [self._row_to_event(row) for row in rows]

    async def get_event_counts(
        self,
        since: Optional[datetime] = None,
        group_by: str = "class_name",
    ) -> dict[str, int]:
        """
        Get event counts grouped by a field.

        Args:
            since: Count events since this time (default: last hour)
            group_by: Field to group by (class_name, source_id, node_id, event_type)

        Returns:
            Dictionary mapping group values to counts
        """
        pool = get_db_pool()

        if not pool.is_initialized:
            return {}

        if since is None:
            since = datetime.utcnow() - timedelta(hours=1)

        # Validate group_by to prevent SQL injection
        valid_fields = {"class_name", "source_id", "node_id", "event_type"}
        if group_by not in valid_fields:
            group_by = "class_name"

        rows = await pool.fetch(
            f"""
            SELECT {group_by} as group_key, COUNT(*) as count
            FROM vision_events
            WHERE event_timestamp >= $1
            GROUP BY {group_by}
            ORDER BY count DESC
            """,
            since,
        )

        return {row["group_key"]: row["count"] for row in rows}

    async def delete_old_events(self, older_than: datetime) -> int:
        """
        Delete events older than a given timestamp.

        Args:
            older_than: Delete events before this time

        Returns:
            Number of deleted events
        """
        pool = get_db_pool()

        if not pool.is_initialized:
            return 0

        result = await pool.execute(
            "DELETE FROM vision_events WHERE event_timestamp < $1",
            older_than,
        )

        count = int(result.split()[-1]) if result else 0
        if count > 0:
            logger.info("Deleted %d old vision events", count)
        return count

    async def get_active_cameras(self, since_minutes: int = 5) -> list[str]:
        """
        Get list of cameras that have reported events recently.

        Args:
            since_minutes: Consider cameras active if event within this many minutes

        Returns:
            List of camera source_ids
        """
        pool = get_db_pool()

        if not pool.is_initialized:
            return []

        since = datetime.utcnow() - timedelta(minutes=since_minutes)

        rows = await pool.fetch(
            """
            SELECT DISTINCT source_id
            FROM vision_events
            WHERE event_timestamp >= $1
            ORDER BY source_id
            """,
            since,
        )

        return [row["source_id"] for row in rows]

    def _row_to_event(self, row) -> VisionEventRecord:
        """Convert a database row to a VisionEventRecord object."""
        return VisionEventRecord(
            id=row["id"],
            event_id=row["event_id"],
            event_type=row["event_type"],
            track_id=row["track_id"],
            class_name=row["class_name"],
            source_id=row["source_id"],
            node_id=row["node_id"],
            bbox_x1=row["bbox_x1"],
            bbox_y1=row["bbox_y1"],
            bbox_x2=row["bbox_x2"],
            bbox_y2=row["bbox_y2"],
            event_timestamp=row["event_timestamp"],
            received_at=row["received_at"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )


# Global repository instance
_vision_event_repo: Optional[VisionEventRepository] = None


def get_vision_event_repo() -> VisionEventRepository:
    """Get the global vision event repository."""
    global _vision_event_repo
    if _vision_event_repo is None:
        _vision_event_repo = VisionEventRepository()
    return _vision_event_repo
