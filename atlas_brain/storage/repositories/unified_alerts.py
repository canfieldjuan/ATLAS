"""
Unified alerts repository for the centralized alert system.

Provides storage and retrieval of alerts from all event sources.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID, uuid4

from ..database import get_db_pool
from ..models import Alert

logger = logging.getLogger("atlas.storage.unified_alerts")


class UnifiedAlertRepository:
    """
    Repository for unified alert storage and retrieval.

    Handles persistence of alerts from all event sources:
    vision, audio, ha_state, security.
    """

    async def save_alert(
        self,
        rule_name: str,
        event_type: str,
        message: str,
        source_id: str,
        event_data: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[Alert]:
        """Save an alert to the database."""
        pool = get_db_pool()

        if not pool.is_initialized:
            logger.debug("Database not initialized, skipping alert save")
            return None

        alert_id = uuid4()
        event_data_json = json.dumps(event_data or {})
        metadata_json = json.dumps(metadata or {})

        try:
            row = await pool.fetchrow(
                """
                INSERT INTO alerts (
                    id, rule_name, event_type, message, source_id,
                    triggered_at, event_data, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb)
                RETURNING id, triggered_at
                """,
                alert_id,
                rule_name,
                event_type,
                message,
                source_id,
                datetime.utcnow(),
                event_data_json,
                metadata_json,
            )

            if row:
                return Alert(
                    id=row["id"],
                    rule_name=rule_name,
                    event_type=event_type,
                    message=message,
                    source_id=source_id,
                    triggered_at=row["triggered_at"],
                    event_data=event_data or {},
                    metadata=metadata or {},
                )
            return None

        except Exception as e:
            logger.error("Failed to save alert: %s", e)
            return None

    async def get_recent_alerts(
        self,
        limit: int = 50,
        event_type: Optional[str] = None,
        include_acknowledged: bool = False,
        rule_name: Optional[str] = None,
        source_id: Optional[str] = None,
        since: Optional[datetime] = None,
        node_id: Optional[str] = None,
    ) -> list[Alert]:
        """Get recent alerts with optional filters."""
        pool = get_db_pool()

        if not pool.is_initialized:
            return []

        conditions = []
        params = []
        param_idx = 1

        if not include_acknowledged:
            conditions.append("acknowledged = FALSE")

        if event_type:
            conditions.append(f"event_type = ${param_idx}")
            params.append(event_type)
            param_idx += 1

        if rule_name:
            conditions.append(f"rule_name = ${param_idx}")
            params.append(rule_name)
            param_idx += 1

        if source_id:
            conditions.append(f"source_id = ${param_idx}")
            params.append(source_id)
            param_idx += 1

        if since:
            conditions.append(f"triggered_at >= ${param_idx}")
            params.append(since)
            param_idx += 1

        if node_id:
            conditions.append(f"metadata->>'node_id' = ${param_idx}")
            params.append(node_id)
            param_idx += 1

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        params.append(limit)

        query = f"""
            SELECT id, rule_name, event_type, message, source_id,
                   triggered_at, acknowledged, acknowledged_at,
                   acknowledged_by, event_data, metadata
            FROM alerts
            {where_clause}
            ORDER BY triggered_at DESC
            LIMIT ${param_idx}
        """

        rows = await pool.fetch(query, *params)
        return [self._row_to_alert(row) for row in rows]

    async def get_unacknowledged_count(
        self,
        event_type: Optional[str] = None,
    ) -> int:
        """Get count of unacknowledged alerts."""
        pool = get_db_pool()

        if not pool.is_initialized:
            return 0

        if event_type:
            count = await pool.fetchval(
                "SELECT COUNT(*) FROM alerts WHERE acknowledged = FALSE AND event_type = $1",
                event_type,
            )
        else:
            count = await pool.fetchval(
                "SELECT COUNT(*) FROM alerts WHERE acknowledged = FALSE"
            )
        return count or 0

    async def acknowledge_alert(
        self,
        alert_id: UUID,
        acknowledged_by: Optional[str] = None,
    ) -> bool:
        """Mark an alert as acknowledged."""
        pool = get_db_pool()

        if not pool.is_initialized:
            return False

        result = await pool.execute(
            """
            UPDATE alerts
            SET acknowledged = TRUE,
                acknowledged_at = $2,
                acknowledged_by = $3
            WHERE id = $1 AND acknowledged = FALSE
            """,
            alert_id,
            datetime.utcnow(),
            acknowledged_by,
        )

        success = result and "UPDATE 1" in result
        if success:
            logger.info("Acknowledged alert: %s", alert_id)
        return success

    async def acknowledge_all(
        self,
        acknowledged_by: Optional[str] = None,
        event_type: Optional[str] = None,
        rule_name: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> int:
        """Acknowledge multiple alerts."""
        pool = get_db_pool()

        if not pool.is_initialized:
            return 0

        conditions = ["acknowledged = FALSE"]
        params = [datetime.utcnow(), acknowledged_by]
        param_idx = 3

        if event_type:
            conditions.append(f"event_type = ${param_idx}")
            params.append(event_type)
            param_idx += 1

        if rule_name:
            conditions.append(f"rule_name = ${param_idx}")
            params.append(rule_name)
            param_idx += 1

        if source_id:
            conditions.append(f"source_id = ${param_idx}")
            params.append(source_id)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        result = await pool.execute(
            f"""
            UPDATE alerts
            SET acknowledged = TRUE,
                acknowledged_at = $1,
                acknowledged_by = $2
            WHERE {where_clause}
            """,
            *params,
        )

        count = int(result.split()[-1]) if result else 0
        if count > 0:
            logger.info("Acknowledged %d alerts", count)
        return count

    async def get_alert_stats(
        self,
        since: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get alert statistics."""
        pool = get_db_pool()

        if not pool.is_initialized:
            return {
                "total": 0,
                "unacknowledged": 0,
                "by_type": {},
                "by_rule": {},
                "by_source": {},
            }

        if since is None:
            since = datetime.utcnow() - timedelta(hours=24)
        elif since.tzinfo is not None:
            # Convert to naive UTC for database comparison
            since = since.replace(tzinfo=None)

        counts = await pool.fetchrow(
            """
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE acknowledged = FALSE) as unacknowledged
            FROM alerts
            WHERE triggered_at >= $1
            """,
            since,
        )

        by_type_rows = await pool.fetch(
            """
            SELECT event_type, COUNT(*) as count
            FROM alerts
            WHERE triggered_at >= $1
            GROUP BY event_type
            ORDER BY count DESC
            """,
            since,
        )

        by_rule_rows = await pool.fetch(
            """
            SELECT rule_name, COUNT(*) as count
            FROM alerts
            WHERE triggered_at >= $1
            GROUP BY rule_name
            ORDER BY count DESC
            """,
            since,
        )

        by_source_rows = await pool.fetch(
            """
            SELECT source_id, COUNT(*) as count
            FROM alerts
            WHERE triggered_at >= $1
            GROUP BY source_id
            ORDER BY count DESC
            LIMIT 20
            """,
            since,
        )

        return {
            "total": counts["total"] if counts else 0,
            "unacknowledged": counts["unacknowledged"] if counts else 0,
            "by_type": {row["event_type"]: row["count"] for row in by_type_rows},
            "by_rule": {row["rule_name"]: row["count"] for row in by_rule_rows},
            "by_source": {row["source_id"]: row["count"] for row in by_source_rows},
        }

    async def delete_old_alerts(self, older_than: datetime) -> int:
        """Delete alerts older than a given timestamp."""
        pool = get_db_pool()

        if not pool.is_initialized:
            return 0

        result = await pool.execute(
            "DELETE FROM alerts WHERE triggered_at < $1",
            older_than,
        )

        count = int(result.split()[-1]) if result else 0
        if count > 0:
            logger.info("Deleted %d old alerts", count)
        return count

    def _row_to_alert(self, row) -> Alert:
        """Convert a database row to an Alert object."""
        return Alert(
            id=row["id"],
            rule_name=row["rule_name"],
            event_type=row["event_type"],
            message=row["message"],
            source_id=row["source_id"],
            triggered_at=row["triggered_at"],
            acknowledged=row["acknowledged"],
            acknowledged_at=row["acknowledged_at"],
            acknowledged_by=row["acknowledged_by"],
            event_data=json.loads(row["event_data"]) if row["event_data"] else {},
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )


_unified_alert_repo: Optional[UnifiedAlertRepository] = None


def get_unified_alert_repo() -> UnifiedAlertRepository:
    """Get the global unified alert repository."""
    global _unified_alert_repo
    if _unified_alert_repo is None:
        _unified_alert_repo = UnifiedAlertRepository()
    return _unified_alert_repo
