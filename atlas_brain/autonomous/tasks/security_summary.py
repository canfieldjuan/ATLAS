"""
Security summary builtin task.

Aggregates vision events and alerts from the database over a
configurable time window and returns a structured summary.
"""

import logging
from datetime import datetime, timedelta, timezone

from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.security_summary")


async def run(task) -> dict:
    """
    Aggregate vision events and alerts for the last N hours.

    Configurable via task.metadata:
        hours (int): Lookback window in hours (default: 24)
    """
    metadata = task.metadata or {}
    hours = metadata.get("hours", 24)
    # DB uses timestamp without time zone stored as UTC -- pass naive UTC
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)

    pool = get_db_pool()

    # Vision event counts by source + class
    vision_by_source_class = await pool.fetch(
        """
        SELECT source_id, class_name, count(*) AS cnt
        FROM vision_events
        WHERE event_timestamp >= $1
        GROUP BY source_id, class_name
        ORDER BY cnt DESC
        """,
        since,
    )

    # Alert counts by rule
    alerts_by_rule = await pool.fetch(
        """
        SELECT rule_name, count(*) AS cnt, max(triggered_at) AS last_triggered
        FROM alerts
        WHERE triggered_at >= $1
        GROUP BY rule_name
        ORDER BY cnt DESC
        """,
        since,
    )

    # Unacknowledged alert count
    unacked_row = await pool.fetchrow(
        """
        SELECT count(*) AS cnt
        FROM alerts
        WHERE acknowledged = false AND triggered_at >= $1
        """,
        since,
    )
    unacked = unacked_row["cnt"] if unacked_row else 0

    # Aggregate vision events
    total_vision = 0
    by_source: dict[str, int] = {}
    by_class: dict[str, int] = {}
    for row in vision_by_source_class:
        cnt = row["cnt"]
        total_vision += cnt
        src = row["source_id"]
        cls = row["class_name"]
        by_source[src] = by_source.get(src, 0) + cnt
        by_class[cls] = by_class.get(cls, 0) + cnt

    # Aggregate alerts
    total_alerts = 0
    by_rule: dict[str, dict] = {}
    for row in alerts_by_rule:
        cnt = row["cnt"]
        total_alerts += cnt
        by_rule[row["rule_name"]] = {
            "count": cnt,
            "last_triggered": row["last_triggered"].isoformat() if row["last_triggered"] else None,
        }

    # Build human-readable summary
    summary_parts = [f"{hours}h:"]

    # Vision
    summary_parts.append(f"{total_vision} vision events")
    if by_class:
        class_items = sorted(by_class.items(), key=lambda x: -x[1])[:5]
        class_str = ", ".join(f"{cnt} {cls}" for cls, cnt in class_items)
        summary_parts[-1] += f" ({class_str})"
    summary_parts[-1] += "."

    # Alerts
    summary_parts.append(f"{total_alerts} alerts ({unacked} unacked).")

    # Top source
    if by_source:
        top_source = max(by_source, key=lambda k: by_source[k])
        summary_parts.append(f"Top source: {top_source} ({by_source[top_source]} events).")

    result = {
        "period_hours": hours,
        "vision_events": {
            "total": total_vision,
            "by_source": by_source,
            "by_class": by_class,
        },
        "alerts": {
            "total": total_alerts,
            "unacknowledged": unacked,
            "by_rule": by_rule,
        },
        "summary": " ".join(summary_parts),
    }

    logger.info("Security summary: %s", result["summary"])
    return result
