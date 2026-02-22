"""
Action escalation builtin task.

Periodically checks pending proactive actions and sends ntfy
notifications for aging items. Classifies by age tier:
stale (7+ days), overdue (3-6 days), pending (1-2 days).
Only fires when the house is occupied.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.action_escalation")

# Age tier thresholds in days (defaults, overridden by config + task.metadata)
STALE_DAYS = 7
OVERDUE_DAYS = 3


async def run(task: ScheduledTask) -> dict[str, Any]:
    """
    Check pending action items and escalate aging ones via notification.

    Configurable via task.metadata:
        stale_days (int): Days before an item is considered stale (default: 7)
        overdue_days (int): Days before an item is considered overdue (default: 3)
    """
    # 1. Presence gate
    from ...autonomous.presence import get_presence_tracker, OccupancyState

    presence = get_presence_tracker()
    if presence.state.state == OccupancyState.EMPTY:
        return {
            "pending_count": 0,
            "escalations": {},
            "notified": False,
            "_skip_synthesis": "Action escalation skipped -- house empty.",
        }

    metadata = task.metadata or {}
    stale_threshold = metadata.get("stale_days", settings.autonomous.action_escalation_stale_days)
    overdue_threshold = metadata.get("overdue_days", settings.autonomous.action_escalation_overdue_days)

    # 2. Query pending actions
    from ...storage.database import get_db_pool

    pool = get_db_pool()
    if not pool.is_initialized:
        return {
            "pending_count": 0,
            "escalations": {},
            "notified": False,
            "_skip_synthesis": "Action escalation skipped -- database not ready.",
        }

    try:
        rows = await pool.fetch(
            """
            SELECT action_text, action_type, created_at
            FROM proactive_actions
            WHERE status = 'pending'
            ORDER BY created_at ASC
            """,
        )
    except Exception as e:
        logger.warning("Failed to query pending actions: %s", e)
        return {"error": str(e)}

    if not rows:
        return {
            "pending_count": 0,
            "escalations": {},
            "notified": False,
            "_skip_synthesis": "No pending action items.",
        }

    # 3. Classify by age tier
    now = datetime.now(timezone.utc)
    stale: list[str] = []
    overdue: list[str] = []
    pending: list[str] = []

    for row in rows:
        created = row["created_at"]
        # Ensure timezone-aware comparison
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = (now - created).days

        text = row["action_text"]
        if age_days >= stale_threshold:
            stale.append(text)
        elif age_days >= overdue_threshold:
            overdue.append(text)
        elif age_days >= 1:
            pending.append(text)
        # Items < 1 day old are too fresh to escalate

    # 4. Only notify if there are stale or overdue items
    notified = False
    if stale or overdue:
        parts = []
        if stale:
            items = ", ".join(stale[:5])
            parts.append(f"Stale ({len(stale)}): {items}")
            if len(stale) > 5:
                parts[-1] += f" and {len(stale) - 5} more"
        if overdue:
            items = ", ".join(overdue[:5])
            parts.append(f"Overdue ({len(overdue)}): {items}")
            if len(overdue) > 5:
                parts[-1] += f" and {len(overdue) - 5} more"

        message = "Pending action items. " + ". ".join(parts) + "."
        priority = "high" if stale else "default"

        if settings.alerts.ntfy_enabled:
            try:
                from ...tools.notify import notify_tool
                await notify_tool._send_notification(
                    message=message,
                    title="Action Items",
                    priority=priority,
                )
                notified = True
            except Exception as e:
                logger.warning("Failed to send escalation notification: %s", e)

    return {
        "pending_count": len(rows),
        "escalations": {
            "stale": {"count": len(stale), "items": stale[:5]},
            "overdue": {"count": len(overdue), "items": overdue[:5]},
            "pending": {"count": len(pending), "items": pending[:5]},
            "fresh": len(rows) - len(stale) - len(overdue) - len(pending),
        },
        "notified": notified,
    }
