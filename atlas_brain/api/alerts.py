"""
Unified alerts API endpoints.

Provides REST API for querying and managing alerts from all event sources
(vision, audio, ha_state, security).
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Query

from ..alerts import AlertRule, get_alert_manager
from ..storage.repositories import get_unified_alert_repo

logger = logging.getLogger("atlas.api.alerts")

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("")
async def get_alerts(
    limit: int = Query(default=50, ge=1, le=500, description="Maximum alerts to return"),
    event_type: Optional[str] = Query(
        default=None,
        description="Filter by event type (vision, audio, ha_state, security)",
    ),
    include_acknowledged: bool = Query(default=False, description="Include acknowledged alerts"),
    rule_name: Optional[str] = Query(default=None, description="Filter by rule name"),
    source_id: Optional[str] = Query(default=None, description="Filter by source ID"),
    since_minutes: Optional[int] = Query(
        default=None, ge=1, le=10080, description="Alerts from last N minutes"
    ),
    node_id: Optional[str] = Query(default=None, description="Filter by edge node ID"),
):
    """
    Get unified alert history from all event sources.

    Returns triggered alerts, most recent first.
    By default only shows unacknowledged alerts.
    """
    repo = get_unified_alert_repo()

    since = None
    if since_minutes:
        since = datetime.utcnow() - timedelta(minutes=since_minutes)

    alerts = await repo.get_recent_alerts(
        limit=limit,
        event_type=event_type,
        include_acknowledged=include_acknowledged,
        rule_name=rule_name,
        source_id=source_id,
        since=since,
        node_id=node_id,
    )

    return {
        "count": len(alerts),
        "alerts": [a.to_dict() for a in alerts],
    }


@router.get("/stats")
async def get_alert_stats(
    since_hours: int = Query(default=24, ge=1, le=168, description="Stats for last N hours"),
):
    """
    Get alert statistics across all event types.

    Returns counts by event type, rule, source, and acknowledgment status.
    """
    repo = get_unified_alert_repo()

    since = datetime.utcnow() - timedelta(hours=since_hours)
    stats = await repo.get_alert_stats(since=since)

    return {
        "since_hours": since_hours,
        **stats,
    }


@router.get("/unacknowledged/count")
async def get_unacknowledged_count(
    event_type: Optional[str] = Query(default=None, description="Filter by event type"),
):
    """Get count of unacknowledged alerts."""
    repo = get_unified_alert_repo()
    count = await repo.get_unacknowledged_count(event_type=event_type)

    return {"unacknowledged_count": count}


@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: Optional[str] = Query(default=None, description="Who acknowledged"),
):
    """Acknowledge a single alert."""
    repo = get_unified_alert_repo()

    try:
        uuid = UUID(alert_id)
    except ValueError:
        return {"success": False, "message": "Invalid alert ID"}

    if await repo.acknowledge_alert(uuid, acknowledged_by):
        return {"success": True, "message": "Alert acknowledged"}
    else:
        return {"success": False, "message": "Alert not found or already acknowledged"}


@router.post("/acknowledge-all")
async def acknowledge_all_alerts(
    acknowledged_by: Optional[str] = Query(default=None, description="Who acknowledged"),
    event_type: Optional[str] = Query(default=None, description="Only this event type"),
    rule_name: Optional[str] = Query(default=None, description="Only this rule"),
    source_id: Optional[str] = Query(default=None, description="Only this source"),
):
    """
    Acknowledge multiple alerts at once.

    Optionally filter by event type, rule, or source.
    """
    repo = get_unified_alert_repo()

    count = await repo.acknowledge_all(
        acknowledged_by=acknowledged_by,
        event_type=event_type,
        rule_name=rule_name,
        source_id=source_id,
    )

    return {
        "success": True,
        "acknowledged_count": count,
    }


@router.delete("/cleanup")
async def cleanup_old_alerts(
    older_than_days: int = Query(default=30, ge=1, le=365, description="Delete alerts older than N days"),
):
    """
    Delete old alerts to free up database space.

    Only deletes alerts older than the specified number of days.
    """
    repo = get_unified_alert_repo()

    older_than = datetime.utcnow() - timedelta(days=older_than_days)
    deleted_count = await repo.delete_old_alerts(older_than)

    return {
        "deleted_count": deleted_count,
        "older_than": older_than.isoformat(),
    }


@router.get("/rules")
async def list_alert_rules(
    event_type: Optional[str] = Query(default=None, description="Filter by event type"),
):
    """List all configured alert rules."""
    manager = get_alert_manager()
    rules = manager.list_rules(event_type=event_type)

    return {
        "count": len(rules),
        "rules": [
            {
                "name": r.name,
                "event_types": r.event_types,
                "source_pattern": r.source_pattern,
                "conditions": r.conditions,
                "message_template": r.message_template,
                "cooldown_seconds": r.cooldown_seconds,
                "enabled": r.enabled,
                "priority": r.priority,
            }
            for r in rules
        ],
    }


@router.post("/rules")
async def create_alert_rule(
    name: str = Query(..., description="Unique rule name"),
    event_types: str = Query(
        default="vision",
        description="Comma-separated event types (vision,audio,ha_state,security) or *",
    ),
    source_pattern: str = Query(..., description="Source pattern (e.g., '*front_door*' or '*')"),
    message_template: str = Query(
        default="{event_type} event at {source}",
        description="Alert message template",
    ),
    cooldown_seconds: int = Query(default=30, ge=5, le=3600, description="Cooldown between alerts"),
    priority: int = Query(default=5, ge=1, le=100, description="Rule priority (higher = more important)"),
):
    """
    Create a new alert rule.

    Message templates can use: {event_type}, {source_id}, {source}, {time},
    and event-specific fields like {class_name}, {sound_class}, {new_state}.
    """
    manager = get_alert_manager()

    event_type_list = [t.strip() for t in event_types.split(",")]

    rule = AlertRule(
        name=name,
        event_types=event_type_list,
        source_pattern=source_pattern,
        message_template=message_template,
        cooldown_seconds=cooldown_seconds,
        priority=priority,
        enabled=True,
    )

    manager.add_rule(rule)

    return {
        "success": True,
        "message": f"Alert rule '{name}' created",
        "rule": {
            "name": rule.name,
            "event_types": rule.event_types,
            "source_pattern": rule.source_pattern,
        },
    }


@router.delete("/rules/{rule_name}")
async def delete_alert_rule(rule_name: str):
    """Delete an alert rule."""
    manager = get_alert_manager()

    if manager.remove_rule(rule_name):
        return {"success": True, "message": f"Rule '{rule_name}' deleted"}
    else:
        return {"success": False, "message": f"Rule '{rule_name}' not found"}


@router.post("/rules/{rule_name}/enable")
async def enable_alert_rule(rule_name: str):
    """Enable an alert rule."""
    manager = get_alert_manager()

    if manager.enable_rule(rule_name):
        return {"success": True, "message": f"Rule '{rule_name}' enabled"}
    else:
        return {"success": False, "message": f"Rule '{rule_name}' not found"}


@router.post("/rules/{rule_name}/disable")
async def disable_alert_rule(rule_name: str):
    """Disable an alert rule."""
    manager = get_alert_manager()

    if manager.disable_rule(rule_name):
        return {"success": True, "message": f"Rule '{rule_name}' disabled"}
    else:
        return {"success": False, "message": f"Rule '{rule_name}' not found"}


@router.post("/test")
async def trigger_test_alert(
    message: str = Query(default="Test alert from Atlas", description="Test message"),
):
    """
    Trigger a test alert to verify the delivery pipeline.

    Creates a test reminder event and processes it through the alert system.
    This will trigger all configured delivery methods (TTS, ntfy, etc.).
    """
    from ..alerts import ReminderAlertEvent

    manager = get_alert_manager()
    test_event = ReminderAlertEvent(
        source_id="test_alert",
        timestamp=datetime.utcnow(),
        message=message,
        reminder_id="test-000",
        event_type="reminder",
    )

    result = await manager.process_event(test_event)

    return {
        "success": bool(result),
        "message": "Alert triggered and delivered" if result else "No matching rule found",
    }
