"""
Vision events API endpoints.

Provides REST API for querying stored vision detection events
from atlas_vision nodes.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Query

from ..storage.repositories import get_vision_event_repo, get_unified_alert_repo
from ..vision import get_vision_subscriber, get_alert_manager
from ..alerts import AlertRule, create_vision_rule

logger = logging.getLogger("atlas.api.vision")

router = APIRouter(prefix="/vision", tags=["vision"])


@router.get("/events")
async def get_vision_events(
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum events to return"),
    source_id: Optional[str] = Query(default=None, description="Filter by camera ID"),
    node_id: Optional[str] = Query(default=None, description="Filter by vision node ID"),
    class_name: Optional[str] = Query(default=None, description="Filter by detected class (person, car, etc.)"),
    event_type: Optional[str] = Query(default=None, description="Filter by event type (new_track, track_lost)"),
    since_minutes: Optional[int] = Query(default=None, ge=1, le=10080, description="Events from last N minutes"),
):
    """
    Get recent vision detection events.

    Returns a list of detection events from atlas_vision nodes,
    optionally filtered by various criteria.
    """
    repo = get_vision_event_repo()

    since = None
    if since_minutes:
        since = datetime.utcnow() - timedelta(minutes=since_minutes)

    events = await repo.get_recent_events(
        limit=limit,
        source_id=source_id,
        node_id=node_id,
        class_name=class_name,
        event_type=event_type,
        since=since,
    )

    return {
        "count": len(events),
        "events": [e.to_dict() for e in events],
    }


@router.get("/events/counts")
async def get_event_counts(
    since_minutes: int = Query(default=60, ge=1, le=10080, description="Count events from last N minutes"),
    group_by: str = Query(
        default="class_name",
        description="Field to group by",
        pattern="^(class_name|source_id|node_id|event_type)$",
    ),
):
    """
    Get event counts grouped by a field.

    Useful for dashboards and statistics.
    """
    repo = get_vision_event_repo()

    since = datetime.utcnow() - timedelta(minutes=since_minutes)
    counts = await repo.get_event_counts(since=since, group_by=group_by)

    return {
        "since_minutes": since_minutes,
        "group_by": group_by,
        "counts": counts,
        "total": sum(counts.values()),
    }


@router.get("/events/range")
async def get_events_in_range(
    start_time: datetime = Query(..., description="Start of time range (ISO format)"),
    end_time: datetime = Query(..., description="End of time range (ISO format)"),
    source_id: Optional[str] = Query(default=None, description="Filter by camera ID"),
):
    """
    Get events within a specific time range.

    Useful for investigating specific incidents.
    """
    repo = get_vision_event_repo()

    events = await repo.get_events_in_range(
        start_time=start_time,
        end_time=end_time,
        source_id=source_id,
    )

    return {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "count": len(events),
        "events": [e.to_dict() for e in events],
    }


@router.get("/cameras")
async def get_active_cameras(
    since_minutes: int = Query(default=5, ge=1, le=60, description="Consider active within N minutes"),
):
    """
    Get list of cameras that have reported events recently.
    """
    repo = get_vision_event_repo()
    cameras = await repo.get_active_cameras(since_minutes=since_minutes)

    return {
        "since_minutes": since_minutes,
        "count": len(cameras),
        "cameras": cameras,
    }


@router.get("/nodes")
async def get_known_nodes():
    """
    Get list of known vision nodes and their status.
    """
    subscriber = get_vision_subscriber()
    nodes = subscriber.known_nodes

    return {
        "count": len(nodes),
        "nodes": [
            {
                "node_id": status.node_id,
                "status": status.status,
                "last_seen": status.timestamp.isoformat(),
            }
            for status in nodes.values()
        ],
    }


@router.delete("/events/cleanup")
async def cleanup_old_events(
    older_than_days: int = Query(default=7, ge=1, le=365, description="Delete events older than N days"),
):
    """
    Delete old vision events to free up database space.

    Only deletes events older than the specified number of days.
    """
    repo = get_vision_event_repo()

    older_than = datetime.utcnow() - timedelta(days=older_than_days)
    deleted_count = await repo.delete_old_events(older_than)

    return {
        "deleted_count": deleted_count,
        "older_than": older_than.isoformat(),
    }


# =============================================================================
# Alert Rules Management
# =============================================================================


@router.get("/alerts/rules")
async def list_alert_rules():
    """
    List all configured alert rules (vision-specific view).

    Note: Use /alerts/rules for the full unified view.
    """
    manager = get_alert_manager()
    rules = manager.list_rules(event_type="vision")

    return {
        "count": len(rules),
        "rules": [
            {
                "name": r.name,
                "source_pattern": r.source_pattern,
                "class_name": r.conditions.get("class_name", "*"),
                "event_type": r.conditions.get("detection_type", "new_track"),
                "message_template": r.message_template,
                "cooldown_seconds": r.cooldown_seconds,
                "enabled": r.enabled,
                "priority": r.priority,
            }
            for r in rules
        ],
    }


@router.post("/alerts/rules")
async def create_alert_rule(
    name: str = Query(..., description="Unique rule name"),
    source_pattern: str = Query(..., description="Camera source pattern (e.g., '*front_door*' or '*')"),
    class_name: str = Query(..., description="Detection class (e.g., 'person', 'car', '*')"),
    detection_type: str = Query(default="new_track", description="Detection type to match"),
    message_template: str = Query(
        default="{class_name} detected at {source}.",
        description="Alert message template"
    ),
    cooldown_seconds: int = Query(default=30, ge=5, le=3600, description="Cooldown between alerts"),
    priority: int = Query(default=5, ge=1, le=100, description="Rule priority (higher = more important)"),
):
    """
    Create a new vision alert rule.

    Message templates can use: {class_name}, {source}, {source_id}, {time}
    Note: Use /alerts/rules for creating rules for other event types.
    """
    manager = get_alert_manager()

    rule = create_vision_rule(
        name=name,
        source_pattern=source_pattern,
        class_name=class_name,
        detection_type=detection_type,
        message_template=message_template,
        cooldown_seconds=cooldown_seconds,
        priority=priority,
    )

    manager.add_rule(rule)

    return {
        "success": True,
        "message": f"Alert rule '{name}' created",
        "rule": {
            "name": rule.name,
            "source_pattern": rule.source_pattern,
            "class_name": class_name,
        },
    }


@router.delete("/alerts/rules/{rule_name}")
async def delete_alert_rule(rule_name: str):
    """Delete an alert rule."""
    manager = get_alert_manager()

    if manager.remove_rule(rule_name):
        return {"success": True, "message": f"Rule '{rule_name}' deleted"}
    else:
        return {"success": False, "message": f"Rule '{rule_name}' not found"}


@router.post("/alerts/rules/{rule_name}/enable")
async def enable_alert_rule(rule_name: str):
    """Enable an alert rule."""
    manager = get_alert_manager()

    if manager.enable_rule(rule_name):
        return {"success": True, "message": f"Rule '{rule_name}' enabled"}
    else:
        return {"success": False, "message": f"Rule '{rule_name}' not found"}


@router.post("/alerts/rules/{rule_name}/disable")
async def disable_alert_rule(rule_name: str):
    """Disable an alert rule."""
    manager = get_alert_manager()

    if manager.disable_rule(rule_name):
        return {"success": True, "message": f"Rule '{rule_name}' disabled"}
    else:
        return {"success": False, "message": f"Rule '{rule_name}' not found"}


# =============================================================================
# Alert History
# =============================================================================


@router.get("/alerts")
async def get_alerts(
    limit: int = Query(default=50, ge=1, le=500, description="Maximum alerts to return"),
    include_acknowledged: bool = Query(default=False, description="Include acknowledged alerts"),
    rule_name: Optional[str] = Query(default=None, description="Filter by rule name"),
    source_id: Optional[str] = Query(default=None, description="Filter by camera ID"),
    since_minutes: Optional[int] = Query(default=None, ge=1, le=10080, description="Alerts from last N minutes"),
):
    """
    Get vision alert history.

    Returns triggered vision alerts, most recent first.
    By default only shows unacknowledged alerts.
    Note: Use /alerts for alerts from all event types.
    """
    repo = get_unified_alert_repo()

    since = None
    if since_minutes:
        since = datetime.utcnow() - timedelta(minutes=since_minutes)

    alerts = await repo.get_recent_alerts(
        limit=limit,
        event_type="vision",
        include_acknowledged=include_acknowledged,
        rule_name=rule_name,
        source_id=source_id,
        since=since,
    )

    return {
        "count": len(alerts),
        "alerts": [a.to_dict() for a in alerts],
    }


@router.get("/alerts/stats")
async def get_alert_stats(
    since_hours: int = Query(default=24, ge=1, le=168, description="Stats for last N hours"),
):
    """
    Get vision alert statistics.

    Returns counts by rule, source, and acknowledgment status.
    Note: Use /alerts/stats for stats from all event types.
    """
    repo = get_unified_alert_repo()

    since = datetime.utcnow() - timedelta(hours=since_hours)
    stats = await repo.get_alert_stats(since=since)

    vision_stats = {
        "since_hours": since_hours,
        "total": stats.get("by_type", {}).get("vision", 0),
        "unacknowledged": stats.get("unacknowledged", 0),
        "by_rule": {k: v for k, v in stats.get("by_rule", {}).items()},
        "by_source": stats.get("by_source", {}),
    }

    return vision_stats


@router.get("/alerts/unacknowledged/count")
async def get_unacknowledged_count():
    """Get count of unacknowledged vision alerts."""
    repo = get_unified_alert_repo()
    count = await repo.get_unacknowledged_count(event_type="vision")

    return {"unacknowledged_count": count}


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: Optional[str] = Query(default=None, description="Who acknowledged"),
):
    """Acknowledge a single alert."""
    from uuid import UUID

    repo = get_unified_alert_repo()

    try:
        uuid = UUID(alert_id)
    except ValueError:
        return {"success": False, "message": "Invalid alert ID"}

    if await repo.acknowledge_alert(uuid, acknowledged_by):
        return {"success": True, "message": "Alert acknowledged"}
    else:
        return {"success": False, "message": "Alert not found or already acknowledged"}


@router.post("/alerts/acknowledge-all")
async def acknowledge_all_alerts(
    acknowledged_by: Optional[str] = Query(default=None, description="Who acknowledged"),
    rule_name: Optional[str] = Query(default=None, description="Only this rule"),
    source_id: Optional[str] = Query(default=None, description="Only this camera"),
):
    """
    Acknowledge multiple vision alerts at once.

    Optionally filter by rule or source.
    """
    repo = get_unified_alert_repo()

    count = await repo.acknowledge_all(
        acknowledged_by=acknowledged_by,
        event_type="vision",
        rule_name=rule_name,
        source_id=source_id,
    )

    return {
        "success": True,
        "acknowledged_count": count,
    }
