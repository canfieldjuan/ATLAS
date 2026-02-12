"""
Device health check builtin task.

Connects to Home Assistant and checks all entities for:
- Unavailable/unknown state
- Low battery levels
- Stale last_updated timestamps
"""

import logging
from datetime import datetime, timedelta, timezone

from ...capabilities.backends.homeassistant import HomeAssistantBackend
from ...config import settings
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.device_health")


async def run(task: ScheduledTask) -> dict:
    """
    Check Home Assistant device health.

    Configurable via task.metadata:
        battery_threshold (int): Low battery % threshold (default: 20)
        stale_hours (int): Hours before entity is considered stale (default: 24)
        domain_filter (list[str]): Domain prefixes to check (default: from HA config)
    """
    if not settings.homeassistant.enabled:
        return {
            "total_entities": 0,
            "healthy": 0,
            "issues": [],
            "summary": "Home Assistant integration is disabled.",
        }

    if not settings.homeassistant.token:
        return {
            "total_entities": 0,
            "healthy": 0,
            "issues": [],
            "summary": "Home Assistant token not configured.",
        }

    metadata = task.metadata or {}
    battery_threshold = metadata.get("battery_threshold", 20)
    stale_hours = metadata.get("stale_hours", 24)
    domain_filter = metadata.get("domain_filter", None)

    backend = HomeAssistantBackend(
        base_url=settings.homeassistant.url,
        access_token=settings.homeassistant.token,
    )

    try:
        await backend.connect()
        entities = await backend.list_entities(domain_filter=domain_filter)
    except Exception as e:
        logger.error("Failed to connect to Home Assistant: %s", e)
        return {
            "total_entities": 0,
            "healthy": 0,
            "issues": [],
            "summary": f"Failed to connect to Home Assistant: {e}",
        }
    finally:
        await backend.disconnect()

    now = datetime.now(timezone.utc)
    stale_cutoff = now - timedelta(hours=stale_hours)
    issues = []

    for entity in entities:
        entity_id = entity.get("entity_id", "")
        state = entity.get("state", "")
        attrs = entity.get("attributes", {})
        friendly_name = attrs.get("friendly_name", entity_id)

        # Check unavailable / unknown
        if state == "unavailable":
            issues.append({
                "entity_id": entity_id,
                "issue": "unavailable",
                "friendly_name": friendly_name,
            })
            continue

        if state == "unknown":
            issues.append({
                "entity_id": entity_id,
                "issue": "unknown",
                "friendly_name": friendly_name,
            })
            continue

        # Check low battery
        battery = attrs.get("battery_level") or attrs.get("battery")
        if battery is not None:
            try:
                battery_pct = int(battery)
                if battery_pct < battery_threshold:
                    issues.append({
                        "entity_id": entity_id,
                        "issue": "low_battery",
                        "battery_pct": battery_pct,
                        "friendly_name": friendly_name,
                    })
            except (ValueError, TypeError):
                pass

        # Check stale last_updated
        last_updated_str = entity.get("last_updated")
        if last_updated_str:
            try:
                last_updated = datetime.fromisoformat(
                    last_updated_str.replace("Z", "+00:00")
                )
                if last_updated < stale_cutoff:
                    issues.append({
                        "entity_id": entity_id,
                        "issue": "stale",
                        "last_updated": last_updated.isoformat(),
                        "friendly_name": friendly_name,
                    })
            except (ValueError, TypeError):
                pass

    total = len(entities)
    # Count unique entity_ids with issues
    issue_entity_ids = {i["entity_id"] for i in issues}
    healthy = total - len(issue_entity_ids)

    # Build summary
    summary_parts = [f"{healthy}/{total} devices healthy."]

    unavail = [i for i in issues if i["issue"] == "unavailable"]
    if unavail:
        names = ", ".join(i["entity_id"] for i in unavail[:5])
        suffix = f" (+{len(unavail) - 5} more)" if len(unavail) > 5 else ""
        summary_parts.append(f"{len(unavail)} unavailable: {names}{suffix}.")

    unknown = [i for i in issues if i["issue"] == "unknown"]
    if unknown:
        names = ", ".join(i["entity_id"] for i in unknown[:3])
        summary_parts.append(f"{len(unknown)} unknown: {names}.")

    low_batt = [i for i in issues if i["issue"] == "low_battery"]
    if low_batt:
        items = ", ".join(f"{i['entity_id']} ({i['battery_pct']}%)" for i in low_batt[:3])
        summary_parts.append(f"{len(low_batt)} low battery: {items}.")

    stale = [i for i in issues if i["issue"] == "stale"]
    if stale:
        summary_parts.append(f"{len(stale)} stale (>{stale_hours}h).")

    result = {
        "total_entities": total,
        "healthy": healthy,
        "issues": issues,
        "summary": " ".join(summary_parts),
    }

    logger.info("Device health: %s", result["summary"])
    return result
