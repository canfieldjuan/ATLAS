"""
Departure security check builtin task.

Queries Home Assistant for security-relevant entities when the house
goes empty: lights left on, doors unlocked, garage/covers open.
"""

import logging
from typing import Any

from ...config import settings
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.departure_check")

# Default domains and states that indicate an issue
DEFAULT_CHECKS: list[dict[str, Any]] = [
    {"domain": "light", "bad_state": "on", "label": "lights on"},
    {"domain": "lock", "bad_state": "unlocked", "label": "locks unlocked"},
    {"domain": "cover", "bad_state": "open", "label": "covers/garage open"},
]


async def run(task: ScheduledTask) -> dict[str, Any]:
    """
    Check HA entities for security issues on departure.

    Configurable via task.metadata:
        domains (list[str]): Override domain list (default: light, lock, cover)
        skip_entities (list[str]): Entity IDs to ignore
    """
    if not settings.homeassistant.enabled:
        return {
            "total_checked": 0,
            "issues": [],
            "summary": "Home Assistant not enabled -- skipped.",
        }

    ha_url = settings.homeassistant.url
    ha_token = settings.homeassistant.token
    if not ha_token:
        return {
            "total_checked": 0,
            "issues": [],
            "summary": "Home Assistant token not configured -- skipped.",
        }

    metadata = task.metadata or {}
    skip_entities: list[str] = metadata.get("skip_entities", [])

    # Build check list from metadata or defaults
    custom_domains = metadata.get("domains")
    if custom_domains:
        checks = []
        for d in custom_domains:
            # Map domain to sensible defaults
            if d == "light":
                checks.append({"domain": "light", "bad_state": "on", "label": "lights on"})
            elif d == "lock":
                checks.append({"domain": "lock", "bad_state": "unlocked", "label": "locks unlocked"})
            elif d == "cover":
                checks.append({"domain": "cover", "bad_state": "open", "label": "covers/garage open"})
            elif d == "switch":
                checks.append({"domain": "switch", "bad_state": "on", "label": "switches on"})
            else:
                checks.append({"domain": d, "bad_state": "on", "label": f"{d} on"})
    else:
        checks = list(DEFAULT_CHECKS)

    # Connect to HA
    from ...capabilities.backends.homeassistant import HomeAssistantBackend

    ha = HomeAssistantBackend(ha_url, ha_token)
    try:
        await ha.connect()
    except Exception as e:
        logger.error("Failed to connect to HA: %s", e)
        return {
            "total_checked": 0,
            "issues": [],
            "summary": f"Failed to connect to Home Assistant: {e}",
        }

    total_checked = 0
    issues: list[dict[str, str]] = []

    try:
        for check in checks:
            domain = check["domain"]
            bad_state = check["bad_state"]
            label = check["label"]

            entities = await ha.list_entities(domain_filter=[f"{domain}."])
            for entity in entities:
                eid = entity.get("entity_id", "")
                if eid in skip_entities:
                    continue

                total_checked += 1
                state = entity.get("state", "")

                if state == bad_state:
                    friendly = entity.get("attributes", {}).get("friendly_name", eid)
                    issues.append({
                        "entity_id": eid,
                        "friendly_name": friendly,
                        "state": state,
                        "issue": label,
                    })
    finally:
        await ha.disconnect()

    # Build summary
    if issues:
        parts = []
        for issue in issues:
            parts.append(f"{issue['friendly_name']} ({issue['issue']})")
        summary = f"Departure check: {len(issues)} issue(s) found -- {', '.join(parts[:5])}"
        if len(issues) > 5:
            summary += f" and {len(issues) - 5} more"
    else:
        summary = f"Departure check: all clear ({total_checked} entities checked)."

    logger.info(summary)
    return {
        "total_checked": total_checked,
        "issues": issues,
        "summary": summary,
    }
