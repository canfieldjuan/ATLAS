"""
Anomaly detection task.

Interval task (every 15 min) that compares recent presence events against
learned temporal patterns and checks HA device states for contextual anomalies.
Sends ntfy notifications for significant deviations.
"""

import logging
import zoneinfo
from datetime import datetime, timedelta
from typing import Any

from ...config import settings
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.anomaly_detection")

# In-memory dedup: keyed by "person:transition:date" to avoid repeat alerts.
_notified_anomalies: set[str] = set()
_MAX_DEDUP_ENTRIES = 200


from ...utils.time import format_minutes as _format_time


async def _send_alert(message: str, title: str = "Atlas Anomaly") -> None:
    """Send an ntfy notification if configured."""
    if not settings.alerts.ntfy_enabled:
        return
    try:
        from ...tools.notify import notify_tool
        await notify_tool._send_notification(
            message=message,
            title=title,
            priority="high",
        )
    except Exception as e:
        logger.warning("Failed to send anomaly alert: %s", e)


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Detect unusual presence patterns and device states."""
    global _notified_anomalies

    from ...storage.database import get_db_pool

    metadata = task.metadata or {}
    deviation_threshold = metadata.get("deviation_threshold", settings.autonomous.anomaly_detection_deviation_threshold)
    min_samples = metadata.get("min_samples", settings.autonomous.anomaly_detection_min_samples)

    pool = get_db_pool()
    if not pool.is_initialized:
        return {
            "error": "Database not initialized",
            "_skip_synthesis": "Anomaly detection skipped -- database not ready.",
        }

    # Check if temporal_patterns table exists and has data
    try:
        pattern_count = await pool.fetchval(
            "SELECT count(*) FROM temporal_patterns"
        )
    except Exception:
        # Table doesn't exist yet (migration 025 not applied)
        return {
            "checked": 0, "anomalies": 0, "details": [],
            "note": "temporal_patterns table not ready",
            "_skip_synthesis": "Anomaly detection skipped -- migration pending.",
        }
    if not pattern_count:
        return {
            "checked": 0, "anomalies": 0, "details": [],
            "note": "No learned patterns yet",
            "_skip_synthesis": "Anomaly detection skipped -- no learned patterns yet.",
        }

    # Use server-local timezone to match presence_events timestamps
    local_tz = zoneinfo.ZoneInfo(settings.autonomous.default_timezone)
    now = datetime.now(local_tz)
    current_dow = now.weekday()  # 0=Mon, 6=Sun
    current_minutes = now.hour * 60 + now.minute

    # --- 1. Check recent presence events (last 15 min) ---
    # Use naive datetime to match presence_events TIMESTAMP (no timezone)
    cutoff = now.replace(tzinfo=None) - timedelta(minutes=15)
    recent_events = await pool.fetch(
        """
        SELECT person_name, transition,
               extract(hour from created_at)::int * 60
                   + extract(minute from created_at)::int AS event_minutes,
               created_at
        FROM presence_events
        WHERE created_at >= $1
          AND person_name IS NOT NULL
        ORDER BY created_at DESC
        """,
        cutoff,
    )

    # Reset dedup set if it grows too large
    if len(_notified_anomalies) > _MAX_DEDUP_ENTRIES:
        _notified_anomalies = set()

    anomalies: list[dict[str, Any]] = []
    today_str = now.strftime("%Y-%m-%d")

    for event in recent_events:
        person = event["person_name"]
        transition = event["transition"]
        event_minutes = event["event_minutes"]

        dedup_key = f"{person}:{transition}:{today_str}"
        if dedup_key in _notified_anomalies:
            continue

        # Look up learned pattern
        pattern = await pool.fetchrow(
            """
            SELECT median_minutes, stddev_minutes, sample_count
            FROM temporal_patterns
            WHERE person_name = $1
              AND pattern_type = $2
              AND day_of_week = $3
            """,
            person, transition, current_dow,
        )

        if not pattern or pattern["sample_count"] < min_samples:
            continue

        median = pattern["median_minutes"]
        stddev = max(pattern["stddev_minutes"], 15)
        deviation = abs(event_minutes - median) / stddev

        if deviation > deviation_threshold:
            _notified_anomalies.add(dedup_key)
            detail = {
                "person": person,
                "transition": transition,
                "event_time": _format_time(event_minutes),
                "typical_time": _format_time(median),
                "stddev_min": stddev,
                "deviation": round(deviation, 1),
            }
            anomalies.append(detail)

            message = (
                f"Unusual {transition}: {person} at {detail['event_time']} "
                f"(typical: {detail['typical_time']} +/- {stddev}min)"
            )
            await _send_alert(message)
            logger.info("Anomaly detected: %s", message)

    # --- 2. Check HA device states for contextual anomalies ---
    device_anomalies = await _check_device_anomalies(pool, current_minutes, today_str)
    anomalies.extend(device_anomalies)

    return {
        "checked": len(recent_events),
        "anomalies": len(anomalies),
        "details": anomalies,
    }


async def _check_device_anomalies(pool, current_minutes: int, today_str: str) -> list[dict[str, Any]]:
    """Check HA for contextual device anomalies (lights at night, garage open)."""
    if not settings.homeassistant.enabled or not settings.homeassistant.token:
        return []

    # Only check during nighttime hours (1-5 AM = 60-300 minutes)
    is_night = 60 <= current_minutes <= 300

    from ...capabilities.backends.homeassistant import HomeAssistantBackend

    ha = HomeAssistantBackend(settings.homeassistant.url, settings.homeassistant.token)
    anomalies: list[dict[str, Any]] = []

    try:
        await ha.connect()

        # Check for lights on during deep night when house is empty
        if is_night:
            from ...autonomous.presence import get_presence_tracker, OccupancyState
            presence = get_presence_tracker()
            if presence.state.state == OccupancyState.EMPTY:
                lights = await ha.list_entities(domain_filter=["light."])
                on_lights = [
                    e for e in lights
                    if e.get("state") == "on"
                ]
                if on_lights:
                    names = [
                        e.get("attributes", {}).get("friendly_name", e.get("entity_id", "?"))
                        for e in on_lights[:3]
                    ]
                    dedup_key = f"lights_night:{today_str}"
                    if dedup_key not in _notified_anomalies:
                        _notified_anomalies.add(dedup_key)
                        detail = {
                            "type": "device",
                            "issue": "lights_on_at_night",
                            "entities": names,
                        }
                        anomalies.append(detail)
                        msg = f"Lights on at night while house is empty: {', '.join(names)}"
                        await _send_alert(msg, title="Atlas Device Anomaly")

        # Check for garage/covers open for extended time at night
        if current_minutes >= 1320 or current_minutes <= 360:  # 10 PM - 6 AM
            covers = await ha.list_entities(domain_filter=["cover."])
            open_covers = [
                e for e in covers
                if e.get("state") == "open"
            ]
            if open_covers:
                names = [
                    e.get("attributes", {}).get("friendly_name", e.get("entity_id", "?"))
                    for e in open_covers[:3]
                ]
                dedup_key = f"covers_night:{today_str}"
                if dedup_key not in _notified_anomalies:
                    _notified_anomalies.add(dedup_key)
                    detail = {
                        "type": "device",
                        "issue": "covers_open_at_night",
                        "entities": names,
                    }
                    anomalies.append(detail)
                    msg = f"Garage/covers open at night: {', '.join(names)}"
                    await _send_alert(msg, title="Atlas Device Anomaly")

    except Exception as e:
        logger.warning("HA device anomaly check failed: %s", e)
    finally:
        try:
            await ha.disconnect()
        except Exception:
            pass

    return anomalies
