"""
Morning briefing builtin task.

Composes a daily briefing by directly calling existing tools/services:
- Calendar events for the day
- Current weather
- Overnight security summary
- Device health check
- Pending proactive actions
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ...config import settings
from ...storage.models import ScheduledTask
from . import security_summary as security_mod
from . import device_health as device_mod

logger = logging.getLogger("atlas.autonomous.tasks.morning_briefing")


@dataclass
class _SubTaskStub:
    """Minimal stand-in for ScheduledTask when calling sub-task handlers internally."""
    metadata: dict[str, Any] = field(default_factory=dict)


async def run(task: ScheduledTask) -> dict:
    """
    Generate a morning briefing combining multiple data sources.

    Configurable via task.metadata:
        calendar_hours (int): Hours ahead for calendar events (default: 12)
        security_hours (int): Lookback hours for security summary (default: 8)
    """
    metadata = task.metadata or {}
    calendar_hours = metadata.get("calendar_hours", 12)
    security_hours = metadata.get("security_hours", 8)

    today = datetime.now().strftime("%Y-%m-%d")
    result: dict = {"date": today}

    # 1. Calendar events
    result["calendar"] = await _get_calendar(calendar_hours)

    # 2. Weather
    result["weather"] = await _get_weather()

    # 3. Overnight security summary (reuse security_summary handler)
    try:
        security_result = await security_mod.run(
            _SubTaskStub(metadata={"hours": security_hours})
        )
        result["security"] = {
            "alerts_overnight": security_result.get("alerts", {}).get("total", 0),
            "unacked": security_result.get("alerts", {}).get("unacknowledged", 0),
            "vision_events": security_result.get("vision_events", {}).get("total", 0),
        }
    except Exception as e:
        logger.warning("Security summary failed: %s", e)
        result["security"] = {"error": str(e)}

    # 4. Device health (reuse device_health handler)
    try:
        health_result = await device_mod.run(
            _SubTaskStub(metadata={})
        )
        result["device_health"] = {
            "issues_count": len(health_result.get("issues", [])),
            "total": health_result.get("total_entities", 0),
            "healthy": health_result.get("healthy", 0),
        }
    except Exception as e:
        logger.warning("Device health failed: %s", e)
        result["device_health"] = {"error": str(e)}

    # 5. Pending proactive actions
    result["actions"] = await _get_pending_actions()

    # Build summary
    result["summary"] = _build_summary(result, security_hours)

    logger.info("Morning briefing: %s", result["summary"])
    return result


async def _get_calendar(hours_ahead: int) -> dict:
    """Fetch calendar events using the existing CalendarTool."""
    try:
        from ...tools.calendar import calendar_tool

        if not settings.tools.calendar_enabled or not settings.tools.calendar_refresh_token:
            return {"events": [], "count": 0, "note": "Calendar not configured"}

        events = await calendar_tool._fetch_events(hours_ahead=hours_ahead, max_results=10)
        return {
            "events": [
                {
                    "summary": e.summary,
                    "start": e.start.isoformat(),
                    "end": e.end.isoformat(),
                    "all_day": e.all_day,
                    "location": e.location,
                }
                for e in events
            ],
            "count": len(events),
        }
    except Exception as e:
        logger.warning("Calendar fetch failed: %s", e)
        return {"events": [], "count": 0, "error": str(e)}


async def _get_weather() -> dict:
    """Fetch current weather using the existing WeatherTool."""
    try:
        from ...tools.weather import weather_tool

        if not settings.tools.weather_enabled:
            return {"note": "Weather not enabled"}

        data = await weather_tool._fetch_weather(
            settings.tools.weather_default_lat,
            settings.tools.weather_default_lon,
        )
        return {
            "temp": data.get("temperature"),
            "unit": data.get("unit", "F"),
            "condition": data.get("condition", "Unknown"),
            "windspeed": data.get("windspeed"),
        }
    except Exception as e:
        logger.warning("Weather fetch failed: %s", e)
        return {"error": str(e)}


async def _get_pending_actions() -> dict:
    """Fetch pending proactive actions from DB."""
    try:
        from ...storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return {"count": 0, "items": []}

        rows = await pool.fetch(
            """
            SELECT action_text, action_type, created_at
            FROM proactive_actions
            WHERE status = 'pending'
            ORDER BY created_at DESC
            LIMIT 5
            """,
        )

        return {
            "count": len(rows),
            "items": [
                {"text": r["action_text"], "type": r["action_type"]}
                for r in rows
            ],
        }
    except Exception as e:
        logger.warning("Pending actions fetch failed: %s", e)
        return {"count": 0, "items": [], "error": str(e)}


def _build_summary(result: dict, security_hours: int) -> str:
    """Build a human-readable morning briefing summary."""
    parts = ["Good morning."]

    # Calendar
    cal = result.get("calendar", {})
    count = cal.get("count", 0)
    if count > 0:
        events = cal.get("events", [])
        event_names = [e["summary"] for e in events[:5]]
        parts.append(f"{count} events today: {', '.join(event_names)}.")
    else:
        parts.append("No events scheduled.")

    # Weather
    wx = result.get("weather", {})
    if "error" not in wx and wx.get("temp") is not None:
        temp = wx["temp"]
        unit = wx.get("unit", "F")
        condition = wx.get("condition", "")
        parts.append(f"Currently {temp} {unit} and {condition.lower()}.")

    # Security
    sec = result.get("security", {})
    if "error" not in sec:
        alerts = sec.get("alerts_overnight", 0)
        unacked = sec.get("unacked", 0)
        vision = sec.get("vision_events", 0)
        if alerts > 0 or vision > 0:
            sec_str = f"{alerts} overnight alerts"
            if unacked > 0:
                sec_str += f" ({unacked} unacked)"
            sec_str += f", {vision} vision events."
            parts.append(sec_str)
        else:
            parts.append("Quiet overnight, no security events.")

    # Device health
    dh = result.get("device_health", {})
    if "error" not in dh:
        issues = dh.get("issues_count", 0)
        if issues > 0:
            parts.append(f"{issues} device issue(s) detected.")
        else:
            parts.append("All devices healthy.")

    # Pending actions
    actions = result.get("actions", {})
    if actions.get("count", 0) > 0:
        items = [a["text"] for a in actions["items"][:3]]
        parts.append(f"Pending: {', '.join(items)}.")

    return " ".join(parts)
