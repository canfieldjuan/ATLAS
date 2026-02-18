"""
Time tool for current date and time information.

Provides current time, date, day of week, etc.
No external API required.
"""

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from ..config import settings
from .base import ToolParameter, ToolResult

logger = logging.getLogger("atlas.tools.time")


class TimeTool:
    """Current time and date tool."""

    def __init__(self) -> None:
        self._timezone = settings.reminder.default_timezone

    @property
    def name(self) -> str:
        return "get_time"

    @property
    def description(self) -> str:
        return "Get current time, date, and day of week"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="timezone",
                param_type="string",
                description="Timezone (e.g. America/Chicago)",
                required=False,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["time", "current time", "what time", "date", "today"]

    @property
    def category(self) -> str:
        return "utility"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute time query."""
        try:
            tz_name = params.get("timezone", self._timezone)
            tz = ZoneInfo(tz_name)
            now = datetime.now(tz)

            # Use %-I to avoid leading zero (TTS reads "04" as "zero four")
            time_str = now.strftime("%-I:%M %p")

            # TTS-friendly spoken time (e.g. "6 o'clock" not "6:00")
            hour = now.hour % 12 or 12
            minute = now.minute
            period = "AM" if now.hour < 12 else "PM"
            if minute == 0:
                spoken_time = f"{hour} o'clock {period}"
            elif minute < 10:
                spoken_time = f"{hour} oh {minute} {period}"
            else:
                spoken_time = f"{hour} {minute} {period}"

            data = {
                "time": time_str,
                "time_24h": now.strftime("%H:%M"),
                "date": now.strftime("%B %-d, %Y"),
                "day_of_week": now.strftime("%A"),
                "timezone": tz_name,
                "iso": now.isoformat(),
            }

            message = f"It is {spoken_time}."

            return ToolResult(
                success=True,
                data=data,
                message=message,
            )
        except Exception as e:
            logger.exception("Time tool error")
            return ToolResult(
                success=False,
                error="EXECUTION_ERROR",
                message=str(e),
            )

    async def close(self) -> None:
        """No cleanup needed."""
        pass


# Module-level instance
time_tool = TimeTool()
