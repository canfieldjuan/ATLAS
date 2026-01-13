"""
Time tool for current date and time information.

Provides current time, date, day of week, etc.
No external API required.
"""

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from .base import ToolParameter, ToolResult

logger = logging.getLogger("atlas.tools.time")


class TimeTool:
    """Current time and date tool."""

    def __init__(self) -> None:
        self._timezone = "America/Chicago"

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

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute time query."""
        try:
            tz_name = params.get("timezone", self._timezone)
            tz = ZoneInfo(tz_name)
            now = datetime.now(tz)

            # Use %-I to avoid leading zero (TTS reads "04" as "zero four")
            time_str = now.strftime("%-I:%M %p")

            data = {
                "time": time_str,
                "time_24h": now.strftime("%H:%M"),
                "date": now.strftime("%B %-d, %Y"),
                "day_of_week": now.strftime("%A"),
                "timezone": tz_name,
                "iso": now.isoformat(),
            }

            message = f"It is {time_str} on {data['day_of_week']}, {data['date']}."

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
