"""
Status skill - reports system status, connectivity, and uptime.
"""

import re
import time

from .base import SkillResult

# Module-level start time for uptime calculation
_start_time = time.time()


def _format_uptime(seconds: float) -> str:
    """Format seconds into human-readable uptime."""
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if mins:
        parts.append(f"{mins}m")
    if not parts:
        parts.append(f"{int(secs)}s")
    return " ".join(parts)


class StatusSkill:
    """Reports system status and connectivity."""

    name = "status"
    description = "Reports brain connection, HA status, and uptime"
    patterns = [
        re.compile(r"(?:system\s+)?status"),
        re.compile(r"are\s+you\s+(?:online|working|there|okay|ok)"),
        re.compile(r"how\s+are\s+you(?:\s+doing)?"),
    ]

    async def execute(self, query: str, match: re.Match) -> SkillResult:
        uptime = _format_uptime(time.time() - _start_time)

        # Check brain connection status
        brain_status = "unknown"
        try:
            from ..brain.connection import get_brain_connection
            conn = await get_brain_connection()
            brain_status = "connected" if conn.is_connected else "disconnected"
        except Exception:
            brain_status = "unavailable"

        # Check Home Assistant status
        ha_status = "unknown"
        try:
            from ..capabilities.homeassistant import get_homeassistant
            ha = await get_homeassistant()
            ha_status = "connected" if ha and ha.is_connected else "disconnected"
        except Exception:
            ha_status = "unavailable"

        # Friendly response
        if match.re == self.patterns[2]:
            # "how are you doing" - more conversational
            if brain_status == "connected":
                text = f"I'm doing well! Brain is connected, uptime {uptime}."
            else:
                text = f"I'm running in offline mode. Brain is {brain_status}, but I can still help with device control and local skills. Uptime: {uptime}."
        else:
            text = f"Brain: {brain_status}. Home Assistant: {ha_status}. Uptime: {uptime}."

        return SkillResult(
            success=True,
            response_text=text,
            skill_name=self.name,
        )
