"""
Time skill - handles time, date, and day queries locally.
"""

import re
from datetime import datetime

from .base import Skill, SkillResult


class TimeSkill:
    """Answers time, date, and day queries."""

    name = "time"
    description = "Tells the current time, date, or day of the week"
    patterns = [
        re.compile(r"(?:what(?:'s| is)?\s+)?(?:the\s+)?(?:current\s+)?time(?:\s+(?:is it|right now))?"),
        re.compile(r"(?:what(?:'s| is)?\s+)?(?:the\s+)?(?:today'?s?\s+)?date(?:\s+today)?"),
        re.compile(r"what\s+day\s+(?:is\s+it|of\s+the\s+week)"),
        re.compile(r"(?:tell\s+me\s+)?the\s+time"),
    ]

    def __init__(self, timezone: str = "America/Chicago"):
        self._timezone_name = timezone

    def _now(self) -> datetime:
        """Get current datetime in configured timezone."""
        try:
            from zoneinfo import ZoneInfo
            return datetime.now(ZoneInfo(self._timezone_name))
        except Exception:
            return datetime.now()

    async def execute(self, query: str, match: re.Match) -> SkillResult:
        now = self._now()
        query_lower = query.lower()

        if "date" in query_lower:
            text = now.strftime("Today is %A, %B %-d, %Y.")
        elif "day" in query_lower and "time" not in query_lower:
            text = now.strftime("Today is %A, %B %-d.")
        else:
            text = now.strftime("It's %-I:%M %p.")

        return SkillResult(
            success=True,
            response_text=text,
            skill_name=self.name,
        )
