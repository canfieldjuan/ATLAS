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

    @staticmethod
    def _format_date(now: datetime) -> str:
        """Format a date string without platform-specific strftime directives."""
        return f"Today is {now.strftime('%A')}, {now.strftime('%B')} {now.day}, {now.year}."

    @staticmethod
    def _format_time(now: datetime) -> str:
        """Format a 12-hour time string without platform-specific strftime directives."""
        hour_12 = now.hour % 12 or 12
        return f"It's {hour_12}:{now.minute:02d} {now.strftime('%p')}."

    async def execute(self, query: str, match: re.Match) -> SkillResult:
        now = self._now()
        query_lower = query.lower()

        if "date" in query_lower:
            text = self._format_date(now)
        elif "day" in query_lower and "time" not in query_lower:
            text = f"Today is {now.strftime('%A')}, {now.strftime('%B')} {now.day}."
        else:
            text = self._format_time(now)

        return SkillResult(
            success=True,
            response_text=text,
            skill_name=self.name,
        )
