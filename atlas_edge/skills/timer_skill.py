"""
Timer skill - set, check, and cancel countdown timers locally.
"""

import asyncio
import logging
import re
import time
from typing import Callable, Optional

from .base import SkillResult

logger = logging.getLogger("atlas.edge.skills.timer")

# Duration unit multipliers in seconds
_UNIT_SECONDS = {
    "second": 1,
    "seconds": 1,
    "minute": 60,
    "minutes": 60,
    "hour": 3600,
    "hours": 3600,
}

_MAX_TIMER_SECONDS = 24 * 3600  # 24 hours


class TimerSkill:
    """Manages countdown timers with asyncio tasks."""

    name = "timer"
    description = "Set, check, and cancel countdown timers"
    patterns = [
        re.compile(r"set\s+(?:a\s+)?timer\s+(?:for\s+)?(\d+)\s*(seconds?|minutes?|hours?)"),
        re.compile(r"(?:how\s+much\s+time|how\s+long)\s+(?:is\s+)?(?:left|remaining)(?:\s+on\s+(?:the\s+)?timer)?"),
        re.compile(r"(?:cancel|stop|clear|delete)\s+(?:the\s+)?timer(?:s)?"),
    ]

    def __init__(
        self,
        max_timers: int = 10,
        on_timer_done: Optional[Callable[[str], None]] = None,
    ):
        self._max_timers = max_timers
        self._on_timer_done = on_timer_done
        # name -> (task, end_time, duration_label)
        self._timers: dict[str, tuple[asyncio.Task, float, str]] = {}
        self._counter = 0

    def shutdown(self) -> None:
        """Cancel all active timers."""
        for name, (task, _, _) in list(self._timers.items()):
            if not task.done():
                task.cancel()
        self._timers.clear()

    async def execute(self, query: str, match: re.Match) -> SkillResult:
        query_lower = query.lower()

        # Set a timer
        if match.re == self.patterns[0]:
            return await self._set_timer(match)

        # Check remaining time
        if match.re == self.patterns[1]:
            return self._check_timers()

        # Cancel timer(s)
        if match.re == self.patterns[2]:
            return self._cancel_timers()

        return SkillResult(success=False, skill_name=self.name, error="unmatched")

    async def _set_timer(self, match: re.Match) -> SkillResult:
        amount = int(match.group(1))
        unit = match.group(2).lower()

        if amount <= 0:
            return SkillResult(
                success=False,
                response_text="Timer duration must be greater than zero.",
                skill_name=self.name,
            )

        if len(self._timers) >= self._max_timers:
            return SkillResult(
                success=False,
                response_text=f"You already have {len(self._timers)} timers running. Cancel one first.",
                skill_name=self.name,
            )

        multiplier = _UNIT_SECONDS.get(unit, 60)
        total_seconds = amount * multiplier

        if total_seconds > _MAX_TIMER_SECONDS:
            return SkillResult(
                success=False,
                response_text="Timers are limited to 24 hours.",
                skill_name=self.name,
            )

        self._counter += 1
        timer_name = f"timer_{self._counter}"
        label = f"{amount} {unit}"

        task = asyncio.create_task(self._run_timer(timer_name, total_seconds, label))
        self._timers[timer_name] = (task, time.time() + total_seconds, label)

        return SkillResult(
            success=True,
            response_text=f"Timer set for {label}.",
            skill_name=self.name,
        )

    async def _run_timer(self, name: str, seconds: float, label: str) -> None:
        try:
            await asyncio.sleep(seconds)
            logger.info("Timer '%s' (%s) finished", name, label)
            if self._on_timer_done:
                cb = self._on_timer_done
                msg = f"Your {label} timer is done!"
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, cb, msg)
        except asyncio.CancelledError:
            logger.info("Timer '%s' cancelled", name)
        finally:
            self._timers.pop(name, None)

    def _check_timers(self) -> SkillResult:
        # Clean up finished timers
        self._timers = {
            k: v for k, v in self._timers.items() if not v[0].done()
        }

        if not self._timers:
            return SkillResult(
                success=True,
                response_text="No active timers.",
                skill_name=self.name,
            )

        now = time.time()
        parts = []
        for name, (task, end_time, label) in self._timers.items():
            remaining = max(0, end_time - now)
            mins, secs = divmod(int(remaining), 60)
            hours, mins = divmod(mins, 60)
            if hours:
                parts.append(f"{label}: {hours}h {mins}m left")
            elif mins:
                parts.append(f"{label}: {mins}m {secs}s left")
            else:
                parts.append(f"{label}: {secs}s left")

        return SkillResult(
            success=True,
            response_text="; ".join(parts) + ".",
            skill_name=self.name,
        )

    def _cancel_timers(self) -> SkillResult:
        count = 0
        for name, (task, _, _) in list(self._timers.items()):
            if not task.done():
                task.cancel()
                count += 1
        self._timers.clear()

        if count == 0:
            return SkillResult(
                success=True,
                response_text="No active timers to cancel.",
                skill_name=self.name,
            )

        return SkillResult(
            success=True,
            response_text=f"Cancelled {count} timer{'s' if count != 1 else ''}.",
            skill_name=self.name,
        )
