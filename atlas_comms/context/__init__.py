"""
Context routing for incoming calls and messages.

Determines which business context should handle a communication
based on the phone number it was received on.
"""

import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from ..core.config import BusinessContext, BusinessHours, DEFAULT_PERSONAL_CONTEXT

logger = logging.getLogger("atlas.comms.context")


class ContextRouter:
    """
    Routes incoming communications to the appropriate business context.

    Maintains a registry of contexts and their associated phone numbers.
    """

    def __init__(self):
        self._contexts: dict[str, BusinessContext] = {}
        self._number_to_context: dict[str, str] = {}

        # Register default personal context
        self.register_context(DEFAULT_PERSONAL_CONTEXT)

    def register_context(self, context: BusinessContext) -> None:
        """Register a business context."""
        self._contexts[context.id] = context

        # Map phone numbers to this context
        for number in context.phone_numbers:
            normalized = self._normalize_number(number)
            self._number_to_context[normalized] = context.id
            logger.info(
                "Registered number %s for context '%s'",
                normalized,
                context.id,
            )

        logger.info("Registered context: %s (%s)", context.id, context.name)

    def unregister_context(self, context_id: str) -> bool:
        """Remove a business context."""
        if context_id not in self._contexts:
            return False

        context = self._contexts.pop(context_id)

        # Remove number mappings
        for number in context.phone_numbers:
            normalized = self._normalize_number(number)
            self._number_to_context.pop(normalized, None)

        logger.info("Unregistered context: %s", context_id)
        return True

    def get_context(self, context_id: str) -> Optional[BusinessContext]:
        """Get a context by ID."""
        return self._contexts.get(context_id)

    def get_context_for_number(self, to_number: str) -> BusinessContext:
        """
        Get the business context for a phone number.

        Returns the default personal context if no specific context is found.
        """
        normalized = self._normalize_number(to_number)
        context_id = self._number_to_context.get(normalized)

        if context_id:
            context = self._contexts.get(context_id)
            if context:
                return context

        # Fall back to default
        logger.debug(
            "No context found for number %s, using default",
            normalized,
        )
        return self._contexts.get("personal", DEFAULT_PERSONAL_CONTEXT)

    def list_contexts(self) -> list[BusinessContext]:
        """List all registered contexts."""
        return list(self._contexts.values())

    def is_within_hours(
        self,
        context: BusinessContext,
        at_time: Optional[datetime] = None,
    ) -> bool:
        """
        Check if the current time is within business hours.

        Args:
            context: The business context to check
            at_time: Optional time to check (defaults to now)

        Returns:
            True if within business hours, False otherwise
        """
        if at_time is None:
            # Get current time in business timezone
            tz = ZoneInfo(context.hours.timezone)
            at_time = datetime.now(tz)
        else:
            # Convert to business timezone
            tz = ZoneInfo(context.hours.timezone)
            at_time = at_time.astimezone(tz)

        # Get day of week (0 = Monday)
        day = at_time.weekday()
        hours = context.hours

        # Get open/close times for this day
        day_names = [
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday"
        ]
        day_name = day_names[day]

        open_time = getattr(hours, f"{day_name}_open")
        close_time = getattr(hours, f"{day_name}_close")

        if open_time is None or close_time is None:
            # Closed this day
            return False

        # Parse times
        open_hour, open_min = map(int, open_time.split(":"))
        close_hour, close_min = map(int, close_time.split(":"))

        current_minutes = at_time.hour * 60 + at_time.minute
        open_minutes = open_hour * 60 + open_min
        close_minutes = close_hour * 60 + close_min

        return open_minutes <= current_minutes < close_minutes

    def get_business_status(
        self,
        context: BusinessContext,
    ) -> dict:
        """
        Get the current business status.

        Returns dict with:
        - is_open: bool
        - next_open: datetime or None
        - message: str (appropriate greeting or after-hours message)
        """
        tz = ZoneInfo(context.hours.timezone)
        now = datetime.now(tz)

        is_open = self.is_within_hours(context, now)

        if is_open:
            return {
                "is_open": True,
                "next_open": None,
                "message": context.greeting,
            }
        else:
            # Find next open time
            next_open = self._find_next_open_time(context, now)
            return {
                "is_open": False,
                "next_open": next_open,
                "message": context.after_hours_message,
            }

    def _find_next_open_time(
        self,
        context: BusinessContext,
        from_time: datetime,
    ) -> Optional[datetime]:
        """Find the next time the business opens."""
        tz = ZoneInfo(context.hours.timezone)
        hours = context.hours

        day_names = [
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday"
        ]

        # Check next 7 days
        for days_ahead in range(7):
            check_date = from_time.date()
            if days_ahead > 0:
                from datetime import timedelta
                check_date = (from_time + timedelta(days=days_ahead)).date()

            day_idx = check_date.weekday()
            day_name = day_names[day_idx]

            open_time = getattr(hours, f"{day_name}_open")
            if open_time is None:
                continue

            open_hour, open_min = map(int, open_time.split(":"))
            open_dt = datetime(
                check_date.year,
                check_date.month,
                check_date.day,
                open_hour,
                open_min,
                tzinfo=tz,
            )

            if open_dt > from_time:
                return open_dt

        return None

    def _normalize_number(self, number: str) -> str:
        """
        Normalize a phone number for comparison.

        Strips formatting, ensures E.164 format.
        """
        # Remove all non-digit characters except leading +
        cleaned = ""
        for i, char in enumerate(number):
            if char == "+" and i == 0:
                cleaned += char
            elif char.isdigit():
                cleaned += char

        # Add country code if missing (assume US)
        if not cleaned.startswith("+"):
            if len(cleaned) == 10:
                cleaned = "+1" + cleaned
            elif len(cleaned) == 11 and cleaned.startswith("1"):
                cleaned = "+" + cleaned

        return cleaned


# Global router instance
_context_router: Optional[ContextRouter] = None


def get_context_router() -> ContextRouter:
    """Get or create the global context router."""
    global _context_router
    if _context_router is None:
        _context_router = ContextRouter()
    return _context_router
