"""
ToolBridge for PersonaPlex integration.

Monitors conversation text from PersonaPlex and triggers tools
when appropriate intents are detected.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..tools import tool_registry, ToolResult

logger = logging.getLogger("atlas.comms.tool_bridge")


@dataclass
class ConversationContext:
    """Accumulated context from conversation."""

    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    customer_email: Optional[str] = None
    service_address: Optional[str] = None
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None
    service_type: Optional[str] = None
    notes: Optional[str] = None
    accumulated_text: str = ""
    turn_count: int = 0

    def to_booking_params(self) -> dict[str, Any]:
        """Convert context to book_appointment parameters."""
        params = {}
        if self.customer_name:
            params["customer_name"] = self.customer_name
        if self.customer_phone:
            params["customer_phone"] = self.customer_phone
        if self.customer_email:
            params["customer_email"] = self.customer_email
        if self.service_address:
            params["address"] = self.service_address
        if self.preferred_date:
            params["date"] = self.preferred_date
        if self.preferred_time:
            params["time"] = self.preferred_time
        if self.service_type:
            params["service_type"] = self.service_type
        if self.notes:
            params["notes"] = self.notes
        return params


BOOKING_TRIGGER_PATTERNS = [
    r"let me book that",
    r"i('ll| will) book that",
    r"booking (it|that|this)",
    r"schedule (it|that|this)",
    r"confirm(ing)? (the|this|that|your) appointment",
    r"go ahead and book",
    r"book (it|that|this) for you",
]

NAME_PATTERNS = [
    r"(?:my name is|this is|i'm|i am) ([A-Z][a-z]+ [A-Z][a-z]+)",
    r"([A-Z][a-z]+ [A-Z][a-z]+) (?:speaking|here|calling)",
]

PHONE_PATTERNS = [
    r"(?:number is|reach me at|call me at) (\d{3}[-.\s]?\d{3}[-.\s]?\d{4})",
    r"(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})",
]

ADDRESS_PATTERNS = [
    r"(?:address is|located at|at) (\d+[^,]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|way|court|ct|boulevard|blvd)[^,]*)",
]

DATE_PATTERNS = [
    r"(?:on|for) (tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
    r"(?:on|for) (next (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))",
    r"(?:on|for) (\d{1,2}/\d{1,2}(?:/\d{2,4})?)",
]

TIME_PATTERNS = [
    r"(?:at|around) (\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.))",
    r"(?:in the) (morning|afternoon|evening)",
]


class ToolBridge:
    """Bridges PersonaPlex conversation with Atlas tool system."""

    def __init__(
        self,
        on_tool_result: Optional[Callable[[str, ToolResult], None]] = None,
    ):
        self._context = ConversationContext()
        self._on_tool_result = on_tool_result
        self._pending_tool: Optional[str] = None
        self._tool_lock = asyncio.Lock()

    @property
    def context(self) -> ConversationContext:
        """Get the current conversation context."""
        return self._context

    def reset(self) -> None:
        """Reset conversation context for new call."""
        self._context = ConversationContext()
        self._pending_tool = None
        logger.debug("ToolBridge context reset")

    def process_text(self, text: str) -> None:
        """Process text token from PersonaPlex."""
        self._context.accumulated_text += text
        self._extract_entities(text)

    def _extract_entities(self, text: str) -> None:
        """Extract entities from text using pattern matching."""
        text_lower = text.lower()

        for pattern in NAME_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not self._context.customer_name:
                self._context.customer_name = match.group(1)
                logger.debug("Extracted name: %s", self._context.customer_name)
                break

        for pattern in PHONE_PATTERNS:
            match = re.search(pattern, text)
            if match and not self._context.customer_phone:
                self._context.customer_phone = match.group(1)
                logger.debug("Extracted phone: %s", self._context.customer_phone)
                break

        for pattern in ADDRESS_PATTERNS:
            match = re.search(pattern, text_lower)
            if match and not self._context.service_address:
                self._context.service_address = match.group(1)
                logger.debug("Extracted address: %s", self._context.service_address)
                break

        for pattern in DATE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match and not self._context.preferred_date:
                self._context.preferred_date = match.group(1)
                logger.debug("Extracted date: %s", self._context.preferred_date)
                break

        for pattern in TIME_PATTERNS:
            match = re.search(pattern, text_lower)
            if match and not self._context.preferred_time:
                self._context.preferred_time = match.group(1)
                logger.debug("Extracted time: %s", self._context.preferred_time)
                break

    def detect_booking_intent(self) -> bool:
        """Check if accumulated text indicates booking intent."""
        text_lower = self._context.accumulated_text.lower()
        for pattern in BOOKING_TRIGGER_PATTERNS:
            if re.search(pattern, text_lower):
                logger.info("Booking intent detected: %s", pattern)
                return True
        return False

    def has_required_booking_info(self) -> bool:
        """Check if we have minimum info for booking."""
        ctx = self._context
        has_name = ctx.customer_name is not None
        has_phone = ctx.customer_phone is not None
        has_date = ctx.preferred_date is not None
        has_time = ctx.preferred_time is not None
        return has_name and has_phone and has_date and has_time

    async def check_and_execute_tools(self) -> Optional[ToolResult]:
        """Check for tool triggers and execute if appropriate."""
        async with self._tool_lock:
            if self._pending_tool:
                return None

            if self.detect_booking_intent() and self.has_required_booking_info():
                self._pending_tool = "book_appointment"
                result = await self._execute_booking()
                self._pending_tool = None
                return result

        return None

    async def _execute_booking(self) -> ToolResult:
        """Execute booking with accumulated context."""
        params = self._context.to_booking_params()
        logger.info("Executing book_appointment with params: %s", params)

        result = await tool_registry.execute("book_appointment", params)

        if self._on_tool_result:
            self._on_tool_result("book_appointment", result)

        return result

    def get_missing_info_prompt(self) -> Optional[str]:
        """Get prompt for missing required information."""
        missing = []
        if not self._context.customer_name:
            missing.append("name")
        if not self._context.customer_phone:
            missing.append("phone number")
        if not self._context.preferred_date:
            missing.append("preferred date")
        if not self._context.preferred_time:
            missing.append("preferred time")

        if not missing:
            return None

        if len(missing) == 1:
            return f"Could you please provide your {missing[0]}?"
        return f"I still need your {', '.join(missing[:-1])}, and {missing[-1]}."

    def format_tool_result(self, tool_name: str, result: ToolResult) -> str:
        """Format tool result for speech injection."""
        if tool_name == "book_appointment":
            if result.success:
                return result.message or "Your appointment has been booked."
            return f"I could not complete the booking. {result.message}"
        return result.message or ""
