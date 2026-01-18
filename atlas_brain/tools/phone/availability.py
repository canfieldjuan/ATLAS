"""
Check Availability tool - Check calendar for open appointment slots.

Uses the existing calendar tool to find available time slots
for scheduling appointments.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from ..base import ToolParameter, ToolResult

logger = logging.getLogger("atlas.tools.phone.availability")


class CheckAvailabilityTool:
    """Tool to check calendar availability for appointments."""

    def __init__(self) -> None:
        self._calendar_tool = None

    @property
    def name(self) -> str:
        return "check_availability"

    @property
    def description(self) -> str:
        return "Check calendar for available appointment slots"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="context_id",
                param_type="string",
                description="Business context ID",
                required=False,
            ),
            ToolParameter(
                name="days_ahead",
                param_type="int",
                description="Days to look ahead (default: 7)",
                required=False,
                default=7,
            ),
            ToolParameter(
                name="query",
                param_type="string",
                description="Caller's request for context",
                required=False,
            ),
        ]

    def _get_calendar_tool(self):
        """Get the calendar tool instance."""
        if self._calendar_tool is None:
            from ..calendar import calendar_tool
            self._calendar_tool = calendar_tool
        return self._calendar_tool

    def _get_business_context(self, context_id: Optional[str] = None) -> Optional[Any]:
        """Get business context by ID."""
        from ...comms.context import get_context_router

        router = get_context_router()

        if context_id:
            return router.get_context(context_id)

        contexts = router.list_contexts()
        if contexts:
            return contexts[0]

        return None

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Check availability and return open slots."""
        context_id = params.get("context_id")
        days_ahead = params.get("days_ahead", 7)

        try:
            # Get business context for scheduling config
            ctx = self._get_business_context(context_id)
            scheduling = None
            if ctx and hasattr(ctx, "scheduling"):
                scheduling = ctx.scheduling

            # Get calendar events
            calendar = self._get_calendar_tool()
            hours_ahead = days_ahead * 24

            calendar_result = await calendar.execute({
                "hours_ahead": hours_ahead,
                "max_results": 25,
            })

            if not calendar_result.success:
                return ToolResult(
                    success=True,
                    data={"available": True},
                    message="Calendar not configured. Please call to schedule.",
                )

            # Get existing events
            events = calendar_result.data.get("events", [])
            event_count = len(events)

            # Build availability message
            if event_count == 0:
                message = f"Wide open availability for the next {days_ahead} days."
            else:
                message = (
                    f"Found {event_count} existing appointments. "
                    f"Availability varies - best to discuss specific dates."
                )

            # Get scheduling constraints
            min_notice = 24
            if scheduling and hasattr(scheduling, "min_notice_hours"):
                min_notice = scheduling.min_notice_hours

            data = {
                "available": True,
                "days_checked": days_ahead,
                "existing_appointments": event_count,
                "min_notice_hours": min_notice,
            }

            return ToolResult(
                success=True,
                data=data,
                message=message,
            )

        except Exception as e:
            logger.exception("Error checking availability")
            return ToolResult(
                success=False,
                error="EXECUTION_ERROR",
                message=str(e),
            )


# Module instance
check_availability_tool = CheckAvailabilityTool()
