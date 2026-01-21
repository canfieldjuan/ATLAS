"""
Reminder tool for natural language reminder creation.

Parses time expressions like "in 2 hours", "tomorrow at 5pm", "next Tuesday"
using dateparser and creates reminders via the ReminderService.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import dateparser

from ..config import settings
from ..services.reminders import get_reminder_service
from ..storage import DatabaseUnavailableError, DatabaseOperationError
from .base import ToolParameter, ToolResult

logger = logging.getLogger("atlas.tools.reminder")


class ReminderTool:
    """
    Tool for creating and managing reminders via natural language.

    Supports:
    - Natural language time parsing ("in 5 minutes", "tomorrow at 3pm")
    - Recurring reminders ("every day", "every week")
    - Listing active reminders
    - Completing/deleting reminders
    """

    def __init__(self) -> None:
        # dateparser settings for best results
        self._parser_settings = {
            "PREFER_DATES_FROM": "future",
            "PREFER_DAY_OF_MONTH": "first",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TIMEZONE": settings.reminder.default_timezone,
        }

    @property
    def name(self) -> str:
        return "set_reminder"

    @property
    def description(self) -> str:
        return "Create a reminder for a specific time. Use for requests like 'remind me to...', 'set a reminder for...'"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="message",
                param_type="string",
                description="What to remind about (e.g., 'call mom', 'take medicine')",
                required=True,
            ),
            ToolParameter(
                name="when",
                param_type="string",
                description="When to remind (e.g., 'in 30 minutes', 'at 5pm', 'tomorrow morning')",
                required=True,
            ),
            ToolParameter(
                name="repeat",
                param_type="string",
                description="Optional repeat pattern: 'daily', 'weekly', 'monthly'",
                required=False,
                default=None,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["reminder", "remind", "remind me", "set reminder"]

    @property
    def category(self) -> str:
        return "utility"

    def _normalize_time_text(self, when_text: str) -> str:
        """
        Normalize time text for better dateparser compatibility.

        Handles patterns like 'next Tuesday' that dateparser struggles with.
        """
        import re

        text = when_text.strip().lower()

        # Handle "now" variants that dateparser doesn't recognize
        now_variants = ("right now", "immediately", "asap", "straight away")
        if text in now_variants:
            return "in 2 seconds"

        weekdays = (
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday"
        )

        pattern = r"^next\s+(" + "|".join(weekdays) + r")\b(.*)$"
        match = re.match(pattern, text, re.IGNORECASE)

        if match:
            weekday = match.group(1)
            rest = match.group(2).strip()
            if rest:
                return f"{weekday} {rest}"
            return weekday

        return when_text

    def _parse_time(self, when_text: str) -> Optional[datetime]:
        """
        Parse natural language time expression.

        Returns timezone-aware datetime or None if parsing fails.
        """
        try:
            normalized = self._normalize_time_text(when_text)
            parsed = dateparser.parse(
                normalized,
                settings=self._parser_settings,
            )

            if parsed:
                # Ensure timezone-aware
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed

            return None

        except Exception as e:
            logger.warning("Time parsing error for '%s': %s", when_text, e)
            return None

    def _validate_repeat(self, repeat: Optional[str]) -> Optional[str]:
        """Validate and normalize repeat pattern."""
        if not repeat:
            return None

        repeat_lower = repeat.lower().strip()

        # Normalize common variations
        if repeat_lower in ("daily", "every day", "each day"):
            return "daily"
        elif repeat_lower in ("weekly", "every week", "each week"):
            return "weekly"
        elif repeat_lower in ("monthly", "every month", "each month"):
            return "monthly"
        elif repeat_lower in ("yearly", "every year", "each year", "annually"):
            return "yearly"
        else:
            logger.warning("Unknown repeat pattern: %s", repeat)
            return None

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the reminder creation."""
        if not settings.reminder.enabled:
            return ToolResult(
                success=False,
                error="TOOL_DISABLED",
                message="Reminder system is disabled",
            )

        message = params.get("message", "").strip()
        when_text = params.get("when", "").strip()
        repeat = params.get("repeat")

        if not message:
            return ToolResult(
                success=False,
                error="MISSING_MESSAGE",
                message="Please specify what you want to be reminded about",
            )

        if not when_text:
            return ToolResult(
                success=False,
                error="MISSING_TIME",
                message="Please specify when you want to be reminded",
            )

        # Parse the time expression
        due_at = self._parse_time(when_text)
        if not due_at:
            return ToolResult(
                success=False,
                error="INVALID_TIME",
                message=f"I couldn't understand '{when_text}'. Try something like 'in 30 minutes' or 'tomorrow at 5pm'",
            )

        # For immediate reminders ("now"), adjust to near-future
        now = datetime.now(timezone.utc)
        if due_at <= now:
            due_at = now + timedelta(seconds=2)
            logger.info("Adjusted immediate reminder to near-future: %s", due_at)

        # Validate repeat pattern
        repeat_pattern = self._validate_repeat(repeat)

        try:
            service = get_reminder_service()
            reminder = await service.create_reminder(
                message=message,
                due_at=due_at,
                repeat_pattern=repeat_pattern,
                source="voice",
            )

            if not reminder:
                return ToolResult(
                    success=False,
                    error="CREATE_FAILED",
                    message="Failed to create reminder. Please try again.",
                )

            # Format the response
            time_str = self._format_time(due_at)
            response_msg = f"I'll remind you to {message} {time_str}"

            if repeat_pattern:
                response_msg += f" ({repeat_pattern})"

            return ToolResult(
                success=True,
                data={
                    "reminder_id": str(reminder.id),
                    "message": message,
                    "due_at": due_at.isoformat(),
                    "repeat_pattern": repeat_pattern,
                },
                message=response_msg,
            )

        except DatabaseUnavailableError:
            logger.error("Database unavailable for reminder creation")
            return ToolResult(
                success=False,
                error="DATABASE_UNAVAILABLE",
                message="Database is currently unavailable. Please try again later.",
            )
        except DatabaseOperationError as e:
            logger.error("Database error creating reminder: %s", e.cause)
            return ToolResult(
                success=False,
                error="DATABASE_ERROR",
                message="Failed to save reminder. Please try again.",
            )
        except Exception as e:
            logger.exception("Reminder creation error")
            return ToolResult(
                success=False,
                error="EXECUTION_ERROR",
                message=str(e),
            )

    def _format_time(self, due_at: datetime) -> str:
        """Format datetime for human-readable response."""
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        diff = due_at - now

        # Helper for cross-platform time formatting
        hour = due_at.hour % 12 or 12
        minute = due_at.strftime('%M')
        ampm = 'AM' if due_at.hour < 12 else 'PM'
        time_str = f"{hour}:{minute} {ampm}"

        # Same day
        if due_at.date() == now.date():
            return f"today at {time_str}"

        # Tomorrow (use timedelta to avoid month boundary issues)
        tomorrow = (now + timedelta(days=1)).date()
        if due_at.date() == tomorrow:
            return f"tomorrow at {time_str}"

        # Within a week
        if diff.days < 7:
            return f"on {due_at.strftime('%A')} at {time_str}"

        # Further out
        day = due_at.day
        return f"on {due_at.strftime('%B')} {day} at {time_str}"


class ListRemindersTool:
    """Tool for listing active reminders."""

    @property
    def name(self) -> str:
        return "list_reminders"

    @property
    def description(self) -> str:
        return "List your upcoming reminders"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="limit",
                param_type="int",
                description="Maximum number of reminders to return (default: 5)",
                required=False,
                default=5,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["reminders", "my reminders", "list reminders", "show reminders"]

    @property
    def category(self) -> str:
        return "utility"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the list reminders query."""
        if not settings.reminder.enabled:
            return ToolResult(
                success=False,
                error="TOOL_DISABLED",
                message="Reminder system is disabled",
            )

        limit = min(params.get("limit", 5), 20)

        try:
            service = get_reminder_service()
            reminders = await service.list_reminders(limit=limit)

            if not reminders:
                return ToolResult(
                    success=True,
                    data={"reminders": [], "count": 0},
                    message="You have no upcoming reminders",
                )

            reminder_list = []
            now = datetime.now(timezone.utc)

            for r in reminders:
                due_at = r.due_at
                if due_at.tzinfo is None:
                    due_at = due_at.replace(tzinfo=timezone.utc)

                reminder_list.append({
                    "id": str(r.id),
                    "message": r.message,
                    "due_at": due_at.isoformat(),
                    "repeat_pattern": r.repeat_pattern,
                })

            # Format message
            if len(reminders) == 1:
                r = reminders[0]
                msg = f"You have 1 reminder: {r.message}"
            else:
                msg = f"You have {len(reminders)} reminders"

            return ToolResult(
                success=True,
                data={"reminders": reminder_list, "count": len(reminder_list)},
                message=msg,
            )

        except DatabaseUnavailableError:
            logger.error("Database unavailable for listing reminders")
            return ToolResult(
                success=False,
                error="DATABASE_UNAVAILABLE",
                message="Database is currently unavailable. Please try again later.",
            )
        except DatabaseOperationError as e:
            logger.error("Database error listing reminders: %s", e.cause)
            return ToolResult(
                success=False,
                error="DATABASE_ERROR",
                message="Failed to retrieve reminders. Please try again.",
            )
        except Exception as e:
            logger.exception("List reminders error")
            return ToolResult(
                success=False,
                error="EXECUTION_ERROR",
                message=str(e),
            )


class CompleteReminderTool:
    """Tool for completing/dismissing a reminder."""

    @property
    def name(self) -> str:
        return "complete_reminder"

    @property
    def description(self) -> str:
        return "Mark a reminder as complete or dismiss it"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="reminder_id",
                param_type="string",
                description="The ID of the reminder to complete",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="all",
                param_type="boolean",
                description="Complete all reminders (use with caution)",
                required=False,
                default=False,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["done", "dismiss reminder", "clear reminder", "complete reminder"]

    @property
    def category(self) -> str:
        return "utility"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute the complete reminder action."""
        if not settings.reminder.enabled:
            return ToolResult(
                success=False,
                error="TOOL_DISABLED",
                message="Reminder system is disabled",
            )

        reminder_id = params.get("reminder_id")
        complete_all = params.get("all", False)

        if not reminder_id and not complete_all:
            # Try to complete the next upcoming reminder
            try:
                service = get_reminder_service()
                next_reminder = await service.get_next_reminder()

                if not next_reminder:
                    return ToolResult(
                        success=False,
                        error="NO_REMINDERS",
                        message="You have no active reminders to complete",
                    )

                success = await service.complete_reminder(next_reminder.id)
                if success:
                    return ToolResult(
                        success=True,
                        data={"completed_id": str(next_reminder.id)},
                        message=f"Completed reminder: {next_reminder.message}",
                    )
                else:
                    return ToolResult(
                        success=False,
                        error="COMPLETE_FAILED",
                        message="Failed to complete the reminder",
                    )

            except DatabaseUnavailableError:
                logger.error("Database unavailable for completing reminder")
                return ToolResult(
                    success=False,
                    error="DATABASE_UNAVAILABLE",
                    message="Database is currently unavailable. Please try again later.",
                )
            except DatabaseOperationError as e:
                logger.error("Database error completing reminder: %s", e.cause)
                return ToolResult(
                    success=False,
                    error="DATABASE_ERROR",
                    message="Failed to complete reminder. Please try again.",
                )
            except Exception as e:
                logger.exception("Complete reminder error")
                return ToolResult(
                    success=False,
                    error="EXECUTION_ERROR",
                    message=str(e),
                )

        # Complete specific reminder by ID
        if reminder_id:
            try:
                from uuid import UUID
                service = get_reminder_service()
                success = await service.complete_reminder(UUID(reminder_id))

                if success:
                    return ToolResult(
                        success=True,
                        data={"completed_id": reminder_id},
                        message="Reminder completed",
                    )
                else:
                    return ToolResult(
                        success=False,
                        error="NOT_FOUND",
                        message="Reminder not found or already completed",
                    )

            except ValueError:
                return ToolResult(
                    success=False,
                    error="INVALID_ID",
                    message="Invalid reminder ID format",
                )
            except DatabaseUnavailableError:
                logger.error("Database unavailable for completing reminder")
                return ToolResult(
                    success=False,
                    error="DATABASE_UNAVAILABLE",
                    message="Database is currently unavailable. Please try again later.",
                )
            except DatabaseOperationError as e:
                logger.error("Database error completing reminder: %s", e.cause)
                return ToolResult(
                    success=False,
                    error="DATABASE_ERROR",
                    message="Failed to complete reminder. Please try again.",
                )
            except Exception as e:
                logger.exception("Complete reminder error")
                return ToolResult(
                    success=False,
                    error="EXECUTION_ERROR",
                    message=str(e),
                )

        return ToolResult(
            success=False,
            error="INVALID_PARAMS",
            message="Please specify which reminder to complete",
        )


# Module-level instances
reminder_tool = ReminderTool()
list_reminders_tool = ListRemindersTool()
complete_reminder_tool = CompleteReminderTool()
