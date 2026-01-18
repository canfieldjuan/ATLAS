"""
Book Appointment tool - Create calendar appointments.

Creates calendar events for scheduled appointments.
Uses the Google Calendar API through the existing calendar tool setup.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx

from ..base import ToolParameter, ToolResult

logger = logging.getLogger("atlas.tools.phone.booking")

CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"


class BookAppointmentTool:
    """Tool to book appointments on the calendar."""

    def __init__(self) -> None:
        self._calendar_tool = None
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def name(self) -> str:
        return "book_appointment"

    @property
    def description(self) -> str:
        return "Book an appointment on the calendar"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="date",
                param_type="string",
                description="Appointment date (YYYY-MM-DD)",
                required=True,
            ),
            ToolParameter(
                name="time",
                param_type="string",
                description="Appointment time (HH:MM)",
                required=True,
            ),
            ToolParameter(
                name="duration_minutes",
                param_type="int",
                description="Duration in minutes (default: 60)",
                required=False,
                default=60,
            ),
            ToolParameter(
                name="caller_name",
                param_type="string",
                description="Caller's name",
                required=True,
            ),
            ToolParameter(
                name="caller_phone",
                param_type="string",
                description="Caller's phone number",
                required=False,
            ),
            ToolParameter(
                name="service_type",
                param_type="string",
                description="Type of service requested",
                required=False,
            ),
            ToolParameter(
                name="notes",
                param_type="string",
                description="Additional notes",
                required=False,
            ),
            ToolParameter(
                name="context_id",
                param_type="string",
                description="Business context ID",
                required=False,
            ),
        ]

    def _get_calendar_tool(self):
        """Get the calendar tool for auth."""
        if self._calendar_tool is None:
            from ..calendar import calendar_tool
            self._calendar_tool = calendar_tool
        return self._calendar_tool

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Book an appointment."""
        date_str = params.get("date")
        time_str = params.get("time")
        duration = params.get("duration_minutes", 60)
        caller_name = params.get("caller_name", "Customer")
        caller_phone = params.get("caller_phone", "")
        service_type = params.get("service_type", "Appointment")
        notes = params.get("notes", "")

        # Validate required fields
        if not date_str or not time_str:
            return ToolResult(
                success=False,
                error="MISSING_PARAMS",
                message="Date and time are required to book an appointment.",
            )

        try:
            # Parse datetime
            datetime_str = f"{date_str} {time_str}"
            start_dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
            start_dt = start_dt.astimezone()
            end_dt = start_dt + timedelta(minutes=duration)

            # Build event summary
            summary = f"{service_type} - {caller_name}"

            # Build description
            desc_parts = [f"Booked via phone by Atlas"]
            if caller_phone:
                desc_parts.append(f"Phone: {caller_phone}")
            if notes:
                desc_parts.append(f"Notes: {notes}")
            description = "\n".join(desc_parts)

            # Get calendar tool for auth token
            calendar = self._get_calendar_tool()
            auth_header = await calendar._get_auth_header()

            # Create event via Google Calendar API
            client = await self._ensure_client()

            event_body = {
                "summary": summary,
                "description": description,
                "start": {
                    "dateTime": start_dt.isoformat(),
                    "timeZone": "America/Chicago",
                },
                "end": {
                    "dateTime": end_dt.isoformat(),
                    "timeZone": "America/Chicago",
                },
            }

            url = f"{CALENDAR_API_BASE}/calendars/primary/events"
            response = await client.post(
                url,
                headers=auth_header,
                json=event_body,
            )

            if response.status_code in (200, 201):
                event_data = response.json()
                event_id = event_data.get("id", "")

                # Format confirmation
                formatted_date = start_dt.strftime("%A, %B %d")
                formatted_time = start_dt.strftime("%-I:%M %p")

                return ToolResult(
                    success=True,
                    data={
                        "event_id": event_id,
                        "date": date_str,
                        "time": time_str,
                        "duration_minutes": duration,
                        "caller_name": caller_name,
                    },
                    message=(
                        f"Appointment booked for {caller_name} on "
                        f"{formatted_date} at {formatted_time}."
                    ),
                )
            else:
                logger.error("Calendar API error: %s", response.text)
                return ToolResult(
                    success=False,
                    error="API_ERROR",
                    message="Could not create appointment. Please try again.",
                )

        except ValueError as e:
            return ToolResult(
                success=False,
                error="INVALID_DATE",
                message=f"Invalid date or time format: {e}",
            )
        except Exception as e:
            logger.exception("Error booking appointment")
            return ToolResult(
                success=False,
                error="EXECUTION_ERROR",
                message=str(e),
            )


# Module instance
book_appointment_tool = BookAppointmentTool()
