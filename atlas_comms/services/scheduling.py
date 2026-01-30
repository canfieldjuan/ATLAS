"""
Appointment scheduling service for external communications.

Integrates with Google Calendar to:
- Check availability for appointments
- Book new appointments
- Cancel or reschedule appointments
- Find available time slots
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, time as dt_time
from typing import Any, Optional
from zoneinfo import ZoneInfo

import httpx

from ..core.config import comms_settings, BusinessContext
from .base import TimeSlot, Appointment

logger = logging.getLogger("atlas.comms.scheduling")

# Google Calendar API endpoints
CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"
TOKEN_URL = "https://oauth2.googleapis.com/token"


class SchedulingService:
    """
    Appointment scheduling service with Google Calendar integration.

    Provides availability checking and booking for business contexts.
    """

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._access_token: Optional[str] = None
        self._token_expires: float = 0.0
        self._refresh_lock = asyncio.Lock()

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=15.0,
                limits=httpx.Limits(max_keepalive_connections=5),
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _refresh_token(self) -> str:
        """Refresh OAuth2 access token."""
        async with self._refresh_lock:
            if self._access_token and time.time() < self._token_expires - 60:
                return self._access_token

            calendar_config = comms_settings.calendar
            if not calendar_config.enabled:
                raise ValueError("Calendar integration not enabled")
            if not calendar_config.client_id or not calendar_config.client_secret:
                raise ValueError("Calendar OAuth credentials not configured")
            if not calendar_config.refresh_token:
                raise ValueError("Calendar refresh token not configured")

            client = await self._ensure_client()
            data = {
                "client_id": calendar_config.client_id,
                "client_secret": calendar_config.client_secret,
                "refresh_token": calendar_config.refresh_token,
                "grant_type": "refresh_token",
            }

            response = await client.post(TOKEN_URL, data=data)
            response.raise_for_status()
            token_data = response.json()

            self._access_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
            self._token_expires = time.time() + expires_in

            return self._access_token

    async def _get_auth_header(self) -> dict:
        """Get authorization header."""
        if not self._access_token or time.time() >= self._token_expires - 60:
            await self._refresh_token()
        return {"Authorization": "Bearer " + str(self._access_token)}

    async def get_available_slots(
        self,
        context: BusinessContext,
        date: Optional[datetime] = None,
        duration_minutes: Optional[int] = None,
        days_ahead: int = 7,
    ) -> list:
        """
        Get available appointment slots for a business context.

        Args:
            context: Business context with scheduling config
            date: Specific date to check (None = check multiple days)
            duration_minutes: Appointment duration (defaults to context config)
            days_ahead: How many days ahead to search

        Returns:
            List of available TimeSlot objects
        """
        if not context.scheduling.enabled:
            return []

        if not context.scheduling.calendar_id:
            logger.warning("No calendar_id configured for context %s", context.id)
            return []

        tz = ZoneInfo(context.hours.timezone)
        now = datetime.now(tz)
        duration = duration_minutes or context.scheduling.default_duration_minutes
        buffer = context.scheduling.buffer_minutes

        # Determine date range to check
        if date:
            start_date = date.date()
            end_date = start_date
        else:
            start_date = now.date()
            max_days = min(days_ahead, context.scheduling.max_advance_days)
            end_date = start_date + timedelta(days=max_days)

        # Respect min_notice_hours
        min_start = now + timedelta(hours=context.scheduling.min_notice_hours)

        # Get existing events from calendar
        busy_times = await self._get_busy_times(
            context.scheduling.calendar_id,
            datetime.combine(start_date, dt_time.min).replace(tzinfo=tz),
            datetime.combine(end_date + timedelta(days=1), dt_time.min).replace(tzinfo=tz),
        )

        # Generate available slots
        slots = []
        current_date = start_date

        while current_date <= end_date:
            day_slots = self._get_day_slots(
                context=context,
                date=current_date,
                duration=duration,
                buffer=buffer,
                busy_times=busy_times,
                min_start=min_start,
                tz=tz,
            )
            slots.extend(day_slots)
            current_date += timedelta(days=1)

        return slots

    def _get_day_slots(
        self,
        context: BusinessContext,
        date: Any,
        duration: int,
        buffer: int,
        busy_times: list,
        min_start: datetime,
        tz: ZoneInfo,
    ) -> list:
        """Get available slots for a single day."""
        day_names = [
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday"
        ]
        day_idx = date.weekday()
        day_name = day_names[day_idx]

        # Get business hours for this day
        open_time = getattr(context.hours, day_name + "_open")
        close_time = getattr(context.hours, day_name + "_close")

        if not open_time or not close_time:
            return []

        # Parse business hours
        open_hour, open_min = map(int, open_time.split(":"))
        close_hour, close_min = map(int, close_time.split(":"))

        day_start = datetime(
            date.year, date.month, date.day,
            open_hour, open_min, tzinfo=tz
        )
        day_end = datetime(
            date.year, date.month, date.day,
            close_hour, close_min, tzinfo=tz
        )

        # Generate slot times (30-minute increments)
        slots = []
        slot_start = day_start
        slot_increment = timedelta(minutes=30)

        while slot_start + timedelta(minutes=duration) <= day_end:
            slot_end = slot_start + timedelta(minutes=duration)

            if slot_start >= min_start:
                buffer_start = slot_start - timedelta(minutes=buffer)
                buffer_end = slot_end + timedelta(minutes=buffer)

                is_available = True
                for busy_start, busy_end in busy_times:
                    if buffer_start < busy_end and buffer_end > busy_start:
                        is_available = False
                        break

                if is_available:
                    slots.append(TimeSlot(start=slot_start, end=slot_end))

            slot_start += slot_increment

        return slots

    async def _get_busy_times(
        self,
        calendar_id: str,
        time_min: datetime,
        time_max: datetime,
    ) -> list:
        """Get busy times from Google Calendar."""
        client = await self._ensure_client()
        headers = await self._get_auth_header()

        url = CALENDAR_API_BASE + "/freeBusy"
        body = {
            "timeMin": time_min.isoformat(),
            "timeMax": time_max.isoformat(),
            "items": [{"id": calendar_id}],
        }

        try:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()

            data = response.json()
            calendar_data = data.get("calendars", {}).get(calendar_id, {})
            busy_list = calendar_data.get("busy", [])

            busy_times = []
            for busy in busy_list:
                start = datetime.fromisoformat(busy["start"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(busy["end"].replace("Z", "+00:00"))
                busy_times.append((start, end))

            return busy_times

        except Exception as e:
            logger.error("Failed to get busy times: %s", e)
            return []

    async def book_appointment(
        self,
        context: BusinessContext,
        slot: TimeSlot,
        customer_name: str,
        customer_phone: str,
        customer_email: Optional[str] = None,
        service_type: Optional[str] = None,
        location: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Optional[Appointment]:
        """
        Book an appointment.

        Args:
            context: Business context
            slot: TimeSlot to book
            customer_name: Customer's name
            customer_phone: Customer's phone number
            customer_email: Customer's email (optional)
            service_type: Type of service (optional)
            location: Location for the appointment
            notes: Additional notes

        Returns:
            Appointment object if successful, None otherwise
        """
        if not context.scheduling.calendar_id:
            logger.error("No calendar_id for context %s", context.id)
            return None

        client = await self._ensure_client()
        headers = await self._get_auth_header()

        # Build event summary
        summary = context.name + " - " + customer_name
        if service_type:
            summary = service_type + " - " + customer_name

        # Build description
        description_parts = [
            "Customer: " + customer_name,
            "Phone: " + customer_phone,
        ]
        if customer_email:
            description_parts.append("Email: " + customer_email)
        if notes:
            description_parts.append("\nNotes: " + notes)
        description_parts.append("\nBooked via: Atlas AI Assistant")

        description = "\n".join(description_parts)

        # Create event
        event_body = {
            "summary": summary,
            "description": description,
            "start": {
                "dateTime": slot.start.isoformat(),
                "timeZone": context.hours.timezone,
            },
            "end": {
                "dateTime": slot.end.isoformat(),
                "timeZone": context.hours.timezone,
            },
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "popup", "minutes": 60},
                    {"method": "popup", "minutes": 15},
                ],
            },
        }

        if location:
            event_body["location"] = location

        url = CALENDAR_API_BASE + "/calendars/" + context.scheduling.calendar_id + "/events"

        try:
            response = await client.post(url, headers=headers, json=event_body)
            response.raise_for_status()

            event_data = response.json()

            appointment = Appointment(
                start=slot.start,
                end=slot.end,
                service_type=service_type or "",
                duration_minutes=int((slot.end - slot.start).total_seconds() / 60),
                customer_name=customer_name,
                customer_phone=customer_phone,
                customer_email=customer_email or "",
                customer_address=location or "",
                calendar_event_id=event_data["id"],
                notes=notes or "",
                business_context_id=context.id,
            )

            logger.info(
                "Booked appointment %s for %s at %s",
                appointment.id,
                customer_name,
                slot.start,
            )

            return appointment

        except Exception as e:
            logger.error("Failed to book appointment: %s", e)
            return None

    async def cancel_appointment(
        self,
        context: BusinessContext,
        appointment_id: str,
    ) -> bool:
        """Cancel an appointment."""
        if not context.scheduling.calendar_id:
            return False

        client = await self._ensure_client()
        headers = await self._get_auth_header()

        url = (
            CALENDAR_API_BASE + "/calendars/" +
            context.scheduling.calendar_id + "/events/" + appointment_id
        )

        try:
            response = await client.delete(url, headers=headers)
            response.raise_for_status()
            logger.info("Cancelled appointment %s", appointment_id)
            return True
        except Exception as e:
            logger.error("Failed to cancel appointment: %s", e)
            return False

    def format_slots_for_speech(
        self,
        slots: list,
        max_slots: int = 5,
    ) -> str:
        """Format available slots for text-to-speech."""
        if not slots:
            return "I don't have any available appointments in that time frame."

        if len(slots) == 1:
            return "I have one opening: " + str(slots[0])

        # Group by day for clearer presentation
        by_day: dict = {}
        for slot in slots[:max_slots]:
            day_key = slot.start.strftime("%A, %B %d")
            if day_key not in by_day:
                by_day[day_key] = []
            by_day[day_key].append(slot)

        parts = ["Here are some available times:"]
        for day, day_slots in by_day.items():
            times = [s.start.strftime("%I:%M %p").lstrip("0") for s in day_slots[:3]]
            if len(times) > 1:
                times_str = ", ".join(times[:-1]) + " or " + times[-1]
            else:
                times_str = times[0]
            parts.append(day + ": " + times_str)

        if len(slots) > max_slots:
            remaining = str(len(slots) - max_slots)
            parts.append("I have " + remaining + " more openings if those don't work.")

        return " ".join(parts)


# Module-level singleton instance
scheduling_service = SchedulingService()
