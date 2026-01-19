"""
Real service implementations for appointment system.

Connects to:
- Resend API for email
- SignalWire for SMS
- Google Calendar for scheduling
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from .services import (
    CalendarService,
    EmailService,
    SMSService,
    TimeSlot,
    Appointment,
    EmailMessage,
)
from ..config import settings

logger = logging.getLogger("atlas.comms.real_services")


class ResendEmailService(EmailService):
    """Email service using Resend API."""

    def __init__(self):
        self._config = settings.email
        self._client = None

    async def _ensure_client(self):
        """Get or create Resend client."""
        if self._client is None:
            import resend
            resend.api_key = self._config.api_key
            self._client = resend
        return self._client

    async def send_email(self, message: EmailMessage) -> bool:
        """Send an email via Resend."""
        if not self._config.enabled:
            logger.warning("Email service disabled")
            return False

        try:
            client = await self._ensure_client()

            params = {
                "from": message.from_address or self._config.default_from,
                "to": [message.to],
                "subject": message.subject,
                "text": message.body_text,
            }

            if message.body_html:
                params["html"] = message.body_html

            if message.reply_to:
                params["reply_to"] = message.reply_to

            response = client.Emails.send(params)
            logger.info("Email sent: %s to %s", response.get("id"), message.to)
            return True

        except Exception as e:
            logger.error("Failed to send email: %s", e)
            return False

    async def send_appointment_confirmation(
        self,
        appointment: Appointment,
        business_name: str,
        business_phone: Optional[str] = None,
    ) -> bool:
        """Send appointment confirmation email."""
        if not appointment.customer_email:
            logger.warning("No customer email for appointment confirmation")
            return False

        date_str = appointment.start.strftime("%A, %B %d, %Y")
        time_str = appointment.start.strftime("%I:%M %p")

        body_text = f"""Dear {appointment.customer_name},

Your appointment has been confirmed!

Service: {appointment.service_type}
Date: {date_str}
Time: {time_str}
Address: {appointment.customer_address}

If you need to reschedule or cancel, please contact us:
"""

        if business_phone:
            body_text += f"Phone: {business_phone}\n"

        body_text += f"""
Thank you for choosing {business_name}!

Best regards,
{business_name}
"""

        message = EmailMessage(
            to=appointment.customer_email,
            subject=f"Appointment Confirmed - {business_name}",
            body_text=body_text,
        )

        return await self.send_email(message)

    async def send_appointment_reminder(
        self,
        appointment: Appointment,
        business_name: str,
        hours_before: int = 24,
    ) -> bool:
        """Send appointment reminder email."""
        if not appointment.customer_email:
            return False

        date_str = appointment.start.strftime("%A, %B %d, %Y")
        time_str = appointment.start.strftime("%I:%M %p")

        body_text = f"""Dear {appointment.customer_name},

This is a reminder about your upcoming appointment:

Service: {appointment.service_type}
Date: {date_str}
Time: {time_str}
Address: {appointment.customer_address}

We look forward to seeing you!

Best regards,
{business_name}
"""

        message = EmailMessage(
            to=appointment.customer_email,
            subject=f"Reminder: Your Appointment Tomorrow - {business_name}",
            body_text=body_text,
        )

        return await self.send_email(message)

    async def send_cancellation_notice(
        self,
        appointment: Appointment,
        business_name: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Send appointment cancellation notice."""
        if not appointment.customer_email:
            return False

        date_str = appointment.start.strftime("%A, %B %d, %Y")
        time_str = appointment.start.strftime("%I:%M %p")

        body_text = f"""Dear {appointment.customer_name},

Your appointment has been cancelled.

Service: {appointment.service_type}
Date: {date_str}
Time: {time_str}
"""

        if reason:
            body_text += f"\nReason: {reason}\n"

        body_text += f"""
If you would like to reschedule, please contact us.

Best regards,
{business_name}
"""

        message = EmailMessage(
            to=appointment.customer_email,
            subject=f"Appointment Cancelled - {business_name}",
            body_text=body_text,
        )

        return await self.send_email(message)


class SignalWireSMSService(SMSService):
    """SMS service using SignalWire via CommsService."""

    def __init__(self, context_id: str = "effingham_maids"):
        self._context_id = context_id
        self._comms_service = None

    def _get_comms_service(self):
        """Get the global comms service."""
        if self._comms_service is None:
            from .service import get_comms_service
            self._comms_service = get_comms_service()
        return self._comms_service

    async def send_sms(
        self,
        to_number: str,
        message: str,
        from_number: Optional[str] = None,
    ) -> bool:
        """Send an SMS via SignalWire."""
        service = self._get_comms_service()

        if not service.is_connected:
            logger.warning("CommsService not connected, cannot send SMS")
            return False

        try:
            result = await service.send_sms(
                to_number=to_number,
                body=message,
                context_id=self._context_id,
            )
            return result is not None

        except Exception as e:
            logger.error("Failed to send SMS: %s", e)
            return False

    async def send_appointment_confirmation_sms(
        self,
        appointment: Appointment,
        business_name: str,
        from_number: Optional[str] = None,
    ) -> bool:
        """Send appointment confirmation via SMS."""
        if not appointment.customer_phone:
            logger.warning("No customer phone for SMS confirmation")
            return False

        date_str = appointment.start.strftime("%m/%d")
        time_str = appointment.start.strftime("%I:%M %p")

        message = (
            f"{business_name}: Your {appointment.service_type} appointment "
            f"is confirmed for {date_str} at {time_str}. "
            f"Reply STOP to cancel."
        )

        return await self.send_sms(
            to_number=appointment.customer_phone,
            message=message,
            from_number=from_number,
        )


class GoogleCalendarService(CalendarService):
    """Calendar service using Google Calendar API."""

    def __init__(self):
        from ..tools.calendar import calendar_tool
        self._calendar_tool = calendar_tool
        self._config = settings.tools

    async def get_available_slots(
        self,
        date_start: datetime,
        date_end: datetime,
        duration_minutes: int = 60,
        buffer_minutes: int = 15,
        calendar_id: Optional[str] = None,
    ) -> list[TimeSlot]:
        """Find available time slots by checking for gaps in the calendar."""
        if not self._config.calendar_enabled:
            logger.warning("Calendar not enabled")
            return []

        try:
            # Get existing events
            events = await self._calendar_tool._fetch_events(
                hours_ahead=int((date_end - datetime.now().astimezone()).total_seconds() / 3600),
                max_results=50,
            )

            # Generate potential slots (9 AM - 5 PM weekdays)
            available = []
            current = date_start.replace(hour=9, minute=0, second=0, microsecond=0)

            while current < date_end:
                # Skip weekends
                if current.weekday() < 5:
                    for hour in range(9, 17):
                        slot_start = current.replace(hour=hour, minute=0)
                        slot_end = slot_start + timedelta(minutes=duration_minutes)

                        if slot_start < date_start or slot_end > date_end:
                            continue

                        # Check for conflicts
                        has_conflict = False
                        for event in events:
                            event_start = event.start
                            event_end = event.end

                            # Check if slot overlaps with event (including buffer)
                            buffered_start = slot_start - timedelta(minutes=buffer_minutes)
                            buffered_end = slot_end + timedelta(minutes=buffer_minutes)

                            if buffered_start < event_end and buffered_end > event_start:
                                has_conflict = True
                                break

                        if not has_conflict:
                            available.append(TimeSlot(
                                start=slot_start,
                                end=slot_end,
                                calendar_id=calendar_id,
                            ))

                current += timedelta(days=1)

            return available[:10]

        except Exception as e:
            logger.error("Failed to get available slots: %s", e)
            return []

    async def create_event(
        self,
        appointment: Appointment,
        calendar_id: Optional[str] = None,
    ) -> str:
        """Create a calendar event for an appointment."""
        if not self._config.calendar_enabled:
            logger.warning("Calendar not enabled")
            return ""

        try:
            import httpx

            client = await self._calendar_tool._ensure_client()
            headers = await self._calendar_tool._get_auth_header()

            cal_id = calendar_id or "primary"
            url = f"https://www.googleapis.com/calendar/v3/calendars/{cal_id}/events"

            event_data = {
                "summary": f"{appointment.service_type} - {appointment.customer_name}",
                "description": (
                    f"Customer: {appointment.customer_name}\n"
                    f"Phone: {appointment.customer_phone}\n"
                    f"Email: {appointment.customer_email}\n"
                    f"Address: {appointment.customer_address}\n"
                    f"Notes: {appointment.notes}"
                ),
                "location": appointment.customer_address,
                "start": {
                    "dateTime": appointment.start.isoformat(),
                    "timeZone": "America/Chicago",
                },
                "end": {
                    "dateTime": appointment.end.isoformat(),
                    "timeZone": "America/Chicago",
                },
            }

            response = await client.post(url, headers=headers, json=event_data)
            response.raise_for_status()

            event_id = response.json().get("id", "")
            logger.info("Created calendar event: %s", event_id)
            return event_id

        except Exception as e:
            logger.error("Failed to create calendar event: %s", e)
            return ""

    async def update_event(
        self,
        event_id: str,
        appointment: Appointment,
        calendar_id: Optional[str] = None,
    ) -> bool:
        """Update an existing calendar event."""
        if not self._config.calendar_enabled or not event_id:
            return False

        try:
            client = await self._calendar_tool._ensure_client()
            headers = await self._calendar_tool._get_auth_header()

            cal_id = calendar_id or "primary"
            url = f"https://www.googleapis.com/calendar/v3/calendars/{cal_id}/events/{event_id}"

            event_data = {
                "summary": f"{appointment.service_type} - {appointment.customer_name}",
                "description": (
                    f"Customer: {appointment.customer_name}\n"
                    f"Phone: {appointment.customer_phone}\n"
                    f"Email: {appointment.customer_email}\n"
                    f"Address: {appointment.customer_address}\n"
                    f"Notes: {appointment.notes}"
                ),
                "location": appointment.customer_address,
                "start": {
                    "dateTime": appointment.start.isoformat(),
                    "timeZone": "America/Chicago",
                },
                "end": {
                    "dateTime": appointment.end.isoformat(),
                    "timeZone": "America/Chicago",
                },
            }

            response = await client.put(url, headers=headers, json=event_data)
            response.raise_for_status()

            logger.info("Updated calendar event: %s", event_id)
            return True

        except Exception as e:
            logger.error("Failed to update calendar event: %s", e)
            return False

    async def delete_event(
        self,
        event_id: str,
        calendar_id: Optional[str] = None,
    ) -> bool:
        """Delete a calendar event."""
        if not self._config.calendar_enabled or not event_id:
            return False

        try:
            client = await self._calendar_tool._ensure_client()
            headers = await self._calendar_tool._get_auth_header()

            cal_id = calendar_id or "primary"
            url = f"https://www.googleapis.com/calendar/v3/calendars/{cal_id}/events/{event_id}"

            response = await client.delete(url, headers=headers)
            response.raise_for_status()

            logger.info("Deleted calendar event: %s", event_id)
            return True

        except Exception as e:
            logger.error("Failed to delete calendar event: %s", e)
            return False

    async def check_conflicts(
        self,
        start: datetime,
        end: datetime,
        calendar_id: Optional[str] = None,
    ) -> bool:
        """Check if a time slot has conflicts."""
        if not self._config.calendar_enabled:
            return False

        try:
            events = await self._calendar_tool._fetch_events(
                hours_ahead=int((end - datetime.now().astimezone()).total_seconds() / 3600),
                max_results=50,
            )

            for event in events:
                if start < event.end and end > event.start:
                    return True

            return False

        except Exception as e:
            logger.error("Failed to check conflicts: %s", e)
            return False


# Factory functions
def get_email_service() -> EmailService:
    """Get the real email service."""
    if settings.email.enabled:
        return ResendEmailService()
    from .services import StubEmailService
    return StubEmailService()


def get_sms_service(context_id: str = "effingham_maids") -> SMSService:
    """Get the real SMS service."""
    from .config import comms_settings
    if comms_settings.enabled:
        return SignalWireSMSService(context_id)
    from .services import StubSMSService
    return StubSMSService()


def get_calendar_service() -> CalendarService:
    """Get the real calendar service."""
    if settings.tools.calendar_enabled:
        return GoogleCalendarService()
    from .services import StubCalendarService
    return StubCalendarService()
