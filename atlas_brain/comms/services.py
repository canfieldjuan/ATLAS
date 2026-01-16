"""
Service interfaces for external integrations.

Defines protocols for Calendar, Email, and SMS services that the
appointment system depends on. Implementations are provided separately.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4


# === Data Models ===


@dataclass
class TimeSlot:
    """Represents an available time slot for booking."""
    start: datetime
    end: datetime
    calendar_id: Optional[str] = None

    @property
    def duration_minutes(self) -> int:
        return int((self.end - self.start).total_seconds() / 60)

    def __str__(self) -> str:
        return f"{self.start.strftime('%A, %B %d at %I:%M %p')}"


@dataclass
class Appointment:
    """Represents a booked appointment."""
    id: UUID = field(default_factory=uuid4)

    # Timing
    start: datetime = field(default_factory=datetime.now)
    end: datetime = field(default_factory=datetime.now)

    # Service details
    service_type: str = ""
    duration_minutes: int = 60

    # Customer info
    customer_name: str = ""
    customer_phone: str = ""
    customer_email: str = ""
    customer_address: str = ""

    # Booking metadata
    calendar_event_id: Optional[str] = None
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    # Context
    business_context_id: str = ""
    call_id: Optional[UUID] = None  # Link to the call that created this

    # Status
    confirmed: bool = False
    confirmation_sent: bool = False
    reminder_sent: bool = False

    def to_calendar_event(self) -> dict:
        """Convert to calendar event format."""
        return {
            "summary": f"{self.service_type} - {self.customer_name}",
            "description": (
                f"Customer: {self.customer_name}\n"
                f"Phone: {self.customer_phone}\n"
                f"Email: {self.customer_email}\n"
                f"Address: {self.customer_address}\n"
                f"Notes: {self.notes}"
            ),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "location": self.customer_address,
        }


@dataclass
class EmailMessage:
    """Email message to be sent."""
    to: str
    subject: str
    body_text: str
    body_html: Optional[str] = None
    from_address: Optional[str] = None
    reply_to: Optional[str] = None

    # Metadata
    template_id: Optional[str] = None
    template_data: dict = field(default_factory=dict)


# === Service Protocols ===


class CalendarService(ABC):
    """
    Protocol for calendar operations.

    Implementations connect to Google Calendar, Outlook, etc.
    """

    @abstractmethod
    async def get_available_slots(
        self,
        date_start: datetime,
        date_end: datetime,
        duration_minutes: int = 60,
        buffer_minutes: int = 15,
        calendar_id: Optional[str] = None,
    ) -> list[TimeSlot]:
        """
        Find available time slots within a date range.

        Args:
            date_start: Start of the search window
            date_end: End of the search window
            duration_minutes: Required appointment duration
            buffer_minutes: Buffer time between appointments
            calendar_id: Specific calendar to check (default: primary)

        Returns:
            List of available TimeSlot objects
        """
        pass

    @abstractmethod
    async def create_event(
        self,
        appointment: Appointment,
        calendar_id: Optional[str] = None,
    ) -> str:
        """
        Create a calendar event for an appointment.

        Args:
            appointment: The appointment to book
            calendar_id: Target calendar (default: primary)

        Returns:
            The created event ID
        """
        pass

    @abstractmethod
    async def update_event(
        self,
        event_id: str,
        appointment: Appointment,
        calendar_id: Optional[str] = None,
    ) -> bool:
        """
        Update an existing calendar event.

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def delete_event(
        self,
        event_id: str,
        calendar_id: Optional[str] = None,
    ) -> bool:
        """
        Delete a calendar event.

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def check_conflicts(
        self,
        start: datetime,
        end: datetime,
        calendar_id: Optional[str] = None,
    ) -> bool:
        """
        Check if a time slot has conflicts.

        Returns:
            True if there ARE conflicts (slot is NOT available)
        """
        pass


class EmailService(ABC):
    """
    Protocol for email operations.

    Implementations connect to Gmail, Resend, SendGrid, etc.
    """

    @abstractmethod
    async def send_email(self, message: EmailMessage) -> bool:
        """
        Send an email message.

        Returns:
            True if sent successfully
        """
        pass

    @abstractmethod
    async def send_appointment_confirmation(
        self,
        appointment: Appointment,
        business_name: str,
        business_phone: Optional[str] = None,
    ) -> bool:
        """
        Send appointment confirmation email.

        Uses a predefined template with appointment details.

        Returns:
            True if sent successfully
        """
        pass

    @abstractmethod
    async def send_appointment_reminder(
        self,
        appointment: Appointment,
        business_name: str,
        hours_before: int = 24,
    ) -> bool:
        """
        Send appointment reminder email.

        Returns:
            True if sent successfully
        """
        pass

    @abstractmethod
    async def send_cancellation_notice(
        self,
        appointment: Appointment,
        business_name: str,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Send appointment cancellation notice.

        Returns:
            True if sent successfully
        """
        pass


class SMSService(ABC):
    """
    Protocol for SMS operations.

    Can be implemented by TwilioProvider or standalone SMS service.
    """

    @abstractmethod
    async def send_sms(
        self,
        to_number: str,
        message: str,
        from_number: Optional[str] = None,
    ) -> bool:
        """
        Send an SMS message.

        Returns:
            True if sent successfully
        """
        pass

    @abstractmethod
    async def send_appointment_confirmation_sms(
        self,
        appointment: Appointment,
        business_name: str,
        from_number: Optional[str] = None,
    ) -> bool:
        """
        Send appointment confirmation via SMS.

        Returns:
            True if sent successfully
        """
        pass


# === Stub Implementations (for testing/development) ===


class StubCalendarService(CalendarService):
    """Stub implementation that logs actions without external calls."""

    def __init__(self):
        self._events: dict[str, Appointment] = {}
        self._event_counter = 0

    async def get_available_slots(
        self,
        date_start: datetime,
        date_end: datetime,
        duration_minutes: int = 60,
        buffer_minutes: int = 15,
        calendar_id: Optional[str] = None,
    ) -> list[TimeSlot]:
        """Return fake available slots for testing."""
        from datetime import timedelta

        slots = []
        current = date_start.replace(hour=9, minute=0, second=0, microsecond=0)

        while current < date_end:
            # Skip weekends
            if current.weekday() < 5:  # Mon-Fri
                # Morning and afternoon slots
                for hour in [9, 11, 14, 16]:
                    slot_start = current.replace(hour=hour)
                    slot_end = slot_start + timedelta(minutes=duration_minutes)

                    if slot_start >= date_start and slot_end <= date_end:
                        slots.append(TimeSlot(start=slot_start, end=slot_end))

            current += timedelta(days=1)

        return slots[:10]  # Return max 10 slots

    async def create_event(
        self,
        appointment: Appointment,
        calendar_id: Optional[str] = None,
    ) -> str:
        self._event_counter += 1
        event_id = f"stub_event_{self._event_counter}"
        self._events[event_id] = appointment
        return event_id

    async def update_event(
        self,
        event_id: str,
        appointment: Appointment,
        calendar_id: Optional[str] = None,
    ) -> bool:
        if event_id in self._events:
            self._events[event_id] = appointment
            return True
        return False

    async def delete_event(
        self,
        event_id: str,
        calendar_id: Optional[str] = None,
    ) -> bool:
        return self._events.pop(event_id, None) is not None

    async def check_conflicts(
        self,
        start: datetime,
        end: datetime,
        calendar_id: Optional[str] = None,
    ) -> bool:
        # No conflicts in stub
        return False


class StubEmailService(EmailService):
    """Stub implementation that logs emails without sending."""

    def __init__(self):
        self.sent_emails: list[EmailMessage] = []

    async def send_email(self, message: EmailMessage) -> bool:
        self.sent_emails.append(message)
        return True

    async def send_appointment_confirmation(
        self,
        appointment: Appointment,
        business_name: str,
        business_phone: Optional[str] = None,
    ) -> bool:
        message = EmailMessage(
            to=appointment.customer_email,
            subject=f"Appointment Confirmed - {business_name}",
            body_text=f"Your appointment for {appointment.service_type} is confirmed for {appointment.start}",
        )
        return await self.send_email(message)

    async def send_appointment_reminder(
        self,
        appointment: Appointment,
        business_name: str,
        hours_before: int = 24,
    ) -> bool:
        message = EmailMessage(
            to=appointment.customer_email,
            subject=f"Reminder: Appointment Tomorrow - {business_name}",
            body_text=f"Reminder: Your appointment for {appointment.service_type} is scheduled for {appointment.start}",
        )
        return await self.send_email(message)

    async def send_cancellation_notice(
        self,
        appointment: Appointment,
        business_name: str,
        reason: Optional[str] = None,
    ) -> bool:
        message = EmailMessage(
            to=appointment.customer_email,
            subject=f"Appointment Cancelled - {business_name}",
            body_text=f"Your appointment for {appointment.service_type} has been cancelled.",
        )
        return await self.send_email(message)


class StubSMSService(SMSService):
    """Stub implementation that logs SMS without sending."""

    def __init__(self):
        self.sent_messages: list[tuple[str, str]] = []

    async def send_sms(
        self,
        to_number: str,
        message: str,
        from_number: Optional[str] = None,
    ) -> bool:
        self.sent_messages.append((to_number, message))
        return True

    async def send_appointment_confirmation_sms(
        self,
        appointment: Appointment,
        business_name: str,
        from_number: Optional[str] = None,
    ) -> bool:
        message = (
            f"{business_name}: Your appointment for {appointment.service_type} "
            f"is confirmed for {appointment.start.strftime('%m/%d at %I:%M %p')}. "
            f"Reply CANCEL to cancel."
        )
        return await self.send_sms(appointment.customer_phone, message, from_number)
