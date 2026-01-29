"""
Service interfaces for external integrations.

DEPRECATED: This module re-exports from atlas_comms for backward compatibility.
Import from atlas_comms directly for new code:
    from atlas_comms.services import CalendarService, EmailService, SMSService
"""

# Re-export everything from atlas_comms for backward compatibility
from atlas_comms.services import (
    CalendarService,
    EmailService,
    SMSService,
    TimeSlot,
    Appointment,
    EmailMessage,
    StubCalendarService,
    StubEmailService,
    StubSMSService,
)

__all__ = [
    "CalendarService",
    "EmailService",
    "SMSService",
    "TimeSlot",
    "Appointment",
    "EmailMessage",
    "StubCalendarService",
    "StubEmailService",
    "StubSMSService",
]
