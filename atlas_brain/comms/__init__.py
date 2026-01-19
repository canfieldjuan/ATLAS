"""
External Communications Module for Atlas.

Handles phone calls and SMS messaging through programmable telephony providers.
Supports multiple business contexts with independent configurations.

Architecture:
- Provider-agnostic abstraction layer
- Context-based routing (business vs personal)
- Integration with Atlas voice pipeline (STT/LLM/TTS)
- Appointment scheduling via calendar integration
"""

from .config import CommsConfig, BusinessContext, comms_settings
from .protocols import (
    TelephonyProvider,
    CallState,
    CallDirection,
    Call,
    SMSMessage,
    SMSDirection,
)
from .services import (
    CalendarService,
    EmailService,
    SMSService,
    Appointment,
    TimeSlot,
    EmailMessage,
    StubCalendarService,
    StubEmailService,
    StubSMSService,
)
from .real_services import (
    ResendEmailService,
    SignalWireSMSService,
    GoogleCalendarService,
    get_email_service,
    get_sms_service,
    get_calendar_service,
)
from .context import ContextRouter, get_context_router
from .scheduling import SchedulingService, scheduling_service
from .service import (
    CommsService,
    get_comms_service,
    init_comms_service,
    shutdown_comms_service,
)

__all__ = [
    # Config
    "CommsConfig",
    "BusinessContext",
    "comms_settings",
    # Protocols
    "TelephonyProvider",
    "CallState",
    "CallDirection",
    "Call",
    "SMSMessage",
    "SMSDirection",
    # Services (stubs and interfaces)
    "CalendarService",
    "EmailService",
    "SMSService",
    "EmailMessage",
    "StubCalendarService",
    "StubEmailService",
    "StubSMSService",
    # Real service implementations
    "ResendEmailService",
    "SignalWireSMSService",
    "GoogleCalendarService",
    "get_email_service",
    "get_sms_service",
    "get_calendar_service",
    # Context
    "ContextRouter",
    "get_context_router",
    # Scheduling
    "SchedulingService",
    "scheduling_service",
    "TimeSlot",
    "Appointment",
    # Service
    "CommsService",
    "get_comms_service",
    "init_comms_service",
    "shutdown_comms_service",
]
