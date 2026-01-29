"""
External Communications Module for Atlas Brain.

This module re-exports from atlas_comms service for backward compatibility.
Phone processing (STT/LLM/TTS integration) remains local in atlas_brain.

Architecture:
- Core comms functionality provided by atlas_comms service
- Phone call processing stays in atlas_brain (requires local AI services)
- Provider-agnostic abstraction layer
- Context-based routing (business vs personal)
"""

# Re-export from atlas_comms for backward compatibility
from atlas_comms import (
    CommsConfig,
    BusinessContext,
    comms_settings,
    TelephonyProvider,
    CallState,
    CallDirection,
    Call,
    SMSMessage,
    SMSDirection,
    ContextRouter,
    get_context_router,
    CommsService,
    get_comms_service,
    init_comms_service,
    shutdown_comms_service,
)
from atlas_comms.core import (
    BusinessHours,
    SchedulingConfig,
    DEFAULT_PERSONAL_CONTEXT,
    EFFINGHAM_MAIDS_CONTEXT,
)
from atlas_comms.services import (
    CalendarService,
    EmailService,
    SMSService,
    Appointment,
    TimeSlot,
    EmailMessage,
    StubCalendarService,
    StubEmailService,
    StubSMSService,
    SchedulingService,
    scheduling_service,
)
from atlas_comms.providers import get_provider, list_providers

# Local imports for phone processing (requires atlas_brain AI services)
from .real_services import (
    ResendEmailService,
    SignalWireSMSService,
    GoogleCalendarService,
    get_email_service,
    get_sms_service,
    get_calendar_service,
)

__all__ = [
    # Config (from atlas_comms)
    "CommsConfig",
    "BusinessContext",
    "BusinessHours",
    "SchedulingConfig",
    "comms_settings",
    "DEFAULT_PERSONAL_CONTEXT",
    "EFFINGHAM_MAIDS_CONTEXT",
    # Protocols (from atlas_comms)
    "TelephonyProvider",
    "CallState",
    "CallDirection",
    "Call",
    "SMSMessage",
    "SMSDirection",
    # Services (from atlas_comms)
    "CalendarService",
    "EmailService",
    "SMSService",
    "EmailMessage",
    "StubCalendarService",
    "StubEmailService",
    "StubSMSService",
    # Scheduling (from atlas_comms)
    "SchedulingService",
    "scheduling_service",
    "TimeSlot",
    "Appointment",
    # Providers (from atlas_comms)
    "get_provider",
    "list_providers",
    # Context (from atlas_comms)
    "ContextRouter",
    "get_context_router",
    # Service (from atlas_comms)
    "CommsService",
    "get_comms_service",
    "init_comms_service",
    "shutdown_comms_service",
    # Real service implementations (local - uses atlas_brain config)
    "ResendEmailService",
    "SignalWireSMSService",
    "GoogleCalendarService",
    "get_email_service",
    "get_sms_service",
    "get_calendar_service",
]
