"""
External Communications Module for Atlas.

Handles phone calls and SMS messaging through programmable telephony providers.
Supports multiple business contexts with independent configurations.

Architecture:
- Provider-agnostic abstraction layer
- Context-based routing (business vs personal)
- Integration with Atlas voice pipeline (STT/LLM/TTS)
- Appointment scheduling via calendar integration

Usage:
    from atlas_brain.comms import (
        AppointmentStateMachine,
        create_appointment_machine,
        BusinessContext,
    )

    # Create appointment flow
    machine = create_appointment_machine(business_context)
    response = await machine.start(caller_phone="+1234567890")
    response = await machine.process_input("I'd like to schedule a cleaning")
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
from .appointment import (
    AppointmentStateMachine,
    AppointmentState,
    AppointmentEvent,
    AppointmentContext,
    create_appointment_machine,
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
    # Services
    "CalendarService",
    "EmailService",
    "SMSService",
    "Appointment",
    "TimeSlot",
    "EmailMessage",
    "StubCalendarService",
    "StubEmailService",
    "StubSMSService",
    # Appointment State Machine
    "AppointmentStateMachine",
    "AppointmentState",
    "AppointmentEvent",
    "AppointmentContext",
    "create_appointment_machine",
]
