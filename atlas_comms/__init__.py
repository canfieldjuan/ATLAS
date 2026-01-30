"""
Atlas Communications Service.

Handles phone calls and SMS messaging through programmable telephony providers.
Supports multiple business contexts with independent configurations.

Architecture:
- Provider-agnostic abstraction layer (Twilio, SignalWire)
- Context-based routing (business vs personal)
- Integration with Atlas Brain for AI (STT/LLM/TTS via HTTP)
- Appointment scheduling via Google Calendar
"""

from .core import (
    CommsConfig,
    BusinessContext,
    BusinessHours,
    SchedulingConfig,
    comms_settings,
    Call,
    CallState,
    CallDirection,
    SMSMessage,
    SMSDirection,
    TelephonyProvider,
)
from .context import ContextRouter, get_context_router
from .service import (
    CommsService,
    get_comms_service,
    init_comms_service,
    shutdown_comms_service,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "CommsConfig",
    "BusinessContext",
    "BusinessHours",
    "SchedulingConfig",
    "comms_settings",
    # Protocols
    "Call",
    "CallState",
    "CallDirection",
    "SMSMessage",
    "SMSDirection",
    "TelephonyProvider",
    # Context
    "ContextRouter",
    "get_context_router",
    # Service
    "CommsService",
    "get_comms_service",
    "init_comms_service",
    "shutdown_comms_service",
]
