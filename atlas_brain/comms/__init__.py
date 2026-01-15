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
]
