"""
Protocols and data models for the communications system.

DEPRECATED: This module re-exports from atlas_comms for backward compatibility.
Import from atlas_comms directly for new code:
    from atlas_comms.core import Call, CallState, TelephonyProvider
"""

# Re-export everything from atlas_comms for backward compatibility
from atlas_comms.core.protocols import (
    Call,
    CallState,
    CallDirection,
    SMSMessage,
    SMSDirection,
    TelephonyProvider,
    AudioChunkCallback,
    CallEventCallback,
    SMSCallback,
)

__all__ = [
    "Call",
    "CallState",
    "CallDirection",
    "SMSMessage",
    "SMSDirection",
    "TelephonyProvider",
    "AudioChunkCallback",
    "CallEventCallback",
    "SMSCallback",
]
