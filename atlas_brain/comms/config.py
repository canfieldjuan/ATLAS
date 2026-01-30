"""
Configuration for the external communications system.

DEPRECATED: This module re-exports from atlas_comms for backward compatibility.
Import from atlas_comms directly for new code:
    from atlas_comms.core import CommsConfig, BusinessContext, comms_settings
"""

# Re-export everything from atlas_comms for backward compatibility
from atlas_comms.core.config import (
    BusinessHours,
    SchedulingConfig,
    BusinessContext,
    ServerConfig,
    TwilioProviderConfig,
    SignalWireProviderConfig,
    CalendarConfig,
    AtlasBrainConfig,
    CommsConfig,
    comms_settings,
    DEFAULT_PERSONAL_CONTEXT,
    EFFINGHAM_MAIDS_CONTEXT,
)

__all__ = [
    "BusinessHours",
    "SchedulingConfig",
    "BusinessContext",
    "ServerConfig",
    "TwilioProviderConfig",
    "SignalWireProviderConfig",
    "CalendarConfig",
    "AtlasBrainConfig",
    "CommsConfig",
    "comms_settings",
    "DEFAULT_PERSONAL_CONTEXT",
    "EFFINGHAM_MAIDS_CONTEXT",
]
