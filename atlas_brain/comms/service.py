"""
Main Communications Service.

DEPRECATED: This module re-exports from atlas_comms for backward compatibility.
Import from atlas_comms directly for new code:
    from atlas_comms import CommsService, get_comms_service
"""

# Re-export everything from atlas_comms for backward compatibility
from atlas_comms import (
    CommsService,
    get_comms_service,
    init_comms_service,
    shutdown_comms_service,
)

__all__ = [
    "CommsService",
    "get_comms_service",
    "init_comms_service",
    "shutdown_comms_service",
]
