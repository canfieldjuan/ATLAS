"""
Appointment scheduling service for external communications.

DEPRECATED: This module re-exports from atlas_comms for backward compatibility.
Import from atlas_comms directly for new code:
    from atlas_comms.services import SchedulingService, scheduling_service
"""

# Re-export everything from atlas_comms for backward compatibility
from atlas_comms.services import (
    SchedulingService,
    scheduling_service,
    TimeSlot,
    Appointment,
)

__all__ = [
    "SchedulingService",
    "scheduling_service",
    "TimeSlot",
    "Appointment",
]
