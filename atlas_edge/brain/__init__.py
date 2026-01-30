"""
Brain connectivity for edge devices.

Handles WebSocket connection to brain server for escalation.
"""

from .connection import BrainConnectionManager
from .escalation import BrainEscalation, EscalationResult

__all__ = [
    "BrainConnectionManager",
    "BrainEscalation",
    "EscalationResult",
]
