"""
Context routing for incoming calls and messages.

DEPRECATED: This module re-exports from atlas_comms for backward compatibility.
Import from atlas_comms directly for new code:
    from atlas_comms.context import ContextRouter, get_context_router
"""

# Re-export everything from atlas_comms for backward compatibility
from atlas_comms.context import (
    ContextRouter,
    get_context_router,
)

__all__ = [
    "ContextRouter",
    "get_context_router",
]
