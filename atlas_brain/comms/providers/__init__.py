"""
Telephony provider implementations.

DEPRECATED: This module re-exports from atlas_comms for backward compatibility.
Import from atlas_comms directly for new code:
    from atlas_comms.providers import get_provider, list_providers
"""

# Re-export everything from atlas_comms for backward compatibility
from atlas_comms.providers import (
    get_provider,
    list_providers,
    register_provider,
)

__all__ = [
    "get_provider",
    "list_providers",
    "register_provider",
]
