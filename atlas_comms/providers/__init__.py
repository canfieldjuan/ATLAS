"""
Telephony provider implementations.

Each provider implements the TelephonyProvider protocol for a specific
service (Twilio, SignalWire, etc.).
"""

from typing import Optional

from ..core.protocols import TelephonyProvider
from ..core.config import comms_settings

# Provider registry
_providers: dict[str, type[TelephonyProvider]] = {}


def register_provider(name: str):
    """Decorator to register a provider implementation."""
    def decorator(cls: type[TelephonyProvider]):
        _providers[name] = cls
        return cls
    return decorator


def get_provider(name: Optional[str] = None) -> TelephonyProvider:
    """
    Get a provider instance by name.

    Args:
        name: Provider name. If None, uses the configured default.

    Returns:
        Initialized provider instance.

    Raises:
        ValueError: If provider not found.
    """
    if name is None:
        name = comms_settings.provider

    if name not in _providers:
        available = ", ".join(_providers.keys()) or "none"
        raise ValueError(
            f"Unknown provider: {name}. Available: {available}"
        )

    provider_cls = _providers[name]
    return provider_cls()


def list_providers() -> list[str]:
    """List available provider names."""
    return list(_providers.keys())


# Import providers to trigger registration
# These are imported at the bottom to avoid circular imports
try:
    from . import twilio
except ImportError:
    pass  # Twilio not installed

try:
    from . import signalwire
except ImportError:
    pass  # SignalWire not installed
