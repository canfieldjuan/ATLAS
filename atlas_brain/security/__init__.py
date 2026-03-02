"""
Network security monitoring for Atlas.

Provides WiFi threat detection, network IDS, and security asset tracking.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .monitor import SecurityMonitor


def get_security_monitor():
    """Lazy import to avoid importing scapy during test collection."""
    from .monitor import get_security_monitor as _get_security_monitor

    return _get_security_monitor()


def __getattr__(name: str) -> Any:
    if name == "SecurityMonitor":
        from .monitor import SecurityMonitor as _SecurityMonitor

        return _SecurityMonitor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SecurityMonitor",
    "get_security_monitor",
]
