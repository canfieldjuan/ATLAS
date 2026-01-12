"""
Device discovery module for Atlas Brain.

Provides network scanning and automatic device detection using
multiple protocols (SSDP, mDNS, etc.).
"""

from .service import (
    DiscoveryService,
    get_discovery_service,
    init_discovery,
    shutdown_discovery,
    run_discovery_scan,
)

__all__ = [
    "DiscoveryService",
    "get_discovery_service",
    "init_discovery",
    "shutdown_discovery",
    "run_discovery_scan",
]
