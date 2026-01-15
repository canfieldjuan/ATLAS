"""
Network scanners for device discovery.

Each scanner implements a specific discovery protocol:
- SSDP: Simple Service Discovery Protocol (UPnP devices)
- mDNS: Multicast DNS / Bonjour (Atlas nodes, Apple, Google devices)
"""

from .base import BaseScanner, ScanResult
from .ssdp import SSDPScanner
from .mdns import MDNSScanner

__all__ = [
    "BaseScanner",
    "ScanResult",
    "SSDPScanner",
    "MDNSScanner",
]
