"""
Network scanners for device discovery.

Each scanner implements a specific discovery protocol:
- SSDP: Simple Service Discovery Protocol (UPnP devices)
- mDNS: Multicast DNS / Bonjour (Apple, Google devices)
"""

from .base import BaseScanner, ScanResult
from .ssdp import SSDPScanner

__all__ = [
    "BaseScanner",
    "ScanResult",
    "SSDPScanner",
]
