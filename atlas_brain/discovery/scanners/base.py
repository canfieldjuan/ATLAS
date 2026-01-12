"""
Base scanner protocol for device discovery.

All network scanners must implement this interface.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("atlas.discovery.scanners.base")


@dataclass
class ScanResult:
    """Result from a network scan."""

    # Required fields
    host: str  # IP address
    protocol: str  # Discovery protocol used (ssdp, mdns, etc.)

    # Device identification
    device_type: Optional[str] = None  # roku, chromecast, etc.
    name: Optional[str] = None  # Friendly name
    manufacturer: Optional[str] = None
    model: Optional[str] = None

    # Network info
    port: Optional[int] = None
    mac_address: Optional[str] = None

    # Protocol-specific data
    headers: dict[str, str] = field(default_factory=dict)
    services: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)

    def generate_device_id(self) -> str:
        """Generate a unique device ID from this scan result."""
        # Replace dots with underscores for valid identifier
        host_safe = self.host.replace(".", "_")

        if self.device_type:
            return f"{self.device_type}.{host_safe}"

        # Fallback to protocol-based ID
        return f"{self.protocol}.{host_safe}"


class BaseScanner(ABC):
    """
    Abstract base class for network scanners.

    Scanners detect devices on the local network using
    specific discovery protocols.
    """

    @property
    @abstractmethod
    def protocol_name(self) -> str:
        """Name of the discovery protocol (e.g., 'ssdp', 'mdns')."""
        ...

    @abstractmethod
    async def scan(self, timeout: float = 5.0) -> list[ScanResult]:
        """
        Perform a network scan and return discovered devices.

        Args:
            timeout: Maximum time to wait for responses (seconds)

        Returns:
            List of ScanResult objects for discovered devices
        """
        ...

    async def is_available(self) -> bool:
        """
        Check if this scanner can run on the current system.

        Override if the scanner has system requirements.
        """
        return True

    def identify_device_type(self, result: ScanResult) -> Optional[str]:
        """
        Attempt to identify the device type from scan data.

        Override to implement protocol-specific identification logic.
        """
        return None
