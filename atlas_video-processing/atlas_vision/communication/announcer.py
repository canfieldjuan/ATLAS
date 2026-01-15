"""
mDNS service announcer for Atlas Vision nodes.

Registers the node as a discoverable service on the local network
using Zeroconf/mDNS (Bonjour-compatible).
"""

import asyncio
import logging
import socket
from typing import Optional

from zeroconf import IPVersion, ServiceInfo
from zeroconf.asyncio import AsyncZeroconf

from ..core.config import settings

logger = logging.getLogger("atlas.vision.communication.announcer")

# Service type for Atlas nodes
SERVICE_TYPE = "_atlas-node._tcp.local."


def _get_local_ip() -> str:
    """Get the local IP address."""
    try:
        # Create a socket to determine the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _get_hostname() -> str:
    """Get the hostname for service naming."""
    try:
        return socket.gethostname().split(".")[0].lower()
    except Exception:
        return "unknown"


class NodeAnnouncer:
    """
    Announces this Atlas Vision node via mDNS.

    Other Atlas components (especially Atlas Brain) can discover
    this node by browsing for _atlas-node._tcp.local. services.
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        port: Optional[int] = None,
        node_type: str = "gateway",
    ):
        """
        Initialize the announcer.

        Args:
            node_id: Unique node identifier (auto-generated if None)
            port: API port (from settings if None)
            node_type: Node type (gateway, edge, mobile)
        """
        hostname = _get_hostname()
        self.node_id = node_id or f"atlas-vision-{hostname}"
        self.port = port or settings.server.port
        self.node_type = node_type

        self._zeroconf: Optional[AsyncZeroconf] = None
        self._service_info: Optional[ServiceInfo] = None
        self._running = False

    def _build_properties(self) -> dict[str, str]:
        """Build TXT record properties."""
        from ..devices.registry import device_registry
        from ..core.constants import DeviceType

        # Count cameras
        cameras = device_registry.list_by_type(DeviceType.CAMERA)
        camera_count = len(cameras)

        return {
            "node_id": self.node_id,
            "type": self.node_type,
            "version": "0.1.0",
            "cameras": str(camera_count),
            "capabilities": "camera,detection,security",
            "api_port": str(self.port),
        }

    def _create_service_info(self) -> ServiceInfo:
        """Create the mDNS ServiceInfo."""
        local_ip = _get_local_ip()
        properties = self._build_properties()

        # Service name must be unique on the network
        service_name = f"{self.node_id}.{SERVICE_TYPE}"

        return ServiceInfo(
            type_=SERVICE_TYPE,
            name=service_name,
            addresses=[socket.inet_aton(local_ip)],
            port=self.port,
            properties=properties,
            server=f"{self.node_id}.local.",
        )

    async def start(self) -> bool:
        """
        Start announcing the node via mDNS.

        Returns:
            True if successfully started, False otherwise
        """
        if self._running:
            logger.warning("Announcer already running")
            return True

        try:
            # Create Zeroconf instance
            self._zeroconf = AsyncZeroconf(ip_version=IPVersion.V4Only)

            # Create and register service
            self._service_info = self._create_service_info()

            await self._zeroconf.async_register_service(self._service_info)

            self._running = True
            logger.info(
                "mDNS service registered: %s on port %d",
                self._service_info.name,
                self.port,
            )
            logger.info("Node ID: %s, IP: %s", self.node_id, _get_local_ip())

            return True

        except Exception as e:
            logger.error("Failed to start mDNS announcer: %s", e)
            return False

    async def stop(self) -> None:
        """Stop announcing and unregister the service."""
        if not self._running:
            return

        try:
            if self._zeroconf and self._service_info:
                await self._zeroconf.async_unregister_service(self._service_info)
                await self._zeroconf.async_close()

            self._running = False
            logger.info("mDNS service unregistered: %s", self.node_id)

        except Exception as e:
            logger.error("Error stopping mDNS announcer: %s", e)

    async def update_properties(self) -> None:
        """Update TXT record properties (e.g., after camera count changes)."""
        if not self._running or not self._zeroconf or not self._service_info:
            return

        try:
            # Unregister old service
            await self._zeroconf.async_unregister_service(self._service_info)

            # Create new service info with updated properties
            self._service_info = self._create_service_info()

            # Register updated service
            await self._zeroconf.async_register_service(self._service_info)

            logger.debug("mDNS properties updated for %s", self.node_id)

        except Exception as e:
            logger.warning("Failed to update mDNS properties: %s", e)

    @property
    def is_running(self) -> bool:
        """Check if announcer is running."""
        return self._running


# Global announcer instance
_node_announcer: Optional[NodeAnnouncer] = None


def get_node_announcer() -> NodeAnnouncer:
    """Get or create the global node announcer."""
    global _node_announcer
    if _node_announcer is None:
        _node_announcer = NodeAnnouncer()
    return _node_announcer
