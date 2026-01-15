"""
mDNS (Multicast DNS) scanner for Atlas node discovery.

Discovers Atlas Vision nodes on the local network that announce
themselves via the _atlas-node._tcp.local. service type.
"""

import asyncio
import logging
from typing import Optional

from zeroconf import ServiceStateChange, Zeroconf, IPVersion
from zeroconf.asyncio import AsyncServiceBrowser, AsyncZeroconf, AsyncServiceInfo

from .base import BaseScanner, ScanResult

logger = logging.getLogger("atlas.discovery.scanners.mdns")

# Service type for Atlas nodes
ATLAS_SERVICE_TYPE = "_atlas-node._tcp.local."


class MDNSScanner(BaseScanner):
    """
    mDNS/Zeroconf network scanner for Atlas nodes.

    Browses for _atlas-node._tcp.local. services and returns
    discovered Atlas Vision nodes with their metadata.
    """

    @property
    def protocol_name(self) -> str:
        return "mdns"

    async def scan(self, timeout: float = 5.0) -> list[ScanResult]:
        """
        Scan the network for Atlas nodes via mDNS.

        Args:
            timeout: How long to browse for services (seconds)

        Returns:
            List of discovered Atlas nodes
        """
        logger.info("Starting mDNS scan for Atlas nodes (timeout=%.1fs)", timeout)

        discovered_services: dict[str, dict] = {}

        # Callback for service state changes
        def on_service_state_change(
            zeroconf: Zeroconf,
            service_type: str,
            name: str,
            state_change: ServiceStateChange,
        ) -> None:
            if state_change == ServiceStateChange.Added:
                logger.debug("mDNS: Found service %s", name)
                # Store name for later resolution
                discovered_services[name] = {"name": name, "resolved": False}

        try:
            # Create async zeroconf instance
            aiozc = AsyncZeroconf(ip_version=IPVersion.V4Only)

            # Start browsing for Atlas services
            browser = AsyncServiceBrowser(
                aiozc.zeroconf,
                [ATLAS_SERVICE_TYPE],
                handlers=[on_service_state_change],
            )

            # Wait for services to be discovered
            await asyncio.sleep(timeout)

            # Resolve each discovered service to get details
            results = []
            for service_name in list(discovered_services.keys()):
                try:
                    service_info = AsyncServiceInfo(ATLAS_SERVICE_TYPE, service_name)
                    if await service_info.async_request(aiozc.zeroconf, timeout=2.0):
                        result = self._service_info_to_result(service_info)
                        if result:
                            results.append(result)
                            logger.debug(
                                "Resolved Atlas node: %s at %s:%d",
                                result.name,
                                result.host,
                                result.port or 0,
                            )
                except Exception as e:
                    logger.warning("Failed to resolve service %s: %s", service_name, e)

            # Cleanup
            await browser.async_cancel()
            await aiozc.async_close()

            logger.info("mDNS scan complete: found %d Atlas nodes", len(results))
            return results

        except Exception as e:
            logger.error("mDNS scan failed: %s", e)
            return []

    def _service_info_to_result(self, info: AsyncServiceInfo) -> Optional[ScanResult]:
        """Convert a ServiceInfo to a ScanResult."""
        # Get IP address
        addresses = info.parsed_addresses()
        if not addresses:
            logger.warning("No addresses for service %s", info.name)
            return None

        host = addresses[0]  # Use first address

        # Parse TXT record properties
        properties = {}
        if info.properties:
            for key, value in info.properties.items():
                if isinstance(key, bytes):
                    key = key.decode("utf-8", errors="ignore")
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="ignore")
                properties[key] = value

        # Extract node information
        node_id = properties.get("node_id", "unknown")
        node_type = properties.get("type", "unknown")
        version = properties.get("version", "unknown")
        cameras = properties.get("cameras", "0")
        capabilities = properties.get("capabilities", "")

        # Build friendly name
        name = node_id
        if node_type:
            name = f"{node_id} ({node_type})"

        return ScanResult(
            host=host,
            protocol="mdns",
            device_type="atlas_node",
            name=name,
            manufacturer="Atlas",
            model=f"Vision Node v{version}",
            port=info.port,
            headers={},
            services=capabilities.split(",") if capabilities else [],
            raw_data={
                "node_id": node_id,
                "node_type": node_type,
                "version": version,
                "cameras": cameras,
                "capabilities": capabilities,
                "service_name": info.name,
                "server": info.server,
            },
        )

    def identify_device_type(self, result: ScanResult) -> Optional[str]:
        """
        Identify device type from mDNS result.

        For Atlas nodes, always returns 'atlas_node'.
        """
        # All services from this scanner are Atlas nodes
        return "atlas_node"
