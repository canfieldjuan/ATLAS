"""
Discovery Service - Orchestrates network device discovery.

Coordinates scanners, persists results, and optionally auto-registers
discovered devices to the capability registry.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4

from ..config import settings
from ..storage.database import get_db_pool
from ..storage.models import DiscoveredDevice
from ..storage.repositories.device import get_device_repo
from .scanners import SSDPScanner, MDNSScanner, ScanResult

logger = logging.getLogger("atlas.discovery.service")

DEVICE_TYPE_TO_ASSET_TYPE = {
    "roku": "sensor",
    "chromecast": "sensor",
    "smart_tv": "sensor",
    "media_renderer": "sensor",
    "router": "sensor",
    "speaker": "sensor",
    "drone": "drone",
    "vehicle": "vehicle",
    "camera": "sensor",
    "thermostat": "sensor",
    "sensor": "sensor",
}


class DiscoveryService:
    """
    Main discovery service that coordinates network scanning.

    Features:
    - Multiple scanner support (SSDP, mDNS, etc.)
    - Database persistence of discovered devices
    - Optional auto-registration to capability registry
    - Periodic background scanning
    """

    def __init__(self):
        self._scanners = []
        self._initialized = False
        self._scan_task: Optional[asyncio.Task] = None
        self._running = False

        # Initialize scanners based on config
        if settings.discovery.ssdp_enabled:
            self._scanners.append(SSDPScanner())
            logger.info("SSDP scanner enabled")

        # mDNS scanner for Atlas node discovery
        if settings.discovery.mdns_enabled:
            self._scanners.append(MDNSScanner())
            logger.info("mDNS scanner enabled")

    async def initialize(self) -> None:
        """Initialize the discovery service."""
        if self._initialized:
            return

        logger.info("Initializing discovery service")

        # Run migrations if database is available
        pool = get_db_pool()
        if pool.is_initialized:
            try:
                from ..storage.migrations import run_migrations
                await run_migrations(pool)
            except Exception as e:
                logger.warning("Migration check failed (may already exist): %s", e)

        self._initialized = True
        logger.info("Discovery service initialized with %d scanners", len(self._scanners))

    async def shutdown(self) -> None:
        """Shutdown the discovery service."""
        logger.info("Shutting down discovery service")

        # Stop periodic scanning
        self._running = False
        if self._scan_task and not self._scan_task.done():
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass

        self._initialized = False
        logger.info("Discovery service shutdown complete")

    async def scan(self, timeout: float = 5.0) -> list[DiscoveredDevice]:
        """
        Run a network scan using all enabled scanners.

        Args:
            timeout: Scan timeout in seconds

        Returns:
            List of discovered devices
        """
        if not self._scanners:
            logger.warning("No scanners configured")
            return []

        logger.info("Starting network scan...")

        all_results: list[ScanResult] = []

        # Run all scanners
        for scanner in self._scanners:
            try:
                if await scanner.is_available():
                    results = await scanner.scan(timeout=timeout)
                    all_results.extend(results)
                    logger.info(
                        "%s scanner found %d devices",
                        scanner.protocol_name.upper(),
                        len(results),
                    )
                else:
                    logger.warning("%s scanner not available", scanner.protocol_name)
            except Exception as e:
                logger.error("%s scanner failed: %s", scanner.protocol_name, e)

        # Convert to DiscoveredDevice and persist
        devices = []
        repo = get_device_repo()

        for result in all_results:
            device = self._scan_result_to_device(result)
            devices.append(device)

            # Persist to database if enabled
            if settings.discovery.persist_devices:
                try:
                    await repo.save_device(device)
                except Exception as e:
                    logger.warning("Failed to persist device %s: %s", device.device_id, e)

            # Auto-register if enabled
            if settings.discovery.auto_register:
                try:
                    registered = await self._auto_register_device(device)
                    if registered:
                        device.auto_registered = True
                        await repo.mark_registered(device.device_id)
                except Exception as e:
                    logger.warning("Failed to auto-register %s: %s", device.device_id, e)

            # Notify security monitor for asset tracking
            await self._notify_security_asset_tracker(device)

        logger.info("Scan complete: %d devices discovered", len(devices))
        return devices

    def _scan_result_to_device(self, result: ScanResult) -> DiscoveredDevice:
        """Convert a ScanResult to a DiscoveredDevice."""
        device_id = result.generate_device_id()
        now = datetime.utcnow()

        return DiscoveredDevice(
            id=uuid4(),
            device_id=device_id,
            name=result.name or f"{result.device_type or 'Unknown'} ({result.host})",
            device_type=result.device_type or "unknown",
            protocol=result.protocol,
            host=result.host,
            port=result.port,
            discovered_at=now,
            last_seen_at=now,
            is_active=True,
            auto_registered=False,
            metadata={
                "headers": result.headers,
                "services": result.services,
                "manufacturer": result.manufacturer,
                "model": result.model,
            },
        )

    async def _auto_register_device(self, device: DiscoveredDevice) -> bool:
        """
        Attempt to auto-register a device to the capability registry.

        Only registers device types we have backend support for.

        Returns:
            True if registered, False otherwise
        """
        from ..capabilities.registry import capability_registry

        # Check if already registered
        if capability_registry.get(device.device_id):
            logger.debug("Device %s already registered", device.device_id)
            return True

        # Auto-registration of discovered devices is handled by Home Assistant.
        # Discovery still detects devices (e.g., Roku via SSDP) for informational
        # purposes, but HA owns device lifecycle and control.

        logger.debug(
            "No auto-register handler for device type: %s (%s)",
            device.device_type,
            device.device_id,
        )
        return False

    async def _notify_security_asset_tracker(self, device: DiscoveredDevice) -> None:
        """
        Notify security monitor when a device is discovered for asset tracking.

        Maps discovered device types to security asset types (drone, vehicle, sensor).
        """
        if not settings.security.asset_tracking_enabled:
            return

        asset_type = DEVICE_TYPE_TO_ASSET_TYPE.get(device.device_type)
        if not asset_type:
            logger.debug(
                "No asset mapping for device type: %s",
                device.device_type,
            )
            return

        try:
            from ..security import get_security_monitor

            monitor = get_security_monitor()
            if not monitor.is_running:
                return

            metadata = {
                "host": device.host,
                "name": device.name,
                "protocol": device.protocol,
                "manufacturer": device.metadata.get("manufacturer"),
                "model": device.metadata.get("model"),
            }

            monitor.observe_asset(
                asset_type=asset_type,
                identifier=device.device_id,
                metadata=metadata,
            )
            logger.debug(
                "Notified security tracker: %s -> %s",
                device.device_id,
                asset_type,
            )
        except Exception as e:
            logger.warning(
                "Failed to notify security asset tracker for %s: %s",
                device.device_id,
                e,
            )

    async def start_periodic_scan(self) -> None:
        """Start periodic background scanning."""
        if self._running:
            return

        self._running = True
        interval = settings.discovery.scan_interval_seconds

        async def _periodic_scan():
            while self._running:
                try:
                    await asyncio.sleep(interval)
                    if self._running:
                        await self.scan()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Periodic scan failed: %s", e)

        self._scan_task = asyncio.create_task(_periodic_scan())
        logger.info("Started periodic scanning (interval=%ds)", interval)

    async def stop_periodic_scan(self) -> None:
        """Stop periodic background scanning."""
        self._running = False
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped periodic scanning")

    async def get_discovered_devices(self) -> list[DiscoveredDevice]:
        """Get all discovered devices from database."""
        try:
            repo = get_device_repo()
            return await repo.get_all_devices()
        except Exception as e:
            logger.warning("Failed to get discovered devices: %s", e)
            return []

    async def get_active_devices(self) -> list[DiscoveredDevice]:
        """Get all active (reachable) devices from database."""
        try:
            repo = get_device_repo()
            return await repo.get_active_devices()
        except Exception as e:
            logger.warning("Failed to get active devices: %s", e)
            return []


# Global service instance
_discovery_service: Optional[DiscoveryService] = None


def get_discovery_service() -> DiscoveryService:
    """Get or create the global discovery service."""
    global _discovery_service
    if _discovery_service is None:
        _discovery_service = DiscoveryService()
    return _discovery_service


async def init_discovery() -> None:
    """Initialize the discovery service (call from app startup)."""
    service = get_discovery_service()
    await service.initialize()


async def shutdown_discovery() -> None:
    """Shutdown the discovery service (call from app shutdown)."""
    service = get_discovery_service()
    await service.shutdown()


async def run_discovery_scan(timeout: float = 5.0) -> list[DiscoveredDevice]:
    """Run a discovery scan and return results."""
    service = get_discovery_service()
    return await service.scan(timeout=timeout)
