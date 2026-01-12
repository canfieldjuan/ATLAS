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
from .scanners import SSDPScanner, ScanResult

logger = logging.getLogger("atlas.discovery.service")


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

        # Future: mDNS scanner
        # if settings.discovery.mdns_enabled:
        #     self._scanners.append(MDNSScanner())

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

        # Only auto-register known device types
        if device.device_type == "roku":
            try:
                from ..capabilities.backends.roku import RokuBackend
                from ..capabilities.devices.media import RokuTV

                backend = RokuBackend(device.host)
                await backend.connect()

                roku_device = RokuTV(device.device_id, device.name, backend)
                capability_registry.register(roku_device)

                logger.info("Auto-registered Roku: %s (%s)", device.device_id, device.name)
                return True

            except Exception as e:
                logger.warning("Failed to auto-register Roku %s: %s", device.device_id, e)
                return False

        # Future: Add handlers for other device types
        # elif device.device_type == "chromecast":
        #     ...

        logger.debug(
            "No auto-register handler for device type: %s (%s)",
            device.device_type,
            device.device_id,
        )
        return False

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
        repo = get_device_repo()
        return await repo.get_all_devices()

    async def get_active_devices(self) -> list[DiscoveredDevice]:
        """Get all active (reachable) devices from database."""
        repo = get_device_repo()
        return await repo.get_active_devices()


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
