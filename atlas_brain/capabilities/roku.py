"""
Roku integration bootstrap.

Handles connection and device registration for Roku TVs.
"""

import logging
from typing import Optional

from ..config import settings
from .backends.roku import RokuBackend
from .devices.media import RokuTV
from .registry import capability_registry

logger = logging.getLogger("atlas.capabilities.roku")

# Module-level backend references for lifecycle management
_roku_backends: list[RokuBackend] = []


async def init_roku() -> list[str]:
    """
    Initialize Roku devices from configuration.

    Returns:
        List of registered device IDs
    """
    global _roku_backends

    if not settings.roku.enabled:
        logger.info("Roku integration disabled")
        return []

    devices = settings.roku.devices
    if not devices:
        logger.warning("Roku enabled but no devices configured")
        return []

    registered = []

    for device_config in devices:
        host = device_config.get("host")
        name = device_config.get("name", f"Roku ({host})")

        if not host:
            logger.warning("Roku device config missing 'host', skipping")
            continue

        device_id = f"roku.{host.replace('.', '_')}"

        try:
            backend = RokuBackend(host)
            await backend.connect()
            _roku_backends.append(backend)

            # Use device name from Roku if not specified
            if name == f"Roku ({host})":
                name = backend.device_name

            device = RokuTV(device_id, name, backend)
            capability_registry.register(device)
            registered.append(device_id)

            logger.info("Registered Roku: %s (%s) at %s", device_id, name, host)

        except Exception as e:
            logger.error("Failed to connect to Roku at %s: %s", host, e)

    logger.info("Registered %d Roku devices", len(registered))
    return registered


async def shutdown_roku() -> None:
    """Disconnect all Roku devices."""
    global _roku_backends

    for backend in _roku_backends:
        try:
            await backend.disconnect()
        except Exception as e:
            logger.warning("Error disconnecting Roku: %s", e)

    _roku_backends = []
    logger.info("Roku backends disconnected")
