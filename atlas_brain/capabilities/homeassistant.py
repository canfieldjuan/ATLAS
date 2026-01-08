"""
Home Assistant integration bootstrap.

Handles connection, auto-discovery, and device registration.
"""

import logging
from typing import Optional

from ..config import settings
from .backends.homeassistant import HomeAssistantBackend
from .devices.lights import HomeAssistantLight
from .devices.switches import HomeAssistantSwitch
from .registry import capability_registry

logger = logging.getLogger("atlas.capabilities.homeassistant")

# Module-level backend reference for lifecycle management
_ha_backend: Optional[HomeAssistantBackend] = None


def _friendly_name_from_entity(entity: dict) -> str:
    """Extract friendly name from HA entity, fallback to entity_id."""
    attrs = entity.get("attributes", {})
    friendly_name = attrs.get("friendly_name")
    if friendly_name:
        return friendly_name
    # Fallback: convert entity_id to readable name
    entity_id = entity.get("entity_id", "unknown")
    # "light.living_room" -> "Living Room"
    name_part = entity_id.split(".", 1)[-1]
    return name_part.replace("_", " ").title()


async def init_homeassistant() -> list[str]:
    """
    Initialize Home Assistant backend and auto-discover devices.

    Returns:
        List of registered entity IDs
    """
    global _ha_backend

    if not settings.homeassistant.enabled:
        logger.info("Home Assistant integration disabled")
        return []

    if not settings.homeassistant.token:
        logger.warning("Home Assistant enabled but no token configured")
        return []

    url = settings.homeassistant.url
    token = settings.homeassistant.token
    entity_filter = settings.homeassistant.entity_filter

    logger.info("Connecting to Home Assistant at %s", url)

    try:
        _ha_backend = HomeAssistantBackend(url, token)
        await _ha_backend.connect()
    except Exception as e:
        logger.error("Failed to connect to Home Assistant: %s", e)
        _ha_backend = None
        return []

    # Discover entities
    try:
        entities = await _ha_backend.list_entities(entity_filter)
        logger.info("Discovered %d entities from Home Assistant", len(entities))
    except Exception as e:
        logger.error("Failed to list Home Assistant entities: %s", e)
        return []

    registered = []

    for entity in entities:
        entity_id = entity.get("entity_id", "")
        name = _friendly_name_from_entity(entity)

        try:
            if entity_id.startswith("light."):
                device = HomeAssistantLight(entity_id, name, _ha_backend)
                capability_registry.register(device)
                registered.append(entity_id)
                logger.info("Registered HA light: %s (%s)", entity_id, name)

            elif entity_id.startswith("switch."):
                device = HomeAssistantSwitch(entity_id, name, _ha_backend)
                capability_registry.register(device)
                registered.append(entity_id)
                logger.info("Registered HA switch: %s (%s)", entity_id, name)

        except Exception as e:
            logger.warning("Failed to register %s: %s", entity_id, e)

    logger.info("Registered %d Home Assistant devices", len(registered))
    return registered


async def shutdown_homeassistant() -> None:
    """Disconnect from Home Assistant."""
    global _ha_backend

    if _ha_backend and _ha_backend.is_connected:
        await _ha_backend.disconnect()
        logger.info("Disconnected from Home Assistant")

    _ha_backend = None
