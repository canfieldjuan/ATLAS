"""
Home Assistant integration bootstrap.

Handles connection, auto-discovery, and device registration.
Supports real-time state updates via WebSocket subscriptions.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from ..config import settings
from .backends.homeassistant import HomeAssistantBackend
from .backends.homeassistant_ws import (
    HomeAssistantWebSocket,
    init_ws_client,
    shutdown_ws_client,
)
from .devices.lights import HomeAssistantLight
from .devices.media import HomeAssistantMediaPlayer
from .devices.switches import HomeAssistantSwitch
from .registry import capability_registry
from .state_cache import get_state_cache

logger = logging.getLogger("atlas.capabilities.homeassistant")

# Module-level backend references for lifecycle management
_ha_backend: Optional[HomeAssistantBackend] = None
_ws_client: Optional[HomeAssistantWebSocket] = None


async def _on_state_changed(event_data: dict[str, Any]) -> None:
    """Handle state_changed event from WebSocket."""
    cache = get_state_cache()
    await cache.update_from_event(event_data)

    # Log state changes at DEBUG level
    entity_id = event_data.get("entity_id", "unknown")
    new_state = event_data.get("new_state", {})
    state_value = new_state.get("state", "unknown")
    logger.debug("WS state_changed: %s -> %s", entity_id, state_value)


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

    Optionally starts WebSocket client for real-time state updates.

    Returns:
        List of registered entity IDs
    """
    global _ha_backend, _ws_client

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

    # Initialize WebSocket for real-time state updates (if enabled)
    if settings.homeassistant.websocket_enabled:
        try:
            _ws_client = await init_ws_client(
                url=url,
                token=token,
                on_state_changed=_on_state_changed,
            )
            logger.info("Home Assistant WebSocket connected for real-time state")

            # Pre-populate state cache with discovered entities
            cache = get_state_cache()
            for entity in entities:
                entity_id = entity.get("entity_id", "")
                if not entity_id:
                    continue

                # Parse timestamps safely
                last_changed = None
                last_updated = None
                try:
                    if entity.get("last_changed"):
                        last_changed = datetime.fromisoformat(
                            entity["last_changed"].replace("Z", "+00:00")
                        )
                    if entity.get("last_updated"):
                        last_updated = datetime.fromisoformat(
                            entity["last_updated"].replace("Z", "+00:00")
                        )
                except (ValueError, TypeError):
                    pass

                await cache.set(
                    entity_id=entity_id,
                    state=entity.get("state", "unknown"),
                    attributes=entity.get("attributes", {}),
                    last_changed=last_changed,
                    last_updated=last_updated,
                )

            logger.info("State cache populated with %d entities", len(entities))

        except Exception as e:
            logger.warning(
                "WebSocket init failed, falling back to REST-only: %s", e
            )
            _ws_client = None

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

            elif entity_id.startswith("input_boolean."):
                # Treat input_boolean as a switch (same on/off behavior)
                device = HomeAssistantSwitch(entity_id, name, _ha_backend)
                capability_registry.register(device)
                registered.append(entity_id)
                logger.info("Registered HA input_boolean as switch: %s (%s)", entity_id, name)

            elif entity_id.startswith("media_player."):
                device = HomeAssistantMediaPlayer(entity_id, name, _ha_backend)
                capability_registry.register(device)
                registered.append(entity_id)
                logger.info("Registered HA media_player: %s (%s)", entity_id, name)

        except Exception as e:
            logger.warning("Failed to register %s: %s", entity_id, e)

    # Invalidate device resolver index so it rebuilds with new devices
    try:
        from .device_resolver import get_device_resolver
        get_device_resolver().invalidate()
    except Exception:
        pass

    logger.info("Registered %d Home Assistant devices", len(registered))
    return registered


async def shutdown_homeassistant() -> None:
    """Disconnect from Home Assistant (REST and WebSocket)."""
    global _ha_backend, _ws_client

    # Stop WebSocket client first
    if _ws_client:
        try:
            await shutdown_ws_client()
            logger.info("Home Assistant WebSocket disconnected")
        except Exception as e:
            logger.warning("Error stopping WebSocket: %s", e)
        _ws_client = None

    # Clear state cache
    try:
        cache = get_state_cache()
        await cache.clear()
    except Exception as e:
        logger.warning("Error clearing state cache: %s", e)

    # Disconnect REST client
    if _ha_backend and _ha_backend.is_connected:
        await _ha_backend.disconnect()
        logger.info("Disconnected from Home Assistant REST API")

    _ha_backend = None
