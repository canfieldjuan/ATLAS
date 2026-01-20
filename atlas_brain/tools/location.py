"""
Location tool using Home Assistant device tracker.

Provides GPS location from phone via HA Companion App.
"""

import logging
from typing import Any

import httpx

from ..config import settings
from .base import Tool, ToolParameter, ToolResult

logger = logging.getLogger("atlas.tools.location")


class LocationTool:
    """Location tracking tool using Home Assistant."""

    def __init__(self) -> None:
        self._ha_config = settings.homeassistant
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "get_location"

    @property
    def description(self) -> str:
        return "Get current GPS location from phone"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="entity_id",
                param_type="string",
                description="Device tracker or person entity ID (optional)",
                required=False,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["location", "where am i", "my location", "gps"]

    @property
    def category(self) -> str:
        return "utility"

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Execute location query."""
        if not self._ha_config.enabled:
            return ToolResult(
                success=False,
                error="HA_DISABLED",
                message="Home Assistant integration is disabled",
            )

        if not self._ha_config.token:
            return ToolResult(
                success=False,
                error="NO_TOKEN",
                message="Home Assistant token not configured",
            )

        entity_id = params.get("entity_id")

        try:
            if entity_id:
                location_data = await self._fetch_entity_location(entity_id)
            else:
                location_data = await self._find_location()

            if not location_data:
                return ToolResult(
                    success=False,
                    error="NO_LOCATION",
                    message="No location data available. Check Companion app settings.",
                )

            return ToolResult(
                success=True,
                data=location_data,
                message=self._format_message(location_data),
            )
        except httpx.HTTPStatusError as e:
            logger.error("HA API HTTP error: %s", e)
            return ToolResult(
                success=False,
                error="API_ERROR",
                message=f"Home Assistant API error: {e.response.status_code}",
            )
        except Exception as e:
            logger.exception("Location tool error")
            return ToolResult(
                success=False,
                error="EXECUTION_ERROR",
                message=str(e),
            )

    async def _fetch_entity_location(self, entity_id: str) -> dict[str, Any] | None:
        """Fetch location from a specific entity."""
        client = await self._ensure_client()

        url = f"{self._ha_config.url}/api/states/{entity_id}"
        headers = {
            "Authorization": f"Bearer {self._ha_config.token}",
            "Content-Type": "application/json",
        }

        response = await client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        attrs = data.get("attributes", {})
        lat = attrs.get("latitude")
        lon = attrs.get("longitude")

        if lat is None or lon is None:
            return None

        return {
            "entity_id": entity_id,
            "latitude": lat,
            "longitude": lon,
            "gps_accuracy": attrs.get("gps_accuracy"),
            "state": data.get("state"),
            "friendly_name": attrs.get("friendly_name", entity_id),
            "battery_level": attrs.get("battery_level"),
        }

    async def _find_location(self) -> dict[str, Any] | None:
        """Find location from any available device tracker or person."""
        client = await self._ensure_client()

        url = f"{self._ha_config.url}/api/states"
        headers = {
            "Authorization": f"Bearer {self._ha_config.token}",
            "Content-Type": "application/json",
        }

        response = await client.get(url, headers=headers)
        response.raise_for_status()
        entities = response.json()

        for entity in entities:
            eid = entity.get("entity_id", "")
            if not (eid.startswith("device_tracker.") or eid.startswith("person.")):
                continue

            attrs = entity.get("attributes", {})
            lat = attrs.get("latitude")
            lon = attrs.get("longitude")

            if lat is not None and lon is not None:
                return {
                    "entity_id": eid,
                    "latitude": lat,
                    "longitude": lon,
                    "gps_accuracy": attrs.get("gps_accuracy"),
                    "state": entity.get("state"),
                    "friendly_name": attrs.get("friendly_name", eid),
                    "battery_level": attrs.get("battery_level"),
                }

        return None

    def _format_message(self, data: dict[str, Any]) -> str:
        """Format location data as human-readable message."""
        lat = data.get("latitude")
        lon = data.get("longitude")
        name = data.get("friendly_name", "Device")
        state = data.get("state", "unknown")
        accuracy = data.get("gps_accuracy")

        msg = f"{name} location: {lat}, {lon}"
        if state and state not in ("unknown", "not_home"):
            msg = f"{name} is at {state}. Coordinates: {lat}, {lon}"
        elif state == "not_home":
            msg = f"{name} is away from home. Coordinates: {lat}, {lon}"

        if accuracy:
            msg += f" (accuracy: {accuracy}m)"

        return msg


# Module-level instance
location_tool = LocationTool()
