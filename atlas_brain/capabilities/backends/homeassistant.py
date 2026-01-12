"""
Home Assistant REST API backend.

Enables control of devices through a Home Assistant instance.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger("atlas.backends.homeassistant")


class HomeAssistantBackend:
    """Home Assistant REST API backend."""

    def __init__(
        self,
        base_url: str,
        access_token: str,
    ):
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token
        self._client: Any = None
        self._connected = False

    @property
    def backend_type(self) -> str:
        return "homeassistant"

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Connect to the Home Assistant API."""
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed. Run: pip install httpx")
            raise RuntimeError("httpx package required for Home Assistant backend")

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

        # Verify connection
        try:
            resp = await self._client.get("/api/")
            resp.raise_for_status()
            self._connected = True
            logger.info("Connected to Home Assistant at %s", self.base_url)
        except Exception as e:
            await self._client.aclose()
            self._client = None
            logger.error("Failed to connect to Home Assistant: %s", e)
            raise

    async def disconnect(self) -> None:
        """Disconnect from Home Assistant."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._connected = False
            logger.info("Disconnected from Home Assistant")

    async def send_command(
        self,
        service_path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Call a Home Assistant service.

        Args:
            service_path: Service path, e.g., "light/turn_on" or "switch/toggle"
            payload: Service data (entity_id, etc.)

        Returns:
            Response from Home Assistant
        """
        if not self._connected or not self._client:
            raise RuntimeError("Home Assistant client not connected")

        url = f"/api/services/{service_path}"
        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()

        logger.info("HA service called: %s with %s", service_path, payload)
        return resp.json() if resp.content else {"status": "ok"}

    async def get_state(self, entity_id: str) -> dict[str, Any]:
        """
        Get state of a Home Assistant entity.

        Checks the state cache first for instant response. Falls back to REST
        API if cache miss or expired, and updates the cache with fresh data.

        Args:
            entity_id: Entity ID, e.g., "light.living_room"

        Returns:
            Entity state dict
        """
        # Try cache first for instant response
        try:
            from ..state_cache import get_state_cache

            cache = get_state_cache()
            cached = await cache.get(entity_id)

            if cached:
                logger.debug("Cache hit: %s", entity_id)
                return cached.to_dict()

            logger.debug("Cache miss: %s", entity_id)
        except Exception as e:
            logger.debug("Cache lookup failed for %s: %s", entity_id, e)

        # Fallback to REST API
        if not self._connected or not self._client:
            raise RuntimeError("Home Assistant client not connected")

        resp = await self._client.get(f"/api/states/{entity_id}")
        resp.raise_for_status()
        state_data = resp.json()

        # Update cache with fresh data
        try:
            from datetime import datetime
            from ..state_cache import get_state_cache

            cache = get_state_cache()

            last_changed = None
            last_updated = None
            try:
                if state_data.get("last_changed"):
                    last_changed = datetime.fromisoformat(
                        state_data["last_changed"].replace("Z", "+00:00")
                    )
                if state_data.get("last_updated"):
                    last_updated = datetime.fromisoformat(
                        state_data["last_updated"].replace("Z", "+00:00")
                    )
            except (ValueError, TypeError):
                pass

            await cache.set(
                entity_id=entity_id,
                state=state_data.get("state", "unknown"),
                attributes=state_data.get("attributes", {}),
                last_changed=last_changed,
                last_updated=last_updated,
            )
        except Exception as e:
            logger.debug("Cache update failed for %s: %s", entity_id, e)

        return state_data

    async def list_entities(self, domain_filter: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """
        List all entities, optionally filtered by domain.

        Args:
            domain_filter: List of domain prefixes to include (e.g., ["light.", "switch."])

        Returns:
            List of entity states
        """
        if not self._connected or not self._client:
            raise RuntimeError("Home Assistant client not connected")

        resp = await self._client.get("/api/states")
        resp.raise_for_status()
        entities = resp.json()

        if domain_filter:
            entities = [
                e for e in entities
                if any(e["entity_id"].startswith(d) for d in domain_filter)
            ]

        return entities
