"""
Home Assistant client for edge devices.

Provides REST API access to Home Assistant for device control,
optimized for local network operation without brain connectivity.
"""

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger("atlas.edge.capabilities.homeassistant")


class EdgeHomeAssistant:
    """
    Home Assistant client for edge devices.

    Provides:
    - REST API for device control
    - Entity state queries
    - Entity name to ID resolution
    """

    def __init__(
        self,
        base_url: str,
        access_token: str,
    ):
        """
        Initialize Home Assistant client.

        Args:
            base_url: Home Assistant base URL (e.g., http://homeassistant.local:8123)
            access_token: Long-lived access token
        """
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token
        self._client: Any = None
        self._connected = False
        self._entity_cache: dict[str, str] = {}  # name -> entity_id
        self._state_cache: dict[str, dict] = {}  # entity_id -> state

    @property
    def is_connected(self) -> bool:
        """Check if connected to Home Assistant."""
        return self._connected

    async def connect(self) -> None:
        """Connect to Home Assistant API."""
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed. Run: pip install httpx")
            raise RuntimeError("httpx package required")

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

        try:
            resp = await self._client.get("/api/")
            resp.raise_for_status()
            self._connected = True
            logger.info("Connected to Home Assistant at %s", self.base_url)

            # Load entity cache
            await self._load_entity_cache()

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
            self._entity_cache.clear()
            self._state_cache.clear()
            logger.info("Disconnected from Home Assistant")

    async def _load_entity_cache(self) -> None:
        """Load entity names for resolution."""
        if not self._connected or not self._client:
            return

        try:
            resp = await self._client.get("/api/states")
            resp.raise_for_status()
            entities = resp.json()

            for entity in entities:
                entity_id = entity.get("entity_id", "")
                attrs = entity.get("attributes", {})
                friendly_name = attrs.get("friendly_name", "")

                if friendly_name:
                    # Normalize name for lookup
                    name_lower = friendly_name.lower()
                    self._entity_cache[name_lower] = entity_id

                    # Also cache by last part of entity_id
                    if "." in entity_id:
                        short_name = entity_id.split(".", 1)[1].replace("_", " ")
                        self._entity_cache[short_name] = entity_id

            logger.info("Cached %d entities for name resolution", len(self._entity_cache))

        except Exception as e:
            logger.warning("Failed to load entity cache: %s", e)

    def resolve_entity_id(
        self,
        name: str,
        domain: Optional[str] = None,
    ) -> Optional[str]:
        """
        Resolve entity name to entity_id.

        Args:
            name: Friendly name or partial name
            domain: Optional domain filter (light, switch, etc.)

        Returns:
            entity_id if found, None otherwise
        """
        if not name:
            return None

        name_lower = name.lower().strip()

        # Direct lookup
        if name_lower in self._entity_cache:
            entity_id = self._entity_cache[name_lower]
            if not domain or entity_id.startswith(f"{domain}."):
                return entity_id

        # Try with domain prefix
        if domain:
            prefixed = f"{domain}.{name_lower.replace(' ', '_')}"
            for cached_name, entity_id in self._entity_cache.items():
                if entity_id == prefixed:
                    return entity_id

        # Fuzzy match
        for cached_name, entity_id in self._entity_cache.items():
            if name_lower in cached_name or cached_name in name_lower:
                if not domain or entity_id.startswith(f"{domain}."):
                    return entity_id

        logger.debug("Could not resolve entity: %s (domain=%s)", name, domain)
        return None

    async def call_service(
        self,
        service_path: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Call a Home Assistant service.

        Args:
            service_path: Service path (e.g., "light/turn_on")
            data: Service data (entity_id, etc.)

        Returns:
            Service response
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Home Assistant")

        url = f"/api/services/{service_path}"

        try:
            resp = await self._client.post(url, json=data)
            resp.raise_for_status()
            result = resp.json() if resp.content else {"status": "ok"}

            logger.info("HA service: %s with %s -> success", service_path, data)
            return result

        except Exception as e:
            logger.error("HA service failed: %s with %s -> %s", service_path, data, e)
            raise

    async def get_state(self, entity_id: str) -> dict[str, Any]:
        """
        Get state of an entity.

        Args:
            entity_id: Entity ID

        Returns:
            Entity state dict
        """
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Home Assistant")

        # Check cache first
        if entity_id in self._state_cache:
            return self._state_cache[entity_id]

        try:
            resp = await self._client.get(f"/api/states/{entity_id}")
            resp.raise_for_status()
            state = resp.json()

            # Cache the state
            self._state_cache[entity_id] = state
            return state

        except Exception as e:
            logger.error("Failed to get state for %s: %s", entity_id, e)
            raise

    async def turn_on(
        self,
        entity_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Turn on an entity.

        Args:
            entity_id: Entity ID
            **kwargs: Additional service data

        Returns:
            Service response
        """
        domain = entity_id.split(".")[0]
        data = {"entity_id": entity_id, **kwargs}
        return await self.call_service(f"{domain}/turn_on", data)

    async def turn_off(
        self,
        entity_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Turn off an entity.

        Args:
            entity_id: Entity ID
            **kwargs: Additional service data

        Returns:
            Service response
        """
        domain = entity_id.split(".")[0]
        data = {"entity_id": entity_id, **kwargs}
        return await self.call_service(f"{domain}/turn_off", data)

    async def toggle(
        self,
        entity_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Toggle an entity.

        Args:
            entity_id: Entity ID
            **kwargs: Additional service data

        Returns:
            Service response
        """
        domain = entity_id.split(".")[0]
        data = {"entity_id": entity_id, **kwargs}
        return await self.call_service(f"{domain}/toggle", data)


# Singleton instance
_ha_client: Optional[EdgeHomeAssistant] = None


async def get_homeassistant() -> EdgeHomeAssistant:
    """Get or create global Home Assistant client."""
    global _ha_client
    if _ha_client is None:
        from ..config import settings

        _ha_client = EdgeHomeAssistant(
            base_url=settings.homeassistant.url,
            access_token=settings.homeassistant.token or "",
        )
        await _ha_client.connect()
    return _ha_client
