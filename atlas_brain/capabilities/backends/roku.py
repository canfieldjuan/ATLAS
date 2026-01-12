"""
Roku External Control Protocol (ECP) backend.

Direct control of Roku devices over the local network.
"""

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger("atlas.backends.roku")


class RokuBackend:
    """Roku ECP backend for direct device control."""

    def __init__(self, host: str, port: int = 8060):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self._client: Optional[httpx.AsyncClient] = None
        self._device_info: Optional[dict] = None

    @property
    def backend_type(self) -> str:
        return "roku"

    async def connect(self) -> None:
        """Connect to the Roku device and fetch device info."""
        self._client = httpx.AsyncClient(timeout=10.0)

        try:
            # Fetch device info to verify connection
            resp = await self._client.get(f"{self.base_url}/query/device-info")
            resp.raise_for_status()
            self._device_info = self._parse_xml(resp.text)
            logger.info(
                "Connected to Roku: %s at %s",
                self._device_info.get("friendly-device-name", "Unknown"),
                self.host,
            )
        except Exception as e:
            await self._client.aclose()
            self._client = None
            logger.error("Failed to connect to Roku at %s: %s", self.host, e)
            raise

    async def disconnect(self) -> None:
        """Disconnect from the Roku device."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._device_info = None
            logger.info("Disconnected from Roku at %s", self.host)

    def _parse_xml(self, xml_text: str) -> dict:
        """Simple XML to dict parser for Roku responses."""
        import re
        result = {}
        # Match <tag>value</tag> patterns
        pattern = r"<([^/>]+)>([^<]*)</\1>"
        for match in re.finditer(pattern, xml_text):
            result[match.group(1)] = match.group(2)
        return result

    @property
    def device_name(self) -> str:
        """Get the friendly device name."""
        if self._device_info:
            return self._device_info.get(
                "friendly-device-name",
                self._device_info.get("user-device-name", "Roku"),
            )
        return "Roku"

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    async def get_power_state(self) -> str:
        """Get current power state (PowerOn, Standby, etc.)."""
        if not self._client:
            raise RuntimeError("Roku client not connected")

        try:
            resp = await self._client.get(f"{self.base_url}/query/device-info")
            resp.raise_for_status()
            info = self._parse_xml(resp.text)
            return info.get("power-mode", "unknown")
        except Exception:
            # Device in standby may not respond
            return "Standby"

    async def keypress(self, key: str) -> bool:
        """Send a keypress command to the Roku."""
        if not self._client:
            raise RuntimeError("Roku client not connected")

        try:
            resp = await self._client.post(f"{self.base_url}/keypress/{key}")
            resp.raise_for_status()
            logger.debug("Sent keypress: %s", key)
            return True
        except Exception as e:
            logger.warning("Keypress %s failed: %s", key, e)
            return False

    async def power_on(self) -> bool:
        """Wake up / power on the Roku."""
        return await self.keypress("PowerOn")

    async def power_off(self) -> bool:
        """Put Roku to sleep / standby."""
        return await self.keypress("PowerOff")

    async def power_toggle(self) -> bool:
        """Toggle power state."""
        return await self.keypress("Power")

    async def launch_app(self, app_id: str) -> bool:
        """Launch an app by ID."""
        if not self._client:
            raise RuntimeError("Roku client not connected")

        try:
            resp = await self._client.post(f"{self.base_url}/launch/{app_id}")
            resp.raise_for_status()
            logger.info("Launched app: %s", app_id)
            return True
        except Exception as e:
            logger.warning("Failed to launch app %s: %s", app_id, e)
            return False

    async def get_apps(self) -> list[dict]:
        """Get list of installed apps."""
        if not self._client:
            raise RuntimeError("Roku client not connected")

        try:
            resp = await self._client.get(f"{self.base_url}/query/apps")
            resp.raise_for_status()
            # Parse app list XML
            import re
            apps = []
            pattern = r'<app id="([^"]+)"[^>]*>([^<]+)</app>'
            for match in re.finditer(pattern, resp.text):
                apps.append({"id": match.group(1), "name": match.group(2)})
            return apps
        except Exception as e:
            logger.warning("Failed to get apps: %s", e)
            return []

    async def get_active_app(self) -> Optional[dict]:
        """Get the currently active app."""
        if not self._client:
            raise RuntimeError("Roku client not connected")

        try:
            resp = await self._client.get(f"{self.base_url}/query/active-app")
            resp.raise_for_status()
            import re
            match = re.search(r'<app id="([^"]+)"[^>]*>([^<]+)</app>', resp.text)
            if match:
                return {"id": match.group(1), "name": match.group(2)}
            return None
        except Exception:
            return None
