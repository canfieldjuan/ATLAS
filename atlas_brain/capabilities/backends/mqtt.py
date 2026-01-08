"""
MQTT backend for device communication.

Supports devices like Tasmota, Shelly, or custom ESP devices.
"""

import asyncio
import json
import logging
from typing import Any, Optional

logger = logging.getLogger("atlas.backends.mqtt")


class MQTTBackend:
    """MQTT-based backend for device communication."""

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        self._client: Any = None
        self._state_cache: dict[str, Any] = {}
        self._connected = False

    @property
    def backend_type(self) -> str:
        return "mqtt"

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Connect to the MQTT broker."""
        try:
            import aiomqtt
        except ImportError:
            logger.error("aiomqtt not installed. Run: pip install aiomqtt")
            raise RuntimeError("aiomqtt package required for MQTT backend")

        try:
            self._client = aiomqtt.Client(
                hostname=self.broker_host,
                port=self.broker_port,
                username=self.username,
                password=self.password,
            )
            await self._client.__aenter__()
            self._connected = True
            logger.info("MQTT connected to %s:%d", self.broker_host, self.broker_port)
        except Exception as e:
            logger.error("Failed to connect to MQTT broker: %s", e)
            raise

    async def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        if self._client and self._connected:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("Error during MQTT disconnect: %s", e)
            finally:
                self._connected = False
                self._client = None
            logger.info("MQTT disconnected")

    async def send_command(
        self,
        topic: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Publish a command to an MQTT topic.

        Args:
            topic: MQTT topic to publish to
            payload: JSON-serializable payload

        Returns:
            Status dict indicating success
        """
        if not self._connected or not self._client:
            raise RuntimeError("MQTT client not connected")

        payload_str = json.dumps(payload)
        await self._client.publish(topic, payload_str)
        logger.info("MQTT published to %s: %s", topic, payload)

        return {"status": "sent", "topic": topic}

    async def get_state(self, topic: str) -> dict[str, Any]:
        """
        Get cached state for a topic.

        Note: For real-time state, devices should be subscribed to
        and state updated via message handler.
        """
        return self._state_cache.get(topic, {})

    def update_state_cache(self, topic: str, state: dict[str, Any]) -> None:
        """Update the state cache for a topic (called from subscription handler)."""
        self._state_cache[topic] = state

    async def subscribe(self, topic: str) -> None:
        """Subscribe to a topic for state updates."""
        if not self._connected or not self._client:
            raise RuntimeError("MQTT client not connected")

        await self._client.subscribe(topic)
        logger.info("MQTT subscribed to %s", topic)
