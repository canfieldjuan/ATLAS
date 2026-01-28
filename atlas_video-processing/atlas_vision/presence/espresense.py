"""
ESPresense MQTT subscriber for BLE-based room presence detection.

ESPresense uses ESP32 devices as room-level BLE beacons. Each ESP32 reports
distance to tracked BLE devices (phones, watches, etc.) via MQTT.

Topic format: espresense/rooms/{room}/{device_id}
Payload: {"distance": 1.5, "rssi": -65, "raw": -70.5, ...}

See: https://espresense.com/
"""

import asyncio
import json
import logging
import re
from typing import Optional

from .service import PresenceService, get_presence_service
from .config import presence_config

logger = logging.getLogger("atlas.vision.presence.espresense")


class ESPresenseSubscriber:
    """
    Subscribes to ESPresense MQTT topics and feeds readings to PresenceService.
    """

    def __init__(
        self,
        presence_service: Optional[PresenceService] = None,
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
        mqtt_username: Optional[str] = None,
        mqtt_password: Optional[str] = None,
        topic_prefix: str = "espresense/rooms",
    ):
        self.presence_service = presence_service
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.mqtt_username = mqtt_username
        self.mqtt_password = mqtt_password
        self.topic_prefix = topic_prefix

        self._client = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Topic pattern: espresense/rooms/{room}/{device_id}
        self._topic_pattern = re.compile(
            rf"^{re.escape(topic_prefix)}/([^/]+)/([^/]+)$"
        )

    @property
    def is_running(self) -> bool:
        """Check if subscriber is running."""
        return self._running

    async def start(self) -> None:
        """Start the MQTT subscriber."""
        if self._running:
            return

        self._running = True

        # Lazy load presence service
        if self.presence_service is None:
            self.presence_service = get_presence_service()

        self._task = asyncio.create_task(self._run())
        logger.info(
            "ESPresense subscriber started (host=%s, topic=%s/#)",
            self.mqtt_host,
            self.topic_prefix,
        )

    async def stop(self) -> None:
        """Stop the MQTT subscriber."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ESPresense subscriber stopped")

    async def _run(self) -> None:
        """Main loop - connect and process messages."""
        try:
            import aiomqtt
        except ImportError:
            logger.error("aiomqtt not installed. Run: pip install aiomqtt")
            return

        while self._running:
            try:
                async with aiomqtt.Client(
                    hostname=self.mqtt_host,
                    port=self.mqtt_port,
                    username=self.mqtt_username,
                    password=self.mqtt_password,
                ) as client:
                    # Subscribe to all ESPresense room messages
                    await client.subscribe(f"{self.topic_prefix}/#")
                    logger.info("Subscribed to %s/#", self.topic_prefix)

                    async for message in client.messages:
                        try:
                            await self._handle_message(message)
                        except Exception as e:
                            logger.warning("Error handling message: %s", e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("MQTT connection error: %s", e)
                if self._running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)

    async def _handle_message(self, message) -> None:
        """Handle an incoming MQTT message."""
        topic = str(message.topic)
        match = self._topic_pattern.match(topic)

        if not match:
            # Could be a different ESPresense topic (settings, etc.)
            return

        room_id = match.group(1)
        device_id = match.group(2)

        try:
            payload = json.loads(message.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.debug("Invalid JSON from %s: %s", topic, e)
            return

        # Extract distance (required)
        distance = payload.get("distance")
        if distance is None:
            return

        # Extract optional RSSI
        rssi = payload.get("rssi")

        logger.debug(
            "BLE reading: device=%s, room=%s, distance=%.2fm, rssi=%s",
            device_id,
            room_id,
            distance,
            rssi,
        )

        # Send to presence service
        await self.presence_service.handle_ble_reading(
            device_id=device_id,
            room_id=room_id,
            distance=distance,
            rssi=rssi,
        )


# Singleton
_espresense_subscriber: Optional[ESPresenseSubscriber] = None


async def start_espresense_subscriber(
    mqtt_host: str = "localhost",
    mqtt_port: int = 1883,
    mqtt_username: Optional[str] = None,
    mqtt_password: Optional[str] = None,
) -> ESPresenseSubscriber:
    """Start the ESPresense subscriber singleton."""
    global _espresense_subscriber

    if _espresense_subscriber is None:
        _espresense_subscriber = ESPresenseSubscriber(
            mqtt_host=mqtt_host,
            mqtt_port=mqtt_port,
            mqtt_username=mqtt_username,
            mqtt_password=mqtt_password,
            topic_prefix=presence_config.espresense_topic_prefix,
        )

    await _espresense_subscriber.start()
    return _espresense_subscriber


async def stop_espresense_subscriber() -> None:
    """Stop the ESPresense subscriber."""
    global _espresense_subscriber
    if _espresense_subscriber:
        await _espresense_subscriber.stop()
        _espresense_subscriber = None


def get_espresense_subscriber() -> Optional[ESPresenseSubscriber]:
    """Get the ESPresense subscriber singleton."""
    return _espresense_subscriber
