"""
MQTT publisher for detection events.

Publishes track events and status to MQTT topics for
Atlas Brain and other subscribers.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Optional

from ..core.config import settings
from ..core.models import Track, DetectionEvent

logger = logging.getLogger("atlas.vision.communication.mqtt")

# Topic patterns
# {node_id} is replaced with the actual node ID
TOPIC_EVENTS = "atlas/vision/{node_id}/events"
TOPIC_TRACKS = "atlas/vision/{node_id}/tracks"
TOPIC_STATUS = "atlas/vision/{node_id}/status"


class MQTTPublisher:
    """
    MQTT publisher for detection events.

    Publishes to topics:
    - atlas/vision/{node_id}/events - Detection events (new_track, track_lost)
    - atlas/vision/{node_id}/tracks - Active tracks (periodic updates)
    - atlas/vision/{node_id}/status - Node status (online, offline)
    """

    def __init__(
        self,
        node_id: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize MQTT publisher.

        Args:
            node_id: Node identifier for topic names
            host: MQTT broker host
            port: MQTT broker port
            username: MQTT username (optional)
            password: MQTT password (optional)
        """
        # Get node_id from announcer if not provided
        if node_id is None:
            from .announcer import _get_hostname
            node_id = f"atlas-vision-{_get_hostname()}"

        self.node_id = node_id
        self.host = host or settings.mqtt.host
        self.port = port or settings.mqtt.port
        self.username = username or settings.mqtt.username
        self.password = password or settings.mqtt.password

        self._client = None
        self._connected = False
        self._publish_task: Optional[asyncio.Task] = None

        # Build topic names
        self.topic_events = TOPIC_EVENTS.format(node_id=self.node_id)
        self.topic_tracks = TOPIC_TRACKS.format(node_id=self.node_id)
        self.topic_status = TOPIC_STATUS.format(node_id=self.node_id)

        logger.info(
            "MQTTPublisher initialized: broker=%s:%d, node=%s",
            self.host, self.port, self.node_id
        )

    async def connect(self) -> bool:
        """
        Connect to the MQTT broker.

        Returns:
            True if connected successfully
        """
        if self._connected:
            return True

        try:
            import aiomqtt
        except ImportError:
            logger.error("aiomqtt not installed. Run: pip install aiomqtt")
            return False

        try:
            self._client = aiomqtt.Client(
                hostname=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
            )
            await self._client.__aenter__()
            self._connected = True

            # Publish online status
            await self._publish_status("online")

            logger.info("MQTT connected to %s:%d", self.host, self.port)
            return True

        except Exception as e:
            logger.error("Failed to connect to MQTT broker: %s", e)
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        if not self._connected:
            return

        try:
            # Publish offline status before disconnecting
            await self._publish_status("offline")

            if self._client:
                await self._client.__aexit__(None, None, None)

        except Exception as e:
            logger.warning("Error during MQTT disconnect: %s", e)
        finally:
            self._connected = False
            self._client = None
            logger.info("MQTT disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        return self._connected

    async def publish_event(self, event: DetectionEvent) -> bool:
        """
        Publish a detection event.

        Args:
            event: Detection event to publish

        Returns:
            True if published successfully
        """
        if not self._connected or not self._client:
            return False

        try:
            payload = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "track_id": event.track_id,
                "class": event.class_name,
                "source_id": event.source_id,
                "timestamp": event.timestamp.isoformat(),
                "node_id": self.node_id,
            }

            if event.bbox:
                payload["bbox"] = event.bbox.to_dict()

            if event.metadata:
                payload["metadata"] = event.metadata

            await self._client.publish(
                self.topic_events,
                json.dumps(payload),
            )

            logger.debug("Published event: %s", event.event_type.value)
            return True

        except Exception as e:
            logger.warning("Failed to publish event: %s", e)
            return False

    async def publish_tracks(self, tracks: list[Track]) -> bool:
        """
        Publish current active tracks.

        Args:
            tracks: List of active tracks

        Returns:
            True if published successfully
        """
        if not self._connected or not self._client:
            return False

        try:
            payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "node_id": self.node_id,
                "count": len(tracks),
                "tracks": [t.to_dict() for t in tracks],
            }

            await self._client.publish(
                self.topic_tracks,
                json.dumps(payload),
            )

            logger.debug("Published %d tracks", len(tracks))
            return True

        except Exception as e:
            logger.warning("Failed to publish tracks: %s", e)
            return False

    async def _publish_status(self, status: str) -> bool:
        """Publish node status."""
        if not self._client:
            return False

        try:
            payload = {
                "node_id": self.node_id,
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
            }

            await self._client.publish(
                self.topic_status,
                json.dumps(payload),
                retain=True,  # Retain status for new subscribers
            )

            logger.debug("Published status: %s", status)
            return True

        except Exception as e:
            logger.warning("Failed to publish status: %s", e)
            return False


# Global publisher instance
_mqtt_publisher: Optional[MQTTPublisher] = None


def get_mqtt_publisher() -> MQTTPublisher:
    """Get or create the global MQTT publisher."""
    global _mqtt_publisher
    if _mqtt_publisher is None:
        _mqtt_publisher = MQTTPublisher()
    return _mqtt_publisher


async def start_mqtt_publisher() -> bool:
    """Start the MQTT publisher (call from app startup)."""
    if not settings.mqtt.enabled:
        logger.info("MQTT publishing disabled")
        return False

    publisher = get_mqtt_publisher()
    return await publisher.connect()


async def stop_mqtt_publisher() -> None:
    """Stop the MQTT publisher (call from app shutdown)."""
    global _mqtt_publisher
    if _mqtt_publisher and _mqtt_publisher.is_connected:
        await _mqtt_publisher.disconnect()
