"""
MQTT subscriber for vision events from atlas_vision nodes.

Subscribes to detection events and node status updates,
dispatching them to registered handlers.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Awaitable, Callable, Optional
from uuid import uuid4

from ..config import settings
from ..storage.models import VisionEventRecord
from ..storage.repositories import get_vision_event_repo
from .models import EventType, NodeStatus, VisionEvent

logger = logging.getLogger("atlas.brain.vision.subscriber")

# Topic patterns for subscription
# + is a wildcard that matches any single level (node_id)
TOPIC_EVENTS = "atlas/vision/+/events"
TOPIC_STATUS = "atlas/vision/+/status"
TOPIC_TRACKS = "atlas/vision/+/tracks"


# Callback type aliases
EventCallback = Callable[[VisionEvent], Awaitable[None]]
StatusCallback = Callable[[NodeStatus], Awaitable[None]]


class VisionSubscriber:
    """
    MQTT subscriber for atlas_vision events.

    Subscribes to:
    - atlas/vision/+/events - Detection events (new_track, track_lost)
    - atlas/vision/+/status - Node status (online, offline)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize the vision subscriber.

        Args:
            host: MQTT broker host (default from settings)
            port: MQTT broker port (default from settings)
            username: MQTT username (optional)
            password: MQTT password (optional)
        """
        self.host = host or settings.mqtt.host
        self.port = port or settings.mqtt.port
        self.username = username or settings.mqtt.username
        self.password = password or settings.mqtt.password

        self._client = None
        self._running = False
        self._listen_task: Optional[asyncio.Task] = None

        # Event callbacks
        self._event_callbacks: list[EventCallback] = []
        self._status_callbacks: list[StatusCallback] = []

        # Track known nodes
        self._known_nodes: dict[str, NodeStatus] = {}

        logger.info(
            "VisionSubscriber initialized: broker=%s:%d",
            self.host, self.port
        )

    def register_event_callback(self, callback: EventCallback) -> None:
        """Register a callback for vision events."""
        self._event_callbacks.append(callback)
        logger.debug("Registered event callback: %s", callback.__name__)

    def register_status_callback(self, callback: StatusCallback) -> None:
        """Register a callback for node status updates."""
        self._status_callbacks.append(callback)
        logger.debug("Registered status callback: %s", callback.__name__)

    @property
    def is_running(self) -> bool:
        """Check if subscriber is running."""
        return self._running

    @property
    def known_nodes(self) -> dict[str, NodeStatus]:
        """Get currently known vision nodes."""
        return self._known_nodes.copy()

    async def start(self) -> bool:
        """
        Start the subscriber and connect to MQTT broker.

        Returns:
            True if started successfully
        """
        if self._running:
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

            # Subscribe to topics
            await self._client.subscribe(TOPIC_EVENTS)
            await self._client.subscribe(TOPIC_STATUS)
            logger.info("Subscribed to vision topics")

            # Start listening task
            self._running = True
            self._listen_task = asyncio.create_task(self._listen_loop())

            logger.info("VisionSubscriber started, connected to %s:%d", self.host, self.port)
            return True

        except Exception as e:
            logger.error("Failed to start VisionSubscriber: %s", e)
            self._running = False
            return False

    async def stop(self) -> None:
        """Stop the subscriber and disconnect."""
        if not self._running:
            return

        self._running = False

        # Cancel listen task
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        # Disconnect client
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("Error disconnecting MQTT client: %s", e)
            self._client = None

        logger.info("VisionSubscriber stopped")

    async def _listen_loop(self) -> None:
        """Main loop to listen for MQTT messages."""
        try:
            async for message in self._client.messages:
                if not self._running:
                    break

                try:
                    await self._handle_message(message)
                except Exception as e:
                    logger.warning("Error handling message on %s: %s", message.topic, e)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Listen loop error: %s", e)
            self._running = False

    async def _handle_message(self, message) -> None:
        """Handle an incoming MQTT message."""
        topic = str(message.topic)
        payload = json.loads(message.payload.decode())

        # Determine message type from topic
        if "/events" in topic:
            await self._handle_event(payload)
        elif "/status" in topic:
            await self._handle_status(payload)
        else:
            logger.debug("Unknown topic: %s", topic)

    async def _handle_event(self, payload: dict) -> None:
        """Handle a detection event."""
        try:
            event = VisionEvent.from_mqtt_payload(payload)
            logger.debug(
                "Received event: %s track=%d class=%s from %s",
                event.event_type.value, event.track_id, event.class_name, event.source_id
            )

            # Store event in database
            await self._store_event(event)

            # Process through alert rules
            await self._process_alerts(event)

            # Dispatch to callbacks
            for callback in self._event_callbacks:
                try:
                    await callback(event)
                except Exception as e:
                    logger.warning("Event callback error: %s", e)

        except Exception as e:
            logger.warning("Failed to parse event: %s", e)

    async def _process_alerts(self, event: VisionEvent) -> None:
        """Process event through centralized alert system."""
        try:
            from ..alerts import VisionAlertEvent, get_alert_manager

            alert_event = VisionAlertEvent.from_vision_event(event)
            manager = get_alert_manager()
            await manager.process_event(alert_event)
        except Exception as e:
            logger.warning("Failed to process alerts: %s", e)

    async def _store_event(self, event: VisionEvent) -> None:
        """Store a vision event in the database."""
        try:
            # Convert VisionEvent to VisionEventRecord
            record = VisionEventRecord(
                id=uuid4(),
                event_id=event.event_id,
                event_type=event.event_type.value,
                track_id=event.track_id,
                class_name=event.class_name,
                source_id=event.source_id,
                node_id=event.node_id,
                bbox_x1=event.bbox.x1 if event.bbox else None,
                bbox_y1=event.bbox.y1 if event.bbox else None,
                bbox_x2=event.bbox.x2 if event.bbox else None,
                bbox_y2=event.bbox.y2 if event.bbox else None,
                event_timestamp=event.timestamp,
                received_at=datetime.utcnow(),
                metadata=event.metadata,
            )

            repo = get_vision_event_repo()
            await repo.save_event(record)

        except Exception as e:
            logger.warning("Failed to store event in database: %s", e)

    async def _handle_status(self, payload: dict) -> None:
        """Handle a node status update."""
        try:
            status = NodeStatus.from_mqtt_payload(payload)

            # Update known nodes
            self._known_nodes[status.node_id] = status

            logger.info(
                "Vision node %s is now %s",
                status.node_id, status.status
            )

            # Dispatch to callbacks
            for callback in self._status_callbacks:
                try:
                    await callback(status)
                except Exception as e:
                    logger.warning("Status callback error: %s", e)

        except Exception as e:
            logger.warning("Failed to parse status: %s", e)


# Global subscriber instance
_vision_subscriber: Optional[VisionSubscriber] = None


def get_vision_subscriber() -> VisionSubscriber:
    """Get or create the global vision subscriber."""
    global _vision_subscriber
    if _vision_subscriber is None:
        _vision_subscriber = VisionSubscriber()
    return _vision_subscriber


async def start_vision_subscriber() -> bool:
    """Start the vision subscriber (call from app startup)."""
    if not settings.mqtt.enabled:
        logger.info("MQTT disabled, vision subscriber not started")
        return False

    subscriber = get_vision_subscriber()
    return await subscriber.start()


async def stop_vision_subscriber() -> None:
    """Stop the vision subscriber (call from app shutdown)."""
    global _vision_subscriber
    if _vision_subscriber and _vision_subscriber.is_running:
        await _vision_subscriber.stop()
