"""
Security event consumer for Atlas Brain.

Subscribes to Kafka topics from video processing system and
triggers proactive announcements through TTS.
"""

import asyncio
import json
import logging
from typing import Callable, Optional

logger = logging.getLogger("atlas.services.security_events")

# Try to import aiokafka, but don't fail if not installed
try:
    from aiokafka import AIOKafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.warning("aiokafka not installed - security events disabled")


class SecurityEventConsumer:
    """
    Consumes security events from Kafka and triggers announcements.

    Events from video processing:
    - person_detected
    - motion_detected
    - unknown_face
    - vehicle_detected
    - package_detected
    - camera_offline
    """

    # Kafka topics to subscribe to
    TOPICS = [
        "security.person_detected",
        "security.motion_detected",
        "security.alerts",
        "security.camera_status",
    ]

    def __init__(
        self,
        kafka_bootstrap: str = "localhost:9093",
        group_id: str = "atlas-brain",
    ):
        self.kafka_bootstrap = kafka_bootstrap
        self.group_id = group_id
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._running = False
        self._announce_callback: Optional[Callable[[str], asyncio.Future]] = None

        # Import alert handler
        from ..tools.security import alert_handler
        self.alert_handler = alert_handler

    def set_announce_callback(self, callback: Callable[[str], asyncio.Future]):
        """
        Set callback for announcements.

        The callback should accept text and speak it via TTS.
        Example: orchestrator.announce(text)
        """
        self._announce_callback = callback

    async def start(self):
        """Start consuming events."""
        if not KAFKA_AVAILABLE:
            logger.warning("Kafka not available, security events disabled")
            return

        if self._running:
            return

        try:
            self._consumer = AIOKafkaConsumer(
                *self.TOPICS,
                bootstrap_servers=self.kafka_bootstrap,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",  # Only new events
            )
            await self._consumer.start()
            self._running = True
            logger.info("Security event consumer started (topics: %s)", self.TOPICS)

            # Start consuming in background
            asyncio.create_task(self._consume_loop())

        except Exception as e:
            logger.error("Failed to start security event consumer: %s", e)
            self._running = False

    async def stop(self):
        """Stop consuming events."""
        self._running = False
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None
            logger.info("Security event consumer stopped")

    async def _consume_loop(self):
        """Main consumption loop."""
        try:
            async for msg in self._consumer:
                if not self._running:
                    break

                try:
                    await self._handle_event(msg.topic, msg.value)
                except Exception as e:
                    logger.error("Error handling event: %s", e)

        except Exception as e:
            logger.error("Consumer loop error: %s", e)
        finally:
            self._running = False

    async def _handle_event(self, topic: str, event: dict):
        """Handle a single security event."""
        logger.debug("Security event: %s -> %s", topic, event)

        # Process through centralized alert system
        try:
            from ..alerts import SecurityAlertEvent, get_alert_manager

            alert_event = SecurityAlertEvent.from_kafka_event(event)
            manager = get_alert_manager()
            message = await manager.process_event(alert_event)

            if message and self._announce_callback:
                await self._announce_callback(message)
        except Exception as e:
            logger.warning("Failed to process security alert: %s", e)

            # Fallback to legacy alert handler
            announcement = await self.alert_handler.handle_event(event)
            if announcement and self._announce_callback:
                try:
                    await self._announce_callback(announcement)
                except Exception as e2:
                    logger.error("Announcement failed: %s", e2)


# Global consumer instance
_consumer: Optional[SecurityEventConsumer] = None


def get_security_consumer() -> SecurityEventConsumer:
    """Get or create security event consumer."""
    global _consumer
    if _consumer is None:
        _consumer = SecurityEventConsumer()
    return _consumer


async def init_security_events(announce_callback: Callable[[str], asyncio.Future]):
    """
    Initialize security event consumer.

    Call this from main.py during startup:

        from .services.security_events import init_security_events

        async def announce(text: str):
            tts = tts_registry.get_active()
            audio = await tts.synthesize(text)
            # Play audio...

        await init_security_events(announce)
    """
    consumer = get_security_consumer()
    consumer.set_announce_callback(announce_callback)
    await consumer.start()


async def shutdown_security_events():
    """Shutdown security event consumer."""
    global _consumer
    if _consumer:
        await _consumer.stop()
        _consumer = None
