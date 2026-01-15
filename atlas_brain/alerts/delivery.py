"""
Alert delivery mechanisms.

Provides callbacks for delivering alerts via TTS, logging, and other channels.
"""

import logging
from typing import Any, Optional

from .events import AlertEvent
from .rules import AlertRule

logger = logging.getLogger("atlas.alerts.delivery")


async def log_alert_callback(
    message: str,
    rule: AlertRule,
    event: AlertEvent,
) -> None:
    """Default callback that logs alerts."""
    logger.info(
        "ALERT [%s/%s]: %s (source=%s)",
        rule.name,
        event.event_type,
        message,
        event.source_id,
    )


class TTSDelivery:
    """Delivers alerts via text-to-speech to connected voice clients."""

    def __init__(self, tts_registry: Any, connection_manager: Any):
        """
        Initialize TTS delivery.

        Args:
            tts_registry: The TTS service registry for synthesizing speech
            connection_manager: WebSocket connection manager for delivery
        """
        self._tts_registry = tts_registry
        self._connection_manager = connection_manager

    async def deliver(
        self,
        message: str,
        rule: AlertRule,
        event: AlertEvent,
    ) -> None:
        """Deliver alert via TTS to connected voice clients."""
        import base64

        try:
            active_tts = self._tts_registry.get_active()
            if not active_tts:
                logger.debug("TTS not available for alert delivery")
                return

            audio_bytes = await active_tts.synthesize(message)
            logger.info(
                "TTS generated %d bytes for alert [%s]",
                len(audio_bytes),
                rule.name,
            )

            audio_base64 = base64.b64encode(audio_bytes).decode()
            delivered = await self._connection_manager.queue_announcement({
                "state": "alert",
                "rule": rule.name,
                "event_type": event.event_type,
                "text": message,
                "audio_base64": audio_base64,
            })

            if delivered:
                logger.info("Alert delivered to connected clients")
            else:
                logger.debug("Alert queued (no clients connected)")

        except Exception as e:
            logger.warning("TTS delivery failed: %s", e)


class NtfyDelivery:
    """Delivers alerts via ntfy push notification server."""

    def __init__(self, base_url: str = "http://localhost:8090", topic: str = "atlas-alerts"):
        """
        Initialize ntfy delivery.

        Args:
            base_url: ntfy server URL
            topic: Topic to publish to
        """
        self._url = f"{base_url.rstrip('/')}/{topic}"

    async def deliver(
        self,
        message: str,
        rule: AlertRule,
        event: AlertEvent,
    ) -> None:
        """Deliver alert via ntfy."""
        try:
            import httpx

            # Map rule priority to ntfy priority (1-5, 5=max)
            priority = "default"
            if rule.priority >= 15:
                priority = "urgent"
            elif rule.priority >= 10:
                priority = "high"
            elif rule.priority <= 3:
                priority = "low"

            headers = {
                "Title": f"Atlas: {rule.name}",
                "Priority": priority,
                "Tags": event.event_type,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._url,
                    content=message,
                    headers=headers,
                    timeout=10.0,
                )
                response.raise_for_status()
                logger.info("ntfy notification sent for [%s]", rule.name)

        except ImportError:
            logger.warning("httpx not installed, ntfy delivery unavailable")
        except Exception as e:
            logger.warning("ntfy delivery failed: %s", e)


class WebhookDelivery:
    """Delivers alerts via HTTP webhook."""

    def __init__(self, webhook_url: str, headers: Optional[dict[str, str]] = None):
        """
        Initialize webhook delivery.

        Args:
            webhook_url: URL to POST alerts to
            headers: Optional HTTP headers
        """
        self._webhook_url = webhook_url
        self._headers = headers or {}

    async def deliver(
        self,
        message: str,
        rule: AlertRule,
        event: AlertEvent,
    ) -> None:
        """Deliver alert via webhook."""
        try:
            import httpx

            payload = {
                "rule_name": rule.name,
                "event_type": event.event_type,
                "message": message,
                "source_id": event.source_id,
                "timestamp": event.timestamp.isoformat(),
                "priority": rule.priority,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._webhook_url,
                    json=payload,
                    headers=self._headers,
                    timeout=10.0,
                )
                response.raise_for_status()
                logger.debug("Webhook delivered for alert [%s]", rule.name)

        except ImportError:
            logger.warning("httpx not installed, webhook delivery unavailable")
        except Exception as e:
            logger.warning("Webhook delivery failed: %s", e)


def setup_default_callbacks(manager: Any) -> None:
    """Set up default alert callbacks on the manager."""
    manager.register_callback(log_alert_callback)
