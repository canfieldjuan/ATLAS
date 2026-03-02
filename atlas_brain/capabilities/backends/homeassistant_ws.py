"""
Home Assistant WebSocket client for real-time state subscriptions.

Maintains persistent connection and handles reconnection automatically.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger("atlas.backends.homeassistant_ws")


class HomeAssistantWebSocket:
    """
    WebSocket client for Home Assistant real-time events.

    Features:
    - Automatic authentication
    - State change subscriptions
    - Auto-reconnection with exponential backoff
    - Message ID tracking
    """

    def __init__(
        self,
        url: str,
        access_token: str,
        on_state_changed: Optional[Callable[[dict], Any]] = None,
    ):
        # Convert http(s) to ws(s)
        ws_url = url.replace("http://", "ws://").replace("https://", "wss://")
        if not ws_url.endswith("/api/websocket"):
            ws_url = ws_url.rstrip("/") + "/api/websocket"
        self._ws_url = ws_url

        self._access_token = access_token
        self._on_state_changed = on_state_changed

        self._ws = None
        self._msg_id = 0
        self._subscription_id: Optional[int] = None
        self._connected = False
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected and authenticated."""
        return self._connected

    def _next_id(self) -> int:
        """Get next message ID for HA protocol."""
        self._msg_id += 1
        return self._msg_id

    async def connect(self) -> None:
        """Connect and authenticate to Home Assistant WebSocket."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets not installed. Run: pip install websockets")
            raise RuntimeError("websockets package required for HA WebSocket")

        logger.info("Connecting to HA WebSocket: %s", self._ws_url)

        try:
            self._ws = await websockets.connect(
                self._ws_url,
                open_timeout=15,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=5,
            )

            # Wait for auth_required message
            raw_msg = await asyncio.wait_for(self._ws.recv(), timeout=10)
            msg = json.loads(raw_msg)

            if msg.get("type") != "auth_required":
                raise RuntimeError(f"Expected auth_required, got: {msg.get('type')}")

            logger.debug("HA requires auth (version: %s)", msg.get("ha_version"))

            # Send authentication
            await self._ws.send(json.dumps({
                "type": "auth",
                "access_token": self._access_token,
            }))

            # Wait for auth response
            raw_msg = await asyncio.wait_for(self._ws.recv(), timeout=10)
            msg = json.loads(raw_msg)

            if msg.get("type") == "auth_invalid":
                raise RuntimeError(f"Authentication failed: {msg.get('message')}")

            if msg.get("type") != "auth_ok":
                raise RuntimeError(f"Expected auth_ok, got: {msg.get('type')}")

            self._connected = True
            logger.info(
                "HA WebSocket authenticated (HA version: %s)",
                msg.get("ha_version", "unknown")
            )

        except asyncio.TimeoutError:
            logger.error("HA WebSocket connection timed out")
            await self._cleanup_connection()
            raise RuntimeError("Connection timeout")
        except Exception as e:
            logger.error("HA WebSocket connection failed: %s", e)
            await self._cleanup_connection()
            raise

    async def _cleanup_connection(self) -> None:
        """Clean up WebSocket connection."""
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def subscribe_state_changes(self) -> int:
        """Subscribe to state_changed events. Returns subscription ID."""
        if not self._connected or not self._ws:
            raise RuntimeError("WebSocket not connected")

        msg_id = self._next_id()
        await self._ws.send(json.dumps({
            "id": msg_id,
            "type": "subscribe_events",
            "event_type": "state_changed",
        }))

        # Wait for subscription confirmation
        raw_msg = await asyncio.wait_for(self._ws.recv(), timeout=10)
        response = json.loads(raw_msg)

        if response.get("id") != msg_id:
            # Might be an event, keep reading
            logger.debug("Got message while waiting for sub confirm: %s", response.get("type"))

        if not response.get("success", False) and response.get("type") == "result":
            raise RuntimeError(f"Subscription failed: {response}")

        self._subscription_id = msg_id
        logger.info("Subscribed to state_changed events (subscription_id=%d)", msg_id)
        return msg_id

    async def start(self) -> None:
        """Start the WebSocket client with auto-reconnection."""
        self._running = True

        await self.connect()
        await self.subscribe_state_changes()

        # Start receive loop in background
        self._receive_task = asyncio.create_task(self._receive_loop())
        logger.info("HA WebSocket receive loop started")

    async def _receive_loop(self) -> None:
        """Main receive loop for WebSocket messages with auto-reconnect."""
        reconnect_delay = 1
        max_delay = 60

        while self._running:
            try:
                # Reconnect if not connected
                if not self._connected:
                    logger.info("Reconnecting to HA WebSocket in %ds...", reconnect_delay)
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, max_delay)

                    try:
                        await self.connect()
                        await self.subscribe_state_changes()
                        reconnect_delay = 1  # Reset on success
                        logger.info("HA WebSocket reconnected successfully")
                    except Exception as e:
                        logger.warning("Reconnection failed: %s", e)
                        continue

                # Receive message
                try:
                    raw_msg = await self._ws.recv()
                    msg = json.loads(raw_msg)
                    await self._handle_message(msg)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning("WebSocket receive error: %s", e)
                    self._connected = False
                    await self._cleanup_connection()

            except asyncio.CancelledError:
                logger.debug("Receive loop cancelled")
                break
            except Exception as e:
                logger.error("Unexpected error in receive loop: %s", e)
                self._connected = False
                await self._cleanup_connection()

    async def _send_to_alerts(self, event_data: dict[str, Any]) -> None:
        """Send state change to centralized alert system."""
        try:
            from ...alerts import HAStateAlertEvent, get_alert_manager

            alert_event = HAStateAlertEvent.from_ha_event(event_data)
            manager = get_alert_manager()
            await manager.process_event(alert_event)
        except Exception as e:
            logger.debug("Failed to send HA event to alerts: %s", e)

    def _update_context_device(self, event_data: dict[str, Any]) -> None:
        """Feed HA device state into ContextAggregator."""
        try:
            from ...orchestration.context import get_context

            new_state = event_data.get("new_state")
            if not new_state:
                return

            entity_id = event_data.get("entity_id", "")
            attrs = new_state.get("attributes", {})
            friendly_name = attrs.get("friendly_name", entity_id)
            state_val = new_state.get("state", "unknown")

            ctx = get_context()
            ctx.update_device(
                device_id=entity_id,
                name=friendly_name,
                state={"state": state_val},
            )
        except Exception as e:
            logger.debug("Could not update context from HA event: %s", e)

    async def _handle_message(self, msg: dict[str, Any]) -> None:
        """Handle incoming WebSocket message."""
        msg_type = msg.get("type")

        if msg_type == "event":
            event = msg.get("event", {})
            event_type = event.get("event_type")

            if event_type == "state_changed":
                event_data = event.get("data", {})
                entity_id = event_data.get("entity_id", "unknown")

                # Feed device state into ContextAggregator for LLM awareness
                self._update_context_device(event_data)

                # Send to centralized alert system
                await self._send_to_alerts(event_data)

                # Call user handler if set (may be sync or async)
                if self._on_state_changed:
                    try:
                        result = self._on_state_changed(event_data)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.warning("State change handler error for %s: %s", entity_id, e)

        elif msg_type == "result":
            # Command response - log failures
            if not msg.get("success"):
                logger.warning("HA command failed (id=%s): %s", msg.get("id"), msg.get("error"))

        elif msg_type == "pong":
            # Keepalive response
            pass

        else:
            logger.debug("Unhandled message type: %s", msg_type)

    async def stop(self) -> None:
        """Stop the WebSocket client gracefully."""
        logger.info("Stopping HA WebSocket client...")
        self._running = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        await self._cleanup_connection()
        logger.info("HA WebSocket client stopped")


# Module-level instance management
_ws_client: Optional[HomeAssistantWebSocket] = None


def get_ws_client() -> Optional[HomeAssistantWebSocket]:
    """Get the WebSocket client instance (may be None if not initialized)."""
    return _ws_client


async def init_ws_client(
    url: str,
    token: str,
    on_state_changed: Callable[[dict], Any],
) -> HomeAssistantWebSocket:
    """Initialize and start the WebSocket client."""
    global _ws_client

    if _ws_client and _ws_client.is_connected:
        logger.warning("WebSocket client already running, stopping first")
        await shutdown_ws_client()

    _ws_client = HomeAssistantWebSocket(url, token, on_state_changed)
    await _ws_client.start()
    return _ws_client


async def shutdown_ws_client() -> None:
    """Stop the WebSocket client."""
    global _ws_client

    if _ws_client:
        await _ws_client.stop()
        _ws_client = None
