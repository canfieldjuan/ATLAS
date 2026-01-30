"""
Brain Connection Manager for edge devices.

Handles WebSocket connection to brain server with:
- Automatic reconnection with exponential backoff
- Health checks
- Offline detection
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Optional

logger = logging.getLogger("atlas.edge.brain.connection")


class ConnectionState(Enum):
    """Brain connection state."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()


@dataclass
class BrainHealth:
    """Brain server health status."""

    is_healthy: bool
    latency_ms: float = 0.0
    last_check: float = 0.0
    error: Optional[str] = None


class BrainConnectionManager:
    """
    Manages WebSocket connection to brain server.

    Features:
    - Automatic reconnection with exponential backoff
    - Health monitoring
    - Message queuing during disconnection
    - Callbacks for state changes
    """

    def __init__(
        self,
        brain_url: str,
        location_id: str = "default",
        reconnect_interval: int = 5,
        max_reconnect_interval: int = 60,
        health_check_interval: int = 30,
    ):
        """
        Initialize brain connection manager.

        Args:
            brain_url: Brain server WebSocket URL
            location_id: Edge device location identifier
            reconnect_interval: Base reconnection interval (seconds)
            max_reconnect_interval: Maximum reconnection interval (seconds)
            health_check_interval: Health check interval (seconds)
        """
        self._brain_url = brain_url.rstrip("/")
        self._location_id = location_id
        self._reconnect_interval = reconnect_interval
        self._max_reconnect_interval = max_reconnect_interval
        self._health_check_interval = health_check_interval

        self._ws = None
        self._state = ConnectionState.DISCONNECTED
        self._health = BrainHealth(is_healthy=False)
        self._reconnect_attempts = 0
        self._last_message_time = 0.0

        # Callbacks
        self._on_connected: Optional[Callable[[], None]] = None
        self._on_disconnected: Optional[Callable[[], None]] = None
        self._on_message: Optional[Callable[[dict], None]] = None

        # Background tasks
        self._reconnect_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None

        # Message queue for offline operation
        self._pending_messages: list[dict] = []

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected to brain."""
        return self._state == ConnectionState.CONNECTED

    @property
    def is_healthy(self) -> bool:
        """Check if brain is healthy."""
        return self._health.is_healthy

    @property
    def health(self) -> BrainHealth:
        """Get brain health status."""
        return self._health

    def on_connected(self, callback: Callable[[], None]) -> None:
        """Set callback for connection established."""
        self._on_connected = callback

    def on_disconnected(self, callback: Callable[[], None]) -> None:
        """Set callback for disconnection."""
        self._on_disconnected = callback

    def on_message(self, callback: Callable[[dict], None]) -> None:
        """Set callback for received messages."""
        self._on_message = callback

    async def connect(self) -> bool:
        """
        Connect to brain server.

        Returns:
            True if connected successfully
        """
        if self._state == ConnectionState.CONNECTED:
            return True

        self._state = ConnectionState.CONNECTING

        try:
            import websockets

            url = f"{self._brain_url}/api/v1/ws/edge/{self._location_id}"
            logger.info("Connecting to brain at %s", url)

            self._ws = await asyncio.wait_for(
                websockets.connect(url),
                timeout=10.0,
            )

            self._state = ConnectionState.CONNECTED
            self._reconnect_attempts = 0
            self._health = BrainHealth(is_healthy=True, last_check=time.time())

            logger.info("Connected to brain server")

            # Start background tasks
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._health_task = asyncio.create_task(self._health_check_loop())

            # Send any pending messages
            await self._flush_pending_messages()

            if self._on_connected:
                self._on_connected()

            return True

        except asyncio.TimeoutError:
            logger.warning("Connection to brain timed out")
            self._state = ConnectionState.DISCONNECTED
            self._health = BrainHealth(is_healthy=False, error="timeout")
            return False

        except Exception as e:
            logger.warning("Failed to connect to brain: %s", e)
            self._state = ConnectionState.DISCONNECTED
            self._health = BrainHealth(is_healthy=False, error=str(e))
            return False

    async def disconnect(self) -> None:
        """Disconnect from brain server."""
        # Cancel background tasks
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None

        # Close WebSocket
        if self._ws:
            await self._ws.close()
            self._ws = None

        self._state = ConnectionState.DISCONNECTED
        self._health = BrainHealth(is_healthy=False)

        logger.info("Disconnected from brain server")

        if self._on_disconnected:
            self._on_disconnected()

    async def send(self, message: dict[str, Any]) -> bool:
        """
        Send message to brain.

        Args:
            message: Message dict to send

        Returns:
            True if sent successfully
        """
        if not self._ws or self._state != ConnectionState.CONNECTED:
            # Queue message for later
            self._pending_messages.append(message)
            logger.debug("Queued message for later delivery")
            return False

        try:
            await self._ws.send(json.dumps(message))
            self._last_message_time = time.time()
            return True

        except Exception as e:
            logger.warning("Failed to send message: %s", e)
            self._pending_messages.append(message)
            await self._handle_disconnect()
            return False

    async def send_and_receive(
        self,
        message: dict[str, Any],
        timeout: float = 5.0,
    ) -> Optional[dict[str, Any]]:
        """
        Send message and wait for response.

        Args:
            message: Message to send
            timeout: Response timeout in seconds

        Returns:
            Response dict or None if failed
        """
        if not await self.send(message):
            return None

        try:
            response_text = await asyncio.wait_for(
                self._ws.recv(),
                timeout=timeout,
            )
            return json.loads(response_text)

        except asyncio.TimeoutError:
            logger.warning("Response timeout")
            return None

        except Exception as e:
            logger.warning("Failed to receive response: %s", e)
            return None

    async def _receive_loop(self) -> None:
        """Background task to receive messages."""
        while self._state == ConnectionState.CONNECTED:
            try:
                message_text = await self._ws.recv()
                message = json.loads(message_text)

                if self._on_message:
                    self._on_message(message)

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.warning("Receive error: %s", e)
                await self._handle_disconnect()
                break

    async def _health_check_loop(self) -> None:
        """Background task for health checks."""
        while self._state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(self._health_check_interval)

                # Send ping
                start = time.time()
                await self._ws.ping()
                latency = (time.time() - start) * 1000

                self._health = BrainHealth(
                    is_healthy=True,
                    latency_ms=latency,
                    last_check=time.time(),
                )

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.warning("Health check failed: %s", e)
                self._health = BrainHealth(
                    is_healthy=False,
                    last_check=time.time(),
                    error=str(e),
                )
                await self._handle_disconnect()
                break

    async def _handle_disconnect(self) -> None:
        """Handle unexpected disconnection."""
        if self._state == ConnectionState.RECONNECTING:
            return

        self._state = ConnectionState.RECONNECTING
        self._health = BrainHealth(is_healthy=False, error="disconnected")

        if self._on_disconnected:
            self._on_disconnected()

        # Start reconnection task
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Background task for reconnection."""
        while self._state == ConnectionState.RECONNECTING:
            self._reconnect_attempts += 1

            # Exponential backoff
            delay = min(
                self._reconnect_interval * (2 ** (self._reconnect_attempts - 1)),
                self._max_reconnect_interval,
            )

            logger.info(
                "Reconnecting in %ds (attempt %d)",
                delay,
                self._reconnect_attempts,
            )

            await asyncio.sleep(delay)

            if await self.connect():
                break

    async def _flush_pending_messages(self) -> None:
        """Send any pending messages."""
        if not self._pending_messages:
            return

        messages = self._pending_messages.copy()
        self._pending_messages.clear()

        for message in messages:
            try:
                await self._ws.send(json.dumps(message))
            except Exception as e:
                logger.warning("Failed to send pending message: %s", e)


# Singleton instance
_connection: Optional[BrainConnectionManager] = None


async def get_brain_connection() -> BrainConnectionManager:
    """Get or create global brain connection."""
    global _connection
    if _connection is None:
        from ..config import settings

        _connection = BrainConnectionManager(
            brain_url=settings.brain.url,
            location_id=settings.location_id,
            reconnect_interval=settings.brain.reconnect_interval,
            health_check_interval=settings.brain.health_check_interval,
        )

    return _connection
