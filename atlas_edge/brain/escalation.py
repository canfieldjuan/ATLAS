"""
Brain escalation protocol for edge devices.

Handles query escalation to brain server with:
- Streaming response support
- Timeout handling
- Offline fallback
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

from .connection import BrainConnectionManager, get_brain_connection
from ..responses.templates import OFFLINE_FALLBACK_MESSAGE

logger = logging.getLogger("atlas.edge.brain.escalation")


@dataclass
class EscalationResult:
    """Result from brain escalation."""

    success: bool
    response_text: str = ""
    action_type: str = "conversation"
    is_streaming: bool = False
    total_ms: float = 0.0
    was_offline: bool = False
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BrainEscalation:
    """
    Handles query escalation to brain server.

    When the edge device cannot handle a query locally
    (conversations, tool queries, low confidence), it
    escalates to the brain server for processing.
    """

    def __init__(
        self,
        connection: Optional[BrainConnectionManager] = None,
        timeout: float = 5.0,
    ):
        """
        Initialize brain escalation handler.

        Args:
            connection: Brain connection manager (uses global if None)
            timeout: Escalation timeout in seconds
        """
        self._connection = connection
        self._timeout = timeout

    async def _get_connection(self) -> BrainConnectionManager:
        """Get brain connection."""
        if self._connection is None:
            self._connection = await get_brain_connection()
        return self._connection

    async def escalate(
        self,
        query: str,
        session_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> EscalationResult:
        """
        Escalate a query to the brain server.

        Args:
            query: User query text
            session_id: Session identifier
            speaker_id: Speaker identifier
            context: Additional context

        Returns:
            EscalationResult with response
        """
        start_time = time.perf_counter()
        connection = await self._get_connection()

        # Check if brain is available
        if not connection.is_connected:
            # Try to connect
            if not await connection.connect():
                return EscalationResult(
                    success=False,
                    response_text=OFFLINE_FALLBACK_MESSAGE,
                    was_offline=True,
                    error="brain_offline",
                )

        # Build escalation message
        message = {
            "type": "query",
            "query": query,
            "session_id": session_id,
            "speaker_id": speaker_id,
            "context": context or {},
            "timestamp": time.time(),
        }

        try:
            # Send and wait for response
            response = await connection.send_and_receive(
                message,
                timeout=self._timeout,
            )

            if response is None:
                return EscalationResult(
                    success=False,
                    response_text=OFFLINE_FALLBACK_MESSAGE,
                    was_offline=True,
                    error="timeout",
                    total_ms=(time.perf_counter() - start_time) * 1000,
                )

            total_ms = (time.perf_counter() - start_time) * 1000

            return EscalationResult(
                success=response.get("success", False),
                response_text=response.get("response", ""),
                action_type=response.get("action_type", "conversation"),
                total_ms=total_ms,
                metadata=response.get("metadata", {}),
            )

        except asyncio.TimeoutError:
            return EscalationResult(
                success=False,
                response_text=OFFLINE_FALLBACK_MESSAGE,
                was_offline=True,
                error="timeout",
                total_ms=(time.perf_counter() - start_time) * 1000,
            )

        except Exception as e:
            logger.error("Escalation failed: %s", e)
            return EscalationResult(
                success=False,
                response_text=OFFLINE_FALLBACK_MESSAGE,
                was_offline=True,
                error=str(e),
                total_ms=(time.perf_counter() - start_time) * 1000,
            )

    async def escalate_streaming(
        self,
        query: str,
        session_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Escalate a query with streaming response.

        Yields response tokens as they arrive from the brain.

        Args:
            query: User query text
            session_id: Session identifier
            speaker_id: Speaker identifier
            context: Additional context

        Yields:
            Response tokens
        """
        connection = await self._get_connection()

        # Check if brain is available
        if not connection.is_connected:
            if not await connection.connect():
                yield OFFLINE_FALLBACK_MESSAGE
                return

        # Build streaming escalation message
        message = {
            "type": "query_stream",
            "query": query,
            "session_id": session_id,
            "speaker_id": speaker_id,
            "context": context or {},
            "timestamp": time.time(),
        }

        try:
            # Send message
            if not await connection.send(message):
                yield OFFLINE_FALLBACK_MESSAGE
                return

            # Receive streaming response
            while True:
                try:
                    response_text = await asyncio.wait_for(
                        connection._ws.recv(),
                        timeout=self._timeout,
                    )

                    response = json.loads(response_text)
                    msg_type = response.get("type", "")

                    if msg_type == "token":
                        token = response.get("token", "")
                        if token:
                            yield token

                    elif msg_type == "complete":
                        # Stream complete
                        break

                    elif msg_type == "error":
                        logger.error("Stream error: %s", response.get("error"))
                        break

                except asyncio.TimeoutError:
                    logger.warning("Stream timeout")
                    break

        except Exception as e:
            logger.error("Streaming escalation failed: %s", e)
            yield OFFLINE_FALLBACK_MESSAGE

    def is_brain_available(self) -> bool:
        """
        Check if brain is currently available.

        Returns:
            True if brain is connected and healthy
        """
        if self._connection is None:
            return False
        return self._connection.is_connected and self._connection.is_healthy


# Singleton instance
_escalation: Optional[BrainEscalation] = None


async def get_escalation() -> BrainEscalation:
    """Get or create global escalation handler."""
    global _escalation
    if _escalation is None:
        from ..config import settings

        _escalation = BrainEscalation(timeout=settings.brain.escalation_timeout)
    return _escalation
