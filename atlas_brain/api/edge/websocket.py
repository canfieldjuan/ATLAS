"""
WebSocket endpoint for edge device connectivity.

Provides:
- Query escalation from edge devices
- Streaming response support
- Health monitoring
"""

import asyncio
import json
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger("atlas.api.edge.websocket")

router = APIRouter(prefix="/api/v1/ws/edge", tags=["edge"])


class EdgeConnection:
    """Represents a connected edge device."""

    def __init__(
        self,
        websocket: WebSocket,
        location_id: str,
    ):
        self.websocket = websocket
        self.location_id = location_id
        self.connected_at = time.time()
        self.last_message = time.time()
        self.message_count = 0

    async def send(self, message: dict[str, Any]) -> None:
        """Send message to edge device."""
        await self.websocket.send_json(message)
        self.last_message = time.time()

    async def send_token(self, token: str) -> None:
        """Send streaming token."""
        await self.send({"type": "token", "token": token})

    async def send_complete(self, metadata: Optional[dict] = None) -> None:
        """Send stream complete message."""
        await self.send({"type": "complete", "metadata": metadata or {}})

    async def send_error(self, error: str) -> None:
        """Send error message."""
        await self.send({"type": "error", "error": error})


# Track connected edge devices
_connections: dict[str, EdgeConnection] = {}


def get_connection(location_id: str) -> Optional[EdgeConnection]:
    """Get connection for a location."""
    return _connections.get(location_id)


def get_all_connections() -> dict[str, EdgeConnection]:
    """Get all active connections."""
    return _connections.copy()


@router.websocket("/{location_id}")
async def edge_websocket(
    websocket: WebSocket,
    location_id: str,
):
    """
    WebSocket endpoint for edge device connectivity.

    Edge devices connect here to:
    - Escalate queries that can't be handled locally
    - Receive streaming responses
    - Report health status

    Message Types (from edge):
    - query: Escalate a query for processing
    - query_stream: Escalate with streaming response
    - health: Health check ping

    Message Types (to edge):
    - response: Query response
    - token: Streaming token
    - complete: Stream complete
    - error: Error message
    """
    await websocket.accept()
    connection = EdgeConnection(websocket, location_id)
    _connections[location_id] = connection

    logger.info("Edge device connected: %s", location_id)

    try:
        while True:
            # Receive message
            try:
                message = await websocket.receive_json()
            except json.JSONDecodeError:
                await connection.send_error("Invalid JSON")
                continue

            connection.message_count += 1
            connection.last_message = time.time()

            msg_type = message.get("type", "")

            if msg_type == "query":
                # Handle query escalation
                await _handle_query(connection, message)

            elif msg_type == "query_stream":
                # Handle streaming query
                await _handle_streaming_query(connection, message)

            elif msg_type == "health":
                # Health check response
                await connection.send({"type": "health_ack", "timestamp": time.time()})

            else:
                await connection.send_error(f"Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        logger.info("Edge device disconnected: %s", location_id)

    except Exception as e:
        logger.exception("Edge WebSocket error for %s: %s", location_id, e)

    finally:
        # Clean up connection
        if location_id in _connections:
            del _connections[location_id]


async def _handle_query(
    connection: EdgeConnection,
    message: dict[str, Any],
) -> None:
    """Handle query escalation from edge device."""
    query = message.get("query", "")
    session_id = message.get("session_id")
    speaker_id = message.get("speaker_id")
    context = message.get("context", {})

    if not query:
        await connection.send_error("Missing query")
        return

    logger.info(
        "Query from %s: '%s'",
        connection.location_id,
        query[:50],
    )

    try:
        # Use the AtlasAgent graph for processing
        from ...agents.graphs import get_atlas_agent_langgraph

        agent = get_atlas_agent_langgraph(session_id=session_id)
        result = await agent.run(
            input_text=query,
            session_id=session_id,
            speaker_id=speaker_id,
            runtime_context=context,
        )

        # Send response
        await connection.send({
            "type": "response",
            "success": result.get("success", False),
            "response": result.get("response_text", ""),
            "action_type": result.get("action_type", "conversation"),
            "metadata": {
                "timing": result.get("timing", {}),
                "location_id": connection.location_id,
            },
        })

    except Exception as e:
        logger.exception("Query processing failed: %s", e)
        await connection.send_error(str(e))


async def _handle_streaming_query(
    connection: EdgeConnection,
    message: dict[str, Any],
) -> None:
    """Handle streaming query from edge device."""
    query = message.get("query", "")
    session_id = message.get("session_id")
    speaker_id = message.get("speaker_id")
    context = message.get("context", {})

    if not query:
        await connection.send_error("Missing query")
        return

    logger.info(
        "Streaming query from %s: '%s'",
        connection.location_id,
        query[:50],
    )

    try:
        # Use streaming agent
        from ...agents.graphs import get_streaming_atlas_agent

        agent = get_streaming_atlas_agent(session_id=session_id)

        # Stream tokens to edge device
        full_response = []
        async for token in agent.stream(
            input_text=query,
            session_id=session_id,
            speaker_id=speaker_id,
        ):
            full_response.append(token)
            await connection.send_token(token)

        # Send completion
        await connection.send_complete({
            "full_response": "".join(full_response),
            "location_id": connection.location_id,
        })

    except Exception as e:
        logger.exception("Streaming query failed: %s", e)
        await connection.send_error(str(e))


# HTTP endpoint for edge device status


@router.get("/status")
async def get_edge_status():
    """Get status of all connected edge devices."""
    return {
        "connected_devices": len(_connections),
        "devices": [
            {
                "location_id": conn.location_id,
                "connected_at": conn.connected_at,
                "last_message": conn.last_message,
                "message_count": conn.message_count,
            }
            for conn in _connections.values()
        ],
    }
