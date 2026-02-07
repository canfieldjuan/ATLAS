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
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger("atlas.api.edge.websocket")

router = APIRouter(prefix="/ws/edge", tags=["edge"])


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

            elif msg_type == "vision":
                # Handle vision detections from edge node
                await _handle_vision_event(connection, message)

            elif msg_type == "transcript":
                # Handle speech transcript from edge node
                await _handle_transcript(connection, message)

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


async def _handle_vision_event(
    connection: EdgeConnection,
    message: dict[str, Any],
) -> None:
    """Handle vision detections from an edge node."""
    detections = message.get("detections", [])
    node_id = message.get("node_id", connection.location_id)
    frame_shape = message.get("frame_shape", [480, 640])
    ts = message.get("ts", time.time())
    frame_h, frame_w = frame_shape[0], frame_shape[1]

    logger.debug(
        "Vision event from %s: %d detections",
        connection.location_id,
        len(detections),
    )

    for det in detections:
        try:
            from ...vision.models import BoundingBox, EventType, VisionEvent
            from ...storage.models import VisionEventRecord
            from ...storage.repositories import get_vision_event_repo
            from ...alerts import VisionAlertEvent, get_alert_manager

            bbox_raw = det.get("bbox", [0, 0, 0, 0])
            # Normalize pixel coords to 0-1 range
            bbox = BoundingBox(
                x1=bbox_raw[0] / frame_w,
                y1=bbox_raw[1] / frame_h,
                x2=bbox_raw[2] / frame_w,
                y2=bbox_raw[3] / frame_h,
            )

            event_id = f"{node_id}-{uuid4().hex[:8]}"
            event = VisionEvent(
                event_id=event_id,
                event_type=EventType.TRACK_UPDATE,
                track_id=0,
                class_name=det.get("label", "unknown"),
                source_id=f"{node_id}/camera",
                node_id=node_id,
                timestamp=datetime.fromtimestamp(ts),
                bbox=bbox,
                metadata={"confidence": det.get("confidence", 0)},
            )

            # Persist to DB
            record = VisionEventRecord(
                id=uuid4(),
                event_id=event.event_id,
                event_type=event.event_type.value,
                track_id=event.track_id,
                class_name=event.class_name,
                source_id=event.source_id,
                node_id=event.node_id,
                bbox_x1=bbox.x1,
                bbox_y1=bbox.y1,
                bbox_x2=bbox.x2,
                bbox_y2=bbox.y2,
                event_timestamp=event.timestamp,
                received_at=datetime.utcnow(),
                metadata=event.metadata,
            )
            repo = get_vision_event_repo()
            await repo.save_event(record)

            # Process alerts
            try:
                alert_event = VisionAlertEvent.from_vision_event(event)
                manager = get_alert_manager()
                await manager.process_event(alert_event)
            except Exception as e:
                logger.warning("Alert processing failed: %s", e)

        except Exception as e:
            logger.warning("Failed to process detection: %s", e)

    await connection.send({"type": "vision_ack", "count": len(detections)})


async def _handle_transcript(
    connection: EdgeConnection,
    message: dict[str, Any],
) -> None:
    """Handle speech transcript from an edge node."""
    text = message.get("text", "").strip()
    node_id = message.get("node_id", connection.location_id)

    if not text:
        return

    logger.info(
        "Transcript from %s: '%s'",
        connection.location_id,
        text[:80],
    )

    try:
        from ...agents.graphs import get_atlas_agent_langgraph

        session_id = f"edge-{node_id}"
        agent = get_atlas_agent_langgraph(session_id=session_id)
        result = await agent.run(
            input_text=text,
            session_id=session_id,
            speaker_id=node_id,
            runtime_context={"source": "edge_stt", "node_id": node_id},
        )

        await connection.send({
            "type": "response",
            "success": result.get("success", False),
            "response": result.get("response_text", ""),
            "action_type": result.get("action_type", "conversation"),
            "metadata": {
                "timing": result.get("timing", {}),
                "node_id": node_id,
            },
        })

    except Exception as e:
        logger.exception("Transcript processing failed: %s", e)
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
