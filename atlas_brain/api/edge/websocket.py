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

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ...agents.graphs import get_atlas_agent_langgraph, get_streaming_atlas_agent
from ...alerts import VisionAlertEvent, get_alert_manager
from ...config import settings
from ...storage.models import VisionEventRecord
from ...storage.repositories import get_vision_event_repo
from ...storage.repositories.identity import get_identity_repo
from ...storage.repositories.unified_alerts import get_unified_alert_repo
from ...storage.database import get_db_pool
from ...vision.models import BoundingBox, EventType, VisionEvent

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
        self._pending_tasks: set[asyncio.Task] = set()
        self._llm_semaphore = asyncio.Semaphore(settings.edge.max_concurrent_llm)
        self._send_lock = asyncio.Lock()

    async def send(self, message: dict[str, Any]) -> None:
        """Send message to edge device.

        Protected by a lock to prevent concurrent Starlette WebSocket
        state-machine violations when multiple background tasks send
        responses simultaneously.
        """
        async with self._send_lock:
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

    def _spawn_task(self, coro, *, name: str | None = None) -> asyncio.Task:
        """Spawn a background task tracked for cleanup on disconnect."""
        task = asyncio.create_task(coro, name=name)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task

    async def cancel_pending(self) -> None:
        """Cancel all pending tasks (called on disconnect)."""
        tasks = list(self._pending_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._pending_tasks.clear()


class TokenBatcher:
    """Batches streaming tokens and flushes every interval or when buffer is full.

    Reduces WebSocket frame overhead from ~100 frames/sec (per-token) to
    ~20 frames/sec (batched) while adding at most 50ms latency.
    """

    def __init__(self, connection: EdgeConnection, interval_ms: int = 50, max_size: int = 10):
        self._connection = connection
        self._interval = interval_ms / 1000.0
        self._max_size = max_size
        self._buffer: list[str] = []
        self._flush_task: asyncio.Task | None = None

    async def add(self, token: str) -> None:
        """Add a token to the buffer, flushing if full."""
        self._buffer.append(token)
        if len(self._buffer) >= self._max_size:
            await self._flush()
        elif self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._timed_flush())

    async def _timed_flush(self) -> None:
        """Flush after the configured interval."""
        await asyncio.sleep(self._interval)
        await self._flush()

    async def _flush(self) -> None:
        """Send buffered tokens as a single message."""
        if not self._buffer:
            return
        # Swap buffer so new tokens arriving during send go to a fresh list
        tokens = self._buffer
        self._buffer = []
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            self._flush_task = None
        try:
            await self._connection.send({"type": "tokens", "tokens": tokens})
        except Exception:
            # Restore unsent tokens (prepend before any new ones added during send)
            tokens.extend(self._buffer)
            self._buffer = tokens
            raise

    async def close(self) -> None:
        """Flush remaining tokens and cancel any pending timer."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except (asyncio.CancelledError, Exception):
                pass
            self._flush_task = None
        await self._flush()


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
    - token: Single streaming token (legacy)
    - tokens: Batched streaming tokens
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
                # LLM-bound: fire-and-forget task
                connection._spawn_task(
                    _guarded_handle_query(connection, message),
                    name=f"query-{connection.location_id}",
                )

            elif msg_type == "query_stream":
                # LLM-bound: fire-and-forget task
                connection._spawn_task(
                    _guarded_handle_streaming_query(connection, message),
                    name=f"stream-{connection.location_id}",
                )

            elif msg_type == "health":
                # Fast I/O: stay inline
                await connection.send({"type": "health_ack", "timestamp": time.time()})

            elif msg_type == "vision":
                # Fast I/O: stay inline
                await _handle_vision_event(connection, message)

            elif msg_type == "transcript":
                # LLM-bound: fire-and-forget task
                connection._spawn_task(
                    _guarded_handle_transcript(connection, message),
                    name=f"transcript-{connection.location_id}",
                )

            elif msg_type == "identity_sync_request":
                # Handle identity sync from edge node
                await _handle_identity_sync_request(connection, message)

            elif msg_type == "identity_register":
                # Handle new identity registration from edge node
                await _handle_identity_register(connection, message)

            elif msg_type == "security":
                # Handle security events from edge node
                await _handle_security_event(connection, message)

            elif msg_type == "recognition":
                # Handle recognition events from edge node (speaker, face, gait)
                await _handle_recognition_event(connection, message)

            else:
                await connection.send_error(f"Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        logger.info("Edge device disconnected: %s", location_id)

    except Exception as e:
        logger.exception("Edge WebSocket error for %s: %s", location_id, e)

    finally:
        # Cancel any pending LLM tasks before removing connection
        await connection.cancel_pending()
        if location_id in _connections:
            del _connections[location_id]


async def _guarded_handle_query(connection: EdgeConnection, message: dict[str, Any]) -> None:
    """Semaphore-guarded wrapper for _handle_query (runs as background task)."""
    try:
        async with connection._llm_semaphore:
            await _handle_query(connection, message)
    except asyncio.CancelledError:
        logger.debug("Query task cancelled for %s", connection.location_id)
    except Exception as e:
        logger.exception("Background query failed: %s", e)
        try:
            await connection.send_error(str(e))
        except Exception:
            pass


async def _guarded_handle_streaming_query(connection: EdgeConnection, message: dict[str, Any]) -> None:
    """Semaphore-guarded wrapper for _handle_streaming_query (runs as background task)."""
    try:
        async with connection._llm_semaphore:
            await _handle_streaming_query(connection, message)
    except asyncio.CancelledError:
        logger.debug("Streaming query task cancelled for %s", connection.location_id)
    except Exception as e:
        logger.exception("Background streaming query failed: %s", e)
        try:
            await connection.send_error(str(e))
        except Exception:
            pass


async def _guarded_handle_transcript(connection: EdgeConnection, message: dict[str, Any]) -> None:
    """Semaphore-guarded wrapper for _handle_transcript (runs as background task)."""
    try:
        async with connection._llm_semaphore:
            await _handle_transcript(connection, message)
    except asyncio.CancelledError:
        logger.debug("Transcript task cancelled for %s", connection.location_id)
    except Exception as e:
        logger.exception("Background transcript failed: %s", e)
        try:
            await connection.send_error(str(e))
        except Exception:
            pass


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

    batcher = TokenBatcher(
        connection,
        interval_ms=settings.edge.token_batch_interval_ms,
        max_size=settings.edge.token_batch_max_size,
    )
    try:
        # Use streaming agent
        agent = get_streaming_atlas_agent(session_id=session_id)

        # Stream tokens to edge device via batcher
        full_response = []
        async for token in agent.stream(
            input_text=query,
            session_id=session_id,
            speaker_id=speaker_id,
        ):
            full_response.append(token)
            await batcher.add(token)

        await batcher.close()

        # Send completion
        await connection.send_complete({
            "full_response": "".join(full_response),
            "location_id": connection.location_id,
        })

    except Exception as e:
        # Clean up batcher to cancel any pending _timed_flush task
        try:
            await batcher.close()
        except Exception:
            pass
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

    # Phase 1: Build records (no I/O)
    records: list[VisionEventRecord] = []
    events: list[VisionEvent] = []
    for det in detections:
        try:
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
            events.append(event)

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
            records.append(record)
        except Exception as e:
            logger.warning("Failed to build detection: %s", e)

    # Phase 2: Batch DB save (single transaction)
    if records:
        try:
            repo = get_vision_event_repo()
            await repo.save_events_batch(records)
        except Exception as e:
            logger.warning("Batch vision save failed: %s", e)

    # Phase 3: Concurrent alert processing
    if events:
        try:
            manager = get_alert_manager()
            coros = [
                manager.process_event(VisionAlertEvent.from_vision_event(e))
                for e in events
            ]
            await asyncio.gather(*coros, return_exceptions=True)
        except Exception as e:
            logger.warning("Alert processing failed: %s", e)

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


async def _handle_identity_sync_request(
    connection: EdgeConnection,
    message: dict[str, Any],
) -> None:
    """Handle identity sync request from an edge node.

    Edge sends its manifest of {modality: [names]}.
    We diff against the master DB and respond with missing embeddings + deletions.
    """
    edge_manifest = message.get("current", {})
    node_id = message.get("node_id", connection.location_id)

    logger.info(
        "Identity sync request from %s: %s",
        node_id,
        {mod: len(names) for mod, names in edge_manifest.items()},
    )

    try:
        repo = get_identity_repo()
        to_send, to_delete, need_from_edge = await repo.diff_manifest(edge_manifest)

        await connection.send({
            "type": "identity_sync",
            "identities": to_send,
            "delete": to_delete,
            "need_from_edge": need_from_edge,
        })

        sent_count = sum(len(v) for v in to_send.values())
        del_count = sum(len(v) for v in to_delete.values())
        need_count = sum(len(v) for v in need_from_edge.values())
        logger.info(
            "Identity sync response to %s: %d to send, %d to delete, %d needed from edge",
            node_id, sent_count, del_count, need_count,
        )

    except Exception as e:
        logger.exception("Identity sync failed for %s: %s", node_id, e)
        await connection.send_error(f"Identity sync failed: {e}")


async def _handle_identity_register(
    connection: EdgeConnection,
    message: dict[str, Any],
) -> None:
    """Handle a new identity registration from an edge node.

    Saves to master DB, then broadcasts to all OTHER connected edges.
    """
    name = message.get("name")
    modality = message.get("modality")
    embedding_list = message.get("embedding")
    node_id = message.get("node_id", connection.location_id)

    if not name or not modality or embedding_list is None:
        await connection.send_error("identity_register: missing name, modality, or embedding")
        return

    if modality not in ("face", "gait", "speaker"):
        await connection.send_error(f"identity_register: invalid modality '{modality}'")
        return

    logger.info(
        "Identity register from %s: %s/%s (dim=%d)",
        node_id, modality, name, len(embedding_list),
    )

    try:
        repo = get_identity_repo()
        embedding = np.array(embedding_list, dtype=np.float32)
        await repo.upsert(name, modality, embedding, source_node=node_id)

        # Broadcast to all OTHER connected edges
        update_msg = {
            "type": "identity_update",
            "name": name,
            "modality": modality,
            "embedding": embedding_list,
            "source_node": node_id,
        }
        for loc_id, conn in _connections.items():
            if loc_id != connection.location_id:
                try:
                    await conn.send(update_msg)
                    logger.debug("Broadcast identity_update %s/%s to %s", modality, name, loc_id)
                except Exception as e:
                    logger.warning("Failed to broadcast to %s: %s", loc_id, e)

    except Exception as e:
        logger.exception("Identity register failed: %s", e)
        await connection.send_error(f"Identity register failed: {e}")


async def _handle_security_event(
    connection: EdgeConnection,
    message: dict[str, Any],
) -> None:
    """Handle security events from an edge node.

    Event types: motion_detected, person_entered, person_left, unknown_face
    """
    event = message.get("event", "unknown")
    node_id = message.get("node_id", connection.location_id)
    ts = message.get("ts", time.time())

    logger.info(
        "Security event from %s: %s",
        node_id,
        event,
    )

    # Build human-readable message (always, even if DB save fails)
    if event == "person_entered":
        name = message.get("name", "unknown")
        is_known = message.get("is_known", False)
        confidence = message.get("combined_confidence", 0)
        if is_known:
            msg = f"{name} entered (confidence: {confidence:.1%})"
        else:
            msg = "Unknown person entered"
    elif event == "person_left":
        name = message.get("name", "unknown")
        duration = message.get("duration", 0)
        msg = f"{name} left after {duration:.0f}s"
    elif event == "motion_detected":
        confidence = message.get("confidence", 0)
        msg = f"Motion detected (level: {confidence:.1%})"
    elif event == "unknown_face":
        name = message.get("name", "unknown")
        msg = f"Unknown face auto-enrolled as {name}"
    else:
        msg = f"Security event: {event}"

    try:
        repo = get_unified_alert_repo()

        # Strip fields already used for top-level params
        metadata = {k: v for k, v in message.items()
                    if k not in ("type", "event", "node_id", "ts")}

        await repo.save_alert(
            rule_name=f"edge_security_{event}",
            event_type="security",
            message=msg,
            source_id=f"{node_id}/security",
            event_data=metadata,
            metadata={"node_id": node_id, "timestamp": ts},
        )

    except Exception as e:
        logger.warning("Failed to store security event: %s", e)

    # Feed presence tracker (person_entered / person_left)
    if event in ("person_entered", "person_left"):
        try:
            if settings.autonomous.presence_enabled:
                from ...autonomous.presence import get_presence_tracker

                tracker = get_presence_tracker()
                await tracker.on_security_event(event, message)
        except Exception as e:
            logger.warning("Presence tracker update failed: %s", e)

    # Escalation evaluation
    escalation_result = None
    evaluator = None
    if settings.escalation.enabled:
        try:
            from ...escalation import get_escalation_evaluator

            evaluator = get_escalation_evaluator()
            escalation_result = await evaluator.evaluate(event, message, node_id)
        except Exception as e:
            logger.warning("Escalation evaluation failed: %s", e)

    # Build enhanced security_ack
    ack: dict[str, Any] = {"type": "security_ack", "event": event}
    if settings.escalation.narration_hint_enabled:
        try:
            from ...autonomous.presence import get_presence_tracker

            tracker = get_presence_tracker()
            ack["narration"] = {
                "classify": "suppressed" if (escalation_result and escalation_result.should_escalate) else "routine",
                "hint": msg,
                "occupancy_state": tracker.state.state.value,
                "occupants": list(tracker.state.occupants.keys()),
            }
        except Exception:
            ack["narration"] = {
                "classify": "routine",
                "hint": msg,
                "occupancy_state": "unknown",
                "occupants": [],
            }
    await connection.send(ack)

    # Async escalation (non-blocking)
    if escalation_result and escalation_result.should_escalate and evaluator is not None:
        asyncio.create_task(
            evaluator.synthesize_and_send(escalation_result, connection),
            name=f"escalation-{node_id}",
        )




async def _handle_recognition_event(
    connection: EdgeConnection,
    message: dict[str, Any],
) -> None:
    """Handle recognition events from an edge node.

    Recognition types: speaker, face, gait, face+gait
    Stores to recognition_events table.
    """
    person_name = message.get("person_name", "unknown")
    recognition_type = message.get("recognition_type", "unknown")
    confidence = message.get("confidence", 0.0)
    camera_source = message.get("camera_source", "cam1")
    node_id = message.get("node_id", connection.location_id)
    ts = message.get("ts", time.time())

    logger.info(
        "Recognition event from %s: %s identified as %s (%.3f)",
        node_id, recognition_type, person_name, confidence,
    )

    try:
        pool = get_db_pool()

        if pool.is_initialized:
            # Look up person_id by name (optional FK)
            person_id = await pool.fetchval(
                "SELECT id FROM persons WHERE name = $1 LIMIT 1",
                person_name,
            )

            metadata = {
                "node_id": node_id,
                "timestamp": ts,
            }
            # Include extra fields from edge message
            for k in ("track_id", "gait_confidence", "face_confidence", "combined_confidence"):
                if k in message:
                    metadata[k] = message[k]

            await pool.execute(
                """
                INSERT INTO recognition_events
                    (person_id, recognition_type, confidence, camera_source, matched, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                """,
                person_id,
                recognition_type,
                confidence,
                camera_source,
                person_id is not None,
                json.dumps(metadata),
            )

    except Exception as e:
        logger.warning("Failed to store recognition event: %s", e)

    await connection.send({"type": "recognition_ack", "person_name": person_name})

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
