"""
Track API endpoints for real-time object tracking.

Provides REST endpoints and WebSocket streaming for
tracked objects.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from ..core.models import DetectionEventType
from ..processing.tracking import get_track_store
from ..processing.detection import get_yolo_detector

logger = logging.getLogger("atlas.vision.api.tracks")

router = APIRouter()


@router.get("/active")
async def get_active_tracks(
    source_id: Optional[str] = Query(None, description="Filter by camera/source ID"),
) -> dict:
    """
    Get all currently tracked objects.

    Returns active tracks across all cameras (or filtered by source).
    """
    store = get_track_store()
    tracks = store.get_active_tracks(source_id=source_id)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "count": len(tracks),
        "tracks": [t.to_dict() for t in tracks],
    }


@router.get("/all")
async def get_all_tracks(
    source_id: Optional[str] = Query(None, description="Filter by camera/source ID"),
) -> dict:
    """
    Get all tracks including lost ones.

    Useful for debugging and understanding track lifecycle.
    """
    store = get_track_store()
    tracks = store.get_all_tracks(source_id=source_id)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "count": len(tracks),
        "tracks": [t.to_dict() for t in tracks],
    }


@router.get("/events")
async def get_events(
    source_id: Optional[str] = Query(None, description="Filter by source"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    limit: int = Query(100, ge=1, le=1000, description="Max events to return"),
) -> dict:
    """
    Get recent detection events.

    Events include: new_track, track_lost, etc.
    """
    store = get_track_store()

    # Parse event type
    evt_type = None
    if event_type:
        try:
            evt_type = DetectionEventType(event_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid event_type. Valid: {[e.value for e in DetectionEventType]}",
            )

    events = store.get_events(source_id=source_id, event_type=evt_type, limit=limit)

    return {
        "count": len(events),
        "events": [e.to_dict() for e in events],
    }


@router.get("/summary")
async def get_track_summary() -> dict:
    """
    Get summary of tracked objects by source and class.

    Useful for dashboard displays showing counts.
    """
    store = get_track_store()
    counts = store.get_track_counts()

    # Calculate totals
    total_tracks = 0
    class_totals: dict[str, int] = {}

    for source_counts in counts.values():
        for class_name, count in source_counts.items():
            total_tracks += count
            class_totals[class_name] = class_totals.get(class_name, 0) + count

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "total_tracks": total_tracks,
        "by_class": class_totals,
        "by_source": counts,
    }


@router.get("/{track_id}")
async def get_track(track_id: int) -> dict:
    """
    Get a specific track by ID.

    Returns detailed track info including movement path.
    """
    store = get_track_store()
    track = store.get_track(track_id)

    if not track:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

    return track.to_dict_with_path()


@router.get("/{track_id}/history")
async def get_track_history(
    track_id: int,
    limit: int = Query(50, ge=1, le=500, description="Max points to return"),
) -> dict:
    """
    Get movement history for a specific track.

    Returns the path the object has traveled.
    """
    store = get_track_store()
    track = store.get_track(track_id)

    if not track:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

    path = track.path[-limit:]

    return {
        "track_id": track_id,
        "class": track.class_name,
        "point_count": len(path),
        "points": [p.to_dict() for p in path],
    }


# WebSocket connection manager
class TrackConnectionManager:
    """Manages WebSocket connections for track streaming."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info("Track WebSocket connected (total: %d)", len(self.active_connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info("Track WebSocket disconnected (total: %d)", len(self.active_connections))

    async def broadcast(self, message: dict) -> None:
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return

        data = json.dumps(message)
        disconnected = []

        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.send_text(data)
                except Exception:
                    disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            await self.disconnect(conn)


# Global connection manager
_connection_manager = TrackConnectionManager()


def get_connection_manager() -> TrackConnectionManager:
    """Get the global connection manager."""
    return _connection_manager


@router.websocket("/ws")
async def track_websocket(
    websocket: WebSocket,
    source_id: Optional[str] = Query(None, description="Filter by source"),
):
    """
    WebSocket endpoint for real-time track updates.

    Streams track updates as JSON messages:
    - {"event": "update", "tracks": [...]}
    - {"event": "new", "track": {...}}
    - {"event": "lost", "track_id": 42}
    """
    manager = get_connection_manager()
    await manager.connect(websocket)

    try:
        # Send initial state
        store = get_track_store()
        tracks = store.get_active_tracks(source_id=source_id)
        await websocket.send_json({
            "event": "init",
            "tracks": [t.to_dict() for t in tracks],
        })

        # Keep connection alive and handle ping/pong
        while True:
            try:
                # Wait for messages (client can send ping or filter updates)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Handle client messages if needed
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"event": "keepalive"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("WebSocket error: %s", e)
    finally:
        await manager.disconnect(websocket)


async def broadcast_track_update(source_id: str, tracks: list) -> None:
    """
    Broadcast track update to all WebSocket clients.

    Called by the detection pipeline when tracks are updated.
    """
    manager = get_connection_manager()

    if not manager.active_connections:
        return

    await manager.broadcast({
        "event": "update",
        "source_id": source_id,
        "timestamp": datetime.utcnow().isoformat(),
        "tracks": [t.to_dict() for t in tracks],
    })


async def broadcast_event(event) -> None:
    """
    Broadcast detection event to all WebSocket clients.
    """
    manager = get_connection_manager()

    if not manager.active_connections:
        return

    await manager.broadcast({
        "event": event.event_type.value,
        "data": event.to_dict(),
    })
