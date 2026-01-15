"""
Detection and event endpoints.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Query

from ..core.constants import DeviceType, DetectionType
from ..devices.registry import device_registry

logger = logging.getLogger("atlas.vision.api.detections")

router = APIRouter()


@router.get("/detections/current")
async def get_current_detections(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    type: Optional[str] = Query(None, description="Filter by detection type"),
):
    """Get current/recent detections."""
    cameras = device_registry.list_by_type(DeviceType.CAMERA)
    detections = []

    for camera in cameras:
        if camera_id and camera.device_id != camera_id:
            continue

        # Get detections from camera
        camera_detections = await camera.get_current_detections()

        for det in camera_detections:
            if type and det.detection_type.value != type:
                continue
            detections.append(det.to_dict())

    return {"detections": detections, "count": len(detections)}


@router.get("/events")
async def get_events(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    type: Optional[str] = Query(None, description="Event type filter"),
    since: Optional[str] = Query(None, description="ISO timestamp to filter from"),
):
    """Get motion/detection events."""
    cameras = device_registry.list_by_type(DeviceType.CAMERA)
    events = []

    # Parse since timestamp
    since_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            since_dt = datetime.now() - timedelta(hours=1)

    for camera in cameras:
        if camera_id and camera.device_id != camera_id:
            continue

        # Get events from camera
        camera_events = await camera.get_motion_events(since=since_dt)

        for event in camera_events:
            if type and type != "motion":
                continue
            events.append(event.to_dict())

    return {"events": events, "count": len(events)}
