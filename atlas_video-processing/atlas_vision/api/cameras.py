"""
Camera management endpoints.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..core.constants import DeviceType
from ..devices.registry import device_registry

logger = logging.getLogger("atlas.vision.api.cameras")

router = APIRouter()


class RecordRequest(BaseModel):
    """Recording control request."""
    action: str  # "start" or "stop"


class RegisterCameraRequest(BaseModel):
    """Camera registration request."""
    camera_id: str
    name: str
    location: str
    rtsp_url: str
    fps: int = 10
    enable_motion: bool = True


@router.get("")
async def list_cameras():
    """List all registered cameras."""
    cameras = device_registry.list_by_type(DeviceType.CAMERA)
    camera_list = []

    for camera in cameras:
        status = await camera.get_status()
        camera_list.append(status)

    return {"cameras": camera_list}


@router.get("/{camera_id}")
async def get_camera(camera_id: str):
    """Get camera status by ID."""
    camera = device_registry.get(camera_id)

    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera not found: {camera_id}")

    return await camera.get_status()


@router.post("/{camera_id}/record")
async def control_recording(camera_id: str, request: RecordRequest):
    """Start or stop recording on a camera."""
    camera = device_registry.get(camera_id)

    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera not found: {camera_id}")

    if request.action == "start":
        success = await camera.start_recording()
        message = f"Recording started on {camera.name}" if success else "Failed to start recording"
    elif request.action == "stop":
        success = await camera.stop_recording()
        message = f"Recording stopped on {camera.name}" if success else "Failed to stop recording"
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")

    return {"success": success, "message": message, "camera_id": camera_id}


@router.get("/{camera_id}/snapshot")
async def get_snapshot(camera_id: str):
    """Get a snapshot from a camera."""
    camera = device_registry.get(camera_id)

    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera not found: {camera_id}")

    # For now, return placeholder
    return {
        "camera_id": camera_id,
        "timestamp": "2026-01-14T00:00:00",
        "message": "Snapshot endpoint ready - image streaming not yet implemented",
    }


@router.post("/register")
async def register_camera(request: RegisterCameraRequest):
    """Register a new RTSP camera."""
    from ..devices.cameras import RTSPCamera
    from ..processing.detection import get_motion_detector

    # Check if already exists
    existing = device_registry.get(request.camera_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Camera already registered: {request.camera_id}"
        )

    try:
        # Create RTSP camera
        camera = RTSPCamera(
            device_id=request.camera_id,
            name=request.name,
            location=request.location,
            rtsp_url=request.rtsp_url,
            fps=request.fps,
        )

        # Attach motion detector if enabled
        if request.enable_motion:
            motion_detector = get_motion_detector()
            camera.set_motion_detector(motion_detector)

        # Connect to stream
        connected = await camera.connect()
        if not connected:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to connect to RTSP stream: {request.rtsp_url}"
            )

        # Register with device registry
        device_registry.register(camera)

        # Add to detection pipeline if running
        from ..processing.pipeline import get_detection_pipeline
        pipeline = get_detection_pipeline()
        if pipeline.is_running:
            await pipeline.add_camera(request.camera_id)

        logger.info("Registered RTSP camera: %s (%s)", request.camera_id, request.name)

        return {
            "success": True,
            "message": f"Camera '{request.name}' registered successfully",
            "camera_id": request.camera_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to register camera %s: %s", request.camera_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{camera_id}")
async def unregister_camera(camera_id: str):
    """Unregister a camera."""
    camera = device_registry.get(camera_id)

    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera not found: {camera_id}")

    try:
        # Disconnect if it's an RTSP camera
        if hasattr(camera, "disconnect"):
            await camera.disconnect()

        # Remove from registry
        device_registry.unregister(camera_id)

        logger.info("Unregistered camera: %s", camera_id)

        return {
            "success": True,
            "message": f"Camera '{camera_id}' unregistered",
            "camera_id": camera_id,
        }

    except Exception as e:
        logger.error("Failed to unregister camera %s: %s", camera_id, e)
        raise HTTPException(status_code=500, detail=str(e))
