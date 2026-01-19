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


class RegisterWebcamRequest(BaseModel):
    """Webcam registration request."""
    camera_id: str
    name: str
    location: str
    device_index: int = 0
    fps: int = 15
    width: int = 640
    height: int = 480
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
    """Get a JPEG snapshot from a camera."""
    import cv2
    from datetime import datetime
    from fastapi.responses import Response

    camera = device_registry.get(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera not found: {camera_id}")

    frame = await camera.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available")

    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(
        content=jpeg.tobytes(),
        media_type="image/jpeg",
        headers={"X-Timestamp": datetime.now().isoformat()},
    )


@router.get("/{camera_id}/stream")
async def stream_camera(camera_id: str):
    """MJPEG stream from a camera."""
    import cv2
    import asyncio
    from fastapi.responses import StreamingResponse

    camera = device_registry.get(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera not found: {camera_id}")

    async def generate_frames():
        while True:
            frame = await camera.get_frame()
            if frame is not None:
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                )
            await asyncio.sleep(0.066)  # ~15 fps

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/{camera_id}/recognition_stream")
async def stream_camera_with_recognition(
    camera_id: str,
    face: bool = True,
    pose: bool = True,
):
    """
    MJPEG stream with face detection boxes and pose skeleton overlays.

    Detection runs in background threads while frames stream continuously.

    Args:
        camera_id: Camera identifier
        face: Enable face detection boxes (default: true)
        pose: Enable pose skeleton overlay (default: true)
    """
    import cv2
    import asyncio
    import threading
    from fastapi.responses import StreamingResponse
    from ..processing.detection import get_face_detector, get_pose_detector

    camera = device_registry.get(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera not found: {camera_id}")

    # Get detectors if enabled
    face_detector = get_face_detector() if face else None
    pose_detector = get_pose_detector() if pose else None

    async def generate_recognition_frames():
        # Shared state for detection results (protected by lock)
        detection_lock = threading.Lock()
        cached_faces = []
        cached_poses = []
        detection_frame = None
        detection_running = False
        stop_detection = threading.Event()

        def run_detection():
            """Background thread for running detection."""
            nonlocal cached_faces, cached_poses, detection_frame, detection_running

            while not stop_detection.is_set():
                # Get frame to process
                with detection_lock:
                    frame_to_process = detection_frame
                    detection_frame = None
                    if frame_to_process is None:
                        detection_running = False

                if frame_to_process is None:
                    stop_detection.wait(0.05)  # Wait briefly
                    continue

                # Run detection
                new_faces = []
                new_poses = []

                if face_detector:
                    new_faces = face_detector.detect(frame_to_process)
                if pose_detector:
                    new_poses = pose_detector.detect(frame_to_process)

                # Update cached results
                with detection_lock:
                    cached_faces = new_faces
                    cached_poses = new_poses
                    detection_running = False

        # Start detection thread
        detection_thread = threading.Thread(target=run_detection, daemon=True)
        detection_thread.start()

        try:
            while True:
                frame = await camera.get_frame()
                if frame is not None:
                    # Submit frame for detection if not busy
                    with detection_lock:
                        if not detection_running:
                            detection_frame = frame.copy()
                            detection_running = True

                        # Get current detection results
                        faces_to_draw = cached_faces
                        poses_to_draw = cached_poses

                    # Draw cached overlays
                    if face_detector and faces_to_draw:
                        face_detector.draw_detections(frame, faces_to_draw)
                    if pose_detector and poses_to_draw:
                        pose_detector.draw_detections(frame, poses_to_draw)

                    # Encode and yield frame
                    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                    )

                await asyncio.sleep(0.066)  # ~15 fps streaming rate
        finally:
            stop_detection.set()
            detection_thread.join(timeout=1.0)

    return StreamingResponse(
        generate_recognition_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


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


@router.post("/register/webcam")
async def register_webcam(request: RegisterWebcamRequest):
    """Register a local USB webcam."""
    from ..devices.cameras import WebcamCamera
    from ..processing.detection import get_motion_detector

    existing = device_registry.get(request.camera_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Camera already registered: {request.camera_id}"
        )

    try:
        camera = WebcamCamera(
            device_id=request.camera_id,
            name=request.name,
            location=request.location,
            device_index=request.device_index,
            fps=request.fps,
            width=request.width,
            height=request.height,
        )

        if request.enable_motion:
            motion_detector = get_motion_detector()
            camera.set_motion_detector(motion_detector)

        connected = await camera.connect()
        if not connected:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to open webcam at /dev/video{request.device_index}"
            )

        device_registry.register(camera)

        from ..processing.pipeline import get_detection_pipeline
        pipeline = get_detection_pipeline()
        if pipeline.is_running:
            await pipeline.add_camera(request.camera_id)

        logger.info("Registered webcam: %s (%s) at /dev/video%d",
                    request.camera_id, request.name, request.device_index)

        return {
            "success": True,
            "message": f"Webcam '{request.name}' registered successfully",
            "camera_id": request.camera_id,
            "device": f"/dev/video{request.device_index}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to register webcam %s: %s", request.camera_id, e)
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
