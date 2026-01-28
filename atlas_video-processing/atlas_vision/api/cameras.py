"""
Camera management endpoints.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
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


# =============================================================================
# Full Recognition Streaming (Face + Gait + YOLO Tracking)
# =============================================================================


@router.get("/{camera_id}/stream/recognition/full")
async def stream_full_recognition(
    camera_id: str,
    fps: int = Query(default=15, ge=1, le=30),
    face_threshold: float = Query(default=0.6, ge=0.3, le=1.0),
    gait_threshold: float = Query(default=0.5, ge=0.3, le=1.0),
    person_threshold: float = Query(default=0.5, ge=0.3, le=1.0),
    auto_enroll: bool = Query(default=True),
    enroll_gait: bool = Query(default=True),
):
    """
    MJPEG stream with full face + gait recognition and YOLO person tracking.

    Tracks multiple people simultaneously using YOLO ByteTrack.
    Each person has independent face recognition and gait collection.

    Visual indicators:
    - Green box: Identified person (face+gait combined match)
    - Light green box: Identified by face only
    - Orange box: Newly enrolled person
    - Gray box: Unidentified person being tracked
    - Orange progress bar: Gait collection progress
    """
    import cv2
    import asyncio
    from typing import AsyncGenerator
    from fastapi.responses import StreamingResponse

    from ..core.config import settings
    from ..processing.detection.yolo import get_yolo_detector
    from ..recognition import (
        get_face_service,
        get_gait_service,
        get_person_repository,
        get_track_manager,
    )

    camera = device_registry.get(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera not found: {camera_id}")

    async def generate_recognition_frames() -> AsyncGenerator[bytes, None]:
        cfg = settings.recognition
        recognition_interval = cfg.recognition_interval

        yolo = get_yolo_detector()
        face_service = get_face_service()
        gait_service = get_gait_service()
        repo = get_person_repository()
        track_manager = get_track_manager()

        frame_interval = 1.0 / fps
        track_last_face_rec: dict[int, float] = {}

        try:
            while True:
                start_time = asyncio.get_event_loop().time()

                frame = await camera.get_frame()
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue

                frame_h, frame_w = frame.shape[:2]

                # Run YOLO tracking for person detection
                tracks = await yolo.track(frame, camera_id)
                person_tracks = [t for t in tracks if t.class_name == "person"]

                # Update track manager with YOLO tracks
                for track in person_tracks:
                    bbox = [
                        int(track.bbox.x1 * frame_w),
                        int(track.bbox.y1 * frame_h),
                        int(track.bbox.x2 * frame_w),
                        int(track.bbox.y2 * frame_h),
                    ]
                    track_manager.create_or_update_track(
                        camera_id, track.track_id, bbox
                    )

                # Detect all faces in frame
                faces = await asyncio.to_thread(face_service.detect_faces, frame)

                # Associate faces with person tracks
                face_to_track: dict[int, dict] = {}
                for face in faces:
                    face_bbox = face["bbox"]
                    best_track = track_manager.find_track_containing_bbox(
                        camera_id, face_bbox
                    )
                    if best_track:
                        face_to_track[best_track.track_id] = face

                # Process each person track for recognition
                for track in person_tracks:
                    track_id = track.track_id
                    mgr_track = track_manager.get_track(camera_id, track_id)
                    if not mgr_track:
                        continue

                    # Face recognition for this track
                    if track_id in face_to_track:
                        last_rec = track_last_face_rec.get(track_id, 0)
                        if start_time - last_rec > recognition_interval:
                            track_last_face_rec[track_id] = start_time
                            try:
                                result = await face_service.recognize_face(
                                    frame=frame,
                                    threshold=face_threshold,
                                    auto_enroll_unknown=auto_enroll,
                                    use_averaged=cfg.use_averaged,
                                )
                                if result and (result.get("matched") or result.get("auto_enrolled")):
                                    track_manager.associate_person(
                                        camera_id,
                                        track_id,
                                        result["person_id"],
                                        result["name"],
                                        result.get("is_known", False),
                                        result["similarity"],
                                    )
                                    if enroll_gait and result.get("is_known"):
                                        counts = await repo.get_person_embedding_counts(
                                            result["person_id"]
                                        )
                                        if counts["gait_embeddings"] == 0:
                                            track_manager.mark_needs_gait_enrollment(
                                                camera_id, track_id
                                            )
                            except Exception as e:
                                logger.warning("Face rec error track %d: %s", track_id, e)

                # Extract pose and process gait
                pose = gait_service.extract_pose(frame)
                if pose:
                    pose_bbox = gait_service.extract_pose_bbox(pose, frame_w, frame_h)
                    if pose_bbox:
                        pose_track = track_manager.find_track_containing_bbox(
                            camera_id, pose_bbox
                        )
                        if pose_track:
                            gait_service.add_pose_to_track(
                                camera_id, pose_track.track_id, pose
                            )
                            if gait_service.is_track_buffer_full(camera_id, pose_track.track_id):
                                pt_track = track_manager.get_track(camera_id, pose_track.track_id)
                                if pt_track and pt_track.needs_gait_enrollment:
                                    try:
                                        emb_id = await gait_service.enroll_track_gait(
                                            camera_source=camera_id,
                                            track_id=pose_track.track_id,
                                            person_id=pt_track.person_id,
                                            walking_direction="mixed",
                                            source="auto_stream",
                                        )
                                        if emb_id:
                                            track_manager.mark_gait_enrolled(
                                                camera_id, pose_track.track_id
                                            )
                                    except Exception as e:
                                        logger.error("Gait enroll error: %s", e)
                                elif pt_track and pt_track.person_id:
                                    try:
                                        gait_result = await gait_service.recognize_track_gait(
                                            camera_source=camera_id,
                                            track_id=pose_track.track_id,
                                            threshold=gait_threshold,
                                            use_averaged=cfg.use_averaged,
                                        )
                                        if gait_result and gait_result.get("matched"):
                                            track_manager.update_gait_match(
                                                camera_id,
                                                pose_track.track_id,
                                                gait_result["similarity"],
                                            )
                                    except Exception as e:
                                        logger.warning("Gait rec error: %s", e)

                # Cleanup stale tracks
                track_manager.cleanup_stale_tracks(max_age=cfg.track_timeout)

                # Draw overlays
                for track in person_tracks:
                    track_id = track.track_id
                    x1 = int(track.bbox.x1 * frame_w)
                    y1 = int(track.bbox.y1 * frame_h)
                    x2 = int(track.bbox.x2 * frame_w)
                    y2 = int(track.bbox.y2 * frame_h)
                    mgr_track = track_manager.get_track(camera_id, track_id)

                    if mgr_track and mgr_track.person_id:
                        if mgr_track.combined_similarity > 0:
                            color = (0, 255, 0)
                            label = f"#{track_id} {mgr_track.person_name} ({mgr_track.combined_similarity:.0%})"
                        elif mgr_track.face_similarity > 0:
                            color = (0, 200, 0)
                            label = f"#{track_id} {mgr_track.person_name} ({mgr_track.face_similarity:.0%})"
                        else:
                            color = (0, 165, 255)
                            label = f"#{track_id} NEW: {mgr_track.person_name}"
                    else:
                        color = (128, 128, 128)
                        label = f"#{track_id} Person ({track.confidence:.0%})"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Gait progress bar
                    gait_progress = gait_service.get_track_progress(camera_id, track_id)
                    bar_y1 = min(y2 + 2, frame_h - 10)
                    bar_y2 = min(y2 + 8, frame_h - 2)
                    cv2.rectangle(frame, (x1, bar_y1), (x2, bar_y2), (100, 100, 100), 1)
                    if gait_progress > 0:
                        bar_width = max(int((x2 - x1) * min(gait_progress, 1.0)), 2)
                        cv2.rectangle(frame, (x1, bar_y1), (x1 + bar_width, bar_y2), (0, 165, 255), -1)

                    # Label
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Stats overlay
                summary = track_manager.get_track_summary(camera_id)
                cv2.rectangle(frame, (5, 5), (300, 45), (0, 0, 0), -1)
                stats_line1 = f"Persons: {summary['total_tracks']} | Identified: {summary['identified']}"
                stats_line2 = f"With Gait: {summary['with_gait']} | FPS: {fps}"
                cv2.putText(frame, stats_line1, (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(frame, stats_line2, (10, 38),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # Encode and yield
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                )

                elapsed = asyncio.get_event_loop().time() - start_time
                if frame_interval - elapsed > 0:
                    await asyncio.sleep(frame_interval - elapsed)

        except Exception as e:
            logger.error("Recognition stream error: %s", e)
        finally:
            logger.info("Recognition stream closed: %s", camera_id)

    return StreamingResponse(
        generate_recognition_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
