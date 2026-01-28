"""
Video streaming endpoints for Atlas.

Provides MJPEG streaming with real-time object detection and tracking
using YOLO-World (open vocabulary) and ByteTrack.
"""

import asyncio
import logging
import colorsys
from typing import AsyncGenerator, Optional

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse

from ..config import settings

logger = logging.getLogger("atlas.api.video")

router = APIRouter(prefix="/video", tags=["video"])


# =============================================================================
# Unified Frame Source - fetches frames from atlas_vision (single camera owner)
# =============================================================================

def get_atlas_vision_url() -> str:
    """Get atlas_vision API URL."""
    return settings.security.video_processing_url


class AtlasVisionFrameSource:
    """
    Frame source that fetches from atlas_vision API.

    Provides same interface as cv2.VideoCapture but fetches frames
    from atlas_vision's /cameras/{id}/snapshot endpoint.
    """

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.base_url = get_atlas_vision_url()
        self._client: Optional[httpx.AsyncClient] = None
        self._opened = False

    async def open(self) -> bool:
        """Open connection to atlas_vision."""
        try:
            self._client = httpx.AsyncClient(timeout=5.0)
            # Test connection
            response = await self._client.get(f"{self.base_url}/cameras/{self.camera_id}")
            if response.status_code == 200:
                self._opened = True
                return True
            logger.warning("Camera %s not found in atlas_vision", self.camera_id)
            return False
        except Exception as e:
            logger.error("Failed to connect to atlas_vision: %s", e)
            return False

    def isOpened(self) -> bool:
        return self._opened

    async def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Fetch a frame from atlas_vision."""
        if not self._opened or not self._client:
            return False, None

        try:
            response = await self._client.get(
                f"{self.base_url}/cameras/{self.camera_id}/snapshot"
            )
            if response.status_code != 200:
                return False, None

            # Decode JPEG to numpy array
            jpeg_bytes = response.content
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return False, None
            return True, frame

        except Exception as e:
            logger.warning("Frame fetch error: %s", e)
            return False, None

    async def release(self) -> None:
        """Close connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._opened = False


# Map device indices to camera IDs in atlas_vision
DEVICE_TO_CAMERA_MAP = {
    0: "webcam_office",
    # Add more mappings as needed
}


async def get_frame_source(source: str) -> AtlasVisionFrameSource:
    """
    Get a frame source for the given source identifier.

    Args:
        source: Either a device index ("0") or camera ID ("webcam_office")

    Returns:
        AtlasVisionFrameSource connected to atlas_vision
    """
    # Map device index to camera ID
    if source.isdigit():
        device_idx = int(source)
        camera_id = DEVICE_TO_CAMERA_MAP.get(device_idx)
        if not camera_id:
            raise HTTPException(
                status_code=400,
                detail=f"No camera mapped for device index {device_idx}. "
                       f"Available: {DEVICE_TO_CAMERA_MAP}"
            )
    else:
        camera_id = source

    frame_source = AtlasVisionFrameSource(camera_id)
    if not await frame_source.open():
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to camera {camera_id} via atlas_vision"
        )
    return frame_source

# Shared YOLO models
_yolo_model = None
_yolo_world_model = None

# Comprehensive class list for YOLO-World autonomous detection
# Covers people, animals, household items, electronics, furniture, etc.
YOLO_WORLD_CLASSES = [
    # People & body parts
    "person", "face", "hand", "head",
    # Animals
    "cat", "dog", "bird", "fish", "hamster", "rabbit",
    # Furniture
    "chair", "couch", "sofa", "bed", "table", "desk", "shelf", "cabinet",
    "drawer", "wardrobe", "nightstand", "ottoman", "stool", "bench",
    # Electronics
    "laptop", "computer", "monitor", "keyboard", "mouse", "phone", "cell phone",
    "smartphone", "tablet", "tv", "television", "remote control", "speaker",
    "headphones", "earbuds", "charger", "cable", "webcam", "microphone",
    "game controller", "router", "printer",
    # Kitchen items
    "cup", "mug", "glass", "bottle", "bowl", "plate", "fork", "knife", "spoon",
    "pan", "pot", "kettle", "toaster", "microwave", "refrigerator", "blender",
    "coffee maker", "food", "fruit", "apple", "banana", "orange",
    # Office items
    "book", "notebook", "pen", "pencil", "paper", "envelope", "scissors",
    "stapler", "tape", "folder", "binder", "calendar", "clock", "lamp",
    # Personal items
    "wallet", "keys", "watch", "glasses", "sunglasses", "hat", "bag",
    "backpack", "purse", "umbrella", "shoes", "jacket", "shirt",
    # Household
    "pillow", "blanket", "towel", "plant", "flower", "vase", "picture frame",
    "mirror", "curtain", "rug", "trash can", "broom", "vacuum",
    # Toys & misc
    "toy", "ball", "stuffed animal", "teddy bear", "doll", "puzzle", "board game",
    # Vehicles (for outdoor cams)
    "car", "truck", "motorcycle", "bicycle", "bus", "van",
    # Other common objects
    "door", "window", "light", "fan", "air conditioner", "heater",
    "smoke detector", "thermostat", "outlet", "switch",
]

# COCO classes for regular YOLO
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def _get_color_for_class(class_name: str) -> tuple[int, int, int]:
    """Generate a unique color based on class name hash."""
    hash_val = hash(class_name) % 360
    hue = hash_val / 360.0
    rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
    return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))


def _get_yolo_model():
    """Lazy load standard YOLO model."""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            logger.info("Loading YOLOv8m...")
            _yolo_model = YOLO("yolov8m.pt")
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _yolo_model(dummy, verbose=False)
            logger.info("YOLOv8m loaded")
        except Exception as e:
            logger.error("Failed to load YOLO: %s", e)
    return _yolo_model


def _get_yolo_world_model():
    """Lazy load YOLO-World model with comprehensive classes."""
    global _yolo_world_model
    if _yolo_world_model is None:
        try:
            from ultralytics import YOLO
            logger.info("Loading YOLO-World...")
            _yolo_world_model = YOLO("yolov8l-world.pt")
            # Set comprehensive class list
            _yolo_world_model.set_classes(YOLO_WORLD_CLASSES)
            # Warm up
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _yolo_world_model(dummy, verbose=False)
            logger.info("YOLO-World loaded with %d classes", len(YOLO_WORLD_CLASSES))
        except Exception as e:
            logger.error("Failed to load YOLO-World: %s", e)
    return _yolo_world_model


def _draw_detections(
    frame: np.ndarray,
    results,
    class_names: list[str],
    show_conf: bool = True,
    show_track_id: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """Draw detection boxes and labels on frame."""
    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

            # Get track ID if available
            track_id = None
            if hasattr(box, 'id') and box.id is not None:
                track_id = int(box.id[0])

            color = _get_color_for_class(class_name)

            # Draw bounding box with thickness based on confidence
            thickness = 2 if conf < 0.7 else 3
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Build label
            label_parts = [class_name]
            if show_track_id and track_id is not None:
                label_parts.insert(0, f"#{track_id}")
            if show_conf:
                label_parts.append(f"{conf:.0%}")
            label = " ".join(label_parts)

            # Draw label background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            detections.append({
                "class_name": class_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "track_id": track_id,
            })

    return frame, detections


async def _generate_mjpeg(
    source: str,
    fps: int = 30,
    detect: bool = True,
    track: bool = True,
    use_world: bool = True,
    threshold: float = 0.3,
    width: int = 640,
    height: int = 480,
    custom_classes: Optional[list[str]] = None,
) -> AsyncGenerator[bytes, None]:
    """
    Generate MJPEG frames with detection/tracking overlay.

    Args:
        source: Video source (device index or camera ID)
        fps: Target frame rate
        detect: Run object detection
        track: Enable object tracking
        use_world: Use YOLO-World (True) or standard YOLO (False)
        threshold: Confidence threshold
        custom_classes: Custom class list for YOLO-World (None = use default comprehensive list)
    """
    # Use atlas_vision as frame source (unified camera access)
    frame_source = await get_frame_source(source)

    frame_interval = 1.0 / fps

    # Select model
    if use_world:
        model = _get_yolo_world_model()
        class_names = custom_classes if custom_classes else YOLO_WORLD_CLASSES
        if custom_classes:
            model.set_classes(custom_classes)
    else:
        model = _get_yolo_model()
        class_names = COCO_CLASSES

    detection_count = 0
    unique_classes = set()

    try:
        while True:
            start_time = asyncio.get_event_loop().time()

            ret, frame = await frame_source.read()
            if not ret or frame is None:
                await asyncio.sleep(0.1)
                continue

            detections = []

            if detect and model:
                if track:
                    results = await asyncio.to_thread(
                        lambda: model.track(
                            frame,
                            verbose=False,
                            persist=True,
                            conf=threshold,
                            tracker="bytetrack.yaml",
                        )
                    )
                else:
                    results = await asyncio.to_thread(
                        lambda: model(frame, verbose=False, conf=threshold)
                    )

                frame, detections = _draw_detections(frame, results, class_names)
                detection_count = len(detections)
                unique_classes = set(d["class_name"] for d in detections)

            # Draw stats overlay
            model_name = "YOLO-World" if use_world else "YOLOv8m"
            stats_line1 = f"{model_name} | FPS: {fps} | Objects: {detection_count}"
            stats_line2 = f"Classes: {', '.join(sorted(unique_classes)[:5])}" if unique_classes else ""

            # Background for stats
            cv2.rectangle(frame, (5, 5), (450, 55 if stats_line2 else 35), (0, 0, 0), -1)
            cv2.putText(frame, stats_line1, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if stats_line2:
                cv2.putText(frame, stats_line2, (10, 48),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = jpeg.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            elapsed = asyncio.get_event_loop().time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    finally:
        await frame_source.release()
        logger.info("Video stream closed: %s", source)


@router.get("/webcam")
async def stream_webcam(
    device: int = Query(default=0, description="Webcam device index"),
    fps: int = Query(default=30, ge=1, le=30, description="Target FPS"),
    detect: bool = Query(default=True, description="Enable object detection"),
    track: bool = Query(default=True, description="Enable object tracking"),
    world: bool = Query(default=True, description="Use YOLO-World (True) or standard YOLO (False)"),
    threshold: float = Query(default=0.3, ge=0.1, le=1.0, description="Confidence threshold"),
    classes: Optional[str] = Query(default=None, description="Custom comma-separated classes for YOLO-World"),
    width: int = Query(default=640, description="Frame width"),
    height: int = Query(default=480, description="Frame height"),
):
    """
    Stream webcam with real-time object detection and tracking.

    **YOLO-World Mode (default):**
    Detects 100+ object types automatically including people, furniture, electronics,
    kitchen items, personal items, and more.

    **URLs:**
    - Auto-detect everything: http://localhost:8002/api/v1/video/webcam
    - Standard YOLO (80 classes): http://localhost:8002/api/v1/video/webcam?world=false
    - Custom classes: http://localhost:8002/api/v1/video/webcam?classes=person,cat,dog,laptop
    - Lower threshold: http://localhost:8002/api/v1/video/webcam?threshold=0.2
    """
    logger.info("Starting webcam stream: device=%d, fps=%d, world=%s, track=%s",
                device, fps, world, track)

    custom_classes = None
    if classes:
        custom_classes = [c.strip() for c in classes.split(",")]

    return StreamingResponse(
        _generate_mjpeg(
            source=str(device),
            fps=fps,
            detect=detect,
            track=track,
            use_world=world,
            threshold=threshold,
            width=width,
            height=height,
            custom_classes=custom_classes,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/rtsp/{camera_id}")
async def stream_rtsp(
    camera_id: str,
    fps: int = Query(default=10, ge=1, le=30, description="Target FPS"),
    detect: bool = Query(default=True, description="Enable object detection"),
    track: bool = Query(default=True, description="Enable object tracking"),
    world: bool = Query(default=True, description="Use YOLO-World"),
    threshold: float = Query(default=0.3, ge=0.1, le=1.0, description="Confidence threshold"),
    classes: Optional[str] = Query(default=None, description="Custom classes"),
):
    """Stream RTSP camera with YOLO-World detection."""
    from ..config import get_settings
    import json

    settings = get_settings()

    rtsp_url = None
    if settings.rtsp.cameras_json:
        try:
            cameras = json.loads(settings.rtsp.cameras_json)
            for cam in cameras:
                if cam.get("camera_id") == camera_id:
                    rtsp_url = cam.get("rtsp_url")
                    break
        except json.JSONDecodeError:
            pass

    if not rtsp_url:
        rtsp_url = f"rtsp://{settings.rtsp.wyze_bridge_host}:{settings.rtsp.wyze_bridge_port}/{camera_id}"

    logger.info("Starting RTSP stream: %s -> %s", camera_id, rtsp_url)

    custom_classes = None
    if classes:
        custom_classes = [c.strip() for c in classes.split(",")]

    return StreamingResponse(
        _generate_mjpeg(
            source=rtsp_url,
            fps=fps,
            detect=detect,
            track=track,
            use_world=world,
            threshold=threshold,
            custom_classes=custom_classes,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/snapshot/webcam")
async def snapshot_webcam(
    device: int = Query(default=0, description="Webcam device index"),
    detect: bool = Query(default=True, description="Run detection"),
    world: bool = Query(default=True, description="Use YOLO-World"),
    threshold: float = Query(default=0.3, description="Confidence threshold"),
):
    """Get a single snapshot with detection."""
    # Use atlas_vision as frame source (unified camera access)
    frame_source = await get_frame_source(str(device))

    ret, frame = await frame_source.read()
    await frame_source.release()

    if not ret or frame is None:
        raise HTTPException(status_code=503, detail="Failed to capture frame from atlas_vision")

    if detect:
        model = _get_yolo_world_model() if world else _get_yolo_model()
        class_names = YOLO_WORLD_CLASSES if world else COCO_CLASSES
        if model:
            results = model(frame, verbose=False, conf=threshold)
            frame, _ = _draw_detections(frame, results, class_names)

    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return StreamingResponse(iter([jpeg.tobytes()]), media_type="image/jpeg")


@router.get("/classes")
async def list_classes():
    """List all detection classes."""
    return {
        "yolo_world_classes": YOLO_WORLD_CLASSES,
        "coco_classes": COCO_CLASSES,
        "yolo_world_count": len(YOLO_WORLD_CLASSES),
        "coco_count": len(COCO_CLASSES),
    }


# =============================================================================
# Recognition Streaming - MOVED TO ATLAS_VISION
# =============================================================================
#
# The recognition streaming endpoints have been migrated to atlas_vision service.
# Use the following endpoint instead:
#
#   GET http://{atlas_vision_url}/cameras/{camera_id}/stream/recognition/full
#
# Parameters:
#   - fps: Target FPS (1-30)
#   - face_threshold: Face recognition threshold (0.3-1.0)
#   - gait_threshold: Gait recognition threshold (0.3-1.0)
#   - person_threshold: Person detection threshold (0.3-1.0)
#   - auto_enroll: Auto-enroll unknown faces
#   - enroll_gait: Auto-enroll gait for known faces
#


@router.get("/webcam/recognition")
async def stream_webcam_recognition_deprecated(
    device: int = Query(default=0),
):
    """
    DEPRECATED: Recognition streaming moved to atlas_vision.

    Use atlas_vision endpoint instead:
    GET /cameras/{camera_id}/stream/recognition/full
    """
    from ..config import settings
    vision_url = settings.security.video_processing_url
    raise HTTPException(
        status_code=410,
        detail={
            "message": "This endpoint has been moved to atlas_vision",
            "new_endpoint": f"{vision_url}/cameras/webcam_{device}/stream/recognition/full",
            "reason": "GPU workload consolidated in atlas_vision service",
        }
    )


@router.get("/webcam/recognition/multitrack")
async def stream_webcam_recognition_multitrack_deprecated(
    device: int = Query(default=0),
):
    """
    DEPRECATED: Multi-track recognition streaming moved to atlas_vision.

    Use atlas_vision endpoint instead:
    GET /cameras/{camera_id}/stream/recognition/full
    """
    from ..config import settings
    vision_url = settings.security.video_processing_url
    raise HTTPException(
        status_code=410,
        detail={
            "message": "This endpoint has been moved to atlas_vision",
            "new_endpoint": f"{vision_url}/cameras/webcam_{device}/stream/recognition/full",
            "reason": "GPU workload consolidated in atlas_vision service",
        }
    )

