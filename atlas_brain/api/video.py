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
import numpy as np
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse

from ..config import settings

logger = logging.getLogger("atlas.api.video")

router = APIRouter(prefix="/video", tags=["video"])

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
        source: Video source (device index or RTSP URL)
        fps: Target frame rate
        detect: Run object detection
        track: Enable object tracking
        use_world: Use YOLO-World (True) or standard YOLO (False)
        threshold: Confidence threshold
        custom_classes: Custom class list for YOLO-World (None = use default comprehensive list)
    """
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise HTTPException(status_code=503, detail=f"Cannot open video source: {source}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Warm up camera
    for _ in range(30):
        cap.read()

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

            ret, frame = await asyncio.to_thread(cap.read)
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
        cap.release()
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
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise HTTPException(status_code=503, detail="Cannot open webcam")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(30):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise HTTPException(status_code=503, detail="Failed to capture frame")

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


# Face recognition cache to avoid repeated DB queries
_face_cache: dict = {}


async def _generate_recognition_mjpeg(
    source: str,
    fps: int = 15,
    threshold: Optional[float] = None,
    auto_enroll: Optional[bool] = None,
    use_averaged: Optional[bool] = None,
    width: int = 640,
    height: int = 480,
    enroll_gait: bool = True,
) -> AsyncGenerator[bytes, None]:
    """Generate MJPEG frames with face + gait recognition overlay."""
    from ..services.recognition import get_face_service, get_gait_service, get_person_repository

    # Use config defaults when not specified
    cfg = settings.recognition
    threshold = threshold if threshold is not None else cfg.face_threshold
    auto_enroll = auto_enroll if auto_enroll is not None else cfg.auto_enroll_unknown
    use_averaged = use_averaged if use_averaged is not None else cfg.use_averaged
    recognition_interval = cfg.recognition_interval

    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise HTTPException(status_code=503, detail=f"Cannot open video source: {source}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    for _ in range(30):
        cap.read()

    frame_interval = 1.0 / fps
    face_service = get_face_service()
    gait_service = get_gait_service()
    repo = get_person_repository()
    last_recognition = 0
    last_result = None

    # Gait enrollment state
    gait_enroll_person_id = None
    gait_enroll_person_name = None
    gait_enroll_started = False

    try:
        while True:
            start_time = asyncio.get_event_loop().time()

            ret, frame = await asyncio.to_thread(cap.read)
            if not ret or frame is None:
                await asyncio.sleep(0.1)
                continue

            faces = await asyncio.to_thread(face_service.detect_faces, frame)

            # Run face recognition periodically
            if start_time - last_recognition > recognition_interval and faces:
                last_recognition = start_time
                try:
                    last_result = await face_service.recognize_face(
                        frame=frame,
                        threshold=threshold,
                        auto_enroll_unknown=auto_enroll,
                        use_averaged=use_averaged,
                    )

                    # Check if known person needs gait enrollment
                    if enroll_gait and last_result and last_result.get("matched") and last_result.get("is_known"):
                        person_id = last_result["person_id"]
                        # Only start if not already enrolling this person
                        if gait_enroll_person_id != person_id:
                            counts = await repo.get_person_embedding_counts(person_id)
                            if counts["gait_embeddings"] == 0:
                                gait_service.clear_buffer()
                                gait_enroll_person_id = person_id
                                gait_enroll_person_name = last_result["name"]
                                gait_enroll_started = True
                                logger.info("Starting gait enrollment for %s", gait_enroll_person_name)

                except Exception as e:
                    logger.warning("Recognition error: %s", e)
                    last_result = None

            # Collect gait poses and run recognition/enrollment
            gait_progress = 0
            gait_result = None
            pose = gait_service.extract_pose(frame)
            if pose:
                buffer_len = gait_service.add_pose_to_buffer(pose)
                gait_progress = int((buffer_len / gait_service.sequence_length) * 100)

                # Buffer full - decide: enroll or recognize
                if gait_service.is_buffer_full():
                    if gait_enroll_started and gait_enroll_person_id:
                        # Enrollment mode
                        try:
                            embedding_id = await gait_service.enroll_gait(
                                person_id=gait_enroll_person_id,
                                walking_direction="mixed",
                                source="auto_enrollment",
                            )
                            if embedding_id:
                                logger.info("Gait enrolled for %s: %s", gait_enroll_person_name, embedding_id)
                        except Exception as e:
                            logger.error("Gait enrollment failed: %s", e)
                        finally:
                            gait_enroll_person_id = None
                            gait_enroll_person_name = None
                            gait_enroll_started = False
                    else:
                        # Recognition mode
                        try:
                            gait_result = await gait_service.recognize_gait(
                                threshold=cfg.gait_threshold,
                                use_averaged=use_averaged,
                                auto_enroll_unknown=False,
                            )
                        except Exception as e:
                            logger.warning("Gait recognition error: %s", e)

            # Calculate combined score
            combined_score = None
            combined_name = None
            if last_result and last_result.get("matched") and gait_result and gait_result.get("matched"):
                if last_result["person_id"] == gait_result["person_id"]:
                    combined_score = (last_result["similarity"] + gait_result["similarity"]) / 2
                    combined_name = last_result["name"]

            # Draw face boxes
            for face in faces:
                x1, y1, x2, y2 = face["bbox"]
                conf = face["det_score"]

                # Color based on recognition
                if combined_score:
                    color = (0, 255, 0)  # Bright green for combined match
                    label = f"{combined_name} ({combined_score:.0%} combined)"
                elif last_result and last_result.get("matched"):
                    color = (0, 200, 0)  # Green for face match
                    label = f"{last_result['name']} ({last_result['similarity']:.0%})"
                elif last_result and last_result.get("auto_enrolled"):
                    color = (0, 165, 255)  # Orange for new
                    label = f"NEW: {last_result['name']}"
                else:
                    color = (0, 0, 255)  # Red for unknown
                    label = f"Face ({conf:.0%})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 7),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Stats overlay
            overlay_height = 55 if (combined_score or gait_result) else 35
            cv2.rectangle(frame, (5, 5), (400, overlay_height), (0, 0, 0), -1)

            # Line 1: Face status
            face_status = f"Face: {last_result['name']} ({last_result['similarity']:.0%})" if last_result and last_result.get("matched") else "Face: Scanning..."
            cv2.putText(frame, face_status, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Line 2: Gait status
            if gait_enroll_started:
                gait_status = f"Gait: Enrolling {gait_progress}%"
            elif gait_result and gait_result.get("matched"):
                gait_status = f"Gait: {gait_result['name']} ({gait_result['similarity']:.0%})"
            elif gait_progress > 0:
                gait_status = f"Gait: Collecting {gait_progress}%"
            else:
                gait_status = "Gait: Waiting..."

            if combined_score:
                gait_status += f" | COMBINED: {combined_score:.0%}"

            cv2.putText(frame, gait_status, (10, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )

            elapsed = asyncio.get_event_loop().time() - start_time
            if frame_interval - elapsed > 0:
                await asyncio.sleep(frame_interval - elapsed)

    finally:
        cap.release()
        logger.info("Recognition stream closed: %s", source)


@router.get("/webcam/recognition")
async def stream_webcam_recognition(
    device: int = Query(default=0, description="Webcam device index"),
    fps: int = Query(default=15, ge=1, le=30, description="Target FPS"),
    threshold: float = Query(default=0.6, ge=0.3, le=1.0, description="Recognition threshold"),
    auto_enroll: bool = Query(default=True, description="Auto-enroll unknown faces"),
    enroll_gait: bool = Query(default=True, description="Auto-enroll gait for known faces without gait"),
    width: int = Query(default=640, description="Frame width"),
    height: int = Query(default=480, description="Frame height"),
):
    """
    Stream webcam with real-time face + gait recognition.

    - Green box: Known person (matched)
    - Orange box: Newly auto-enrolled person
    - Red box: Unknown face

    When a known face is detected without gait enrolled, gait frames are
    automatically collected and enrolled (shown as "Gait: X%" in overlay).

    **URL:** http://localhost:8002/api/v1/video/webcam/recognition
    """
    logger.info("Starting recognition stream: device=%d, threshold=%.2f, enroll_gait=%s",
                device, threshold, enroll_gait)

    return StreamingResponse(
        _generate_recognition_mjpeg(
            source=str(device),
            fps=fps,
            threshold=threshold,
            auto_enroll=auto_enroll,
            enroll_gait=enroll_gait,
            width=width,
            height=height,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
