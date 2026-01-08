"""
Audio event detection API endpoints.

Provides endpoints for:
- Classifying audio for events
- Managing continuous monitoring
- Querying event history
"""

import base64
import logging
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..orchestration.context import get_context
from ..services.audio_events import (
    INTERESTING_EVENTS,
    AudioMonitor,
    AudioMonitorConfig,
    MonitoredEvent,
    get_audio_monitor,
    set_audio_monitor,
)
from ..services.registry import audio_events_registry

logger = logging.getLogger("atlas.api.audio_events")

router = APIRouter(prefix="/audio-events", tags=["audio-events"])


class ClassifyRequest(BaseModel):
    """Request for audio classification."""

    audio_base64: str
    top_k: int = 5
    min_confidence: float = 0.1
    interesting_only: bool = False


class ClassifyResponse(BaseModel):
    """Response from audio classification."""

    events: list[dict]
    processing_ms: float


class MonitorConfig(BaseModel):
    """Configuration for audio monitoring."""

    enabled: bool = True
    location: Optional[str] = None
    min_confidence: float = 0.3
    interesting_only: bool = True
    event_cooldown_seconds: float = 5.0


@router.get("/")
async def get_audio_events_status():
    """Get audio event detection status."""
    active = audio_events_registry.get_active()
    monitor = get_audio_monitor()

    return {
        "classifier": {
            "available": audio_events_registry.list_available(),
            "active": active.model_info.to_dict() if active else None,
        },
        "monitor": {
            "running": monitor._running if monitor else False,
            "location": monitor.location if monitor else None,
        },
        "interesting_events": list(INTERESTING_EVENTS.keys()),
    }


@router.post("/activate")
async def activate_classifier(body: dict):
    """
    Activate an audio event classifier.

    Request body: {"name": "yamnet"}
    """
    name = body.get("name", "yamnet")

    try:
        classifier = audio_events_registry.activate(name)
        return {
            "status": "ok",
            "message": f"Audio classifier '{name}' activated",
            "model": classifier.model_info.to_dict(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to activate audio classifier")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deactivate")
async def deactivate_classifier():
    """Deactivate the current audio classifier."""
    audio_events_registry.deactivate()
    return {"status": "ok", "message": "Audio classifier deactivated"}


@router.post("/classify")
async def classify_audio(request: ClassifyRequest):
    """
    Classify audio for events.

    Accepts base64-encoded audio (WAV or raw PCM).
    """
    classifier = audio_events_registry.get_active()
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="No audio classifier active. Call /activate first.",
        )

    try:
        audio_bytes = base64.b64decode(request.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")

    from ..services.base import InferenceTimer

    with InferenceTimer() as timer:
        if request.interesting_only:
            events = classifier.get_interesting_events(
                audio_bytes,
                min_confidence=request.min_confidence,
            )
        else:
            events = classifier.classify(
                audio_bytes,
                top_k=request.top_k,
                min_confidence=request.min_confidence,
            )

    # Update context with detected events
    context = get_context()
    for event in events:
        context.add_audio_event(
            label=event.label,
            confidence=event.confidence,
        )

    return {
        "events": [e.to_dict() for e in events],
        "processing_ms": timer.duration * 1000,
    }


@router.post("/classify/file")
async def classify_audio_file(
    audio_file: UploadFile = File(...),
    top_k: int = 5,
    min_confidence: float = 0.1,
    interesting_only: bool = False,
):
    """
    Classify an uploaded audio file for events.
    """
    classifier = audio_events_registry.get_active()
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="No audio classifier active. Call /activate first.",
        )

    audio_bytes = await audio_file.read()

    from ..services.base import InferenceTimer

    with InferenceTimer() as timer:
        if interesting_only:
            events = classifier.get_interesting_events(
                audio_bytes,
                min_confidence=min_confidence,
            )
        else:
            events = classifier.classify(
                audio_bytes,
                top_k=top_k,
                min_confidence=min_confidence,
            )

    # Update context
    context = get_context()
    for event in events:
        context.add_audio_event(
            label=event.label,
            confidence=event.confidence,
        )

    return {
        "filename": audio_file.filename,
        "events": [e.to_dict() for e in events],
        "processing_ms": timer.duration * 1000,
    }


@router.post("/monitor/start")
async def start_monitor(config: MonitorConfig):
    """
    Start continuous audio monitoring.

    Note: This creates an AudioMonitor but does NOT capture from microphone
    by default (requires pyaudio and system access). Audio must be pushed
    via the /monitor/feed endpoint or WebSocket.
    """
    classifier = audio_events_registry.get_active()
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="No audio classifier active. Activate one first.",
        )

    # Create monitor
    monitor_config = AudioMonitorConfig(
        min_confidence=config.min_confidence,
        interesting_only=config.interesting_only,
        event_cooldown_seconds=config.event_cooldown_seconds,
    )

    monitor = AudioMonitor(config=monitor_config, location=config.location)

    # Set up context integration
    context = get_context()

    def on_event(monitored_event: MonitoredEvent):
        context.add_audio_event(
            label=monitored_event.event.label,
            confidence=monitored_event.event.confidence,
            location=monitored_event.location,
        )

    monitor.set_event_callback(on_event)

    # Start the monitor
    await monitor.start()
    set_audio_monitor(monitor)

    return {
        "status": "ok",
        "message": "Audio monitor started",
        "location": config.location,
    }


@router.post("/monitor/stop")
async def stop_monitor():
    """Stop continuous audio monitoring."""
    monitor = get_audio_monitor()
    if monitor:
        await monitor.stop()
        set_audio_monitor(None)
        return {"status": "ok", "message": "Audio monitor stopped"}

    return {"status": "ok", "message": "No monitor was running"}


@router.post("/monitor/feed")
async def feed_audio(request: dict):
    """
    Feed audio data to the monitor for processing.

    Request body: {"audio_base64": "..."}
    """
    monitor = get_audio_monitor()
    if monitor is None or not monitor._running:
        raise HTTPException(
            status_code=400,
            detail="Monitor not running. Start it first.",
        )

    audio_base64 = request.get("audio_base64", "")
    if not audio_base64:
        raise HTTPException(status_code=400, detail="No audio data provided")

    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data")

    monitor.add_audio(audio_bytes)

    return {"status": "ok", "bytes_added": len(audio_bytes)}


@router.get("/monitor/events")
async def get_monitor_events(
    seconds: int = 60,
    priority: Optional[str] = None,
):
    """Get recent events from the monitor."""
    monitor = get_audio_monitor()
    if monitor is None:
        return {"events": [], "message": "No monitor active"}

    events = monitor.get_recent_events(seconds=seconds, priority=priority)

    return {
        "events": [
            {
                "label": e.event.label,
                "confidence": e.event.confidence,
                "timestamp": e.timestamp.isoformat(),
                "location": e.location,
                "priority": e.priority,
            }
            for e in events
        ],
        "count": len(events),
    }


@router.get("/classes")
async def get_event_classes():
    """Get all available event classes from the classifier."""
    classifier = audio_events_registry.get_active()
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="No audio classifier active.",
        )

    try:
        classes = classifier.get_all_class_names()
        return {
            "count": len(classes),
            "classes": classes,
        }
    except AttributeError:
        return {
            "count": 0,
            "classes": [],
            "message": "Classifier does not expose class names",
        }


@router.get("/interesting")
async def get_interesting_events_list():
    """Get the list of events considered 'interesting' for home automation."""
    return {
        "events": [
            {
                "label": label,
                "priority": info["priority"],
                "action": info["action"],
            }
            for label, info in INTERESTING_EVENTS.items()
        ]
    }
