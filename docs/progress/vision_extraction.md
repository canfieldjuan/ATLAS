# Vision Extraction Progress Log

**Created**: 2026-01-28
**Branch**: brain-extraction
**Status**: PLANNING

---

## Context

Atlas Brain currently runs vision/camera detection in-process, competing for GPU resources with voice pipeline (TTS/LLM). This extraction moves detection responsibilities to the existing `atlas_vision` service.

### Current State Analysis (Verified)

**atlas_brain vision code** (`atlas_brain/vision/`):
| File | Purpose | Lines | GPU Usage |
|------|---------|-------|-----------|
| `webcam_detector.py` | YOLO on local webcam | ~280 | YES |
| `rtsp_detector.py` | YOLO on RTSP cameras | ~320 | YES |
| `subscriber.py` | MQTT consumer for events | ~280 | No |
| `models.py` | VisionEvent, BoundingBox dataclasses | ~70 | No |
| `__init__.py` | Exports | ~60 | No |

**atlas_brain recognition code** (`atlas_brain/services/recognition/`):
| File | Purpose | Lines | GPU Usage |
|------|---------|-------|-----------|
| `face.py` | Face recognition (InsightFace) | ~280 | YES |
| `gait.py` | Gait recognition (MediaPipe) | ~650 | YES |
| `repository.py` | Person storage (embeddings) | ~550 | No |
| `tracker.py` | Multi-person tracking | ~260 | No |

**atlas_vision** (already exists in `atlas_video-processing/atlas_vision/`):
- Phase 1-2 complete: REST API, camera registry, motion detection
- Missing: YOLO detection, face/gait recognition, MQTT publishing

### Dependencies Map (Verified via grep)

**Who imports vision?**
```
main.py:286         → get_vision_subscriber (MQTT consumer - KEEP)
main.py:399         → start_webcam_detector (YOLO - EXTRACT)
main.py:421         → start_rtsp_cameras (YOLO - EXTRACT)
main.py:473         → stop_webcam_detector (EXTRACT)
main.py:482         → stop_rtsp_cameras (EXTRACT)
api/vision.py       → get_vision_subscriber, get_alert_manager (KEEP)
presence/camera.py  → get_vision_subscriber (KEEP)
storage/repositories → VisionEventRepository (KEEP)
```

**Who imports recognition?**
```
api/video.py:546    → face_service, gait_service, person_repository
api/video.py:777    → recognition services
api/recognition.py  → all recognition exports
```

### Key Insight

The subscriber pattern is already correct - atlas_brain **consumes** vision events via MQTT. The problem is that atlas_brain also **produces** them by running detectors locally.

**Current (broken)**:
```
atlas_brain
├── Runs YOLO detectors (webcam, RTSP) → GPU contention
├── Runs face/gait recognition → GPU contention
├── Subscribes to MQTT events → correct
└── Serves API endpoints → correct
```

**Target (extracted)**:
```
atlas_vision (separate process)
├── Runs YOLO detectors
├── Runs face/gait recognition
├── Publishes to MQTT topics
└── Serves camera/detection API

atlas_brain (orchestrator only)
├── Subscribes to MQTT events → already works
├── Serves vision API (proxies to atlas_vision)
└── No local GPU vision work
```

---

## Implementation Plan

### Phase 1: Migrate Detection Code to atlas_vision

**Goal**: Move YOLO detection from atlas_brain to atlas_vision

**Files to copy** (atlas_brain → atlas_vision):
1. `vision/webcam_detector.py` → `atlas_vision/detection/webcam.py`
2. `vision/rtsp_detector.py` → `atlas_vision/detection/rtsp.py`
3. `vision/models.py` → `atlas_vision/core/events.py` (merge with existing)

**Files to create** in atlas_vision:
1. `atlas_vision/detection/__init__.py`
2. `atlas_vision/detection/yolo.py` - Shared YOLO wrapper
3. `atlas_vision/communication/mqtt_publisher.py` - Publish events

**Validation**:
- [ ] atlas_vision starts with webcam detection
- [ ] atlas_vision publishes to `atlas/vision/+/events`
- [ ] atlas_brain subscriber receives events (no code change needed)

### Phase 2: Migrate Recognition Code to atlas_vision

**Goal**: Move face/gait recognition from atlas_brain to atlas_vision

**Files to copy**:
1. `services/recognition/face.py` → `atlas_vision/recognition/face.py`
2. `services/recognition/gait.py` → `atlas_vision/recognition/gait.py`
3. `services/recognition/tracker.py` → `atlas_vision/recognition/tracker.py`
4. `services/recognition/repository.py` → `atlas_vision/recognition/repository.py`

**Files to create** in atlas_vision:
1. `atlas_vision/recognition/__init__.py`
2. `atlas_vision/api/recognition.py` - Recognition endpoints

**Validation**:
- [ ] Face enrollment works via atlas_vision API
- [ ] Gait enrollment works via atlas_vision API
- [ ] Person identification events published to MQTT

### Phase 3: Update atlas_brain to Consume Only

**Goal**: Remove detection code from atlas_brain, keep subscriber

**Files to modify** in atlas_brain:

1. **main.py** - Remove detector startup
   - Line 395-414: Remove webcam_detector block
   - Line 416-429: Remove rtsp_manager block
   - Line 471-478: Remove webcam stop
   - Line 480-486: Remove rtsp stop

2. **api/video.py** - Proxy to atlas_vision
   - Change `AtlasVisionFrameSource` to use atlas_vision API (already does!)
   - Update streaming endpoints to fetch from atlas_vision

3. **api/recognition.py** - Proxy to atlas_vision
   - Keep API contract same
   - Forward requests to atlas_vision

**Files to KEEP unchanged** in atlas_brain:
- `vision/subscriber.py` - Still consumes MQTT events
- `vision/models.py` - Event dataclasses still needed
- `storage/repositories/vision.py` - Event storage
- `api/vision.py` - Event query endpoints (use subscriber)
- `presence/camera.py` - Consumes from subscriber

**Files to DELETE** from atlas_brain (after validation):
- `vision/webcam_detector.py`
- `vision/rtsp_detector.py`
- `services/recognition/` (entire directory)

**Validation**:
- [ ] atlas_brain starts without GPU vision load
- [ ] Vision events still flow via MQTT
- [ ] Presence detection still works
- [ ] Recognition API still works (via proxy)
- [ ] No breaking changes to existing functionality

### Phase 4: Configuration Updates

**atlas_vision** environment:
```bash
ATLAS_VISION_MQTT_HOST=localhost
ATLAS_VISION_MQTT_PORT=1883
ATLAS_VISION_WEBCAM_ENABLED=true
ATLAS_VISION_WEBCAM_DEVICE=0
ATLAS_VISION_YOLO_MODEL=yolov8n.pt
```

**atlas_brain** environment:
```bash
# Remove these (no longer needed)
# ATLAS_WEBCAM_ENABLED=true
# ATLAS_RTSP_ENABLED=true

# Add these
ATLAS_VISION_URL=http://localhost:5002  # Already exists as security.video_processing_url
```

---

## Files Affected Summary

### atlas_brain - MODIFY
| File | Change Type | Risk |
|------|-------------|------|
| `main.py` | Remove detector startup/shutdown | Low |
| `api/video.py` | Minor - already proxies | Low |
| `api/recognition.py` | Proxy to atlas_vision | Medium |
| `vision/__init__.py` | Remove detector exports | Low |
| `config.py` | Remove webcam/rtsp configs | Low |

### atlas_brain - DELETE (after validation)
| File | Reason |
|------|--------|
| `vision/webcam_detector.py` | Moved to atlas_vision |
| `vision/rtsp_detector.py` | Moved to atlas_vision |
| `services/recognition/` | Moved to atlas_vision |

### atlas_brain - KEEP UNCHANGED
| File | Reason |
|------|--------|
| `vision/subscriber.py` | MQTT consumer - correct role |
| `vision/models.py` | Event dataclasses needed |
| `presence/camera.py` | Uses subscriber correctly |
| `api/vision.py` | Queries subscriber/storage |
| `storage/repositories/vision.py` | Event persistence |
| `alerts/` | Uses vision events correctly |

### atlas_vision - CREATE/MODIFY
| File | Change Type |
|------|-------------|
| `detection/webcam.py` | New - from atlas_brain |
| `detection/rtsp.py` | New - from atlas_brain |
| `detection/yolo.py` | New - shared YOLO wrapper |
| `recognition/face.py` | New - from atlas_brain |
| `recognition/gait.py` | New - from atlas_brain |
| `recognition/tracker.py` | New - from atlas_brain |
| `recognition/repository.py` | New - from atlas_brain |
| `communication/mqtt_publisher.py` | New - event publishing |
| `api/recognition.py` | New - recognition endpoints |
| `core/events.py` | Update - merge event models |

---

## Rollback Plan

If extraction causes issues:
1. atlas_brain code is still in git (just disabled via config)
2. Set `ATLAS_WEBCAM_ENABLED=true` to restore old behavior
3. Set `ATLAS_VISION_URL=""` to disable proxy

---

## Session Log

### 2026-01-28 - Planning Session

1. Created worktree `Atlas-brain-extraction` on branch `brain-extraction`
2. Analyzed vision code dependencies via grep
3. Discovered atlas_vision already has Phase 1-2 complete (REST API, motion detection)
4. Identified exact files to migrate vs keep vs delete
5. Created this implementation plan

**Key Findings**:
- Subscriber pattern already correct (atlas_brain consumes, not produces)
- atlas_vision REST API already exists and is tested
- api/video.py already has `AtlasVisionFrameSource` that fetches from atlas_vision
- Recognition needs to be migrated (face.py, gait.py, tracker.py, repository.py)

### 2026-01-28 - Phase 1 Implementation COMPLETED

**Changes Made**:

1. **atlas_brain/main.py** - Removed local detector startup/shutdown
   - Removed webcam_detector startup block (lines 395-414)
   - Removed rtsp_manager startup block (lines 416-433)
   - Removed webcam_detector shutdown block (lines 470-477)
   - Removed rtsp_manager shutdown block (lines 479-486)
   - Added comment explaining detection moved to atlas_vision

2. **atlas_brain/vision/__init__.py** - Updated exports
   - Removed imports from webcam_detector.py and rtsp_detector.py
   - Kept imports for subscriber.py, models.py, and alerts
   - Added __getattr__ for deprecation notices on removed functions

**Validation**:
- [x] main.py syntax valid (py_compile passed)
- [x] vision/__init__.py syntax valid (py_compile passed)
- [x] Vision module imports work (VisionSubscriber, VisionEvent, etc.)
- [x] Deprecation notice works for removed functions
- [x] No other files import removed functions

**Files NOT deleted yet** (waiting for full validation):
- atlas_brain/vision/webcam_detector.py
- atlas_brain/vision/rtsp_detector.py

**Next Steps**:
- Configure atlas_vision to run with MQTT enabled
- Register webcam via atlas_vision API
- Verify end-to-end event flow
- Delete detector files after full validation

---

## Exact Code Changes (Pending Approval)

### Phase 1: main.py Changes

**File**: `atlas_brain/main.py`

**REMOVE** Lines 395-414 (webcam detector startup):
```python
    # Start webcam person detector if enabled
    webcam_detector = None
    if settings.webcam.enabled:
        try:
            from .vision import start_webcam_detector
            # ... 15 lines
        except Exception as e:
            logger.error("Failed to start webcam detector: %s", e)
```

**REMOVE** Lines 416-433 (RTSP detector startup):
```python
    # Start RTSP camera detectors if enabled
    rtsp_manager = None
    if settings.rtsp.enabled:
        try:
            # ... 14 lines
        except Exception as e:
            logger.error("Failed to start RTSP cameras: %s", e)
```

**REMOVE** Lines 470-486 (detector shutdown):
```python
    # Stop webcam detector
    if webcam_detector:
        # ... 8 lines

    # Stop RTSP camera detectors
    if rtsp_manager:
        # ... 8 lines
```

### Phase 1: config.py Changes

**File**: `atlas_brain/config.py`

**DEPRECATE** (not delete, just mark deprecated):
- Lines 515-524: `WebcamConfig` - add deprecation note
- Lines 536-546: `RTSPConfig` - add deprecation note

**ADD** to `SecurityConfig` (lines 549-573):
```python
    vision_service_url: str = Field(
        default="http://localhost:5002",
        alias="video_processing_url",  # backwards compatible
        description="Atlas Vision service URL"
    )
```

### Phase 1: vision/__init__.py Changes

**File**: `atlas_brain/vision/__init__.py`

**REMOVE** exports (lines 19-31):
```python
from .webcam_detector import (...)
from .rtsp_detector import (...)
```

**KEEP** exports (lines 10-18):
```python
from ..alerts import AlertManager, AlertRule, get_alert_manager
from .models import BoundingBox, EventType, NodeStatus, VisionEvent
from .subscriber import (...)
```

### Phase 2: Files to Copy to atlas_vision

| Source (atlas_brain) | Destination (atlas_vision) |
|----------------------|---------------------------|
| `vision/webcam_detector.py` | `detection/webcam.py` |
| `vision/rtsp_detector.py` | `detection/rtsp.py` |
| `services/recognition/face.py` | `recognition/face.py` |
| `services/recognition/gait.py` | `recognition/gait.py` |
| `services/recognition/tracker.py` | `recognition/tracker.py` |
| `services/recognition/repository.py` | `recognition/repository.py` |

### Phase 3: Files to Delete from atlas_brain (AFTER validation)

| File | Validation Required |
|------|---------------------|
| `vision/webcam_detector.py` | atlas_vision serving detections |
| `vision/rtsp_detector.py` | atlas_vision serving detections |
| `services/recognition/face.py` | atlas_vision serving recognition |
| `services/recognition/gait.py` | atlas_vision serving recognition |
| `services/recognition/tracker.py` | atlas_vision serving tracking |
| `services/recognition/repository.py` | atlas_vision serving person data |

---

## Validation Checklist

### Before Any Changes
- [ ] atlas_vision service starts on port 5002
- [ ] atlas_vision REST API responds to `/health`
- [ ] MQTT broker running and accessible

### After Phase 1
- [ ] atlas_brain starts without detector code
- [ ] No errors about missing webcam/rtsp imports
- [ ] Vision subscriber still receives MQTT events
- [ ] Presence service still works (if enabled)

### After Phase 2
- [ ] atlas_vision detects persons on webcam
- [ ] atlas_vision publishes to `atlas/vision/+/events`
- [ ] atlas_brain subscriber receives events
- [ ] Recognition API works via proxy

### After Phase 3 (File Deletion)
- [ ] No orphaned imports anywhere
- [ ] All tests pass
- [ ] Full end-to-end flow works

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| MQTT events stop flowing | Keep subscriber unchanged, only remove producers |
| Recognition API breaks | Proxy to atlas_vision, same contract |
| Presence detection breaks | Subscriber unchanged, consumes from atlas_vision |
| Rollback needed | Config flags can re-enable, code still in git |

---

## Next Steps (Pending Approval)

1. [ ] Copy detection code to atlas_vision
2. [ ] Add MQTT publisher to atlas_vision
3. [ ] Test atlas_vision produces events
4. [ ] Modify atlas_brain main.py (remove startup code)
5. [ ] Verify end-to-end flow
6. [ ] Copy recognition code to atlas_vision
7. [ ] Update atlas_brain recognition API to proxy
8. [ ] Delete migrated files from atlas_brain
