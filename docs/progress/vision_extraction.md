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

### Phase 2 Assessment

**Recognition API Analysis**:
- 20+ endpoints in `api/recognition.py`
- Uses `cv2.VideoCapture` directly for frame capture
- Stores embeddings in PostgreSQL database
- GPU usage: InsightFace (face) + MediaPipe (gait) - lighter than YOLO

**Decision**: Recognition migration is LOWER priority because:
1. InsightFace/MediaPipe are lighter than YOLO
2. Database access would need to be shared/proxied
3. Recognition is about WHO, not IF (detection)
4. Main GPU contention was YOLO detection (now in atlas_vision)

### 2026-01-28 - Phase 2 Implementation COMPLETED

**Recognition migrated to atlas_vision**:

**Files created in atlas_vision**:
- `storage/__init__.py` - Database module exports
- `storage/config.py` - Database settings (ATLAS_VISION_DB_*)
- `storage/database.py` - Connection pool
- `recognition/__init__.py` - Recognition module exports
- `recognition/repository.py` - Person/embedding storage
- `recognition/face.py` - Face recognition service
- `recognition/gait.py` - Gait recognition service
- `recognition/tracker.py` - Multi-person tracking
- `api/recognition.py` - REST API endpoints

**Files updated in atlas_vision**:
- `core/config.py` - Added RecognitionConfig
- `api/main.py` - Added database init/close, recognition router
- `.env.example` - Added DB and recognition settings

**API Endpoints** (atlas_vision /recognition/):
- `POST /persons` - Create person
- `GET /persons` - List persons
- `GET /persons/{id}` - Get person
- `DELETE /persons/{id}` - Delete person
- `GET /persons/{id}/embeddings` - Get embedding counts
- `GET /events` - Get recognition events

**Next Steps**:
- Update atlas_brain api/recognition.py to proxy to atlas_vision
- Delete atlas_brain services/recognition/ after validation
- Test full integration

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

---

## 2026-01-28 - Missing Recognition API Endpoints

### Gap Analysis

The atlas_vision `api/recognition.py` has 6 endpoints implemented.
The atlas_brain proxy expects 18 total endpoints.

**12 MISSING ENDPOINTS** that must be added to atlas_vision:

### Group 1: Person Update (1 endpoint)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/persons/{person_id}` | PATCH | Update person name, is_known, metadata |

### Group 2: Face Enrollment/Identification (2 endpoints)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/enroll/face` | POST | Capture frame, detect face, enroll embedding |
| `/identify/face` | POST | Capture frame, detect face, match against DB |

### Group 3: Gait Enrollment (5 endpoints)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/enroll/gait/start` | POST | Start gait enrollment for person_id |
| `/enroll/gait/frame` | POST | Capture frame, add pose to buffer |
| `/enroll/gait/complete` | POST | Finalize enrollment from buffer |
| `/enroll/gait/status` | GET | Get current enrollment buffer status |
| `/enroll/gait/cancel` | POST | Cancel ongoing enrollment |

### Group 4: Gait Identification (3 endpoints)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/identify/gait/start` | POST | Clear buffer for new identification |
| `/identify/gait/frame` | POST | Capture frame, add pose to buffer |
| `/identify/gait/match` | POST | Match collected poses against DB |

### Group 5: Combined Identification (1 endpoint)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/identify/combined` | POST | Face + gait combined identification |

### Implementation File

**Target**: `atlas_video-processing/atlas_vision/api/recognition.py`

**Current state**: 221 lines, ends at line 221

**Insertion point**: After line 220 (end of get_recognition_events function)

### Dependencies Required

1. Frame capture from camera registry
2. Face service (get_face_service)
3. Gait service (get_gait_service)
4. Config settings for thresholds

### Implementation Order

1. PATCH /persons/{person_id} - simplest, just DB update
2. POST /enroll/face - uses face service + camera
3. POST /identify/face - uses face service + camera
4. Gait enrollment group (5 endpoints)
5. Gait identification group (3 endpoints)
6. POST /identify/combined - uses both services

### 2026-01-28 - Missing Endpoints IMPLEMENTED

**All 12 missing endpoints added to atlas_vision/api/recognition.py**

**Imports added**:
- `from ..core.config import settings`
- `from ..devices.registry import device_registry`

**Request models added**:
- `UpdatePersonRequest` - for PATCH /persons/{id}
- `EnrollFaceRequest` - for POST /enroll/face
- `IdentifyRequest` - for POST /identify/face
- `StartGaitEnrollRequest` - for POST /enroll/gait/start
- `GaitIdentifyRequest` - for POST /identify/gait/match
- `CombinedIdentifyRequest` - for POST /identify/combined

**Helper functions added**:
- `_check_recognition_enabled()` - validates recognition is enabled
- `_get_camera_frame(camera_id)` - fetches frame from device registry

**Module-level state added**:
- `_gait_enrollment_state` - tracks active gait enrollment session

**Endpoints implemented** (all 18 total):
1. POST /persons - Create person
2. GET /persons - List persons
3. GET /persons/{id} - Get person
4. DELETE /persons/{id} - Delete person
5. PATCH /persons/{id} - Update person (NEW)
6. GET /persons/{id}/embeddings - Get embedding counts
7. GET /events - Get recognition events
8. POST /enroll/face - Face enrollment (NEW)
9. POST /identify/face - Face identification (NEW)
10. POST /enroll/gait/start - Start gait enrollment (NEW)
11. POST /enroll/gait/frame - Add gait frame (NEW)
12. POST /enroll/gait/complete - Complete gait enrollment (NEW)
13. GET /enroll/gait/status - Get gait enrollment status (NEW)
14. POST /enroll/gait/cancel - Cancel gait enrollment (NEW)
15. POST /identify/gait/start - Start gait identification (NEW)
16. POST /identify/gait/frame - Add gait identify frame (NEW)
17. POST /identify/gait/match - Match gait (NEW)
18. POST /identify/combined - Combined face+gait identification (NEW)

**Validation**:
- [x] Syntax valid (py_compile passed)
- [x] Imports work (atlas_vision.api.recognition imports successfully)
- [x] 18 routes registered on router

**File stats**:
- Original: 221 lines
- After changes: 768 lines
- Lines added: 547

### 2026-01-28 - Recognition Streaming Migration COMPLETED

**Goal**: Migrate recognition streaming endpoints from atlas_brain to atlas_vision

**Analysis**:
- atlas_brain/api/video.py had two recognition streaming functions:
  - `_generate_recognition_mjpeg` (~180 lines) - single person
  - `_generate_multitrack_recognition_mjpeg` (~300 lines) - multi-person with YOLO
- Both used `services/recognition/` imports (GPU work in atlas_brain)
- Consolidated into ONE endpoint in atlas_vision

**Changes in atlas_vision**:

1. **api/cameras.py** - Added consolidated endpoint:
   - `GET /cameras/{camera_id}/stream/recognition/full`
   - Uses YOLO ByteTrack for multi-person tracking
   - Face recognition per track
   - Gait collection and recognition per track
   - Auto-enrollment support
   - Visual overlays (green=identified, gray=tracking, orange=new)

**Changes in atlas_brain**:

1. **api/video.py** - Removed recognition streaming:
   - Deleted `_generate_recognition_mjpeg` function
   - Deleted `_generate_multitrack_recognition_mjpeg` function
   - Replaced endpoints with deprecation notices (HTTP 410)
   - Deprecation includes redirect URL to atlas_vision
   - File reduced from 1100 lines to 592 lines

2. **services/recognition/** - DELETED:
   - face.py
   - gait.py
   - repository.py
   - tracker.py
   - __init__.py

**Validation**:
- [x] atlas_vision api/cameras.py syntax valid
- [x] atlas_brain api/video.py syntax valid
- [x] No orphan imports from services/recognition
- [x] services/recognition/ deleted

**Result**:
- atlas_brain no longer has ANY GPU vision/recognition code
- All vision GPU work consolidated in atlas_vision
- Plain video streaming still works via AtlasVisionFrameSource proxy

### 2026-01-28 - Config Deprecation COMPLETED

**Goal**: Mark deprecated configs in atlas_brain/config.py

**Changes**:

1. **WebcamConfig** (line 515):
   - Added deprecation notice in docstring
   - Points to `POST /cameras/register/webcam` in atlas_vision
   - All fields marked DEPRECATED

2. **RTSPCameraConfig** (line 527):
   - Added deprecation notice in docstring
   - Points to `POST /cameras/register` in atlas_vision

3. **RTSPConfig** (line 536):
   - Added deprecation notice in docstring
   - Points to `POST /cameras/register` in atlas_vision
   - All fields marked DEPRECATED

**Validation**:
- [x] config.py syntax valid

---

## Final Status

### Phase 1: Detection Migration - COMPLETE
- [x] Remove detector startup from main.py
- [x] Delete webcam_detector.py
- [x] Delete rtsp_detector.py
- [x] Update vision/__init__.py

### Phase 2: Recognition Migration - COMPLETE
- [x] Copy face.py, gait.py, tracker.py, repository.py to atlas_vision
- [x] Create api/recognition.py with 18 endpoints
- [x] Add storage module for database access

### Phase 3: atlas_brain Cleanup - COMPLETE
- [x] Convert api/recognition.py to proxy
- [x] Remove recognition streaming from api/video.py
- [x] Delete services/recognition/ directory
- [x] Mark WebcamConfig deprecated
- [x] Mark RTSPConfig deprecated

### Phase 4: Consolidated Streaming - COMPLETE
- [x] Add /cameras/{id}/stream/recognition/full to atlas_vision
- [x] Consolidates single and multi-track recognition

---

## Commits Summary

| Commit | Description |
|--------|-------------|
| b240c64 | Remove local detection from main.py |
| 3d1eb8b | Delete migrated detector files |
| 9ba5a5a | Migrate recognition to atlas_vision |
| 6b55797 | Add missing API endpoints (18 total) |
| 3fc6fa0 | Consolidate recognition streaming |
| 9402d39 | Mark config classes deprecated |

---

## GPU Workload Distribution (Final)

**atlas_brain** (orchestrator):
- Voice pipeline (STT/TTS/LLM)
- API routing and proxying
- MQTT event subscription
- No vision GPU work

**atlas_vision** (vision service):
- YOLO person detection
- Face recognition (InsightFace)
- Gait recognition (MediaPipe)
- Recognition streaming
- Camera management
