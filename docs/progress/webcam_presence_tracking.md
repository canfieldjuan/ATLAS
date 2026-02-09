# Webcam Presence Tracking Implementation

## Status: Complete
## Created: 2026-01-20
## Last Updated: 2026-01-20

---

## Overview

Enable room-aware device control using USB webcam for person detection.
When user says "turn on the lights", Atlas determines which room based on
camera detection and controls the appropriate devices.

---

## Architecture

```
USB Webcam (/dev/video0)
       |
       v
WebcamPersonDetector (vision/webcam_detector.py)
  - Runs YOLO person detection
  - Feeds detections to PresenceService
       |
       v
PresenceService (presence/service.py)
  - Maintains room state per user
  - Maps camera_source_id -> room
       |
       v
Presence Tools (tools/presence.py)
  - lights_near_user, media_near_user, etc.
  - Resolves "the lights" to specific entity_ids
       |
       v
Home Assistant
  - Controls actual devices
```

---

## Audit Results

### Two Camera Detection Paths Found

1. **WebcamPersonDetector** (`vision/webcam_detector.py`)
   - Direct USB webcam detection using YOLO
   - No MQTT required
   - Simpler setup for local webcams
   - **CANONICAL for USB webcams**

2. **CameraPresenceConsumer** (`presence/camera.py`)
   - Consumes events from VisionSubscriber (MQTT-based)
   - Designed for network cameras (Frigate, Wyze, RTSP)
   - More complex but supports multiple camera sources
   - **KEEP for network cameras later**

### Conclusion
No duplicates - these serve different purposes. Use WebcamPersonDetector for USB webcam.

---

## Current Configuration

### .env Settings (Current)
```
ATLAS_PRESENCE_ENABLED=true
ATLAS_PRESENCE_CAMERA_ENABLED=false  # VisionSubscriber/MQTT path - not needed
ATLAS_WEBCAM_ENABLED=true            # Direct USB webcam - ENABLED
ATLAS_WEBCAM_DEVICE_INDEX=0
ATLAS_WEBCAM_DEVICE_NAME=USB Camera  # For reliable device resolution
ATLAS_WEBCAM_SOURCE_ID=webcam_office
ATLAS_WEBCAM_FPS=5                   # Low FPS saves CPU
```

### Room Configuration (presence/config.py)
Office room already configured:
```python
RoomConfig(
    id="office",
    name="Office",
    camera_sources=["webcam_office"],  # Matches ATLAS_WEBCAM_SOURCE_ID
    switches=["input_boolean.office_light", "input_boolean.test_light"],
)
```

### Hardware Detected
- `/dev/video0`: USB 2.0 Camera
- `/dev/video1`: USB 2.0 Camera (secondary interface)

---

## Implementation Plan

### Phase 1: Enable Webcam Detection
1. Update .env to enable webcam
2. Set appropriate FPS (5 is enough, saves CPU)
3. Restart server and verify detection starts

### Phase 2: Verify Presence Service Integration
1. Test that person detection updates PresenceService
2. Verify room state changes when person detected/lost
3. Check logs for proper flow

### Phase 3: Test Room-Aware Tools
1. Test `where_am_i` tool to verify detection
2. Test `lights_near_user` to verify device resolution
3. Verify HA commands are sent correctly

### Phase 4: Production Tuning
1. Adjust confidence thresholds if needed
2. Tune room_enter_threshold and hysteresis
3. Add device_name for reliable device resolution

---

## Changes Made

### 2026-01-20: Initial Setup
- [x] Enable ATLAS_WEBCAM_ENABLED=true in .env
- [x] Set ATLAS_WEBCAM_FPS=5 for CPU efficiency
- [x] Add ATLAS_WEBCAM_DEVICE_NAME=USB Camera for reliable device detection
- [x] Verify webcam detector starts on server startup
- [x] Test presence detection flow

### 2026-01-20: Bug Fixes
- [x] Fixed WebcamConfig not reading .env (missing env_file in model_config)
- [x] Fixed hysteresis timeout by sending periodic detection updates
- [x] Added webcam/presence status to /health endpoint for debugging
- [x] Fixed intent routing: "what room am I in" now routes to where_am_i tool
  - Added keyword override in intent_router.py to catch room queries
  - Added where_am_i to PARAMETERLESS_TOOLS for fast path

---

## Testing Commands

```bash
# Check if webcam detector is running
curl http://localhost:8000/api/v1/health

# Test where_am_i tool
curl -X POST http://localhost:8000/api/v1/query/text \
  -H "Content-Type: application/json" \
  -d '{"query_text": "where am I?"}'

# Test lights_near_user (once you have lights)
curl -X POST http://localhost:8000/api/v1/query/text \
  -H "Content-Type: application/json" \
  -d '{"query_text": "turn on the lights"}'
```

---

## Related Files

- `atlas_brain/vision/webcam_detector.py` - WebcamPersonDetector class
- `atlas_brain/presence/service.py` - PresenceService class
- `atlas_brain/presence/config.py` - Room configuration
- `atlas_brain/tools/presence.py` - Room-aware tools (where_am_i, lights_near_user)
- `atlas_brain/services/intent_router.py` - Keyword override routing for room queries
- `atlas_brain/config.py` - WebcamConfig class
- `atlas_brain/api/health.py` - Health endpoint with webcam/presence status
- `atlas_brain/main.py` - Startup/shutdown hooks

---

## Notes

- USB webcams use direct YOLO detection, not MQTT path
- FPS of 5 is sufficient for presence detection
- Higher FPS uses more CPU/GPU without benefit
- Room mapping uses camera_source_id to match room config
