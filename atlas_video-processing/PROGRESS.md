# Atlas Vision Progress Log

## Session: 2026-01-14

### Current State Analysis

**Codebase Status**: ~2% implemented (confirmed)

#### Existing Files
| File | Status | Notes |
|------|--------|-------|
| `ingest/drone_client/drone_client.py` | Partial | Kafka producer, simulated data only |
| `processing/video_stream_processor/video_stream_processor.py` | Hollow | Kafka consumer, no actual CV processing |
| `docker-compose.yml` | Present | Kafka + Zookeeper + PostgreSQL, hardcoded creds |

#### Empty Directories (structure only)
- `ingest/camera_client/` - No files
- `processing/object_detection/` - No files
- `control/mission_planner/` - No files
- `control/fleet_management/` - No files
- `persistence/storage/` - No files
- `persistence/database/` - No files
- `llm_service/` - No files

#### Critical Issues Found
1. **No REST API** - atlas_brain/tools/security.py expects:
   - `GET /cameras` - List cameras
   - `GET /cameras/{id}` - Camera status
   - `GET /detections/current` - Current detections
   - `GET /events` - Motion/detection events
   - `POST /security/arm` - Arm zone
   - `POST /security/disarm` - Disarm zone
   - `POST /cameras/{id}/record` - Recording control
   - `POST /query` - LLM service queries (port 5004)

2. **Hardcoded values**:
   - `localhost:9092` in both Python files (breaks Docker networking)
   - PostgreSQL credentials in docker-compose.yml

3. **No discovery** - Nodes cannot announce themselves to Atlas Brain

4. **Legacy patterns**:
   - Print statements instead of logging
   - Old Kafka API version (0,10,1)
   - No graceful shutdown
   - No configuration management

### Integration Points with atlas_brain

**atlas_brain expects**:
- Video processing service on port 5002
- LLM service on port 5004
- security.py tools call these endpoints

**atlas_brain has**:
- Agent framework (BaseAgent, protocols) - we should mirror this
- Discovery service (SSDP scanner) - we should announce via mDNS
- Capability registry - nodes should register as capabilities

---

## Implementation Plan

### Decision: Remove Kafka, Use MQTT + Direct RTSP

**Rationale**:
- Kafka is overkill for home/small deployment
- MQTT is already used by Home Assistant
- RTSP is standard for IP cameras
- Reduces complexity by ~80%

### Phase 1: Core Foundation (THIS SESSION)

**Goal**: Create working structure with REST API that atlas_brain can call

#### 1.1 Project Restructure
- Rename to `atlas_vision` (cleaner naming)
- Create proper Python package structure
- Add pyproject.toml for modern packaging

#### 1.2 Core Module
```
atlas_vision/core/
  __init__.py
  config.py       - Pydantic settings (mirror atlas_brain pattern)
  protocols.py    - Shared dataclasses and protocols
  constants.py    - Constants and enums
```

#### 1.3 REST API (Priority - unblocks atlas_brain)
```
atlas_vision/api/
  __init__.py
  main.py         - FastAPI app entry point
  health.py       - /health, /info endpoints
  cameras.py      - Camera endpoints
  detections.py   - Detection endpoints
  security.py     - Arm/disarm endpoints
```

#### 1.4 Device Registry (Foundation for cameras/drones)
```
atlas_vision/devices/
  __init__.py
  protocols.py    - Device capability protocol
  registry.py     - Local device registry
```

### Phase 2: Camera Support

#### 2.1 Camera Adapters
```
atlas_vision/devices/cameras/
  __init__.py
  base.py         - BaseCameraCapability
  rtsp.py         - RTSPCamera (IP cameras)
  mock.py         - MockCamera (testing)
```

#### 2.2 Frame Processing
```
atlas_vision/processing/
  __init__.py
  frame_buffer.py - Efficient frame handling
```

### Phase 3: Detection & Agents

#### 3.1 Detection Module
```
atlas_vision/processing/detection/
  __init__.py
  base.py         - BaseDetector
  yolo.py         - YOLODetector
  motion.py       - MotionDetector
```

#### 3.2 Node Agents (mirror atlas_brain)
```
atlas_vision/agents/
  __init__.py
  base.py         - BaseNodeAgent
  protocols.py    - NodeContext, NodeResult
  detection.py    - DetectionAgent
```

### Phase 4: Discovery & Communication

#### 4.1 Discovery
```
atlas_vision/communication/
  __init__.py
  announcer.py    - mDNS service announcement
  mqtt.py         - MQTT client (telemetry/events)
```

#### 4.2 Atlas Brain Integration
- Add MDNSScanner to atlas_brain/discovery/scanners/
- Auto-register discovered nodes

### Phase 5: Drones & Vehicles

#### 5.1 Mobile Device Support
```
atlas_vision/devices/drones/
  __init__.py
  base.py         - BaseDroneCapability
  mavlink.py      - MAVLink protocol support
```

---

## Files Affected in atlas_brain

**Will need updates (Phase 4)**:
1. `atlas_brain/discovery/scanners/__init__.py` - Add mDNS scanner export
2. `atlas_brain/discovery/service.py` - Enable mDNS scanner
3. `atlas_brain/config.py` - Add mDNS discovery settings

**No changes needed (already prepared)**:
- `atlas_brain/tools/security.py` - Already expects REST API on :5002/:5004

---

## Session Notes

### 2026-01-14 - Initial Analysis
- Explored full codebase structure
- Identified all empty directories
- Read existing Python files (drone_client.py, video_stream_processor.py)
- Analyzed atlas_brain integration points (tools/security.py)
- Confirmed agent framework pattern in atlas_brain
- Confirmed discovery service pattern in atlas_brain
- Created implementation plan
- Received approval to begin Phase 1

### 2026-01-14 - Phase 1 Implementation COMPLETED

#### Files Created (18 Python files)
```
atlas_vision/
  __init__.py
  __main__.py
  core/
    __init__.py
    config.py
    constants.py
    protocols.py
  api/
    __init__.py
    main.py
    health.py
    cameras.py
    detections.py
    security.py
  devices/
    __init__.py
    protocols.py
    registry.py
    cameras/
      __init__.py
      base.py
      mock.py
```

#### Additional Files
- `requirements.txt` - Dependencies (fastapi, uvicorn, pydantic, etc.)
- `Dockerfile.vision` - Container build file

#### Endpoints Tested Successfully
| Endpoint | Method | Status |
|----------|--------|--------|
| `/health` | GET | PASS |
| `/info` | GET | PASS |
| `/cameras` | GET | PASS |
| `/cameras/{id}` | GET | PASS |
| `/cameras/{id}/record` | POST | PASS |
| `/detections/current` | GET | PASS |
| `/events` | GET | PASS |
| `/security` | GET | PASS |
| `/security/arm` | POST | PASS |
| `/security/disarm` | POST | PASS |

#### Test Results
- Server starts on port 5002
- 6 mock cameras auto-registered on startup
- All endpoints return valid JSON
- Security zones arm/disarm correctly
- Mock detections and events generated

#### No Breaking Changes
- Existing `ingest/` code untouched
- Existing `processing/` code untouched
- No changes to atlas_brain

---

### 2026-01-14 - Phase 2 Implementation COMPLETED

#### Files Created (6 new files)
```
atlas_vision/
  processing/
    __init__.py
    frame_buffer.py           # Thread-safe frame storage
    detection/
      __init__.py
      base.py                  # BaseDetector protocol
      motion.py                # Motion detection (MOG2)
  devices/cameras/
    rtsp.py                    # RTSP camera adapter
```

#### Files Updated
- `devices/cameras/__init__.py` - Added RTSPCamera export
- `api/cameras.py` - Added /register and DELETE endpoints
- `requirements.txt` - Added opencv-python-headless, numpy

#### New Endpoints
| Endpoint | Method | Status |
|----------|--------|--------|
| `/cameras/register` | POST | PASS |
| `/cameras/{id}` | DELETE | PASS |

#### Test Results
- Created test video with moving rectangle
- Registered video as camera source
- Motion detection triggered correctly (2 regions detected)
- Bounding boxes returned with confidence scores
- Camera unregistration works
- Phase 1 mock cameras still functional (6 cameras)

#### Features Implemented
- **RTSPCamera**: Async frame capture with reconnection
- **FrameBuffer**: Thread-safe ring buffer per source
- **MotionDetector**: Background subtraction (MOG2) with cooldown
- **Dynamic Registration**: Add/remove cameras at runtime

---

## Next Steps (Phase 3+)

1. **YOLO Detection** - Object detection (person, vehicle, etc.)
2. **mDNS Discovery** - Node announcement to atlas_brain
3. **MQTT Events** - Real-time event publishing
4. **Drone/Vehicle Support** - Mobile device capabilities
5. **Video Recording** - Save clips on detection
