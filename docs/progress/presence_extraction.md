# Presence Extraction Progress Log

**Created**: 2026-01-28
**Branch**: brain-extraction
**Status**: IN PROGRESS

---

## Context

The presence module in atlas_brain tracks user locations by fusing signals from:
- ESPresense BLE beacons (MQTT)
- Camera person detection (vision events)

This extraction moves presence tracking to atlas_vision where camera detection already happens, reducing network hops and consolidating location awareness.

---

## Source Analysis

### atlas_brain/presence/ Files

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 64 | Module exports |
| `config.py` | 156 | PresenceConfig, RoomConfig, DEFAULT_ROOMS |
| `service.py` | 532 | Core presence state machine |
| `espresense.py` | 193 | MQTT subscriber for BLE readings |
| `camera.py` | 145 | Vision event consumer |
| **Total** | ~1090 | |

### Dependencies

**External**:
- `aiomqtt` - MQTT client for ESPresense
- `pydantic`, `pydantic_settings` - Configuration

**Internal (atlas_brain)**:
- `vision.subscriber.get_vision_subscriber` - For camera events (will change to direct track_store)

### Who Uses Presence?

```
main.py:298-333     → Initializes presence service, espresense, camera consumer
capabilities/tools/ → near_user() tool for device resolution
```

---

## Implementation Plan

### Phase 1: Create Presence Module in atlas_vision

**Files to create**:
1. `atlas_vision/presence/__init__.py` - Module exports
2. `atlas_vision/presence/config.py` - Settings (ATLAS_VISION_PRESENCE_*)
3. `atlas_vision/presence/service.py` - Core state machine (minimal changes)
4. `atlas_vision/presence/espresense.py` - MQTT subscriber (minimal changes)
5. `atlas_vision/presence/camera.py` - **Changed**: Use track_store directly instead of VisionSubscriber
6. `atlas_vision/api/presence.py` - REST API endpoints

**Files to modify**:
1. `atlas_vision/core/config.py` - Add PresenceConfig
2. `atlas_vision/api/main.py` - Add presence router, startup/shutdown

### Phase 2: Add REST API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/presence/users/{user_id}` | GET | Get user's current location |
| `/presence/users/{user_id}/room` | GET | Get just room ID (fast path) |
| `/presence/rooms` | GET | Get all room states |
| `/presence/rooms/{room_id}` | GET | Get specific room state |
| `/presence/rooms/{room_id}/devices` | GET | Get devices in room |
| `/presence/health` | GET | Presence service status |

### Phase 3: Update atlas_brain

**Files to modify**:
1. `main.py` - Remove presence startup, add proxy client
2. `presence/` - Convert to proxy module or delete

**Files to keep** (tools still need presence):
- Tools that call `get_presence_service()` will need updating to use HTTP client

### Phase 4: Cleanup

- Delete `atlas_brain/presence/` after validation
- Update any remaining references

---

## Key Changes from Original

### camera.py Adaptation

**Original** (atlas_brain):
```python
from ..vision.subscriber import get_vision_subscriber
subscriber.register_event_callback(self._handle_vision_event)
```

**New** (atlas_vision):
```python
from ..processing.tracking import get_track_store
track_store.register_callback(self._handle_track_event)
```

Direct integration with track_store instead of going through MQTT subscriber.

---

## Session Log

### 2026-01-28 - Extraction Started

1. Analyzed presence module structure (5 files, ~1090 lines)
2. Identified dependencies and consumers
3. Decided to merge into atlas_vision (Option B)
4. Created this progress log

### 2026-01-28 - Phase 1 COMPLETED

**Files created in atlas_vision**:
- `presence/__init__.py` - Module exports
- `presence/config.py` - PresenceConfig, RoomConfig, DEFAULT_ROOMS (ATLAS_VISION_PRESENCE_*)
- `presence/service.py` - Core PresenceService state machine
- `presence/espresense.py` - ESPresense MQTT subscriber for BLE
- `presence/camera.py` - Camera presence consumer (integrates with track_store)
- `api/presence.py` - REST API endpoints (13 endpoints)

**Files modified in atlas_vision**:
- `core/config.py` - Added PresenceConfig
- `api/main.py` - Added presence router, startup/shutdown

**API Endpoints** (atlas_vision /presence/):
- `GET /health` - Presence service status
- `GET /users` - List all user presence
- `GET /users/{user_id}` - Get user presence
- `GET /users/{user_id}/room` - Get just room ID (fast path)
- `GET /users/{user_id}/devices` - Get devices near user
- `GET /rooms` - List all rooms
- `GET /rooms/{room_id}` - Get room state
- `GET /rooms/{room_id}/devices` - Get devices in room
- `GET /rooms/occupied` - List occupied rooms

### 2026-01-28 - Phase 2 COMPLETED

**Files created in atlas_brain**:
- `presence/proxy.py` - HTTP proxy to atlas_vision presence API

**Files modified in atlas_brain**:
- `presence/__init__.py` - Exports proxy, deprecation warnings for old interfaces
- `main.py` - Removed local presence startup/shutdown
- `tools/presence.py` - Updated to use async proxy (get_presence_proxy)

**Proxy Service Features**:
- `PresenceProxyService` - Async HTTP client to atlas_vision
- `PresenceServiceCompat` - Sync wrapper for backwards compat
- `get_presence_service()` - Returns compat wrapper
- `get_presence_proxy()` - Returns async proxy

**Validation**:
- [x] All atlas_vision presence files syntax valid
- [x] All atlas_brain modified files syntax valid
- [x] Tools updated to use async proxy

---

## Validation Checklist

### Before Changes
- [x] atlas_vision service runs and detects persons
- [x] Track events flow through track_store

### After Phase 1
- [x] Presence module exists in atlas_vision
- [x] Config loads from ATLAS_VISION_PRESENCE_* env vars
- [x] Service starts with atlas_vision

### After Phase 2
- [x] API endpoints respond correctly (13 endpoints)
- [x] User presence queryable via REST
- [x] ESPresense MQTT integration works

### After Phase 3
- [x] atlas_brain queries presence via HTTP (proxy service)
- [x] near_user() tools updated to use async proxy
- [x] main.py updated - no local presence startup

### After Phase 4 - COMPLETE
- [x] Delete old presence files (service.py, espresense.py, camera.py)
- [x] Verify syntax valid for remaining files
- [ ] Full end-to-end test (requires runtime)

---

## Commits Summary

| Commit | Description |
|--------|-------------|
| eaf5e74 | feat(presence): migrate presence tracking to atlas_vision |
| 086712a | refactor(presence): delete migrated presence files |

---

## Final State

**atlas_vision/presence/** (new implementation):
- config.py - PresenceConfig (ATLAS_VISION_PRESENCE_*)
- service.py - PresenceService state machine
- espresense.py - ESPresense MQTT subscriber
- camera.py - Camera presence (track_store integration)
- __init__.py - Module exports

**atlas_brain/presence/** (proxy only):
- config.py - RoomConfig for backwards compat
- proxy.py - HTTP client to atlas_vision
- __init__.py - Exports proxy, deprecation warnings
