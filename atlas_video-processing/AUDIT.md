# Video Processing Audit

**Date**: 2026-01-13
**Status**: Pre-Alpha (~2% implemented)

---

## Summary

The video processing module has good architecture but almost no implementation. Only basic Kafka pub/sub works. The REST API that atlas_brain expects does not exist.

## Structure

```
atlas_video-processing/
├── ingest/
│   ├── drone_client/           [IMPLEMENTED] Basic Kafka producer
│   └── camera_client/          [EMPTY]
├── processing/
│   ├── video_stream_processor/ [HOLLOW] OpenCV imported but unused
│   └── object_detection/       [EMPTY]
├── control/
│   ├── mission_planner/        [EMPTY]
│   └── fleet_management/       [EMPTY]
├── persistence/                [EMPTY] PostgreSQL in compose but no code
├── llm_service/                [EMPTY]
└── kafka/                      [OK] Vendored binaries
```

## Critical Gaps

1. **No REST API** - atlas_brain expects :5002/:5004 endpoints
2. **No object detection** - placeholder sleep() only
3. **No database connection** - PostgreSQL unused
4. **No camera management** - can't register/discover cameras
5. **No event publishing** - nothing feeds atlas_brain

## Required Endpoints (from atlas_brain/tools/security.py)

```
GET  /cameras              - List cameras
GET  /cameras/{id}         - Camera status
GET  /detections/current   - Current detections
GET  /events               - Motion/detection events
POST /security/arm         - Arm zone
POST /security/disarm      - Disarm zone
POST /cameras/{id}/record  - Recording control
POST /query                - LLM service queries
```

**None implemented.**

## Code Issues

- Hardcoded `localhost:9092` (breaks Docker)
- Print statements instead of logging
- Old Kafka API version (0,10,1)
- No graceful shutdown
- No configuration management
- Credentials in docker-compose.yml

## Options When Resuming

1. **Build out** - Implement REST API + real processing
2. **Simplify** - Remove Kafka, direct camera → detection flow
3. **Replace** - Use Frigate/ZoneMinder with Atlas adapter
4. **Hybrid** - Frigate for detection, custom for drone/mission control

## Priority Order (if building out)

1. REST API wrapper service (FastAPI)
2. Camera registration/discovery
3. Object detection (YOLO/similar)
4. Event publishing to Kafka topics atlas_brain consumes
5. Database persistence
6. Mission planner (for drones)
7. LLM service integration
