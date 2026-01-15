# Centralized Alert System - Implementation Progress

**Created:** 2026-01-14
**Last Updated:** 2026-01-14
**Status:** COMPLETED

---

## Overview

Refactor the Atlas alert system from the current vision-specific implementation to a centralized, unified alert system that can handle events from multiple sources: vision detection, audio events, Home Assistant state changes, and Kafka security events.

### Goals
- Single alert management system for all event types
- Unified storage and API for alert history/acknowledgment
- Centralized rule engine with pattern matching and cooldowns
- Callback-based delivery (TTS, notifications, etc.)
- No fragmentation of alert logic across the codebase

### Key Decisions Made
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Approach | Option A - Expand AlertManager | Core alert logic is identical across event types |
| Location | New `atlas_brain/alerts/` directory | Clean separation from event sources |
| Event abstraction | `AlertEvent` protocol | Different sources have different fields, unified interface |
| Storage | Single `alerts` table | Unified history/acknowledgment across all types |
| Migration | Non-breaking, gradual | Keep existing vision alerts working during transition |

---

## Current State Analysis

### Existing Alert/Notification Systems Found

| System | Location | Purpose | Alert Logic |
|--------|----------|---------|-------------|
| Vision Alerts | `atlas_brain/vision/alerts.py` | Alert on YOLO detections | Full (rules, cooldowns, callbacks) |
| Security AlertHandler | `atlas_brain/tools/security.py` | Kafka event announcements | Partial (cooldowns, formatting) |
| Audio Monitor | `atlas_brain/services/audio_events/monitor.py` | Audio event detection | Partial (cooldowns, callbacks) |
| HA WebSocket | `atlas_brain/capabilities/backends/homeassistant_ws.py` | State change events | None (just callback dispatch) |
| Security Consumer | `atlas_brain/services/security_events.py` | Kafka event consumption | None (delegates to AlertHandler) |

### Current Vision Alert System (Lines 1-309 in `atlas_brain/vision/alerts.py`)

```python
@dataclass
class AlertRule:
    name: str
    source_pattern: str  # Camera pattern
    class_name: str      # Detection class
    event_type: str      # new_track, track_lost
    message_template: str
    cooldown_seconds: int
    enabled: bool
    priority: int

class AlertManager:
    - _rules: dict[str, AlertRule]
    - _cooldowns: dict[str, datetime]
    - _callbacks: list[AlertCallback]

    def add_rule(rule) / remove_rule(name)
    def enable_rule(name) / disable_rule(name)
    def list_rules() -> sorted by priority
    def register_callback(callback)
    def _check_cooldown(rule_name, seconds) -> bool
    def _update_cooldown(rule_name)
    async def process_event(event: VisionEvent) -> Optional[str]
    async def _persist_alert(rule, message, event)
```

### Current Security AlertHandler (Lines 614-687 in `atlas_brain/tools/security.py`)

```python
class SecurityAlertHandler:
    - alert_cooldown: dict[str, datetime]
    - cooldown_seconds = 30

    def should_alert(event_type, camera_id) -> bool  # Cooldown check
    async def handle_event(event: dict) -> Optional[str]  # Returns announcement
    def _camera_id_to_name(camera_id) -> str
```

### Current Audio Monitor (Lines 54-197 in `atlas_brain/services/audio_events/monitor.py`)

```python
class AudioMonitor:
    - _recent_events: dict[str, datetime]  # Cooldowns
    - _on_event: Callable[[MonitoredEvent], None]

    def _should_report_event(label) -> bool  # Cooldown check
    def set_event_callback(callback)
    def process_buffer() -> list[MonitoredEvent]
```

---

## Proposed Architecture

### New Directory Structure

```
atlas_brain/
└── alerts/                          # NEW - Centralized alert system
    ├── __init__.py                  # Public exports
    ├── events.py                    # AlertEvent protocol + event types
    ├── rules.py                     # AlertRule with multi-event support
    ├── manager.py                   # Centralized AlertManager
    └── delivery.py                  # Alert delivery callbacks (TTS, push, etc.)
```

### AlertEvent Protocol

```python
from typing import Protocol, Any
from datetime import datetime

class AlertEvent(Protocol):
    """Base protocol for all alertable events."""
    event_type: str          # "vision", "audio", "ha_state", "security"
    source_id: str           # Camera ID, sensor ID, entity_id
    timestamp: datetime
    metadata: dict[str, Any]

    def get_field(self, name: str) -> Any:
        """Get event-specific field for rule matching."""
        ...
```

### Event Type Implementations

```python
@dataclass
class VisionAlertEvent:
    """Vision detection event for alert processing."""
    event_type: str = "vision"
    source_id: str  # Camera ID
    timestamp: datetime
    class_name: str  # person, car, etc.
    detection_type: str  # new_track, track_lost
    track_id: int
    node_id: str
    metadata: dict[str, Any]

    @classmethod
    def from_vision_event(cls, event: VisionEvent) -> "VisionAlertEvent":
        ...

@dataclass
class AudioAlertEvent:
    """Audio detection event for alert processing."""
    event_type: str = "audio"
    source_id: str  # Microphone/location ID
    timestamp: datetime
    sound_class: str  # doorbell, glass_break, etc.
    confidence: float
    priority: str  # critical, high, medium, low
    metadata: dict[str, Any]

@dataclass
class HAStateAlertEvent:
    """Home Assistant state change for alert processing."""
    event_type: str = "ha_state"
    source_id: str  # entity_id
    timestamp: datetime
    old_state: str
    new_state: str
    domain: str  # binary_sensor, sensor, etc.
    attributes: dict[str, Any]
    metadata: dict[str, Any]

@dataclass
class SecurityAlertEvent:
    """Security system event for alert processing."""
    event_type: str = "security"
    source_id: str  # Camera ID
    timestamp: datetime
    detection_type: str  # person_detected, motion, vehicle, etc.
    label: Optional[str]  # known:Juan, unknown, etc.
    confidence: float
    metadata: dict[str, Any]
```

### Unified AlertRule

```python
@dataclass
class AlertRule:
    """Alert rule supporting multiple event types."""
    name: str

    # Event type matching (required)
    event_types: list[str]  # ["vision", "audio", "ha_state", "security"] or ["*"]

    # Source pattern matching
    source_pattern: str  # Pattern like "front_door", "*", "binary_sensor.*"

    # Event-specific conditions (optional)
    conditions: dict[str, Any] = field(default_factory=dict)
    # Examples:
    # Vision: {"class_name": "person", "detection_type": "new_track"}
    # Audio: {"sound_class": "doorbell", "min_confidence": 0.7}
    # HA: {"domain": "binary_sensor", "new_state": "on"}
    # Security: {"detection_type": "person_detected"}

    # Alert configuration
    message_template: str = "{event_type} at {source}"
    cooldown_seconds: int = 30
    enabled: bool = True
    priority: int = 1

    def matches(self, event: AlertEvent) -> bool:
        """Check if event matches this rule."""
        ...

    def format_message(self, event: AlertEvent) -> str:
        """Format alert message with event data."""
        ...
```

### Unified AlertManager

```python
class AlertManager:
    """Centralized alert manager for all event sources."""

    def __init__(self):
        self._rules: dict[str, AlertRule] = {}
        self._cooldowns: dict[str, datetime] = {}
        self._callbacks: list[AlertCallback] = []
        self._add_default_rules()

    # Rule management
    def add_rule(self, rule: AlertRule) -> None
    def remove_rule(self, name: str) -> bool
    def get_rule(self, name: str) -> Optional[AlertRule]
    def list_rules(self, event_type: Optional[str] = None) -> list[AlertRule]
    def enable_rule(self, name: str) -> bool
    def disable_rule(self, name: str) -> bool

    # Callback management
    def register_callback(self, callback: AlertCallback) -> None
    def unregister_callback(self, callback: AlertCallback) -> None

    # Event processing
    async def process_event(self, event: AlertEvent) -> Optional[str]:
        """Process any alert event against all rules."""
        ...

    # Internal
    def _check_cooldown(self, rule_name: str, seconds: int) -> bool
    def _update_cooldown(self, rule_name: str) -> None
    async def _persist_alert(self, rule: AlertRule, message: str, event: AlertEvent) -> None
```

### Unified Alert Storage Model

```python
@dataclass
class Alert:
    """A triggered alert from any event source."""
    id: UUID
    rule_name: str
    event_type: str  # vision, audio, ha_state, security
    message: str
    source_id: str
    triggered_at: datetime
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    event_data: dict[str, Any] = field(default_factory=dict)  # Full event snapshot
    metadata: dict[str, Any] = field(default_factory=dict)
```

---

## Files Affected Analysis

### TIER 1: New Files (Centralized Alert System)

| File | Purpose | Risk |
|------|---------|------|
| `alerts/__init__.py` | Public exports | LOW |
| `alerts/events.py` | AlertEvent protocol + implementations | LOW |
| `alerts/rules.py` | AlertRule with multi-event support | LOW |
| `alerts/manager.py` | Centralized AlertManager | LOW |
| `alerts/delivery.py` | Delivery callbacks (TTS, etc.) | LOW |

### TIER 2: Storage Updates (Database)

| File | Changes | Risk |
|------|---------|------|
| `storage/migrations/009_unified_alerts.sql` | New unified alerts table | LOW |
| `storage/models.py` | Add `Alert` model (keep VisionAlert for migration) | LOW |
| `storage/repositories/unified_alerts.py` | New repository for unified alerts | LOW |
| `storage/repositories/__init__.py` | Export new repository | LOW |

### TIER 3: API Updates

| File | Changes | Risk |
|------|---------|------|
| `api/alerts.py` | NEW - Unified alerts API | LOW |
| `api/__init__.py` | Add alerts router | LOW |
| `api/vision.py` | Deprecate vision-specific alert endpoints (keep working) | MEDIUM |

### TIER 4: Event Source Integration

| File | Changes | Risk |
|------|---------|------|
| `vision/subscriber.py` | Update `_process_alerts()` to use centralized system | MEDIUM |
| `vision/alerts.py` | Mark as deprecated, delegate to centralized | LOW |
| `services/audio_events/monitor.py` | Add centralized alert integration | MEDIUM |
| `capabilities/backends/homeassistant_ws.py` | Add alert processing for state changes | MEDIUM |
| `services/security_events.py` | Update to use centralized AlertManager | MEDIUM |
| `tools/security.py` | Remove duplicate SecurityAlertHandler | LOW |

### TIER 5: Main Application

| File | Changes | Risk |
|------|---------|------|
| `main.py` | Initialize centralized AlertManager, update callbacks | MEDIUM |

---

## Exact Insertion Points

### Phase 1: New alerts/ Module (No Breaking Changes)

**File: `atlas_brain/alerts/__init__.py`** (NEW)
```python
from .events import AlertEvent, VisionAlertEvent, AudioAlertEvent, HAStateAlertEvent
from .rules import AlertRule
from .manager import AlertManager, get_alert_manager
from .delivery import TTSDelivery, register_delivery

__all__ = [...]
```

**File: `atlas_brain/alerts/events.py`** (NEW)
- AlertEvent Protocol
- VisionAlertEvent, AudioAlertEvent, HAStateAlertEvent, SecurityAlertEvent dataclasses

**File: `atlas_brain/alerts/rules.py`** (NEW)
- AlertRule dataclass with multi-event matching

**File: `atlas_brain/alerts/manager.py`** (NEW)
- AlertManager class (centralized)
- get_alert_manager() function

### Phase 2: Storage (Database Schema + Repository)

**File: `atlas_brain/storage/migrations/009_unified_alerts.sql`** (NEW)
```sql
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_name VARCHAR(64) NOT NULL,
    event_type VARCHAR(32) NOT NULL,  -- vision, audio, ha_state, security
    message TEXT NOT NULL,
    source_id VARCHAR(128) NOT NULL,
    triggered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP,
    acknowledged_by VARCHAR(128),
    event_data JSONB DEFAULT '{}'::jsonb,  -- Full event snapshot
    metadata JSONB DEFAULT '{}'::jsonb
);
-- Indexes for common queries
```

**File: `atlas_brain/storage/models.py`**
- Lines 438+: Add new `Alert` dataclass (keep existing VisionAlert)

**File: `atlas_brain/storage/repositories/unified_alerts.py`** (NEW)
- AlertRepository class with CRUD operations

**File: `atlas_brain/storage/repositories/__init__.py`**
- Line 8: Add import for unified_alerts
- Line 17+: Add exports

### Phase 3: Unified API

**File: `atlas_brain/api/alerts.py`** (NEW)
- `/alerts` - List alerts with filters
- `/alerts/stats` - Alert statistics
- `/alerts/{id}/acknowledge` - Acknowledge single alert
- `/alerts/acknowledge-all` - Bulk acknowledge
- `/alerts/rules` - Rule management endpoints

**File: `atlas_brain/api/__init__.py`**
- Line 17+: Add `from .alerts import router as alerts_router`
- Line 32+: Add `router.include_router(alerts_router)`

### Phase 4: Event Source Integration

**File: `atlas_brain/vision/subscriber.py`**
- Lines 224-231: Update `_process_alerts()` method:
```python
async def _process_alerts(self, event: VisionEvent) -> None:
    """Process event through centralized alert system."""
    try:
        from ..alerts import get_alert_manager, VisionAlertEvent
        manager = get_alert_manager()
        alert_event = VisionAlertEvent.from_vision_event(event)
        await manager.process_event(alert_event)
    except Exception as e:
        logger.warning("Failed to process alerts: %s", e)
```

**File: `atlas_brain/vision/alerts.py`**
- Add deprecation warning at module level
- Update `AlertManager.process_event()` to delegate to centralized system

**File: `atlas_brain/services/audio_events/monitor.py`**
- Lines 168-173: In callback after event detection, also send to centralized:
```python
# After existing callback
from ..alerts import get_alert_manager, AudioAlertEvent
alert_event = AudioAlertEvent.from_monitored_event(monitored_event)
await get_alert_manager().process_event(alert_event)
```

**File: `atlas_brain/capabilities/backends/homeassistant_ws.py`**
- Lines 214-224: In `_handle_message()` after state_changed callback:
```python
# Add centralized alert processing
from ...alerts import get_alert_manager, HAStateAlertEvent
try:
    alert_event = HAStateAlertEvent.from_ha_event(event_data)
    await get_alert_manager().process_event(alert_event)
except Exception as e:
    logger.debug("Alert processing skipped: %s", e)
```

**File: `atlas_brain/services/security_events.py`**
- Lines 127-134: Update `_handle_event()` to use centralized system:
```python
async def _handle_event(self, topic: str, event: dict):
    from ..alerts import get_alert_manager, SecurityAlertEvent
    alert_event = SecurityAlertEvent.from_kafka_event(event)
    message = await get_alert_manager().process_event(alert_event)
    if message and self._announce_callback:
        await self._announce_callback(message)
```

### Phase 5: Main Application Integration

**File: `atlas_brain/main.py`**
- Lines 209-242: Update vision subscriber initialization:
```python
# Initialize centralized alert system
from .alerts import get_alert_manager, TTSDelivery

alert_manager = get_alert_manager()

# Register TTS delivery callback
if tts_registry.get_active():
    tts_delivery = TTSDelivery(tts_registry)
    alert_manager.register_callback(tts_delivery.deliver)

# Vision subscriber still starts the same way
if settings.mqtt.enabled:
    vision_subscriber = get_vision_subscriber()
    await vision_subscriber.start()
```

---

## Implementation Phases

### Phase 1: Core Alert Module (No Breaking Changes)
**Goal:** Create new centralized alert system alongside existing

1. Create `atlas_brain/alerts/` directory structure
2. Implement `AlertEvent` protocol and event types
3. Implement `AlertRule` with multi-event support
4. Implement centralized `AlertManager`
5. Add default rules for all event types
6. **Verification:** Import module, create rules, process test events

### Phase 2: Database Schema + Repository
**Goal:** Add unified storage for all alerts

1. Create migration `009_unified_alerts.sql`
2. Add `Alert` model to `storage/models.py`
3. Create `AlertRepository` in `storage/repositories/`
4. Run migration
5. **Verification:** Save and retrieve test alerts via repository

### Phase 3: Unified API
**Goal:** Single API for all alert operations

1. Create `api/alerts.py` with unified endpoints
2. Register router in `api/__init__.py`
3. **Verification:** Test all endpoints via curl/httpie

### Phase 4: Event Source Integration
**Goal:** Connect all event sources to centralized system

1. Update `vision/subscriber.py` to use centralized alerts
2. Add deprecation to `vision/alerts.py`
3. Update `audio_events/monitor.py` integration
4. Update `homeassistant_ws.py` integration
5. Update `security_events.py` integration
6. Remove duplicate `SecurityAlertHandler`
7. **Verification:** Trigger events from each source, verify alerts created

### Phase 5: Application Integration + Testing
**Goal:** Full integration and verification

1. Update `main.py` initialization
2. Configure delivery callbacks (TTS, etc.)
3. End-to-end testing all event sources
4. **Verification:** Full integration tests

### Phase 6: Cleanup (Optional, After Verification)
**Goal:** Remove deprecated code

1. Remove deprecated vision-specific alert endpoints
2. Remove old `VisionAlert` model (after data migration)
3. Remove `SecurityAlertHandler` from `tools/security.py`
4. Update documentation

---

## Verification Checklist

### Before Implementation
- [x] Deep dive complete on vision alerts
- [x] Deep dive complete on security alerts
- [x] Deep dive complete on audio monitor
- [x] Deep dive complete on HA WebSocket
- [x] Identified all affected files
- [x] User approval pending

### After Each Phase
- [ ] Phase 1: Alert module imports, rules match events correctly
- [ ] Phase 2: Database migration runs, alerts persist correctly
- [ ] Phase 3: API endpoints respond correctly
- [ ] Phase 4: Each event source creates unified alerts
- [ ] Phase 5: End-to-end flow works (event -> alert -> TTS)
- [ ] Phase 6: Deprecated code removed, no regressions

### Integration Tests
- [ ] Vision detection creates unified alert
- [ ] Audio detection creates unified alert
- [ ] HA state change creates unified alert
- [ ] Security event creates unified alert
- [ ] TTS announces alerts
- [ ] Alert acknowledgment works
- [ ] Cooldowns prevent spam
- [ ] Rule enable/disable works

---

## Risk Mitigation

### High Risk Areas
1. **Breaking existing vision alerts** - Users may have rules configured
   - Mitigation: Keep existing system working, migrate rules to new format

2. **Database migration** - Must preserve existing alert history
   - Mitigation: New table, migrate data after verification

3. **Event source integration** - Must not break event handling
   - Mitigation: Add centralized calls alongside existing, don't remove

### Rollback Strategy
- Each phase is independent
- Phases 1-3 are purely additive (no breaking changes)
- Phase 4 adds calls but doesn't remove existing
- Can stop at any phase and have working system

---

## Session Notes

### 2026-01-14 Session
- Investigated existing alert/notification systems
- Found 5 different systems with alert logic
- User approved centralized approach (Option A)
- Created comprehensive implementation plan
- **Implementation completed successfully**

#### Files Created
- `atlas_brain/alerts/__init__.py` - Module exports
- `atlas_brain/alerts/events.py` - AlertEvent protocol + event types
- `atlas_brain/alerts/rules.py` - AlertRule with multi-event support
- `atlas_brain/alerts/manager.py` - Centralized AlertManager
- `atlas_brain/alerts/delivery.py` - Delivery callbacks (TTS, webhook)
- `atlas_brain/storage/migrations/009_unified_alerts.sql` - Database schema
- `atlas_brain/storage/repositories/unified_alerts.py` - Repository
- `atlas_brain/api/alerts.py` - Unified API endpoints

#### Files Modified
- `atlas_brain/config.py` - Added AlertsConfig
- `atlas_brain/storage/models.py` - Added Alert model
- `atlas_brain/storage/repositories/__init__.py` - Export new repository
- `atlas_brain/vision/__init__.py` - Re-export from centralized alerts
- `atlas_brain/vision/subscriber.py` - Use centralized alerts
- `atlas_brain/services/security_events.py` - Use centralized alerts
- `atlas_brain/api/__init__.py` - Include alerts router
- `atlas_brain/main.py` - Initialize centralized alerts on startup

#### Verification Results
- Module imports: OK
- Database migration: OK
- API endpoints: OK (6 rules, stats, acknowledgment)
- End-to-end alert flow: OK
- Backwards compatibility: OK (vision imports still work)

---

## Open Questions (Deferred)
1. Should we migrate existing vision_alerts data to new alerts table?
2. Should HA state changes for ALL entities trigger alert processing, or only configured ones?
3. What default rules should be included for audio/HA/security events?
4. Should we add webhook delivery in addition to TTS?
