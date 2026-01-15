# Reminder System Implementation

## Status: Complete
**Started**: 2026-01-14
**Completed**: 2026-01-14
**Last Updated**: 2026-01-14

## Overview

Implementing a reminder system for Atlas with:
- Natural language time parsing ("remind me in 2 hours", "remind me at 5pm tomorrow")
- PostgreSQL persistence (survives restarts)
- In-memory asyncio scheduler (sub-second delivery precision)
- Integration with centralized alert system for delivery

## Architecture

```
User: "remind me to call mom at 5pm"
              |
              v
      ┌──────────────────┐
      │   ReminderTool   │  ← Natural language parsing (dateparser)
      │   (set_reminder) │
      └────────┬─────────┘
               |
               v
      ┌──────────────────┐
      │ ReminderService  │  ← Core service with scheduler
      │  - PostgreSQL    │  ← Persistence (source of truth)
      │  - In-memory     │  ← Asyncio timers (instant delivery)
      └────────┬─────────┘
               |
               v (when due)
      ┌──────────────────┐
      │  AlertManager    │  ← Centralized delivery
      │  - TTS           │
      │  - Notifications │
      └──────────────────┘
```

## Phased Implementation Plan

### Phase 1: Foundation (Database + Service Core)
- [x] Create progress log
- [x] Create phased implementation plan
- [x] Add `dateparser` to requirements.txt
- [x] Add `ReminderConfig` to config.py
- [x] Add `Reminder` dataclass to storage/models.py
- [x] Create migration 010_reminders.sql
- [x] Create storage/repositories/reminder.py

### Phase 2: Scheduler Service
- [x] Create services/reminders.py with ReminderService
- [x] Implement in-memory asyncio scheduler
- [x] Implement startup loading from PostgreSQL
- [x] Implement create/list/complete/delete operations
- [x] Add service initialization to main.py lifespan

### Phase 3: Tool Integration
- [x] Create tools/reminder.py with ReminderTool
- [x] Implement natural language time parsing
- [x] Register tool in tools/__init__.py
- [x] Add "reminder" to TOOL_MAP in agents/tools.py
- [x] Add "reminder" to agent capabilities

### Phase 4: Alert System Integration
- [x] Connect scheduler to TTS callback for delivery
- [x] Reminder service has dedicated callback (not using rule-based AlertManager)
- [x] TTS integration in main.py lifespan

### Phase 5: Verification
- [x] Test tool import and syntax
- [x] Test dateparser expressions
- [x] Test database persistence
- [x] Test end-to-end reminder creation

## Files to Create

| File | Purpose |
|------|---------|
| `atlas_brain/storage/migrations/010_reminders.sql` | Database schema |
| `atlas_brain/storage/repositories/reminder.py` | Repository for CRUD |
| `atlas_brain/services/reminders.py` | Service with scheduler |
| `atlas_brain/tools/reminder.py` | Tool for agent |

## Files to Modify

| File | Change |
|------|--------|
| `requirements.txt` | Add dateparser |
| `atlas_brain/config.py` | Add ReminderConfig |
| `atlas_brain/storage/models.py` | Add Reminder dataclass |
| `atlas_brain/tools/__init__.py` | Register reminder tool |
| `atlas_brain/agents/tools.py` | Add "reminder" to TOOL_MAP |
| `atlas_brain/agents/atlas.py` | Add "reminder" to capabilities |
| `atlas_brain/alerts/events.py` | Add ReminderAlertEvent |
| `atlas_brain/main.py` | Initialize reminder service |

## Design Decisions

### 1. Time Parsing: dateparser
- Handles "in 5 minutes", "tomorrow at 3pm", "next Tuesday"
- Runs only at creation time (no delivery latency)
- Widely used, well-maintained

### 2. Scheduler: Hybrid PostgreSQL + In-Memory
- PostgreSQL: Source of truth, survives restarts
- In-memory asyncio timers: Sub-second precision
- On startup: Load pending from DB, schedule in memory
- On create: Write to DB, schedule in memory

### 3. Delivery: Via AlertManager
- Reuses existing TTS/notification infrastructure
- Respects global quiet hours and preferences
- Consistent delivery across all alert types

## Progress Log

### 2026-01-14 (Completed)
- Explored codebase structure
- Identified patterns for tools, services, alerts
- Created implementation plan

**Phase 1 Complete:**
- Added `dateparser>=1.2.0` to requirements.txt
- Added `ReminderConfig` to config.py with default timezone (America/Chicago)
- Added `Reminder` dataclass to storage/models.py
- Created `010_reminders.sql` migration
- Created `storage/repositories/reminder.py` with full CRUD

**Phase 2 Complete:**
- Created `services/reminders.py` with ReminderService
- Implemented in-memory asyncio scheduler with `call_later` for sub-second precision
- Loads pending reminders from DB on startup
- Full create/list/complete/delete operations

**Phase 3 Complete:**
- Created `tools/reminder.py` with three tools:
  - `set_reminder` - Create reminders with natural language time
  - `list_reminders` - Show upcoming reminders
  - `complete_reminder` - Mark reminder done
- Registered all tools in `tools/__init__.py`
- Added to TOOL_MAP in `agents/tools.py`
- Added "reminder" to agent capabilities

**Phase 4 Complete:**
- Connected reminder service to TTS callback in main.py
- Service uses direct callback pattern (simpler than rule-based alerts)

**Phase 5 Verification:**
- All imports work correctly
- dateparser handles common expressions ("in 5 minutes", "tomorrow at 3pm")
- Database persistence tested
- End-to-end reminder creation verified

### Gap Analysis and Fixes (2026-01-14)

**Critical Gaps Fixed:**
1. **Intent Parser Missing Reminder Tools** - Added `calendar,reminder,reminders` to TOOLS list in `capabilities/intent_parser.py`
2. **TTS Not Played for Reminders** - Updated callback to use `ConnectionManager.broadcast()` to push audio to connected voice clients
3. **Migration Numbering Conflict** - Renamed `010_reminders.sql` to `011_reminders.sql`

**Additional Fixes:**
4. Added `complete_reminder` to TOOL_MAP in `agents/tools.py`
5. Exported `ReminderService` and `get_reminder_service` from `services/__init__.py`
6. Added reminder and calendar env vars to `CLAUDE.md`
7. Added reminder examples to intent parser prompt

**Known Limitations (Resolved):**
- ~~Monthly recurrence uses 30-day intervals~~ - Fixed: Now uses `dateutil.relativedelta` for proper month arithmetic
- ~~Reminders only play on connected voice clients~~ - Fixed: Added `queue_announcement` to queue reminders for delivery when client connects
- ~~"next Tuesday at 10am" doesn't parse~~ - Fixed: Added `_normalize_time_text` to strip "next" prefix from weekdays

**Remaining Limitations:**
- dateparser doesn't support natural time words like "morning" or "evening" - users should say "at 9am" instead

### Second Gap Analysis (2026-01-14)

**Critical Gaps Found and Fixed:**

1. **Reminders > 7 days not loaded on startup**
   - Issue: Only reminders due within 7 days were loaded from DB on startup
   - Impact: Long-term reminders would be missed after server restart
   - Fix: Added `_periodic_reload_loop()` to reload reminders hourly

2. **scheduler_check_interval_seconds config unused**
   - Issue: Config setting existed but was never implemented
   - Impact: No mechanism to pick up long-term reminders
   - Fix: Integrated into periodic reload loop

3. **Deprecated asyncio.get_event_loop()**
   - Issue: Using deprecated API in Python 3.10+
   - Impact: Deprecation warnings
   - Fix: Changed to `asyncio.get_running_loop()`

**Changes Made:**
- `atlas_brain/services/reminders.py`:
  - Added `_reload_task` attribute to track background task
  - Added `_periodic_reload_loop()` method for hourly reload
  - Modified `initialize()` to start reload task
  - Modified `shutdown()` to cancel reload task
  - Changed `get_event_loop()` to `get_running_loop()`

### Alert System Integration Verification (2026-01-14)

**Verification Results:**
- ReminderAlertEvent conforms to AlertEvent protocol
- AlertManager has "reminder_due" rule for reminder events
- Rule matching works correctly
- Message formatting works ("Reminder: {message}")

**Gap Found and Fixed:**
- Alert system's `tts_alert_callback` was only synthesizing TTS but not delivering to voice clients
- Fixed: Updated callback to use `connection_manager.queue_announcement()` like reminder callback

**Reminder Delivery Flow (Conditional):**
- If `alerts.enabled` AND `alerts.tts_enabled`: Uses alert system path only
- Otherwise: Uses direct callback as fallback

**Critical Bug Fixed - Duplicate Delivery:**
- Issue: Both paths were active simultaneously, causing reminders to be announced twice
- Fix: Direct callback only enabled when alert system TTS is disabled
- Code: `if not (settings.alerts.enabled and settings.alerts.tts_enabled):`

### Deep Audit Fixes (2026-01-14)

**23 issues identified across 5 files. Critical fixes applied:**

1. **Month-end crash in `_format_time()`** (CRITICAL)
   - Issue: `now.date().replace(day=now.day + 1)` crashes on month-end (Jan 31 + 1 = invalid)
   - Fix: Use `(now + timedelta(days=1)).date()` instead
   - Also fixed platform-dependent strftime `%-I` format

2. **Race condition in scheduler** (CRITICAL)
   - Issue: No locking on `_scheduled` dictionary, concurrent access could cause duplicate deliveries
   - Fix: Added `threading.Lock()` to protect all `_scheduled` modifications
   - Also added check to skip already completed/delivered reminders

3. **Deprecated `datetime.utcnow()`** (HIGH)
   - Issue: `datetime.utcnow()` deprecated in Python 3.12+, returns naive datetime
   - Fix: Replaced with `datetime.now(timezone.utc)` in:
     - `storage/repositories/reminder.py` (5 occurrences)
     - `alerts/events.py` (1 occurrence in ReminderAlertEvent)

4. **`max_reminders_per_user` not enforced** (MEDIUM)
   - Issue: Config setting existed but was never checked
   - Fix: Added limit check in `ReminderService.create_reminder()`

5. **Silent database failures** (HIGH)
   - Issue: Repository methods returned None/empty list when DB unavailable
   - Fix: Created custom exceptions in `storage/exceptions.py`:
     - `DatabaseUnavailableError` - raised when DB not initialized
     - `DatabaseOperationError` - raised when DB operation fails
   - Updated all repository methods to raise instead of return None
   - Updated service layer to handle exceptions appropriately
   - Updated tool layer to return user-friendly error messages

**All documented issues resolved.**

### Config Validation Fixes (2026-01-14)

**Issues Fixed:**
1. **Config timezone validation** - Added `field_validator` that validates against `zoneinfo` (Python stdlib) with `pytz` fallback
2. **Config scheduler_check_interval bounds** - Added `ge=0.1, le=3600.0` constraints (0.1s to 1hr)
3. **Config max_reminders_per_user bounds** - Added `ge=1, le=1000` constraints

**Validation Behavior:**
- Invalid timezone: Raises `ValidationError` with helpful message listing valid format examples
- Out-of-bounds interval: Raises `ValidationError` with clear min/max constraints
- All validations occur at config load time (fail-fast)

### Transaction Context Fix (2026-01-14)

**Issue:** Recurring reminders could be lost if `reschedule_recurring()` failed after `mark_delivered()` succeeded. These were two separate database operations without transaction protection.

**Solution:**
1. Added `DatabasePool.transaction()` async context manager for atomic operations
2. Added `ReminderRepository.deliver_recurring()` method that atomically:
   - Marks current reminder as delivered
   - Creates next occurrence for recurring reminders
3. Updated `ReminderService._on_reminder_due()` to use `deliver_recurring()`

**Implementation Details:**
- Uses asyncpg's native transaction support via `connection.transaction()`
- On any failure, the entire operation rolls back (reminder stays undelivered)
- Extracted `_calculate_next_due()` helper for reuse and testing
- All repeat patterns (daily, weekly, monthly, yearly) tested including edge cases
