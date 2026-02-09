# Tools & Workflows Audit

**Date:** 2026-02-07
**Status:** Complete (5/5 issues resolved)

## Registered Tools (26 total)

| # | Tool Name | Type | Fast-Path | LLM? | Routed? |
|---|-----------|------|-----------|------|---------|
| 1 | `get_time` | Query | Yes | No (logic) | Yes - `get_time` route |
| 2 | `get_weather` | Query | Yes | No (API) | Yes - `get_weather` route |
| 3 | `get_traffic` | Query | Yes | No (API) | Yes - `get_traffic` route |
| 4 | `get_location` | Query | Yes | No (HA+API) | Yes - `get_location` route |
| 5 | `get_calendar` | Query | Yes | No (Google API) | Yes - `get_calendar` route |
| 6 | `list_reminders` | Query | Yes | No (DB) | Yes - `list_reminders` route |
| 7 | `send_notification` | Action | No | No (ntfy API) | NO ROUTE |
| 8 | `lights_near_user` | Presence | No | No (logic) | NO ROUTE |
| 9 | `media_near_user` | Presence | No | No (logic) | NO ROUTE |
| 10 | `scene_near_user` | Presence | No | No (logic) | NO ROUTE |
| 11 | `where_am_i` | Presence | No | No (logic) | NO ROUTE |
| 12 | `list_cameras` | Security | No | No (API) | NO ROUTE |
| 13 | `get_camera_status` | Security | No | No (API) | NO ROUTE |
| 14 | `start_recording` | Security | No | No (API) | NO ROUTE |
| 15 | `stop_recording` | Security | No | No (API) | NO ROUTE |
| 16 | `ptz_control` | Security | No | No (API) | NO ROUTE |
| 17 | `get_current_detections` | Security | No | No (API) | NO ROUTE |
| 18 | `query_detections` | Security | No | No (API) | NO ROUTE |
| 19 | `get_person_at_location` | Security | No | No (API) | NO ROUTE |
| 20 | `get_motion_events` | Security | No | No (API) | NO ROUTE |
| 21 | `list_zones` | Security | No | No (API) | NO ROUTE |
| 22 | `get_zone_status` | Security | No | No (API) | NO ROUTE |
| 23 | `arm_zone` | Security | No | No (API) | NO ROUTE |
| 24 | `disarm_zone` | Security | No | No (API) | NO ROUTE |
| 25 | `show_camera_feed` | Display | No | No (ffplay) | NO ROUTE |
| 26 | `close_camera_feed` | Display | No | No (ffplay) | NO ROUTE |

### Not Registered (workflow-only tools)

`set_reminder`, `complete_reminder`, `send_email`, `estimate_email`, `proposal_email`, `book_appointment`, `cancel_appointment`, `reschedule_appointment`, `lookup_customer`, `check_availability`

---

## Workflows (6 total)

| # | Workflow | Wired? | Route | USE_REAL_TOOLS? | Multi-Turn |
|---|----------|--------|-------|-----------------|------------|
| 1 | Booking | Yes | `booking` | Mock by default | Yes |
| 2 | Reminder | Yes | `reminder` | Mock by default | Yes |
| 3 | Email | Yes | `email` | Mock by default | Yes |
| 4 | Calendar | Yes | `calendar_write` | Mock by default | Yes |
| 5 | Security | Yes | `security` | Mock by default | No |
| 6 | Presence | Yes | `presence` | Mock by default | No |

---

## Hardcoded Prompts (10 found)

| # | File | Line | Prompt | Configurable? |
|---|------|------|--------|---------------|
| 1 | `intent_parser.py` | 72-93 | `UNIFIED_INTENT_PROMPT` (1026 tokens) | No |
| 2 | `intent_parser.py` | 284 | `"You parse intents. Output ONLY valid JSON."` | No |
| 3 | `intent_router.py` | 324-328 | LLM fallback classifier prompt | No |
| 4 | `booking.py` | 40-45 | `FIELD_PROMPTS` (name/address/date/time extraction) | No |
| 5 | `receptionist.py` | 266-316 | Full receptionist system prompt | No |
| 6 | `home.py` | 407-409 | `"You are a helpful home assistant."` | No |
| 7 | `home.py` | 436-438 | `"You are a helpful assistant. Use tools when needed."` | No |
| 8 | `atlas.py` | 750-754 | `"You are Atlas, a capable personal assistant..."` | No |
| 9 | `launcher.py` | 330-334 | `_PREFILL_SYSTEM_PROMPT` (KV-cache warmup) | No |
| 10 | `streaming.py` | 270-272 | `"You are a helpful home assistant."` | No |

---

## Critical Issues

### Issue 1: 20/26 registered tools have NO route -- unreachable by voice

**Affected tools:** 7-26 (notification, presence, security, display)
**Root cause:** No entries in `ROUTE_DEFINITIONS`, `ROUTE_TO_ACTION`, or `ROUTE_TO_WORKFLOW` in `intent_router.py`
**Impact:** "Send me a notification", "show me the camera", "where am I" route to `conversation` and get a chat response instead of executing the tool
**Fix location:** `atlas_brain/services/intent_router.py` lines 47-146

**Status:** [x] Phase 1 complete (standalone tools routed)

**Phase 1 changes (2026-02-08):**
- Added `where_am_i` route (8 exemplars) + `PARAMETERLESS_TOOLS` (fast path, ~10ms)
- Added `notification` route (7 exemplars) -> `send_notification` tool (LLM slow path for message extraction)
- Added `show_camera` route (9 exemplars) -> `show_camera_feed` tool (LLM slow path for camera_name/display params)
- Separated `get_location` (GPS/geo via HA phone tracker) from `where_am_i` (indoor room detection via presence system)
- File: `atlas_brain/services/intent_router.py` -- `ROUTE_DEFINITIONS`, `ROUTE_TO_ACTION`, `PARAMETERLESS_TOOLS`
- Validation: 16/16 routing accuracy test cases passed (new + existing routes)
- Remaining: 13 security tools + 3 presence tools deferred to Issue 2 (workflow wiring)

---

### Issue 2: Security & Presence workflows built but NOT wired

**Affected files:** `atlas_brain/agents/graphs/security.py`, `atlas_brain/agents/graphs/presence.py`
**Root cause:** Neither imported in `atlas.py`, neither in `ROUTE_TO_WORKFLOW`
**Impact:** Full LangGraph implementations exist but are impossible to reach via voice/text
**Fix locations:**
- `atlas_brain/services/intent_router.py` (add routes + exemplars)
- `atlas_brain/agents/graphs/atlas.py` (import + dispatch in `start_workflow`/`continue_workflow`)

**Status:** [x] Complete

**Changes (2026-02-08):**
- Added `SECURITY_WORKFLOW_TYPE = "security"` to `security.py`
- Added `PRESENCE_WORKFLOW_TYPE = "presence"` to `presence.py`
- Added `security` route (18 exemplars) covering cameras, detections, zones
- Added `presence` route (16 exemplars) covering scene/mood commands and proximity-qualified device control
- Added both to `ROUTE_TO_ACTION` and `ROUTE_TO_WORKFLOW` in `intent_router.py`
- Imported `run_security_workflow` and `run_presence_workflow` in `atlas.py`
- Added dispatch cases in both `start_workflow()` and `continue_workflow()`
- Presence workflow passes `speaker_id` as `user_id` for room resolution
- Validation: 26/27 routing accuracy (1 known overlap: "dim the lights in here" -> device_command)
- No regressions on existing routes (27/27 existing test cases pass)
- Graph compilation verified, StreamingAtlasAgent imports clean
- Known limitation: proximity-qualified device commands ("dim the lights in here") may route to device_command instead of presence due to semantic overlap -- LLM fallback handles correctly

---

### Issue 3: USE_REAL_TOOLS not set -- all workflows run in MOCK mode

**Affected files:** `booking.py`, `reminder.py`, `email.py`, `calendar.py`, `security.py`, `presence.py`
**Root cause:** `os.environ.get("USE_REAL_TOOLS", "false")` defaults to `"false"`, not set in `.env`
**Impact:** Every workflow silently returns fake data. No logging when mock mode is active.
**Fix locations:**
- `.env` (add `USE_REAL_TOOLS=true`)
- Or refactor to per-workflow config in `atlas_brain/config.py`

**Status:** [x] Complete

**Changes (2026-02-08):**
- Added `WorkflowConfig` to `atlas_brain/config.py` with `use_real_tools: bool = True`
- Env var: `ATLAS_WORKFLOW_USE_REAL_TOOLS` (default: true)
- Added `workflows: WorkflowConfig` field to `Settings` class
- Updated 3 Pattern-1 files (booking, reminder, calendar): replaced `os.environ.get("USE_REAL_TOOLS")` with `settings.workflows.use_real_tools`
- Updated 3 Pattern-2 files (email, security, presence): replaced `_use_real_tools()` body with config access
- Added startup logging in booking, reminder, calendar (logs "real" or "mock" at import)
- Added `ATLAS_WORKFLOW_USE_REAL_TOOLS=true` to `.env`
- Override verified: `ATLAS_WORKFLOW_USE_REAL_TOOLS=false` correctly disables real tools
- All 6 workflows confirmed reading `True` from config
- Graph compilation and 27 existing tests pass with no regressions

---

### Issue 4: UNIFIED_INTENT_PROMPT is stale

**Affected file:** `atlas_brain/capabilities/intent_parser.py` lines 72-93
**Root cause:** Prompt still contains booking/reminder examples that now route via `ROUTE_TO_WORKFLOW` before intent parser is called
**Impact:** Burns 1026 tokens on examples that are never executed via this path. Missing examples for notification, camera, presence, security tools.
**Fix location:** `atlas_brain/capabilities/intent_parser.py` line 72

**Status:** [x] Complete

**Changes (2026-02-08):**
- Removed 8 stale examples: time query, calendar read, reminder (x2), booking (x4), availability
- Added 2 new examples: notification with message param, camera feed with camera_name + display params
- Kept 5 examples: device on/off, pronoun refs (it/them), brightness, conversation
- Token reduction: ~1026 -> ~309 tokens (70% reduction)
- Added comment block documenting which queries reach the parser vs are handled earlier
- No Unicode characters in file (verified)
- Graph compilation and 27 existing tests pass with no regressions

---

### Issue 5: No .env.local file exists

**Root cause:** File does not exist on disk
**Impact:** No local environment overrides available

**Status:** [x] Complete

**Changes (2026-02-08):**
- Added `.env.local` to `.gitignore` (machine-specific, never committed)
- Updated `atlas_brain/main.py`: loads `.env` then `.env.local` with `override=True`
- Updated `atlas_edge/main.py`: same pattern
- `.env.local` is optional -- `load_dotenv` silently returns `False` for missing files
- Override verified: `.env.local` values correctly override `.env` values
- Pydantic `BaseSettings` classes read from `os.environ` first (populated by `load_dotenv`), so no `config.py` changes needed
- Usage: create `.env.local` with machine-specific overrides (API keys, ports, debug flags)

---

## Intelligence Breakdown

| Layer | Method | Latency | Used For |
|-------|--------|---------|----------|
| Semantic Router | Embedding cosine sim | ~10ms | Route classification (all queries) |
| Device Resolver | Embedding cosine sim | ~20ms | Device name matching (device commands) |
| Keyword Extraction | String matching | <1ms | Action parsing (turn on/off/toggle) |
| LLM Fallback Router | Qwen3-30b | ~500ms | Low-confidence route classification |
| LLM Intent Parser | Qwen3-30b | ~14s | Parameter extraction (slow path) |
| LLM Field Extraction | Qwen3-30b | ~2-5s | Booking workflow field extraction |
| LLM Response Gen | Qwen3-30b | ~2-8s | Final response generation |

### Latency by Path

- Fast path (6 tools): Router(10ms) -> Execute -> Respond = ~200ms total
- Device commands: Router(10ms) -> DeviceResolver(20ms) -> Execute = ~50ms total
- Workflows: Router(10ms) -> Workflow(LLM calls) = ~5-15s
- Everything else: Router -> LLM Parse(14s) -> Execute -> LLM Respond = ~20s+

---

## File Reference

| File | Role |
|------|------|
| `atlas_brain/tools/__init__.py` | Tool registration (26 tools) |
| `atlas_brain/tools/registry.py` | ToolRegistry singleton |
| `atlas_brain/services/intent_router.py` | ROUTE_DEFINITIONS, ROUTE_TO_ACTION, ROUTE_TO_WORKFLOW, PARAMETERLESS_TOOLS |
| `atlas_brain/agents/graphs/atlas.py` | Main agent graph, classify_and_route, start_workflow, continue_workflow |
| `atlas_brain/agents/graphs/booking.py` | Booking workflow |
| `atlas_brain/agents/graphs/reminder.py` | Reminder workflow |
| `atlas_brain/agents/graphs/email.py` | Email workflow |
| `atlas_brain/agents/graphs/calendar.py` | Calendar workflow |
| `atlas_brain/agents/graphs/security.py` | Security workflow |
| `atlas_brain/agents/graphs/presence.py` | Presence workflow |
| `atlas_brain/capabilities/intent_parser.py` | UNIFIED_INTENT_PROMPT, IntentParser |
| `atlas_brain/voice/launcher.py` | _PREFILL_SYSTEM_PROMPT |
