# Communications Extraction Progress Log

**Created**: 2026-01-28
**Branch**: brain-extraction
**Status**: IN PROGRESS

---

## Context

The comms module handles telephony (phone calls, SMS) via Twilio/SignalWire providers. It includes:
- Provider abstraction layer
- Business context routing
- Appointment scheduling
- Voice pipeline integration for phone calls

### Complexity Analysis

**AI Dependencies in comms:**
```
phone_processor.py → agents (create_receptionist_agent)
phone_processor.py → services (stt_registry, tts_registry)
personaplex_processor.py → services/personaplex
tool_bridge.py → tools (tool_registry)
real_services.py → tools/calendar
scheduling.py → tools/calendar
```

**Challenge**: Phone calls need real-time AI (STT/TTS/LLM) for voice conversation.

---

## Source Analysis

### atlas_brain/comms/ Files

| File | Lines | AI Deps? | Purpose |
|------|-------|----------|---------|
| `config.py` | 270 | No | CommsConfig, BusinessContext |
| `protocols.py` | 301 | No | Call, SMS, Provider protocols |
| `context.py` | 252 | No | Context routing |
| `services.py` | 451 | No | CalendarService, EmailService stubs |
| `real_services.py` | 503 | Tools | Google Calendar, Resend, SignalWire SMS |
| `scheduling.py` | 465 | Tools | Appointment scheduling |
| `service.py` | 351 | No | Main CommsService orchestrator |
| `providers/__init__.py` | 66 | No | Provider factory |
| `providers/twilio_provider.py` | 464 | No | Twilio implementation |
| `providers/signalwire_provider.py` | 523 | No | SignalWire implementation |
| `phone_processor.py` | 430 | **YES** | Voice call processing (STT/LLM/TTS) |
| `personaplex_processor.py` | 239 | **YES** | PersonaPlex speech-to-speech |
| `tool_bridge.py` | 228 | **YES** | Tool execution during calls |
| **Total** | ~4,543 | | |

### atlas_brain/api/comms/ Files

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 17 | Router exports |
| `management.py` | 403 | Admin endpoints |
| `webhooks.py` | 742 | Twilio/SignalWire webhooks |
| **Total** | 1,162 | |

---

## Extraction Strategy

### Option A: Full Extraction with AI Callbacks

Move ALL comms code to `atlas_comms` service. For phone calls:
- atlas_comms receives webhook
- Streams audio to atlas_brain STT API
- Gets LLM response from atlas_brain
- Streams TTS back from atlas_brain

**Pros**: Clean separation, comms fully independent
**Cons**: Higher latency for voice calls, complex streaming

### Option B: Hybrid - Core to atlas_comms, Phone Processor stays

Move to atlas_comms:
- config, protocols, context, services
- real_services, scheduling
- providers (Twilio, SignalWire)

Keep in atlas_brain:
- phone_processor (needs local AI)
- personaplex_processor
- tool_bridge

**Pros**: Low latency for calls, simpler
**Cons**: Split module, comms depends on brain

### Recommended: Option A with HTTP streaming

Create atlas_comms as standalone service:
1. All comms code moves to atlas_comms
2. Phone processor makes HTTP calls to atlas_brain for:
   - STT: `POST /api/v1/stt/stream`
   - LLM: `POST /api/v1/llm/chat`
   - TTS: `POST /api/v1/tts/stream`
3. SMS auto-reply can use atlas_brain LLM or own logic
4. Scheduling is fully independent (Google Calendar API)

---

## Implementation Plan

### Phase 1: Create atlas_comms Service

**Directory structure:**
```
atlas_comms/
├── __init__.py
├── __main__.py
├── core/
│   ├── __init__.py
│   ├── config.py          # From comms/config.py
│   └── protocols.py       # From comms/protocols.py
├── context/
│   ├── __init__.py
│   └── router.py          # From comms/context.py
├── services/
│   ├── __init__.py
│   ├── base.py            # From comms/services.py
│   ├── calendar.py        # From comms/real_services.py (calendar)
│   ├── email.py           # From comms/real_services.py (email)
│   ├── sms.py             # From comms/real_services.py (sms)
│   └── scheduling.py      # From comms/scheduling.py
├── providers/
│   ├── __init__.py
│   ├── base.py
│   ├── twilio.py          # From providers/twilio_provider.py
│   └── signalwire.py      # From providers/signalwire_provider.py
├── phone/
│   ├── __init__.py
│   ├── processor.py       # From comms/phone_processor.py (adapted)
│   ├── personaplex.py     # From comms/personaplex_processor.py (adapted)
│   └── tool_bridge.py     # From comms/tool_bridge.py (adapted)
├── api/
│   ├── __init__.py
│   ├── main.py            # FastAPI app
│   ├── webhooks.py        # From api/comms/webhooks.py
│   └── management.py      # From api/comms/management.py
└── service.py             # From comms/service.py
```

### Phase 2: Adapt AI Dependencies

For phone processor, create HTTP clients to atlas_brain:
- `AtlasBrainSTTClient` - streams audio, gets transcripts
- `AtlasBrainLLMClient` - sends messages, gets responses
- `AtlasBrainTTSClient` - sends text, streams audio

### Phase 3: Update atlas_brain

- Remove comms/ directory
- Update main.py - no local comms init
- Update api/ - remove comms router
- Keep tools/scheduling.py → calls atlas_comms API

### Phase 4: API Endpoints for atlas_comms

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/webhooks/voice` | POST | Incoming call webhook |
| `/webhooks/voice/status` | POST | Call status callback |
| `/webhooks/sms` | POST | Incoming SMS webhook |
| `/calls` | GET | List active calls |
| `/calls/{id}` | GET | Get call details |
| `/calls/{id}/hangup` | POST | Hang up call |
| `/calls/outbound` | POST | Make outbound call |
| `/sms` | POST | Send SMS |
| `/contexts` | GET | List business contexts |
| `/scheduling/available` | GET | Get available slots |
| `/scheduling/book` | POST | Book appointment |

---

## Session Log

### 2026-01-28 - Planning

1. Analyzed comms module structure (14 files, ~4,500 lines)
2. Identified AI dependencies in phone_processor, personaplex_processor, tool_bridge
3. Decided on Option A - full extraction with HTTP callbacks for AI
4. Created this progress log

---

## Progress

### 2026-01-28 - Phase 1 Started

**Files created in atlas_comms:**
- `__init__.py`, `__main__.py` - Module entry points
- `service.py` - Main CommsService
- `core/config.py` - CommsConfig, BusinessContext (ATLAS_COMMS_*)
- `core/protocols.py` - Call, SMS, TelephonyProvider protocols
- `core/__init__.py` - Core exports
- `context/__init__.py` - ContextRouter
- `providers/__init__.py` - Provider registry
- `providers/twilio.py` - Twilio provider
- `providers/signalwire.py` - SignalWire provider
- `services/__init__.py`, `services/base.py` - Service stubs
- `api/main.py` - FastAPI app
- `api/health.py` - Health endpoints
- `api/calls.py` - Call management endpoints
- `api/sms.py` - SMS endpoints
- `api/contexts.py` - Business context endpoints

**API Endpoints** (atlas_comms):
- `GET /health` - Service health
- `GET /ping` - Ping
- `GET /calls` - List active calls
- `GET /calls/{id}` - Get call details
- `POST /calls/outbound` - Make outbound call
- `POST /calls/{id}/hangup` - Hang up call
- `POST /calls/{id}/transfer` - Transfer call
- `POST /sms/send` - Send SMS
- `GET /contexts` - List business contexts
- `GET /contexts/{id}` - Get context details
- `GET /contexts/{id}/status` - Get open/closed status

**Remaining:**
- [ ] Phone processor (deferred - requires AI integration)
- [ ] Webhooks for incoming calls/SMS (deferred - part of phone processing)
- [x] Scheduling service - COMPLETE
- [ ] Update atlas_brain to proxy/remove comms
- [x] Calendar integration - COMPLETE (via scheduling service)

### 2026-01-28 - Scheduling Service Added

**Files created:**
- `services/scheduling.py` - SchedulingService with Google Calendar integration
- `api/scheduling.py` - Scheduling API endpoints

**Files updated:**
- `core/config.py` - Added CalendarConfig class with ATLAS_COMMS_CALENDAR_* env vars
- `services/__init__.py` - Added SchedulingService exports
- `api/main.py` - Added scheduling router

**New API Endpoints:**
- `GET /scheduling/available` - Get available appointment slots
- `POST /scheduling/book` - Book an appointment
- `POST /scheduling/cancel` - Cancel an appointment

**Environment Variables (calendar):**
- `ATLAS_COMMS_CALENDAR_ENABLED` - Enable calendar integration
- `ATLAS_COMMS_CALENDAR_CLIENT_ID` - Google OAuth client ID
- `ATLAS_COMMS_CALENDAR_CLIENT_SECRET` - Google OAuth client secret
- `ATLAS_COMMS_CALENDAR_REFRESH_TOKEN` - Google OAuth refresh token

---

## Current Status Summary

### Completed in atlas_comms

| Component | Files | Status |
|-----------|-------|--------|
| Core config | `core/config.py`, `core/protocols.py` | COMPLETE |
| Context routing | `context/__init__.py` | COMPLETE |
| Providers | `providers/twilio.py`, `providers/signalwire.py` | COMPLETE |
| Services base | `services/base.py`, `services/__init__.py` | COMPLETE |
| Scheduling | `services/scheduling.py`, `api/scheduling.py` | COMPLETE |
| API endpoints | `api/main.py`, `api/health.py`, `api/calls.py`, `api/sms.py`, `api/contexts.py` | COMPLETE |
| Main service | `service.py` | COMPLETE |

### Deferred (Stays in atlas_brain)

| Component | Reason |
|-----------|--------|
| `phone_processor.py` | Requires local STT/TTS/LLM (AI integration) |
| `personaplex_processor.py` | Requires PersonaPlex service (AI integration) |
| `tool_bridge.py` | Requires tool_registry (AI integration) |
| `api/comms/webhooks.py` | Part of phone processing |

### atlas_brain/comms/ - Now Compatibility Layer

All core comms files now re-export from atlas_comms for backward compatibility:

| File | Status |
|------|--------|
| `__init__.py` | Re-exports from atlas_comms |
| `config.py` | Re-exports from atlas_comms.core.config |
| `context.py` | Re-exports from atlas_comms.context |
| `protocols.py` | Re-exports from atlas_comms.core.protocols |
| `services.py` | Re-exports from atlas_comms.services |
| `scheduling.py` | Re-exports from atlas_comms.services |
| `service.py` | Re-exports from atlas_comms |
| `providers/__init__.py` | Re-exports from atlas_comms.providers |
| `providers/twilio_provider.py` | DELETED (using atlas_comms) |
| `providers/signalwire_provider.py` | DELETED (using atlas_comms) |
| `real_services.py` | LOCAL - email/SMS implementations |
| `phone_processor.py` | LOCAL - requires atlas_brain AI |
| `personaplex_processor.py` | LOCAL - requires atlas_brain AI |
| `tool_bridge.py` | LOCAL - requires atlas_brain tools |

### Completed Steps

1. [x] Update `atlas_brain/comms/__init__.py` to re-export from atlas_comms
2. [x] Update all submodule files to re-export from atlas_comms
3. [x] Delete redundant provider implementation files
4. [x] Verify all imports work correctly

### Remaining (Deferred)

- Phone processing (phone_processor.py, personaplex_processor.py, tool_bridge.py)
- Webhooks (api/comms/webhooks.py)
- real_services.py (email/SMS implementations)

---

## 2026-01-28 - Comms Finalization Complete

**Changes made:**

1. Updated `atlas_brain/comms/__init__.py`:
   - Now imports from atlas_comms instead of local modules
   - Re-exports everything for backward compatibility

2. Updated submodule files to re-export from atlas_comms:
   - `config.py` - re-exports from atlas_comms.core.config
   - `protocols.py` - re-exports from atlas_comms.core.protocols
   - `context.py` - re-exports from atlas_comms.context
   - `services.py` - re-exports from atlas_comms.services
   - `scheduling.py` - re-exports from atlas_comms.services
   - `service.py` - re-exports from atlas_comms
   - `providers/__init__.py` - re-exports from atlas_comms.providers

3. Deleted redundant files:
   - `providers/twilio_provider.py`
   - `providers/signalwire_provider.py`

4. Kept local files (require atlas_brain internals):
   - `real_services.py` - uses atlas_brain config for OAuth
   - `phone_processor.py` - uses STT/TTS/LLM registries
   - `personaplex_processor.py` - uses PersonaPlex service
   - `tool_bridge.py` - uses tool_registry

**Validation:**
- [x] All comms imports work correctly
- [x] tools/scheduling.py imports work
- [x] main.py style imports work
- [x] Syntax valid for all updated files
