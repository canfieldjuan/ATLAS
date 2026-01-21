# Scheduling Architecture Decision Log

**Date:** 2026-01-19 (Updated: 2026-01-20)
**Status:** Implemented

## Current State

### ReceptionistAgent (Phone Mode)
- Designed for **inbound customer calls**
- Multi-turn conversational flow:
  1. Greeting phase
  2. Info collection (name, address, time preference)
  3. Confirmation phase
  4. Booking execution via `book_appointment` tool
- Works but has model/VRAM constraints
- Was functional enough for MVP consideration

### AtlasAgent (Voice Commands)
- Handles direct tool execution
- Single-turn commands: "what time is it" → executes `get_time`
- Should handle direct scheduling commands

## Problem Identified

Voice commands like "create an appointment for John Smith tomorrow at 9am" go through AtlasAgent but scheduling tools expect the phone conversation flow context (caller_name, service_address collected over multiple turns).

## Desired Architecture

### 1. Inbound Customer Calls (Future)
**Options evaluated:**
- ~~SignalWire service~~ - Would handle conversation but less control
- **NVIDIA PersonaFlex** - Voice agent model with:
  - Back-and-forth conversation capability
  - Built-in voice synthesis
  - Character/persona setting
  - Handles the hard multi-turn conversational problem

**Decision:** NVIDIA PersonaFlex for voice agent layer
- Deploy on serverless GPU in cloud
- Use Together AI MOE model for tool calling (reasoning layer)
- Current limitation: Uses all VRAM locally, needs CPU offloading
- Architecture: PersonaFlex (voice) -> Together AI (tools) -> Atlas Tool Registry

### 2. Direct Voice Commands (Atlas)
- User says: "Book an appointment for John Smith at 123 Main St tomorrow morning"
- AtlasAgent detects `book_appointment` tool
- Executes directly with parsed parameters
- No multi-turn conversation needed

## Implementation Notes

### For Direct Scheduling via Atlas
The `book_appointment` tool should accept:
- `customer_name` - extracted from command
- `address` - extracted from command
- `date` - parsed from "tomorrow", "next Monday", etc.
- `time` - parsed from "morning", "9am", etc.
- `service_type` - default to "Free Estimate" or specified

### Intent Parser Updates Needed
Add examples for direct booking:
```
"book appointment for John Smith tomorrow" → book_appointment tool
"schedule estimate at 123 Main St on Monday" → book_appointment tool
```

## References

- ReceptionistAgent: `atlas_brain/agents/receptionist.py`
- Scheduling tools: `atlas_brain/tools/scheduling.py`
- Phone providers: `atlas_brain/comms/providers/`

## Testing Results (2026-01-19)

### Direct Booking via AtlasAgent - WORKS
```
Query: "book an appointment for John Smith at 123 Main Street tomorrow at 9am"
Action Type: tool_use
Response: Calendar not configured. Set ATLAS_COMMS_EFFINGHAM_MAIDS_CALENDAR_ID.
```
- Intent parsing correctly detects `book_appointment` tool
- Tool executes (config error is expected - calendar not configured in test env)
- No multi-turn conversation needed

## Implementation Plan: Separate Phone Flow from Voice Commands

**Date:** 2026-01-19
**Status:** COMPLETED

### Analysis Summary

**Current Issue:** ReceptionistAgent is used in TWO places:
1. **Mode routing** (`atlas_brain/agents/atlas.py:118`) - used for voice commands via mode system
2. **Phone calls** (`atlas_brain/comms/phone_processor.py:88`) - used for inbound customer calls

**Files Referencing ReceptionistAgent:**
- `atlas_brain/agents/atlas.py:114,118` - imports and maps to ModeType.RECEPTIONIST
- `atlas_brain/modes/manager.py:170-173` - aliases "business", "scheduling", "appointment" to receptionist
- `atlas_brain/modes/config.py:16,59` - ModeType.RECEPTIONIST definition and tool config
- `atlas_brain/comms/phone_processor.py:88` - creates agent for phone calls (KEEP)
- `atlas_brain/api/comms/webhooks.py` - webhook processing (KEEP - uses phone_processor)

### Phase 1: Remove ReceptionistAgent from Voice Mode Routing

**Files to modify:**
1. `atlas_brain/agents/atlas.py`
   - Remove line 114: `from .receptionist import get_receptionist_agent`
   - Remove line 118: `ModeType.RECEPTIONIST: get_receptionist_agent(),`

**Result:** When mode is RECEPTIONIST, AtlasAgent handles it directly (line 186-187 fallback)

### Phase 2: No Changes Needed

The following should remain unchanged:
- `atlas_brain/modes/config.py` - ModeType.RECEPTIONIST keeps scheduling tools
- `atlas_brain/modes/manager.py` - aliases keep routing to RECEPTIONIST mode (for tool filtering)
- `atlas_brain/comms/phone_processor.py` - phone calls use `create_receptionist_agent()` directly

### Verification Checklist

- [x] Direct scheduling commands via voice work (AtlasAgent handles)
- [x] Phone calls still work (phone_processor.py creates ReceptionistAgent directly)
- [x] Mode switching still works ("switch to scheduling mode")
- [x] Scheduling tools available in RECEPTIONIST mode (16 tools including 5 scheduling tools)
- [x] No import errors or circular dependencies

### Verification Results (2026-01-19)

```
Mode switching: receptionist mode works
Scheduling tools in receptionist mode: check_availability, book_appointment,
  cancel_appointment, reschedule_appointment, lookup_customer
Mode agents in AtlasAgent: [HOME only] - RECEPTIONIST not in dict (uses fallback)
Imports verified: AtlasAgent OK, PhoneCallProcessor OK
```

## All Modes Verification (2026-01-19)

**Status:** VERIFIED CORRECT

All modes now follow the same pattern as RECEPTIONIST (no dedicated agent, AtlasAgent handles directly):

| Mode | Dedicated Agent | Tools | Status |
|------|-----------------|-------|--------|
| HOME | HomeAgent | 8 (4 presence + 4 shared) | Has agent |
| RECEPTIONIST | None (fallback) | 16 | AtlasAgent direct |
| COMMS | None (fallback) | 10 | AtlasAgent direct |
| SECURITY | None (fallback) | 19 | AtlasAgent direct |
| CHAT | None (fallback) | 4 (shared only) | AtlasAgent direct |

**Key Files:**
- `atlas_brain/agents/atlas.py:118-120` - `_mode_agents` only contains HOME
- `atlas_brain/agents/atlas.py:186-188` - Fallback to AtlasAgent for modes without agents
- `atlas_brain/tools/__init__.py` - All 36 tools registered
- `atlas_brain/modes/config.py` - Mode configurations with tool lists

**Architecture Pattern:**
1. User speaks command
2. ModeManager determines current mode
3. AtlasAgent checks `_mode_agents` for dedicated agent
4. If found (HOME only): delegate to HomeAgent
5. If not found: AtlasAgent handles directly with mode-specific tools

## Completed Items

1. ~~Test direct `book_appointment` via AtlasAgent~~ DONE - works
2. ~~Verify all modes follow same pattern~~ DONE - verified
3. ~~Evaluate SignalWire vs NVIDIA model for inbound calls~~ DONE - PersonaPlex selected
4. ~~Add cloud LLM provider~~ DONE - Together AI added (2026-01-20)
5. ~~STT/TTS auto-load on startup~~ DONE - loads by default now
6. ~~Configure calendar for production use~~ DONE - `ATLAS_COMMS_EFFINGHAM_MAIDS_CALENDAR_ID` set

## PersonaPlex Integration Decision (2026-01-20)

**Decision:** Integrate NVIDIA PersonaPlex for inbound business calls

**Rationale:**
- PersonaPlex is a 7B speech-to-speech model with ~170ms latency
- Handles natural conversation, interruptions, turn-taking natively
- Replaces separate STT + LLM + TTS pipeline with unified model
- Better voice quality and more natural conversation flow

**Implementation:**
- PersonaPlex handles voice layer (listen + speak)
- Existing tool infrastructure (book_appointment, etc.) used via ToolBridge
- ToolBridge monitors PersonaPlex text output, triggers tools when needed
- Feature-flagged: `personaplex_enabled` toggle in config

**Hardware Requirement:**
- ~19GB VRAM (tested on RTX 3090)
- Recommendation: Cloud GPU (A100) for production

**See:** `docs/progress/personaplex_integration.md` for full implementation plan

## Remaining Items

1. Build PersonaPlex pipeline for serverless GPU deployment
2. Integrate PersonaPlex with Together AI for tool calling
3. Potentially deprecate ReceptionistAgent if PersonaPlex handles all phone flows
