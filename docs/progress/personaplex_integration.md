# PersonaPlex Integration for Receptionist Mode

**Date Started:** 2026-01-20
**Status:** Implementation Complete (Pending Verification)
**Branch:** personaplex-integration

## Executive Summary

Integrate NVIDIA PersonaPlex (speech-to-speech conversational AI) into the Atlas receptionist mode for handling business phone calls. PersonaPlex replaces the current STT → LLM → TTS pipeline with a unified, low-latency voice model while preserving existing tool-calling infrastructure.

## Background

### Current Architecture (phone calls)
```
SignalWire (mulaw 8kHz)
    ↓ WebSocket
webhooks.py:handle_audio_stream()
    ↓
PhoneCallProcessor.process_audio_chunk()
    ├── VAD (Voice Activity Detection)
    ├── mulaw → PCM conversion
    ├── STT (Nemotron) → transcript
    ├── ReceptionistAgent.run() → response text
    ├── TTS (Kokoro) → audio
    └── PCM → mulaw conversion
    ↓
SignalWire (mulaw 8kHz)
```

### Target Architecture (with PersonaPlex)
```
SignalWire (mulaw 8kHz)
    ↓ WebSocket
webhooks.py:handle_audio_stream()
    ↓
PersonaPlexProcessor (NEW)
    ├── mulaw 8kHz → PCM 24kHz resampling
    ├── PCM → Opus encoding
    ├── PersonaPlex WebSocket (voice in → voice out)
    │       ↓ emits text tokens (0x02)
    ├── ToolBridge monitors text → triggers tools
    │       ├── check_availability
    │       ├── book_appointment
    │       └── send_confirmation_email
    ├── Tool results injected into PersonaPlex text_prompt
    ├── Opus → PCM decoding
    └── PCM 24kHz → mulaw 8kHz resampling
    ↓
SignalWire (mulaw 8kHz)
```

## PersonaPlex Details

- **Model:** nvidia/personaplex-7b-v1 (7B params, bf16)
- **VRAM:** ~19GB (tested on RTX 3090, works but tight)
- **Audio:** 24kHz, Opus codec
- **Latency:** ~170ms turn-taking, ~240ms interrupt
- **Protocol:** WebSocket at `wss://host:8998/api/chat`
- **Configuration via query params:**
  - `text_prompt`: Persona/instructions
  - `voice_prompt`: Voice style (NATF0-3, NATM0-3, etc.)
  - `seed`: Reproducibility

### Message Protocol
| Byte | Direction | Type |
|------|-----------|------|
| 0x00 | Server→Client | Handshake |
| 0x01 | Both | Audio (Opus) |
| 0x02 | Server→Client | Text token |

## Files Analysis

### Files to CREATE
| File | Purpose |
|------|---------|
| `atlas_brain/services/personaplex/__init__.py` | PersonaPlex service module |
| `atlas_brain/services/personaplex/service.py` | PersonaPlex client wrapper |
| `atlas_brain/services/personaplex/config.py` | PersonaPlex configuration |
| `atlas_brain/services/personaplex/audio.py` | Audio conversion utilities |
| `atlas_brain/comms/personaplex_processor.py` | Phone call processor using PersonaPlex |
| `atlas_brain/comms/tool_bridge.py` | Monitors PersonaPlex text, triggers tools |

### Files to MODIFY
| File | Lines | Change |
|------|-------|--------|
| `atlas_brain/config.py` | ~end | Add PersonaPlexConfig section |
| `atlas_brain/api/comms/webhooks.py` | 478-618 | Add PersonaPlex processor option |
| `atlas_brain/comms/config.py` | ~end | Add personaplex_enabled flag |

### Files UNCHANGED (no modifications needed)
| File | Reason |
|------|--------|
| `atlas_brain/agents/receptionist.py` | Keep for fallback/reference |
| `atlas_brain/comms/phone_processor.py` | Keep as legacy option |
| `atlas_brain/tools/scheduling.py` | Used by ToolBridge as-is |
| `atlas_brain/comms/providers/signalwire_provider.py` | No changes needed |

## Dependency Analysis

### New Dependencies Required
```
sphn>=0.1.4    # Opus audio codec (already in personaplex venv)
```

### Existing Dependencies Used
```
audioop        # stdlib - mulaw conversion
asyncio        # stdlib - async WebSocket
aiohttp        # PersonaPlex server uses this
```

### VRAM Considerations
- PersonaPlex needs ~19GB VRAM
- Cannot run alongside Nemotron STT + Kokoro TTS
- Recommendation: Run PersonaPlex on dedicated GPU server (cloud)

## Phased Implementation Plan

### Phase 1: PersonaPlex Service Wrapper
**Goal:** Create standalone PersonaPlex client that can connect and stream audio

**Files:**
- `atlas_brain/services/personaplex/__init__.py`
- `atlas_brain/services/personaplex/service.py`
- `atlas_brain/services/personaplex/config.py`
- `atlas_brain/services/personaplex/audio.py`

**Deliverables:**
- [x] PersonaPlexConfig with host, port, voice_prompt, text_prompt
- [x] PersonaPlexService class with connect/disconnect
- [x] Audio conversion: PCM 8kHz mulaw ↔ PCM 24kHz ↔ Opus
- [x] WebSocket client for PersonaPlex protocol
- [x] Text token callback mechanism

**Verification:**
- [ ] Can connect to PersonaPlex server
- [ ] Can send audio and receive audio response
- [ ] Can receive text tokens

### Phase 2: Tool Bridge
**Goal:** Monitor PersonaPlex conversation and trigger tools

**Files:**
- `atlas_brain/comms/tool_bridge.py`

**Deliverables:**
- [x] ToolBridge class that accumulates text tokens
- [x] Intent detection from accumulated text
- [x] Tool parameter extraction
- [x] Tool execution via existing tool_registry
- [x] Result formatting for injection into conversation

**Verification:**
- [ ] Detects booking intent from phrases like "let me book that"
- [ ] Extracts name, address, date/time from conversation
- [ ] Successfully calls book_appointment tool
- [ ] Handles tool errors gracefully

### Phase 3: PersonaPlex Phone Processor
**Goal:** Replace PhoneCallProcessor for PersonaPlex mode

**Files:**
- `atlas_brain/comms/personaplex_processor.py`

**Deliverables:**
- [x] PersonaPlexProcessor class matching PhoneCallProcessor interface
- [x] Integration with PersonaPlexService
- [x] Integration with ToolBridge
- [x] Bidirectional audio streaming
- [x] Call state management

**Verification:**
- [ ] Receives audio from SignalWire
- [ ] Sends audio to PersonaPlex
- [ ] Receives PersonaPlex audio response
- [ ] Sends audio back to SignalWire
- [ ] Tools triggered correctly during conversation

### Phase 4: Configuration & Webhook Integration
**Goal:** Wire PersonaPlex into the webhook handler

**Files to modify:**
- `atlas_brain/config.py` (add PersonaPlexConfig)
- `atlas_brain/comms/config.py` (add personaplex_enabled)
- `atlas_brain/api/comms/webhooks.py` (add processor selection)

**Deliverables:**
- [x] PersonaPlexConfig in main config (in services/personaplex/config.py)
- [x] Environment variable support (ATLAS_PERSONAPLEX_*)
- [x] personaplex_enabled toggle in comms config
- [x] Webhook handler selects processor based on config

**Verification:**
- [ ] Config loads correctly from environment
- [ ] personaplex_enabled=true uses PersonaPlexProcessor
- [ ] personaplex_enabled=false uses PhoneCallProcessor (legacy)
- [ ] No breaking changes to existing functionality

### Phase 5: Business Context Integration
**Goal:** Pass business persona to PersonaPlex

**Deliverables:**
- [x] Build text_prompt from BusinessContext
- [x] Voice selection based on context config
- [x] Dynamic prompt updates for tool results (via ToolBridge callback)

**Verification:**
- [ ] PersonaPlex uses correct business persona
- [ ] Voice matches configured preference
- [ ] Tool results appear in conversation naturally

## Breaking Change Analysis

### Risk Assessment
| Change | Risk | Mitigation |
|--------|------|------------|
| New processor | Low | Feature flag, legacy processor remains |
| Config additions | Low | Additive only, no existing fields changed |
| Webhook modification | Medium | Conditional logic, existing path unchanged |
| New dependencies | Low | Optional, only loaded when enabled |

### Rollback Plan
1. Set `personaplex_enabled=false` in config
2. System falls back to existing PhoneCallProcessor
3. No code changes required for rollback

## Testing Strategy

### Unit Tests
- [ ] PersonaPlexService: connection, audio send/receive
- [ ] ToolBridge: intent detection, tool execution
- [ ] Audio conversion: sample rate, codec

### Integration Tests
- [ ] End-to-end call with PersonaPlex
- [ ] Tool execution during call
- [ ] Fallback to legacy processor

### Manual Testing
- [ ] Real phone call via SignalWire
- [ ] Verify voice quality
- [ ] Verify latency (<1s round-trip)
- [ ] Verify appointment booking works

## Open Questions

1. **Hosting:** Where to run PersonaPlex? (cloud GPU recommended)
2. **Failover:** What happens if PersonaPlex server is down?
3. **Cost:** PersonaPlex GPU cost vs current STT+LLM+TTS cost?
4. **Voice consistency:** Can we match current TTS voice quality?

## Progress Log

### 2026-01-20
- Explored PersonaPlex repo and documentation
- Tested PersonaPlex on RTX 3090 (works, 19GB VRAM)
- Created custom loader for CPU->GPU model loading
- Analyzed Atlas codebase for integration points
- Created this planning document
- Identified exact files and insertion points
- **Phase 1 Complete:**
  - Created `atlas_brain/services/personaplex/` directory
  - Implemented `config.py` with PersonaPlexConfig (env vars: ATLAS_PERSONAPLEX_*)
  - Implemented `audio.py` with AudioConverter (mulaw 8kHz <-> PCM 24kHz <-> Opus)
  - Implemented `service.py` with PersonaPlexService WebSocket client
  - All deliverables complete, pending verification tests
- **Phase 2 Complete:**
  - Created `atlas_brain/comms/tool_bridge.py`
  - ToolBridge class with ConversationContext tracking
  - Pattern-based entity extraction (name, phone, address, date, time)
  - Booking intent detection with trigger patterns
  - Tool execution via tool_registry.execute()
  - Result formatting for speech injection
- **Phase 3 Complete:**
  - Created `atlas_brain/comms/personaplex_processor.py`
  - PersonaPlexProcessor with PersonaPlexCallState
  - Integrates PersonaPlexService for WebSocket communication
  - Integrates ToolBridge for tool detection and execution
  - Bidirectional audio: mulaw 8kHz <-> Opus 24kHz
  - Module-level functions: create/get/remove processor, is_enabled check
- **Phase 4 Complete:**
  - Added `personaplex_enabled` to CommsConfig in `atlas_brain/comms/config.py`
  - Modified `atlas_brain/api/comms/webhooks.py` to support both modes
  - Conditional imports based on comms_settings.personaplex_enabled
  - PersonaPlex uses callback for audio responses, legacy uses return value
  - PersonaPlex generates greeting via text_prompt, legacy uses TTS
- **Phase 5 Complete:**
  - Enhanced `_build_text_prompt()` to include persona, pricing, and service area
  - Added `_get_voice_prompt()` for voice selection based on context
  - Updated PersonaPlexService.connect() to accept voice_prompt parameter
  - Voice mapping: female names -> NATF0, male names -> NATM0
- **Gap Analysis Complete:**
  - Created `atlas_brain/services/omni/__init__.py` (fixes import error)
  - Added error handling to `send_audio_mulaw()` in service.py
  - Added error handling to `_on_audio_response()` in personaplex_processor.py
  - Added connection failure handling in webhooks.py (removes processor on fail)
  - Fixed `is_personaplex_enabled()` to use correct config source (comms_settings)
  - Removed unused `enabled` field from PersonaPlexConfig (dead code)
  - Verified no Unicode characters in new Python files
  - All Python syntax validated
  - All module imports verified working

## References

- PersonaPlex repo: `/home/juan-canfield/Desktop/live-translator/personaplex/`
- Model weights: `~/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/`
- Custom server script: `/home/juan-canfield/Desktop/live-translator/personaplex/run_server.py`
- Atlas comms docs: `docs/progress/external_comms_system.md`
- Atlas scheduling docs: `docs/progress/scheduling_architecture.md`
