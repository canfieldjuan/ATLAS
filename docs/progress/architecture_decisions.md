# Architecture Decisions Log

**Created**: 2026-01-28
**Branch**: brain-extraction

---

## Decision 1: Brain as Orchestrator, Not Middleman

**Date**: 2026-01-28

### Context

Atlas brain was acting as a proxy/middleman for everything, creating a centralization bottleneck.

### Decision

Brain responsibilities (KEEP):
- Intent classification for complex queries needing LLM reasoning
- LLM reasoning and tool execution
- Maintaining global state (PostgreSQL: sessions, conversations, users, presence aggregation)
- Cross-location coordination ("heading to office 102" triggers prep there)
- Training/updating edge models, pushing config changes

Brain should NOT do (DELEGATE):
- Be in hot path for device control (direct to Home Assistant)
- Run STT/TTS for local voice (edge devices own this)
- Proxy everything through itself

### Implementation

- Edge devices talk directly to Home Assistant for local automation
- Modules communicate peer-to-peer when appropriate
- Brain only handles complex reasoning requests

---

## Decision 2: capabilities/ Module - Cross-Location Only

**Date**: 2026-01-28

### Context

`atlas_brain/capabilities/` contains Home Assistant integration, device control, intent parsing.

### Decision

KEEP `capabilities/` in brain ONLY for:
1. Cross-location coordination (actions that span multiple HA instances)
2. Complex multi-step reasoning about devices
3. Global state queries ("is anyone home?" across all locations)

Device commands that don't need reasoning should go direct to HA from edge devices.

### Files Affected

- `capabilities/homeassistant.py` - Keep for cross-location
- `capabilities/intent_parser.py` - Keep for complex intents
- `capabilities/actions.py` - Keep for coordination logic
- Direct device commands - Edge devices should call HA directly

---

## Decision 3: Voice Pipeline - Piper TTS, Not Kokoro

**Date**: 2026-01-28

### Context

Code has two TTS paths:
1. `tts_registry` (default: Piper) - used by phone_processor, alerts, API
2. `pipecat/pipeline.py` hardcodes Kokoro - potentially unused

### Decision

Piper is the actual TTS in use (runs on CPU). The Pipecat pipeline with Kokoro is either unused or optional.

Actual voice pipeline uses:
- STT: `stt_registry.get_active()` -> Nemotron (GPU)
- TTS: `tts_registry.get_active()` -> Piper (CPU)
- LLM: `llm_registry.get_active()` -> llama-cpp or Ollama

---

## Decision 4: Presence to Home Assistant (Future)

**Date**: 2026-01-28

### Context

Presence tracking currently in atlas_vision (recently extracted from brain).

### Decision

Long-term: Presence should be in Home Assistant, not atlas_vision.
- HA already has presence detection integrations
- Brain should query HA for "who's where" rather than owning presence
- For now, atlas_vision owns it (recently migrated there)

### Future Work

- Evaluate HA presence integrations
- Migrate presence from atlas_vision to HA
- Brain queries HA for presence state

---

## Extraction Status Summary

| Module | Status | Target |
|--------|--------|--------|
| vision/ | COMPLETE | atlas_vision |
| presence/ | COMPLETE | atlas_vision (future: HA) |
| comms/ | IN PROGRESS | atlas_comms |
| services/stt,tts | Future | Edge devices |
| pipecat/ | Future | Edge devices |
| capabilities/ | Keep (limited) | Cross-location only |
| agents/ | Keep | Brain core |
| memory/ | Keep | Brain core |
| storage/ | Keep | Brain core |
| tools/ | Keep | Brain core |
| modes/ | Keep | Brain core |

---

## Next Steps

1. Complete comms extraction (phone processor deferred)
2. Update atlas_brain to proxy to atlas_comms for scheduling
3. Clean up atlas_brain/comms/ directory
4. Assess remaining modules for extraction
