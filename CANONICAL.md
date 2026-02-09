# Canonical Implementations

**Last Updated**: 2025-01-22
**Status**: P0 AUDIT COMPLETE

This file answers: "Which one is the REAL implementation?"

---

## Voice Pipeline (P0) - AUDITED

| Component | Canonical | Location | Status |
|-----------|-----------|----------|--------|
| Wake Detector | `WakeWordDetector` | `services/wakeword/detector.py` | CANONICAL (OpenWakeWord) |
| STT | `NemotronSTT` | `services/stt/nemotron.py` | CANONICAL |
| TTS | `KokoroTTS` | `services/tts/kokoro.py` | CANONICAL |
| Agent | `AtlasAgent` | `agents/atlas.py` | CANONICAL |
| Local Loop | `run_voice_loop()` | `orchestration/__init__.py` | CANONICAL (mic on server) |
| Remote WS | `/api/v1/ws/orchestrated` | `api/orchestration.py` | CANONICAL (remote runner/UI) |

**Voice Entry Points**
- Local mic: `main.py:353-397` starts `run_voice_loop()` when `ATLAS_VOICE_ENABLED=true`
- Remote clients: WebSocket `/api/v1/ws/orchestrated` (Atlas UI and atlas_voice runner)

**Full Pipeline Flow**:
```
Mic/WS Audio -> STT -> AtlasAgent -> TTS -> Speaker (optional wake word upstream)
```
Wake word detection uses OpenWakeWord when enabled by the client or local loop.

---

## Tool System (P2)

| Component | Canonical | Location | Status |
|-----------|-----------|----------|--------|
| Tool Executor | `execute_with_tools()` | `services/tool_executor.py` | CANONICAL |
| Tools | `AtlasAgentTools` | `agents/tools.py` | CANONICAL |

**DEPRECATED - Do not modify:**
- `services/tool_router.py` - Gorilla-based, experimental
- `capabilities/intent_parser.py` - Old VLM-based parsing

---

## Home Assistant (P1)

| Component | Canonical | Location | Status |
|-----------|-----------|----------|--------|
| HA Backend | `HomeAssistantBackend` | `capabilities/backends/homeassistant.py` | CANONICAL |
| HA WebSocket | `HomeAssistantWSBackend` | `capabilities/backends/homeassistant_ws.py` | CANONICAL (real-time) |
| Device Registry | `capability_registry` | `capabilities/registry.py` | CANONICAL |

---

## LLM Services

| Purpose | Canonical | Location | Status |
|---------|-----------|----------|--------|
| Chat/Reasoning | Cloud LLM (Groq+Together) | `services/llm/cloud.py` | CANONICAL |
| Local Fallback | Ollama | `services/llm/__init__.py` | CANONICAL |
| Tool Calling | via `execute_with_tools()` | `services/tool_executor.py` | CANONICAL |

---

## API Endpoints

| Endpoint | Canonical | Wired To | Status |
|----------|-----------|----------|--------|
| POST /query/text | `api/query/text.py` | Atlas Agent → tool_executor | CANONICAL |
| WS /voice/stream | `api/comms/webhooks.py` | Phone calls only | NOT for local voice |
| WS /ws/orchestrated | `api/orchestration.py` | Atlas Agent + STT/TTS | CANONICAL |

**Local voice**: Handled by `run_voice_loop()` (not HTTP/WS API)

---

## Configuration

| Setting | Purpose | Default |
|---------|---------|---------|
| `ATLAS_VOICE_ENABLED` | Enable voice pipeline | true |
| `ATLAS_VOICE_INPUT_DEVICE_NAME` | Mic name substring | None |
| `ATLAS_VOICE_OUTPUT_DEVICE_NAME` | Speaker name substring | None |
| `ATLAS_VOICE_INPUT_SAMPLE_RATE` | Mic sample rate | 44100 |
| `ATLAS_ORCH_WAKEWORD_ENABLED` | Enable OpenWakeWord | true |
| `ATLAS_ORCH_WAKEWORD_THRESHOLD` | Detection threshold | 0.5 |
| `ATLAS_TTS_VOICE` | Kokoro voice | "am_michael" |
| `ATLAS_TTS_SPEED` | TTS speed | 1.15 |

---

## Decision Log

| Date | Component | Decision | Rationale |
|------|-----------|----------|-----------|
| 2026-01-23 | Wake Word | WakeWordGate (OpenWakeWord) | Audio-based detection runs BEFORE STT, saves compute |
| 2025-01-22 | STT | NemotronSTTService | Default in pipeline, streaming-optimized |
| 2025-01-22 | TTS | StreamingKokoroTTSService | Lower latency than non-streaming |
| 2025-01-22 | Agent | AtlasAgentProcessor | Routes through AtlasAgent for tools/devices |

---

## Rules

1. **One canonical per component** - If two exist, one must be deprecated
2. **Deprecated = don't touch** - Unless explicitly removing it
3. **New code must wire to canonical** - No new floating implementations
4. **Voice goes through agent/STT/TTS stack** - either local loop or `/ws/orchestrated`
