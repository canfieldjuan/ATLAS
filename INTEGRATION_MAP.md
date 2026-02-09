# Atlas Integration Map

**Last Updated**: 2025-01-22
**Status**: P0 AUDITED

---

## Legend

```
───▶  Connected and working
- - ▶  Exists but NOT wired
╳      Missing/not implemented
```

---

## P0 Voice Pipeline - COMPLETE

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              PIPECAT VOICE PIPELINE                                      │
│                                                                                          │
│  ┌─────────┐   ┌───────────┐   ┌─────────┐   ┌───────────┐   ┌────────────┐             │
│  │   Mic   │──▶│ Resampler │──▶│   VAD   │──▶│    STT    │──▶│ WakeFilter │             │
│  │         │   │ 44k→16k   │   │ Silero  │   │ Nemotron  │   │ "hey atlas"│             │
│  └─────────┘   └───────────┘   └─────────┘   └───────────┘   └─────┬──────┘             │
│       ▲                                                            │                     │
│       │                                                            ▼                     │
│       │        ┌───────────┐   ┌───────────────────────────────────────┐                │
│       │        │  Speaker  │◀──│           AtlasAgentProcessor          │                │
│       │        └───────────┘   │                                       │                │
│       │              ▲         │  TranscriptionFrame                   │                │
│       │              │         │        │                              │                │
│       │         ┌────┴────┐    │        ▼                              │                │
│       │         │   TTS   │    │  ┌─────────────┐                      │                │
│       │         │ Kokoro  │◀───│  │ AtlasAgent  │                      │                │
│       │         │Streaming│    │  │             │                      │                │
│       │         └─────────┘    │  │ - Tools     │                      │                │
│       │                        │  │ - Devices   │                      │                │
│       │                        │  │ - Memory    │                      │                │
│       └────────────────────────│  └─────────────┘                      │                │
│                                └───────────────────────────────────────┘                │
│                                                                                          │
│  Entry: main.py:353 → run_voice_pipeline()                                              │
│  Config: ATLAS_VOICE_ENABLED=true                                                       │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

**Status**: WIRED ✓

---

## Text API Path - WORKING

```
HTTP POST /api/v1/query/text
         │
         ▼
┌──────────────────┐
│   text.py        │───▶ AtlasAgent ───▶ Tools ───▶ Response
│   endpoint       │
└──────────────────┘
```

**Status**: WIRED ✓

---

## Phone/SMS Path - BROKEN (Not P0)

```
SignalWire ───▶ webhooks.py ───▶ PersonaPlex - - - ▶ [LATENCY ISSUES]
                    │
                    └───▶ Legacy phone_processor - - - ▶ [DEPRECATED]
```

**Status**: TABLED (not P0 priority)

---

## Deprecated Components (DO NOT USE)

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEPRECATED / NOT WIRED                        │
│                                                                  │
│   services/wakeword/detector.py    ← OpenWakeWord (not in pipe) │
│   services/stt/nemotron.py         ← Use pipecat/stt.py         │
│   services/tts/piper.py            ← Use pipecat/tts.py         │
│   services/tts/kokoro.py           ← Use pipecat/tts.py         │
│   services/tool_router.py          ← Gorilla experimental       │
│   pipecat/router.py                ← FunctionGemma legacy       │
│   capabilities/intent_parser.py    ← Old VLM parsing            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## P1 - Home Assistant

```
┌──────────────────────────────────────────────────────────────────┐
│                     HOME ASSISTANT (P1)                           │
│                                                                   │
│   AtlasAgent                                                      │
│       │                                                           │
│       ▼                                                           │
│   AtlasAgentTools ───▶ HA Backend ───▶ Home Assistant API        │
│                             │                                     │
│                             ▼                                     │
│                    capability_registry ───▶ Devices               │
│                                                                   │
│   Status: Components exist, need to verify E2E via voice         │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## P2 - Native Tools

```
┌──────────────────────────────────────────────────────────────────┐
│                      NATIVE TOOLS (P2)                            │
│                                                                   │
│   AtlasAgent ───▶ AtlasAgentTools ───▶ tool_executor             │
│                         │                                         │
│                         ├── get_time ✓                           │
│                         ├── get_weather (needs API key?)         │
│                         ├── calendar (Google Calendar)           │
│                         ├── reminders                            │
│                         └── ...                                   │
│                                                                   │
│   Status: Tools exist in agents/tools.py, need E2E test         │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## File Locations (Canonical Only)

| Component | File | Purpose |
|-----------|------|---------|
| STT | `services/stt/nemotron.py` | Nemotron STT |
| TTS | `services/tts/kokoro.py` | Kokoro TTS |
| WS Orchestrator | `api/orchestration.py` | `/api/v1/ws/orchestrated` |
| Atlas Agent | `agents/atlas.py` | Main brain |
| Tools | `agents/tools.py` | AtlasAgentTools |
| HA Backend | `capabilities/backends/homeassistant.py` | HA connection |
| Config | `config.py` | VoiceClientConfig, etc. |
| Entry | `main.py` | Starts local voice loop |

---

## Next Steps

1. [x] Audit P0 components - DONE
2. [ ] Test P0 end-to-end (say "hey atlas", get response)
3. [ ] If broken, fix canonical pipeline (not create new one)
4. [ ] Verify P1 HA commands work via voice
5. [ ] Verify P2 tools work via voice
