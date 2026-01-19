# Atlas Modes Architecture

## Overview

Atlas moves from a monolithic "one model does everything" approach to a modular **Modes** system where each mode has:
- Appropriately sized LLM model
- Mode-specific tools
- Standalone capability (can run outside Atlas)
- Optimized prompts for the use case

## Design Principles

1. **Resource Efficiency**: Small tasks use small models (3-7B), complex tasks use larger models (7-14B)
2. **Modularity**: Each mode is a standalone repo/package that works independently
3. **Hardware Accessible**: Target 8-16GB VRAM for most modes
4. **Explicit Switching**: Mode changes via explicit command ("Atlas transition to scheduling mode")

## Mode Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ATLAS CORE                               │
│                    (Orchestration Layer)                         │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │Mode Router  │  │   STT/TTS   │  │ User State  │              │
│  │ (keyword)   │  │  (shared)   │  │  (shared)   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                  │
│  Shared Tools: time, weather, location                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                    Mode Switching Command:
                 "Atlas transition to [mode] mode"
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  atlas-home   │   │atlas-schedule│   │ atlas-security│
│   (3-7B)      │   │   (7-14B)    │   │  (7-14B)      │
│   ~5GB VRAM   │   │   ~9GB VRAM  │   │  standalone   │
│               │   │               │   │               │
│ • lights      │   │ • calendar    │   │ • cameras     │
│ • TV/media    │   │ • reminders   │   │ • person ID   │
│ • thermostat  │   │ • appointments│   │ • alerts      │
│ • switches    │   │ • conflicts   │   │ • monitoring  │
│ • scenes      │   │ • availability│   │ • zones       │
└───────────────┘   └───────────────┘   └───────────────┘

┌───────────────┐   ┌───────────────┐
│atlas-business │   │  atlas-chat   │
│   (7-14B)     │   │  (cloud API)  │
│   ~9GB VRAM   │   │   0GB VRAM    │
│               │   │               │
│ • email       │   │ • conversation│
│ • proposals   │   │ • Q&A         │
│ • SMS/calls   │   │ • reasoning   │
│ • contacts    │   │ • creativity  │
│ • invoicing   │   │ • general     │
└───────────────┘   └───────────────┘
```

## Planned Modes

### 1. atlas-home (Priority: High)
- **Purpose**: Smart home device control
- **Model**: qwen3:8b (~5GB) or smaller
- **Tools**: lights, media, switches, thermostats, scenes
- **Complexity**: Low - simple command → action

### 2. atlas-schedule (Priority: HIGH - First Implementation)
- **Purpose**: Calendar, reminders, appointments, scheduling
- **Model**: qwen3:14b (~9GB)
- **Tools**: calendar CRUD, reminders, availability check, conflict detection
- **Complexity**: High - multi-step reasoning, time calculations

### 3. atlas-business (Priority: Medium)
- **Purpose**: Business communications and management
- **Model**: qwen3:14b (~9GB)
- **Tools**: email, SMS, proposals, contacts, Effingham Maids specific
- **Complexity**: High - context-aware communication

### 4. atlas-security (Priority: Medium)
- **Purpose**: Camera monitoring, alerts, person recognition
- **Model**: Specialized (VLM + small LLM)
- **Tools**: camera feeds, person detection, alert rules, zone monitoring
- **Complexity**: High - real-time processing, standalone operation

### 5. atlas-chat (Priority: Low)
- **Purpose**: General conversation, Q&A, reasoning
- **Model**: Cloud API (GPT-4, Claude)
- **Tools**: web search, general knowledge
- **Complexity**: Variable - defers to capable cloud models

## Mode Switching

### Explicit Commands
```
"Atlas transition to scheduling mode"
"Atlas switch to business mode"
"Atlas go to home mode"
```

### Keyword Hints (within mode)
Once in a mode, keywords keep you there:
- scheduling: "calendar", "remind", "appointment", "schedule", "meeting"
- home: "turn on", "turn off", "dim", "brightness", "play", "pause"
- business: "email", "send", "proposal", "call", "text"

### Default Mode
- Start in "home" mode (most common, fastest)
- Or configurable per user/time

## Standalone Package Structure

Each mode as independent repo:

```
atlas-schedule/
├── atlas_schedule/
│   ├── __init__.py
│   ├── agent.py              # Main ScheduleAgent class
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── calendar.py       # Google Calendar integration
│   │   ├── reminders.py      # Reminder CRUD
│   │   ├── availability.py   # Check free/busy
│   │   └── conflicts.py      # Detect scheduling conflicts
│   ├── prompts/
│   │   ├── system.py         # System prompt for scheduling
│   │   └── templates.py      # Response templates
│   ├── models.py             # Pydantic models
│   └── config.py             # Model preferences, API keys
├── tests/
│   └── ...
├── cli.py                    # Standalone CLI interface
├── api.py                    # Standalone FastAPI server
├── pyproject.toml
└── README.md
```

## Model Configuration

```python
# atlas_schedule/config.py
class ModelConfig:
    # Preferred models in order
    PRIMARY_MODEL = "qwen3:14b"        # 9GB, good reasoning
    FALLBACK_MODEL = "qwen3:8b"        # 5GB, faster
    CLOUD_FALLBACK = "gpt-4o-mini"     # Complex edge cases

    # Ollama settings
    OLLAMA_URL = "http://localhost:11434"
    KEEP_ALIVE = "5m"  # Keep model loaded for 5 minutes

    # Performance
    MAX_TOKENS = 256
    TEMPERATURE = 0.7
```

## Integration with Atlas

```python
# atlas_brain/modes/manager.py
class ModeManager:
    def __init__(self):
        self.current_mode = "home"
        self.modes = {
            "home": HomeMode(),
            "scheduling": ScheduleMode(),  # imports atlas-schedule
            "business": BusinessMode(),
            "security": SecurityMode(),
            "chat": ChatMode(),
        }

    async def switch_mode(self, mode_name: str):
        """Switch to a different mode, handles model swapping"""
        if mode_name not in self.modes:
            return False

        # Unload current mode's model
        await self.modes[self.current_mode].unload()

        # Load new mode's model
        self.current_mode = mode_name
        await self.modes[mode_name].load()

        return True

    async def process(self, query: str, user_id: str):
        """Process query with current mode"""
        return await self.modes[self.current_mode].process(query, user_id)
```

## Implementation Order

1. **Phase 1: atlas-schedule** (This PR)
   - Standalone repo with calendar, reminders, availability
   - Integration point in Atlas
   - Model swap on mode change

2. **Phase 2: atlas-home**
   - Extract existing device control
   - Optimize for small model

3. **Phase 3: atlas-business**
   - Email, SMS, proposals
   - Effingham Maids specific features

4. **Phase 4: atlas-security**
   - Camera integration
   - Person recognition
   - Alert system

5. **Phase 5: atlas-chat**
   - Cloud API integration
   - Fallback for unknown queries

## Resource Requirements

| Mode | Model | VRAM | Latency (warm) |
|------|-------|------|----------------|
| home | qwen3:8b | ~5GB | ~1-2s |
| scheduling | qwen3:14b | ~9GB | ~2-3s |
| business | qwen3:14b | ~9GB | ~2-3s |
| security | custom | ~4GB | ~1s |
| chat | cloud | 0GB | ~1-2s |

With STT (~3GB), a 16GB GPU can run home+STT comfortably.
A 24GB GPU can run any mode + STT + TTS.

## Open Questions

1. Should modes share conversation history?
2. How to handle queries that span modes? (e.g., "Schedule a reminder to turn off the lights")
3. Should there be a "supervisor" mode for complex multi-mode tasks?

---

*Created: 2026-01-18*
*Status: Planning*
*First Implementation: atlas-schedule*
