# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 🎯 PROJECT VISION (Read First!)

**Atlas is NOT just a home assistant.** It's an extensible AI "Brain" designed to grow from home automation into a comprehensive intelligent system.

### The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ATLAS BRAIN                               │
│              (Cloud/Server - Central Intelligence)               │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │     LLM     │  │     VLM     │  │     STT     │   AI Models  │
│  │  (Reasoning)│  │   (Vision)  │  │   (Speech)  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                  │
│  ┌─────────────────────────────────────────────────┐            │
│  │              Unified Voice Interface             │            │
│  │    "Hey Atlas" → STT → Router → Action/LLM → TTS │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                  │
│  ┌─────────────────────────────────────────────────┐            │
│  │           PostgreSQL (Persistence)              │            │
│  │   Sessions | Conversations | Users | State      │            │
│  └─────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CENTRAL HUB                                 │
│                    (Jetson Nano)                                 │
│         Local processing, device coordination                    │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │  Node 1  │   │  Node 2  │   │  Node N  │
        │ (Jetson) │   │ (Jetson) │   │ (Jetson) │
        │ Camera   │   │ Sensors  │   │ Display  │
        │ Mic/Spk  │   │ Motion   │   │ Control  │
        └──────────┘   └──────────┘   └──────────┘
```

### Current Capabilities (Implemented)
- ✅ Voice-activated device control ("Hey Atlas, turn off the TV")
- ✅ Natural language intent parsing
- ✅ Home Assistant integration (WebSocket real-time state, media players)
- ✅ LLM for conversations and reasoning
- ✅ VLM for vision queries
- ✅ STT/TTS for voice interface
- ✅ PostgreSQL for conversation persistence

### Future Capabilities (Planned)
- 🔲 Unified always-on voice interface (wake word "Hey Atlas")
- 🔲 Smart routing: device commands vs conversation vs queries
- 🔲 Human tracking and recognition
- 🔲 Object detection and tracking
- 🔲 Distributed node architecture (Jetson Nanos)
- 🔲 Context-aware conversations ("dim them" → knows "them" = last mentioned lights)
- 🔲 Calendar, reminders, proactive notifications
- 🔲 Multi-room audio/video coordination

### Design Principles
1. **Extensibility First**: Every component should be pluggable and replaceable
2. **Seamless Experience**: One interface for everything (chat, control, queries)
3. **Local Processing**: Prefer edge compute, cloud for heavy lifting only
4. **Persistence**: Remember conversations, learn preferences, track state
5. **Privacy**: User data stays local, no external telemetry

### Current Session Focus
When working on Atlas, always ask: "Does this fit the big picture?"
- Don't over-engineer for today, but don't block tomorrow
- Keep interfaces clean for future node distribution
- Maintain conversation context across interactions

---

## Project Overview

Atlas is a centralized AI "Brain" server and extensible automation platform. It provides:
- **AI Services**: Text, vision, and speech-to-text inference via REST API
- **Device Control**: Extensible capability system for IoT devices, home automation
- **Intent Dispatch**: Natural language commands to device actions via VLM
- **Voice Interface**: Wake word activated, seamless chat + control

## Build and Run Commands

### Local Development (Recommended for fast iteration)

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run with hot reload on port 8001
# Note: WebSocket ping settings prevent timeout during voice streaming
uvicorn atlas_brain.main:app --host 0.0.0.0 --port 8001 --reload --ws-ping-interval 60 --ws-ping-timeout 120
```

### ASR Server (Required for Voice Pipeline)

The ASR server provides speech-to-text for the voice pipeline. It runs separately from the main Atlas server.

```bash
# Install ASR dependencies (first time only)
pip install -r requirements.asr.txt

# Start ASR server on GPU 0, port 8081
python asr_server.py --model nvidia/nemotron-speech-streaming-en-0.6b --port 8081 --device cuda:0
```

**Endpoints:**
- `GET /health` - Server status
- `POST /v1/asr` - Batch transcription (WAV file)
- `WS /v1/asr/stream` - Streaming transcription (PCM chunks)

**Note:** The voice pipeline expects ASR at `http://127.0.0.1:8081`. Configure via `ATLAS_VOICE_ASR_URL` in `.env`.

### LLM Model (Ollama)

The main LLM runs via Ollama. Using `qwen3-30b-a3b-small` (17GB Q4_K_S quantization) to fit alongside ASR on a single 24GB GPU.

```bash
# Pull/create the model (if not already done)
ollama create qwen3-30b-a3b-small -f Modelfile.qwen3-30b-a3b-small

# Test the model
ollama run qwen3-30b-a3b-small "Hello"
```

### Docker (Production)

```bash
# Build and start the server (requires NVIDIA Container Toolkit)
docker compose up --build -d

# Restart after code changes (volumes mount atlas_brain/, so rebuild not always needed)
docker compose restart

# View logs
docker compose logs -f brain

# Stop the server
docker compose down
```

## Testing Endpoints

```bash
# Health check
curl http://127.0.0.1:8000/api/v1/ping

# Detailed health with service status
curl http://127.0.0.1:8000/api/v1/health

# Text query
curl -X POST http://127.0.0.1:8000/api/v1/query/text \
  -H "Content-Type: application/json" \
  -d '{"query_text": "What is 2+2?"}'

# Vision query (image + optional prompt)
curl -X POST http://127.0.0.1:8000/api/v1/query/vision \
  -F "image_file=@image.jpg" \
  -F "prompt_text=What is in this image?"

# Audio transcription
curl -X POST http://127.0.0.1:8000/api/v1/query/audio \
  -F "audio_file=@audio.wav"

# List available VLM models
curl http://127.0.0.1:8000/api/v1/models/vlm

# Hot-swap VLM model
curl -X POST http://127.0.0.1:8000/api/v1/models/vlm/activate \
  -H "Content-Type: application/json" \
  -d '{"name": "moondream"}'

# List registered devices
curl http://127.0.0.1:8000/api/v1/devices/

# Execute device action
curl -X POST http://127.0.0.1:8000/api/v1/devices/{device_id}/action \
  -H "Content-Type: application/json" \
  -d '{"action": "turn_on"}'

# Natural language device control
curl -X POST http://127.0.0.1:8000/api/v1/devices/intent \
  -H "Content-Type: application/json" \
  -d '{"query": "turn on the living room lights"}'
```

## Architecture

```
atlas_brain/
├── main.py                      # FastAPI app with lifespan management
├── config.py                    # Pydantic Settings for configuration
│
├── api/                         # API layer (routing only)
│   ├── dependencies.py          # FastAPI Depends (get_vlm, get_stt)
│   ├── health.py                # /ping, /health
│   ├── query/                   # AI inference endpoints
│   │   ├── text.py              # POST /query/text
│   │   ├── audio.py             # POST /query/audio, WS /ws/query/audio
│   │   └── vision.py            # POST /query/vision
│   ├── models/                  # Model management
│   │   └── management.py        # GET/POST /models/vlm, /models/stt
│   └── devices/                 # Device control
│       └── control.py           # /devices/*, /devices/intent
│
├── schemas/                     # Pydantic request/response models
│   └── query.py
│
├── services/                    # AI model services
│   ├── protocols.py             # VLMService, STTService protocols
│   ├── base.py                  # BaseModelService with shared utilities
│   ├── registry.py              # ServiceRegistry for hot-swapping
│   ├── vlm/
│   │   └── moondream.py         # @register_vlm("moondream")
│   └── stt/
│       └── nemotron.py          # @register_stt("nemotron")
│
└── capabilities/                # Device/integration system
    ├── protocols.py             # Capability, CapabilityState, ActionResult
    ├── registry.py              # CapabilityRegistry
    ├── actions.py               # ActionDispatcher, Intent
    ├── intent_parser.py         # VLM → Intent extraction
    ├── backends/                # Communication backends
    │   ├── base.py              # Backend protocol
    │   ├── mqtt.py              # MQTTBackend
    │   └── homeassistant.py     # HomeAssistantBackend
    └── devices/                 # Device implementations
        ├── lights.py            # MQTTLight, HomeAssistantLight
        └── switches.py          # MQTTSwitch, HomeAssistantSwitch
```

## Key Patterns

**Service Registry**: AI models are managed via registries that support runtime hot-swapping:
```python
from atlas_brain.services import vlm_registry
vlm_registry.activate("moondream")  # Load model
vlm_registry.deactivate()            # Unload to free VRAM
```

**Capability System**: Devices implement the Capability protocol and are registered:
```python
from atlas_brain.capabilities import capability_registry
capability_registry.register(my_light)
```

**Intent Dispatch**: Natural language → structured intent → device action:
```python
from atlas_brain.capabilities import action_dispatcher, intent_parser
intent = await intent_parser.parse("turn on the lights")
result = await action_dispatcher.dispatch_intent(intent)
```

## Adding New Models

Create a new file in `services/vlm/` or `services/stt/`:
```python
from ..registry import register_vlm
from ..base import BaseModelService

@register_vlm("my-model")
class MyVLM(BaseModelService):
    def load(self): ...
    def unload(self): ...
    def process_text(self, query): ...
    async def process_vision(self, image_bytes, prompt): ...
```

## Adding New Device Types

Create in `capabilities/devices/`:
```python
from ..protocols import Capability, CapabilityType, ActionResult

class ThermostatCapability:
    capability_type = CapabilityType.THERMOSTAT
    supported_actions = ["set_temperature", "read"]

    async def execute_action(self, action, params): ...
```

## Environment Variables

```bash
# AI Models
ATLAS_VLM_DEFAULT_MODEL=moondream
ATLAS_STT_WHISPER_MODEL_SIZE=small.en
ATLAS_LOAD_VLM_ON_STARTUP=true
ATLAS_LOAD_STT_ON_STARTUP=false

# MQTT Backend (optional)
ATLAS_MQTT_ENABLED=false
ATLAS_MQTT_HOST=localhost
ATLAS_MQTT_PORT=1883

# Home Assistant Backend (optional)
ATLAS_HA_ENABLED=false
ATLAS_HA_URL=http://homeassistant.local:8123
ATLAS_HA_TOKEN=your_token

# Reminder System
ATLAS_REMINDER_ENABLED=true
ATLAS_REMINDER_DEFAULT_TIMEZONE=America/Chicago
ATLAS_REMINDER_MAX_REMINDERS_PER_USER=100

# Calendar Tool (Google Calendar)
ATLAS_TOOLS_CALENDAR_ENABLED=true
ATLAS_TOOLS_CALENDAR_CLIENT_ID=your_client_id
ATLAS_TOOLS_CALENDAR_CLIENT_SECRET=your_client_secret
ATLAS_TOOLS_CALENDAR_REFRESH_TOKEN=your_refresh_token
```

## Environment Requirements

- NVIDIA GPU with 24GB+ VRAM (RTX 3090/4090) - single GPU setup
  - LLM (qwen3-30b-a3b-small): ~18GB VRAM
  - ASR (Nemotron 0.6B): ~2GB VRAM
- NVIDIA Container Toolkit installed on host (see `install_nvidia_toolkit.sh`)
- Docker and Docker Compose
- Ollama for LLM serving
