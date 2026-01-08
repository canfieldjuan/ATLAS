# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Atlas is a centralized AI "Brain" server and extensible automation platform. It provides:
- **AI Services**: Text, vision, and speech-to-text inference via REST API
- **Device Control**: Extensible capability system for IoT devices, home automation
- **Intent Dispatch**: Natural language commands to device actions via VLM

## Build and Run Commands

### Local Development (Recommended for fast iteration)

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run with hot reload on port 8001
uvicorn atlas_brain.main:app --host 0.0.0.0 --port 8001 --reload
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
│       └── faster_whisper.py    # @register_stt("faster-whisper")
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
```

## Environment Requirements

- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed on host (see `install_nvidia_toolkit.sh`)
- Docker and Docker Compose
