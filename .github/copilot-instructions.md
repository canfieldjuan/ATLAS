# Copilot Instructions for ATLAS

## Project Overview

Atlas is an extensible AI "Brain" server and automation platform. It provides AI services (text, vision, speech-to-text), device control via a capability system, natural language intent dispatch, and a voice interface. The long-term vision is a distributed intelligent system spanning a central cloud brain, a local Jetson Nano hub, and edge nodes for cameras, sensors, and displays.

**Seven MCP servers** expose Atlas capabilities to MCP clients (Claude Desktop, Cursor):
CRM (port 8056), Email (8057), Twilio (8058), Calendar (8059), Invoicing (8060), Intelligence (8061), B2B Churn (8062).

## Repository Layout

```
atlas_brain/          # Core FastAPI AI brain (Python)
  api/                # Routing layer (health, query, devices, models)
  capabilities/       # Device/integration capability system
  mcp/                # MCP servers (crm, email, twilio, calendar, invoicing, intelligence, b2b_churn)
  services/           # AI model services (LLM registry, STT, CRM provider, email provider)
  schemas/            # Pydantic request/response models
atlas_comms/          # Communications module (Twilio, SMS, call flows)
atlas_edge/           # Edge computing / Jetson node support
atlas_video-processing/ # Video analysis pipeline
atlas-ui/             # React + TypeScript main UI (Vite)
atlas-admin-ui/       # Admin dashboard
atlas-churn-ui/       # Churn management UI
atlas-intel-ui/       # Intelligence UI
atlas-mobile/         # Mobile app
lib/graphrag/         # TypeScript GraphRAG library
graphiti-wrapper/     # Python FastAPI wrapper for GraphRAG
tests/                # Python pytest suite (60+ files)
scripts/              # Utility and debug scripts
```

## Build & Run

### Python backend (atlas_brain)

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run with hot reload (port 8001)
uvicorn atlas_brain.main:app --host 0.0.0.0 --port 8001 --reload \
  --ws-ping-interval 60 --ws-ping-timeout 120
```

### ASR server (required for voice pipeline)

```bash
pip install -r requirements.asr.txt
python asr_server.py --model nvidia/nemotron-speech-streaming-en-0.6b --port 8081 --device cuda:0
```

### LLM (Ollama)

```bash
ollama pull qwen3:14b          # local model (~10 GB VRAM)
ollama pull minimax-m2:cloud   # cloud relay for business workflows
```

### Frontend (React + TypeScript)

```bash
cd atlas-ui   # or atlas-admin-ui / atlas-churn-ui / atlas-intel-ui
npm install
npm run dev
```

### Docker (production)

```bash
docker compose up --build -d
docker compose logs -f brain
```

## Testing

```bash
# Python unit/integration tests
source .venv/bin/activate
pytest                        # run all tests
pytest tests/test_crm.py      # single file
pytest -m "not integration"   # skip integration markers

# TypeScript tests (lib/graphrag)
cd lib/graphrag
npm test
```

**Pytest configuration** is in `pytest.ini`. Custom markers: `integration`, `e2e`, `slow`. Async mode is auto-enabled.

## Key Patterns & Conventions

### Service Registry (LLM hot-swapping)

```python
from atlas_brain.services import llm_registry
llm_registry.activate("ollama")
llm_registry.deactivate()
```

### CRM Provider (direct asyncpg)

```python
from atlas_brain.services.crm_provider import get_crm_provider
crm = get_crm_provider()
contacts = await crm.search_contacts(phone="618-555-1234")
await crm.log_interaction(contact_id, "call", "note")
```

### Email Provider (Gmail preferred, Resend fallback)

```python
from atlas_brain.services.email_provider import get_email_provider
email = get_email_provider()
await email.send(to=["alice@example.com"], subject="Hi", body="...")
```

### Capability System (devices)

```python
from atlas_brain.capabilities import capability_registry, action_dispatcher, intent_parser
capability_registry.register(my_device)
intent = await intent_parser.parse("turn on the lights")
result = await action_dispatcher.dispatch_intent(intent)
```

### Adding a new device type

Create a class in `atlas_brain/capabilities/devices/` that implements the `Capability` protocol defined in `capabilities/protocols.py`. Register it on startup.

## Coding Standards

- **Python**: Follow existing patterns; use `async`/`await` throughout; Pydantic models for all API schemas; no bare `except` clauses.
- **TypeScript**: Strict mode; prefer `async`/`await`; no `any` types unless unavoidable; functional React components with hooks.
- **Tests**: Every new service or endpoint should have a corresponding test in `tests/`. Match the naming pattern `test_<module>.py`.
- **Configuration**: All tunables go through `atlas_brain/config.py` (Pydantic `Settings`); never hard-code credentials or URLs.
- **Secrets**: Use environment variables (`.env` file locally, Docker secrets in production). Never commit secrets.

## Environment Variables (key subset)

```bash
ATLAS_LLM_OLLAMA_MODEL=qwen3:14b
ATLAS_LOAD_LLM_ON_STARTUP=true
ATLAS_HA_ENABLED=false          # Home Assistant backend
ATLAS_MQTT_ENABLED=false        # MQTT backend
ATLAS_REMINDER_ENABLED=true
ATLAS_MCP_TRANSPORT=stdio       # stdio or sse
ATLAS_MCP_AUTH_TOKEN=           # Bearer token for SSE mode
```

See `CLAUDE.md` in the repository root for the full list of environment variables and additional context about the project.

## Known Gotchas

- The ASR server must be running before the voice pipeline endpoint is usable (`/ws/query/audio`).
- `uvicorn` WebSocket ping settings (`--ws-ping-interval 60 --ws-ping-timeout 120`) are required to prevent timeouts during voice streaming.
- NocoDB (port 8090) is a **read-only admin UI** over PostgreSQL; Atlas never depends on it at runtime.
- MCP servers are independent processes; run each one separately in stdio or SSE mode.
- The LLM router (`llm_router.py`) selects between the local Ollama model and the cloud relay based on task type.
