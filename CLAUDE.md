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
│  │     LLM     │  │     STT     │  │     TTS     │   AI Models  │
│  │  (Reasoning)│  │   (Speech)  │  │   (Voice)   │              │
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
- ✅ STT/TTS for voice interface
- ✅ PostgreSQL for conversation persistence
- ✅ Contacts CRM — `contacts` table + NocoDB browser UI (http://localhost:8090)
- ✅ 7 MCP servers: CRM (10 tools), Email (8), Twilio (10), Calendar (8), Invoicing (15), Intelligence (8), B2B Churn (10)

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
- **Intent Dispatch**: Natural language commands to device actions via LLM
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

### LLM Models (Ollama)

**Local LLM**: `qwen3:14b` (~10GB VRAM) -- conversation, reminders, calendar, intent classification.

**Cloud LLM**: `minimax-m2:cloud` (Ollama cloud relay) -- business workflows (booking, email, security escalation). Routed via `llm_router.py`.

```bash
# Pull local model
ollama pull qwen3:14b

# Pull cloud model
ollama pull minimax-m2:cloud

# Test
ollama run qwen3:14b "Hello"
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
│   ├── dependencies.py          # FastAPI Depends (inject services)
│   ├── health.py                # /ping, /health
│   ├── query/                   # AI inference endpoints
│   │   ├── text.py              # POST /query/text
│   │   ├── audio.py             # POST /query/audio, WS /ws/query/audio
│   │   └── vision.py            # POST /query/vision
│   └── models/                  # Model management
│       └── management.py        # GET/POST /models/stt, /models/llm
│   └── devices/                 # Device control
│       └── control.py           # /devices/*, /devices/intent
│
├── schemas/                     # Pydantic request/response models
│   └── query.py
│
├── services/                    # AI model services
│   ├── protocols.py             # LLMService protocol
│   ├── base.py                  # BaseModelService with shared utilities
│   ├── registry.py              # ServiceRegistry for hot-swapping (LLM)
│   ├── crm_provider.py          # CRM: DatabaseCRMProvider (direct asyncpg)
│   ├── email_provider.py        # Email: GmailEmailProvider + ResendEmailProvider
│   └── stt/
│       └── nemotron.py          # @register_stt("nemotron")
│
└── capabilities/                # Device/integration system
    ├── protocols.py             # Capability, CapabilityState, ActionResult
    ├── registry.py              # CapabilityRegistry
    ├── actions.py               # ActionDispatcher, Intent
    ├── intent_parser.py         # LLM → Intent extraction
    ├── backends/                # Communication backends
    │   ├── base.py              # Backend protocol
    │   ├── mqtt.py              # MQTTBackend
    │   └── homeassistant.py     # HomeAssistantBackend
    └── devices/                 # Device implementations
        ├── lights.py            # MQTTLight, HomeAssistantLight
        └── switches.py          # MQTTSwitch, HomeAssistantSwitch

atlas_brain/mcp/                 # MCP servers (Claude Desktop / Cursor compatible)
├── crm_server.py                # CRM MCP server          (10 tools, port 8056)
├── email_server.py              # Email MCP server         (8 tools, port 8057)
├── twilio_server.py             # Twilio MCP server        (10 tools, port 8058)
├── calendar_server.py           # Calendar MCP server      (8 tools, port 8059)
├── invoicing_server.py          # Invoicing MCP server     (15 tools, port 8060)
├── intelligence_server.py       # Intelligence MCP server  (17 tools, port 8061)
└── b2b_churn_server.py          # B2B Churn MCP server     (18 tools, port 8062)
```

## Key Patterns

**Service Registry**: LLM services are managed via a registry supporting runtime hot-swapping:
```python
from atlas_brain.services import llm_registry
llm_registry.activate("ollama")  # Load a registered LLM implementation
llm_registry.deactivate()         # Unload to free resources
```

**CRM Provider**: Single source of truth for all customer/contact data:
```python
from atlas_brain.services.crm_provider import get_crm_provider
crm = get_crm_provider()                       # DatabaseCRMProvider (direct asyncpg)
contacts = await crm.search_contacts(phone="618-555-1234")
await crm.log_interaction(contact_id, "call", "Booked cleaning for Monday")
```

**Email Provider**: Provider-agnostic send + read (Gmail preferred, Resend fallback):
```python
from atlas_brain.services.email_provider import get_email_provider
email = get_email_provider()
await email.send(to=["alice@example.com"], subject="Estimate", body="...")
messages = await email.list_messages("is:unread newer_than:1d")
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
# AI Models (LLM)
ATLAS_LLM_OLLAMA_MODEL=qwen3:14b
ATLAS_LLM_CLOUD_ENABLED=true
ATLAS_LLM_CLOUD_OLLAMA_MODEL=minimax-m2:cloud
ATLAS_LOAD_LLM_ON_STARTUP=true

# STT
ATLAS_STT_WHISPER_MODEL_SIZE=small.en
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

# NocoDB (browser UI over the contacts/appointments tables — no token needed for brain)
# Admin UI: http://localhost:8090  (auto-discovers all Postgres tables)

# MCP Servers (Claude Desktop / Cursor integration)
# Default transport is stdio. Set ATLAS_MCP_TRANSPORT=sse to expose as HTTP.
ATLAS_MCP_TRANSPORT=stdio            # stdio (Claude Desktop/Cursor) or sse (HTTP)
ATLAS_MCP_HOST=0.0.0.0              # Bind host for SSE mode
ATLAS_MCP_AUTH_TOKEN=                # Bearer token for SSE mode (optional)
ATLAS_MCP_CRM_ENABLED=true          # Enable/disable individual servers
ATLAS_MCP_EMAIL_ENABLED=true
ATLAS_MCP_TWILIO_ENABLED=true
ATLAS_MCP_CALENDAR_ENABLED=true
ATLAS_MCP_INVOICING_ENABLED=true
ATLAS_MCP_INTELLIGENCE_ENABLED=true
ATLAS_MCP_B2B_CHURN_ENABLED=true
ATLAS_MCP_CRM_PORT=8056
ATLAS_MCP_EMAIL_PORT=8057
ATLAS_MCP_TWILIO_PORT=8058
ATLAS_MCP_CALENDAR_PORT=8059
ATLAS_MCP_INVOICING_PORT=8060
ATLAS_MCP_INTELLIGENCE_PORT=8061
ATLAS_MCP_B2B_CHURN_PORT=8062

# IMAP — provider-agnostic email reading (works with Gmail, Outlook, any IMAP server)
# Leave blank to fall back to Gmail API reading
ATLAS_EMAIL_IMAP_HOST=imap.gmail.com
ATLAS_EMAIL_IMAP_PORT=993
ATLAS_EMAIL_IMAP_USERNAME=
ATLAS_EMAIL_IMAP_PASSWORD=         # For Gmail: 16-char app password (myaccount.google.com/apppasswords)
ATLAS_EMAIL_IMAP_SSL=true
ATLAS_EMAIL_IMAP_MAILBOX=INBOX
```

## NocoDB CRM Setup

NocoDB provides a browser UI over the existing Postgres tables (contacts,
contact_interactions, appointments).  Atlas itself uses DatabaseCRMProvider
(direct asyncpg) — NocoDB is purely a human-facing admin interface.

```bash
# 1. Start Postgres + NocoDB
docker compose up -d postgres nocodb

# 2. Open the NocoDB UI
open http://localhost:8090

# 3. On first launch, create an account, then connect to the existing DB:
#    Source: Postgres | Host: postgres | Port: 5432 | DB: atlas | User: atlas
```

The `contacts` table (created by migration `035_contacts.sql`) is the CRM schema.
`DatabaseCRMProvider` always queries it directly via asyncpg — no token or extra
service required.

## MCP Servers

Seven MCP servers expose Atlas capabilities to any MCP client (Claude Desktop, Cursor, custom agents).
All share `ATLAS_MCP_TRANSPORT` (stdio/sse), `ATLAS_MCP_HOST`, and `ATLAS_MCP_AUTH_TOKEN` config.
Each server has an independent enable/disable toggle (`ATLAS_MCP_<NAME>_ENABLED`).

### CRM MCP Server (10 tools)
```bash
# stdio mode (Claude Desktop / Cursor)
python -m atlas_brain.mcp.crm_server

# SSE HTTP mode (port 8056)
python -m atlas_brain.mcp.crm_server --sse
```

Tools: `search_contacts`, `get_contact`, `create_contact`, `update_contact`,
`delete_contact`, `list_contacts`, `log_interaction`, `get_interactions`,
`get_contact_appointments`, `get_customer_context`

### Email MCP Server (8 tools)
```bash
# stdio mode (Claude Desktop / Cursor)
python -m atlas_brain.mcp.email_server

# SSE HTTP mode (port 8057)
python -m atlas_brain.mcp.email_server --sse
```

Tools: `send_email`, `send_estimate`, `send_proposal`, `list_inbox`,
`get_message`, `search_inbox`, `get_thread`, `list_sent_history`

**Sending**: Gmail preferred (OAuth2); falls back to Resend if Gmail is not configured.
**Reading**: IMAP (provider-agnostic) when configured; Gmail API fallback.

IMAP works with any mail server — Gmail, Outlook, Yahoo, or custom:
```bash
ATLAS_EMAIL_IMAP_HOST=imap.gmail.com      # or outlook.office365.com, etc.
ATLAS_EMAIL_IMAP_PORT=993
ATLAS_EMAIL_IMAP_USERNAME=you@gmail.com
ATLAS_EMAIL_IMAP_PASSWORD=your_app_password   # Google: 16-char app password
ATLAS_EMAIL_IMAP_SSL=true
```

### Twilio MCP Server (10 tools)
```bash
# stdio mode (Claude Desktop / Cursor)
python -m atlas_brain.mcp.twilio_server

# SSE HTTP mode (port 8058)
python -m atlas_brain.mcp.twilio_server --sse
```

Tools: `make_call`, `get_call`, `list_calls`, `hangup_call`,
`start_recording`, `stop_recording`, `list_recordings`, `get_recording`,
`send_sms`, `lookup_phone`

**Outbound call recording**: Use `make_call(record=True)` to record from call creation.
Use `start_recording(call_sid)` to begin recording on an already-active call.

```bash
ATLAS_COMMS_TWILIO_ACCOUNT_SID=ACxxxxxxxx…
ATLAS_COMMS_TWILIO_AUTH_TOKEN=your_auth_token
ATLAS_COMMS_RECORD_CALLS=true          # enable recording globally
ATLAS_COMMS_WEBHOOK_BASE_URL=https://your-domain.com
```

### Calendar MCP Server (8 tools)
```bash
# stdio mode (Claude Desktop / Cursor)
python -m atlas_brain.mcp.calendar_server

# SSE HTTP mode (port 8059)
python -m atlas_brain.mcp.calendar_server --sse
```

Tools: `list_calendars`, `list_events`, `get_event`, `create_event`,
`update_event`, `delete_event`, `find_free_slots`, `sync_appointment`

**Provider-agnostic** — swap providers without touching the MCP layer:
- **Google Calendar** (default): set `ATLAS_TOOLS_CALENDAR_ENABLED=true` + run `scripts/setup_google_oauth.py`
- **CalDAV**: set `ATLAS_TOOLS_CALDAV_URL` + credentials (works with Nextcloud, Apple Calendar, Fastmail, Proton Calendar, SOGo, Baikal, Radicale)

**Does NocoDB have a calendar?** NocoDB can display date columns in a calendar view but
does not manage calendar events.
The `appointments` table in PostgreSQL is the schedule; `appointments.calendar_event_id`
links each booking to a calendar event.  Use `sync_appointment` to keep them in sync.

```bash
# Google Calendar (OAuth2)
ATLAS_TOOLS_CALENDAR_ENABLED=true
ATLAS_TOOLS_CALENDAR_CLIENT_ID=your_client_id
ATLAS_TOOLS_CALENDAR_CLIENT_SECRET=your_client_secret
ATLAS_TOOLS_CALENDAR_REFRESH_TOKEN=your_refresh_token  # written by setup_google_oauth.py

# CalDAV (alternative — overrides Google Calendar when set)
ATLAS_TOOLS_CALDAV_URL=https://nextcloud.example.com/remote.php/dav
ATLAS_TOOLS_CALDAV_USERNAME=your_username
ATLAS_TOOLS_CALDAV_PASSWORD=your_password
ATLAS_TOOLS_CALDAV_CALENDAR_URL=   # optional; auto-discovered via PROPFIND if blank

ATLAS_MCP_CALENDAR_PORT=8059  # Calendar MCP server (SSE mode only)
```

### Invoicing MCP Server (15 tools)
```bash
# stdio mode (Claude Desktop / Cursor)
python -m atlas_brain.mcp.invoicing_server

# SSE HTTP mode (port 8060)
python -m atlas_brain.mcp.invoicing_server --sse
```

Tools: `create_invoice`, `get_invoice`, `list_invoices`, `update_invoice`,
`send_invoice`, `record_payment`, `mark_void`, `customer_balance`,
`payment_history`, `create_service`, `list_services`, `get_service`,
`update_service`, `set_service_status`, `search_invoices`

### Intelligence MCP Server (17 tools)
```bash
# stdio mode (Claude Desktop / Cursor)
python -m atlas_brain.mcp.intelligence_server

# SSE HTTP mode (port 8061)
python -m atlas_brain.mcp.intelligence_server --sse
```

Tools (Strategic): `generate_intelligence_report`, `list_intelligence_reports`,
`get_intelligence_report`, `list_pressure_baselines`, `analyze_risk_sensors`,
`run_intervention_pipeline`, `list_pending_approvals`, `review_approval`

Tools (Consumer Product): `search_product_reviews`, `get_product_review`,
`list_pain_points`, `list_brands`, `get_brand_intelligence`,
`list_market_reports`, `get_market_report`, `get_consumer_pipeline_status`,
`list_complaint_content`

**Intelligence + Consumer product reviews**: Strategic entity intelligence
(pressure baselines, behavioral risk, interventions) plus consumer product
review data (Amazon reviews, brand health scores, pain points, competitive
flows, generated content). Two-pass enrichment pipeline.

### B2B Churn Intelligence MCP Server (18 tools)
```bash
# stdio mode (Claude Desktop / Cursor)
python -m atlas_brain.mcp.b2b_churn_server

# SSE HTTP mode (port 8062)
python -m atlas_brain.mcp.b2b_churn_server --sse
```

Tools: `list_churn_signals`, `get_churn_signal`, `list_high_intent_companies`,
`get_vendor_profile`, `list_reports`, `get_report`, `search_reviews`,
`get_review`, `get_pipeline_status`, `list_scrape_targets`,
`get_product_profile`, `match_products_tool`, `list_blog_posts`,
`get_blog_post`, `add_scrape_target`, `manage_scrape_target`,
`delete_scrape_target`, `list_affiliate_partners`

**B2B churn intelligence**: Query vendor churn signals, search enriched reviews,
read intelligence reports, identify high-intent companies, monitor pipeline health,
manage scrape targets, view blog posts, and list affiliate partners.
Data sourced from 16 review sites (incl. Twitter/X via Web Unlocker).

```bash
ATLAS_MCP_B2B_CHURN_ENABLED=true
ATLAS_MCP_B2B_CHURN_PORT=8062
```

### Claude Desktop config (`~/.claude/claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "atlas-crm": {
      "command": "python",
      "args": ["-m", "atlas_brain.mcp.crm_server"],
      "cwd": "/path/to/ATLAS"
    },
    "atlas-email": {
      "command": "python",
      "args": ["-m", "atlas_brain.mcp.email_server"],
      "cwd": "/path/to/ATLAS"
    },
    "atlas-twilio": {
      "command": "python",
      "args": ["-m", "atlas_brain.mcp.twilio_server"],
      "cwd": "/path/to/ATLAS"
    },
    "atlas-calendar": {
      "command": "python",
      "args": ["-m", "atlas_brain.mcp.calendar_server"],
      "cwd": "/path/to/ATLAS"
    },
    "atlas-invoicing": {
      "command": "python",
      "args": ["-m", "atlas_brain.mcp.invoicing_server"],
      "cwd": "/path/to/ATLAS"
    },
    "atlas-intelligence": {
      "command": "python",
      "args": ["-m", "atlas_brain.mcp.intelligence_server"],
      "cwd": "/path/to/ATLAS"
    },
    "atlas-b2b-churn": {
      "command": "python",
      "args": ["-m", "atlas_brain.mcp.b2b_churn_server"],
      "cwd": "/path/to/ATLAS"
    }
  }
}
```

## Environment Requirements

- NVIDIA GPU with 24GB+ VRAM (RTX 3090/4090) - single GPU setup
  - LLM (qwen3:14b): ~10GB VRAM
  - ASR (Nemotron 0.6B): ~2GB VRAM
- NVIDIA Container Toolkit installed on host (see `install_nvidia_toolkit.sh`)
- Docker and Docker Compose
- Ollama for LLM serving
