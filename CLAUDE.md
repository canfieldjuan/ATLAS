# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## What Atlas Is (Today)

Atlas is a **multi-product intelligence platform** built on a single FastAPI
backend (`atlas_brain/`), a 9-server MCP surface, six standalone Python
packages (`extracted_*/`), and six React+Vite frontends. The headline
products in priority order:

| Product | Surface | Status |
|---|---|---|
| **B2B Churn Intelligence** | `atlas-churn-ui`, `atlas-intel-ui`, MCP `b2b_churn_server` (83 tools), 19 review sources, displacement graph, weekly reports, calibration loop, webhooks | **Shipped** — Intelligence Platform Roadmap phases 0–7 complete |
| **Consumer Intelligence** | MCP `intelligence_server` (33 tools), brand registry, displacement edges, market reports, PDF export | **Shipped** — Consumer Roadmap phases 0–6+ complete |
| **Content Ops Pipeline** | `extracted_content_pipeline/` (~77 KLOC), blog post + B2B campaign + landing page + report + sales-brief generation, signal extraction, generated-asset review API | **Active iteration** — 39 of 65 open plans; signal extraction + scope wiring + landing-page wired; 4 more generators pending LLM/repo factories |
| **Communications + CRM + Calendar + Invoicing** | MCP servers (CRM 10, Email 9, Twilio 10, Calendar 8, Invoicing 18), Postgres `contacts`/`appointments`, Gmail/IMAP/Resend, Google Calendar/CalDAV, NocoDB admin UI | **Shipped** |
| **Knowledge graph memory** | MCP `memory_server` (15 tools), Postgres short-term + Neo4j/Graphiti long-term via `graphiti-wrapper` (port 8001) | **Shipped** |
| **Universal scraper** | MCP `scraper_server` (5 tools), LLM-driven schema extraction, Playwright JS rendering | **Shipped** |
| **Voice + Home Automation** | `atlas_brain/voice/`, ASR (Nemotron 0.6B, port 8081), wake-word + VAD + capture, Home Assistant + MQTT capability registry, intent dispatch | **Built but not yet routed through agent** — per `CONTEXT.md`, Pipecat pipeline + agent routing are not unified yet (P0 in `BUILD_SPEC.md`) |
| **Multi-tenant SaaS auth** | `atlas_brain/auth/` — JWT, password hashing, plan tiers | **Shipped** |
| **Autonomous task scheduler** | `atlas_brain/autonomous/` — APScheduler-driven, 150+ task modules (B2B churn, blog gen, email digest, campaign send, invoice reminders, weekly briefings, …) | **Shipped** |

The original home-automation framing (wake word "Hey Atlas" → STT → router →
device action → TTS) is still on the roadmap and the components are in the
tree (`atlas_brain/voice/`, `atlas_brain/capabilities/`,
`atlas_brain/discovery/`), but the unified voice-to-agent pipeline is the
**P0** in `BUILD_SPEC.md` — not a shipped capability today.

### Design Principles
1. **Extensibility First** — every component pluggable behind a typed port
2. **Provider-agnostic** — email, calendar, LLM, CRM all swap behind a single port
3. **Single source of truth** — config via `atlas_brain/config.py`; CRM via `crm_provider.get_crm_provider()`; never sneak around the port
4. **Plan first** — non-trivial PRs ship a plan doc at `plans/PR-<Slice>.md` per `AGENTS.md`
5. **Local processing** — prefer edge compute; cloud LLMs only for heavy lifting
6. **Privacy** — user data stays local, no external telemetry

---

## Build and Run Commands

### Local Development (Recommended for fast iteration)

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run with hot reload on port 8001
# Scope reload to atlas_brain so watchfiles does not scan data/postgres
# Note: WebSocket ping settings prevent timeout during voice streaming
uvicorn atlas_brain.main:app --host 0.0.0.0 --port 8001 --reload --reload-dir atlas_brain --reload-exclude data/postgres --reload-exclude data/postgres/** --ws-ping-interval 60 --ws-ping-timeout 120
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

## Repository Layout

The repo is a monorepo: the Python `atlas_brain` server, several extracted standalone Python packages (`extracted_*`), the standalone `atlas_comms` / `atlas_edge` / `atlas_video-processing` services, six React+Vite frontends (`atlas-*-ui`, `atlas-mobile`, `portfolio-ui`), and a Neo4j/Graphiti knowledge-graph wrapper.

```
ATLAS/
├── atlas_brain/                 # Main FastAPI server (the "Brain")
├── atlas_comms/                 # Standalone communications service (Twilio/SignalWire)
├── atlas_edge/                  # Edge-node capabilities (Jetson)
├── atlas_video-processing/      # Video pipeline (SAM 3, etc.)
├── graphiti-wrapper/            # Neo4j + Graphiti GraphRAG service (port 8001)
├── asr_server.py                # Standalone ASR (Nemotron) FastAPI server, port 8081
├── webhook_dispatcher.py        # Outbound webhook delivery for B2B intelligence
│
├── extracted_competitive_intelligence/  # Standalone competitive-intel package
├── extracted_content_pipeline/          # Blog post + B2B campaign generation (active iteration)
├── extracted_evidence_to_story/         # Evidence-to-narrative pipeline
├── extracted_llm_infrastructure/        # LLM provider ports + adapters
├── extracted_quality_gate/              # Quality / validation gates
├── extracted_reasoning_core/            # Cross-domain reasoning event bus
│
├── atlas-admin-ui/   atlas-churn-ui/   atlas-intel-ui/   atlas-ui/
├── atlas-mobile/     portfolio-ui/     animated-robot-logo/
│
├── tests/                       # 600+ tests, pytest-based
├── scripts/                     # 159 audit/run/validate/backfill scripts
├── docs/                        # Architecture, audits, runbooks, roadmaps
├── plans/                       # 65+ per-PR plan docs (AGENTS.md workflow)
│
├── AGENTS.md                    # Multi-session builder/reviewer contract (READ FIRST)
├── AUDITOR_PROMPT.md            # Cross-cutting audit prompt
├── CANONICAL.md                 # Which implementation is the real one
├── INTEGRATION_MAP.md           # What's wired to what
├── BUILD_SPEC.md                # P0/P1/P2 priorities, definition of done
└── CLAUDE.md                    # (this file)
```

### `atlas_brain/` package map (35 packages)

```
atlas_brain/
├── main.py                      # FastAPI app + lifespan
├── config.py                    # Pydantic Settings (env_prefix=ATLAS_*)
├── _content_ops_*.py            # Content-Ops scope/services/infra wiring
│
├── api/                         # 50+ FastAPI routers (health, query, devices,
│                                #   b2b_*, content-ops, blog, billing, identity, …)
├── agents/                      # LangGraph agent orchestration (memory, tools, entity tracker)
├── alerts/                      # Centralized alert system (vision, audio, HA, security)
├── auth/                        # SaaS auth: JWT, password hashing, plan tiers
├── autonomous/                  # Scheduled & alert-driven headless tasks (incl. blog gen)
├── capabilities/                # Device protocols, registry, action dispatch
│   ├── backends/                # mqtt, homeassistant
│   └── devices/                 # lights, switches, …
├── comms/                       # Re-exports atlas_comms (phone STT/LLM/TTS local)
├── discovery/                   # SSDP / mDNS device discovery
├── escalation/                  # Security event classification + LLM synthesis
├── events/                      # System event broadcast (real-time UI feed)
├── jobs/                        # Background jobs (e.g. NightlyMemorySync)
├── mcp/                         # 10 MCP servers (see "MCP Servers" below)
│   └── b2b/                     # B2B-churn server module split
├── memory/                      # MemoryService, RAG client, token budgeting
├── modes/                       # Operating modes (tool groupings, model prefs)
├── orchestration/               # Runtime context (faces, speakers, objects), CUDA lock
├── pipelines/                   # Pipeline registry (news, complaints, SaaS reviews, …)
├── presence/                    # Proxies to atlas_vision occupancy
├── reasoning/                   # Event-driven cross-domain reasoning
├── schemas/                     # Pydantic request/response models
├── security/                    # WiFi threat detection, network IDS
├── services/                    # 55 modules: providers, embeddings, registries, audits
│   ├── b2b/                     # B2B-specific services
│   ├── llm/                     # LLM router + adapters
│   ├── embedding/               # Embedding services
│   ├── scraping/                # Universal scrape engine
│   └── speaker_id/              # Speaker recognition
├── skills/                      # Markdown-prompt skills (b2b, call, digest, email,
│                                #   intelligence, invoicing, security, sms)
├── storage/                     # Postgres persistence (sessions, conversations, terminals)
├── templates/                   # Email + message templates
├── tools/                       # Info-query tools (weather, traffic, calendar, email, …)
├── utils/                       # Time/format helpers
├── vision/                      # MQTT subscription to atlas_vision detections
└── voice/                       # Local voice-to-voice (wake word, VAD, capture, playback)
```

### `atlas_brain/mcp/` MCP servers (10)

| Server                  | Port | Tools | Module                                |
|-------------------------|------|-------|---------------------------------------|
| CRM                     | 8056 | 10    | `atlas_brain.mcp.crm_server`          |
| Email                   | 8057 | 9     | `atlas_brain.mcp.email_server`        |
| Twilio                  | 8058 | 10    | `atlas_brain.mcp.twilio_server`       |
| Calendar                | 8059 | 8     | `atlas_brain.mcp.calendar_server`     |
| Invoicing               | 8060 | 18    | `atlas_brain.mcp.invoicing_server`    |
| Invoicing Readonly      | 8065 | 8     | `atlas_brain.mcp.invoicing_readonly_server` |
| Intelligence            | 8061 | 33    | `atlas_brain.mcp.intelligence_server` |
| B2B Churn Intelligence  | 8062 | 83    | `atlas_brain.mcp.b2b_churn_server` (split across 17 modules in `mcp/b2b/`) |
| Universal Scraper       | 8063 | 5     | `atlas_brain.mcp.scraper_server`      |
| Memory (Graphiti+Postgres) | 8064 | 15 | `atlas_brain.mcp.memory_server`       |

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
ATLAS_MCP_AUTH_TOKEN=                # Bearer token for SSE mode; required for invoicing-readonly HTTP
ATLAS_MCP_INVOICING_READONLY_AUTH_MODE=bearer  # bearer (direct clients) or oauth (ChatGPT)
ATLAS_MCP_INVOICING_READONLY_OAUTH_ISSUER_URL=
ATLAS_MCP_INVOICING_READONLY_OAUTH_RESOURCE_URL=
ATLAS_MCP_INVOICING_READONLY_OAUTH_APPROVAL_TOKEN=
ATLAS_MCP_CRM_ENABLED=true          # Enable/disable individual servers
ATLAS_MCP_EMAIL_ENABLED=true
ATLAS_MCP_TWILIO_ENABLED=true
ATLAS_MCP_CALENDAR_ENABLED=true
ATLAS_MCP_INVOICING_ENABLED=true
ATLAS_MCP_INVOICING_READONLY_ENABLED=true
ATLAS_MCP_INTELLIGENCE_ENABLED=true
ATLAS_MCP_B2B_CHURN_ENABLED=true
ATLAS_MCP_CRM_PORT=8056
ATLAS_MCP_EMAIL_PORT=8057
ATLAS_MCP_TWILIO_PORT=8058
ATLAS_MCP_CALENDAR_PORT=8059
ATLAS_MCP_INVOICING_PORT=8060
ATLAS_MCP_INVOICING_READONLY_PORT=8065
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

Ten MCP servers expose Atlas capabilities to any MCP client (Claude Desktop, Cursor, custom agents).
All share `ATLAS_MCP_TRANSPORT` (stdio/sse) and `ATLAS_MCP_HOST`; HTTP deployments should set `ATLAS_MCP_AUTH_TOKEN`, and the read-only invoicing HTTP server refuses to start without a non-placeholder token of at least 24 characters because it exposes customer financial data.
Each server has an independent enable/disable toggle (`ATLAS_MCP_<NAME>_ENABLED`).

For ChatGPT online connector rollout, use
`docs/MCP_CHATGPT_OAUTH_ROLLOUT_RUNBOOK.md`. It captures the proven
read-only invoicing OAuth pattern, Tailscale well-known route shape, discovery
and e2e smoke requirements, and operator launcher checklist for future MCP
servers.

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

### Email MCP Server (9 tools)
```bash
# stdio mode (Claude Desktop / Cursor)
python -m atlas_brain.mcp.email_server

# SSE HTTP mode (port 8057)
python -m atlas_brain.mcp.email_server --sse
```

Tools: `send_email`, `send_estimate`, `send_proposal`, `list_inbox`,
`get_message`, `search_inbox`, `get_thread`, `list_sent_history`, `list_folders`

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
The appointments table in PostgreSQL is the schedule; `appointments.calendar_event_id`
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

### Invoicing MCP Server (18 tools)
```bash
# stdio mode (Claude Desktop / Cursor)
python -m atlas_brain.mcp.invoicing_server

# SSE HTTP mode (port 8060)
python -m atlas_brain.mcp.invoicing_server --sse
```

Tools: `create_invoice`, `get_invoice`, `list_invoices`, `update_invoice`,
`send_invoice`, `record_payment`, `mark_void`, `customer_balance`,
`payment_history`, `create_service`, `list_services`, `get_service`,
`update_service`, `set_service_status`, `search_invoices`,
`list_pending_drafts`, `approve_and_send`, `export_invoice_pdf`

### Invoicing Readonly MCP Server (8 tools)
```bash
# stdio mode (read-only tools only)
python -m atlas_brain.mcp.invoicing_readonly_server

# SSE HTTP mode (port 8065, requires ATLAS_MCP_AUTH_TOKEN)
ATLAS_MCP_AUTH_TOKEN=<token> python -m atlas_brain.mcp.invoicing_readonly_server --sse

# ChatGPT-compatible OAuth mode
ATLAS_MCP_INVOICING_READONLY_AUTH_MODE=oauth \
ATLAS_MCP_INVOICING_READONLY_OAUTH_ISSUER_URL=https://atlas-brain.tailc7bd29.ts.net/invoicing-readonly \
ATLAS_MCP_INVOICING_READONLY_OAUTH_RESOURCE_URL=https://atlas-brain.tailc7bd29.ts.net/invoicing-readonly/mcp \
ATLAS_MCP_INVOICING_READONLY_OAUTH_APPROVAL_TOKEN=<long-operator-token> \
python -m atlas_brain.mcp.invoicing_readonly_server --sse

# Operator launcher (loads .env/.env.local, validates OAuth config, prints smokes)
.venv/bin/python scripts/start_invoicing_readonly_oauth_server.py

# connector boundary smoke (auth + exact read-only tool list; no invoice reads)
python scripts/check_invoicing_readonly_mcp_connector.py \
  --url http://127.0.0.1:8065/mcp \
  --token "$ATLAS_MCP_AUTH_TOKEN"

# OAuth public-discovery smoke (metadata + 401 challenge; no invoice reads)
.venv/bin/python scripts/check_invoicing_readonly_oauth_discovery.py \
  --issuer-url https://atlas-brain.tailc7bd29.ts.net/invoicing-readonly \
  --resource-url https://atlas-brain.tailc7bd29.ts.net/invoicing-readonly/mcp

# OAuth e2e smoke (registration + approval + token + list_tools; no invoice reads)
.venv/bin/python scripts/check_invoicing_readonly_oauth_e2e.py \
  --issuer-url https://atlas-brain.tailc7bd29.ts.net/invoicing-readonly \
  --resource-url https://atlas-brain.tailc7bd29.ts.net/invoicing-readonly/mcp
```

Tools: `get_invoice`, `list_invoices`, `search_invoices`,
`list_pending_drafts`, `customer_balance`, `payment_history`,
`list_services`, `get_service`

This surface is for authenticated ChatGPT-style connector review when only
read tools should be available. It deliberately omits
create/update/approve/send/payment/void/PDF-export and service mutation tools,
but it still requires bearer auth in HTTP mode because the remaining tools
expose customer financial data.

Do not use placeholder or session-local tokens such as `test-token` or
`test-readonly-token` for public HTTP exposure. Generate a long random token,
start the server with that value, then run the connector boundary smoke above
before attaching a ChatGPT-style connector.

ChatGPT online should use OAuth mode, not raw bearer mode. OAuth mode adds MCP
authorization-server metadata, protected-resource metadata, dynamic client
registration, and an operator approval page at `/oauth/approve`. The approval
page posts back to the current external URL, so a path-prefixed public approval
URL keeps its prefix on submit. The page still requires
`ATLAS_MCP_INVOICING_READONLY_OAUTH_APPROVAL_TOKEN`, so the connector is not
silently auto-approved.

Run `scripts/check_invoicing_readonly_oauth_discovery.py` against the public
issuer/resource URLs before attaching ChatGPT. The current public URL is
path-prefixed under `/invoicing-readonly`; if the smoke cannot reach the OAuth
`/.well-known/...` routes, add the missing Tailscale serve route or use a
dedicated hostname before retrying the connector.

For the current Tailscale Funnel shape, the protected-resource metadata route
must preserve the backend path:

```bash
tailscale funnel --bg --yes \
  --set-path /.well-known/oauth-protected-resource \
  http://127.0.0.1:8065/.well-known/oauth-protected-resource
```

After discovery passes, run
`scripts/check_invoicing_readonly_oauth_e2e.py`. It dynamically registers a
temporary OAuth client, approves it with
`ATLAS_MCP_INVOICING_READONLY_OAUTH_APPROVAL_TOKEN`, exchanges the authorization
code for a bearer token, and lists the MCP tools. It must report exactly the
eight read-only tools above and does not call invoice/service/balance/payment
tools.

Use `scripts/start_invoicing_readonly_oauth_server.py` for local operator
startup instead of shell-sourcing `.env` manually. The launcher loads `.env` and
`.env.local`, forces OAuth mode, validates required public URLs and approval
token length, starts the read-only server in the foreground, and prints the
discovery/e2e smoke commands. It masks bearer and approval tokens in output.

### Intelligence MCP Server (33 tools)
```bash
# stdio mode (Claude Desktop / Cursor)
python -m atlas_brain.mcp.intelligence_server

# SSE HTTP mode (port 8061)
python -m atlas_brain.mcp.intelligence_server --sse
```

Tools (Strategic, 8): `generate_intelligence_report`,
`list_intelligence_reports`, `get_intelligence_report`,
`list_pressure_baselines`, `analyze_risk_sensors`,
`run_intervention_pipeline`, `list_pending_approvals`, `review_approval`

Tools (Consumer product reviews, 9): `search_product_reviews`,
`get_product_review`, `list_pain_points`, `list_brands`,
`get_brand_intelligence`, `list_market_reports`, `get_market_report`,
`get_consumer_pipeline_status`, `list_complaint_content`

Tools (Brand registry + fuzzy matching, 4): `list_brand_registry`,
`fuzzy_brand_search`, `add_brand_to_registry`, `add_brand_alias`

Tools (Brand history + change events, 4): `get_brand_history`,
`list_product_change_events`, `list_concurrent_events`,
`get_brand_correlation`

Tools (Consumer corrections, 3): `create_consumer_correction`,
`list_consumer_corrections`, `revert_consumer_correction`

Tools (Displacement + delivery, 5): `list_product_displacement_edges`,
`get_product_displacement_history`, `export_market_report_pdf`,
`export_brand_report_pdf`, `send_brand_health_digest`

**Intelligence + Consumer product reviews**: Strategic entity intelligence
(pressure baselines, behavioral risk, interventions) plus consumer product
review data (Amazon reviews, brand health scores, pain points, competitive
flows, generated content). Two-pass enrichment pipeline.

### B2B Churn Intelligence MCP Server (83 tools, 17 modules)
```bash
# stdio mode (Claude Desktop / Cursor)
python -m atlas_brain.mcp.b2b_churn_server

# SSE HTTP mode (port 8062)
python -m atlas_brain.mcp.b2b_churn_server --sse
```

For the full pipeline / schema / tool-module breakdown, see the
**B2B Churn Intelligence Pipeline** product section below.

**Read intelligence:** `list_churn_signals`, `get_churn_signal`, `list_high_intent_companies`,
`get_vendor_profile`, `get_vendor_history`, `compare_vendor_periods`

**Reviews & reports:** `search_reviews`, `get_review`, `list_reports`, `get_report`, `export_report_pdf`

**Product intelligence:** `get_product_profile`, `get_product_profile_history`, `match_products_tool`

**Displacement graph:** `list_displacement_edges`, `get_displacement_history`, `list_vendor_pain_points`,
`list_vendor_use_cases`, `list_vendor_integrations`, `list_vendor_buyer_profiles`

**Vendor registry:** `list_vendors_registry`, `fuzzy_vendor_search`, `fuzzy_company_search`,
`add_vendor_to_registry`, `add_vendor_alias`

**Scrape target admin:** `list_scrape_targets`, `add_scrape_target`, `manage_scrape_target`, `delete_scrape_target`

**Pipeline & health:** `get_pipeline_status`, `get_parser_version_status`, `get_source_health`,
`get_source_telemetry`, `get_source_capabilities`, `get_operational_overview`, `get_parser_health`

**Corrections:** `create_data_correction`, `list_data_corrections`, `revert_data_correction`,
`get_data_correction`, `get_correction_stats`, `get_source_correction_impact`

**Calibration:** `get_calibration_weights`, `trigger_score_calibration`, `record_campaign_outcome`,
`get_signal_effectiveness`, `get_outcome_distribution`

**Change events:** `list_change_events`, `list_concurrent_events`, `get_vendor_correlation`

**Webhooks:** `list_webhook_subscriptions`, `send_test_webhook_tool`, `update_webhook`, `get_webhook_delivery_summary`

**CRM events:** `list_crm_pushes`, `list_crm_events`, `ingest_crm_event`, `get_crm_enrichment_stats`

**Content:** `list_blog_posts`, `get_blog_post`, `list_affiliate_partners`

**Account intelligence:** `list_account_intelligence`, `get_account_intelligence`,
`build_accounts_in_motion`

**Category dynamics:** `list_category_dynamics`, `get_category_dynamics`

**Segment & temporal intelligence:** `list_segment_intelligence`, `get_segment_intelligence`,
`list_temporal_intelligence`, `get_temporal_intelligence`

**Displacement dynamics:** `list_displacement_dynamics`, `get_displacement_dynamics`,
`get_source_impact_ledger`

**Evidence vaults:** `list_evidence_vaults`, `get_evidence_vault`

**Cross-vendor conclusions:** `list_cross_vendor_conclusions`, `get_cross_vendor_conclusion`,
`persist_conclusion`, `persist_report`, `reason_vendor`, `compare_vendors`

**Generated assets:** `build_challenger_brief`, `draft_campaign`

Data sourced from 19 review sites (incl. Twitter/X via Web Unlocker). See
the **B2B Churn Intelligence Pipeline** section for the full source list.

```bash
ATLAS_MCP_B2B_CHURN_ENABLED=true
ATLAS_MCP_B2B_CHURN_PORT=8062
```

### Universal Scraper MCP Server (5 tools)
```bash
# stdio mode (Claude Desktop / Cursor)
python -m atlas_brain.mcp.scraper_server

# SSE HTTP mode (port 8063)
python -m atlas_brain.mcp.scraper_server --sse
```

Tools: `scrape_url`, `scrape_multi`, `get_scrape_job`, `get_scrape_results`,
`list_scrape_jobs`

LLM-powered extraction from any site with caller-supplied schema. Supports
pagination and Playwright JS rendering for dynamic pages.

### Memory MCP Server (15 tools)
```bash
# stdio mode (Claude Desktop / Cursor)
python -m atlas_brain.mcp.memory_server

# SSE HTTP mode (port 8064)
python -m atlas_brain.mcp.memory_server --sse
```

**Graph tools** (Postgres + Neo4j knowledge graph via `graphiti-wrapper` on
`localhost:8001`): `search_memory`, `search_memory_enhanced`,
`search_memory_temporal`, `get_entity`, `traverse_graph`, `find_shortest_path`,
`add_fact`, `add_episode`, `delete_episode`, `enhance_prompt`,
`analyze_sentiment`

**Conversation tools** (Postgres conversation_turns): `search_conversations`,
`get_session_history`, `list_sessions`

**Combined**: `get_context` (parallel graph + conversation lookup, unified result).

The wrapper service is started with `start-graphiti.sh` (compose file
`docker-compose.graphiti.yml`).

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
    "atlas-invoicing-readonly": {
      "command": "python",
      "args": ["-m", "atlas_brain.mcp.invoicing_readonly_server"],
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
    },
    "atlas-scraper": {
      "command": "python",
      "args": ["-m", "atlas_brain.mcp.scraper_server"],
      "cwd": "/path/to/ATLAS"
    },
    "atlas-memory": {
      "command": "python",
      "args": ["-m", "atlas_brain.mcp.memory_server"],
      "cwd": "/path/to/ATLAS"
    }
  }
}
```

The repo root also contains a `.mcp.json`, but treat it as a personal
scratch — at the time of writing it only registers three servers and
uses a machine-local Python path. Don't follow it as the canonical
configuration; use the snippet above and adjust `cwd` to your checkout.

## Environment Requirements

- NVIDIA GPU with 24GB+ VRAM (RTX 3090/4090) - single GPU setup
  - LLM (qwen3:14b): ~10GB VRAM
  - ASR (Nemotron 0.6B): ~2GB VRAM
- NVIDIA Container Toolkit installed on host (see `install_nvidia_toolkit.sh`)
- Docker and Docker Compose
- Ollama for LLM serving

---

## B2B Churn Intelligence Pipeline (headline product)

End-to-end: 19 review sources → enrichment → weekly churn signals →
displacement graph → reports → webhooks. The whole pipeline is exposed via
the `b2b_churn_server` MCP (83 tools, 17 modules under `atlas_brain/mcp/b2b/`).

### Data flow

```
APScheduler                                    Tenant integrations
(autonomous/scheduler.py)                              ▲
       │                                               │
       ▼                                       (HMAC-SHA256 signed)
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Scrape (19      │    │  Enrichment      │    │  Webhook         │
│  sources, see    │───▶│  (~60 *.py under │    │  dispatcher      │
│  ReviewSource    │    │  services/b2b/)  │    │  (root-level     │
│  enum)           │    │                  │    │  webhook_dispat- │
└──────────────────┘    └──────────────────┘    │  cher.py)        │
       │                       │                └──────────────────┘
       ▼                       ▼                          ▲
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ b2b_reviews      │    │ b2b_churn_signals│    │ b2b_intelligence │
│ (raw + parser_   │    │ (weekly UPSERT   │    │ (reports:        │
│  version)        │    │  per vendor x    │    │  weekly_churn,   │
│                  │    │  category)       │    │  vendor_score-   │
│                  │    │                  │    │  card, …)        │
└──────────────────┘    └──────────────────┘    └──────────────────┘
                               │
                               ▼
                       ┌──────────────────┐
                       │ b2b_displacement_│
                       │ edges (append-   │
                       │ only time-series,│
                       │ velocity 7d/30d) │
                       └──────────────────┘
```

### 19 review sources (`atlas_brain/services/scraping/sources.py`)

G2, Capterra, TrustRadius, Gartner, PeerSpot, GetApp, Product Hunt,
Trustpilot, Reddit, Hacker News, GitHub, YouTube, Stack Overflow, Quora,
Twitter/X (via Web Unlocker), RSS, Software Advice, SourceForge, Slashdot.
`REQUIRED_SCRAPE_SOURCES` constrains deployments to at least Capterra,
TrustRadius, Software Advice. Source groupings: `VERIFIED_SOURCES` (8),
`SLUG_SOURCES` (11), `SEARCH_SOURCES` (7), `API_SOURCES` (6).

### Schema highlights

| Table | Migration | Shape |
|---|---|---|
| `b2b_reviews` | 055 | Raw imports; `source`, `imported_at`, `enriched_at`, `enrichment_status`, `parser_version` |
| `b2b_churn_signals` | 055 (+243 indices) | UPSERT per (vendor_name, COALESCE(product_category, '')); ~27 numeric/text + 10 JSONB cols (top_pain_categories, top_competitors, top_feature_gaps, sentiment_distribution, buyer_authority_summary, insider_signal_count, keyword_spike_count, archetype, reasoning_mode, …) |
| `b2b_company_signals` | 099 | UPSERT per (company_name, vendor_name); urgency, pain, buying_stage, decision_maker, seat_count, contract_end |
| `b2b_displacement_edges` | 099 | Append-only; (from_vendor, to_vendor, computed_date) unique; `signal_strength` enum (strong/moderate/emerging), `primary_driver`, `velocity_7d`, `velocity_30d`, `key_quote`, `confidence_score` |
| `b2b_intelligence` | 055 | Reports — `report_type` enum (weekly_churn_feed / vendor_scorecard / displacement_report / category_overview / vendor_retention / challenger_intel), `intelligence_data` JSONB, `data_density`, `status`, `llm_model` |
| `b2b_scrape_targets` | 056 | Target planning (vendor, product, source, max_pages, priority, scrape_mode) |
| `b2b_scrape_log` | 055 | Audit trail (source, status, reviews_found, reviews_inserted, duration_ms) |
| `b2b_webhook_subscriptions` | 107 | Tenant subscriptions (account_id, url, secret, event_types[]) |
| `b2b_webhook_delivery_log` | 107 | Append-only delivery attempts (status_code, duration_ms, attempt) |

### Calibration & parser-version telemetry

- **Calibration loop** (`mcp/b2b/calibration.py`): `record_campaign_outcome`
  writes outcomes (meeting_booked / deal_opened / deal_won / deal_lost /
  no_opportunity / disqualified) with optional revenue + notes.
  `get_signal_effectiveness` groups completed sequences by signal dimension
  (buying_stage, role_type, urgency_bucket, opportunity_score_bucket) and
  returns positive_outcome_rate per group.
- **Parser-version telemetry** (`mcp/b2b/pipeline.py`): `get_parser_version_status`
  returns `current_version` / `total_reviews` / `current_count` /
  `outdated_count` / `unknown_count` per source. Outdated rows auto-requeue
  for re-enrichment.

### MCP tool layout (`atlas_brain/mcp/b2b/`, 83 tools / 17 modules)

The full breakdown by module (per-row counts sum to 83; default
`mcp_tool_groups=all` in `MCPConfig` registers every group):

| Module | Tools | Domain |
|---|---|---|
| `signals.py` | 6 | List / get churn signals, high-intent companies, trends, anomalies, suppress |
| `displacement.py` | 6 | Edges, history, pain points, evidence, competitive set, competitor analysis |
| `reports.py` | 3 | List / get / export PDF |
| `webhooks.py` | 4 | List subscriptions, send test, update, delivery log |
| `calibration.py` | 5 | Outcome record + effectiveness/distribution/variance |
| `pipeline.py` | 8 | Pipeline status, parser-version, source health, enrichment queue mgmt |
| `evidence.py` | 12 | List / detail / search by claim or reviewer / tag / suppress / provenance / credibility / gaps / export |
| `products.py` | 3 | Product profiles |
| `vendor_registry.py` | 5 | List / get / add / merge / bulk-import |
| `vendor_history.py` | 5 | Timeline, historical signal, version compare, snapshot, export |
| `scrape_targets.py` | 4 | List / get / create / update |
| `crm_events.py` | 4 | Record / list / get / bulk-import CRM events |
| `cross_vendor.py` | 2 | Cross-vendor churn analysis + correlation |
| `corrections.py` | 6 | Analyst corrections (CRUD + batch + export) |
| `content.py` | 3 | Content pieces |
| `reviews.py` | 2 | Search reviews + review detail |
| `write_intelligence.py` | 5 | Draft report types + publish |

### Frontend pages (`atlas-intel-ui/src/App.tsx`)

`/b2b` (KPI dashboard) · `/b2b/onboarding` · `/b2b/signals` (sortable
vendor table) · `/b2b/signals/:vendor` (deep-dive + competing vendors +
quotable evidence) · `/b2b/leads` (high-intent pipeline, urgency >= 7,
last 30d) · `/b2b/leads/:company` · `/b2b/displacement` (Sankey of
from→to flows) · `/b2b/reports` · `/b2b/reviews` · `/b2b/campaigns`.

### Adjacent: Consumer Intelligence

Same architectural shape ported to consumer product reviews (Amazon, etc.):
brand registry + fuzzy matching (`pg_trgm` + difflib), displacement edges,
pain points, market reports, MCP `intelligence_server` (33 tools — 8
strategic + 9 consumer product + 4 brand registry + 4 brand history/change
events + 3 corrections + 5 displacement/delivery). See
`docs/consumer_intelligence_roadmap.md` (phases 0–6+ marked complete).

---

## Content Ops Pipeline (active iteration)

`extracted_content_pipeline/` (~77K LOC, 153 files) is the active build
slice. It generates blog posts, B2B campaigns, landing pages, reports, and
sales briefs from structured customer data, with a generated-asset review
API for human approval before send.

The host brain wires it through three flat modules at `atlas_brain/`:
`_content_ops_scope.py`, `_content_ops_services.py`,
`_content_ops_infrastructure.py` (see *Extracted Packages → Wiring*
above).

### What's wired today (E0–E3 landed)

- **E0** Contract tests, domain layer, API adapter
- **E1** `signal_extraction` service (deterministic, no LLM)
- **E2** Tenant scope ContextVar + landing-page generator wired into the
  execution-services bundle
- **E3** Control surfaces reasoning provider — route-level reasoning context
  seam shared across all 5 generators

### What's in flight (E4+ — 39 of 65 open plans are Content-Ops)

- Wire `campaign`, `blog_post`, `report`, `sales_brief` generators (each
  needs `IntelligenceRepository` host factory + `LLMClient` + `SkillStore`)
- Frontend Screen 1 (New Run controls), Screen 2 (Plan Preview verdict /
  cost / missing-input validation), Screen 3 (Asset Review)
- Per-output dynamic input forms; result summary + export APIs

The recent commit log (`PR-Content-Ops-*`) is the source of truth for
which slices have landed. See `plans/PR-Content-Ops-*.md` for plan docs.

---

## Planned / In-Flight Work

### P0 (per `BUILD_SPEC.md`) — Voice-to-Voice agent unification

Components built but not unified:
- `atlas_brain/voice/` — wake word (OpenWakeWord), VAD (WebRTC), audio
  capture, frame processor, Pipecat pipeline, launcher.
- `asr_server.py` — Nemotron 0.6B streaming ASR (port 8081).
- TTS — Piper, wired direct in `atlas_brain/voice/pipeline.py`.

**Gap (per `CONTEXT.md`):** The Pipecat voice pipeline does not currently
flow through the unified Atlas Agent — voice and text APIs use different
code paths. P0 is to consolidate behind a single agent entry point so
both surfaces share intent routing, conversation state, and tool use.

### P1 / P2 (blocked by P0)

- **P1** Home Assistant integration through unified agent (capability
  registry already exists in `atlas_brain/capabilities/`).
- **P2** Atlas-native voice-first features — time/date, weather, timers,
  reminders, calendar (no external HA dependency).

### Roadmaps marked complete

- `docs/intelligence_platform_roadmap.md` — B2B phases 0–7 (source
  normalization → product language repositioning).
- `docs/consumer_intelligence_roadmap.md` — Consumer phases 0–6+.

### Other planned but not started

- Human tracking / face recognition, object detection — design docs only
  (`docs/progress/gui_camera_audit.md`); no detector code in tree.
- Distributed Jetson Nano nodes / multi-room A/V coordination — diagram
  only in CLAUDE.md vision; no code.
- PersonaPlex (phone integration) — partial; deprioritized at 22s latency.

### How to find the rest

- `plans/` — 65 plan docs, ~39 Content-Ops, ~8 Audit/Testing, ~5
  Caching/Optimization, ~8 B2B/Reasoning. Sorted by mtime is the rough
  current-iteration view.
- `BUILD_SPEC.md` — P0/P1/P2 tiers + definition of done.
- `CONTEXT.md` — known debt + session notes.
- `INTEGRATION_MAP.md` — what's actually wired to what (vs. what looks
  wired in the file tree).
- `CANONICAL.md` — when two implementations exist, which one is real.

---

## Multi-Session PR Workflow (AGENTS.md)

Atlas uses **two coordinated Claude Code sessions** for non-trivial work: a
**builder** (writes the plan, the code, the PR) and a **reviewer** (audits the
PR independently). The full contract lives in `AGENTS.md`; the highlights:

- **Plan first.** Every non-trivial PR ships a plan doc at
  `plans/PR-<Slice-Name>.md` with these 7 required sections, in order:
  Why this slice exists / Scope / Mechanism / Intentional / Deferred /
  Verification / Estimated diff size.
- **Diff budget:** target **<400 LOC** per PR (soft cap). Over-budget PRs
  must justify the overage in *Why this slice exists*.
- **PR body** mirrors the plan-doc framing, with `Plan: plans/PR-<Slice>.md`
  as the lead line. Same shape goes in the commit message.
- **Branch naming:** `claude/pr-<slice-name>` for builder branches;
  `claude/<topic>` for non-PR scratch.
- **Open ready for review** by default. Do not open draft PRs unless the
  operator explicitly asks for a draft; automated review tools do not review
  draft PRs.
- **Reviewer verdicts:** `BLOCKER` / `MAJOR` / `NIT` / `LGTM`. Reviewer
  reproduces the builder's verification commands; doesn't trust claims.
- **No "while I was here" cleanups.** Plan and implementation ship together;
  off-scope changes go to a follow-up slice (added to *Deferred*).

When extending a Claude Code session in this repo, **read `AGENTS.md` first**
if the task is a non-trivial PR. For one-off scratch / exploration, the
contract doesn't apply.

Companion docs:
- `AUDITOR_PROMPT.md` — cross-cutting auditor prompt (canonical / integration / scope / debt)
- `CANONICAL.md` — which implementation is the real one
- `INTEGRATION_MAP.md` — what's wired to what
- `BUILD_SPEC.md` — P0/P1/P2 priorities, definition of done
- `CONTEXT.md` — session notes, known debt

### PR Reviews

When asked to review a PR — whether by a local Claude Code session, a GitHub
Actions run (e.g., `claude-code-review.yml`), or an `@claude` mention — the
deliverable is **comments on the PR**, not a chat summary. Findings must be
posted via:

```bash
gh api repos/{owner}/{repo}/pulls/{pr}/reviews --input - < /tmp/review.json
```

with `event: "COMMENT"` and an inline `comments[]` array. A chat-only
narration of findings is not a deliverable; the comments on the PR are.

Rules:

- **Inline comments only resolve on lines in the PR diff.** GitHub returns
  HTTP 422 "Line could not be resolved" for out-of-diff line targets. For
  findings on unchanged lines, either move the comment to the closest
  in-diff line or include them in the review `body` with a `file:line`
  reference.
- **Use stdin** (`--input - < file.json`) rather than `--input
  /path/file.json` — the `gh` CLI sometimes errors with "no such file" on
  direct file paths under sandboxing.
- **Verify line numbers against the actual file at the PR head**
  (`git show <sha>:<path>` or fetch the branch) before posting. Don't
  trust the agent's summary blindly.
- **Chat summary AFTER posting is fine** as a recap — but the review on
  the PR is the deliverable.
- **For review-on-a-PR tasks specifically, posting is mandatory.** If the
  task only asked you to *analyze* (not post), surface findings in the
  return message in the structured form.

The `code-review-agent` definition at `.claude/agents/code-review-agent.md`
encodes the same rules. The `claude-code-review.yml` GitHub Action spawns
this agent on every PR open/push and inherits these conventions from this
file.

---

## Testing

Pytest is the only test runner. Config is in `pytest.ini`:

```ini
[pytest]
testpaths = tests
asyncio_mode = auto
markers =
    integration: marks tests as integration tests (require database)
    e2e:         marks tests as end-to-end tests (require all services)
    slow:        marks tests as slow running
```

```bash
# Activate venv first
source .venv/bin/activate

# Full unit suite (skip DB-bound tests)
pytest -m "not integration and not e2e"

# Just one test file
pytest tests/test_blog_post_postgres.py -v

# Run integration tests (requires Postgres + Neo4j up)
docker compose up -d postgres
pytest -m integration

# Single keyword filter
pytest -k "campaign and not slow"
```

`tests/` has 600+ files at the top level plus `tests/atlas_edge/`,
`tests/security/`, `tests/fixtures/`, and a shared `conftest.py`.

### Per-package validation gauntlets

Each `extracted_*` package has a CI-equivalent local sweep. Run before
pushing changes that touch that package:

```bash
# extracted_content_pipeline (active iteration)
bash scripts/validate_extracted_content_pipeline.sh
python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
python scripts/audit_extracted_standalone.py --fail-on-debt
bash scripts/check_ascii_python.sh
bash scripts/run_extracted_pipeline_checks.sh                      # full CI mirror

# Sync touched files from atlas_brain → extracted (only if synced files changed)
bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline

# Other packages
bash scripts/validate_extracted_competitive_intelligence.sh
bash scripts/run_extracted_competitive_intelligence_checks.sh
bash scripts/validate_extracted_llm_infrastructure.sh
bash scripts/run_extracted_llm_infrastructure_checks.sh
bash scripts/run_extracted_evidence_to_story_checks.sh

# Reasoning rollout / hybrid checks
python scripts/check_reasoning_rollout_readiness.py
bash scripts/run_hybrid_reasoning_checks.sh
bash scripts/run_reasoning_provider_port_compat_checks.sh
```

`scripts/` also holds ~150 backfill / audit utilities (e.g.
`audit_g2_raw_capture.py`, `backfill_blog_seo.py`,
`run_b2b_enrichment_until_exhausted.py`). These are one-off ops tools, not
part of CI.

---

## Extracted Packages

Atlas extracts cohesive subsystems into standalone Python packages so they
can be exercised, tested, and (eventually) shipped independently of the main
brain. Six packages exist today:

| Package                              | LOC   | Files | Status                  | Purpose |
|--------------------------------------|-------|-------|-------------------------|---------|
| `extracted_content_pipeline/`        | ~77K  | 153   | **Active iteration (E2+)** | Blog post + B2B campaign + landing-page + report + sales-brief generation, signal extraction, draft export/review APIs, campaign lifecycle (gen → draft → review → queue → send), webhook ingestion, sequence progression, seller outreach |
| `extracted_competitive_intelligence/`| ~20K  | 85    | Phase 1 complete        | Vendor registry + dedupe, displacement edges, battle cards, weekly vendor briefings, cross-vendor reasoning, win/loss inputs |
| `extracted_llm_infrastructure/`      | ~9K   | 42    | Phase 2 substrate landed; Phase 3 = DI seams | 8 LLM providers (Anthropic, OpenRouter, Ollama, vLLM, Groq, Together, Hybrid, GCP), Anthropic batches w/ dedup+replay, semantic+exact cache, FTL tracing, cost-closure (savings/drift/budget gates) |
| `extracted_reasoning_core/`          | ~5.7K | 19    | Mature                  | Hierarchical multi-tier reasoning (shallow/balanced/deep), 10-archetype scoring, evidence evaluation, temporal evidence, narrative planning, semantic-cache key derivation, wedge/pack registries |
| `extracted_quality_gate/`            | ~4.7K | 14    | PR-B5c (May 2026)       | Deterministic safety gate (regex catalogue), packs: blog / campaign / witness specificity / evidence coverage / source quality |
| `extracted_evidence_to_story/`       | ~640  | 3     | Pre-alpha (schema only) | Typed claim ledger contract (factual / timeline / entity / emotional_inference / disputed / reveal / transition × verified / inferred / disputed / unknown). LLM extractor deferred. |

### Wiring into the host brain

`extracted_content_pipeline` is wired into `atlas_brain/` via three
intentionally-flat scaffolding modules at the package root (so tests can
import them without booting the router init chain):

- `atlas_brain/_content_ops_scope.py` — `capture_content_ops_auth_user()`
  ContextVar bridge from `AuthUser` → typed `TenantScope`.
- `atlas_brain/_content_ops_services.py` — builds the
  `ContentOpsExecutionServices` bundle (LLM, skill store, Postgres pool);
  `enable_db_services` flag gates DB-backed services.
- `atlas_brain/_content_ops_infrastructure.py` — `HostLLMClient` +
  `HostSkillStore` adapters bridging Atlas's LLM/skill registries into the
  package's ports; runs host `chat()` in a worker thread for async parity.

`extracted_competitive_intelligence` is wired in via
`atlas_brain/reasoning/single_pass_prompts/cross_vendor_battle.py` and
`atlas_brain/autonomous/tasks/b2b_vendor_briefing.py`.

`extracted_llm_infrastructure` is wired in via `atlas_brain/services/llm_router.py`
and `atlas_brain/services/b2b/anthropic_batch.py`.

`extracted_reasoning_core` is wired in throughout
`atlas_brain/reasoning/` (archetypes, evidence engine, semantic cache keys).

`extracted_quality_gate` re-exports through
`atlas_brain/autonomous/tasks/_b2b_specificity.py` and
`atlas_brain/services/b2b/witness_render_gate.py`.

`extracted_evidence_to_story` is **not** wired in yet (schema-only).

### Manifest discipline (synced vs owned)

Each `extracted_<name>/manifest.json` has `mappings` entries in two shapes:

- **Synced** — entries with both `source` (path under `atlas_brain/`) and
  `target`. The source is canonical; edit `atlas_brain/...` and run
  `extracted/_shared/scripts/sync_extracted.sh <package>` to propagate.
- **Owned** — entries with `target` only. The package copy is canonical.
  Edit in place. The sync script does not overwrite it.

```bash
grep -B2 '"target": "<path>"' <package>/manifest.json   # check which side
```

A `source` line means synced; absence (just `target`) means owned.

### Shared tooling under `extracted/_shared/scripts/`

| Script | Purpose |
|---|---|
| `sync_extracted.sh` | Refresh extracted targets from `atlas_brain/` source paths in `manifest.json` |
| `forbid_atlas_reasoning_imports.py` | Hard-fail any `atlas_brain.reasoning` import in extracted packages (must use `extracted_reasoning_core` instead) |
| `forbid_hard_atlas_imports.py` | Allow gated `atlas_brain` imports (try/except, env branches); fail on hard top-level imports |
| `check_extracted_imports.py` | Validate package import structure |
| `validate_extracted.sh` | Run forbid checks + import validation per package |
| `check_ascii_python.sh` | ASCII-only `.py` enforcement |

---

## Sub-Projects

| Path                          | Stack             | Purpose |
|-------------------------------|-------------------|---------|
| `atlas-admin-ui/`             | React + Vite + TS | Internal admin dashboard |
| `atlas-churn-ui/`             | React + Vite + TS | B2B churn intelligence dashboard |
| `atlas-intel-ui/`             | React + Vite + TS | Strategic intelligence dashboard |
| `atlas-ui/`                   | React + Vite + TS | Main consumer dashboard |
| `atlas-mobile/`               | React Native      | Mobile app |
| `portfolio-ui/`               | React + Vite + TS | Portfolio management UI |
| `animated-robot-logo/`        | static / animation| Logo animation assets |
| `atlas_comms/`                | Python service    | Standalone Twilio/SignalWire bridge |
| `atlas_edge/`                 | Python service    | Edge-node capabilities (Jetson) |
| `atlas_video-processing/`     | Python service    | Video pipeline (SAM 3) |
| `graphiti-wrapper/`           | Python service    | Neo4j + Graphiti GraphRAG service (port 8001) |

UI projects each have their own `package.json`; run `npm install && npm run dev`
inside the directory. Python sub-services have their own `requirements*.txt`
and Dockerfiles (`Dockerfile.graphiti`, `Dockerfile`).

Compose files for sub-services:
- `docker-compose.yml` — main brain + Postgres + NocoDB
- `docker-compose.graphiti.yml` — Neo4j + Graphiti wrapper (use `start-graphiti.sh`)
- `docker-compose.ha.yml` / `docker-compose.homeassistant.yml` — Home Assistant
- `docker-compose.wyze.yml` — Wyze cam bridge

---

## Key Conventions

- **Config**: every setting goes through `atlas_brain/config.py`
  (Pydantic Settings, `env_prefix=ATLAS_*`). Never read `os.environ`
  directly — add a typed field on the relevant `BaseSettings` subclass.
- **Async-first**: all I/O is `async def`. Database access is `asyncpg` or
  the typed providers in `atlas_brain/services/` — no synchronous DB calls.
- **Single source of truth for CRM**: `crm_provider.get_crm_provider()`
  returns `DatabaseCRMProvider`. Don't reach into the `contacts` table from
  random callers.
- **Provider-agnostic ports**: email (`email_provider`), calendar
  (`calendar_provider`), LLM (`llm_router`) all expose ports with multiple
  adapters. Add new providers behind the existing port; don't fork.
- **MCP servers are thin**: each `atlas_brain/mcp/<name>_server.py` should
  delegate to a service in `atlas_brain/services/`, not embed business
  logic. Tools = transport, services = behavior.
- **Skills = injectable prompts**: domain prompts live as markdown under
  `atlas_brain/skills/<domain>/`. Load via `skills/registry.py`. Don't
  hard-code domain prompts inline.
- **Plans are mandatory** for non-trivial PRs (see *Multi-Session PR
  Workflow* above). No code without a plan doc.
- **ASCII Python**: `scripts/check_ascii_python.sh` is part of CI.
  Non-ASCII characters in `.py` files break the gate. Use ASCII-only
  identifiers and string literals.
- **No `--no-verify`**: never bypass pre-commit / CI gates without explicit
  user authorization.
