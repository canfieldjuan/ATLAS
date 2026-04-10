# Atlas

A multi-modal AI platform that combines personal automation, voice control, B2B sales intelligence, and consumer product analytics into a single extensible system. Atlas runs as a central "Brain" server backed by local LLMs, with edge nodes for distributed sensing, 4 web dashboards, 8 MCP servers (130+ tools), 57 autonomous scheduled tasks, a full telephony stack, and 500+ REST/WebSocket API endpoints.

---

## System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          ATLAS BRAIN                                  │
│                   (FastAPI · 500+ API endpoints)                    │
│                                                                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐   │
│  │  LLM Pool  │  │    STT     │  │    TTS     │  │   Intent     │   │
│  │ Ollama     │  │ Nemotron   │  │ Piper      │  │   Router     │   │
│  │ vLLM       │  │ SenseVoice │  │ Kokoro     │  │ (Embeddings) │   │
│  │ Claude     │  └────────────┘  └────────────┘  └──────────────┘   │
│  │ OpenRouter │                                                       │
│  └────────────┘                                                       │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                    12 LangGraph Workflows                     │    │
│  │  Voice · Email · Calendar · Booking · Reminder · Call         │    │
│  │  Security · Presence · Receptionist · Home · Streaming        │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                  57 Autonomous Tasks (APScheduler)            │    │
│  │  Enrichment · Campaigns · Briefings · Intelligence · Alerts   │    │
│  │  Memory Sync · Pattern Learning · Anomaly Detection           │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                  8 MCP Servers (130+ tools)                   │    │
│  │  CRM · Email · Twilio · Calendar · Invoicing                 │    │
│  │  Intelligence · B2B Churn (61 tools) · Memory                 │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐       │
│  │  PostgreSQL    │  │  Neo4j         │  │  68 Skill Docs   │       │
│  │  (50+ tables)  │  │  (GraphRAG)    │  │  (LLM Prompts)   │       │
│  └────────────────┘  └────────────────┘  └──────────────────┘       │
└──────────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐
│  Edge Node   │   │  Telephony   │   │    4 Web Dashboards   │
│  Orange Pi   │   │  Twilio      │   │  Main · Intel · Churn │
│  RK3588      │   │  SignalWire  │   │  Admin Ops            │
│  YOLO·Piper  │   │  Call/SMS    │   │  React + Next.js      │
└──────────────┘   └──────────────┘   └──────────────────────┘
```

---

## What Atlas Does

### Personal AI Assistant
- **Voice interface** — Wake word ("Hey Atlas"), streaming STT, semantic intent routing, TTS response. Works via brain server or edge node.
- **Device control** — "Turn off the TV", "dim the bedroom lights to 30%". Parses natural language into structured intents, dispatches to Home Assistant (WebSocket) or MQTT devices.
- **Multi-turn workflows** — Booking appointments (slot checking, confirmation), composing emails (draft → review → send), setting reminders, querying calendar — all as stateful LangGraph conversations.
- **Telephony** — Inbound/outbound calls via Twilio or SignalWire. Call transcription, receptionist routing, business-hours logic, multi-context phone numbers, SMS auto-reply via LLM.
- **Morning briefings** — Autonomous daily digest: calendar, weather, unread email triage, device health, security events, proactive actions.

### B2B Sales Intelligence
- **Review scraping** — Ingests from 16 sources (G2, Capterra, TrustRadius, Reddit, Gartner, GetApp, GitHub, HackerNews, PeerSpot, ProductHunt, Quora, StackOverflow, TrustPilot, Twitter/X, YouTube, RSS).
- **LLM enrichment** — Each review extracted for pain categories, urgency scores, churn intent, company signals, budget pressure, competitive mentions, buying stage.
- **Churn signal aggregation** — Per-vendor metrics: churn intent rate, NPS proxy, decision-maker churn rate, price complaint rate. Classified into archetypes (pricing_shock, feature_gap, acquisition_decay, support_collapse, etc.) with risk levels (imminent → stable).
- **Evidence pools** — Deterministic intermediate representations (Evidence Vault, Segment Intelligence, Temporal Intelligence, Account Intelligence, Displacement Dynamics, Category Dynamics) computed once and consumed by all downstream artifacts.
- **Displacement graph** — Vendor A → Vendor B competitive flows with mention counts, signal strength, primary switch drivers, velocity tracking, and source distribution.
- **Reasoning synthesis** — Claude-powered compression of evidence pools into structured reasoning contracts (vendor core, displacement, category, account reasoning) with schema validation and citation tracing.
- **Battle cards** — Pairwise competitive positioning generated from reasoning contracts and displacement evidence.
- **Campaign generation** — Opportunity scoring (role weight + buying stage + urgency + seat count + pain categories), calibrated from real CRM outcomes. Auto-generates personalized cold email, follow-up, and LinkedIn variants.
- **Score calibration** — Closed-loop: CRM events (deal_won, deal_lost, meeting_booked) feed back into opportunity scoring weights via `score_calibration_weights` table.
- **Blog generation** — Auto-writes vendor alternatives, migration guides, pricing reality checks, vendor showdowns, switching stories from churn intelligence. Role-aware topic boosting (CFO → pricing, CTO → migration).
- **Vendor briefings** — Email-delivered intelligence packages pulling from 10 data sources (signals, evidence vaults, product profiles, reasoning synthesis, segment/temporal intelligence, displacement dynamics).
- **Product profiles** — Per-vendor knowledge cards: strengths, weaknesses, pain addressed, use cases, typical company size/industry, top integrations, commonly compared/switched-from vendors.
- **Account resolution** — Matches reviews to canonical company identities. Builds witness packets with decision-maker counts, contract end dates, seat counts, buying stage, org pressure signals.
- **CRM integration** — Ingests events from HubSpot, Salesforce, Pipedrive. Matches to campaign sequences. Webhook dispatch to external CRMs with retry logic.

### Consumer Product Intelligence
- **Amazon review pipeline** — Deep enrichment of product reviews (pain categories, product names, alternatives, proof quotes).
- **Brand health scoring** — Review aggregation by brand with safety signals and risk scoring.
- **Complaint analysis** — Customer complaint classification, content generation, competitive flow visualization.
- **Market reports** — Category-level intelligence and ecosystem mapping.

### Strategic Intelligence (Chase Hughes Framework)
- **Risk sensors** — Alignment, urgency, and rigidity detection with correlation analysis.
- **SORAM classification** — Behavioral risk classification on entities.
- **Intelligence reports** — Report orchestration, intervention pipelines, narrative architecture.
- **Simulation** — Simulated evolution of entity behavior under pressure.

---

## Project Structure

```
atlas_brain/                    # Core server (FastAPI, 500+ endpoints)
├── api/                        # REST + WebSocket endpoints
│   ├── query/                  #   Text inference
│   ├── devices/                #   Device control + intent dispatch
│   └── ...                     #   B2B, campaigns, email, calendar, blog, admin
├── agents/graphs/              # 12 LangGraph workflow state machines
├── autonomous/                 # 57 scheduled tasks (APScheduler)
│   ├── scheduler.py            #   Cron/interval task orchestration
│   ├── runner.py               #   Execution engine + ntfy notifications
│   └── tasks/                  #   Task implementations
├── capabilities/               # Device control system
│   ├── backends/               #   Home Assistant (WebSocket), MQTT
│   ├── devices/                #   Lights, switches, media players
│   └── intent_parser.py        #   NL → structured intent
├── mcp/                        # 8 MCP servers (130+ tools)
│   ├── b2b/                    #   B2B churn (14 domain files, 61 tools)
│   ├── memory_server.py        #   GraphRAG memory (13 tools)
│   ├── crm_server.py           #   CRM (10 tools)
│   ├── email_server.py         #   Email (8 tools)
│   ├── twilio_server.py        #   Telephony (10 tools)
│   ├── calendar_server.py      #   Calendar (8 tools)
│   ├── invoicing_server.py     #   Invoicing (15 tools)
│   └── intelligence_server.py  #   Intelligence (17 tools)
├── memory/                     # RAG client, quality detection, embeddings
├── reasoning/                  # Evidence engine, archetypes, narrative synthesis
├── pipelines/                  # Generalized pipeline registry + LLM routing
├── services/                   # Business logic layer
│   ├── b2b/                    #   Account resolution, product matching, PDF render
│   ├── scraping/               #   16-source review scraper
│   ├── embedding/              #   Sentence transformer embeddings
│   ├── recognition/            #   Face/gait recognition
│   ├── speaker_id/             #   Speaker identification
│   ├── llm_router.py           #   Multi-model LLM routing
│   ├── crm_provider.py         #   Direct asyncpg CRM
│   ├── email_provider.py       #   Gmail + Resend + IMAP
│   ├── calendar_provider.py    #   Google Calendar + CalDAV
│   └── mcp_client.py           #   MCP tool consumer (stdio subprocesses)
├── skills/                     # 68 injectable LLM prompt documents
│   ├── digest/                 #   46 analysis/synthesis skills
│   ├── email/                  #   7 email composition skills
│   ├── call/                   #   4 call handling skills
│   ├── invoicing/              #   3 billing skills
│   ├── intelligence/           #   6 strategic analysis skills
│   └── ...
├── storage/                    # PostgreSQL (50+ tables, asyncpg)
│   └── migrations/             #   Versioned SQL migrations
├── voice/                      # End-to-end voice pipeline
│   ├── pipeline.py             #   Main voice loop
│   ├── vad/                    #   Dual VAD (WebRTC + Silero)
│   └── launcher.py             #   Pipeline lifecycle
└── tools/                      # Registered tool implementations

atlas_edge/                     # Edge node (Orange Pi RK3588 / Jetson)
├── capabilities/               #   Local HA device control
├── pipeline/                   #   Voice pipeline (Piper TTS, SenseVoice STT)
├── intent/                     #   Local intent parsing
├── skills/                     #   Offline skills (time, timer, math)
└── brain/                      #   Brain server connection + offline fallback

atlas_comms/                    # Telephony system
├── providers/                  #   Twilio + SignalWire backends
├── services/                   #   Scheduling, context routing
└── api/                        #   Call/SMS/webhook endpoints

atlas_video-processing/         # Distributed vision
├── devices/                    #   Webcam, RTSP, mock cameras
├── processing/                 #   YOLO detection, tracking
└── communication/              #   MQTT device discovery

atlas-ui/                       # Main conversational UI (React + Vite)
atlas-intel-ui/                 # Consumer intelligence dashboard (React)
atlas-churn-ui/                 # B2B churn intelligence dashboard (React)
atlas-admin-ui/                 # Operations dashboard (React)
atlas-intel-next/               # Next.js B2B dashboard (migration in progress)
atlas-mobile/                   # Mobile app

graphiti-wrapper/               # GraphRAG backend (FastAPI + Neo4j)
├── query_utils.py              #   Query decomposition, expansion, temporal detection
├── reranker.py                 #   BM25 + cross-encoder reranking
└── embedder_factory.py         #   OpenAI / local embedder selection

tests/                          # Test suite
scripts/                        # Data import, backfill, setup utilities
```

---

## Voice Pipeline

End-to-end voice interaction, works on both the brain server and edge nodes:

```
Microphone
  → Wake Word Detection (OpenWakeWord: "Hey Atlas")
  → Voice Activity Detection (WebRTC VAD + Silero dual-gate)
  → Audio Capture (buffered PCM frames, 16kHz)
  → ASR (Nemotron 0.6B streaming, or SenseVoice on edge)
  → Semantic Intent Router (all-MiniLM-L6-v2 embeddings, cosine similarity)
  → Route Decision:
      ├─ Device command → Intent Parser → Home Assistant action
      ├─ Parameterless tool → Direct execution (time, weather, calendar)
      ├─ Workflow trigger → LangGraph state machine (booking, email, etc.)
      └─ Conversation → LLM with tool access (CRM, calendar, email, telephony)
  → TTS (Piper on edge, Kokoro on brain)
  → Speaker Output
```

The edge node runs offline-capable local skills (time, timer, math, status) without brain round-trip. When the brain is reachable, queries route there for full LLM + tool access.

---

## LLM Routing

Atlas uses multiple models for different workloads:

| Workload | Model | Where |
|----------|-------|-------|
| Conversation, intent, reminders | qwen3:14b | Local (Ollama) |
| Business workflows (booking, email) | minimax-m2:cloud | Ollama cloud relay |
| Email drafts | Claude Sonnet | Anthropic API |
| Email triage | Claude Haiku | Anthropic API |
| B2B reasoning synthesis | Claude / OpenRouter | Cloud |
| B2B enrichment | Qwen3-30B-A3B | Local (vLLM) |
| Edge node (offline) | SenseVoice (STT) + Piper (TTS) | Orange Pi RK3588 |

The intent router uses `all-MiniLM-L6-v2` (384-dim, CPU) for fast semantic classification before any LLM call. Parameterless tools (get_time, get_weather, etc.) execute without touching the LLM at all.

---

## Autonomous Tasks

57 scheduled tasks run via APScheduler with PostgreSQL as source of truth. Key categories:

**Daily Consumer Tasks:**
- `morning_briefing` — Calendar + email + weather + device health synthesis
- `gmail_digest` — Unread email triage and summary
- `calendar_reminder` — Upcoming event alerts (every 5 min)
- `device_health_check` — Smart home device status
- `security_summary` — Camera/motion event summary (every 6h)
- `email_intake` / `email_draft` / `email_auto_approve` — Email processing pipeline
- `model_swap_day` / `model_swap_night` — GPU memory management

**B2B Intelligence Pipeline:**
- `b2b_enrichment` — LLM extraction on pending reviews (every 5 min)
- `b2b_churn_intelligence` — Weekly signal aggregation + evidence pool building
- `b2b_reasoning_synthesis` — Claude-powered reasoning contracts from evidence pools
- `b2b_product_profiles` — Daily vendor knowledge card generation
- `b2b_campaign_generation` — Nightly opportunity scoring + outreach drafting
- `b2b_score_calibration` — Weekly calibration from CRM outcomes
- `b2b_battle_cards` — Competitive positioning generation
- `b2b_blog_generation` — Auto-write vendor alternative guides
- `b2b_vendor_briefing` — Email-delivered intelligence packages
- `b2b_accounts_in_motion` — High-intent account identification
- `b2b_scrape_target_pruning` — Optimize scrape spend

**Intelligence & Learning:**
- `daily_intelligence` / `competitive_intelligence` / `news_intelligence` — Market signals
- `pattern_learning` / `preference_learning` — User behavior modeling
- `anomaly_detection` — Churn signal anomaly alerting (every 15 min)
- `knowledge_graph_sync` — Nightly Neo4j sync from conversation history
- `reasoning_reflection` — Persistent reasoning updates

All tasks support ntfy push notifications, per-task opt-out, and `_skip_synthesis` for empty results.

---

## MCP Servers

Eight MCP servers expose Atlas to any MCP-compatible client (Claude Desktop, Cursor, custom agents). All support stdio (for local clients) and SSE HTTP (for remote access).

| Server | Tools | Port | Description |
|--------|-------|------|-------------|
| **Memory** | 13 | 8001 | Semantic search, entity lookup, graph traversal, episode ingestion, sentiment analysis |
| **CRM** | 10 | 8056 | Contact CRUD, interaction logging, appointment tracking, customer context |
| **Email** | 8 | 8057 | Send/read email (Gmail OAuth + IMAP), estimates, proposals, thread search |
| **Twilio** | 10 | 8058 | Calls, SMS, recordings, phone lookup |
| **Calendar** | 8 | 8059 | Google Calendar + CalDAV, free slot finder, appointment sync |
| **Invoicing** | 15 | 8060 | Invoice lifecycle, payments, services, customer balance |
| **Intelligence** | 17 | 8061 | Strategic entity intelligence + consumer product reviews, brand health, market reports |
| **B2B Churn** | 61 | 8062 | Full B2B intelligence: signals, reviews, displacement, campaigns, calibration, corrections, webhooks |

### B2B Churn MCP (61 tools across 14 domains)

The largest MCP server, organized by domain:

- **Signals** — `list_churn_signals`, `get_churn_signal`, `get_vendor_profile`, `list_high_intent_companies`
- **Reviews** — `search_reviews`, `get_review` (enriched with pain, urgency, churn intent)
- **Displacement** — `list_displacement_edges`, `get_displacement_history`, `list_vendor_pain_points`, `list_vendor_use_cases`, `list_vendor_integrations`, `list_vendor_buyer_profiles`
- **Products** — `get_product_profile`, `get_product_profile_history`, `match_products_tool`
- **Evidence Pools** — `get_evidence_vault`, `get_segment_intelligence`, `get_temporal_intelligence`, `get_account_intelligence`, `get_displacement_dynamics`, `get_category_dynamics`
- **Cross-Vendor** — `list_cross_vendor_conclusions`, `get_cross_vendor_conclusion` (battle cards, category councils)
- **Reports** — `list_reports`, `get_report`, `export_report_pdf`
- **Vendor Registry** — `list_vendors_registry`, `fuzzy_vendor_search`, `fuzzy_company_search`, `add_vendor_to_registry`, `add_vendor_alias`
- **Scrape Admin** — `list_scrape_targets`, `add_scrape_target`, `manage_scrape_target`, `delete_scrape_target`
- **Corrections** — `create_data_correction`, `list_data_corrections`, `revert_data_correction`, `get_correction_stats`, `get_source_correction_impact`
- **Calibration** — `record_campaign_outcome`, `get_signal_effectiveness`, `get_outcome_distribution`, `trigger_score_calibration`, `get_calibration_weights`
- **Change Events** — `list_change_events`, `list_concurrent_events`, `get_vendor_correlation`
- **Webhooks** — `list_webhook_subscriptions`, `update_webhook`, `send_test_webhook_tool`, `get_webhook_delivery_summary`
- **CRM Events** — `list_crm_events`, `ingest_crm_event`, `get_crm_enrichment_stats`
- **Content** — `list_blog_posts`, `get_blog_post`, `list_affiliate_partners`
- **Write** — `persist_report`, `persist_conclusion`, `draft_campaign`, `build_challenger_brief`, `build_accounts_in_motion`
- **Pipeline** — `get_pipeline_status`, `get_parser_health`, `get_source_health`, `get_source_telemetry`, `get_operational_overview`

---

## B2B Intelligence Pipeline

The full data flow from raw reviews to outbound campaigns:

```
16 Review Sources (G2, Capterra, Reddit, Gartner, etc.)
    │
    ▼
b2b_enrichment (every 5 min, vLLM extraction)
    │  pain_category, urgency_score, churn_intent, company_signals,
    │  budget_pressure, competitive_mentions, buying_stage
    ▼
b2b_churn_intelligence (weekly)
    │  Aggregates per-vendor metrics, computes archetypes + risk levels
    │  Builds 6 evidence pools:
    │    Evidence Vault · Segment Intelligence · Temporal Intelligence
    │    Account Intelligence · Displacement Dynamics · Category Dynamics
    ▼
b2b_reasoning_synthesis (after intelligence)
    │  Claude compresses pools → structured reasoning contracts
    │  Validates citations, enforces schema, decomposes into:
    │    vendor_core · displacement · category · account reasoning
    ▼
┌───────────────────────────────────────────────────────┐
│  Downstream consumers (all compose from pools/contracts) │
│                                                         │
│  battle_cards ─── competitive positioning               │
│  campaigns ────── opportunity-scored outreach            │
│  blog_posts ──── SEO content (alternatives, guides)     │
│  vendor_briefings ── email intelligence packages        │
│  product_profiles ── vendor knowledge cards             │
│  tenant_reports ─── per-customer intelligence           │
│  churn_alerts ──── high-intent notifications            │
│  accounts_in_motion ── at-risk account identification   │
└───────────────────────────────────────────────────────┘
    │
    ▼
CRM Feedback Loop
    ingest_crm_event (deal_won/lost/meeting_booked)
    → record_campaign_outcome
    → score_calibration (weekly, updates opportunity weights)
```

---

## Web Dashboards

### Main UI (`atlas-ui/`)
Sci-fi themed conversational interface. Real-time audio waveform visualization, system metrics (CPU/memory/network), conversation history, live event feed, voice/text input with privacy mode, settings for voice pipeline and integrations.

### B2B Churn Dashboard (`atlas-churn-ui/`)
Vendor intelligence operations center. Pages: Dashboard (KPIs + pipeline health), Vendors (filterable list with urgency/archetype badges), Vendor Detail (churn trajectory, product profile, reviews, companies), Reviews (enriched search), Leads (high-intent companies), Prospects (outreach management), Reports (generated intelligence with quality status), Campaigns (review + diagnostics), Blog (auto-generated content library), Challengers (displacement analysis).

### Consumer Intelligence Dashboard (`atlas-intel-ui/`)
Product review analytics. Pages: Dashboard (pipeline + enrichment metrics), Brands (top brands, comparison), Reviews (deep enrichment panels), Flows (churn flow visualization), Safety (risk signals), plus B2B sub-routes for cross-system views.

### Admin Operations Dashboard (`atlas-admin-ui/`)
System monitoring. LLM cost tracking by provider/model with daily trends, system resource usage, scheduled task health + success rates, scraping pipeline status per source (success rate, block frequency, proxy usage), recent API calls with latency, reasoning activity (Reddit posts by signal type, per-subreddit breakdown), error timeline.

---

## Edge Nodes

Orange Pi RK3588 (or Jetson Nano) running `atlas_edge/`:

- **Voice** — Piper TTS (~6.6x realtime), SenseVoice STT (int8 ONNX, CPU)
- **Vision** — YOLO-World (NPU core 0), RetinaFace (NPU core 1), MobileFaceNet (NPU core 2), YOLOv8n-pose (NPU core 1 timeshared)
- **Motion gate** — MOG2 on CPU prevents NPU inference when no motion
- **Local skills** — Time, timer, math, status queries skip brain round-trip
- **Home Assistant** — Runs on the edge node via Docker, direct entity control
- **Offline fallback** — Functions independently when brain is unreachable
- **Brain connection** — Tailscale mesh, WebSocket with zlib compression, token batching

---

## Skills System

68 injectable markdown documents that enrich LLM system prompts at runtime. Organized by domain:

- **Digest** (46) — B2B campaign generation, churn extraction, intelligence synthesis, battle cards, vendor outreach, complaint analysis, morning briefing, email triage, security summary, and more
- **Email** (7) — Cleaning confirmation, reply, query, proposal, sentiment response, estimate
- **Call** (4) — Call extraction, action planning, confirmation email/SMS
- **Invoicing** (3) — Payment reminder, overdue notification, invoice email
- **Intelligence** (6) — Report building, narrative architecture, intervention, simulation
- **Security** (1) — Escalation narration
- **B2B** (1) — Product profile synthesis

Skills are lazy-loaded and referenced by domain/name (e.g., `email/draft`, `digest/b2b_churn_intelligence`).

---

## Memory & Knowledge Graph

Dual-layer memory system:

- **Semantic memory** — Neo4j knowledge graph via Graphiti wrapper. Entity relationships, facts, temporal awareness. Query decomposition, synonym expansion, cross-encoder reranking.
- **Episodic memory** — PostgreSQL conversation persistence. Session transcripts with metadata, memory quality detection (correction patterns, repetition via cosine similarity).
- **Quality signals** — 10 regex patterns for correction detection, embedding-based repetition detection (threshold 0.85). Stored in `metadata` JSONB on `conversation_turns`.
- **Nightly sync** — Conversation history → Neo4j knowledge graph with quality metadata.
- **RAG retrieval** — Unified `RAGClient` with `retrieve_memory()` returning structured `SearchSource[]`. Source tracking on both text and voice paths.

---

## Quick Start

### Requirements

- Python 3.11+
- NVIDIA GPU with 24GB+ VRAM (RTX 3090/4090)
- Docker and Docker Compose
- Ollama
- PostgreSQL + Neo4j

### Setup

```bash
git clone https://github.com/canfieldjuan/ATLAS.git
cd ATLAS

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # Configure credentials

ollama pull qwen3:14b
docker compose up -d postgres

# Dev server with hot reload
uvicorn atlas_brain.main:app \
  --host 0.0.0.0 --port 8001 \
  --reload --reload-dir atlas_brain \
  --reload-exclude 'data/postgres/**' \
  --ws-ping-interval 60 --ws-ping-timeout 120
```

### ASR Server (for voice pipeline)

```bash
pip install -r requirements.asr.txt
python asr_server.py --model nvidia/nemotron-speech-streaming-en-0.6b --port 8081 --device cuda:0
```

### Test

```bash
# Health
curl http://127.0.0.1:8001/api/v1/ping

# Text query
curl -X POST http://127.0.0.1:8001/api/v1/query/text \
  -H "Content-Type: application/json" \
  -d '{"query_text": "Hello Atlas"}'

# Device control
curl -X POST http://127.0.0.1:8001/api/v1/devices/intent \
  -H "Content-Type: application/json" \
  -d '{"query": "turn on the living room lights"}'

# Vision event feed
curl http://127.0.0.1:8001/api/v1/vision/events

# Voice pipeline (WebSocket)
# Connect to ws://127.0.0.1:8001/api/v1/ws/orchestrated
```

---

## API Reference

Below is a curated API overview for the main route families. For a generated mounted-route snapshot from importing `atlas_brain.main:app` in the current checkout, see [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md).

Most REST endpoints are served under `/api/v1/` (port 8001 by default). Root-level exceptions include OpenAI compatibility (`/v1/...`), Ollama compatibility (`/`, `/api/...`), and webhooks (`/webhooks/...`). Most authenticated endpoints use a JWT Bearer token obtained via `/api/v1/auth/login`. B2B-specific routes additionally require the `b2b_plan` claim. Exact mounted route counts can vary with optional routers and local build artifacts.

### Authentication

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/auth/register` | Register a new user; returns JWT access + refresh tokens |
| `POST` | `/api/v1/auth/login` | Password login; returns JWT access + refresh tokens |
| `POST` | `/api/v1/auth/refresh` | Exchange refresh token for a new access token |
| `GET`  | `/api/v1/auth/me` | Returns the authenticated user's profile |

### Billing (Stripe)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/billing/checkout` | Create a Stripe Checkout session |
| `POST` | `/api/v1/billing/portal` | Create a Stripe Customer Portal session |
| `GET`  | `/api/v1/billing/status` | Active subscription tier and feature flags |

### Core AI

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/v1/ping` | Liveness check |
| `GET`  | `/api/v1/health` | Detailed health with service status |
| `POST` | `/api/v1/query/text` | Text query → LLM response |
| `GET`  | `/api/v1/system/stats` | High-level system stats |
| `GET`  | `/api/v1/vision/events` | Vision event feed from atlas_vision nodes |
| `WS`   | `/api/v1/ws/orchestrated` | Full voice pipeline over WebSocket |

### LLM Management

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/v1/llm/available` | List registered LLM implementations and the active model |
| `POST` | `/api/v1/llm/activate` | Load a model into the active slot |
| `POST` | `/api/v1/llm/deactivate` | Unload the active model |
| `POST` | `/api/v1/llm/generate` | One-shot text generation from a prompt |
| `POST` | `/api/v1/llm/chat` | Multi-message chat request through the active LLM |
| `GET`  | `/api/v1/llm/status` | Active model, VRAM usage, health |

### Device Control

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/v1/devices/` | List all registered capability devices |
| `GET`  | `/api/v1/devices/{device_id}` | Device state and metadata |
| `POST` | `/api/v1/devices/{device_id}/action` | Execute a device action directly |
| `POST` | `/api/v1/devices/intent` | Natural language → device action dispatch |

### Session & Memory

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/session/create` | Create a new conversation session |
| `POST` | `/api/v1/session/continue` | Resume or create a user session on another terminal |
| `GET`  | `/api/v1/session/{session_id}` | Session detail with transcript |
| `GET`  | `/api/v1/session/{session_id}/history` | Paginated conversation history for a session |
| `POST` | `/api/v1/session/{session_id}/close` | Mark a session inactive |
| `GET`  | `/api/v1/session/status/db` | Database/session persistence health |

### Email

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/v1/email/drafts/` | List pending email drafts |
| `GET`  | `/api/v1/email/drafts/{draft_id}` | Draft detail |
| `POST` | `/api/v1/email/drafts/{draft_id}/approve` | Approve and send a draft |
| `POST` | `/api/v1/email/drafts/{draft_id}/reject` | Reject a draft |
| `POST` | `/api/v1/email/drafts/{draft_id}/edit` | Edit a generated draft |
| `POST` | `/api/v1/email/drafts/generate/{gmail_message_id}` | Generate a draft for a processed Gmail message |
| `POST` | `/api/v1/email/drafts/{draft_id}/redraft` | Regenerate a draft |
| `POST` | `/api/v1/email/drafts/{draft_id}/skip` | Skip a draft without sending |
| `POST` | `/api/v1/email/actions/{gmail_message_id}/quote` | Generate a quote reply draft |
| `POST` | `/api/v1/email/actions/{gmail_message_id}/escalate` | Escalate a complaint email |
| `POST` | `/api/v1/email/actions/{gmail_message_id}/slots` | Suggest appointment slots |
| `POST` | `/api/v1/email/actions/{gmail_message_id}/send-info` | Send an informational reply |
| `POST` | `/api/v1/email/actions/{gmail_message_id}/archive` | Archive a processed email |
| `GET`  | `/api/v1/email/inbox-rules/` | List inbox routing rules |
| `POST` | `/api/v1/email/inbox-rules/` | Create an inbox routing rule |
| `PUT`  | `/api/v1/email/inbox-rules/{rule_id}` | Update an inbox rule |
| `DELETE` | `/api/v1/email/inbox-rules/{rule_id}` | Delete an inbox rule |
| `POST` | `/api/v1/email/inbox-rules/reorder` | Reorder rule priority |
| `POST` | `/api/v1/email/inbox-rules/test` | Test a rule against sample email content |

### Communications (Calls, SMS, Actions & Webhooks)

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/v1/comms/status` | Communications provider status |
| `GET`  | `/api/v1/comms/contexts` | List registered business contexts |
| `GET`  | `/api/v1/comms/contexts/{context_id}` | Context detail and availability |
| `POST` | `/api/v1/comms/calls` | Initiate an outbound call |
| `POST` | `/api/v1/comms/sms` | Send an outbound SMS |
| `POST` | `/api/v1/comms/availability` | Check availability for a call/SMS workflow |
| `POST` | `/api/v1/comms/appointments` | Create an appointment from a comms workflow |
| `DELETE` | `/api/v1/comms/appointments/{appointment_id}` | Delete an appointment |
| `POST` | `/api/v1/comms/recordings/reconcile` | Reconcile provider recording state |
| `POST` | `/api/v1/comms/call-actions/{transcript_id}/book` | Turn a transcript into an appointment |
| `POST` | `/api/v1/comms/call-actions/{transcript_id}/draft-email` | Draft a follow-up email from a call |
| `POST` | `/api/v1/comms/call-actions/{transcript_id}/draft-sms` | Draft a follow-up SMS from a call |
| `POST` | `/api/v1/comms/call-actions/{transcript_id}/send-email` | Send the drafted email |
| `POST` | `/api/v1/comms/call-actions/{transcript_id}/send-sms` | Send the drafted SMS |
| `GET`  | `/api/v1/comms/calls/search` | Full-text search over call transcripts |
| `GET`  | `/api/v1/contacts/{contact_id}/timeline` | Contact interaction timeline |
| `POST` | `/api/v1/comms/voice/inbound` | Inbound telephony webhook |
| `POST` | `/api/v1/comms/voice/status` | Voice status callback webhook |
| `POST` | `/api/v1/comms/sms/inbound` | Inbound SMS webhook |
| `POST` | `/api/v1/comms/sms/status` | SMS delivery status webhook |
| `WS`   | `/api/v1/comms/voice/stream/{call_sid}` | Live telephony media stream |

### Invoicing

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/v1/invoicing/{invoice_id}` | Invoice detail |
| `POST` | `/api/v1/invoicing/{invoice_id}/send` | Email the invoice to the customer |
| `POST` | `/api/v1/invoicing/{invoice_id}/send-reminder` | Send a reminder for an unpaid invoice |
| `POST` | `/api/v1/invoicing/{invoice_id}/mark-paid` | Record payment / mark invoice paid |

### B2B Intelligence — Admin Dashboard (`/api/v1/b2b/dashboard/`)

Public / admin-facing endpoints — no tenant scope. Require `b2b_plan`.

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/signals` | Global churn signals across all vendors |
| `GET`  | `/slow-burn-watchlist` | Low-urgency vendors with accumulating risk |
| `GET`  | `/signals/{vendor_name}` | Full vendor signal detail |
| `GET`  | `/high-intent` | Companies showing high purchase intent |
| `GET`  | `/vendors/{vendor_name}` | Vendor profile with churn metrics |
| `POST` | `/vendors/{vendor_name}/reason` | Trigger Claude reasoning synthesis for a vendor |
| `POST` | `/vendors/compare-reasoning` | Side-by-side reasoning comparison |
| `GET`  | `/reports` | List generated intelligence reports |
| `GET`  | `/reports/{report_id}` | Report detail |
| `GET`  | `/reports/{report_id}/pdf` | Download report as PDF |
| `POST` | `/reports/compare` | Generate a vendor comparison report |
| `POST` | `/reports/compare-companies` | Account-level comparison report |
| `POST` | `/reports/company-deep-dive` | Single-company deep dive report |
| `GET`  | `/reviews` | Enriched review list with filters |
| `GET`  | `/reviews/{review_id}` | Single review with full enrichment |
| `GET`  | `/pipeline` | Pipeline health: queue depths, error rates, lag |
| `GET`  | `/source-health` | Per-source scraping health (success rate, block rate) |
| `GET`  | `/source-health/telemetry` | Source telemetry metrics |
| `GET`  | `/displacement-edges` | Competitive displacement graph edges |
| `GET`  | `/displacement-history` | Displacement edge velocity over time |
| `GET`  | `/vendor-pain-points` | Aggregated pain categories per vendor |
| `GET`  | `/vendor-use-cases` | Use case coverage per vendor |
| `GET`  | `/vendor-integrations` | Integration graph per vendor |
| `GET`  | `/vendor-buyer-profiles` | Buyer profile segments per vendor |
| `GET`  | `/product-profile` | Vendor product profile knowledge card |
| `GET`  | `/change-events` | Contract/leadership/funding change events |
| `GET`  | `/fuzzy-vendor-search` | Fuzzy vendor name search |
| `GET`  | `/fuzzy-company-search` | Fuzzy company name search |
| `GET`  | `/compare-vendor-periods` | Metric delta between two time windows |
| `GET`  | `/signal-effectiveness` | Per-signal type performance stats |

### B2B Intelligence — Tenant Dashboard (`/api/v1/b2b/tenant/`)

All endpoints are tenant-scoped to the authenticated user's tracked vendors. Require `b2b_plan`.

**Vendor Management**

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/vendors` | List this tenant's tracked vendors |
| `POST` | `/vendors` | Add a vendor to the tenant's tracking list |
| `DELETE` | `/vendors/{vendor_name}` | Remove a vendor from tracking |
| `GET`  | `/vendors/search` | Search the global vendor registry |
| `POST` | `/push-to-crm` | Push a high-intent company to the connected CRM |

**Competitive Sets**

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/competitive-sets` | List tenant competitive sets |
| `POST` | `/competitive-sets` | Create a competitive set |
| `PUT`  | `/competitive-sets/{id}` | Update a competitive set |
| `DELETE` | `/competitive-sets/{id}` | Delete a competitive set |
| `GET`  | `/competitive-sets/{id}/plan` | Preview synthesis plan before running |
| `POST` | `/competitive-sets/{id}/run` | Trigger on-demand intelligence synthesis |

**Intelligence Feeds**

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/overview` | KPI summary: signal counts, risk levels, archetype breakdown |
| `GET`  | `/signals` | Tenant-scoped churn signals with filters |
| `GET`  | `/slow-burn-watchlist` | Slow-burn risk vendors with accumulating pressure |
| `GET`  | `/signals/{vendor_name}` | Full vendor detail scoped to tenant |
| `GET`  | `/accounts-in-motion-feed` | At-risk accounts with decision-maker signals |
| `GET`  | `/vendor-history` | Metric time-series for a tracked vendor |
| `GET`  | `/compare-vendor-periods` | Metric delta between two time windows |
| `GET`  | `/pain-trends` | Pain category trend lines over time |
| `GET`  | `/displacement` | Competitive displacement edges for tracked vendors |
| `GET`  | `/pipeline` | Tenant pipeline status: enrichment lag, queue depth |
| `GET`  | `/leads` | High-intent companies scoped to tenant vendors |
| `GET`  | `/high-intent` | Compact high-intent list for quick scan |
| `GET`  | `/leads/{company}` | Detailed lead profile for a single company |

**Watchlist Views & Alerts**

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/watchlist-views` | List saved watchlist filter configurations |
| `POST` | `/watchlist-views` | Create a watchlist view |
| `PUT`  | `/watchlist-views/{view_id}` | Update view filters or delivery settings |
| `DELETE` | `/watchlist-views/{view_id}` | Delete a watchlist view |
| `GET`  | `/watchlist-views/{view_id}/alert-events` | Alert trigger history for a view |
| `POST` | `/watchlist-views/{view_id}/alert-events/evaluate` | Run alert rules now |
| `GET`  | `/watchlist-views/{view_id}/alert-email-log` | Email delivery log |
| `POST` | `/watchlist-views/{view_id}/alert-events/deliver-email` | Send alert email immediately |

**Reports**

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/reports` | List tenant intelligence reports with trust metadata |
| `GET`  | `/reports/{report_id}` | Report detail with freshness and review state |
| `POST` | `/reports/compare` | Generate a vendor comparison report |
| `POST` | `/reports/compare-companies` | Account-level comparison report |
| `POST` | `/reports/company-deep-dive` | Single-company deep dive |
| `POST` | `/reports/battle-card` | Competitive battle card |
| `GET`  | `/report-subscriptions/{scope_type}/{scope_key}` | Get recurring delivery subscription |
| `PUT`  | `/report-subscriptions/{scope_type}/{scope_key}` | Create or update recurring delivery |

**Reviews, Campaigns & Exports**

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/reviews` | Tenant-scoped enriched reviews |
| `GET`  | `/reviews/{review_id}` | Single review detail |
| `GET`  | `/campaigns` | Outreach campaigns scoped to tenant |
| `POST` | `/campaigns/generate` | Trigger campaign generation now |
| `PATCH` | `/campaigns/{campaign_id}` | Update campaign status or content |
| `GET`  | `/opportunity-dispositions` | Manual disposition overrides |
| `POST` | `/opportunity-dispositions` | Set a disposition for an opportunity |
| `POST` | `/opportunity-dispositions/bulk` | Bulk set dispositions |
| `POST` | `/opportunity-dispositions/remove` | Remove disposition overrides |
| `GET`  | `/export/signals` | CSV export of signals |
| `GET`  | `/export/reviews` | CSV export of reviews |
| `GET`  | `/export/high-intent` | CSV export of high-intent companies |
| `GET`  | `/export/source-health` | CSV export of source health metrics |

### B2B Evidence Explorer (`/api/v1/b2b/evidence/`)

Trust layer exposing raw evidence behind every signal, report, and account card. Requires `b2b_plan`.

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/witnesses` | List witness records for a vendor (filters: urgency, pain category, buying stage, date) |
| `GET`  | `/witnesses/{id}` | Single witness with full review context and company signals |
| `GET`  | `/vault` | Evidence vault claims (weakness/strength) with citation counts |
| `GET`  | `/trace` | Full claim-to-review reasoning trace (sample of 20 witnesses) |
| `GET`  | `/annotations` | Human annotation overrides on witness records |
| `POST` | `/annotations` | Add or update a witness annotation |
| `POST` | `/annotations/remove` | Remove an annotation |

### B2B Win/Loss Predictor (`/api/v1/b2b/predict/`)

Aggregates displacement, churn, pain, and buyer data into a probability score with evidence. Requires `b2b_plan`.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/win-loss` | Predict win probability for a target vendor vs. a challenger |
| `POST` | `/win-loss/compare` | Head-to-head comparison across multiple challengers |
| `GET`  | `/win-loss/recent` | Recent predictions with outcome labels |
| `GET`  | `/win-loss/{prediction_id}` | Prediction detail with factor breakdown |
| `GET`  | `/win-loss/{prediction_id}/csv` | Export prediction as CSV |

Scoring factors: `displacement_momentum` (25%), `churn_severity` (20%), `pain_concentration` (15%), `dm_engagement` (15%), `historical_outcomes` (15%), `segment_match` (10%). Vendors with insufficient data return `is_gated=true` instead of a fake probability.

### B2B Campaigns (`/api/v1/b2b/campaigns/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | List campaigns with status and quality filters |
| `GET`  | `/stats` | Aggregate campaign KPIs |
| `GET`  | `/quality-trends` | Quality score trends over time |
| `GET`  | `/analytics/funnel` | Full outreach funnel metrics |
| `GET`  | `/analytics/by-vendor` | Per-vendor campaign analytics |
| `GET`  | `/suppressions` | Email suppression list |
| `POST` | `/suppressions` | Add a suppression entry |
| `DELETE` | `/suppressions/{id}` | Remove a suppression entry |
| `GET`  | `/review-queue` | Campaigns awaiting human review |

### B2B Scrape Management (`/api/v1/b2b/scrape/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/targets` | List active scrape targets |
| `GET`  | `/targets/probation-telemetry` | Low-yield target diagnostics |
| `POST` | `/targets/onboard-vendor` | Provision scrape targets for a new vendor |
| `GET`  | `/targets/coverage-plan` | Coverage gap analysis |
| `POST` | `/targets/coverage-plan/seed-missing-core` | Auto-provision missing core-source targets |

### B2B CRM Events (`/api/v1/b2b/crm/`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/events` | Ingest a single CRM event (deal_won, deal_lost, meeting_booked) |
| `POST` | `/events/batch` | Batch ingest CRM events |
| `POST` | `/events/hubspot` | HubSpot webhook receiver |
| `POST` | `/events/salesforce` | Salesforce webhook receiver |

### B2B Vendor Briefings (`/api/v1/b2b/briefings/`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/preview` | Preview briefing content before sending |
| `POST` | `/generate` | Generate and send a vendor intelligence briefing email |
| `POST` | `/gate` | Check data readiness before briefing generation |
| `POST` | `/checkout` | Schedule a recurring briefing delivery |

### B2B Prospects (`/api/v1/b2b/prospects/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | List prospects with pipeline stage filters |
| `GET`  | `/stats` | Prospect pipeline stats |
| `GET`  | `/manual-queue` | Manually queued prospects awaiting review |
| `GET`  | `/company-overrides` | Manual targeting overrides |

### Consumer Product Intelligence (`/api/v1/consumer/dashboard/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/asins` | List tracked Amazon ASINs |
| `POST` | `/asins` | Add an ASIN to tracking |
| `DELETE` | `/asins/{asin}` | Remove an ASIN from tracking |
| `GET`  | `/pipeline` | Enrichment pipeline health |
| `GET`  | `/categories` | Product category list with review counts |
| `GET`  | `/brands` | Brand list with health scores |
| `GET`  | `/brands/compare` | Side-by-side brand comparison |
| `GET`  | `/brands/{brand_name}` | Brand detail: health score, pain map, safety signals |
| `GET`  | `/flows` | Churn flow visualization data |
| `GET`  | `/features` | Feature mention matrix |
| `GET`  | `/safety` | Safety risk signals |
| `GET`  | `/reviews` | Enriched product reviews with filters |
| `GET`  | `/reviews/{review_id}` | Review detail with full enrichment |
| `GET`  | `/brand-history` | Brand health score time-series |

### Strategic Intelligence (`/api/v1/intelligence/`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/report` | Generate an intelligence report |
| `POST` | `/intervention` | Trigger an intervention pipeline |
| `GET`  | `/reports` | List intelligence reports |
| `GET`  | `/reports/{report_id}` | Report detail |
| `GET`  | `/pressure` | Entity pressure baseline records |
| `GET`  | `/approvals` | Pending approval queue |
| `GET`  | `/approvals/{approval_id}` | Approval detail |
| `POST` | `/approvals/{approval_id}/approve` | Approve a pending action |
| `POST` | `/approvals/{approval_id}/reject` | Reject a pending action |

### Identity, Presence, Speaker ID & Recognition

| Route family | Description |
|-------------|-------------|
| `/api/v1/identity/*` | Canonical identity records, names, creation, and modality cleanup |
| `/api/v1/presence/*` | Occupancy status and transition history |
| `/api/v1/speaker/*` | Speaker enrollment, verification, and enrolled-user management |
| `/api/v1/recognition/*` | Face/gait enrollment, identification, and recognition-event history |

### Alerts, Vision, Video, Security & Settings

| Route family | Description |
|-------------|-------------|
| `/api/v1/alerts/*` | Unified cross-source alert history, rules, acknowledgements, and cleanup |
| `/api/v1/vision/*` | Vision events, cameras, alert rules, and alert history |
| `/api/v1/video/*` | Webcam / RTSP streams, snapshots, and recognition views |
| `/api/v1/security/*` | Security asset telemetry, observations, and threat summaries |
| `/api/v1/settings/*` | Voice, email, daily, intelligence, LLM, notifications, and integrations settings |
| `/api/v1/system/*` | High-level system stats |

### Additional Business Route Families

| Route family | Description |
|-------------|-------------|
| `/api/v1/b2b/tenant/affiliates/*` | Affiliate opportunities, partner registry, and click tracking |
| `/api/v1/b2b/vendor-targets/*` | Vendor target CRUD, claiming, and report generation |
| `/api/v1/seller/*` | Seller-side targets, campaigns, and intelligence refresh |
| `/api/v1/scraper/*` | Universal scrape job creation, status, results, cancel, and delete |
| `/api/v1/actions/*` | Proactive action queue acknowledgement/dismiss flows |

### Blog (`/api/v1/blog/`, `/api/v1/admin/blog/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/blog/published` | List published posts (public, no auth) |
| `GET`  | `/blog/published/{slug}` | Single published post (public) |
| `GET`  | `/admin/blog/drafts` | List blog drafts |
| `GET`  | `/admin/blog/drafts/{id}` | Draft detail |
| `PATCH` | `/admin/blog/drafts/{id}` | Edit draft content or metadata |
| `POST` | `/admin/blog/drafts/{id}/publish` | Publish a draft |
| `GET`  | `/admin/blog/quality-trends` | Quality score trends |

### Pipeline Visibility (`/api/v1/pipeline/visibility/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/extraction-health` | Enrichment extraction success rates |
| `GET`  | `/summary` | High-level pipeline summary |
| `GET`  | `/watchlist-delivery` | Watchlist alert delivery health |
| `GET`  | `/queue` | Enrichment queue state |
| `GET`  | `/events` | Recent pipeline events |

### Admin & Cost Tracking (`/api/v1/admin/costs/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/summary` | LLM cost summary (all providers) |
| `GET`  | `/burn-dashboard` | Daily spend burn rate |
| `GET`  | `/by-provider` | Cost breakdown by LLM provider |
| `GET`  | `/by-model` | Cost breakdown by model |
| `GET`  | `/by-workflow` | Cost breakdown by workflow |
| `GET`  | `/reconciliation` | Cost reconciliation against usage data |

### Autonomous Tasks (`/api/v1/autonomous/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/status/summary` | Scheduler / hook-manager status summary |
| `GET`  | `/` | List all scheduled tasks |
| `POST` | `/` | Create a scheduled task |
| `GET`  | `/{task_id}` | Task detail |
| `PUT`  | `/{task_id}` | Update a task |
| `DELETE` | `/{task_id}` | Delete a task |
| `POST` | `/{task_id}/run` | Trigger a task immediately |
| `POST` | `/{task_id}/enable` | Enable and schedule a task |
| `POST` | `/{task_id}/disable` | Disable and unschedule a task |
| `GET`  | `/{task_id}/executions` | Task execution history |

### Reasoning Events (`/api/v1/reasoning/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/events` | Pending reasoning events |
| `GET`  | `/locks` | Active reasoning locks |
| `GET`  | `/queue` | Reasoning work queue |
| `POST` | `/process/{event_id}` | Manually process a reasoning event |

### WebSocket Endpoints

| Path | Description |
|------|-------------|
| `WS /api/v1/ws/edge/{location_id}` | Edge node connection and state sync |
| `WS /api/v1/ws/orchestrated` | Orchestrated full voice pipeline: audio in → ASR → agent → TTS |
| `WS /api/v1/comms/voice/stream/{call_sid}` | Live telephony audio stream for calls |

### OpenAI / Ollama Compatibility

Atlas exposes drop-in compatibility endpoints so any OpenAI SDK or Ollama client can point at the Atlas brain server:

| Path | Compatibility |
|------|---------------|
| `POST /v1/chat/completions` | OpenAI Chat Completions API |
| `GET /` | Ollama root health check (`Ollama is running`) |
| `GET /api/version` | Ollama version endpoint |
| `POST /api/chat` | Ollama `/api/chat` |
| `GET /api/tags` | Ollama `/api/tags` |

### Webhooks

| Path | Description |
|------|-------------|
| `GET /webhooks/unsubscribe` | One-click campaign unsubscribe page |
| `POST /webhooks/campaign-email` | Campaign email ESP event webhook receiver |
| `POST /webhooks/stripe` | Stripe webhook receiver (subscription lifecycle) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.11, FastAPI, asyncpg, APScheduler |
| **Database** | PostgreSQL (50+ tables), Neo4j (GraphRAG) |
| **LLM** | Ollama, vLLM, Claude (Anthropic), OpenRouter |
| **ASR** | Nemotron 0.6B (brain), SenseVoice ONNX (edge) |
| **TTS** | Piper (edge), Kokoro (brain) |
| **Embeddings** | all-MiniLM-L6-v2 (384-dim), mxbai-embed-large-v1 (1024-dim) |
| **Workflows** | LangGraph (12 state machines) |
| **MCP** | FastMCP (8 servers, 130+ tools) |
| **Home Automation** | Home Assistant (WebSocket), MQTT |
| **Telephony** | Twilio, SignalWire |
| **Email** | Gmail (OAuth2), IMAP, Resend |
| **Calendar** | Google Calendar, CalDAV |
| **CRM** | Direct asyncpg, NocoDB admin UI |
| **Billing** | Stripe (subscription lifecycle) |
| **Vision** | YOLOv8, YOLO-World, RetinaFace, MobileFaceNet |
| **Edge** | Orange Pi RK3588 (NPU), Jetson Nano |
| **Frontends** | React 19, Next.js, Vite, Recharts, Tailwind |
| **Networking** | Tailscale mesh, zlib WebSocket compression |

---

## License

Private repository.
