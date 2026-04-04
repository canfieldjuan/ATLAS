# Atlas

A multi-modal AI platform that combines personal automation, voice control, B2B sales intelligence, and consumer product analytics into a single extensible system. Atlas runs as a central "Brain" server backed by local LLMs, with edge nodes for distributed sensing, 4 web dashboards, 8 MCP servers (130+ tools), 57 autonomous scheduled tasks, and a full telephony stack.

---

## System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          ATLAS BRAIN                                  │
│                   (FastAPI · 55+ API routes)                         │
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
atlas_brain/                    # Core server (FastAPI, 55+ routes)
├── api/                        # REST + WebSocket endpoints
│   ├── query/                  #   Text, audio, vision inference
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

# Vision
curl -X POST http://127.0.0.1:8001/api/v1/query/vision \
  -F "image_file=@image.jpg" -F "prompt_text=What is this?"

# Audio transcription
curl -X POST http://127.0.0.1:8001/api/v1/query/audio \
  -F "audio_file=@audio.wav"
```

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
