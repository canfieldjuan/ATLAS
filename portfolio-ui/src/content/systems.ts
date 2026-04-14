export interface Subsystem {
  name: string;
  description: string;
  stats?: { label: string; value: string }[];
  components: string[];
}

export interface IntegrationBridge {
  from: string;
  to: string;
  description: string;
}

export interface SystemDomain {
  id: string;
  title: string;
  subtitle: string;
  icon: string;
  color: string;
  colorBg: string;
  colorBorder: string;
  description: string;
  subsystems: Subsystem[];
  bridges: IntegrationBridge[];
}

export const systemDomains: SystemDomain[] = [
  {
    id: "conversational",
    title: "Conversational Intelligence",
    subtitle: "Voice, language, and real-time orchestration",
    icon: "Mic",
    color: "text-primary-400",
    colorBg: "bg-primary-500/10",
    colorBorder: "border-primary-500/30",
    description:
      "The real-time layer that handles voice input, natural language understanding, intent routing, tool execution, and response generation. Distributed across a GPU server (brain) and ARM edge nodes connected via Tailscale.",
    subsystems: [
      {
        name: "Voice Pipeline",
        description:
          "Wake word activation, streaming STT (Nemotron 0.6B / SenseVoice), semantic intent routing with all-MiniLM-L6-v2 centroids, tool execution loop, TTS response (Piper on edge). Sub-second for local skills, 3-5s for LLM reasoning.",
        stats: [
          { label: "Edge STT", value: "SenseVoice int8" },
          { label: "Brain STT", value: "Nemotron 0.6B" },
          { label: "TTS Speed", value: "6.6x realtime" },
        ],
        components: [
          "atlas_brain/voice/ (13 files)",
          "atlas_edge/voice/ (3 files)",
          "atlas_edge/intent/ (4 files)",
          "atlas_edge/skills/ (5 local skills)",
        ],
      },
      {
        name: "LLM Services",
        description:
          "9 provider adapters behind a unified registry with runtime hot-swap. Local inference (Ollama, vLLM, Llama.cpp) for cost-sensitive workloads, cloud relay (Anthropic, OpenRouter, Google Cloud, Groq, Together) for heavy reasoning. LLM router decides per-query.",
        stats: [
          { label: "Providers", value: "9" },
          { label: "Local Models", value: "3 adapters" },
          { label: "Cloud Models", value: "6 adapters" },
        ],
        components: [
          "atlas_brain/services/llm/ (11 files)",
          "atlas_brain/services/llm_router.py",
          "atlas_brain/services/registry.py",
        ],
      },
      {
        name: "Edge Nodes",
        description:
          "Orange Pi RK3588 running STT, TTS, and computer vision locally. NPU core allocation (3 cores, 5 models), motion gating via MOG2, brain connection over Tailscale WebSocket with zlib compression and token batching.",
        stats: [
          { label: "NPU Models", value: "5 concurrent" },
          { label: "RAM Usage", value: "~1.7 GB" },
          { label: "Hardware Cost", value: "$60" },
        ],
        components: [
          "atlas_edge/ (26 files)",
          "Brain-edge protocol: WebSocket + zlib",
          "NPU: YOLO-World, RetinaFace, MobileFaceNet, YOLOv8n-pose",
        ],
      },
      {
        name: "Skills System",
        description:
          "Injectable markdown documents that enrich LLM system prompts at runtime. Domain-scoped (email, digest, intelligence) so the LLM gets context-appropriate instructions without prompt bloat.",
        components: [
          "atlas_brain/skills/registry.py",
          "Skill files: email (5), digest (7), intelligence (6)",
        ],
      },
    ],
    bridges: [
      {
        from: "Voice Pipeline",
        to: "LLM Services",
        description:
          "Intent router classifies query → routes to streaming LLM (conversational) or tool-calling LLM (action). Streaming path uses chat_stream_async without tools; tool-needing queries route through agent fallback.",
      },
      {
        from: "Edge Nodes",
        to: "Voice Pipeline",
        description:
          "Audio captured on edge → STT runs locally → text sent to brain over Tailscale WebSocket → brain returns LLM response → edge synthesizes TTS locally. Local skills (time, timer, math) skip brain entirely.",
      },
    ],
  },
  {
    id: "refinery",
    title: "Data Refinery",
    subtitle: "Industrial-scale intelligence from unstructured data",
    icon: "Factory",
    color: "text-accent-cyan",
    colorBg: "bg-cyan-500/10",
    colorBorder: "border-cyan-500/30",
    description:
      "The batch processing layer that transforms raw review data from 19 sources into structured intelligence — enrichment, repair, evidence derivation, vendor reasoning, displacement graphs, and downstream artifact generation.",
    subsystems: [
      {
        name: "Scraping Infrastructure",
        description:
          "19 source-specific parsers with a universal engine. Priority-ordered scheduling, per-source semaphores, 3-tier G2 fallback (Web Unlocker → Playwright → residential), CAPTCHA detection and solve tracking, proxy rotation, rate limiting, SERP discovery for target provisioning.",
        stats: [
          { label: "Parsers", value: "19" },
          { label: "Infrastructure", value: "38 files" },
          { label: "Sources Active", value: "15" },
        ],
        components: [
          "atlas_brain/services/scraping/ (38 files)",
          "Parsers: G2, Capterra, Gartner, GetApp, GitHub, HackerNews, PeerSpot, ProductHunt, Quora, Reddit, RSS, Slashdot, SoftwareAdvice, SourceForge, StackOverflow, TrustPilot, TrustRadius, Twitter, YouTube",
          "Engine: orchestrator, html_cleaner, captcha, proxy, rate_limiter",
          "Discovery: SERP, target planning/provisioning/validation",
        ],
      },
      {
        name: "Enrichment Pipeline",
        description:
          "7-stage pipeline: ingest → enrichment (LLM extraction of 10+ fields) → repair (contradiction detection, weak extraction re-queue) → evidence derivation → vendor reasoning → cross-vendor synthesis → artifact generation. Field ownership contracts enforce canonical values and prevent drift.",
        stats: [
          { label: "Extraction Fields", value: "10+" },
          { label: "Pipeline Stages", value: "7" },
          { label: "Reviews Enriched", value: "25K+" },
        ],
        components: [
          "atlas_brain/autonomous/tasks/b2b_enrichment.py",
          "atlas_brain/autonomous/tasks/b2b_enrichment_repair.py",
          "atlas_brain/autonomous/tasks/b2b_reasoning_synthesis.py",
          "atlas_brain/autonomous/tasks/_b2b_shared.py (field contracts)",
          "atlas_brain/services/b2b/ (16 files)",
        ],
      },
      {
        name: "Intelligence Synthesis",
        description:
          "Churn signals, product profiles, displacement edges, pain point rankings, use case maps, buyer profiles, vendor correlation, concurrent event detection. Witness system fact-checks evidence spans and salience before reporting. 8 churn archetypes classify patterns.",
        stats: [
          { label: "Vendors Tracked", value: "56" },
          { label: "Churn Archetypes", value: "8" },
          { label: "Witness Records", value: "3,271" },
        ],
        components: [
          "atlas_brain/autonomous/tasks/b2b_churn_intelligence.py",
          "atlas_brain/autonomous/tasks/b2b_product_profiles.py",
          "atlas_brain/autonomous/tasks/b2b_battle_cards.py",
          "atlas_brain/autonomous/tasks/b2b_vendor_briefing.py",
          "atlas_brain/reasoning/ (38 files)",
        ],
      },
      {
        name: "Consumer Intelligence",
        description:
          "Amazon review pipeline: complaint enrichment, brand health scoring, migration flow analysis, safety signal tracking. Separate from B2B but shares the pipeline framework (registry, shared LLM, notify).",
        components: [
          "atlas_brain/autonomous/tasks/complaint_*.py",
          "atlas_brain/autonomous/tasks/consumer_*.py",
          "atlas_brain/autonomous/tasks/deep_enrichment.py",
          "atlas_brain/pipelines/ (4 modules)",
        ],
      },
    ],
    bridges: [
      {
        from: "Scraping Infrastructure",
        to: "Enrichment Pipeline",
        description:
          "Scrape intake writes raw reviews to b2b_reviews with parser_version tracking. Enrichment picks up pending reviews, runs LLM extraction, writes structured fields. Parser version upgrades auto-trigger re-enrichment.",
      },
      {
        from: "Intelligence Synthesis",
        to: "Business Automation",
        description:
          "Churn signals feed campaign generation → email sequences → CRM webhook outcomes → score calibration loop. Battle cards and blog posts generate from synthesized intelligence.",
      },
    ],
  },
  {
    id: "operations",
    title: "Operations Platform",
    subtitle: "Autonomous orchestration, memory, and observability",
    icon: "Settings",
    color: "text-accent-purple",
    colorBg: "bg-purple-500/10",
    colorBorder: "border-purple-500/30",
    description:
      "The nervous system: 51 scheduled tasks, 9 MCP servers exposing 190 tools, persistent memory across PostgreSQL and Neo4j, multi-step reasoning, and full-system observability with cost tracking.",
    subsystems: [
      {
        name: "Autonomous Task System",
        description:
          "51 scheduled tasks (cron + interval + event-triggered), 99 total handlers. Runner with fail-open patterns, skip-synthesis conventions, LLM synthesis with domain skills, ntfy notification delivery. Per-task opt-out, priority control, and pre-warm for cold LLM starts.",
        stats: [
          { label: "Scheduled Tasks", value: "51" },
          { label: "Total Handlers", value: "99" },
          { label: "Skip-synthesis", value: "saves 10-20s/task" },
        ],
        components: [
          "atlas_brain/autonomous/scheduler.py",
          "atlas_brain/autonomous/runner.py",
          "atlas_brain/autonomous/tasks/ (48+ task files)",
        ],
      },
      {
        name: "MCP Server Mesh",
        description:
          "9 MCP servers exposing 190 tools via stdio (Claude Desktop/Cursor) or SSE (HTTP). B2B Churn alone has 83 tools across 17 modules. MCP client spawns servers as stdio subprocesses, discovers tools dynamically, registers in tool_registry with collision protection.",
        stats: [
          { label: "Servers", value: "9" },
          { label: "Total Tools", value: "190" },
          { label: "B2B Churn Tools", value: "83" },
        ],
        components: [
          "atlas_brain/mcp/ (9 servers)",
          "atlas_brain/services/mcp_client.py",
          "Servers: CRM (10), Email (9), Twilio (10), Calendar (8), Invoicing (17), Intelligence (33), B2B Churn (83), Memory (15), Scraper (5)",
        ],
      },
      {
        name: "Memory & Knowledge Graph",
        description:
          "Dual persistence: PostgreSQL for conversations, turns, and metadata (JSONB quality signals). Neo4j via Graphiti wrapper for long-term knowledge graph with entity relationships. RAG client unifies retrieval with semantic search, temporal queries, and source tracking.",
        stats: [
          { label: "DB Migrations", value: "300" },
          { label: "Memory Tools", value: "15" },
        ],
        components: [
          "atlas_brain/memory/ (RAG client, quality detector)",
          "atlas_brain/storage/ (repositories, migrations)",
          "graphiti-wrapper/ (14 files)",
          "atlas_brain/mcp/memory_server.py",
        ],
      },
      {
        name: "Reasoning Engine",
        description:
          "Multi-step reasoning with stratified depth (L1 Aggregation → L5 Ecosystem). 3 cognitive modes: Recall (cached), Reconstitute (patch), Reason (full LLM). Semantic cache in Postgres, episodic traces in Neo4j. 8 churn archetypes for pattern classification.",
        stats: [
          { label: "Reasoning Files", value: "38" },
          { label: "Depth Levels", value: "5" },
          { label: "Cognitive Modes", value: "3" },
        ],
        components: [
          "atlas_brain/reasoning/ (38 files)",
          "Stratified: L1-L5 depth hierarchy",
          "Dual memory: Postgres semantic + Neo4j episodic",
        ],
      },
      {
        name: "Telemetry & Cost Tracking",
        description:
          "Token usage tracking across all 9 LLM providers. Per-call cost computation, provider spend aggregation, daily charts. Scrape telemetry: success rates, block types, CAPTCHA solve times, proxy usage. FTL tracing on pipeline spans.",
        components: [
          "atlas_brain/services/provider_cost_sync.py",
          "atlas_brain/services/tracing.py",
          "atlas-admin-ui/ (cost dashboard, provider analytics)",
          "atlas_brain/autonomous/tasks/llm_provider_cost_sync.py",
        ],
      },
    ],
    bridges: [
      {
        from: "MCP Server Mesh",
        to: "Conversational Intelligence",
        description:
          "MCP client discovers tools at startup. Voice/text queries that need CRM, calendar, email, or telephony route through MCP tools via execute_with_tools(). Tool results feed back into LLM response generation.",
      },
      {
        from: "Autonomous Task System",
        to: "Data Refinery",
        description:
          "Scheduled tasks orchestrate every refinery stage: scrape intake, enrichment, repair, reasoning synthesis, blog generation, campaign generation. Each task is a thin handler that delegates to pipeline logic.",
      },
    ],
  },
  {
    id: "physical",
    title: "Physical World",
    subtitle: "Devices, cameras, presence, and security",
    icon: "Eye",
    color: "text-amber-400",
    colorBg: "bg-amber-500/10",
    colorBorder: "border-amber-500/30",
    description:
      "The physical layer: IoT device control via Home Assistant and MQTT, video processing with face/gait/pose recognition, presence detection, security event escalation, and a React Native mobile app.",
    subsystems: [
      {
        name: "Device Control",
        description:
          "Capability-based device system with Home Assistant (WebSocket real-time state) and MQTT backends. Intent parser extracts structured actions from natural language. Device resolver maps names to entities. State cache prevents redundant commands.",
        stats: [
          { label: "Device Types", value: "3 (lights, media, switches)" },
          { label: "Backends", value: "2 (HA, MQTT)" },
        ],
        components: [
          "atlas_brain/capabilities/ (17 files)",
          "Backends: homeassistant_ws.py, mqtt.py",
          "Devices: lights.py, media.py, switches.py",
          "Actions: intent_parser → action_dispatcher → device",
        ],
      },
      {
        name: "Video Processing",
        description:
          "Complete surveillance/intelligence pipeline. Motion detection (MOG2) gates expensive inference. Face detection (RetinaFace) → face recognition (MobileFaceNet) → gait analysis. Object detection (YOLO-World). Pose estimation (YOLOv8n-pose). RTSP camera support, frame buffering, MQTT announcements.",
        stats: [
          { label: "Pipeline Files", value: "53" },
          { label: "Detection Models", value: "5" },
          { label: "NPU Inference", value: "94ms" },
        ],
        components: [
          "atlas_video-processing/ (53 files)",
          "Detection: face, motion, pose, YOLO",
          "Recognition: face embeddings, gait patterns",
          "Cameras: RTSP, webcam, mock",
          "Presence: camera + ESPresense fusion",
        ],
      },
      {
        name: "Security System",
        description:
          "Security event detection, classification, and escalation. Presence-based automation (departure triggers). Person tracking across cameras. Alert routing to ntfy, SMS, or voice announcement.",
        stats: [{ label: "Security Files", value: "15" }],
        components: [
          "atlas_brain/security/ (15 files)",
          "atlas_brain/alerts/ (5 files)",
          "atlas_brain/escalation/",
          "Event-triggered tasks: departure_auto_fix",
        ],
      },
      {
        name: "Mobile App",
        description:
          "React Native app (Expo) for iOS and Android. Zustand state management, NativeWind styling (Tailwind for RN), AsyncStorage persistence, Expo Router navigation.",
        components: [
          "atlas-mobile/ (Expo 54 + React Native 0.81)",
          "State: Zustand + AsyncStorage",
          "Styling: NativeWind (Tailwind CSS)",
        ],
      },
    ],
    bridges: [
      {
        from: "Video Processing",
        to: "Security System",
        description:
          "Face detected → recognition attempted → known/unknown classification. Unknown face + motion + time-of-day rules trigger security escalation. Known face updates presence state.",
      },
      {
        from: "Device Control",
        to: "Conversational Intelligence",
        description:
          "\"Hey Atlas, turn off the TV\" → intent parser extracts action + device → action dispatcher routes to Home Assistant WebSocket → state cache updated → TTS confirms. Local on edge for common commands.",
      },
    ],
  },
  {
    id: "business",
    title: "Business Automation",
    subtitle: "CRM, communications, invoicing, and campaigns",
    icon: "Briefcase",
    color: "text-rose-400",
    colorBg: "bg-rose-500/10",
    colorBorder: "border-rose-500/30",
    description:
      "The revenue and operations layer: contact management, email (send + read), telephony (calls + SMS), calendar scheduling, automated invoicing, and campaign generation with outcome-driven calibration.",
    subsystems: [
      {
        name: "CRM & Contacts",
        description:
          "DatabaseCRMProvider queries PostgreSQL directly via asyncpg. Contact search, interaction logging, appointment management. NocoDB provides a browser-based admin UI over the same tables. Apollo.io integration for firmographic enrichment.",
        stats: [
          { label: "CRM Tools", value: "10" },
        ],
        components: [
          "atlas_brain/services/crm_provider.py",
          "atlas_brain/mcp/crm_server.py (10 tools)",
          "NocoDB: http://localhost:8090",
          "Apollo: firmographic backfill",
        ],
      },
      {
        name: "Email & Communications",
        description:
          "Provider-agnostic email: Gmail (OAuth2) preferred, Resend fallback for sending. IMAP for reading (works with any mail server). Twilio + SignalWire for voice calls and SMS. Full call recording, SMS cleaning reminders.",
        stats: [
          { label: "Email Tools", value: "9" },
          { label: "Twilio Tools", value: "10" },
          { label: "Comm Providers", value: "4" },
        ],
        components: [
          "atlas_brain/mcp/email_server.py (9 tools)",
          "atlas_brain/mcp/twilio_server.py (10 tools)",
          "atlas_comms/ (21 files — calls, SMS, scheduling)",
          "Providers: Gmail, Resend, Twilio, SignalWire",
        ],
      },
      {
        name: "Calendar & Scheduling",
        description:
          "Provider-agnostic calendar: Google Calendar (OAuth2) or CalDAV (Nextcloud, Apple, Fastmail). Appointment sync between PostgreSQL appointments table and calendar events. Free slot discovery for booking workflows.",
        stats: [{ label: "Calendar Tools", value: "8" }],
        components: [
          "atlas_brain/mcp/calendar_server.py (8 tools)",
          "atlas_brain/services/calendar_provider.py",
          "Google OAuth: scripts/setup_google_oauth.py",
        ],
      },
      {
        name: "Invoicing",
        description:
          "Monthly auto-generation from calendar events matched to customer services. PDF rendering (fpdf2). Approve-and-send batch flow: generate PDFs, email, mark sent. Overdue checks, payment reminders. SMS cleaning reminders (SignalWire, pending 10DLC).",
        stats: [{ label: "Invoicing Tools", value: "17" }],
        components: [
          "atlas_brain/mcp/invoicing_server.py (17 tools)",
          "atlas_brain/services/invoice_pdf.py",
          "atlas_brain/autonomous/tasks/monthly_invoice_generation.py",
          "atlas_brain/autonomous/tasks/invoice_*.py",
        ],
      },
      {
        name: "Campaign Engine",
        description:
          "Churn signals → lead scoring → campaign sequence generation → multi-step email outreach → ESP webhook ingestion (open, click, bounce, reply) → outcome recording → score calibration loop. CRM webhook ingestion from HubSpot, Salesforce, Pipedrive normalizes events and auto-records outcomes.",
        stats: [
          { label: "CRM Webhooks", value: "3 formats" },
          { label: "Calibration", value: "weekly" },
        ],
        components: [
          "atlas_brain/autonomous/tasks/b2b_campaign_generation.py",
          "atlas_brain/autonomous/tasks/campaign_*.py",
          "Score calibration: calibration_weights table",
          "CRM events: HubSpot, Salesforce, Pipedrive webhooks",
        ],
      },
    ],
    bridges: [
      {
        from: "Campaign Engine",
        to: "Data Refinery",
        description:
          "Calibration feedback loop: campaign outcomes (opened/clicked/replied/converted) feed back into scoring weights. Weekly autonomous task computes per-dimension conversion rates, derives lift adjustments. Weights capped at +/-50% of defaults.",
      },
      {
        from: "Invoicing",
        to: "Calendar & Scheduling",
        description:
          "Monthly invoice task reads calendar events for prior month, matches to customer services by keyword, groups by contact, builds line items (per-visit, per-month, per-hour). Dedup via source_ref prevents double-invoicing.",
      },
    ],
  },
];

export const crossDomainBridges: IntegrationBridge[] = [
  {
    from: "Conversational Intelligence",
    to: "Business Automation",
    description:
      "Voice booking workflow: user says \"book a cleaning for Monday\" → LLM + 3 tools (lookup_availability, lookup_customer, book_appointment) → calendar event created → CRM interaction logged → confirmation spoken via TTS.",
  },
  {
    from: "Data Refinery",
    to: "Operations Platform",
    description:
      "Every refinery stage is orchestrated by autonomous tasks. Scrape intake, enrichment, repair, reasoning, blog generation, campaign generation — all scheduled via cron/interval. MCP tools expose pipeline state to any client.",
  },
  {
    from: "Physical World",
    to: "Conversational Intelligence",
    description:
      "Presence detection (camera + ESPresense) feeds departure events → autonomous task triggers HA automations (lights off, thermostat down, locks engaged). Face recognition results available to voice queries (\"who's at the door?\").",
  },
  {
    from: "Operations Platform",
    to: "Business Automation",
    description:
      "Morning briefing task synthesizes overnight emails, calendar, device status, and security events into a single notification. Gmail digest triages inbox. Invoice overdue check triggers payment reminders via email or SMS.",
  },
  {
    from: "Data Refinery",
    to: "Business Automation",
    description:
      "Churn signals → campaign generation → email outreach → CRM webhook outcomes → score calibration. Battle cards and blog posts auto-generated from synthesized intelligence. Account intelligence feeds sales enablement.",
  },
];
