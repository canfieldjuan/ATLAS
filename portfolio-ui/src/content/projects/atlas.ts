import type { Project } from "@/types";

export const atlasProject: Project = {
  slug: "atlas",
  title: "Atlas",
  tagline: "AI Brain — From Home Automation to Production Intelligence Platform",
  description:
    "An extensible AI system that started as voice-controlled home automation and evolved into a multi-domain intelligence platform. 11 MCP servers, 190 tools, autonomous task orchestration, edge compute on ARM boards, and a B2B churn intelligence pipeline processing 15 review sources.",
  techStack: [
    "Python",
    "FastAPI",
    "PostgreSQL",
    "Neo4j",
    "React",
    "TypeScript",
    "Vite",
    "Tailwind CSS",
    "Ollama (local LLM)",
    "vLLM",
    "MCP Protocol",
    "WebSocket",
    "Tailscale",
    "Orange Pi / Jetson",
    "ONNX Runtime",
    "RKNN NPU",
  ],
  highlights: [
    {
      title: "190 MCP Tools Across 11 Servers",
      description:
        "B2B churn intelligence server alone exposes 83 tools across 17 modules — scraping, enrichment, displacement graphs, calibration loops, webhook delivery, CRM sync.",
      icon: "Wrench",
    },
    {
      title: "Deterministic Pipelines from Non-Deterministic LLMs",
      description:
        "Field ownership contracts, witness verification, calibration feedback loops, archetype classification, score normalization — making LLM output reliable enough for production CRM pushes.",
      icon: "Target",
    },
    {
      title: "Edge / Cloud Split Architecture",
      description:
        "Brain runs on GPU server. Edge nodes (Orange Pi RK3588) handle STT, TTS, computer vision with NPU core allocation and motion gating. Connected via Tailscale.",
      icon: "Network",
    },
    {
      title: "Autonomous Task Orchestration",
      description:
        "51 scheduled tasks, 99 total handlers — cron, interval, and event-triggered — with LLM synthesis, skip-synthesis conventions, fail-open patterns, and notification delivery.",
      icon: "Clock",
    },
    {
      title: "Voice Pipeline",
      description:
        "Wake word activation, streaming STT (Nemotron), intent routing, tool execution, TTS response. Sub-second local processing on $60 ARM board.",
      icon: "Mic",
    },
    {
      title: "Stratified Reasoning",
      description:
        "5-level depth hierarchy with 3 cognitive modes (Recall/Reconstitute/Reason). Dual memory across Postgres semantic cache and Neo4j episodic traces.",
      icon: "Brain",
    },
  ],
  stats: [
    { label: "MCP Servers", value: "11" },
    { label: "Total Tools", value: "190" },
    { label: "Autonomous Tasks", value: "51" },
    { label: "DB Migrations", value: "300" },
    { label: "Python Files", value: "658" },
    { label: "UI Apps", value: "6" },
  ],
  subsystems: [
    {
      name: "Memory & Knowledge Graph",
      description:
        "Dual-store persistence: PostgreSQL for conversations, turns, and JSONB quality signals. Neo4j via Graphiti wrapper for long-term knowledge graph with entity relationships. RAG client unifies retrieval with semantic search, temporal queries, and structured source tracking across both text and voice paths.",
      icon: "Database",
      stats: [
        { label: "Memory MCP Tools", value: "15" },
        { label: "DB Migrations", value: "300" },
      ],
      relatedInsight: "prompting-is-a-science-rag-is-harder-than-you-think",
    },
    {
      name: "Campaign Engine",
      description:
        "End-to-end pipeline: churn signals → lead scoring → campaign sequence generation → multi-step email outreach → ESP webhook ingestion (open, click, bounce, reply) → outcome recording → weekly score calibration loop. CRM webhook ingestion from HubSpot, Salesforce, and Pipedrive normalizes events and auto-records outcomes with rank-based progression.",
      icon: "Megaphone",
      stats: [
        { label: "CRM Webhooks", value: "3 formats" },
        { label: "Calibration", value: "Weekly" },
      ],
      relatedInsight: "autonomy-is-overrated",
    },
    {
      name: "Voice Pipeline",
      description:
        "Distributed across brain (GPU server) and edge (Orange Pi RK3588). Edge handles STT (SenseVoice int8), TTS (Piper, 6.6x realtime), and local skills. Brain handles LLM reasoning, tool calling, and MCP tool access. Connected via Tailscale WebSocket with zlib compression and token batching.",
      icon: "Mic",
      stats: [
        { label: "TTS Speed", value: "6.6x realtime" },
        { label: "NPU Models", value: "5 concurrent" },
        { label: "Hardware Cost", value: "$60" },
      ],
      relatedInsight: "edge-cloud-voice-pipeline-60-dollar-board",
    },
    {
      name: "Invoice Automation",
      description:
        "Monthly cron task reads calendar events, matches to customer service agreements by keyword, groups by contact, generates line items (per-visit, per-month, per-hour), renders branded PDFs via fpdf2, and queues for human approval. Approve-and-send MCP tool handles batch delivery. Overdue checks and payment reminders via email and SMS.",
      icon: "FileText",
      stats: [
        { label: "Invoicing Tools", value: "17" },
        { label: "Accuracy", value: "100%" },
        { label: "Monthly Volume", value: "$16K+" },
      ],
    },
    {
      name: "B2B Churn Intelligence",
      description:
        "7-stage pipeline: 19 parsers scrape review sites → local LLM enrichment (10+ fields per review) → repair pass for contradictions → evidence derivation with confidence scoring → per-vendor reasoning synthesis → cross-vendor displacement graph → downstream artifact generation (battle cards, blog posts, campaigns, PDF reports).",
      icon: "TrendingDown",
      stats: [
        { label: "Parsers", value: "19" },
        { label: "Reviews Enriched", value: "25K+" },
        { label: "Vendors Tracked", value: "56" },
        { label: "B2B MCP Tools", value: "83" },
      ],
      relatedInsight: "building-b2b-churn-intelligence-pipeline",
    },
    {
      name: "Admin Cost Dashboard",
      description:
        "Real-time observability across the entire platform: per-provider LLM cost tracking, token usage aggregation, daily spend charts, scrape success rates per source, CAPTCHA solve metrics, parser version status, task health monitoring, and reasoning depth distribution. Every pipeline stage is surfaced to the UI.",
      icon: "BarChart3",
      stats: [
        { label: "LLM Providers Tracked", value: "9" },
        { label: "Cost Retention", value: "90 days" },
      ],
    },
    {
      name: "Video Processing & Security",
      description:
        "Complete surveillance pipeline: motion detection (MOG2) gates NPU inference → face detection (RetinaFace) → face recognition (MobileFaceNet) → gait analysis → pose estimation (YOLOv8n-pose) → object detection (YOLO-World). RTSP camera support, presence inference fusing camera and ESPresense data, security event escalation to ntfy/SMS/voice.",
      icon: "Eye",
      stats: [
        { label: "Pipeline Files", value: "53" },
        { label: "Detection Models", value: "5" },
        { label: "NPU Inference", value: "94ms" },
      ],
    },
    {
      name: "Autonomous Task System",
      description:
        "51 scheduled tasks (cron, interval, event-triggered) orchestrated by a runner with fail-open patterns. Skip-synthesis saves 10-20s per empty task. LLM synthesis uses domain-specific skill prompts. Per-task notification control via ntfy. Pre-warm for cold LLM starts. Each task is independently testable and debuggable.",
      icon: "Clock",
      stats: [
        { label: "Scheduled Tasks", value: "51" },
        { label: "Total Handlers", value: "99" },
      ],
      relatedInsight: "making-autonomous-ai-tasks-fail-safely",
    },
  ],
  media: [],
};
