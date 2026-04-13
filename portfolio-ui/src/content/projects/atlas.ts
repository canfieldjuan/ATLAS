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
      title: "60+ MCP Tools",
      description:
        "B2B churn intelligence server alone exposes 60+ tools — scraping, enrichment, displacement graphs, calibration loops, webhook delivery, CRM sync.",
      icon: "Wrench",
    },
    {
      title: "Deterministic Pipelines from Non-Deterministic LLMs",
      description:
        "Calibration feedback loops, archetype classification, score normalization — making LLM output reliable enough for production CRM pushes.",
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
        "36 scheduled tasks — cron and interval — with LLM synthesis, skip-synthesis conventions, fail-open patterns, and notification delivery.",
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
    { label: "MCP Servers", value: "7" },
    { label: "Total Tools", value: "100+" },
    { label: "Review Sources", value: "16" },
    { label: "Autonomous Tasks", value: "14" },
    { label: "LLM Cost (B2B pipeline)", value: "$0" },
    { label: "Edge Node Latency", value: "<1s" },
  ],
  media: [],
};
