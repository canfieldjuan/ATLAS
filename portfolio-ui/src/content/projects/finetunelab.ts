import type { Project } from "@/types";

export const finetunelabProject: Project = {
  slug: "finetunelab",
  title: "FineTuneLab.ai",
  tagline: "End-to-End LLM Fine-Tuning Platform",
  description:
    "A SaaS platform for fine-tuning large language models without deep infrastructure expertise. Unified web interface for training, evaluating, and deploying custom LLMs. 225 API endpoints, multi-provider support, GraphRAG knowledge grounding, and enterprise-grade security.",
  techStack: [
    "Next.js 15",
    "React 19",
    "TypeScript",
    "Supabase (PostgreSQL + RLS)",
    "Neo4j (GraphRAG)",
    "PyTorch",
    "Unsloth (2-4x training speedup)",
    "QLoRA / LoRA",
    "RunPod (serverless GPU)",
    "vLLM",
    "Ollama",
    "AES-256-GCM encryption",
    "Recharts",
  ],
  highlights: [
    {
      title: "12+ LLM Provider Adapters",
      description:
        "Unified client supporting OpenAI, Anthropic (with extended thinking + prompt caching), RunPod, Ollama, HuggingFace, SageMaker, Vertex AI — single interface, provider-agnostic.",
      icon: "Layers",
    },
    {
      title: "Training Pipeline with Checkpoints",
      description:
        "Job queue with pause/resume, checkpoint management, WebSocket live metrics, orphan job recovery. ~9,000 lines of training orchestration.",
      icon: "Cpu",
    },
    {
      title: "LLM-as-Judge Evaluation",
      description:
        "3-tier evaluation system: rule-based, human review, and LLM judge with configurable rubrics. Scheduled recurring evals with regression detection.",
      icon: "Scale",
    },
    {
      title: "GraphRAG Knowledge Grounding",
      description:
        "Multi-format document parsing, semantic + keyword hybrid search, source attribution. Knowledge graphs in Neo4j for context-aware fine-tuning.",
      icon: "GitBranch",
    },
    {
      title: "Hierarchical Tracing",
      description:
        "50+ fields per operation — tokens, cost, latency, RAG context. Full observability across training runs, inference calls, and evaluations.",
      icon: "Activity",
    },
    {
      title: "Enterprise Security",
      description:
        "AES-256-GCM with PBKDF2 key derivation for API keys at rest. Row-level security in Supabase. Multi-tier auth (JWT/API key/service role).",
      icon: "Shield",
    },
  ],
  stats: [
    { label: "API Endpoints", value: "225" },
    { label: "LLM Providers", value: "12+" },
    { label: "Training Code", value: "9K lines" },
    { label: "Trace Fields", value: "50+" },
    { label: "Eval Tiers", value: "3" },
    { label: "Encryption", value: "AES-256" },
  ],
  media: [],
};
