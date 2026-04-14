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
  subsystems: [
    {
      name: "Training Pipeline",
      description:
        "Full job lifecycle management: queuing, GPU allocation (RunPod serverless), training with Unsloth (2-4x speedup), checkpoint saving at configurable intervals, pause/resume mid-training, WebSocket live metrics streaming, and orphan job recovery for runs that die mid-flight. ~9,000 lines of Python orchestration.",
      icon: "Cpu",
      stats: [
        { label: "Training Code", value: "9K lines" },
        { label: "Speedup", value: "2-4x (Unsloth)" },
      ],
    },
    {
      name: "LLM-as-Judge Evaluation",
      description:
        "3-tier evaluation: rule-based checks (format, length, keyword presence), human review (manual scoring with rubric), and LLM judge (configurable model evaluates quality against criteria). Scheduled recurring evaluations detect regression — if a fine-tuned model degrades over time, the system flags it before production impact.",
      icon: "Scale",
      stats: [
        { label: "Eval Tiers", value: "3" },
        { label: "Anomaly Detection", value: "Z-score / IQR" },
      ],
    },
    {
      name: "Unified LLM Client",
      description:
        "Single adapter pattern supporting 12+ providers: OpenAI, Anthropic Claude (with extended thinking and prompt caching), RunPod, Ollama, HuggingFace Inference, AWS SageMaker, Google Vertex AI, Together, and more. Provider-agnostic interface means switching models is a config change, not a code change.",
      icon: "Layers",
      stats: [
        { label: "Providers", value: "12+" },
        { label: "Features", value: "Extended thinking, prompt caching" },
      ],
    },
    {
      name: "GraphRAG Knowledge Grounding",
      description:
        "Multi-format document ingestion (PDF, DOCX, TXT, markdown), chunk-level embedding, Neo4j knowledge graph construction with entity and relationship extraction. Hybrid retrieval: semantic vector search + keyword BM25. Source attribution on every retrieved chunk so the model's answers are traceable to documents.",
      icon: "GitBranch",
      stats: [
        { label: "Search Modes", value: "Semantic + keyword hybrid" },
      ],
    },
    {
      name: "Hierarchical Tracing & Observability",
      description:
        "Every LLM call, training step, and evaluation is traced with 50+ fields: input/output tokens, cost, latency, model version, prompt hash, RAG context used, and parent-child relationships between operations. Enables cost attribution per feature, latency profiling, and debugging of complex multi-step workflows.",
      icon: "Activity",
      stats: [
        { label: "Trace Fields", value: "50+" },
        { label: "Cost Attribution", value: "Per-operation" },
      ],
    },
    {
      name: "Security & Multi-Tenancy",
      description:
        "AES-256-GCM encryption with PBKDF2 key derivation for all API keys at rest. Supabase row-level security ensures tenant isolation at the database layer. Multi-tier authentication: JWT for web sessions, API keys for programmatic access, service role for internal operations. Every data query is scoped to the authenticated tenant.",
      icon: "Shield",
      stats: [
        { label: "Encryption", value: "AES-256-GCM" },
        { label: "Auth Tiers", value: "3 (JWT/API key/service)" },
      ],
    },
  ],
  media: [],
};
