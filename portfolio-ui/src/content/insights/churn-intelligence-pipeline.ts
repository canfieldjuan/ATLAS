import type { InsightPost } from "@/types";

export const churnIntelligencePipeline: InsightPost = {
  slug: "building-b2b-churn-intelligence-pipeline",
  title: "Building a B2B Churn Intelligence Pipeline from 16 Review Sources",
  description:
    "How I built a pipeline that scrapes 15 review platforms, enriches reviews with local LLM inference, constructs displacement graphs, and generates calibrated churn signals with hybrid local/cloud inference and multi-provider cost routing.",
  date: "2026-04-10",
  type: "case-study",
  tags: [
    "pipeline architecture",
    "LLM enrichment",
    "web scraping",
    "data engineering",
    "churn intelligence",
  ],
  project: "atlas",
  seoTitle: "Building a B2B Churn Intelligence Pipeline with LLM Enrichment",
  seoDescription:
    "Case study: a 7-stage intelligence pipeline processing 15 review sources with local LLM enrichment, displacement graphs, calibrated scoring, and cost-optimized multi-provider inference routing.",
  targetKeyword: "b2b churn intelligence pipeline",
  secondaryKeywords: [
    "llm data pipeline",
    "review intelligence",
    "churn prediction from reviews",
  ],
  faq: [
    {
      question: "How do you make LLM enrichment deterministic enough for production?",
      answer:
        "Three layers: structured extraction prompts with JSON schema enforcement, a repair pass that catches contradictions and weak extractions, and confidence scoring based on source diversity and mention frequency. Reviews that fail quality gates get re-queued, not silently passed through.",
    },
    {
      question: "Why local LLM instead of cloud APIs?",
      answer:
        "At scale, the cost calculus flips. Processing tens of thousands of reviews through cloud APIs would cost hundreds of dollars per run. Local inference on Qwen3-30B-A3B via vLLM costs $0 per token — the only cost is GPU time, which is already allocated. The tradeoff is latency, not dollars.",
    },
    {
      question: "How does the displacement graph work?",
      answer:
        "When a review mentions switching from Vendor A to Vendor B, that creates a displacement edge — an append-only time-series record with confidence scoring. Over time, these edges form a directed graph showing where customers are flowing. Concurrent events across 3+ vendors surface market-level shifts.",
    },
  ],
  content: `
<h2>The Problem</h2>
<p>B2B software buyers leave signals everywhere — G2 reviews, Reddit threads, Gartner peer reviews, GitHub issues, HackerNews discussions. But this data is scattered across 16+ platforms, unstructured, and noisy. Turning it into actionable churn intelligence requires a pipeline that's both broad (covering all sources) and deep (extracting structured signals from free text).</p>

<p>This is the system I built. It's not a proof-of-concept — it runs in production, processes thousands of reviews, and feeds downstream products including battle cards, campaign sequences, and CRM pushes.</p>

<h2>Pipeline Architecture: 7 Stages</h2>

<h3>Stage 1: Ingest</h3>
<p>15 review sources, each with different access patterns. G2 requires a 3-tier fallback (Web Unlocker → Playwright → residential proxy). Reddit uses API. GitHub and HackerNews are open. Each source has a capability profile defining anti-bot classification, proxy requirements, and data quality tier.</p>
<p>Priority-ordered scheduling with per-source semaphores (4 concurrent for web scrapers, 10 for APIs), exponential backoff on blocks, and cooldown periods. Every scrape is logged: reviews found, pages scraped, duration, proxy type, CAPTCHA attempts.</p>

<h3>Stage 2: Enrichment</h3>
<p>Raw reviews become structured intelligence. A local LLM (Qwen3-30B-A3B on vLLM) extracts 10+ fields per review: pain categories, competitor mentions, switching intent, sentiment, use cases, integration mentions, buyer role signals, and evidence quotes.</p>
<p>This is where most AI pipelines break. The LLM doesn't always return clean JSON. It hallucinates fields. It contradicts itself. Making this work requires JSON schema enforcement, output validation, and a retry loop that catches malformed responses before they poison downstream data.</p>

<h3>Stage 3: Repair</h3>
<p>A second pass catches what enrichment missed. Reviews with low-confidence extractions, contradictory signals (e.g., high praise + high churn intent), or missing critical fields get re-queued through a repair pipeline. This isn't optional — without it, downstream artifacts inherit upstream noise.</p>

<h3>Stage 4: Evidence Derivation</h3>
<p>Enriched reviews become evidence primitives — the atomic units of intelligence. Each piece of evidence has a type (pain point, competitor mention, switching trigger, praise signal), a confidence score based on source diversity and mention frequency, and provenance back to the original review.</p>

<h3>Stage 5: Vendor Reasoning</h3>
<p>Evidence aggregates into per-vendor intelligence: churn signals, product profiles, pain point rankings, use case maps, integration inventories, and buyer profiles. Each synthesis step uses the LLM but constrains it with structured data — the LLM reasons over evidence, not raw text.</p>

<h3>Stage 6: Cross-Vendor Reasoning</h3>
<p>The displacement graph emerges here. When reviews mention switching from Vendor A to Vendor B, that creates a directed edge with confidence scoring. Over time, these edges reveal market dynamics: which vendors are gaining, which are bleeding, and what triggers the shift.</p>
<p>Concurrent event detection finds dates where 3+ vendors experienced the same type of change (urgency spike, NPS shift, new pain category). These are market-level signals, not vendor-specific noise.</p>

<h3>Stage 7: Artifact Generation</h3>
<p>Downstream products consume the intelligence: battle cards, blog posts (80+ generated), campaign sequences, PDF reports, CRM pushes, and webhook deliveries. Each artifact is traceable back through the pipeline to source reviews.</p>

<h2>Making Non-Deterministic Output Deterministic</h2>
<p>This is the core engineering challenge that separates AI pipeline work from chatbot development. Three mechanisms:</p>

<p><strong>Calibration loops:</strong> Campaign outcomes (opened, clicked, replied, converted) feed back into scoring weights. A weekly autonomous task computes per-dimension conversion rates, derives lift adjustments, and updates the score computation. Weights are capped at ±50% of static defaults to prevent wild swings. Requires 20+ sequences with outcomes before producing weights.</p>

<p><strong>Parser versioning:</strong> Every review tracks which parser version extracted it. When parsers improve, outdated reviews automatically re-queue for re-enrichment. No manual intervention needed.</p>

<p><strong>Correction persistence:</strong> Analysts can create data corrections via MCP tools. Corrections are append-only with full audit trails and revert capability. Source-level correction impact is tracked to identify systematically problematic sources.</p>

<h2>The Numbers</h2>
<ul>
  <li><strong>16</strong> review sources actively scraped</li>
  <li><strong>60+</strong> MCP tools exposing the pipeline</li>
  <li><strong>80+</strong> blog posts generated from pipeline data</li>
  <li><strong>$0</strong> LLM inference cost (local vLLM)</li>
  <li><strong>7</strong> pipeline phases, all 100% complete</li>
  <li><strong>3</strong> CRM webhook formats supported (HubSpot, Salesforce, Pipedrive)</li>
</ul>

<h2>What I Learned</h2>
<p>The hard part isn't getting the LLM to extract data. It's building everything around it: the retry logic, the validation, the repair passes, the confidence scoring, the calibration loops, the correction system. The LLM is maybe 15% of the code. The other 85% is making its output trustworthy enough to act on.</p>
`,
};
