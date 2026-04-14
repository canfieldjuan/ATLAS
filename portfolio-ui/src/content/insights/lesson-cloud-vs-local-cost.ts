import type { InsightPost } from "@/types";

export const lessonCloudVsLocalCost: InsightPost = {
  slug: "cloud-vs-local-llm-cost-quality-tradeoff",
  title: "Cloud vs Local LLM: The Cost/Quality Tradeoff Nobody Warns You About",
  description:
    "At thousands of reviews per week, cloud API costs compound fast. Local models save money but sacrifice extraction quality. Flagship model doesn't mean best option. Here's how the tradeoffs actually play out in production.",
  date: "2026-04-12",
  type: "lesson",
  tags: [
    "LLM cost",
    "local inference",
    "cloud API",
    "production tradeoffs",
    "data quality",
  ],
  project: "atlas",
  seoTitle: "Cloud vs Local LLM Cost: Real Production Tradeoffs at Scale",
  seoDescription:
    "Production lesson: cloud API costs explode at scale, local models sacrifice quality. How to make the cost/quality tradeoff when processing thousands of reviews per week.",
  targetKeyword: "cloud vs local llm cost",
  secondaryKeywords: [
    "llm inference cost production",
    "local llm quality tradeoff",
    "ai cost optimization",
  ],
  faq: [
    {
      question: "When should you use local LLM vs cloud API?",
      answer:
        "Use local for high-volume, narrow tasks: structured extraction, classification, sentiment analysis. Use cloud for complex reasoning that requires frontier-model quality: cross-vendor synthesis, battle card generation, nuanced competitive analysis. The worst decision is using a flagship model for a task a 7B model handles fine.",
    },
    {
      question: "How much does cloud LLM inference actually cost at scale?",
      answer:
        "Processing 1,000 reviews through Claude or GPT-4 for enrichment costs $50-150 depending on prompt length and output. Do that weekly across multiple pipeline stages and you're at $200-600/week just for enrichment. Local inference on a GPU you already own costs $0 per token — the tradeoff is quality and latency, not money.",
    },
  ],
  content: `
<h2>The Realization</h2>
<p>When you're processing a few hundred items, cloud API costs are invisible. A few dollars here, a few dollars there. But we're processing thousands of reviews per week across a 7-stage pipeline. Each review touches the LLM multiple times: enrichment, repair, evidence derivation, reasoning. The costs compound fast.</p>

<p>The first instinct is "just use a local model." And that works — until you compare the extraction quality side by side.</p>

<h2>What Actually Happened</h2>
<p>We started with cloud APIs (Anthropic, OpenRouter) for everything. The quality was excellent — structured extraction was clean, pain categories were consistent, competitor mentions were accurate. But the bill grew linearly with review volume.</p>

<p>So we moved enrichment to local inference: Qwen3-30B-A3B on vLLM. Cost dropped to $0. But extraction quality dropped too. Not catastrophically — maybe 85% as good. The problem is that 85% quality in the enrichment stage cascades. By the time you reach downstream artifacts (battle cards, campaigns), the quality gap has multiplied through every stage.</p>

<h2>The Tradeoff Framework We Landed On</h2>

<h3>Local models are extremely capable — in narrow domains</h3>
<p>Structured extraction with a clear JSON schema? Local handles it. Binary classification (churning/not churning)? Local is fine. Sentiment analysis? Local is great. These are constrained tasks where you can validate the output mechanically.</p>

<h3>Flagship models aren't the best option — they're the most expensive option</h3>
<p>Using Claude Opus to classify sentiment is like hiring a senior architect to sort mail. The task doesn't need that capability, and you're paying for reasoning depth you won't use. The right model is the cheapest one that meets the quality bar for that specific task.</p>

<h3>The real skill is tiered routing</h3>
<p>Our LLM router now decides per-task:</p>
<ul>
  <li><strong>Local (Qwen3 via vLLM):</strong> High-volume enrichment, classification, structured extraction — tasks where schema enforcement catches most errors</li>
  <li><strong>Mid-tier cloud (OpenRouter, Groq):</strong> Repair passes, evidence derivation — tasks where some reasoning is needed but frontier quality isn't</li>
  <li><strong>Frontier cloud (Anthropic):</strong> Cross-vendor synthesis, battle cards, campaign messaging — tasks where nuance and quality directly impact the end product</li>
</ul>

<h2>Bottlenecks Compound at Scale</h2>
<p>This isn't just about LLM cost. Every inefficiency multiplies:</p>
<ul>
  <li>An extra API call per review × 1,000 reviews/week = 1,000 unnecessary calls</li>
  <li>Unbatched DB writes × 10 fields per review = 10,000 individual inserts instead of 1,000 batch inserts</li>
  <li>Redundant re-enrichment of already-good reviews = burning tokens on work that's already done</li>
</ul>
<p>At small scale, none of this matters. At production scale, it's the difference between a system that runs in 2 hours and one that runs in 12.</p>

<p>This is why parser versioning exists (only re-enrich when the parser actually improves), why skip-synthesis conventions exist (don't call the LLM when there's nothing to synthesize), why the Token Batcher exists (don't send 1,000 individual WebSocket messages when you can batch them).</p>

<h2>The Uncomfortable Truth</h2>
<p>Saving money means sacrificing data quality somewhere. The skill isn't avoiding the tradeoff — it's choosing <em>where</em> to make it. Sacrifice quality on classification (cheap to validate) rather than on synthesis (expensive to fix). Use local models where the output is mechanically verifiable, cloud models where the output quality is the product.</p>

<p>I got very comfortable with tradeoffs. That's not a euphemism for "I settled for worse." It means I learned which 15% of quality loss is acceptable and which 5% is catastrophic — and they're not the same 5% for every task.</p>
`,
};
