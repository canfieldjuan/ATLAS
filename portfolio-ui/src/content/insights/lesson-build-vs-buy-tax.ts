import type { InsightPost } from "@/types";

export const lessonBuildVsBuyTax: InsightPost = {
  slug: "the-tax-of-third-party-apps-vs-custom-code",
  title: "Abstraction Removes Visible Complexity. It Does Not Remove Actual Complexity.",
  description:
    "Every third-party tool hides complexity. It doesn't eliminate it. It moves it — to the vendor, the framework, the platform, or your future self. That last one gets people. Here's how every Atlas subsystem maps to a build-vs-buy decision and the deciding factor that tipped it.",
  date: "2026-04-13",
  type: "lesson",
  tags: [
    "build vs buy",
    "abstraction",
    "deferred complexity",
    "architecture decisions",
    "infrastructure ownership",
  ],
  project: "atlas",
  seoTitle: "Abstraction Hides Complexity, It Doesn't Remove It: Build vs Buy in AI Systems",
  seoDescription:
    "Abstraction moves complexity to the vendor, the framework, or your future self. How every Atlas subsystem maps to a build-vs-buy decision — and the 6 deciding factors that tipped each one.",
  targetKeyword: "abstraction complexity build vs buy",
  secondaryKeywords: [
    "deferred complexity ai",
    "third party dependency cost",
    "custom code vs saas ai infrastructure",
  ],
  faq: [
    {
      question: "What's the difference between abstracting complexity and eliminating it?",
      answer:
        "Abstraction hides complexity behind an interface — you don't see it, but it still exists. Elimination removes it entirely. Stripe abstracts PCI compliance; you still need PCI compliance, Stripe just handles it. If Stripe disappears, the complexity reappears. Third-party tools abstract. Custom code owns. Neither eliminates.",
    },
    {
      question: "How do you decide what to build custom vs use a third-party tool?",
      answer:
        "Six factors: Do you need tight vendor scoping (control what data goes where)? Do you need control over prompting and reasoning? Is cost sensitivity high at your volume? Is latency critical? Do you need inspectability when something breaks at 3AM? Does your UX depend on the system's real structure? If yes to 2+ of these, build custom. If none apply, use the tool.",
    },
  ],
  content: `
<h2>The Core Insight</h2>
<p><strong>Abstraction removes visible complexity. It does not remove actual complexity.</strong></p>
<p>It just moves that complexity somewhere else. Usually to:</p>
<ul>
  <li>The vendor</li>
  <li>The framework</li>
  <li>The platform</li>
  <li>Or your future self</li>
</ul>
<p>That last one gets people.</p>
<p>Because sometimes what feels "simple" today is just deferred complexity. It comes back later as debugging blind spots, scaling pain, weird limitations, vendor lock-in, rising costs, and brittle workarounds.</p>

<h2>Two Examples That Make This Concrete</h2>

<h3>Stripe Checkout</h3>
<p>If you use Stripe's hosted checkout page, that's abstraction. You don't manage PCI compliance, UI flow, edge-case payment handling. Great. But Stripe owns a lot of that path.</p>
<p>If you build your own payment flow on top of their APIs, you own more: the UX, the validation, the event handling, the recovery logic, the downstream workflow. More control, more responsibility.</p>

<h3>LLM Pipeline</h3>
<p>If you call a workflow tool that says "summarize docs and send report," that's abstraction. Fast, simple. But if summaries degrade, costs spike, or logs are weak, you're boxed in.</p>
<p>If you build your own pipeline — ingestion, chunking, routing, model selection, retries, evals, caching, telemetry — now you own it. That gives you power, but also maintenance burden.</p>

<p>Same pattern both times. The question is never "is abstraction good or bad?" It's <strong>"where does the complexity land when something goes wrong?"</strong></p>

<h2>The 6 Deciding Factors</h2>
<p>Every build-vs-buy decision in Atlas came down to one or more of these:</p>

<table>
  <thead>
    <tr><th>Factor</th><th>What It Means</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Tight vendor scoping</strong></td>
      <td>Control exactly what data goes where, what fields get exposed, what third parties see. No ambient data leakage.</td>
    </tr>
    <tr>
      <td><strong>Control over prompting and reasoning</strong></td>
      <td>Craft the exact prompt, enforce field contracts, choose the model tier per task. Can't do this through someone else's "AI summarizer" button.</td>
    </tr>
    <tr>
      <td><strong>Cost sensitivity</strong></td>
      <td>At thousands of items per week, per-API-call pricing compounds. Owning the pipeline means the marginal cost is your GPU, not someone's rate card.</td>
    </tr>
    <tr>
      <td><strong>Latency</strong></td>
      <td>A voice pipeline that adds 500ms for a third-party round trip isn't a voice pipeline. An invoice that takes 3 seconds to render through a SaaS API isn't fast enough for bulk generation.</td>
    </tr>
    <tr>
      <td><strong>Inspectability</strong></td>
      <td>When something breaks at 3AM, can you read the code that broke? Or do you read logs that say "third-party service returned error 500" and open a support ticket?</td>
    </tr>
    <tr>
      <td><strong>UX depends on real structure</strong></td>
      <td>If your dashboard needs to show pipeline stages, cost per provider, scrape health per source — it needs real data from real infrastructure, not a generic workflow builder's abstraction.</td>
    </tr>
  </tbody>
</table>

<h2>How Every Atlas Subsystem Maps</h2>

<h3>Enrichment Pipeline — custom</h3>
<p><strong>Deciding factors:</strong> prompting control, cost sensitivity, inspectability</p>
<p>A third-party "AI enrichment" service would mean: their prompts, their model, their cost per call, their logs. When enrichment quality degrades — and it will — you'd be filing a support ticket instead of editing a prompt. The 7-stage pipeline (ingest, enrich, repair, evidence, reasoning, cross-vendor, artifacts) exists because each stage has different quality requirements and different failure modes. No workflow builder models that.</p>

<h3>Scraping Infrastructure — custom</h3>
<p><strong>Deciding factors:</strong> vendor scoping, cost sensitivity, inspectability</p>
<p>Scraping SaaS tools exist. They charge per page, per API call, per result. At 19 sources and thousands of reviews per week, the math doesn't work. More importantly: when G2 changes their anti-bot strategy, I need to update a parser, not wait for a vendor to notice. The 3-tier G2 fallback (Web Unlocker, Playwright, residential) exists because I can see exactly where scraping breaks and fix it immediately.</p>

<h3>CRM — custom (on Postgres)</h3>
<p><strong>Deciding factors:</strong> vendor scoping, UX dependency, cost sensitivity</p>
<p>HubSpot would have given me a CRM in a day. But HubSpot owns the data model. Their contact fields, their interaction schema, their API rate limits. The Atlas CRM is 10 MCP tools over raw asyncpg queries. When the invoicing system needs to join contacts to appointments to calendar events, it's one SQL query — not three API calls to three different services with three different auth tokens.</p>

<h3>Invoicing — custom</h3>
<p><strong>Deciding factors:</strong> inspectability, UX dependency, latency</p>
<p>Stripe Invoicing or FreshBooks would abstract the invoice rendering. But when a Unicode em-dash broke the PDF encoder, I fixed 303 lines of fpdf2 code in 10 minutes. Through a SaaS? Support ticket, 24-hour wait, "we'll look into it." The monthly auto-generation task matches calendar events to customer services — that join logic lives in my database, not in someone else's workflow builder.</p>

<h3>Autonomous Task System — custom</h3>
<p><strong>Deciding factors:</strong> prompting control, inspectability, UX dependency</p>
<p>Could have used a task queue SaaS (Temporal, Inngest). But 51 scheduled tasks with fail-open patterns, skip-synthesis conventions, per-task LLM skill loading, and ntfy delivery — that's not a "run this function on a schedule" problem. It's an orchestration layer where every task has different preconditions, different synthesis requirements, and different failure modes. A generic task runner hides the complexity I need to see.</p>

<h3>Voice Pipeline — custom</h3>
<p><strong>Deciding factors:</strong> latency, cost sensitivity, vendor scoping</p>
<p>A voice API like Vapi or Bland would abstract the STT/TTS/LLM chain. But adding a third-party round trip to a pipeline where sub-second matters isn't acceptable. Running STT and TTS on a $60 edge node over Tailscale gives me latency control that no hosted service matches. And the audio never leaves my network — vendor scoping matters when voice data is involved.</p>

<h3>Memory / RAG — custom</h3>
<p><strong>Deciding factors:</strong> prompting control, inspectability, UX dependency</p>
<p>Pinecone or Weaviate would give me a vector DB. But RAG isn't "query a vector DB" — it's retrieval validation, context budget management, grounding checks, dual-store retrieval (Postgres for conversations + Neo4j for knowledge graph), and source tracking. The RAG client unifies all of this with structured <code>SearchSource[]</code> objects. A third-party vector DB is one piece of a system that needs to be inspectable end-to-end.</p>

<h3>Admin / Telemetry — custom</h3>
<p><strong>Deciding factors:</strong> UX dependency, inspectability</p>
<p>Datadog or Grafana could chart my metrics. But the Admin UI shows pipeline-specific telemetry: cost per LLM provider, scrape success rates per source, CAPTCHA solve times, parser version status, reasoning depth distribution. This data comes from 300 migration tables that no off-the-shelf dashboard understands. The UI is built on the system's real structure because the structure IS the insight.</p>

<h2>What I Do Use Third-Party For</h2>
<p>Not everything is custom. The deciding factors pointed the other way for:</p>
<ul>
  <li><strong>Email delivery</strong> (Gmail API, Resend) — deliverability is a specialization. Running my own SMTP would be custom for the sake of custom.</li>
  <li><strong>Telephony</strong> (Twilio, SignalWire) — telecom infrastructure is not my problem. Their abstraction genuinely eliminates complexity I'd never want to own.</li>
  <li><strong>OAuth / Auth</strong> (Google OAuth2) — security-critical, well-standardized, no reason to reimplement.</li>
  <li><strong>Hosting / CDN</strong> (Vercel, Docker) — infrastructure-as-a-service is the right abstraction layer.</li>
  <li><strong>LLM providers</strong> (Anthropic, Ollama) — I don't train models. I use models. That's a clear buy.</li>
</ul>
<p>Pattern: use third-party when the domain is genuinely not your problem and the abstraction doesn't hide something you'll need to debug. Build custom when the complexity is your product's complexity — when hiding it means hiding from your own system.</p>

<h2>The Real Question</h2>
<p>Every abstraction is a bet: "this complexity will never be my problem." Sometimes that bet pays off (Stripe handling PCI compliance). Sometimes it doesn't (a workflow builder hiding the pipeline state you need to debug a quality regression).</p>

<p>The question isn't build or buy. It's: <strong>when this complexity comes back — and it will — do I want to meet it in my own code where I can see everything? Or in someone else's black box where I can see nothing?</strong></p>

<p>After enough 3AM debugging sessions against opaque third-party services, the answer writes itself.</p>
`,
};
