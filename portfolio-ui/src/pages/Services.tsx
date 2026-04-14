import { Link } from "react-router-dom";
import { SeoHead } from "@/components/seo/SeoHead";
import {
  ArrowRight,
  MailCheck,
  Database,
  Megaphone,
  ClipboardList,
  BarChart3,
  Bot,
  Zap,
  CheckCircle,
} from "lucide-react";

const automationCategories = [
  {
    icon: ClipboardList,
    title: "Customer Operations",
    before: "Support tickets manually triaged, slow first response, repeated info requests",
    after: "Auto-classified, context-enriched, routed with draft responses — reviewed before send",
  },
  {
    icon: Database,
    title: "Data Pipelines",
    before: "Manual data collection, copy-paste between tools, inconsistent formatting",
    after: "Scheduled ingestion, automated enrichment, validated outputs, structured storage",
  },
  {
    icon: Megaphone,
    title: "Marketing & Outreach",
    before: "Generic campaigns, manual lead research, no feedback loop on what works",
    after: "Signal-driven targeting, personalized sequences, outcome tracking that improves over time",
  },
  {
    icon: MailCheck,
    title: "Internal Operations",
    before: "Manual invoicing, calendar-based scheduling, email-driven approvals",
    after: "Auto-generated invoices from calendar events, approval workflows with audit trails",
  },
  {
    icon: BarChart3,
    title: "Reporting & Intelligence",
    before: "Weekly manual reports, data pulled from 5 tools, outdated by delivery",
    after: "Automated synthesis, real-time dashboards, scheduled digests with actionable summaries",
  },
];

const processPhases = [
  {
    number: "01",
    title: "Discover",
    subtitle: "Map what actually happens today",
    description:
      "Before touching technology, we map the real workflow — not the documented one. Who touches it, what decisions they make, what information they need, where things fall through the cracks. The goal is to understand the process well enough to classify every step.",
    details: [
      "Stakeholder interviews — the people who do the work, not just the people who manage it",
      "Process mapping — every step, every decision point, every handoff",
      "Pain point identification — where time is wasted, where errors happen, where context is lost",
      "Data audit — what exists, what's accessible, what's missing, what format it's in",
      "Volume assessment — 50 items/day is a different architecture than 5,000/day",
    ],
  },
  {
    number: "02",
    title: "Classify",
    subtitle: "Separate what needs AI from what doesn't",
    description:
      "Most of any workflow is deterministic — rules-based steps that don't need AI at all. Maybe 20-30% actually benefits from intelligence. The rest needs better traditional automation. Putting AI on a deterministic step wastes money and adds unpredictability.",
    details: [
      "Deterministic steps → rules engine or simple automation (routing, filtering, formatting)",
      "Extraction steps → structured data extraction with validation (pull fields from unstructured text)",
      "Judgment steps → classification with confidence scoring (urgency, sentiment, category)",
      "Generation steps → content creation with approval gates (drafts, summaries, responses)",
      "Human-required steps → kept manual, with better tooling to support the decision",
    ],
  },
  {
    number: "03",
    title: "Design",
    subtitle: "Architecture before code",
    description:
      "Every automated workflow needs gates — points where the system checks its own work or waits for human approval before taking external action. The design phase defines what gets automated, what gets augmented, and what stays manual.",
    details: [
      "Gate placement — where does a human review before external action?",
      "Fallback paths — what happens when the AI isn't confident or the extraction fails?",
      "Model selection — which steps need frontier-quality AI vs a rules engine vs a simple classifier?",
      "Data flow — how does information move between steps without losing context?",
      "Cost modeling — what does this cost to run at your volume, daily?",
    ],
  },
  {
    number: "04",
    title: "Build",
    subtitle: "Incremental delivery, not a big bang",
    description:
      "We build in layers: deterministic automation first (immediate value, zero AI risk), then augmented steps (AI assists, human decides), then automated steps (AI acts within validated boundaries). Each layer is testable independently.",
    details: [
      "Layer 1 — Deterministic automation: routing, formatting, data sync. Works day one.",
      "Layer 2 — AI-augmented steps: extraction and classification running alongside human review. Building trust.",
      "Layer 3 — AI-automated steps: generation and action with gates, confidence thresholds, and fallbacks. Earning autonomy.",
      "Integration — connects to your existing tools (CRM, email, calendar, databases) without ripping anything out",
      "Observability — every automated step is logged, traceable, and debuggable",
    ],
  },
  {
    number: "05",
    title: "Validate",
    subtitle: "Prove it works before you depend on it",
    description:
      "AI outputs are probabilistic — the same input can produce different results. Validation isn't a one-time test; it's a continuous system. We build validation layers that run on every execution, catching quality regressions before they reach your customers.",
    details: [
      "Schema validation — does every output meet the required format and constraints?",
      "Confidence gating — low-confidence results get flagged for human review, not pushed through",
      "Outcome tracking — does the automation actually improve the metric it was built for?",
      "Feedback loops — results that get corrected by humans feed back into the system's accuracy",
      "Cost monitoring — is the system staying within the projected operational budget?",
    ],
  },
];

const proofPoints = [
  {
    title: "Invoice Generation from Calendar Events",
    outcome:
      "Monthly invoicing that took 3-4 hours of manual work now runs automatically — matching calendar events to customer service agreements, generating line items, rendering PDFs, and queuing for approval. One click to review and send.",
    tags: ["Internal Operations", "Deterministic + Generation"],
  },
  {
    title: "B2B Intelligence Pipeline",
    outcome:
      "25,000+ reviews from 15 sources automatically ingested, enriched with structured data extraction, validated through quality gates, and synthesized into competitive intelligence reports. Manual research that would take weeks runs on a schedule.",
    tags: ["Data Pipeline", "Extraction + Judgment + Generation"],
  },
  {
    title: "Campaign Engine with Outcome Calibration",
    outcome:
      "Signal-driven outreach that generates personalized email sequences, delivers them on schedule, tracks outcomes (opens, clicks, replies), and feeds results back into scoring. Campaign quality improves automatically over time.",
    tags: ["Marketing & Outreach", "Full automation with feedback loop"],
  },
];

export default function Services() {
  return (
    <>
      <SeoHead
        meta={{
          title: "What We Build",
          description:
            "We turn manual business workflows into repeatable, reliable automated systems. Customer operations, data pipelines, marketing, invoicing, reporting — built with the right tool for each step, not AI for everything.",
          canonicalPath: "/services",
          jsonLd: {
            "@context": "https://schema.org",
            "@type": "Service",
            name: "Business Workflow Automation",
            description:
              "End-to-end automation of business workflows — discovery, classification, design, build, and validation. AI where it helps, traditional automation where it doesn't.",
            provider: {
              "@type": "Person",
              name: "Juan Canfield",
              jobTitle: "AI Systems Architect",
            },
          },
        }}
      />

      {/* Hero */}
      <section className="py-20 px-6">
        <div className="mx-auto max-w-4xl text-center">
          <h1 className="text-4xl sm:text-5xl font-bold text-white leading-tight mb-6">
            We Turn Manual Workflows Into{" "}
            <span className="text-gradient">Repeatable, Reliable Systems</span>
          </h1>
          <p className="text-lg text-surface-200/70 max-w-2xl mx-auto leading-relaxed">
            Not every problem needs AI. Some need better automation. Some need
            both. We figure out which is which, then build the system that
            runs without you — with gates so it never acts without your
            approval.
          </p>
        </div>
      </section>

      {/* What we automate */}
      <section className="py-16 px-6 border-t border-surface-700/30">
        <div className="mx-auto max-w-4xl">
          <h2 className="text-2xl font-bold text-white mb-3">
            What We Automate
          </h2>
          <p className="text-surface-200/60 mb-10 max-w-2xl">
            Tedious tasks, business processes, marketing operations, data
            pipelines, and internal workflows — transformed from manual
            effort into systems that produce consistent output.
          </p>

          <div className="space-y-4">
            {automationCategories.map((cat) => (
              <div
                key={cat.title}
                className="rounded-xl border border-surface-700/50 bg-surface-800/30 p-5"
              >
                <div className="flex items-center gap-3 mb-4">
                  <div className="h-9 w-9 rounded-lg bg-primary-500/10 flex items-center justify-center">
                    <cat.icon size={18} className="text-primary-400" />
                  </div>
                  <h3 className="font-semibold text-white">{cat.title}</h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <span className="text-[10px] uppercase tracking-widest text-rose-400/60 font-medium">
                      Before
                    </span>
                    <p className="text-sm text-surface-200/60 mt-1">
                      {cat.before}
                    </p>
                  </div>
                  <div>
                    <span className="text-[10px] uppercase tracking-widest text-primary-400/60 font-medium">
                      After
                    </span>
                    <p className="text-sm text-surface-200/80 mt-1">
                      {cat.after}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How we work */}
      <section className="py-16 px-6 border-t border-surface-700/30">
        <div className="mx-auto max-w-4xl">
          <h2 className="text-2xl font-bold text-white mb-3">How We Work</h2>
          <p className="text-surface-200/60 mb-12 max-w-2xl">
            A structured process that starts with understanding your
            business — not your tech stack. Technology decisions come after
            we know what needs to happen and why.
          </p>

          <div className="space-y-8">
            {processPhases.map((phase) => (
              <div
                key={phase.number}
                className="rounded-xl border border-surface-700/50 bg-surface-800/30 overflow-hidden"
              >
                <div className="flex items-center gap-4 p-6 border-b border-surface-700/30">
                  <div className="h-12 w-12 rounded-xl bg-primary-500/10 border border-primary-500/30 flex items-center justify-center text-lg font-bold text-primary-400 font-mono">
                    {phase.number}
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-white">
                      {phase.title}
                    </h3>
                    <p className="text-sm text-surface-200/50">
                      {phase.subtitle}
                    </p>
                  </div>
                </div>
                <div className="p-6">
                  <p className="text-sm text-surface-200/70 leading-relaxed mb-4">
                    {phase.description}
                  </p>
                  <ul className="space-y-2">
                    {phase.details.map((detail, i) => (
                      <li
                        key={i}
                        className="text-sm text-surface-200/60 flex items-start gap-2"
                      >
                        <CheckCircle
                          size={14}
                          className="text-primary-500/60 mt-0.5 flex-shrink-0"
                        />
                        {detail}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Measurable outcomes */}
      <section className="py-16 px-6 border-t border-surface-700/30">
        <div className="mx-auto max-w-4xl">
          <h2 className="text-2xl font-bold text-white mb-3">
            Measurable Outcomes
          </h2>
          <p className="text-surface-200/60 mb-10 max-w-2xl">
            Real numbers from production systems we operate — not projections,
            not benchmarks from a controlled environment.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              {
                metric: "50%",
                label: "API cost reduction",
                detail:
                  "Batch API processing cuts per-call cost in half vs synchronous requests. At thousands of calls per week, this is the difference between viable and prohibitive.",
                category: "Cost",
              },
              {
                metric: "$0",
                label: "LLM inference cost for high-volume enrichment",
                detail:
                  "Moved structured extraction to local models (Qwen3-30B on vLLM). Cloud APIs reserved for tasks that need frontier quality. Same pipeline, eliminated the largest recurring cost.",
                category: "Cost",
              },
              {
                metric: "11",
                label: "exact-cache layers across the pipeline",
                detail:
                  "Deterministic LLM calls (same input = same output) are cached with versioned namespaces. Hit tracking shows which stages benefit most. Repeat processing costs zero tokens.",
                category: "Cost",
              },
              {
                metric: "52%",
                label: "faster autonomous task execution",
                detail:
                  "Skip-synthesis convention: if there's no new data, don't call the LLM. Batch execution dropped from ~27s to ~13s. Synthesis runs on Claude Sonnet (cloud); reasoning tasks route to flagship models. Each skipped synthesis saves 10-20 seconds and avoids a cloud API call entirely.",
                category: "Latency",
              },
              {
                metric: "5x",
                label: "fewer WebSocket frames (edge ↔ brain)",
                detail:
                  "Token batching reduces per-token WebSocket overhead. ~100 frames/sec became ~20 frames/sec with 50ms batching. Adds at most 50ms latency — invisible to the user, massive reduction in network overhead.",
                category: "Latency",
              },
              {
                metric: "6.6x",
                label: "realtime TTS on $60 hardware",
                detail:
                  "Text-to-speech generates audio 6.6x faster than playback speed on an Orange Pi RK3588. No cloud round trip. Voice responses start before the sentence is fully synthesized.",
                category: "Latency",
              },
              {
                metric: "100%",
                label: "invoice accuracy (automated vs manual)",
                detail:
                  "10/10 exact match between automated and manually-created invoices in verification audit. 14 invoices, 12 customers, $16,228.27 — generated, rendered as PDF, and queued for approval in one cron run.",
                category: "Reliability",
              },
              {
                metric: "300",
                label: "database migrations without data loss",
                detail:
                  "Schema has evolved 300 times across every subsystem. Each migration is versioned, tested, and applied in sequence. Zero data loss incidents. The schema is a living document of every business requirement.",
                category: "Reliability",
              },
            ].map((item) => (
              <div
                key={item.label}
                className="rounded-xl border border-surface-700/50 bg-surface-800/30 p-5"
              >
                <div className="flex items-baseline gap-3 mb-2">
                  <span className="text-2xl font-black text-white">
                    {item.metric}
                  </span>
                  <span className="text-sm font-medium text-surface-200/80">
                    {item.label}
                  </span>
                </div>
                <p className="text-xs text-surface-200/50 leading-relaxed">
                  {item.detail}
                </p>
                <span className="inline-block mt-3 text-[10px] uppercase tracking-widest text-primary-400/50 font-medium">
                  {item.category}
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* When AI is the answer */}
      <section className="py-16 px-6 border-t border-surface-700/30">
        <div className="mx-auto max-w-4xl">
          <h2 className="text-2xl font-bold text-white mb-3">
            When AI Is the Answer — And When It Isn't
          </h2>
          <p className="text-surface-200/60 mb-10 max-w-2xl">
            We won't put AI where simple automation works. That's how you
            waste money and add unpredictability. Here's the honest breakdown.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="rounded-xl border border-primary-500/30 bg-primary-500/5 p-6">
              <div className="flex items-center gap-2 mb-4">
                <Bot size={20} className="text-primary-400" />
                <h3 className="font-semibold text-white">AI is the right tool when:</h3>
              </div>
              <ul className="space-y-3">
                {[
                  "The input is unstructured (free text, emails, documents)",
                  "The task requires judgment, not just rules (urgency, sentiment, intent)",
                  "The output needs to be generated, not looked up (summaries, drafts, responses)",
                  "The volume justifies the investment",
                  "You can validate the output before acting on it",
                ].map((item, i) => (
                  <li
                    key={i}
                    className="text-sm text-surface-200/70 flex items-start gap-2"
                  >
                    <CheckCircle
                      size={14}
                      className="text-primary-500 mt-0.5 flex-shrink-0"
                    />
                    {item}
                  </li>
                ))}
              </ul>
            </div>

            <div className="rounded-xl border border-surface-700/50 bg-surface-800/30 p-6">
              <div className="flex items-center gap-2 mb-4">
                <Zap size={20} className="text-accent-amber" />
                <h3 className="font-semibold text-white">
                  Simple automation is better when:
                </h3>
              </div>
              <ul className="space-y-3">
                {[
                  "The logic is deterministic (if X then Y — every time)",
                  "The data is already structured (database fields, form inputs)",
                  "The routing rules are known and stable",
                  "Speed matters more than nuance",
                  "The cost of AI per-execution doesn't justify the improvement",
                ].map((item, i) => (
                  <li
                    key={i}
                    className="text-sm text-surface-200/70 flex items-start gap-2"
                  >
                    <CheckCircle
                      size={14}
                      className="text-accent-amber/70 mt-0.5 flex-shrink-0"
                    />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <p className="text-sm text-surface-200/50 mt-6 text-center">
            Most workflows are 70% deterministic automation and 30% AI-augmented.
            We build both — and we know the difference.
          </p>
        </div>
      </section>

      {/* Proof */}
      <section className="py-16 px-6 border-t border-surface-700/30">
        <div className="mx-auto max-w-4xl">
          <h2 className="text-2xl font-bold text-white mb-3">
            Built and Running
          </h2>
          <p className="text-surface-200/60 mb-10 max-w-2xl">
            Systems we've built that run in production — not demos, not
            proofs of concept.
          </p>

          <div className="space-y-4">
            {proofPoints.map((proof) => (
              <div
                key={proof.title}
                className="rounded-xl border border-surface-700/50 bg-surface-800/30 p-6"
              >
                <h3 className="font-semibold text-white mb-2">
                  {proof.title}
                </h3>
                <p className="text-sm text-surface-200/70 leading-relaxed mb-3">
                  {proof.outcome}
                </p>
                <div className="flex flex-wrap gap-2">
                  {proof.tags.map((tag) => (
                    <span
                      key={tag}
                      className="text-[10px] text-primary-400/70 bg-primary-500/10 border border-primary-500/20 rounded-full px-2 py-0.5"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-6 border-t border-surface-700/30">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-2xl font-bold text-white mb-4">
            Have a Workflow That's Eating Your Team's Time?
          </h2>
          <p className="text-surface-200/60 mb-8 leading-relaxed">
            We'll map it, classify every step, and tell you exactly what
            should be automated, what needs AI, and what should stay manual.
            No commitment — just clarity on what's possible.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <a
              href="mailto:campfieldjuan24@gmail.com"
              className="inline-flex items-center gap-2 rounded-lg bg-primary-500 px-6 py-3 text-sm font-semibold text-surface-900 hover:bg-primary-400 transition-colors"
            >
              Start a Conversation
              <ArrowRight size={16} />
            </a>
            <Link
              to="/systems"
              className="inline-flex items-center gap-2 rounded-lg border border-surface-700 px-6 py-3 text-sm font-semibold text-surface-200 hover:border-surface-200/50 hover:text-white transition-colors"
            >
              See the Systems We've Built
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}
