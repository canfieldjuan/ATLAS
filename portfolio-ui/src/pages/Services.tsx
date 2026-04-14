import { Link } from "react-router-dom";
import { SeoHead } from "@/components/seo/SeoHead";
import { allInsights } from "@/content/insights";
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
  BookOpen,
  ExternalLink,
  HelpCircle,
} from "lucide-react";
import type { InsightPost } from "@/types";

type ProcessPhase = {
  number: string;
  title: string;
  subtitle: string;
  description: string;
  details: string[];
  outputs: string[];
  proofSlugs: string[];
};

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

const processPhases: ProcessPhase[] = [
  {
    number: "01",
    title: "Discovery and Process Capture",
    subtitle: "Map repetitive work before proposing any AI",
    description:
      "Before touching technology, we map the real process end-to-end. We document each human decision point, artifact handoff, and failure condition so automation is built on evidence, not assumptions.",
    details: [
      "Stakeholder interviews with operators, not just stakeholders, to capture edge cases and exception handling",
      "Task decomposition: trigger → action → review point → outcome for each repeatable workflow",
      "Data inventory (CRM, email platform, calendar, analytics, support tools) and schema-level validation",
      "Volume and variance baseline (daily/monthly peaks, burst windows, retry frequency, error rate by channel)",
      "Definition of what 'good output' and 'safe output' means by business owner, with acceptance criteria",
    ],
    outputs: [
      "Workflow map (decision tree + handoff map)",
      "Baseline KPI model (throughput, defect rate, cycle time)",
      "Scope exclusions (what we intentionally do not automate yet)",
    ],
    proofSlugs: ["its-not-a-model-problem-its-a-context-problem"],
  },
  {
    number: "02",
    title: "Classification and Control-Point Design",
    subtitle: "Decide what is deterministic logic vs. AI judgment",
    description:
      "Most operations are repeatable logic with a few high-judgment moments. We split the process into deterministic, AI-augmented, and AI-automated layers, then assign explicit control points where humans keep authority.",
    details: [
      "Deterministic layer: routing, normalization, enrichment lookups, deduplication, and scheduling",
      "AI-judgment layer: sentiment, intent, priority, and anomaly scoring where nuance changes outcomes",
      "Generation layer: campaign drafts, summaries, and follow-up recommendations in constrained formats",
      "Guardrails: human approvals and confidence thresholds before any external action",
      "Rollback behavior: how the system recovers safely if data is missing, noisy, or contradictory",
    ],
    outputs: [
      "Automation split map (rule-based / AI / human)",
      "Gating policy (confidence thresholds, exceptions, escalation paths)",
      "Approval matrix for campaign, CRM, and payment-side actions",
    ],
    proofSlugs: [
      "seven-patterns-deterministic-llm-systems",
      "prompting-is-a-science-rag-is-harder-than-you-think",
    ],
  },
  {
    number: "03",
    title: "Campaign and Workflow Blueprint",
    subtitle: "Architecture before code",
    description:
      "With the control model fixed, we create a system blueprint that is specific to the process being automated (marketing cadence, email campaigns, lead routing, reporting, or invoicing). The blueprint makes ownership, sequencing, and failure behavior explicit.",
    details: [
      "Campaign flow design: lead qualification → campaign selection → draft assembly → review queue → send window",
      "Data contracts: source system schema, required fields, and immutable audit fields retained at each step",
      "Context retention strategy so sequence history, vendor notes, and engagement signals remain coherent",
      "Fallback and recovery for campaign drops, API outages, and ambiguous model output",
      "Cost and latency model with expected volume at current and 3x growth scenarios",
    ],
    outputs: [
      "System architecture diagram (nodes, queues, triggers, outputs)",
      "Sequence-state model (what happened, by whom, and why)",
      "Runbook for campaign and operations exceptions",
    ],
    proofSlugs: ["replacing-state-machine-with-llm-tools", "deterministic-infrastructure-for-non-deterministic-intelligence"],
  },
  {
    number: "04",
    title: "Build in Controlled Layers",
    subtitle: "Incremental delivery, not a big bang",
    description:
      "We implement in layers so value appears early and risk stays bounded. Deterministic automation launches first, AI-augmented steps follow, and AI-executed steps are released only after repeated validation.",
    details: [
      "Layer 1: deterministic pipelines (routing, normalization, sync) for immediate time savings",
      "Layer 2: AI-assisted extraction/classification with strict review policies",
      "Layer 3: AI-generated campaigns/notes/outputs with gating and signed approval state",
      "Integration across existing systems (CRM, ESP, calendar, BI) without forced platform migrations",
      "Audit-ready observability from ingestion to outbound action, including queue depth and worker behavior",
    ],
    outputs: [
      "MVP workflow deployed in parallel with existing manual process",
      "Release plan by phase with acceptance criteria for each layer",
      "Automated test coverage for parsing, routing, and decision paths",
    ],
    proofSlugs: [
      "building-b2b-churn-intelligence-pipeline",
      "making-autonomous-ai-tasks-fail-safely",
      "cloud-vs-local-llm-cost-quality-tradeoff",
    ],
  },
  {
    number: "05",
    title: "Validate and Calibrate",
    subtitle: "Prove it works before you depend on it",
    description:
      "We do not move from pilot to full automation without production-style validation. This is where reliability is measured over time, not in a single demo.",
    details: [
      "Schema and policy checks on every execution (format, required fields, timing, approvals)",
      "Campaign A/B and holdout checks before changing outreach logic at scale",
      "Human correction tracking: every override is turned into a retraining or rule adjustment signal",
      "Cost, latency, and failure dashboards visible per subsystem and per campaign",
      "Post-launch governance review: what changed, what broke, and what will be hardened next",
    ],
    outputs: [
      "Control dashboard (quality, cost, latency, conversion outcomes)",
      "Correction log and model/rule update backlog",
      "Quarterly reliability and automation maturity assessment",
    ],
    proofSlugs: [
      "testing-llm-systems-is-expensive",
      "your-fallback-path-is-your-cost-path",
    ],
  },
  {
    number: "06",
    title: "Operate, Iterate, Expand",
    subtitle: "Keep systems stable as volume and variants increase",
    description:
      "After launch, we run workflows as products: monitor, tune, and expand only when constraints are met. Repetitive tasks keep evolving, so the system is measured and adjusted, not set once and forgotten.",
    details: [
      "Campaign fatigue and deliverability controls on outbound email programs",
      "Drift detection for changed lead quality, seasonal campaign patterns, and tool behavior changes",
      "Periodic reclassification of tasks as deterministic or AI-dependent based on outcomes",
      "Operational runbooks for incident response and safe rollback in under an hour",
      "Planned expansion checklist for adjacent workflows (support, onboarding, reporting, outreach)",
    ],
    outputs: [
      "Monthly review pack with outcome deltas and cost/quality trade-offs",
      "Iteration backlog tied to concrete metric changes",
      "Expansion candidate list with confidence and effort scores",
    ],
    proofSlugs: ["the-tax-of-third-party-apps-vs-custom-code"],
  },
];

const servicesCrossLinks = [
  {
    to: "/projects",
    title: "Project Deep Dives",
    blurb:
      "See how each production subsystem was implemented, measured, and hardened over time.",
  },
  {
    to: "/systems",
    title: "System Map",
    blurb:
      "Review the end-to-end architecture before committing to a particular automation stack.",
  },
  {
    to: "/framework",
    title: "Implementation Framework",
    blurb:
      "Understand the discipline behind quality gates, governance, and AI/automation boundaries.",
  },
  {
    to: "/insights",
    title: "Proof Archive",
    blurb:
      "Read build logs, case studies, and lessons from production runs with real failures and fixes.",
  },
];

const externalProofLinks = [
  {
    brand: "churnsignals.co",
    blurb:
      "Public campaign and reporting touchpoints that reinforce the same production reliability principles.",
    links: [
      {
        title: "Freshdesk deep dive",
        url: "https://churnsignals.co/blog/freshdesk-deep-dive",
      },
      {
        title: "Why teams leave Azure (2026)",
        url: "https://churnsignals.co/blog/why-teams-leave-azure-2026-03",
      },
      {
        title: "HubSpot vs Power BI (2026)",
        url: "https://churnsignals.co/blog/hubspot-vs-power-bi-2026-04",
      },
    ],
  },
  {
    brand: "atlasbizintel.co",
    blurb:
      "Live business intelligence surface with campaign and analysis patterns aligned to this same operating model.",
    links: [
      {
        title: "Business intelligence homepage",
        url: "https://www.atlasbizintel.co/",
      },
    ],
  },
];

const whatWeWorkFaq = [
  {
    question: "Do you automate every repetitive task with AI?",
    answer:
      "No. We start with process mapping and split each step into deterministic logic, AI-supported judgment, and AI execution only where it is measurable and controlled.",
  },
  {
    question: "How are marketing email campaigns handled safely?",
    answer:
      "Campaigns are treated as stateful workflows with suppression, cadence, content generation, approval, and reply capture gates. A model can draft and score, but send decisions stay policy-driven with clear rollback behavior.",
  },
  {
    question: "What is your gating model before external action?",
    answer:
      "We use confidence thresholds, human approval paths, and auditable exception rules for every external action path, then tune thresholds based on observed drift and override patterns.",
  },
  {
    question: "How do you prevent hidden drift in long-running automations?",
    answer:
      "Every workflow has a recurring validation cadence: schema checks, quality sampling, and drift alerts on cost, accuracy, and output behavior. Drift triggers controlled re-training or policy updates.",
  },
];

const faqJsonLd = {
  "@context": "https://schema.org",
  "@type": "FAQPage",
  mainEntity: whatWeWorkFaq.map((entry) => ({
    "@type": "Question",
    name: entry.question,
    acceptedAnswer: {
      "@type": "Answer",
      text: entry.answer,
    },
  })),
};

const phaseEvidence = (phase: ProcessPhase) =>
  phase.proofSlugs
    .map((slug) => allInsights.find((post) => post.slug === slug))
    .filter((post): post is InsightPost => Boolean(post));

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
          keywords: [
            "workflow automation",
            "AI systems architect",
            "business process automation",
            "marketing automation",
            "email campaign automation",
            "quality gates",
            "deterministic logic",
            "AI-augmented workflows",
          ],
          canonicalPath: "/services",
          jsonLd: [
            {
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
              hasOfferCatalog: {
                "@type": "OfferCatalog",
                name: "Workflow Systems Services",
                itemListElement: [
                  {
                    "@type": "Offer",
                    itemOffered: {
                      "@type": "Service",
                      name: "Automation Design",
                      description: "Process discovery and deterministic workflow mapping.",
                    },
                  },
                  {
                    "@type": "Offer",
                    itemOffered: {
                      "@type": "Service",
                      name: "AI-Augmented Production Build",
                      description:
                        "Campaign and business process automation with validation gates and approvals.",
                    },
                  },
                ],
              },
            },
            faqJsonLd,
          ],
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
          <p className="text-sm text-surface-200/60 max-w-2xl mx-auto leading-relaxed mt-4">
            From prompt-first automation to production systems: AI where it adds
            value, deterministic logic where it saves risk, and guardrails where
            failure is unacceptable.
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

      {/* Where this process lives */}
      <section className="py-16 px-6 border-t border-surface-700/30">
        <div className="mx-auto max-w-4xl">
          <h2 className="text-2xl font-bold text-white mb-3">
            Explore the Full Workflow Stack
          </h2>
          <p className="text-surface-200/60 mb-6 max-w-2xl">
            If you're deciding where to start, begin with architecture, then move to
            systems, projects, and evidence. Each page is written to support the same
            operating model.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {servicesCrossLinks.map((link) => (
              <Link
                key={link.to}
                to={link.to}
                className="rounded-xl border border-surface-700/50 bg-surface-800/30 p-4 hover:border-primary-500/40 hover:bg-surface-800/50 transition-all"
              >
                <div className="text-base font-semibold text-white mb-1">
                  {link.title}
                </div>
                <p className="text-sm text-surface-200/60">{link.blurb}</p>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* How we work */}
      <section className="py-16 px-6 border-t border-surface-700/30">
        <div className="mx-auto max-w-4xl">
          <h2 className="text-2xl font-bold text-white mb-3">How We Work</h2>
          <p className="text-surface-200/60 mb-12 max-w-2xl">
            A production workflow design sequence for repetitive business tasks:
            capture the operational reality first, split deterministic vs AI
            responsibilities, then add automation layers that can be audited
            per campaign, per workflow, and per business function (marketing,
            outreach, invoicing, support, reporting).
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
                  {phase.outputs && phase.outputs.length > 0 && (
                    <div className="mt-4 rounded-lg border border-surface-700/40 bg-surface-900/40 p-4">
                      <p className="text-xs text-surface-200/70 uppercase tracking-widest mb-2">
                        Deliverables from this phase
                      </p>
                      <ul className="space-y-2">
                        {phase.outputs.map((output) => (
                          <li
                            key={output}
                            className="text-sm text-surface-200/65 flex items-start gap-2"
                          >
                            <CheckCircle
                              size={12}
                              className="text-primary-400/70 mt-0.5 flex-shrink-0"
                            />
                            {output}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {phase.proofSlugs && phase.proofSlugs.length > 0 && (
                    <div className="mt-4 rounded-lg border border-surface-700/40 bg-surface-900/40 p-4">
                      <p className="text-xs text-surface-200/70 uppercase tracking-widest mb-2">
                        Proof from production writeups
                      </p>
                      <ul className="space-y-2">
                        {phaseEvidence(phase).map((proof) => (
                          <li
                            key={proof.slug}
                            className="text-sm text-surface-200/65 flex items-start gap-2"
                          >
                            <BookOpen
                              size={12}
                              className="text-primary-400/70 mt-0.5 flex-shrink-0"
                            />
                            <Link
                              to={`/insights/${proof.slug}`}
                              className="text-primary-400 hover:text-primary-300 transition-colors"
                            >
                              {proof.title}
                            </Link>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How we work FAQ */}
      <section className="py-16 px-6 border-t border-surface-700/30">
        <div className="mx-auto max-w-4xl">
          <div className="flex items-center gap-3 mb-4">
            <HelpCircle size={20} className="text-primary-400" />
            <h2 className="text-2xl font-bold text-white">Process Questions</h2>
          </div>
          <p className="text-surface-200/60 mb-8 max-w-2xl">
            The answers below are the practical checkpoints you should use to
            evaluate whether a prospective automation engagement is realistic and
            trustworthy.
          </p>
          <div className="space-y-4">
            {whatWeWorkFaq.map((item) => (
              <div
                key={item.question}
                className="rounded-xl border border-surface-700/50 bg-surface-800/30 p-5"
              >
                <p className="font-semibold text-white mb-2">{item.question}</p>
                <p className="text-sm text-surface-200/70 leading-relaxed">
                  {item.answer}
                </p>
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

          <div className="mt-8 pt-8 border-t border-surface-700/30">
            <h3 className="text-lg font-semibold text-white mb-4">
              External Production References
            </h3>
            <p className="text-sm text-surface-200/65 mb-4 max-w-2xl">
              Live surfaces connected to these same automation and campaign systems.
              Useful when you want to validate the operating model outside this site.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {externalProofLinks.map((resource) => (
                <div
                  key={resource.brand}
                  className="rounded-xl border border-surface-700/50 bg-surface-800/30 p-5"
                >
                  <p className="font-semibold text-white mb-2">
                    {resource.brand}
                  </p>
                  <p className="text-sm text-surface-200/70 mb-4">
                    {resource.blurb}
                  </p>
                  <div className="space-y-2">
                    {resource.links.map((link) => (
                      <a
                        key={link.title}
                        href={link.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-start gap-2 text-sm text-primary-400 hover:text-primary-300 transition-colors"
                      >
                        <ExternalLink
                          size={12}
                          className="text-primary-400/70 mt-0.5 flex-shrink-0"
                        />
                        <span>{link.title}</span>
                      </a>
                    ))}
                  </div>
                </div>
              ))}
            </div>
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
