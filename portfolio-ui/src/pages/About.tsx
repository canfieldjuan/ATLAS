import { SeoHead } from "@/components/seo/SeoHead";
import { Link } from "react-router-dom";
import {
  ArrowRight,
  Brain,
  Cpu,
  LineChart,
  Network,
  ShieldCheck,
  Target,
  Workflow,
} from "lucide-react";

const platformProof = [
  { value: "2", label: "Production Platforms" },
  { value: "11", label: "MCP Servers (Atlas)" },
  { value: "300+", label: "End-to-End Tools" },
  { value: "56", label: "Vendors Tracked" },
  { value: "25K+", label: "Reviews Enriched" },
  { value: "94ms", label: "NPU Inference" },
];

const operatingPrinciples = [
  {
    icon: Workflow,
    title: "Problem Framing Before Model Framing",
    body: "Map every workflow into deterministic rules, AI judgment points, and approved automation boundaries. Reliability comes from control, not model hype.",
  },
  {
    icon: ShieldCheck,
    title: "Quality Gates as Product Design",
    body: "Schema validation, confidence scoring, repair loops, and approvals are baked into architecture, not added after deployment.",
  },
  {
    icon: Target,
    title: "Measurable Outcomes",
    body: "I measure by throughput, error rates, cost drift, correction frequency, and business impact. Not by demo outputs.",
  },
  {
    icon: Network,
    title: "Graceful Degradation",
    body: "When uncertainty is high, systems shift safely: fail-open patterns, safe routing, and explicit fallback behavior keep workflows resilient.",
  },
];

const audienceSignals = [
  {
    title: "For Repetitive Business Work",
    points: [
      "Customer onboarding and support workflows",
      "Marketing campaign sequencing",
      "CRM and lead-scoring automation",
      "B2B churn intelligence operations",
      "Scheduling, invoicing, and notifications",
    ],
  },
  {
    title: "For AI System Teams",
    points: [
      "How to split deterministic logic, AI judgment, and AI output",
      "How to design approval gates and exception handling",
      "How to control cost when fallbacks and retries multiply",
      "How to keep outputs traceable and auditable",
    ],
  },
];

const externalProofLinks = [
  {
    title: "churnsignals.co",
    url: "https://www.churnsignals.co/",
    notes:
      "Public product surface reflecting campaign intelligence and operational reporting principles.",
  },
  {
    title: "atlasbizintel.co",
    url: "https://www.atlasbizintel.co/",
    notes:
      "Business intelligence surface aligned with the same system approach and reporting workflows.",
  },
];

const quickActions = [
  {
    icon: Brain,
    title: "Process Framework",
    text: "How the method turns fuzzy prompts into controlled operational systems.",
    to: "/framework",
    cta: "Read Framework",
  },
  {
    icon: Cpu,
    title: "Project Detail",
    text: "Atlas and FineTuneLab evidence with architecture and implementation tradeoffs.",
    to: "/projects",
    cta: "Explore Projects",
  },
  {
    icon: LineChart,
    title: "Production Lessons",
    text: "Build logs and post-mortems showing what failed, why, and what was changed.",
    to: "/insights",
    cta: "Open Insights",
  },
];

export default function About() {
  return (
    <>
      <SeoHead
        meta={{
          title: "About — AI Systems Architect",
          description:
            "AI systems architect focused on dependable business automation: production reliability, quality gates, reproducible AI output, and automation for revenue-impacting workflows.",
          keywords: [
            "AI systems architect",
            "production AI reliability",
            "quality gates for AI systems",
            "business process automation engineering",
          ],
          canonicalPath: "/about",
          jsonLd: {
            "@context": "https://schema.org",
            "@type": "AboutPage",
            name: "About Juan Canfield",
            description:
              "AI systems architect focused on production-grade automation for repetitive business workflows.",
            mainEntity: {
              "@type": "Person",
              name: "Juan Canfield",
              jobTitle: "AI Systems Architect",
              sameAs: [
                "https://github.com/canfieldjuan/atlas-portfolio",
                "https://www.linkedin.com/in/juan-canfield-9b2a733b5/",
              ],
            },
          },
        }}
      />

      <section className="py-16 px-6">
        <div className="mx-auto max-w-4xl">
          <header className="mb-12">
            <h1 className="text-4xl font-bold text-white mb-4">
              About Juan Canfield
            </h1>
            <p className="text-surface-200/70 max-w-2xl">
              I build production AI systems for real operations: quality
              controlled, cost-aware, and measurable. Not wrappers. Not demos.
              I build systems that survive the real world: missing inputs, noisy
              data, outages, and silent drift.
            </p>
          </header>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-12">
            {platformProof.map((metric) => (
              <div
                key={metric.label}
                className="rounded-xl border border-surface-700/50 bg-surface-800/30 px-3 py-4"
              >
                <div className="text-xl font-bold text-white">{metric.value}</div>
                <div className="text-[11px] text-surface-200/50 mt-1">
                  {metric.label}
                </div>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
            <section>
              <h2 className="text-2xl font-bold text-white mb-4">
                Positioning
              </h2>
              <p className="text-surface-200/80 mb-4">
                The distinction I make is simple: many teams ship AI features.
                I ship dependable AI systems. I treat AI as an execution layer
                and build the controls that make it safe, explainable, and
                repeatable.
              </p>
              <p className="text-surface-200/80">
                My work spans software architecture, operations, and business
                process design. The result is fewer fragile automations and more
                systems that run correctly under pressure.
              </p>
            </section>

            <section className="rounded-xl border border-primary-500/20 bg-primary-500/5 p-5">
              <h2 className="text-2xl font-bold text-white mb-4">
                What This Portfolio Documents
              </h2>
              <p className="text-surface-200/80 mb-4">
                Evidence from two production stacks:
              </p>
              <div className="space-y-3">
                <Link
                  to="/projects/atlas"
                  className="block rounded-lg border border-surface-700/50 px-3 py-3 bg-surface-800/25 hover:border-primary-400/50 transition-colors"
                >
                  <span className="text-primary-400 font-medium flex items-center gap-2">
                    Atlas <ArrowRight size={14} />
                  </span>
                  <p className="text-xs text-surface-200/70 mt-1">
                    Intelligence platform with autonomous workflow orchestration,
                    churn pipeline processing, and edge + cloud architecture.
                  </p>
                </Link>
                <Link
                  to="/projects/finetunelab"
                  className="block rounded-lg border border-surface-700/50 px-3 py-3 bg-surface-800/25 hover:border-primary-400/50 transition-colors"
                >
                  <span className="text-primary-400 font-medium flex items-center gap-2">
                    FineTuneLab.ai <ArrowRight size={14} />
                  </span>
                  <p className="text-xs text-surface-200/70 mt-1">
                    End-to-end LLM fine-tuning platform with multi-provider
                    adapters, structured training workflows, and evaluation
                    controls.
                  </p>
                </Link>
              </div>
            </section>
          </div>

          <section className="mb-12">
            <h2 className="text-2xl font-bold text-white mb-4">
              Operating Principles
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {operatingPrinciples.map((principle) => {
                const Icon = principle.icon;
                return (
                  <div
                    key={principle.title}
                    className="rounded-xl border border-surface-700/50 bg-surface-800/30 p-5"
                  >
                    <div className="flex items-start gap-3">
                      <Icon size={18} className="text-primary-400 mt-0.5" />
                      <div>
                        <h3 className="font-semibold text-white mb-2">
                          {principle.title}
                        </h3>
                        <p className="text-sm text-surface-200/75 leading-relaxed">
                          {principle.body}
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-12">
            {audienceSignals.map((audience) => (
              <div
                key={audience.title}
                className="rounded-xl border border-surface-700/50 bg-surface-800/30 p-5"
              >
                <h3 className="text-sm uppercase tracking-widest text-primary-400/80 mb-2">
                  {audience.title}
                </h3>
                <ul className="space-y-2 text-sm text-surface-200/75">
                  {audience.points.map((point) => (
                    <li key={point} className="flex items-start gap-2">
                      <span className="text-primary-400 mt-1">▸</span>
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </section>

          <section className="rounded-xl border border-surface-700/50 bg-surface-800/20 p-6 mb-12">
            <h2 className="text-2xl font-bold text-white mb-4">
              Validation Path
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {quickActions.map((item) => {
                const Icon = item.icon;
                return (
                  <Link
                    to={item.to}
                    key={item.title}
                    className="rounded-lg border border-surface-700/50 bg-surface-800/30 p-4 hover:border-primary-400/50 transition-colors"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <Icon size={18} className="text-primary-400" />
                      <span className="text-sm font-semibold text-white">
                        {item.title}
                      </span>
                    </div>
                    <p className="text-xs text-surface-200/70 mb-2">
                      {item.text}
                    </p>
                    <span className="text-[11px] text-primary-400 flex items-center gap-1">
                      {item.cta} <ArrowRight size={12} />
                    </span>
                  </Link>
                );
              })}
            </div>
          </section>

          <section className="mb-12">
            <h2 className="text-2xl font-bold text-white mb-4">
              External Proof Points
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {externalProofLinks.map((site) => (
                <a
                  key={site.title}
                  href={site.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block rounded-xl border border-surface-700/50 bg-surface-800/20 p-5 hover:border-primary-400/50 transition-colors"
                >
                  <span className="text-primary-400 font-semibold text-sm">
                    {site.title}
                  </span>
                  <p className="text-sm text-surface-200/75 mt-1">
                    {site.notes}
                  </p>
                </a>
              ))}
            </div>
          </section>

          <section className="rounded-xl border border-primary-500/20 bg-primary-500/5 p-6">
            <h2 className="text-2xl font-bold text-white mb-3">
              Start with one workflow
            </h2>
            <p className="text-surface-200/80 mb-4">
              The strongest AI implementations start with one repetitive
              bottleneck and a measurable outcome. Then you add guardrails,
              tracing, and approval controls before scaling automation outward.
            </p>
            <div className="text-sm text-surface-200/75 flex flex-wrap gap-3 items-center">
              <span>If that matches your operations:</span>
              <Link
                to="/services"
                className="inline-flex items-center gap-1.5 text-primary-400 hover:text-primary-300"
              >
                How we work <ArrowRight size={14} />
              </Link>
            </div>
          </section>
        </div>
      </section>
    </>
  );
}
