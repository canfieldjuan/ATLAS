import { SeoHead } from "@/components/seo/SeoHead";
import { Link } from "react-router-dom";
import { ProjectCard } from "@/components/sections/ProjectCard";
import { allProjects } from "@/content/projects";
import { allInsights } from "@/content/insights";
import {
  Boxes,
  Network,
  Cpu,
  ShieldCheck,
  ArrowRight,
  BookOpen,
  ExternalLink,
} from "lucide-react";
import type { InsightPost } from "@/types";

const featuredSystems = allProjects
  .flatMap((project) =>
    project.subsystems.slice(0, 2).map((subsystem) => {
      const primaryStat = subsystem.stats?.[0];
      const relatedInsight = subsystem.relatedInsight
        ? allInsights.find((post) => post.slug === subsystem.relatedInsight)
        : undefined;
      return {
        project: project.title,
        name: subsystem.name,
        description:
          subsystem.description.includes(".") && subsystem.description.length > 170
            ? subsystem.description.split(". ").slice(0, 2).join(". ") + "."
            : subsystem.description,
        ownerStat: primaryStat
          ? `${primaryStat.value} ${primaryStat.label}`
          : "Production subsystem",
        projectSlug: project.slug,
        insightSlug: relatedInsight?.slug,
        insightTitle: relatedInsight?.title,
      };
    }),
  )
  .slice(0, 6);

const insightTypeDisplay: Record<InsightPost["type"], string> = {
  "case-study": "Case Study",
  "build-log": "Build Log",
  "industry-insight": "Industry Insight",
  lesson: "Lesson",
};

const toNumber = (value: string): number =>
  Number.parseInt(value.replace(/[^\d]/g, ""), 10) || 0;

const siteCrossLinks = [
  {
    to: "/systems",
    label: "System Architecture",
    blurb: "See how Atlas domains, tasks, and data flows connect end-to-end.",
  },
  {
    to: "/insights",
    label: "Insights Archive",
    blurb: "Case studies, build logs, and hard-won lessons from production.",
  },
  {
    to: "/framework",
    label: "Systems Framework",
    blurb: "The discipline that turns AI outputs into dependable products.",
  },
  {
    to: "/services",
    label: "Service Model",
    blurb: "How I design the right balance of AI, automation, and humans.",
  },
];

const relatedInsights = allInsights
  .filter((post) => post.project)
  .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
  .slice(0, 6);

const externalProofLinks = [
  {
    label: "churnsignals.co",
    url: "https://churnsignals.co",
    blurb:
      "Live product surface with public campaign and reporting touchpoints, plus selected examples that validate the same reliability principles showcased here.",
    links: [
      {
        label: "Freshdesk deep dive",
        url: "https://churnsignals.co/blog/freshdesk-deep-dive",
      },
      {
        label: "Why teams leave Azure (2026)",
        url: "https://churnsignals.co/blog/why-teams-leave-azure-2026-03",
      },
      {
        label: "HubSpot vs Power BI (2026)",
        url: "https://churnsignals.co/blog/hubspot-vs-power-bi-2026-04",
      },
    ],
  },
  {
    label: "atlasbizintel.co",
    url: "https://www.atlasbizintel.co/",
    blurb: "Live business intelligence product context and public entry points.",
  },
];

const platformSnapshot = [
  {
    icon: Boxes,
    label: "Production Platforms",
    value: `${allProjects.length}`,
  },
  {
    icon: Network,
    label: "Total Subsystems",
    value: `${allProjects.reduce(
      (acc, project) => acc + project.subsystems.length,
      0,
    )}`,
  },
  {
    icon: Cpu,
    label: "Production Scale Signals",
    value: `${allProjects.reduce(
      (acc, project) =>
        acc +
        project.stats.reduce((subAcc, stat) => {
          const isCoreScaleSignal =
            stat.label.includes("Tools") ||
            stat.label.includes("Endpoints") ||
            stat.label.includes("Servers") ||
            stat.label.includes("Tasks") ||
            stat.label.includes("Migrations");
          return isCoreScaleSignal ? subAcc + toNumber(stat.value) : subAcc;
        }, 0),
      0,
    )}+`,
  },
  {
    icon: ShieldCheck,
    label: "Unique Stack Components",
    value: `${new Set(allProjects.flatMap((project) => project.techStack)).size}`,
  },
];

export default function Projects() {
  return (
    <>
      <SeoHead
        meta={{
          title: "Projects",
          description:
            "Production AI systems — Atlas (100+ MCP tools, autonomous orchestration, edge compute) and FineTuneLab.ai (225 API endpoints, LLM fine-tuning, GraphRAG).",
          keywords: [
            "production ai systems",
            "AI systems portfolio",
            "quality gates",
            "autonomous orchestration",
            "AI reliability",
          ],
          canonicalPath: "/projects",
          jsonLd: {
            "@context": "https://schema.org",
            "@type": "CollectionPage",
            name: "Projects",
            description:
              "Production AI systems and subsystems documented with architecture decisions, validation gates, and related insights.",
            url: new URL("/projects", window.location.origin).toString(),
            mainEntity: {
              "@type": "ItemList",
              itemListElement: allProjects.map((project, index) => ({
                "@type": "ListItem",
                position: index + 1,
                name: project.title,
                url: new URL(
                  `/projects/${project.slug}`,
                  window.location.origin,
                ).toString(),
              })),
            },
          },
        }}
      />

      <section className="py-16 px-6">
        <div className="mx-auto max-w-4xl">
          <h1 className="text-4xl font-bold text-white mb-4">Projects</h1>
          <p className="text-surface-200/70 mb-12 max-w-2xl">
            Each project below is a production system — not a tutorial repo.
            Click through to see architecture decisions, the tedious parts, and
            what AI-first development actually looks like.
          </p>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-12">
            {platformSnapshot.map((item) => {
              const Icon = item.icon;
              return (
                <div
                  key={item.label}
                  className="rounded-xl border border-surface-700/50 bg-surface-800/30 px-4 py-5"
                >
                  <Icon size={18} className="text-primary-400 mb-2" />
                  <div className="text-2xl font-bold text-white">
                    {item.value}
                  </div>
                  <div className="text-[11px] text-surface-200/50 mt-1">
                    {item.label}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-12">
            {siteCrossLinks.map((link) => (
              <Link
                to={link.to}
                key={link.to}
                className="rounded-xl border border-surface-700/50 bg-surface-800/30 p-4 hover:border-primary-500/40 hover:bg-surface-800/50 transition-all"
              >
                <div className="text-sm font-semibold text-white mb-1">
                  {link.label}
                </div>
                <p className="text-xs text-surface-200/60">{link.blurb}</p>
              </Link>
            ))}
          </div>

          <div className="rounded-xl border border-primary-500/20 bg-primary-500/5 p-6 mb-12">
            <h2 className="text-2xl font-bold text-white mb-4">
              What This Portfolio Actually Contains
            </h2>
            <p className="text-surface-200/70 mb-6 max-w-2xl">
              We only showcase two flagship platforms, but each one ships a stack
              of production subsystems with independent value: acquisition,
              intelligence, invoicing, routing, and monitoring.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {featuredSystems.map((system) => (
                <div
                  key={`${system.project}-${system.name}`}
                  className="rounded-lg border border-surface-700/50 bg-surface-800/30 px-4 py-4"
                >
                  <Link
                    to={`/projects/${system.projectSlug}`}
                    className="grid gap-2 group"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <h3 className="text-sm font-semibold text-white group-hover:text-primary-400 transition-colors">
                        {system.name}
                      </h3>
                      <span className="text-[10px] text-primary-400/70 uppercase tracking-widest">
                        {system.project}
                      </span>
                    </div>
                    <p className="text-xs text-surface-200/60 leading-relaxed">
                      {system.description}
                    </p>
                    <p className="text-xs text-primary-400 font-medium">
                      {system.ownerStat}
                    </p>
                  </Link>
                  {system.insightSlug && system.insightTitle && (
                    <Link
                      to={`/insights/${system.insightSlug}`}
                      className="mt-2 inline-flex items-center gap-1.5 text-xs text-surface-200/70 hover:text-primary-400 transition-colors"
                    >
                      <BookOpen size={12} />
                      Read related insight: {system.insightTitle}
                      <ArrowRight size={12} />
                    </Link>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-xl border border-primary-500/20 bg-primary-500/5 p-6 mb-12">
            <h2 className="text-2xl font-bold text-white mb-4">
              Insights That Validate These Systems
            </h2>
            <p className="text-surface-200/70 mb-6 max-w-2xl">
              Each linked post expands one slice of production design: quality
              gates, calibration, output structure, cost discipline, and
              operational risk control.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {relatedInsights.map((post) => (
                <Link
                  to={`/insights/${post.slug}`}
                  key={post.slug}
                  className="rounded-lg border border-surface-700/50 bg-surface-800/30 px-4 py-4 hover:border-primary-500/40 hover:bg-surface-800/50 transition-all"
                >
                  <div className="flex items-center justify-between gap-3 mb-2">
                    <span className="inline-flex items-center gap-1.5 text-[10px] text-primary-400/80 uppercase tracking-widest">
                      <BookOpen size={12} />
                      {insightTypeDisplay[post.type]}
                    </span>
                    <span className="text-[10px] text-surface-200/50">
                      {new Date(post.date).toLocaleDateString("en-US", {
                        month: "short",
                        day: "numeric",
                        year: "numeric",
                      })}
                    </span>
                  </div>
                  <h3 className="text-sm font-semibold text-white leading-snug mb-2">
                    {post.title}
                  </h3>
                  <p className="text-xs text-surface-200/60 leading-relaxed">
                    {post.description}
                  </p>
                  <div className="mt-3 flex items-center justify-between gap-3">
                    <span className="text-[10px] text-surface-200/50">
                      {post.project === "finetunelab"
                        ? "FineTuneLab.ai"
                        : "Atlas"}
                    </span>
                    <span className="text-xs text-primary-400/80 inline-flex items-center gap-1">
                      Read full writeup
                      <ArrowRight size={12} />
                    </span>
                  </div>
                </Link>
              ))}
            </div>
          </div>

          <div className="rounded-xl border border-accent-cyan/20 bg-accent-cyan/5 p-6 mb-12">
            <h2 className="text-2xl font-bold text-white mb-2">
              Live Product References
            </h2>
            <p className="text-surface-200/70 mb-6 max-w-2xl">
              External systems and references for quick verification of this work in
              production.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {externalProofLinks.map((resource) => (
                <div
                  key={resource.url}
                  className="rounded-lg border border-accent-cyan/30 bg-surface-800/20 px-4 py-4 hover:border-accent-cyan/50 hover:bg-surface-800/40 transition-all"
                >
                  <div className="flex items-center justify-between gap-3 mb-2">
                    <span className="text-sm font-semibold text-white">
                      {resource.label}
                    </span>
                    <ExternalLink size={14} className="text-surface-200/40" />
                  </div>
                  <p className="text-xs text-surface-200/60 leading-relaxed mb-3">
                    {resource.blurb}
                  </p>
                  <a
                    href={resource.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-accent-cyan/80 inline-flex items-center gap-1"
                  >
                    Open site
                    <ArrowRight size={12} />
                  </a>
                  {resource.links?.length ? (
                    <ul className="mt-4 space-y-2">
                      {resource.links.map((proofLink) => (
                        <li key={proofLink.url}>
                          <a
                            href={proofLink.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs text-surface-200/80 hover:text-accent-cyan transition-colors inline-flex items-center gap-1.5"
                          >
                            {proofLink.label}
                            <ArrowRight size={12} />
                          </a>
                        </li>
                      ))}
                    </ul>
                  ) : null}
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-xl border border-surface-700/50 bg-surface-800/20 p-4 mb-12">
            <p className="text-sm text-surface-200/70">
              Prefer direct evidence? Open the{" "}
              <Link to="/insights" className="text-primary-400 underline">
                full insight archive
              </Link>{" "}
              to find more subsystem-level notes, lessons, and implementation
              notes.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {allProjects.map((project) => (
              <ProjectCard key={project.slug} project={project} />
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
