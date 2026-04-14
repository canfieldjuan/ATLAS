import { useState } from "react";
import { SeoHead } from "@/components/seo/SeoHead";
import { systemDomains, crossDomainBridges } from "@/content/systems";
import type { SystemDomain, Subsystem } from "@/content/systems";
import * as Icons from "lucide-react";
import { ChevronDown, ChevronRight, ArrowRight } from "lucide-react";

function DomainIcon({ name, className }: { name: string; className?: string }) {
  const Icon =
    (Icons as unknown as Record<string, Icons.LucideIcon>)[name] ?? Icons.Box;
  return <Icon className={className} size={20} />;
}

function SubsystemCard({ sub }: { sub: Subsystem }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-lg border border-surface-700/40 bg-surface-800/20">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-start gap-3 p-4 text-left hover:bg-surface-800/40 transition-colors rounded-lg"
      >
        <div className="mt-0.5 text-surface-200/40">
          {open ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        </div>
        <div className="flex-1 min-w-0">
          <h4 className="font-semibold text-white text-sm">{sub.name}</h4>
          <p className="text-xs text-surface-200/60 mt-1 leading-relaxed">
            {sub.description}
          </p>
        </div>
        {sub.stats && sub.stats.length > 0 && (
          <div className="hidden sm:flex items-center gap-4 flex-shrink-0">
            {sub.stats.slice(0, 2).map((s) => (
              <div key={s.label} className="text-right">
                <div className="text-sm font-bold text-white">{s.value}</div>
                <div className="text-[10px] text-surface-200/40">{s.label}</div>
              </div>
            ))}
          </div>
        )}
      </button>

      {open && (
        <div className="px-4 pb-4 pt-0 ml-7">
          {/* Stats (mobile) */}
          {sub.stats && sub.stats.length > 0 && (
            <div className="flex flex-wrap gap-4 mb-3 sm:hidden">
              {sub.stats.map((s) => (
                <div key={s.label}>
                  <span className="text-sm font-bold text-white">{s.value}</span>
                  <span className="text-[10px] text-surface-200/40 ml-1">{s.label}</span>
                </div>
              ))}
            </div>
          )}
          {/* All stats on desktop if more than 2 */}
          {sub.stats && sub.stats.length > 2 && (
            <div className="hidden sm:flex flex-wrap gap-4 mb-3">
              {sub.stats.slice(2).map((s) => (
                <div key={s.label}>
                  <span className="text-sm font-bold text-white">{s.value}</span>
                  <span className="text-[10px] text-surface-200/40 ml-1">{s.label}</span>
                </div>
              ))}
            </div>
          )}
          {/* Components */}
          <div className="space-y-1">
            {sub.components.map((c, i) => (
              <div
                key={i}
                className="text-xs text-surface-200/50 font-mono leading-relaxed"
              >
                {c}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function DomainSection({ domain }: { domain: SystemDomain }) {
  return (
    <section id={domain.id} className="scroll-mt-24">
      <div className="rounded-xl border border-surface-700/50 bg-surface-800/30 overflow-hidden">
        {/* Domain header */}
        <div className={`flex items-center gap-4 p-6 border-b border-surface-700/50`}>
          <div
            className={`h-11 w-11 rounded-xl ${domain.colorBg} border ${domain.colorBorder} flex items-center justify-center`}
          >
            <DomainIcon name={domain.icon} className={domain.color} />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">{domain.title}</h2>
            <p className="text-sm text-surface-200/50">{domain.subtitle}</p>
          </div>
        </div>

        {/* Description */}
        <div className="px-6 py-4 border-b border-surface-700/30">
          <p className="text-sm text-surface-200/70 leading-relaxed">
            {domain.description}
          </p>
        </div>

        {/* Subsystems */}
        <div className="p-4 space-y-2">
          {domain.subsystems.map((sub) => (
            <SubsystemCard key={sub.name} sub={sub} />
          ))}
        </div>

        {/* Internal bridges */}
        {domain.bridges.length > 0 && (
          <div className="px-6 py-4 bg-surface-700/10 border-t border-surface-700/30">
            <h3 className="text-xs uppercase tracking-widest text-surface-200/30 mb-3">
              Internal Integration
            </h3>
            <div className="space-y-3">
              {domain.bridges.map((bridge, i) => (
                <div key={i} className="flex items-start gap-2 text-xs">
                  <div className="flex items-center gap-1.5 flex-shrink-0 mt-0.5">
                    <span className={`font-medium ${domain.color}`}>
                      {bridge.from}
                    </span>
                    <ArrowRight size={10} className="text-surface-200/30" />
                    <span className={`font-medium ${domain.color}`}>
                      {bridge.to}
                    </span>
                  </div>
                  <span className="text-surface-200/50 hidden md:inline">
                    — {bridge.description}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

export default function Systems() {
  return (
    <>
      <SeoHead
        meta={{
          title: "System Architecture",
          description:
            "Atlas system architecture: 5 integrated domains, 9 MCP servers, 190 tools, 51 autonomous tasks, 300 database migrations. How 15+ subsystems connect in production.",
          canonicalPath: "/systems",
          jsonLd: {
            "@context": "https://schema.org",
            "@type": "TechArticle",
            headline: "Atlas System Architecture",
            description:
              "Deep dive into the Atlas AI platform architecture — 5 domains, 15+ subsystems, and the integration patterns that connect them.",
            author: { "@type": "Person", name: "Juan Canfield" },
          },
        }}
      />

      <section className="py-16 px-6">
        <div className="mx-auto max-w-4xl">
          {/* Header */}
          <header className="mb-12">
            <h1 className="text-4xl font-bold text-white mb-4">
              System Architecture
            </h1>
            <p className="text-surface-200/70 max-w-2xl leading-relaxed mb-6">
              Atlas is not one system — it's 15+ subsystems organized into 5
              domains, each independently useful but designed to integrate.
              The real complexity lives in the bridges between them.
            </p>

            {/* Quick stats bar */}
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-4 p-4 rounded-xl border border-surface-700/50 bg-surface-800/30">
              {[
                { value: "658", label: "Python Files" },
                { value: "190", label: "MCP Tools" },
                { value: "99", label: "Task Handlers" },
                { value: "300", label: "DB Migrations" },
                { value: "114", label: "Services" },
                { value: "6", label: "UI Apps" },
              ].map((s) => (
                <div key={s.label} className="text-center">
                  <div className="text-lg font-bold text-white">{s.value}</div>
                  <div className="text-[10px] text-surface-200/40">{s.label}</div>
                </div>
              ))}
            </div>
          </header>

          {/* Domain nav */}
          <nav className="flex flex-wrap gap-2 mb-10">
            {systemDomains.map((d) => (
              <a
                key={d.id}
                href={`#${d.id}`}
                className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium border transition-colors ${d.colorBg} ${d.color} ${d.colorBorder} hover:brightness-125`}
              >
                <DomainIcon name={d.icon} className={d.color} />
                {d.title}
              </a>
            ))}
          </nav>

          {/* Domains */}
          <div className="space-y-8">
            {systemDomains.map((domain) => (
              <DomainSection key={domain.id} domain={domain} />
            ))}
          </div>

          {/* Cross-domain bridges */}
          <section className="mt-16">
            <h2 className="text-2xl font-bold text-white mb-4">
              Cross-Domain Integration
            </h2>
            <p className="text-sm text-surface-200/60 mb-8">
              These are the bridges where data flows between domains.
              Each one represents a design decision about where
              responsibilities end and handoffs begin.
            </p>

            <div className="space-y-4">
              {crossDomainBridges.map((bridge, i) => (
                <div
                  key={i}
                  className="rounded-xl border border-surface-700/50 bg-surface-800/30 p-5"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-sm font-semibold text-white">
                      {bridge.from}
                    </span>
                    <ArrowRight size={14} className="text-primary-400" />
                    <span className="text-sm font-semibold text-white">
                      {bridge.to}
                    </span>
                  </div>
                  <p className="text-sm text-surface-200/70 leading-relaxed">
                    {bridge.description}
                  </p>
                </div>
              ))}
            </div>
          </section>

          {/* UI Applications */}
          <section className="mt-16">
            <h2 className="text-2xl font-bold text-white mb-4">
              UI Applications (6)
            </h2>
            <p className="text-sm text-surface-200/60 mb-8">
              Every backend domain has at least one frontend surface.
              All web apps share the same stack: React 19 + Vite + Tailwind CSS.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {[
                {
                  name: "Atlas Churn UI",
                  desc: "B2B intelligence dashboard — vendor tracking, churn signals, account intelligence, 80+ blog posts with charts",
                  tech: "React + Vite + Recharts",
                },
                {
                  name: "Atlas Intel UI",
                  desc: "Consumer intelligence — brand health, complaint analysis, migration flows, safety signals, 14 blog posts",
                  tech: "React + Vite + Recharts",
                },
                {
                  name: "Atlas Admin UI",
                  desc: "Operations dashboard — LLM cost tracking, provider analytics, task health, scraping pipeline, reasoning panel",
                  tech: "React + Vite + Zustand",
                },
                {
                  name: "Atlas UI",
                  desc: "Main dashboard — conversation interface, device control, system status",
                  tech: "React + Vite + Zustand",
                },
                {
                  name: "Atlas Mobile",
                  desc: "iOS + Android app — mobile access to Atlas capabilities",
                  tech: "Expo + React Native + NativeWind",
                },
                {
                  name: "Portfolio UI",
                  desc: "This site — project showcase, insights, skill framework",
                  tech: "React + Vite + Tailwind",
                },
              ].map((app) => (
                <div
                  key={app.name}
                  className="rounded-lg border border-surface-700/40 bg-surface-800/20 p-4"
                >
                  <h3 className="font-semibold text-white text-sm">
                    {app.name}
                  </h3>
                  <p className="text-xs text-surface-200/60 mt-1 leading-relaxed">
                    {app.desc}
                  </p>
                  <p className="text-[10px] text-surface-200/30 font-mono mt-2">
                    {app.tech}
                  </p>
                </div>
              ))}
            </div>
          </section>
        </div>
      </section>
    </>
  );
}
