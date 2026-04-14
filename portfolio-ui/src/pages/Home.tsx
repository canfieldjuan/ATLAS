import { SeoHead } from "@/components/seo/SeoHead";
import { HeroSection } from "@/components/sections/HeroSection";
import { ManifestoSection } from "@/components/sections/ManifestoSection";
import { ProjectCard } from "@/components/sections/ProjectCard";
import { allProjects } from "@/content/projects";

export default function Home() {
  return (
    <>
      <SeoHead
        meta={{
          title: "AI Systems Architect",
          description:
            "Building production AI systems — not chatbot demos. Real infrastructure where non-deterministic LLM outputs become deterministic, repeatable pipelines.",
          canonicalPath: "/",
          jsonLd: {
            "@context": "https://schema.org",
            "@graph": [
              {
                "@type": "Person",
                name: "Juan Canfield",
                jobTitle: "AI Systems Architect",
                description:
                  "Building production AI systems with MCP servers, autonomous task orchestration, and edge compute architecture.",
              },
              {
                "@type": "WebSite",
                name: "Juan Canfield",
                url: window.location.origin,
                inLanguage: "en-US",
              },
              {
                "@type": "Organization",
                name: "Juan Canfield",
                url: window.location.origin,
                sameAs: [
                  "https://github.com/canfieldjuan/atlas-portfolio",
                  "https://www.linkedin.com/in/juan-canfield-9b2a733b5/",
                ],
              },
            ],
          },
        }}
      />

      <HeroSection />

      {/* Scale metrics */}
      <section className="py-12 px-6 border-y border-surface-700/30 bg-surface-900/50">
        <div className="mx-auto max-w-5xl grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-8">
          {[
            { value: "25K+", label: "Reviews Enriched" },
            { value: "56", label: "Vendors Tracked" },
            { value: "280+", label: "DB Migrations" },
            { value: "36", label: "Autonomous Tasks" },
            { value: "3,271", label: "Witness Records" },
            { value: "94ms", label: "NPU Inference" },
          ].map((stat) => (
            <div key={stat.label} className="text-center">
              <div className="text-xl font-bold text-white">{stat.value}</div>
              <div className="text-[11px] text-surface-200/50 mt-1">{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      <ManifestoSection />

      {/* Projects preview */}
      <section className="py-24 px-6">
        <div className="mx-auto max-w-4xl">
          <h2 className="text-3xl font-bold text-white mb-4 text-center">
            Systems I've Built
          </h2>
          <p className="text-surface-200/60 text-center mb-12 max-w-2xl mx-auto">
            Not proof-of-concepts. Production systems handling real data,
            real users, and real failure modes.
          </p>
          <p className="text-sm text-surface-200/60 text-center mb-8 max-w-2xl mx-auto">
            I don't build chat demos. I build reliable AI systems — with
            quality gates, fallback paths, and measurable business outcomes.
          </p>
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
