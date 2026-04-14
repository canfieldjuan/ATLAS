import { SeoHead } from "@/components/seo/SeoHead";
import { ProjectCard } from "@/components/sections/ProjectCard";
import { allProjects } from "@/content/projects";

export default function Projects() {
  return (
    <>
      <SeoHead
        meta={{
          title: "Projects",
          description:
            "Production AI systems — Atlas (100+ MCP tools, autonomous orchestration, edge compute) and FineTuneLab.ai (225 API endpoints, LLM fine-tuning, GraphRAG).",
          canonicalPath: "/projects",
        }}
      />

      <section className="py-16 px-6">
        <div className="mx-auto max-w-4xl">
          <h1 className="text-4xl font-bold text-white mb-4">Projects</h1>
          <p className="text-surface-200/70 mb-12 max-w-2xl">
            Each project below is a production system — not a tutorial repo.
            Click through to see architecture decisions, the tedious parts,
            and what AI-first development actually looks like.
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
