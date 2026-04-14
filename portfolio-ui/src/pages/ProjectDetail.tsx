import { useState } from "react";
import { useParams, Navigate, Link } from "react-router-dom";
import { SeoHead } from "@/components/seo/SeoHead";
import { MediaGallery } from "@/components/media/MediaGallery";
import { getProject } from "@/content/projects";
import { getInsight } from "@/content/insights";
import * as Icons from "lucide-react";
import { ChevronDown, ChevronRight, ArrowRight } from "lucide-react";
import type { ProjectSubsystem } from "@/types";

function SubsystemIcon({
  name,
  className,
}: {
  name: string;
  className?: string;
}) {
  const Icon =
    (Icons as unknown as Record<string, Icons.LucideIcon>)[name] ?? Icons.Box;
  return <Icon className={className} size={18} />;
}

function SubsystemCard({ sub }: { sub: ProjectSubsystem }) {
  const [open, setOpen] = useState(false);
  const relatedPost = sub.relatedInsight
    ? getInsight(sub.relatedInsight)
    : undefined;

  return (
    <div className="rounded-xl border border-surface-700/50 bg-surface-800/30 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-start gap-4 p-5 text-left hover:bg-surface-800/50 transition-colors"
      >
        <div className="h-10 w-10 rounded-lg bg-primary-500/10 flex items-center justify-center flex-shrink-0 mt-0.5">
          <SubsystemIcon name={sub.icon} className="text-primary-400" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-white">{sub.name}</h3>
            {open ? (
              <ChevronDown size={14} className="text-surface-200/40" />
            ) : (
              <ChevronRight size={14} className="text-surface-200/40" />
            )}
          </div>
          <p className="text-sm text-surface-200/60 mt-1 leading-relaxed">
            {sub.description}
          </p>
        </div>
      </button>

      {open && (
        <div className="px-5 pb-5 pt-0 border-t border-surface-700/30">
          <div className="pt-4 flex flex-col gap-4">
            {/* Stats */}
            {sub.stats && sub.stats.length > 0 && (
              <div className="flex flex-wrap gap-4">
                {sub.stats.map((s) => (
                  <div
                    key={s.label}
                    className="rounded-lg bg-surface-700/20 px-3 py-2"
                  >
                    <span className="text-sm font-bold text-white">
                      {s.value}
                    </span>
                    <span className="text-[10px] text-surface-200/40 ml-1.5">
                      {s.label}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {/* Related insight link */}
            {relatedPost && (
              <Link
                to={`/insights/${relatedPost.slug}`}
                className="group flex items-center gap-2 text-sm text-primary-400/70 hover:text-primary-400 transition-colors"
              >
                <ArrowRight
                  size={14}
                  className="group-hover:translate-x-1 transition-transform"
                />
                Read more: {relatedPost.title}
              </Link>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default function ProjectDetail() {
  const { slug } = useParams<{ slug: string }>();
  const project = slug ? getProject(slug) : undefined;

  if (!project) return <Navigate to="/projects" replace />;

  return (
    <>
      <SeoHead
        meta={{
          title: project.title,
          description: project.description.slice(0, 160),
          canonicalPath: `/projects/${project.slug}`,
        }}
      />

      <article className="py-16 px-6">
        <div className="mx-auto max-w-4xl">
          {/* Header */}
          <header className="mb-12">
            <h1 className="text-4xl font-bold text-white mb-2">
              {project.title}
            </h1>
            <p className="text-xl text-primary-400 font-medium mb-4">
              {project.tagline}
            </p>
            <p className="text-surface-200/80 leading-relaxed max-w-3xl">
              {project.description}
            </p>
          </header>

          {/* Stats grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 mb-16">
            {project.stats.map((stat) => (
              <div
                key={stat.label}
                className="text-center p-4 rounded-lg border border-surface-700/50 bg-surface-800/30"
              >
                <p className="text-2xl font-bold text-white">{stat.value}</p>
                <p className="text-xs text-surface-200/50 mt-1">
                  {stat.label}
                </p>
              </div>
            ))}
          </div>

          {/* Highlights */}
          <section className="mb-16">
            <h2 className="text-2xl font-bold text-white mb-8">
              Key Architecture Decisions
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {project.highlights.map((hl) => {
                const Icon =
                  (Icons as unknown as Record<string, Icons.LucideIcon>)[
                    hl.icon
                  ] ?? Icons.Code;
                return (
                  <div
                    key={hl.title}
                    className="p-5 rounded-xl border border-surface-700/50 bg-surface-800/30"
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <div className="h-9 w-9 rounded-lg bg-primary-500/10 flex items-center justify-center">
                        <Icon size={18} className="text-primary-400" />
                      </div>
                      <h3 className="font-semibold text-white">{hl.title}</h3>
                    </div>
                    <p className="text-sm text-surface-200/70 leading-relaxed">
                      {hl.description}
                    </p>
                  </div>
                );
              })}
            </div>
          </section>

          {/* Subsystems */}
          {project.subsystems.length > 0 && (
            <section className="mb-16">
              <h2 className="text-2xl font-bold text-white mb-3">
                Key Subsystems
              </h2>
              <p className="text-sm text-surface-200/50 mb-8">
                Expand any subsystem to see stats and related deep-dive
                content.
              </p>
              <div className="space-y-3">
                {project.subsystems.map((sub) => (
                  <SubsystemCard key={sub.name} sub={sub} />
                ))}
              </div>
            </section>
          )}

          {/* Media gallery */}
          {project.media.length > 0 && (
            <section className="mb-16">
              <h2 className="text-2xl font-bold text-white mb-8">In Action</h2>
              <MediaGallery items={project.media} columns={2} />
            </section>
          )}

          {/* Tech stack */}
          <section>
            <h2 className="text-2xl font-bold text-white mb-6">Tech Stack</h2>
            <div className="flex flex-wrap gap-2">
              {project.techStack.map((tech) => (
                <span
                  key={tech}
                  className="rounded-full border border-surface-700/50 bg-surface-800/30 px-3 py-1.5 text-sm text-surface-200/80"
                >
                  {tech}
                </span>
              ))}
            </div>
          </section>
        </div>
      </article>
    </>
  );
}
