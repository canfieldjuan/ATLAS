import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import type { Project } from "@/types";

interface ProjectCardProps {
  project: Project;
}

export function ProjectCard({ project }: ProjectCardProps) {
  return (
    <Link
      to={`/projects/${project.slug}`}
      className="group block rounded-xl border border-surface-700/50 bg-surface-800/30 p-6 hover:border-primary-500/30 hover:bg-surface-800/50 transition-all duration-300"
    >
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-xl font-bold text-white group-hover:text-primary-400 transition-colors">
            {project.title}
          </h3>
          <p className="text-sm text-surface-200/60 mt-1">{project.tagline}</p>
        </div>
        <ArrowRight
          size={20}
          className="text-surface-200/30 group-hover:text-primary-400 group-hover:translate-x-1 transition-all mt-1"
        />
      </div>

      <p className="text-surface-200/80 text-sm leading-relaxed mb-6 line-clamp-3">
        {project.description}
      </p>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {project.stats.slice(0, 3).map((stat) => (
          <div key={stat.label}>
            <p className="text-lg font-bold text-white">{stat.value}</p>
            <p className="text-xs text-surface-200/50">{stat.label}</p>
          </div>
        ))}
      </div>

      {/* Tech badges */}
      <div className="flex flex-wrap gap-2">
        {project.techStack.slice(0, 6).map((tech) => (
          <span
            key={tech}
            className="rounded-full bg-surface-700/50 px-2.5 py-0.5 text-xs text-surface-200/70"
          >
            {tech}
          </span>
        ))}
        {project.techStack.length > 6 && (
          <span className="rounded-full bg-surface-700/50 px-2.5 py-0.5 text-xs text-surface-200/50">
            +{project.techStack.length - 6} more
          </span>
        )}
      </div>
    </Link>
  );
}
