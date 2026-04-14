import { useState } from "react";
import { Link } from "react-router-dom";
import { SeoHead } from "@/components/seo/SeoHead";
import { allInsights } from "@/content/insights";
import { ArrowRight, BookOpen, Wrench, TrendingUp, Flame } from "lucide-react";
import type { InsightPost } from "@/types";

const TYPE_CONFIG = {
  "case-study": {
    label: "Case Study",
    icon: BookOpen,
    color: "text-primary-400",
    bg: "bg-primary-500/10",
    border: "border-primary-500/30",
  },
  "build-log": {
    label: "Build Log",
    icon: Wrench,
    color: "text-accent-amber",
    bg: "bg-amber-500/10",
    border: "border-amber-500/30",
  },
  "industry-insight": {
    label: "Industry Insight",
    icon: TrendingUp,
    color: "text-accent-cyan",
    bg: "bg-cyan-500/10",
    border: "border-cyan-500/30",
  },
  lesson: {
    label: "Lesson",
    icon: Flame,
    color: "text-rose-400",
    bg: "bg-rose-500/10",
    border: "border-rose-500/30",
  },
} as const;

type FilterType = InsightPost["type"] | "all";

export default function Insights() {
  const [filter, setFilter] = useState<FilterType>("all");

  const filtered =
    filter === "all"
      ? allInsights
      : allInsights.filter((p) => p.type === filter);

  return (
    <>
      <SeoHead
        meta={{
          title: "Insights",
          description:
            "Case studies, build logs, and lessons from building production AI systems. The tedious, essential parts of AI-first development that nobody else documents.",
          canonicalPath: "/insights",
        }}
      />

      <section className="py-16 px-6">
        <div className="mx-auto max-w-4xl">
          <h1 className="text-4xl font-bold text-white mb-4">Insights</h1>
          <p className="text-surface-200/70 mb-10 max-w-2xl">
            Case studies, build logs, and hard-won lessons from building
            production AI systems. The tedious, essential work that tutorials
            skip.
          </p>

          {/* Filter tabs */}
          <div className="flex flex-wrap gap-2 mb-10">
            {(
              [
                "all",
                "case-study",
                "build-log",
                "lesson",
                "industry-insight",
              ] as FilterType[]
            ).map((type) => {
              const isActive = filter === type;
              const label =
                type === "all"
                  ? "All"
                  : TYPE_CONFIG[type].label;
              return (
                <button
                  key={type}
                  onClick={() => setFilter(type)}
                  className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-primary-500 text-surface-900"
                      : "bg-surface-800 text-surface-200/70 hover:text-white hover:bg-surface-700"
                  }`}
                >
                  {label}
                </button>
              );
            })}
          </div>

          {/* Post list */}
          <div className="space-y-6">
            {filtered.map((post) => {
              const config = TYPE_CONFIG[post.type];
              const Icon = config.icon;
              return (
                <Link
                  key={post.slug}
                  to={`/insights/${post.slug}`}
                  className="group block rounded-xl border border-surface-700/50 bg-surface-800/30 p-6 hover:border-primary-500/30 hover:bg-surface-800/50 transition-all"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-3">
                        <span
                          className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${config.bg} ${config.color} border ${config.border}`}
                        >
                          <Icon size={12} />
                          {config.label}
                        </span>
                        <span className="text-xs text-surface-200/40">
                          {new Date(post.date).toLocaleDateString("en-US", {
                            month: "short",
                            day: "numeric",
                            year: "numeric",
                          })}
                        </span>
                      </div>

                      <h2 className="text-lg font-semibold text-white group-hover:text-primary-400 transition-colors mb-2">
                        {post.title}
                      </h2>
                      <p className="text-sm text-surface-200/70 leading-relaxed line-clamp-2">
                        {post.description}
                      </p>

                      <div className="flex flex-wrap gap-2 mt-4">
                        {post.tags.slice(0, 4).map((tag) => (
                          <span
                            key={tag}
                            className="text-xs text-surface-200/50 bg-surface-700/30 rounded-full px-2 py-0.5"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>

                    <ArrowRight
                      size={20}
                      className="text-surface-200/30 group-hover:text-primary-400 group-hover:translate-x-1 transition-all mt-2 flex-shrink-0"
                    />
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      </section>
    </>
  );
}
