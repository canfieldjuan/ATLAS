import { useParams, Navigate, Link } from "react-router-dom";
import { SeoHead } from "@/components/seo/SeoHead";
import { MediaGallery } from "@/components/media/MediaGallery";
import { getInsight } from "@/content/insights";
import { ArrowLeft, BookOpen, Wrench, TrendingUp, Flame } from "lucide-react";

const TYPE_META = {
  "case-study": { label: "Case Study", icon: BookOpen, color: "text-primary-400" },
  "build-log": { label: "Build Log", icon: Wrench, color: "text-accent-amber" },
  "industry-insight": { label: "Industry Insight", icon: TrendingUp, color: "text-accent-cyan" },
  lesson: { label: "Lesson", icon: Flame, color: "text-rose-400" },
} as const;

export default function InsightDetail() {
  const { slug } = useParams<{ slug: string }>();
  const post = slug ? getInsight(slug) : undefined;

  if (!post) return <Navigate to="/insights" replace />;

  const typeMeta = TYPE_META[post.type];
  const TypeIcon = typeMeta.icon;

  const jsonLd: Record<string, unknown> = {
    "@context": "https://schema.org",
    "@type": "Article",
    headline: post.title,
    description: post.seoDescription ?? post.description,
    datePublished: post.date,
    author: { "@type": "Person", name: "Juan Canfield" },
  };

  if (post.faq && post.faq.length > 0) {
    jsonLd.mainEntity = post.faq.map((f) => ({
      "@type": "Question",
      name: f.question,
      acceptedAnswer: { "@type": "Answer", text: f.answer },
    }));
  }

  return (
    <>
      <SeoHead
        meta={{
          title: post.seoTitle ?? post.title,
          description: post.seoDescription ?? post.description,
          canonicalPath: `/insights/${post.slug}`,
          jsonLd,
        }}
      />

      <article className="py-16 px-6">
        <div className="mx-auto max-w-3xl">
          {/* Back link */}
          <Link
            to="/insights"
            className="inline-flex items-center gap-2 text-sm text-surface-200/50 hover:text-surface-200 transition-colors mb-8"
          >
            <ArrowLeft size={14} />
            All Insights
          </Link>

          {/* Header */}
          <header className="mb-12">
            <div className="flex items-center gap-3 mb-4">
              <span
                className={`inline-flex items-center gap-1.5 text-sm font-medium ${typeMeta.color}`}
              >
                <TypeIcon size={14} />
                {typeMeta.label}
              </span>
              <span className="text-sm text-surface-200/40">
                {new Date(post.date).toLocaleDateString("en-US", {
                  month: "long",
                  day: "numeric",
                  year: "numeric",
                })}
              </span>
              {post.project && (
                <Link
                  to={`/projects/${post.project}`}
                  className="text-sm text-primary-400/60 hover:text-primary-400 transition-colors"
                >
                  {post.project === "atlas" ? "Atlas" : "FineTuneLab.ai"}
                </Link>
              )}
            </div>
            <h1 className="text-3xl sm:text-4xl font-bold text-white leading-tight mb-4">
              {post.title}
            </h1>
            <p className="text-lg text-surface-200/70 leading-relaxed">
              {post.description}
            </p>
          </header>

          {/* Content */}
          <div
            className="prose-custom"
            dangerouslySetInnerHTML={{ __html: post.content }}
          />

          {/* Media */}
          {post.media && post.media.length > 0 && (
            <section className="mt-12">
              <MediaGallery items={post.media} columns={2} />
            </section>
          )}

          {/* FAQ (renders as visible content + JSON-LD schema) */}
          {post.faq && post.faq.length > 0 && (
            <section className="mt-16">
              <h2 className="text-2xl font-bold text-white mb-6">FAQ</h2>
              <div className="space-y-6">
                {post.faq.map((item) => (
                  <div
                    key={item.question}
                    className="rounded-xl border border-surface-700/50 bg-surface-800/30 p-5"
                  >
                    <h3 className="font-semibold text-white mb-2">
                      {item.question}
                    </h3>
                    <p className="text-sm text-surface-200/70 leading-relaxed">
                      {item.answer}
                    </p>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Tags */}
          <div className="mt-12 pt-8 border-t border-surface-700/50">
            <div className="flex flex-wrap gap-2">
              {post.tags.map((tag) => (
                <span
                  key={tag}
                  className="rounded-full border border-surface-700/50 bg-surface-800/30 px-3 py-1 text-xs text-surface-200/60"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        </div>
      </article>
    </>
  );
}
