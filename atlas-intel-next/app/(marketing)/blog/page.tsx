import Link from "next/link";
import type { Metadata } from "next";
import { SITE_URL } from "@/lib/constants";
import { fetchAllPosts } from "@/lib/api/blog";

export const metadata: Metadata = {
  title: "Blog",
  description:
    "Data-driven analysis of B2B software churn signals, vendor displacement patterns, and competitive intelligence. Research backed by real reviewer data.",
  alternates: { canonical: `${SITE_URL}/blog` },
  openGraph: {
    title: "Churn Signals Blog",
    description:
      "Data-driven B2B software churn analysis backed by real reviewer data.",
    type: "website",
  },
};

export default async function BlogPage() {
  const POSTS = await fetchAllPosts();
  return (
    <section className="max-w-4xl mx-auto px-6 py-16">
      <h1 className="text-3xl font-bold mb-2">Blog</h1>
      <p className="text-slate-400 mb-12">
        Data-driven analysis of B2B software churn signals and competitive
        intelligence.
      </p>
      <div className="grid gap-8">
        {POSTS.map((post) => (
          <Link
            key={post.slug}
            href={`/blog/${post.slug}`}
            className="block bg-slate-800/60 border border-slate-700/50 rounded-xl p-6 hover:border-slate-600 transition-colors"
          >
            <div className="flex items-center gap-3 text-xs text-slate-500 mb-3">
              <time dateTime={post.date}>{post.date}</time>
              {post.tags.slice(0, 3).map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-0.5 bg-slate-700/50 rounded text-slate-400"
                >
                  {tag}
                </span>
              ))}
            </div>
            <h2 className="text-xl font-semibold mb-2">{post.title}</h2>
            <p className="text-sm text-slate-400 line-clamp-2">
              {post.description}
            </p>
          </Link>
        ))}
      </div>
    </section>
  );
}
