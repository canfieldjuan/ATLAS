/**
 * Blog post fetching for SSG/ISR.
 *
 * At build time (or ISR revalidation), fetches published posts from the
 * Atlas backend API.  Falls back to bundled content if the API is
 * unavailable (e.g., during local dev without a running backend).
 */

import type { BlogPost } from "@/content/blog";
import { POSTS as BUNDLED_POSTS } from "@/content/blog";

const BLOG_API =
  process.env.BLOG_API_URL ||
  process.env.NEXT_PUBLIC_API_BASE ||
  "http://localhost:8000";

interface ApiBlogPost {
  slug: string;
  title: string;
  description: string;
  date: string;
  author: string;
  tags: string[];
  content: string;
  charts?: any[];
  topic_type?: string;
  data_context?: Record<string, unknown>;
  seo_title?: string;
  seo_description?: string;
  target_keyword?: string;
  secondary_keywords?: string[];
  faq?: { question: string; answer: string }[];
  related_slugs?: string[];
}

function apiPostToBlogPost(p: ApiBlogPost): BlogPost {
  return {
    slug: p.slug,
    title: p.title,
    description: p.description,
    date: p.date?.slice(0, 10) || "",
    author: p.author || "Churn Signals",
    tags: p.tags || [],
    content: p.content || "",
    charts: p.charts,
    topic_type: p.topic_type,
    data_context: p.data_context,
    seo_title: p.seo_title,
    seo_description: p.seo_description,
    target_keyword: p.target_keyword,
    secondary_keywords: p.secondary_keywords,
    faq: p.faq,
    related_slugs: p.related_slugs,
  };
}

/**
 * Fetch all published blog posts.
 * Merges API posts with bundled posts (API takes precedence on slug collision).
 */
export async function fetchAllPosts(): Promise<BlogPost[]> {
  let apiPosts: BlogPost[] = [];

  try {
    const res = await fetch(`${BLOG_API}/api/v1/blog/published?limit=200`, {
      next: { revalidate: 3600 }, // ISR: revalidate every hour
    });
    if (res.ok) {
      const data = await res.json();
      apiPosts = (data.posts || []).map(apiPostToBlogPost);
    }
  } catch {
    // API unavailable — use bundled posts only
  }

  // Merge: API posts take precedence, then bundled
  const seen = new Set(apiPosts.map((p) => p.slug));
  const merged = [
    ...apiPosts,
    ...BUNDLED_POSTS.filter((p) => !seen.has(p.slug)),
  ];

  return merged.sort((a, b) => b.date.localeCompare(a.date));
}

/**
 * Fetch a single post by slug.
 */
export async function fetchPostBySlug(
  slug: string,
): Promise<BlogPost | null> {
  // Try API first
  try {
    const res = await fetch(`${BLOG_API}/api/v1/blog/published/${slug}`, {
      next: { revalidate: 3600 },
    });
    if (res.ok) {
      const data = await res.json();
      if (data.post) return apiPostToBlogPost(data.post);
    }
  } catch {
    // Fall through to bundled
  }

  // Fallback to bundled
  return BUNDLED_POSTS.find((p) => p.slug === slug) || null;
}
