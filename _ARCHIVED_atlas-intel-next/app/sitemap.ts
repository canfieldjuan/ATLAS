import type { MetadataRoute } from "next";
import { SITE_URL } from "@/lib/constants";
import { fetchAllPosts } from "@/lib/api/blog";

const REPORT_API =
  process.env.BLOG_API_URL ||
  process.env.NEXT_PUBLIC_API_BASE ||
  "http://localhost:8000";

async function fetchReportVendors(): Promise<string[]> {
  try {
    const res = await fetch(
      `${REPORT_API}/api/v1/blog/published?limit=200`,
      { next: { revalidate: 3600 } },
    );
    if (!res.ok) return [];
    const data = await res.json();
    // Extract unique vendor names from published posts for report URLs
    const vendors = new Set<string>();
    for (const post of data.posts || []) {
      const ctx = post.data_context || {};
      const vendor =
        ctx.vendor || ctx.topic_ctx?.vendor || ctx.topic_ctx?.vendor_a;
      if (vendor) vendors.add(vendor);
    }
    return Array.from(vendors);
  } catch {
    return [];
  }
}

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const posts = await fetchAllPosts();
  const reportVendors = await fetchReportVendors();

  const staticPages: MetadataRoute.Sitemap = [
    {
      url: SITE_URL,
      lastModified: new Date(),
      changeFrequency: "weekly",
      priority: 1.0,
    },
    {
      url: `${SITE_URL}/blog`,
      lastModified: new Date(),
      changeFrequency: "daily",
      priority: 0.9,
    },
  ];

  const blogPages: MetadataRoute.Sitemap = posts.map((post) => ({
    url: `${SITE_URL}/blog/${post.slug}`,
    lastModified: new Date(post.date),
    changeFrequency: "monthly" as const,
    priority: 0.7,
  }));

  // B2B vendor report gate pages (public intelligence landing pages)
  const reportPages: MetadataRoute.Sitemap = reportVendors.map((vendor) => ({
    url: `${SITE_URL}/report?vendor=${encodeURIComponent(vendor)}`,
    lastModified: new Date(),
    changeFrequency: "weekly" as const,
    priority: 0.8,
  }));

  return [...staticPages, ...blogPages, ...reportPages];
}
