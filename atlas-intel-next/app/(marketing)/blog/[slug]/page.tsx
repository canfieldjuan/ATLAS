import { SITE_URL } from "@/lib/constants";
import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { fetchAllPosts, fetchPostBySlug } from "@/lib/api/blog";
import BlogPostContent from "./blog-post-content";

// ---------------------------------------------------------------------------
// SSG: pre-render all known blog slugs at build time
// New posts from the API appear via ISR revalidation (hourly)
// ---------------------------------------------------------------------------
export async function generateStaticParams() {
  const posts = await fetchAllPosts();
  return posts.map((post) => ({ slug: post.slug }));
}

// ---------------------------------------------------------------------------
// Metadata: baked into HTML at build time -- crawlers see it without JS
// ---------------------------------------------------------------------------
type PageProps = { params: Promise<{ slug: string }> };

export async function generateMetadata(props: PageProps): Promise<Metadata> {
  const { slug } = await props.params;
  const post = await fetchPostBySlug(slug);
  if (!post) return {};

  const title = post.seo_title || post.title;
  const description = post.seo_description || post.description;
  const url = `${SITE_URL}/blog/${post.slug}`;

  return {
    title,
    description,
    keywords: post.secondary_keywords,
    alternates: { canonical: url },
    openGraph: {
      title: post.title,
      description,
      type: "article",
      url,
      publishedTime: post.date,
      authors: [post.author],
      tags: post.tags,
      images: [{ url: `${SITE_URL}/og-default.png`, width: 1200, height: 630 }],
    },
    twitter: {
      card: "summary_large_image",
      title: post.title,
      description,
    },
  };
}

// ---------------------------------------------------------------------------
// Page component
// ---------------------------------------------------------------------------
export default async function BlogPostPage(props: PageProps) {
  const { slug } = await props.params;
  const post = await fetchPostBySlug(slug);
  if (!post) notFound();

  // Build JSON-LD structured data (baked into HTML at build time)
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "BlogPosting",
    headline: post.title,
    description: post.description,
    datePublished: post.date,
    dateModified: post.date,
    author: {
      "@type": "Organization",
      name: "Churn Signals",
      url: SITE_URL,
    },
    publisher: {
      "@type": "Organization",
      name: "Churn Signals",
      url: SITE_URL,
    },
    mainEntityOfPage: {
      "@type": "WebPage",
      "@id": `${SITE_URL}/blog/${post.slug}`,
    },
    keywords: post.tags.join(", "),
  };

  const breadcrumbJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      {
        "@type": "ListItem",
        position: 1,
        name: "Home",
        item: SITE_URL,
      },
      {
        "@type": "ListItem",
        position: 2,
        name: "Blog",
        item: `${SITE_URL}/blog`,
      },
      {
        "@type": "ListItem",
        position: 3,
        name: post.title,
      },
    ],
  };

  const faqJsonLd =
    post.faq && post.faq.length > 0
      ? {
          "@context": "https://schema.org",
          "@type": "FAQPage",
          mainEntity: post.faq.map((item) => ({
            "@type": "Question",
            name: item.question,
            acceptedAnswer: {
              "@type": "Answer",
              text: item.answer,
            },
          })),
        }
      : null;

  return (
    <article className="max-w-3xl mx-auto px-6 py-16">
      {/* JSON-LD in HTML source -- crawlers see this without JS */}
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }}
      />
      {faqJsonLd && (
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(faqJsonLd) }}
        />
      )}

      {/* Header */}
      <header className="mb-10">
        <div className="flex items-center gap-3 text-xs text-slate-500 mb-4">
          <time dateTime={post.date}>{post.date}</time>
          <span>{post.author}</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-bold leading-tight mb-4">
          {post.title}
        </h1>
        <p className="text-lg text-slate-400">{post.description}</p>
        <div className="flex flex-wrap gap-2 mt-4">
          {post.tags.map((tag) => (
            <span
              key={tag}
              className="px-2 py-1 text-xs bg-slate-800 border border-slate-700 rounded text-slate-400"
            >
              {tag}
            </span>
          ))}
        </div>
      </header>

      {/* Content -- rendered from markdown at build time */}
      <BlogPostContent content={post.content} charts={post.charts} />

      {/* FAQ section */}
      {post.faq && post.faq.length > 0 && (
        <section className="mt-16 border-t border-slate-800 pt-10">
          <h2 className="text-2xl font-bold mb-6">
            Frequently Asked Questions
          </h2>
          <div className="space-y-6">
            {post.faq.map((item) => (
              <div key={item.question}>
                <h3 className="font-semibold text-lg mb-2">{item.question}</h3>
                <p className="text-slate-400">{item.answer}</p>
              </div>
            ))}
          </div>
        </section>
      )}
    </article>
  );
}
