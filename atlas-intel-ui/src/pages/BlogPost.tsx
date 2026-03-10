import { useParams, Link } from 'react-router-dom'
import { lazy, Suspense, useMemo } from 'react'
import { marked } from 'marked'
import { ArrowLeft } from 'lucide-react'
import PublicLayout from '../components/PublicLayout'
import SeoHead from '../components/SeoHead'
import BlogCardVisual from '../components/BlogCardVisual'
import { POSTS } from '../content/blog'
import type { ChartSpec, BlogPost as BlogPostType } from '../content/blog'

const BlogChart = lazy(() => import('../components/BlogChartRenderer'))

function formatDate(iso: string) {
  return new Date(iso + 'T00:00:00').toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })
}

const CHART_PLACEHOLDER_RE = /(\{\{chart:[^}]+\}\})/

function renderContentWithCharts(content: string, charts?: ChartSpec[]) {
  if (!charts || charts.length === 0) {
    const html = marked.parse(content, { async: false }) as string
    return <div className="blog-prose" dangerouslySetInnerHTML={{ __html: html }} />
  }

  const chartMap = new Map(charts.map(c => [c.chart_id, c]))
  const parts = content.split(CHART_PLACEHOLDER_RE)

  return (
    <div className="blog-prose">
      {parts.map((part, i) => {
        const match = part.match(/^\{\{chart:([^}]+)\}\}$/)
        if (match) {
          const spec = chartMap.get(match[1])
          if (spec) {
            return (
              <Suspense key={i} fallback={null}>
                <BlogChart spec={spec} />
              </Suspense>
            )
          }
          return null
        }
        if (!part.trim()) return null
        const html = marked.parse(part, { async: false }) as string
        return <div key={i} dangerouslySetInnerHTML={{ __html: html }} />
      })}
    </div>
  )
}

export default function BlogPost() {
  const { slug } = useParams<{ slug: string }>()
  const post = POSTS.find(p => p.slug === slug)

  const renderedContent = useMemo(() => {
    if (!post) return null
    return renderContentWithCharts(post.content, post.charts)
  }, [post])

  const seoKeywords = useMemo(() => {
    if (!post) return undefined
    const kws: string[] = []
    if (post.target_keyword) kws.push(post.target_keyword)
    if (post.secondary_keywords) kws.push(...post.secondary_keywords)
    return kws.length > 0 ? kws : undefined
  }, [post])

  const jsonLd = useMemo(() => {
    if (!post) return undefined
    const blogPosting: Record<string, unknown> = {
      '@context': 'https://schema.org',
      '@type': 'BlogPosting',
      headline: post.seo_title || post.title,
      description: post.seo_description || post.description,
      datePublished: post.date,
      dateModified: post.date,
      image: 'https://atlas-intel-ui-two.vercel.app/og-default.png',
      author: {
        '@type': 'Organization',
        name: 'Atlas Intelligence',
        sameAs: ['https://twitter.com/atlasintel', 'https://www.linkedin.com/company/atlas-intelligence'],
      },
      publisher: {
        '@type': 'Organization',
        name: 'Atlas Intelligence',
        url: 'https://atlas-intel-ui-two.vercel.app',
        sameAs: ['https://twitter.com/atlasintel', 'https://www.linkedin.com/company/atlas-intelligence'],
      },
      mainEntityOfPage: {
        '@type': 'WebPage',
        '@id': `https://atlas-intel-ui-two.vercel.app/blog/${post.slug}`,
      },
      isBasedOn: {
        '@type': 'Dataset',
        name: 'Consumer Product Review Intelligence',
        description: 'Aggregated from Amazon verified purchase reviews',
      },
      keywords: seoKeywords?.join(', ') || '',
    }

    // For migration_report topics, extract numbered steps for HowTo schema
    if (post.topic_type === 'migration_report' && post.content) {
      const stepRegex = /<li>\s*<strong>([^<]+)<\/strong>/g
      const steps: { '@type': string; name: string; position: number }[] = []
      let match
      let pos = 1
      while ((match = stepRegex.exec(post.content)) !== null) {
        steps.push({ '@type': 'HowToStep', name: match[1].replace(/\s*--\s*$/, ''), position: pos++ })
      }
      if (steps.length >= 2) {
        return {
          '@context': 'https://schema.org',
          '@graph': [
            blogPosting,
            {
              '@type': 'HowTo',
              name: post.seo_title || post.title,
              description: post.seo_description || post.description,
              step: steps,
            },
          ],
        }
      }
    }

    return blogPosting
  }, [post, seoKeywords])

  const relatedPosts = useMemo(() => {
    if (!post?.related_slugs) return []
    return post.related_slugs
      .map(s => POSTS.find(p => p.slug === s))
      .filter((p): p is BlogPostType => !!p)
  }, [post])

  if (!post) {
    return (
      <PublicLayout>
        <section className="max-w-3xl mx-auto px-6 py-24 text-center">
          <h1 className="text-3xl font-bold mb-4">Post not found</h1>
          <p className="text-slate-400 mb-8">The article you are looking for does not exist.</p>
          <Link to="/blog" className="text-cyan-400 hover:text-cyan-300 transition-colors">
            Back to blog
          </Link>
        </section>
      </PublicLayout>
    )
  }

  return (
    <PublicLayout>
      <SeoHead
        title={`${post.seo_title || post.title} | Atlas Intelligence`}
        description={post.seo_description || post.description}
        canonical={`https://atlas-intel-ui-two.vercel.app/blog/${post.slug}`}
        keywords={seoKeywords}
        faq={post.faq}
        jsonLd={jsonLd}
      />
      <article className="max-w-3xl mx-auto px-6 pt-12 pb-24">
        {/* Back link */}
        <Link
          to="/blog"
          className="inline-flex items-center gap-1.5 text-sm text-slate-400 hover:text-white transition-colors mb-8"
        >
          <ArrowLeft className="h-4 w-4" />
          All posts
        </Link>

        {/* Hero visual */}
        <div className="rounded-xl overflow-hidden mb-8">
          <BlogCardVisual post={post} />
        </div>

        {/* Article header */}
        <header className="mb-10">
          <div className="flex flex-wrap items-center gap-2 mb-4">
            {post.tags.map(tag => (
              <span
                key={tag}
                className="text-xs px-2.5 py-0.5 bg-slate-700/60 text-slate-400 rounded-full"
              >
                {tag}
              </span>
            ))}
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold leading-tight">{post.title}</h1>
          <div className="mt-4 flex items-center gap-3 text-sm text-slate-500">
            <span>{post.author}</span>
            <span className="w-1 h-1 rounded-full bg-slate-600" />
            <time>{formatDate(post.date)}</time>
          </div>
        </header>

        {/* Rendered markdown with inline charts */}
        {renderedContent}

        {/* CTA */}
        <div className="mt-16 p-8 bg-slate-800/60 border border-slate-700/50 rounded-xl text-center">
          <h2 className="text-xl font-bold mb-2">Start tracking these products</h2>
          <p className="text-slate-400 mb-6">
            Get AI-powered review monitoring, competitor migration tracking, and safety alerts. Free for 14 days.
          </p>
          <Link
            to="/signup?product=consumer"
            className="inline-block px-6 py-3 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-white font-semibold transition-colors"
          >
            Start Free Trial
          </Link>
        </div>

        {/* FAQ section -- rendered for SEO (FAQ schema via SeoHead) */}
        {post.faq && post.faq.length > 0 && (
          <section className="mt-12 border-t border-slate-700/50 pt-8">
            <h2 className="text-xl font-bold mb-6">Frequently Asked Questions</h2>
            {post.faq.map((item, i) => (
              <div key={i} className="mb-6">
                <h3 className="font-semibold text-lg text-white">{item.question}</h3>
                <p className="mt-2 text-slate-400">{item.answer}</p>
              </div>
            ))}
          </section>
        )}

        {/* Related posts -- internal linking for SEO */}
        {relatedPosts.length > 0 && (
          <section className="mt-8 border-t border-slate-700/50 pt-8">
            <h2 className="text-lg font-semibold mb-4">Related Analysis</h2>
            <div className="grid gap-3">
              {relatedPosts.map(rp => (
                <Link
                  key={rp.slug}
                  to={`/blog/${rp.slug}`}
                  className="flex items-center gap-3 p-3 rounded-lg bg-slate-800/40 border border-slate-700/30 hover:border-cyan-500/30 transition-colors"
                >
                  <span className="text-sm text-cyan-400 hover:text-cyan-300">{rp.title}</span>
                </Link>
              ))}
            </div>
          </section>
        )}
      </article>
    </PublicLayout>
  )
}
