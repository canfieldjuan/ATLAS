import { useParams, Link } from 'react-router-dom'
import { lazy, Suspense, useMemo, useEffect } from 'react'
import { ArrowLeft } from 'lucide-react'
import PublicLayout from '../components/PublicLayout'
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
    return <div className="blog-prose" dangerouslySetInnerHTML={{ __html: content }} />
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
        return <div key={i} dangerouslySetInnerHTML={{ __html: part }} />
      })}
    </div>
  )
}

function getAffiliateCta(post: BlogPostType) {
  const url = post.data_context?.affiliate_url
  const partner = post.data_context?.affiliate_partner
  const partnerName = partner?.name || partner?.product_name

  if (url && partnerName) {
    return { url, name: partnerName, show: true }
  }
  // No affiliate — don't show a CTA
  return { url: '', name: '', show: false }
}

function hasAffiliateContent(post: BlogPostType): boolean {
  return !!(post.data_context?.affiliate_url || post.content?.includes('try.monday.com'))
}

export default function BlogPost() {
  const { slug } = useParams<{ slug: string }>()
  const post = POSTS.find(p => p.slug === slug)

  const cta = post ? getAffiliateCta(post) : { url: '', name: '', show: false }
  const showDisclosure = post ? hasAffiliateContent(post) : false

  const renderedContent = useMemo(() => {
    if (!post) return null
    return renderContentWithCharts(post.content, post.charts)
  }, [post])

  useEffect(() => {
    document.title = post
      ? `${post.title} | Churn Signals`
      : 'Post Not Found | Churn Signals'
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
      <article className="max-w-3xl mx-auto px-6 pt-12 pb-24">
        {/* Back link */}
        <Link
          to="/blog"
          className="inline-flex items-center gap-1.5 text-sm text-slate-400 hover:text-white transition-colors mb-8"
        >
          <ArrowLeft className="h-4 w-4" />
          All posts
        </Link>

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

        {/* Affiliate disclosure — before content, near the top */}
        {showDisclosure && (
          <div className="mb-8 px-4 py-3 bg-slate-800/40 border border-slate-700/40 rounded-lg text-xs text-slate-500">
            <strong>Disclosure:</strong> This article may contain affiliate links. If you purchase through these links, we may earn a commission at no additional cost to you. Our analysis and recommendations are based on verified review data, not affiliate relationships. See our{' '}
            <Link to="/methodology" className="text-cyan-500 hover:text-cyan-400 underline">methodology</Link>.
          </div>
        )}

        {/* Rendered markdown with inline charts */}
        {renderedContent}

        {/* CTA — only shown when post has a matching affiliate partner */}
        {cta.show && (
          <div className="mt-16 p-8 bg-slate-800/60 border border-slate-700/50 rounded-xl text-center">
            <h2 className="text-xl font-bold mb-2">Want to try {cta.name}?</h2>
            <p className="text-slate-400 mb-6">
              Based on the data, {cta.name} may be worth evaluating for your team. Start free, no credit card required.
            </p>
            <a
              href={cta.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block px-6 py-3 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-white font-semibold transition-colors"
            >
              Try {cta.name} Free
            </a>
            <p className="mt-3 text-xs text-slate-600">Affiliate link — see disclosure above</p>
          </div>
        )}

        {/* Generic CTA — shown when no affiliate partner */}
        {!cta.show && (
          <div className="mt-16 p-8 bg-slate-800/60 border border-slate-700/50 rounded-xl text-center">
            <h2 className="text-xl font-bold mb-2">Want churn intelligence on vendors in this space?</h2>
            <p className="text-slate-400 mb-6">
              We track real-time switching signals across 200+ B2B software vendors. See which accounts are actively evaluating alternatives.
            </p>
            <a
              href={(post.data_context?.booking_url as string) || 'https://cal.com/atlas-intel/15min'}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block px-6 py-3 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-white font-semibold transition-colors"
            >
              Book a 15-Min Intel Briefing
            </a>
          </div>
        )}
      </article>
    </PublicLayout>
  )
}
