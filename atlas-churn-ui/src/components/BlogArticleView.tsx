import { lazy, Suspense, useMemo } from 'react'
import { Link } from 'react-router-dom'
import type { BlogPost as BlogPostType, ChartSpec } from '../content/blog'

const BlogChart = lazy(() => import('./BlogChartRenderer'))

type PreviewViewport = 'desktop' | 'mobile'

interface BlogArticleViewProps {
  post: BlogPostType
  relatedPosts?: BlogPostType[]
  preview?: boolean
  previewViewport?: PreviewViewport
  showBackLink?: boolean
  highlightAffiliateLinks?: boolean
  highlightCharts?: boolean
  highlightCtas?: boolean
}

export interface ResolvedBlogArticleCta {
  mode: 'affiliate' | 'generic'
  url: string
  headline: string
  body: string
  buttonText: string
  name: string
}

const CHART_PLACEHOLDER_RE = /(\{\{chart:[^}]+\}\})/
const AFFILIATE_INLINE_STYLE = [
  'outline:2px dashed rgba(34,211,238,0.85)',
  'outline-offset:2px',
  'background:rgba(34,211,238,0.08)',
  'border-radius:4px',
].join(';')

function formatDate(iso: string) {
  return new Date(iso + 'T00:00:00').toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })
}

export function resolveBlogArticleCta(post: BlogPostType): ResolvedBlogArticleCta {
  const url = String(post.data_context?.affiliate_url || '').trim()
  const partner = post.data_context?.affiliate_partner
  const partnerName = String(partner?.name || partner?.product_name || '').trim()
  const storedCta = post.cta || null

  if (url && partnerName) {
    return {
      mode: 'affiliate',
      url,
      name: partnerName,
      headline: storedCta?.headline?.trim() || `Want to try ${partnerName}?`,
      body:
        storedCta?.body?.trim() ||
        `Based on the data, ${partnerName} may be worth evaluating for your team. Start free, no credit card required.`,
      buttonText: storedCta?.button_text?.trim() || `Try ${partnerName} Free`,
    }
  }
  return {
    mode: 'generic',
    url: String(post.data_context?.booking_url || 'https://cal.com/atlas-intel/15min'),
    name: '',
    headline:
      storedCta?.headline?.trim() || 'Want churn intelligence on vendors in this space?',
    body:
      storedCta?.body?.trim() ||
      'We track real-time switching signals across 200+ B2B software vendors. See which accounts are actively evaluating alternatives.',
    buttonText:
      storedCta?.button_text?.trim() || 'Book a 15-Min Intel Briefing',
  }
}

function hasAffiliateContent(post: BlogPostType): boolean {
  return !!(post.data_context?.affiliate_url || post.content?.includes('try.monday.com'))
}

function decorateAffiliateLinks(
  html: string,
  {
    affiliateUrl,
    highlightAffiliateLinks,
  }: {
    affiliateUrl: string
    highlightAffiliateLinks: boolean
  },
) {
  if (!highlightAffiliateLinks || !affiliateUrl || typeof DOMParser === 'undefined') {
    return html
  }

  try {
    const parser = new DOMParser()
    const doc = parser.parseFromString(`<div>${html}</div>`, 'text/html')
    const anchors = doc.querySelectorAll('a[href]')
    anchors.forEach((anchor) => {
      const href = String(anchor.getAttribute('href') || '').trim()
      if (!href || href !== affiliateUrl) return
      const existingStyle = String(anchor.getAttribute('style') || '').trim()
      anchor.setAttribute(
        'style',
        existingStyle ? `${existingStyle};${AFFILIATE_INLINE_STYLE}` : AFFILIATE_INLINE_STYLE,
      )
      anchor.setAttribute('data-preview-affiliate', 'true')
      anchor.setAttribute('title', 'Affiliate link')
    })
    return doc.body.innerHTML.replace(/^<div>|<\/div>$/g, '')
  } catch {
    return html
  }
}

function renderContentWithCharts(
  content: string,
  charts: ChartSpec[] | undefined,
  options: {
    affiliateUrl: string
    highlightAffiliateLinks: boolean
    highlightCharts: boolean
  },
) {
  if (!charts || charts.length === 0) {
    return (
      <div
        className="blog-prose"
        dangerouslySetInnerHTML={{
          __html: decorateAffiliateLinks(content, options),
        }}
      />
    )
  }

  const chartMap = new Map(charts.map((c) => [c.chart_id, c]))
  const parts = content.split(CHART_PLACEHOLDER_RE)

  return (
    <div className="blog-prose">
      {parts.map((part, i) => {
        const match = part.match(/^\{\{chart:([^}]+)\}\}$/)
        if (match) {
          const spec = chartMap.get(match[1])
          if (!spec) return null
          return (
            <div key={i} className="relative">
              {options.highlightCharts && (
                <div className="mb-2 inline-flex rounded-full border border-cyan-500/40 bg-cyan-500/10 px-2 py-0.5 text-[11px] font-medium text-cyan-300">
                  Chart Block
                </div>
              )}
              <Suspense fallback={null}>
                <BlogChart spec={spec} />
              </Suspense>
            </div>
          )
        }
        if (!part.trim()) return null
        return (
          <div
            key={i}
            dangerouslySetInnerHTML={{
              __html: decorateAffiliateLinks(part, options),
            }}
          />
        )
      })}
    </div>
  )
}

export default function BlogArticleView({
  post,
  relatedPosts = [],
  preview = false,
  previewViewport = 'desktop',
  showBackLink = true,
  highlightAffiliateLinks = false,
  highlightCharts = false,
  highlightCtas = false,
}: BlogArticleViewProps) {
  const cta = resolveBlogArticleCta(post)
  const showDisclosure = hasAffiliateContent(post)

  const renderedContent = useMemo(
    () =>
      renderContentWithCharts(post.content, post.charts, {
        affiliateUrl: String(post.data_context?.affiliate_url || '').trim(),
        highlightAffiliateLinks,
        highlightCharts,
      }),
    [post.content, post.charts, post.data_context, highlightAffiliateLinks, highlightCharts],
  )

  const containerClass = preview
    ? previewViewport === 'mobile'
      ? 'mx-auto max-w-[420px]'
      : 'mx-auto max-w-3xl'
    : 'max-w-3xl mx-auto'

  return (
    <article className={`${containerClass} px-6 pt-12 pb-24`}>
      {showBackLink && (
        <Link
          to="/blog"
          className="inline-flex items-center gap-1.5 text-sm text-slate-400 hover:text-white transition-colors mb-8"
        >
          Back to blog
        </Link>
      )}

      <header className="mb-10">
        <div className="flex flex-wrap items-center gap-2 mb-4">
          {post.tags.map((tag) => (
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

      {showDisclosure && (
        <div className="mb-8">
          {preview && highlightCtas && (
            <div className="mb-2 inline-flex rounded-full border border-cyan-500/40 bg-cyan-500/10 px-2 py-0.5 text-[11px] font-medium text-cyan-300">
              Affiliate Disclosure
            </div>
          )}
          <div className="px-4 py-3 bg-slate-800/40 border border-slate-700/40 rounded-lg text-xs text-slate-500">
            <strong>Disclosure:</strong> This article may contain affiliate links. If you purchase through these links, we may earn a commission at no additional cost to you. Our analysis and recommendations are based on verified review data, not affiliate relationships. See our{' '}
            <Link to="/methodology" className="text-cyan-500 hover:text-cyan-400 underline">methodology</Link>.
          </div>
        </div>
      )}

      {renderedContent}

      <div className="mt-16">
        {preview && highlightCtas && (
          <div className="mb-2 inline-flex rounded-full border border-cyan-500/40 bg-cyan-500/10 px-2 py-0.5 text-[11px] font-medium text-cyan-300">
            Rendered CTA
          </div>
        )}
        <div className="p-8 bg-slate-800/60 border border-slate-700/50 rounded-xl text-center">
          <h2 className="text-xl font-bold mb-2">{cta.headline}</h2>
          <p className="text-slate-400 mb-6">{cta.body}</p>
          <a
            href={cta.url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-block px-6 py-3 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-white font-semibold transition-colors"
          >
            {cta.buttonText}
          </a>
          {cta.mode === 'affiliate' && (
            <p className="mt-3 text-xs text-slate-600">Affiliate link — see disclosure above</p>
          )}
        </div>
      </div>

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

      {relatedPosts.length > 0 && (
        <section className="mt-8 border-t border-slate-700/50 pt-8">
          <h2 className="text-lg font-semibold mb-4">Related Analysis</h2>
          <div className="grid gap-3">
            {relatedPosts.map((rp) => (
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
  )
}
