import { useMemo } from 'react'
import { useParams } from 'react-router-dom'
import PublicLayout from '../components/PublicLayout'
import SeoHead from '../components/SeoHead'
import BlogArticleView from '../components/BlogArticleView'
import useApiData from '../hooks/useApiData'
import { loadPostBySlug, loadPostsBySlugs } from '../content/blog'
import type { BlogPost as BlogPostType } from '../content/blog'
import { buildLoginRedirectPath, buildSignupRedirectPath } from '../auth/redirects'

const RETENTION_SIGNUP = buildSignupRedirectPath('/watchlists', 'b2b_retention')
const CHALLENGER_SIGNUP = buildSignupRedirectPath('/challengers', 'b2b_challenger')
const RETENTION_LOGIN = buildLoginRedirectPath('/watchlists', 'b2b_retention')
const CHALLENGER_LOGIN = buildLoginRedirectPath('/challengers', 'b2b_challenger')

export default function BlogPost() {
  const { slug } = useParams<{ slug: string }>()
  const { data } = useApiData(
    async (): Promise<{ post: BlogPostType | null; relatedPosts: BlogPostType[] }> => {
      if (!slug) {
        return { post: null, relatedPosts: [] }
      }
      const post = await loadPostBySlug(slug)
      if (!post?.related_slugs?.length) {
        return { post, relatedPosts: [] }
      }
      const relatedPosts = await loadPostsBySlugs(post.related_slugs)
      return { post, relatedPosts }
    },
    [slug],
    { refreshOnFocus: false, refreshOnReconnect: false },
  )
  const post = data?.post
  const relatedPosts = data?.relatedPosts ?? []

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
      image: 'https://churnsignals.co/og-default.png',
      author: {
        '@type': 'Organization',
        name: 'Churn Signals',
        sameAs: ['https://twitter.com/churnsignals', 'https://www.linkedin.com/company/churn-signals'],
      },
      publisher: {
        '@type': 'Organization',
        name: 'Churn Signals',
        url: 'https://churnsignals.co',
        sameAs: ['https://twitter.com/churnsignals', 'https://www.linkedin.com/company/churn-signals'],
      },
      mainEntityOfPage: {
        '@type': 'WebPage',
        '@id': `https://churnsignals.co/blog/${post.slug}`,
      },
      isBasedOn: {
        '@type': 'Dataset',
        name: 'B2B SaaS Review Intelligence',
        description: 'Aggregated from G2, Capterra, TrustRadius, and Reddit',
      },
      keywords: seoKeywords?.join(', ') || '',
    }

    if (post.topic_type === 'migration_guide' && post.content) {
      const stepRegex = /<li>\s*<strong>([^<]+)<\/strong>/g
      const steps: { '@type': string; name: string; position: number }[] = []
      let match: RegExpExecArray | null
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

  if (post === undefined) {
    return (
      <PublicLayout>
        <section className="max-w-3xl mx-auto px-6 py-24 text-center">
          <div className="h-10 w-48 mx-auto rounded bg-slate-800/50 animate-pulse" />
          <div className="mt-6 h-4 w-72 mx-auto rounded bg-slate-800/40 animate-pulse" />
        </section>
      </PublicLayout>
    )
  }

  if (!post) {
    return (
      <PublicLayout>
        <section className="max-w-3xl mx-auto px-6 py-24 text-center">
          <h1 className="text-3xl font-bold mb-4">Post not found</h1>
          <p className="text-slate-400 mb-8">The article you are looking for does not exist.</p>
        </section>
      </PublicLayout>
    )
  }

  return (
    <PublicLayout>
      <SeoHead
        title={`${post.seo_title || post.title} | Churn Signals`}
        description={post.seo_description || post.description}
        canonical={`https://churnsignals.co/blog/${post.slug}`}
        keywords={seoKeywords}
        faq={post.faq}
        jsonLd={jsonLd}
      />
      <BlogArticleView post={post} relatedPosts={relatedPosts} />
      <section className="max-w-3xl mx-auto px-6 pb-24">
        <div className="rounded-2xl border border-slate-700/50 bg-slate-800/40 p-6">
          <h2 className="text-2xl font-semibold">Put this analysis into the product workflow</h2>
          <p className="mt-3 text-slate-400">
            Read publicly. Operate privately. Move from article context into the watchlists or challenger workflow that matches the job.
          </p>
          <div className="mt-6 flex flex-wrap items-center gap-3">
            <a
              href={RETENTION_SIGNUP}
              className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-white font-medium transition-colors"
            >
              Start Vendor Retention
            </a>
            <a
              href={CHALLENGER_SIGNUP}
              className="px-4 py-2 border border-amber-500/40 bg-amber-500/10 hover:border-amber-400/50 hover:bg-amber-500/15 rounded-lg text-amber-100 font-medium transition-colors"
            >
              Start Challenger Lead Gen
            </a>
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-4 text-sm text-slate-400">
            <a href={RETENTION_LOGIN} className="hover:text-white transition-colors">
              Sign in to Watchlists
            </a>
            <a href={CHALLENGER_LOGIN} className="hover:text-white transition-colors">
              Sign in to Challengers
            </a>
          </div>
        </div>
      </section>
    </PublicLayout>
  )
}
