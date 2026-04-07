import { useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import PublicLayout from '../components/PublicLayout'
import SeoHead from '../components/SeoHead'
import BlogArticleView from '../components/BlogArticleView'
import { loadPostBySlug, loadPostsBySlugs } from '../content/blog'
import type { BlogPost as BlogPostType } from '../content/blog'

export default function BlogPost() {
  const { slug } = useParams<{ slug: string }>()
  const [post, setPost] = useState<BlogPostType | null | undefined>(undefined)
  const [relatedPosts, setRelatedPosts] = useState<BlogPostType[]>([])

  useEffect(() => {
    let cancelled = false
    setPost(undefined)
    setRelatedPosts([])

    if (!slug) {
      setPost(null)
      return () => {
        cancelled = true
      }
    }

    loadPostBySlug(slug).then((loadedPost) => {
      if (cancelled) return
      setPost(loadedPost)
      if (!loadedPost?.related_slugs?.length) return
      loadPostsBySlugs(loadedPost.related_slugs).then((loadedRelatedPosts) => {
        if (cancelled) return
        setRelatedPosts(loadedRelatedPosts)
      })
    })

    return () => {
      cancelled = true
    }
  }, [slug])

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
    </PublicLayout>
  )
}
