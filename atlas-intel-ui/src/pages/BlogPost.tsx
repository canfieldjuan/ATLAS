import { useParams, Link } from 'react-router-dom'
import { useMemo, useEffect } from 'react'
import { marked } from 'marked'
import { ArrowLeft } from 'lucide-react'
import PublicLayout from '../components/PublicLayout'
import { POSTS } from '../content/blog'

function formatDate(iso: string) {
  return new Date(iso + 'T00:00:00').toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })
}

export default function BlogPost() {
  const { slug } = useParams<{ slug: string }>()
  const post = POSTS.find(p => p.slug === slug)

  const html = useMemo(() => {
    if (!post) return ''
    return marked.parse(post.content, { async: false }) as string
  }, [post])

  useEffect(() => {
    document.title = post
      ? `${post.title} | Atlas Intelligence`
      : 'Post Not Found | Atlas Intelligence'
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

        {/* Rendered markdown */}
        <div
          className="blog-prose"
          dangerouslySetInnerHTML={{ __html: html }}
        />

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
      </article>
    </PublicLayout>
  )
}
