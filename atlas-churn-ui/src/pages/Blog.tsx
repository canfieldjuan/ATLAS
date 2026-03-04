import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import PublicLayout from '../components/PublicLayout'
import { POSTS } from '../content/blog'

function formatDate(iso: string) {
  return new Date(iso + 'T00:00:00').toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })
}

export default function Blog() {
  useEffect(() => { document.title = 'Blog | Churn Intel' }, [])

  return (
    <PublicLayout>
      {/* Hero */}
      <section className="max-w-4xl mx-auto px-6 pt-20 pb-12 text-center">
        <h1 className="text-4xl sm:text-5xl font-bold">Blog</h1>
        <p className="mt-4 text-lg text-slate-400 max-w-2xl mx-auto">
          B2B software churn intelligence, vendor comparisons, and migration analysis backed by real enterprise review data.
        </p>
      </section>

      {/* Post grid */}
      <section className="max-w-5xl mx-auto px-6 pb-24">
        {POSTS.length === 0 ? (
          <div className="text-center py-16 text-slate-500">
            <p className="text-lg">No posts yet. Check back soon.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {POSTS.map(post => (
              <Link
                key={post.slug}
                to={`/blog/${post.slug}`}
                className="group bg-slate-800/60 border border-slate-700/50 rounded-xl overflow-hidden hover:border-slate-600 transition-colors"
              >
                {/* Gradient placeholder */}
                <div className="h-40 bg-gradient-to-br from-cyan-900/40 to-slate-800" />
                <div className="p-6">
                  <div className="flex items-center gap-2 mb-3">
                    <time className="text-xs text-slate-500">{formatDate(post.date)}</time>
                    {post.tags.slice(0, 2).map(tag => (
                      <span
                        key={tag}
                        className="text-xs px-2 py-0.5 bg-slate-700/60 text-slate-400 rounded-full"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                  <h2 className="text-lg font-semibold group-hover:text-cyan-400 transition-colors">
                    {post.title}
                  </h2>
                  <p className="mt-2 text-sm text-slate-400 line-clamp-2">{post.description}</p>
                </div>
              </Link>
            ))}
          </div>
        )}
      </section>
    </PublicLayout>
  )
}
