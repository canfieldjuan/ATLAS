import { useEffect, lazy, Suspense } from 'react'
import { Link } from 'react-router-dom'
import PublicLayout from '../components/PublicLayout'
import BlogCardVisual from '../components/BlogCardVisual'
import { POSTS } from '../content/blog'

const AtlasRobotScene = lazy(() => import('../components/AtlasRobotScene'))

function formatDate(iso: string) {
  return new Date(iso + 'T00:00:00').toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })
}

export default function Blog() {
  useEffect(() => { document.title = 'Blog | Atlas Intelligence' }, [])

  return (
    <PublicLayout>
      {/* Hero */}
      <section className="max-w-4xl mx-auto px-6 pt-16 pb-12 text-center">
        <Suspense fallback={null}>
          <AtlasRobotScene />
        </Suspense>
        <h1 className="text-4xl sm:text-5xl font-bold mt-4">Blog</h1>
        <p className="mt-4 text-lg text-slate-400 max-w-2xl mx-auto">
          Amazon seller intelligence, review monitoring strategies, and competitive analysis insights.
        </p>
      </section>

      {/* Post grid */}
      <section className="max-w-5xl mx-auto px-6 pb-24">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {POSTS.map(post => (
            <Link
              key={post.slug}
              to={`/blog/${post.slug}`}
              className="group bg-slate-800/60 border border-slate-700/50 rounded-xl overflow-hidden hover:border-slate-600 transition-colors"
            >
              <BlogCardVisual post={post} />
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
      </section>
    </PublicLayout>
  )
}
