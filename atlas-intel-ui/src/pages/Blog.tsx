import { lazy, Suspense, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import PublicLayout from '../components/PublicLayout'
import BlogCardVisual from '../components/BlogCardVisual'
import SeoHead from '../components/SeoHead'
import { POSTS } from '../content/blog'
import type { BlogPost } from '../content/blog'
import { fetchPublicBlogPosts } from '../api/blog'

const AtlasRobotScene = lazy(() => import('../components/AtlasRobotScene'))

function formatDate(iso: string) {
  return new Date(iso + 'T00:00:00').toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })
}

const BLOG_DESCRIPTION = 'Amazon seller intelligence, review monitoring strategies, and competitive analysis insights.'

export default function Blog() {
  const [posts, setPosts] = useState<BlogPost[]>(POSTS)

  useEffect(() => {
    let cancelled = false
    fetchPublicBlogPosts()
      .then((generatedPosts) => {
        if (!cancelled) setPosts(mergeBlogPosts(generatedPosts, POSTS))
      })
      .catch(() => {
        if (!cancelled) setPosts(POSTS)
      })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <>
      <SeoHead
        title="Blog | Atlas Intelligence"
        description={BLOG_DESCRIPTION}
        canonical="https://atlas-intel-ui-two.vercel.app/blog"
        ogType="website"
      />
      <PublicLayout>
        {/* Hero */}
        <section className="max-w-4xl mx-auto px-6 pt-16 pb-12 text-center">
          <Suspense fallback={null}>
            <AtlasRobotScene />
          </Suspense>
          <h1 className="text-4xl sm:text-5xl font-bold mt-4">Blog</h1>
          <p className="mt-4 text-lg text-slate-400 max-w-2xl mx-auto">
            {BLOG_DESCRIPTION}
          </p>
        </section>

        {/* Post grid */}
        <section className="max-w-5xl mx-auto px-6 pb-24">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {posts.map(post => (
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
    </>
  )
}

function mergeBlogPosts(generatedPosts: BlogPost[], staticPosts: BlogPost[]): BlogPost[] {
  const bySlug = new Map<string, BlogPost>()
  for (const post of staticPosts) bySlug.set(post.slug, post)
  for (const post of generatedPosts) bySlug.set(post.slug, post)
  return [...bySlug.values()].sort((a, b) => b.date.localeCompare(a.date))
}
