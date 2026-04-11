import React, { startTransition, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import PublicLayout from '../components/PublicLayout'
import SeoHead from '../components/SeoHead'
import { loadAllPosts } from '../content/blog'
import type { BlogPost } from '../content/blog'

const AtlasRobotScene = React.lazy(() => import('../components/AtlasRobotScene'))

const RETENTION_SIGNUP = `/signup?${new URLSearchParams({
  product: 'b2b_retention',
  redirect_to: '/watchlists',
}).toString()}`
const CHALLENGER_SIGNUP = `/signup?${new URLSearchParams({
  product: 'b2b_challenger',
  redirect_to: '/challengers',
}).toString()}`
const RETENTION_LOGIN = `/login?${new URLSearchParams({
  redirect_to: '/watchlists',
  product: 'b2b_retention',
}).toString()}`
const CHALLENGER_LOGIN = `/login?${new URLSearchParams({
  redirect_to: '/challengers',
  product: 'b2b_challenger',
}).toString()}`

function formatDate(iso: string) {
  return new Date(iso + 'T00:00:00').toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })
}

/* ---- Category visual config ---- */
const CATEGORY_STYLES: Record<string, { gradient: string; accent: string }> = {
  'CRM':                  { gradient: 'from-blue-600/30 via-indigo-700/20 to-slate-900', accent: 'text-blue-400' },
  'Helpdesk':             { gradient: 'from-emerald-600/30 via-teal-700/20 to-slate-900', accent: 'text-emerald-400' },
  'Cybersecurity':        { gradient: 'from-red-600/30 via-rose-700/20 to-slate-900', accent: 'text-red-400' },
  'Cloud Infrastructure': { gradient: 'from-violet-600/30 via-purple-700/20 to-slate-900', accent: 'text-violet-400' },
  'Project Management':   { gradient: 'from-amber-600/30 via-orange-700/20 to-slate-900', accent: 'text-amber-400' },
  'Marketing Automation': { gradient: 'from-pink-600/30 via-fuchsia-700/20 to-slate-900', accent: 'text-pink-400' },
  'Data & Analytics':     { gradient: 'from-sky-600/30 via-cyan-700/20 to-slate-900', accent: 'text-sky-400' },
  'Communication':        { gradient: 'from-teal-600/30 via-cyan-700/20 to-slate-900', accent: 'text-teal-400' },
  'HR / HCM':             { gradient: 'from-lime-600/30 via-green-700/20 to-slate-900', accent: 'text-lime-400' },
  'E-commerce':           { gradient: 'from-orange-600/30 via-amber-700/20 to-slate-900', accent: 'text-orange-400' },
}
const DEFAULT_STYLE = { gradient: 'from-cyan-600/30 via-slate-700/20 to-slate-900', accent: 'text-cyan-400' }

/* ---- Topic type icons (inline SVG) ---- */
function TopicIcon({ type }: { type?: string }) {
  const cls = 'w-8 h-8 opacity-80'
  switch (type) {
    case 'vendor_deep_dive':
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="11" cy="11" r="7" /><path d="m21 21-4.35-4.35" />
          <path d="M11 8v6M8 11h6" />
        </svg>
      )
    case 'migration_guide':
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M5 12h14M12 5l7 7-7 7" />
        </svg>
      )
    case 'vendor_showdown':
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M6 6l4 12M18 6l-4 12M8 12h8" />
        </svg>
      )
    case 'pricing_reality_check':
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <line x1="12" y1="1" x2="12" y2="23" /><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
        </svg>
      )
    default:
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" />
        </svg>
      )
  }
}

const TOPIC_LABELS: Record<string, string> = {
  vendor_deep_dive: 'Deep Dive',
  migration_guide: 'Migration Guide',
  vendor_showdown: 'Comparison',
  pricing_reality_check: 'Pricing Analysis',
}

function CardHeader({ post }: { post: BlogPost }) {
  const category = post.tags[0] || ''
  const style = CATEGORY_STYLES[category] || DEFAULT_STYLE
  const topicLabel = TOPIC_LABELS[post.topic_type || ''] || 'Analysis'

  return (
    <div className={`relative h-44 bg-gradient-to-br ${style.gradient} overflow-hidden`}>
      {/* Decorative grid pattern */}
      <div
        className="absolute inset-0 opacity-[0.06]"
        style={{
          backgroundImage: 'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)',
          backgroundSize: '24px 24px',
        }}
      />
      {/* Decorative circle */}
      <div className="absolute -right-8 -top-8 w-32 h-32 rounded-full bg-white/[0.04]" />
      <div className="absolute -right-4 -top-4 w-20 h-20 rounded-full bg-white/[0.03]" />

      {/* Content */}
      <div className="relative h-full flex flex-col justify-between p-5">
        <div className="flex items-center justify-between">
          <span className={`text-xs font-semibold uppercase tracking-wider ${style.accent}`}>
            {category}
          </span>
          <span className="text-[10px] font-medium uppercase tracking-wider text-slate-400 bg-slate-800/60 px-2 py-0.5 rounded">
            {topicLabel}
          </span>
        </div>
        <div className={`self-end ${style.accent}`}>
          <TopicIcon type={post.topic_type} />
        </div>
      </div>
    </div>
  )
}

export default function Blog() {
  const [posts, setPosts] = useState<BlogPost[] | null>(null)
  const [loadFailed, setLoadFailed] = useState(false)

  useEffect(() => {
    let cancelled = false
    loadAllPosts()
      .then((loadedPosts) => {
        if (cancelled) return
        startTransition(() => {
          setPosts(loadedPosts)
          setLoadFailed(false)
        })
      })
      .catch(() => {
        if (cancelled) return
        setLoadFailed(true)
      })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <PublicLayout>
      <SeoHead
        title="B2B Software Churn Intelligence Blog | Churn Signals"
        description="Data-backed analysis of B2B software vendor churn signals, migration patterns, and competitive intelligence from real enterprise reviews."
        canonical="https://churnsignals.co/blog"
        ogType="website"
      />
      {/* Hero */}
      <section className="max-w-4xl mx-auto px-6 pt-16 pb-12 text-center">
        <React.Suspense fallback={<div className="h-40" />}>
          <AtlasRobotScene />
        </React.Suspense>
        <h1 className="mt-4 text-4xl sm:text-5xl font-bold">Blog</h1>
        <p className="mt-4 text-lg text-slate-400 max-w-2xl mx-auto">
          B2B software churn intelligence, vendor comparisons, and migration analysis backed by real enterprise review data.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Link
            to={RETENTION_SIGNUP}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-white font-medium transition-colors"
          >
            Start Vendor Retention
          </Link>
          <Link
            to={CHALLENGER_SIGNUP}
            className="px-4 py-2 border border-amber-500/40 bg-amber-500/10 hover:border-amber-400/50 hover:bg-amber-500/15 rounded-lg text-amber-100 font-medium transition-colors"
          >
            Start Challenger Lead Gen
          </Link>
        </div>
        <div className="mt-3 flex flex-wrap items-center justify-center gap-4 text-sm text-slate-400">
          <Link to={RETENTION_LOGIN} className="hover:text-white transition-colors">
            Sign in to Watchlists
          </Link>
          <Link to={CHALLENGER_LOGIN} className="hover:text-white transition-colors">
            Sign in to Challengers
          </Link>
          <Link to="/methodology" className="hover:text-white transition-colors">
            Read Methodology
          </Link>
        </div>
      </section>

      {/* Post grid */}
      <section className="max-w-5xl mx-auto px-6 pb-24">
        {posts === null && !loadFailed ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Array.from({ length: 6 }).map((_, index) => (
              <div
                key={index}
                className="h-80 rounded-xl border border-slate-700/50 bg-slate-800/40 animate-pulse"
              />
            ))}
          </div>
        ) : loadFailed ? (
          <div className="text-center py-16 text-slate-500">
            <p className="text-lg">Blog posts failed to load. Try refreshing.</p>
          </div>
        ) : posts && posts.length === 0 ? (
          <div className="text-center py-16 text-slate-500">
            <p className="text-lg">No posts yet. Check back soon.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {posts?.map(post => (
              <Link
                key={post.slug}
                to={`/blog/${post.slug}`}
                className="group bg-slate-800/60 border border-slate-700/50 rounded-xl overflow-hidden hover:border-slate-600 transition-colors"
              >
                <CardHeader post={post} />
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
