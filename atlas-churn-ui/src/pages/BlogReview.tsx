import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  FileSearch,
  RefreshCw,
  CheckCircle2,
  Archive,
  X,
  ExternalLink,
  Loader2,
  ChevronRight,
} from 'lucide-react'
import { clsx } from 'clsx'
import useApiData from '../hooks/useApiData'
import DataTable from '../components/DataTable'
import StatCard from '../components/StatCard'
import UrgencyBadge from '../components/UrgencyBadge'
import BlogArticleView, { resolveBlogArticleCta } from '../components/BlogArticleView'
import BlogFailureExplanation from '../components/BlogFailureExplanation'
import type { Column } from '../components/DataTable'
import type { BlogDraftSummary, BlogDraft, BlogEvidence } from '../types'
import { POSTS } from '../content/blog'
import type { BlogPost as BlogPostType } from '../content/blog'
import {
  fetchBlogDrafts,
  fetchBlogDraftSummary,
  fetchBlogDraft,
  fetchBlogEvidence,
  publishBlogDraft,
  updateBlogDraft,
} from '../api/client'

type StatusFilter = 'draft' | 'published' | 'archived' | 'rejected' | 'failed' | ''
type ReviewTab = 'copy' | 'preview'
type PreviewViewport = 'desktop' | 'mobile'
type AffiliatePlacement = { href: string; text: string; placement: string }

function StatusBadge({ status }: { status: string }) {
  return (
    <span
      className={clsx(
        'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium',
        status === 'draft' && 'bg-amber-500/20 text-amber-400',
        status === 'published' && 'bg-green-500/20 text-green-400',
        status === 'rejected' && 'bg-rose-500/20 text-rose-400',
        status === 'failed' && 'bg-red-500/20 text-red-400',
        status === 'archived' && 'bg-slate-500/20 text-slate-400',
      )}
    >
      {status}
    </span>
  )
}

function TopicBadge({ type }: { type: string }) {
  const label = type.replace(/_/g, ' ')
  return (
    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-cyan-500/10 text-cyan-400">
      {label}
    </span>
  )
}

function derivePreviewPost(draft: BlogDraft): BlogPostType {
  const normalizedCta = draft.cta
    ? {
        headline: draft.cta.headline || '',
        body: draft.cta.body || '',
        button_text: draft.cta.button_text || '',
        report_type: draft.cta.report_type || '',
        vendor_filter: draft.cta.vendor_filter,
        category_filter: draft.cta.category_filter,
      }
    : null
  return {
    slug: draft.slug,
    title: draft.title,
    description: draft.description || '',
    date:
      String(draft.source_report_date || draft.created_at || '').slice(0, 10) ||
      new Date().toISOString().slice(0, 10),
    author: 'Churn Signals Team',
    tags: draft.tags || [],
    content: draft.content || '',
    charts: Array.isArray(draft.charts) ? draft.charts as BlogPostType['charts'] : [],
    topic_type: draft.topic_type,
    data_context: draft.data_context || {},
    seo_title: draft.seo_title || draft.title,
    seo_description: draft.seo_description || draft.description || '',
    target_keyword: draft.target_keyword || undefined,
    secondary_keywords: Array.isArray(draft.secondary_keywords) ? draft.secondary_keywords as string[] : [],
    faq: Array.isArray(draft.faq) ? draft.faq as BlogPostType['faq'] : [],
    related_slugs: Array.isArray(draft.related_slugs) ? draft.related_slugs : [],
    cta: normalizedCta,
  }
}

function extractLinks(html: string) {
  if (!html || typeof DOMParser === 'undefined') return []
  try {
    const parser = new DOMParser()
    const doc = parser.parseFromString(`<div>${html}</div>`, 'text/html')
    return Array.from(doc.querySelectorAll('a[href]')).map((anchor) => ({
      href: String(anchor.getAttribute('href') || '').trim(),
      text: String(anchor.textContent || '').trim(),
    })).filter((item) => item.href)
  } catch {
    return []
  }
}

export default function BlogReview() {
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('draft')
  const [selectedDraft, setSelectedDraft] = useState<BlogDraft | null>(null)
  const [activeTab, setActiveTab] = useState<ReviewTab>('copy')
  const [evidence, setEvidence] = useState<BlogEvidence[]>([])
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [publishing, setPublishing] = useState(false)
  const [archiving, setArchiving] = useState(false)
  const [notes, setNotes] = useState('')
  const [savingNotes, setSavingNotes] = useState(false)
  const [previewViewport, setPreviewViewport] = useState<PreviewViewport>('desktop')
  const [highlightAffiliateLinks, setHighlightAffiliateLinks] = useState(true)
  const [highlightCharts, setHighlightCharts] = useState(true)
  const [highlightCtas, setHighlightCtas] = useState(true)

  const { data: drafts, loading, error, refresh, refreshing } = useApiData(
    () => fetchBlogDrafts(statusFilter || undefined),
    [statusFilter],
  )
  const { data: summaryData, loading: summaryLoading } = useApiData(
    fetchBlogDraftSummary,
    [],
  )

  const openDraft = async (row: BlogDraftSummary) => {
    setLoadingDetail(true)
    setSelectedDraft(null)
    setEvidence([])
    setActiveTab('copy')
    try {
      const [detail, ev] = await Promise.all([
        fetchBlogDraft(row.id),
        fetchBlogEvidence(row.id),
      ])
      setSelectedDraft(detail)
      setEvidence(ev.reviews)
      setNotes(detail.reviewer_notes || '')
    } catch {
      // Fallback: show summary info in the detail panel
      setSelectedDraft({
        ...row,
        tags: [],
        content: '<p>Failed to load full content. Please try again.</p>',
        charts: [],
        data_context: null,
        reviewer_notes: null,
        source_report_date: null,
      })
    } finally {
      setLoadingDetail(false)
    }
  }

  const handlePublish = async () => {
    if (!selectedDraft) return
    setPublishing(true)
    try {
      await publishBlogDraft(selectedDraft.id)
      setSelectedDraft(null)
      refresh()
    } finally {
      setPublishing(false)
    }
  }

  const handleArchive = async () => {
    if (!selectedDraft) return
    setArchiving(true)
    try {
      await updateBlogDraft(selectedDraft.id, { status: 'archived' })
      setSelectedDraft(null)
      refresh()
    } finally {
      setArchiving(false)
    }
  }

  const handleSaveNotes = async () => {
    if (!selectedDraft) return
    setSavingNotes(true)
    try {
      await updateBlogDraft(selectedDraft.id, { reviewer_notes: notes })
    } finally {
      setSavingNotes(false)
    }
  }

  const previewPost = useMemo(
    () => (selectedDraft ? derivePreviewPost(selectedDraft) : null),
    [selectedDraft],
  )
  const relatedPreviewPosts = useMemo(() => {
    if (!previewPost?.related_slugs) return []
    return previewPost.related_slugs
      .map((slug) => POSTS.find((post) => post.slug === slug))
      .filter((post): post is BlogPostType => !!post)
  }, [previewPost])
  const renderedLinks = useMemo(
    () => extractLinks(previewPost?.content || ''),
    [previewPost?.content],
  )
  const renderedCta = useMemo(
    () => (previewPost ? resolveBlogArticleCta(previewPost) : null),
    [previewPost],
  )
  const affiliateUrl = String(selectedDraft?.data_context?.affiliate_url || '').trim()
  const affiliateLinks = useMemo<AffiliatePlacement[]>(
    () => {
      const placements = renderedLinks
        .filter((link) => affiliateUrl && link.href === affiliateUrl)
        .map((link) => ({
          href: link.href,
          text: link.text || '(untitled affiliate link)',
          placement: 'Body copy',
        }))
      if (renderedCta?.mode === 'affiliate' && affiliateUrl && renderedCta.url === affiliateUrl) {
        placements.push({
          href: renderedCta.url,
          text: renderedCta.buttonText || renderedCta.headline,
          placement: 'Rendered CTA',
        })
      }
      return placements
    },
    [renderedLinks, affiliateUrl, renderedCta],
  )
  const chartCount = Array.isArray(selectedDraft?.charts) ? selectedDraft?.charts.length : 0
  const storedCta = selectedDraft?.cta || null

  const columns: Column<BlogDraftSummary>[] = [
    {
      key: 'title',
      header: 'Title',
      render: (r) => (
        <div className="max-w-md">
          <span className="text-white font-medium line-clamp-1">{r.title}</span>
        </div>
      ),
      sortable: true,
      sortValue: (r) => r.title,
    },
    {
      key: 'topic_type',
      header: 'Type',
      render: (r) => <TopicBadge type={r.topic_type} />,
      sortable: true,
      sortValue: (r) => r.topic_type,
    },
    {
      key: 'status',
      header: 'Status',
      render: (r) => <StatusBadge status={r.status} />,
      sortable: true,
      sortValue: (r) => r.status,
    },
    {
      key: 'signals',
      header: 'Signals',
      render: (r) => (
        <div className="space-y-0.5 text-xs">
          {(r.quality_score != null || r.quality_threshold != null) && (
            <div className="text-slate-400">
              {r.quality_score != null ? r.quality_score : '--'}
              {r.quality_threshold != null ? ` / ${r.quality_threshold}` : ''}
            </div>
          )}
          {(r.blocker_count ?? 0) > 0 && <div className="text-red-400">{r.blocker_count} blocker{r.blocker_count === 1 ? '' : 's'}</div>}
          {(r.warning_count ?? 0) > 0 && <div className="text-amber-400">{r.warning_count} warn</div>}
          {(r.unresolved_issue_count ?? 0) > 0 && <div className="text-cyan-400">{r.unresolved_issue_count} open</div>}
        </div>
      ),
    },
    {
      key: 'created_at',
      header: 'Created',
      render: (r) => (
        <span className="text-sm text-slate-400">
          {r.created_at ? new Date(r.created_at).toLocaleDateString() : '--'}
        </span>
      ),
      sortable: true,
      sortValue: (r) => r.created_at || '',
    },
    {
      key: 'action',
      header: '',
      render: () => <ChevronRight className="h-4 w-4 text-slate-500" />,
    },
  ]

  const items = drafts ?? []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <FileSearch className="h-6 w-6 text-cyan-400" />
          <h1 className="text-2xl font-bold text-white">Blog Review</h1>
        </div>
        <div className="flex items-center gap-2">
          <Link
            to="/blog-diagnostics"
            className="flex items-center gap-2 px-3 py-2 text-sm text-slate-300 bg-slate-800/50 rounded-lg hover:bg-slate-700/50 transition-colors"
          >
            <ExternalLink className="h-4 w-4" />
            Diagnostics
          </Link>
          <button
            onClick={refresh}
            disabled={refreshing}
            className="flex items-center gap-2 px-3 py-2 text-sm text-slate-300 bg-slate-800/50 rounded-lg hover:bg-slate-700/50 transition-colors"
          >
            <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
            Refresh
          </button>
        </div>
      </div>

      {/* Status filter tabs */}
      <div className="flex gap-1 border-b border-slate-700/50">
        {([['', 'All'], ['draft', 'Draft'], ['rejected', 'Rejected'], ['failed', 'Failed'], ['published', 'Published'], ['archived', 'Archived']] as const).map(
          ([val, label]) => (
            <button
              key={val}
              onClick={() => setStatusFilter(val as StatusFilter)}
              className={clsx(
                'px-4 py-2 text-sm font-medium transition-colors border-b-2',
                statusFilter === val
                  ? 'text-cyan-400 border-cyan-400'
                  : 'text-slate-400 border-transparent hover:text-white',
              )}
            >
              {label}
            </button>
          ),
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Failing Drafts"
          value={summaryData?.quality?.failing ?? 0}
          icon={<FileSearch className="h-4 w-4" />}
          skeleton={summaryLoading}
        />
        <StatCard
          label="Warning Only"
          value={summaryData?.quality?.warning_only ?? 0}
          icon={<RefreshCw className="h-4 w-4" />}
          skeleton={summaryLoading}
        />
        <StatCard
          label="Clean Drafts"
          value={summaryData?.quality?.clean ?? 0}
          icon={<CheckCircle2 className="h-4 w-4" />}
          skeleton={summaryLoading}
        />
        <StatCard
          label="Open Issues"
          value={summaryData?.quality?.unresolved ?? 0}
          icon={<Archive className="h-4 w-4" />}
          skeleton={summaryLoading}
        />
      </div>

      {((summaryData?.quality?.top_blockers?.length ?? 0) > 0 || (summaryData?.quality?.by_failure_step?.length ?? 0) > 0) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
            <h3 className="text-sm font-medium text-white mb-3">Top Blog Blockers</h3>
            <div className="space-y-2">
              {(summaryData?.quality?.top_blockers ?? []).map((item) => (
                <div key={item.reason} className="flex items-start justify-between gap-3 text-sm">
                  <span className="text-slate-300 break-words">{item.reason}</span>
                  <span className="text-red-400 shrink-0">{item.count}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
            <h3 className="text-sm font-medium text-white mb-3">Failure Steps</h3>
            <div className="space-y-2">
              {(summaryData?.quality?.by_failure_step ?? []).map((item) => (
                <div key={item.step} className="flex items-center justify-between gap-3 text-sm">
                  <span className="text-slate-300">{item.step}</span>
                  <span className="text-cyan-400">{item.count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400 text-sm">
          {error.message}
        </div>
      )}

      {/* Table */}
      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden">
        {loading ? (
          <DataTable columns={columns} data={[]} skeletonRows={6} />
        ) : (
          <DataTable
            columns={columns}
            data={items}
            onRowClick={openDraft}
            emptyMessage="No blog posts match this filter"
          />
        )}
      </div>

      {/* Detail Modal */}
      {(selectedDraft || loadingDetail) && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-start justify-center pt-8 overflow-y-auto">
          <div className="bg-slate-900 border border-slate-700/50 rounded-xl w-full max-w-6xl mx-4 mb-8">
            {/* Modal header */}
            <div className="flex items-center justify-between p-5 border-b border-slate-700/50">
              <div className="flex-1 min-w-0">
                {loadingDetail ? (
                  <div className="h-6 w-64 bg-slate-700/50 rounded animate-pulse" />
                ) : (
                  <>
                    <h2 className="text-lg font-semibold text-white truncate">
                      {selectedDraft?.title}
                    </h2>
                    <div className="flex items-center gap-2 mt-1">
                      {selectedDraft && <TopicBadge type={selectedDraft.topic_type} />}
                      {selectedDraft && <StatusBadge status={selectedDraft.status} />}
                      {selectedDraft?.llm_model && (
                        <span className="text-xs text-slate-500">{selectedDraft.llm_model}</span>
                      )}
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2 text-xs">
                      {selectedDraft?.quality_score != null && (
                        <span className="text-slate-400">
                          quality {selectedDraft.quality_score}
                          {selectedDraft.quality_threshold != null ? ` / ${selectedDraft.quality_threshold}` : ''}
                        </span>
                      )}
                      {(selectedDraft?.blocker_count ?? 0) > 0 && (
                        <span className="text-red-400">{selectedDraft?.blocker_count} blocker{selectedDraft?.blocker_count === 1 ? '' : 's'}</span>
                      )}
                      {(selectedDraft?.warning_count ?? 0) > 0 && (
                        <span className="text-amber-400">{selectedDraft?.warning_count} warning{selectedDraft?.warning_count === 1 ? '' : 's'}</span>
                      )}
                      {(selectedDraft?.unresolved_issue_count ?? 0) > 0 && (
                        <span className="text-cyan-400">{selectedDraft?.unresolved_issue_count} open issue{selectedDraft?.unresolved_issue_count === 1 ? '' : 's'}</span>
                      )}
                    </div>
                  </>
                )}
              </div>
              <button
                onClick={() => setSelectedDraft(null)}
                className="ml-4 text-slate-400 hover:text-white"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {loadingDetail ? (
              <div className="p-10 flex items-center justify-center">
                <Loader2 className="h-8 w-8 text-cyan-400 animate-spin" />
              </div>
            ) : selectedDraft ? (
              <div className="grid grid-cols-1 lg:grid-cols-5 divide-y lg:divide-y-0 lg:divide-x divide-slate-700/50">
                {/* Left: Blog content */}
                <div className="lg:col-span-3 p-5 max-h-[70vh] overflow-y-auto">
                  <div className="mb-4 flex items-center gap-1 border-b border-slate-700/50">
                    {([
                      ['copy', 'Copy'],
                      ['preview', 'Preview'],
                    ] as const).map(([key, label]) => (
                      <button
                        key={key}
                        onClick={() => setActiveTab(key)}
                        className={clsx(
                          'px-3 py-2 text-sm font-medium border-b-2 transition-colors',
                          activeTab === key
                            ? 'border-cyan-400 text-cyan-400'
                            : 'border-transparent text-slate-400 hover:text-white',
                        )}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                  {(selectedDraft.rejection_reason || selectedDraft.latest_error_summary) && (
                    <div className="mb-4 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-200">
                      <div className="font-medium">
                        {selectedDraft.status === 'rejected' ? 'Rejected' : 'Latest failure'}
                      </div>
                      <div className="mt-1">
                        {selectedDraft.rejection_reason || selectedDraft.latest_error_summary}
                      </div>
                      {selectedDraft.latest_failure_step && (
                        <div className="mt-1 text-xs text-rose-300/80">
                          step: {selectedDraft.latest_failure_step}
                        </div>
                      )}
                    </div>
                  )}
                  <BlogFailureExplanation explanation={selectedDraft.failure_explanation} />
                  {activeTab === 'copy' ? (
                    <div
                      className="prose prose-invert prose-sm max-w-none"
                      dangerouslySetInnerHTML={{
                        __html: selectedDraft.content,
                      }}
                    />
                  ) : previewPost ? (
                    <div className="space-y-4">
                      <div className="flex flex-wrap items-center gap-4 rounded-lg border border-slate-700/50 bg-slate-950/40 px-4 py-3 text-xs text-slate-400">
                        <div className="flex items-center gap-2">
                          <span>Viewport</span>
                          <div className="flex rounded-lg border border-slate-700/50 overflow-hidden">
                            {([
                              ['desktop', 'Desktop'],
                              ['mobile', 'Mobile'],
                            ] as const).map(([value, label]) => (
                              <button
                                key={value}
                                onClick={() => setPreviewViewport(value)}
                                className={clsx(
                                  'px-2.5 py-1 transition-colors',
                                  previewViewport === value
                                    ? 'bg-cyan-500/20 text-cyan-300'
                                    : 'text-slate-400 hover:text-white',
                                )}
                              >
                                {label}
                              </button>
                            ))}
                          </div>
                        </div>
                        <label className="flex items-center gap-1.5 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={highlightAffiliateLinks}
                            onChange={(e) => setHighlightAffiliateLinks(e.target.checked)}
                            className="accent-cyan-500"
                          />
                          Highlight affiliate links
                        </label>
                        <label className="flex items-center gap-1.5 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={highlightCharts}
                            onChange={(e) => setHighlightCharts(e.target.checked)}
                            className="accent-cyan-500"
                          />
                          Label charts
                        </label>
                        <label className="flex items-center gap-1.5 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={highlightCtas}
                            onChange={(e) => setHighlightCtas(e.target.checked)}
                            className="accent-cyan-500"
                          />
                          Label CTA blocks
                        </label>
                      </div>
                      <div className="rounded-2xl border border-slate-700/50 bg-slate-950/30 p-4">
                        <BlogArticleView
                          post={previewPost}
                          relatedPosts={relatedPreviewPosts}
                          preview
                          previewViewport={previewViewport}
                          showBackLink={false}
                          highlightAffiliateLinks={highlightAffiliateLinks}
                          highlightCharts={highlightCharts}
                          highlightCtas={highlightCtas}
                        />
                      </div>
                    </div>
                  ) : null}
                </div>

                {/* Right: Evidence + actions */}
                <div className="lg:col-span-2 p-5 max-h-[70vh] overflow-y-auto space-y-4">
                  {/* Actions */}
                  {selectedDraft.status === 'draft' && (
                    <div className="flex items-center gap-2">
                      <button
                        onClick={handlePublish}
                        disabled={publishing}
                        className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-green-600 hover:bg-green-500 text-white rounded-lg transition-colors disabled:opacity-50"
                      >
                        {publishing ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />}
                        Publish
                      </button>
                      <button
                        onClick={handleArchive}
                        disabled={archiving}
                        className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded-lg transition-colors disabled:opacity-50"
                      >
                        {archiving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Archive className="h-4 w-4" />}
                        Archive
                      </button>
                    </div>
                  )}

                  {/* Reviewer notes */}
                  <div>
                    <label className="text-xs text-slate-400 block mb-1">Reviewer Notes</label>
                    <textarea
                      value={notes}
                      onChange={(e) => setNotes(e.target.value)}
                      rows={2}
                      className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg p-2 text-sm text-slate-200 placeholder-slate-500 resize-none focus:outline-none focus:border-cyan-500/50"
                      placeholder="Add notes..."
                    />
                    <button
                      onClick={handleSaveNotes}
                      disabled={savingNotes}
                      className="mt-1 text-xs text-cyan-400 hover:text-cyan-300 disabled:opacity-50"
                    >
                      {savingNotes ? 'Saving...' : 'Save Notes'}
                    </button>
                  </div>

                  {activeTab === 'preview' && selectedDraft && (
                    <div className="rounded-xl border border-slate-700/50 bg-slate-800/40 p-4 space-y-3">
                      <h3 className="text-sm font-semibold text-white">Prepublish Map</h3>
                      <div className="grid grid-cols-2 gap-3 text-xs">
                        <div>
                          <div className="text-slate-500">Charts</div>
                          <div className="text-slate-200">{chartCount}</div>
                        </div>
                        <div>
                          <div className="text-slate-500">Affiliate Placements</div>
                          <div className="text-slate-200">{affiliateLinks.length}</div>
                        </div>
                        <div>
                          <div className="text-slate-500">Rendered CTA</div>
                          <div className="text-slate-200">
                            {renderedCta?.mode === 'affiliate' ? 'Affiliate CTA' : 'Generic CTA'}
                          </div>
                        </div>
                        <div>
                          <div className="text-slate-500">Stored CTA Payload</div>
                          <div className="text-slate-200">{storedCta ? 'Present' : 'None'}</div>
                        </div>
                      </div>

                      {affiliateUrl && (
                        <div>
                          <div className="text-xs text-slate-500 mb-1">Affiliate Destination</div>
                          <a
                            href={affiliateUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs text-cyan-400 break-all hover:text-cyan-300"
                          >
                            {affiliateUrl}
                          </a>
                        </div>
                      )}

                      {affiliateLinks.length > 0 && (
                        <div>
                          <div className="text-xs text-slate-500 mb-1">Affiliate Link Placements</div>
                          <div className="space-y-2">
                            {affiliateLinks.map((link, index) => (
                              <div
                                key={`${link.href}-${index}`}
                                className="rounded-lg border border-cyan-500/20 bg-cyan-500/5 p-2"
                              >
                                <div className="text-[11px] text-slate-500">{link.placement}</div>
                                <div className="text-xs text-slate-200 line-clamp-1">
                                  {link.text}
                                </div>
                                <div className="text-[11px] text-cyan-400 break-all">{link.href}</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {storedCta && (
                        <div>
                          <div className="text-xs text-slate-500 mb-1">Stored CTA Payload</div>
                          <div className="rounded-lg border border-slate-700/50 bg-slate-900/50 p-3 space-y-1 text-xs">
                            <div className="text-slate-200">{storedCta.headline || '(no headline)'}</div>
                            {storedCta.body && <div className="text-slate-400">{storedCta.body}</div>}
                            {storedCta.button_text && (
                              <div className="text-cyan-400">Button: {storedCta.button_text}</div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Source Evidence */}
                  <div>
                    <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                      <ExternalLink className="h-4 w-4 text-cyan-400" />
                      Source Evidence ({evidence.length} reviews)
                    </h3>
                    {evidence.length === 0 ? (
                      <p className="text-sm text-slate-500">No matching reviews found for this vendor.</p>
                    ) : (
                      <div className="space-y-3">
                        {evidence.map((ev) => (
                          <div
                            key={ev.id}
                            className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-3"
                          >
                            <div className="flex items-start justify-between gap-2 mb-1">
                              <span className="text-sm font-medium text-white line-clamp-1">
                                {ev.headline || ev.vendor_name}
                              </span>
                              <UrgencyBadge score={ev.urgency_score} />
                            </div>
                            <div className="flex items-center gap-2 text-xs text-slate-400 mb-2">
                              {ev.reviewer_company && <span>{ev.reviewer_company}</span>}
                              {ev.source_site && (
                                <>
                                  <span className="text-slate-600">|</span>
                                  <span>{ev.source_site}</span>
                                </>
                              )}
                              {ev.review_date && (
                                <>
                                  <span className="text-slate-600">|</span>
                                  <span>{new Date(ev.review_date).toLocaleDateString()}</span>
                                </>
                              )}
                            </div>
                            {ev.full_text && (
                              <p className="text-xs text-slate-300 line-clamp-3">{ev.full_text}</p>
                            )}
                            {ev.pain_categories.length > 0 && (
                              <div className="flex flex-wrap gap-1 mt-2">
                                {ev.pain_categories.slice(0, 3).map((cat) => (
                                  <span
                                    key={cat}
                                    className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700/50 text-slate-300"
                                  >
                                    {cat}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  )
}
