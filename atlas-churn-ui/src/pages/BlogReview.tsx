import { useState } from 'react'
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
import { marked } from 'marked'
import useApiData from '../hooks/useApiData'
import DataTable from '../components/DataTable'
import UrgencyBadge from '../components/UrgencyBadge'
import type { Column } from '../components/DataTable'
import type { BlogDraftSummary, BlogDraft, BlogEvidence } from '../types'
import {
  fetchBlogDrafts,
  fetchBlogDraft,
  fetchBlogEvidence,
  publishBlogDraft,
  updateBlogDraft,
} from '../api/client'

type StatusFilter = 'draft' | 'published' | 'archived' | ''

function StatusBadge({ status }: { status: string }) {
  return (
    <span
      className={clsx(
        'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium',
        status === 'draft' && 'bg-amber-500/20 text-amber-400',
        status === 'published' && 'bg-green-500/20 text-green-400',
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

export default function BlogReview() {
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('draft')
  const [selectedDraft, setSelectedDraft] = useState<BlogDraft | null>(null)
  const [evidence, setEvidence] = useState<BlogEvidence[]>([])
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [publishing, setPublishing] = useState(false)
  const [archiving, setArchiving] = useState(false)
  const [notes, setNotes] = useState('')
  const [savingNotes, setSavingNotes] = useState(false)

  const { data: drafts, loading, error, refresh, refreshing } = useApiData(
    () => fetchBlogDrafts(statusFilter || undefined),
    [statusFilter],
  )

  const openDraft = async (row: BlogDraftSummary) => {
    setLoadingDetail(true)
    setSelectedDraft(null)
    setEvidence([])
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
        <button
          onClick={refresh}
          disabled={refreshing}
          className="flex items-center gap-2 px-3 py-2 text-sm text-slate-300 bg-slate-800/50 rounded-lg hover:bg-slate-700/50 transition-colors"
        >
          <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
          Refresh
        </button>
      </div>

      {/* Status filter tabs */}
      <div className="flex gap-1 border-b border-slate-700/50">
        {([['', 'All'], ['draft', 'Draft'], ['published', 'Published'], ['archived', 'Archived']] as const).map(
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
                  <div
                    className="prose prose-invert prose-sm max-w-none"
                    dangerouslySetInnerHTML={{
                      __html: marked.parse(selectedDraft.content, { async: false }) as string,
                    }}
                  />
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
