import { type KeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  ArrowRight,
  CheckCircle2,
  Download,
  Eye,
  Loader2,
  RefreshCw,
  X,
  XCircle,
} from 'lucide-react'
import { clsx } from 'clsx'
import {
  exportGeneratedAssetDraftsCsv,
  fetchGeneratedAssetDrafts,
  reviewGeneratedAssetDraft,
  reviewGeneratedAssetDrafts,
  type GeneratedAssetDraft,
  type GeneratedAssetListResponse,
  type GeneratedAssetType,
} from '../api/contentOps'
import { PageError } from '../components/ErrorBoundary'

type StatusFilter = 'draft' | 'approved' | 'rejected' | 'all'

type AssetPreview = {
  heading: string
  body: string
  meta: string[]
}

type AssetFact = {
  label: string
  value: string
}

const ASSETS: Array<{
  id: GeneratedAssetType
  label: string
  description: string
}> = [
  {
    id: 'report',
    label: 'Reports',
    description: 'Structured market, account, and vendor reports.',
  },
  {
    id: 'blog_post',
    label: 'Blog Posts',
    description: 'SEO-ready long-form content drafts.',
  },
  {
    id: 'landing_page',
    label: 'Landing Pages',
    description: 'Campaign pages with hero, sections, CTA, and metadata.',
  },
  {
    id: 'sales_brief',
    label: 'Sales Briefs',
    description: 'Pre-call and account-facing sales enablement assets.',
  },
]

const STATUSES: StatusFilter[] = ['draft', 'approved', 'rejected', 'all']

export default function ContentOpsAssetsReview() {
  const [asset, setAsset] = useState<GeneratedAssetType>('report')
  const [status, setStatus] = useState<StatusFilter>('draft')
  const [limit, setLimit] = useState(20)
  const [data, setData] = useState<GeneratedAssetListResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [busyId, setBusyId] = useState<string | null>(null)
  const [batchBusy, setBatchBusy] = useState<'approved' | 'rejected' | null>(null)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set())
  const [exporting, setExporting] = useState(false)
  const [detailRow, setDetailRow] = useState<GeneratedAssetDraft | null>(null)

  const params = useMemo(
    () => ({
      status: status === 'all' ? '' : status,
      limit,
    }),
    [limit, status],
  )

  const load = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true)
    else setLoading(true)
    setError(null)
    setActionError(null)
    try {
      const result = await fetchGeneratedAssetDrafts(asset, params)
      setData(result)
      setSelectedIds(new Set())
      setDetailRow(null)
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)))
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [asset, params])

  useEffect(() => {
    void load(false)
  }, [load])

  const handleReview = async (row: GeneratedAssetDraft, nextStatus: 'approved' | 'rejected') => {
    const id = assetId(row)
    if (!id) return
    setBusyId(id)
    setActionError(null)
    try {
      await reviewGeneratedAssetDraft(asset, id, nextStatus)
      await load(true)
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusyId(null)
    }
  }

  const handleBatchReview = async (nextStatus: 'approved' | 'rejected') => {
    const ids = Array.from(selectedIds)
    if (ids.length === 0) return
    setBatchBusy(nextStatus)
    setActionError(null)
    try {
      const result = await reviewGeneratedAssetDrafts(asset, ids, nextStatus)
      await load(true)
      if (result.missing_ids.length > 0) {
        setActionError(
          `${result.updated} updated. ${result.missing_ids.length} item(s) not found and skipped.`,
        )
      }
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err))
    } finally {
      setBatchBusy(null)
    }
  }

  const handleExport = async () => {
    setExporting(true)
    setActionError(null)
    try {
      const csv = await exportGeneratedAssetDraftsCsv(asset, params)
      downloadCsv(csv, `${asset}_drafts.csv`)
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err))
    } finally {
      setExporting(false)
    }
  }

  if (error && !data) {
    return <PageError error={error} onRetry={() => void load(true)} />
  }

  const activeAsset = ASSETS.find((item) => item.id === asset) || ASSETS[0]
  const rows = data?.rows || []
  const reviewableIds = rows.map(assetId).filter(Boolean)
  const selectedCount = reviewableIds.filter((id) => selectedIds.has(id)).length
  const allSelected = reviewableIds.length > 0 && selectedCount === reviewableIds.length
  const toggleAll = () => {
    if (allSelected) {
      setSelectedIds(new Set())
      return
    }
    setSelectedIds(new Set(reviewableIds))
  }
  const toggleRow = (id: string) => {
    setSelectedIds((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-4 py-6 sm:px-6 lg:px-8">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-sm font-medium uppercase tracking-wide text-cyan-300">
            AI Content Ops
          </p>
          <h1 className="mt-2 text-3xl font-semibold text-white">
            Generated Asset Review
          </h1>
          <p className="mt-2 max-w-3xl text-sm text-slate-400">
            Review persisted drafts from the Content Ops generation pipeline,
            approve assets that are ready, reject misses, and export the current
            queue for offline review.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Link
            to="/content-ops/new"
            className="inline-flex items-center gap-2 rounded-md border border-slate-700 px-3 py-2 text-sm text-slate-200 hover:border-cyan-400 hover:text-cyan-200"
          >
            New run
            <ArrowRight className="h-4 w-4" />
          </Link>
          <button
            type="button"
            onClick={() => void load(true)}
            disabled={refreshing}
            className="inline-flex items-center gap-2 rounded-md border border-slate-700 px-3 py-2 text-sm text-slate-200 hover:border-cyan-400 hover:text-cyan-200 disabled:opacity-60"
          >
            <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
            Refresh
          </button>
        </div>
      </header>

      <section className="grid gap-3 md:grid-cols-4">
        {ASSETS.map((item) => (
          <button
            key={item.id}
            type="button"
            onClick={() => setAsset(item.id)}
            className={clsx(
              'rounded-lg border p-4 text-left transition',
              asset === item.id
                ? 'border-cyan-400 bg-cyan-500/10'
                : 'border-slate-800 bg-slate-900/60 hover:border-slate-600',
            )}
          >
            <div className="text-sm font-semibold text-white">{item.label}</div>
            <p className="mt-1 text-xs text-slate-400">{item.description}</p>
          </button>
        ))}
      </section>

      <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-white">{activeAsset.label}</h2>
            <p className="text-sm text-slate-400">{activeAsset.description}</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={toggleAll}
              disabled={reviewableIds.length === 0 || loading}
              className="inline-flex items-center gap-2 rounded-md border border-slate-700 px-3 py-2 text-sm text-slate-200 hover:border-cyan-400 hover:text-cyan-200 disabled:opacity-60"
            >
              {allSelected ? 'Clear' : 'Select all'}
            </button>
            <button
              type="button"
              onClick={() => void handleBatchReview('approved')}
              disabled={selectedCount === 0 || Boolean(batchBusy) || Boolean(busyId)}
              className="inline-flex items-center gap-2 rounded-md border border-emerald-500/40 px-3 py-2 text-sm text-emerald-200 hover:bg-emerald-500/10 disabled:opacity-50"
            >
              {batchBusy === 'approved' ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <CheckCircle2 className="h-4 w-4" />
              )}
              Approve selected
            </button>
            <button
              type="button"
              onClick={() => void handleBatchReview('rejected')}
              disabled={selectedCount === 0 || Boolean(batchBusy) || Boolean(busyId)}
              className="inline-flex items-center gap-2 rounded-md border border-rose-500/40 px-3 py-2 text-sm text-rose-200 hover:bg-rose-500/10 disabled:opacity-50"
            >
              {batchBusy === 'rejected' ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <XCircle className="h-4 w-4" />
              )}
              Reject selected
            </button>
            <select
              value={status}
              onChange={(event) => setStatus(event.target.value as StatusFilter)}
              className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
            >
              {STATUSES.map((item) => (
                <option key={item} value={item}>
                  {item === 'all' ? 'All statuses' : item}
                </option>
              ))}
            </select>
            <select
              value={limit}
              onChange={(event) => setLimit(Number(event.target.value))}
              className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
            >
              {[10, 20, 50, 100].map((item) => (
                <option key={item} value={item}>
                  {item} rows
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => void handleExport()}
              disabled={exporting || loading}
              className="inline-flex items-center gap-2 rounded-md bg-cyan-500 px-3 py-2 text-sm font-medium text-slate-950 hover:bg-cyan-400 disabled:opacity-60"
            >
              {exporting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Download className="h-4 w-4" />
              )}
              Export CSV
            </button>
          </div>
        </div>

        {actionError && (
          <div className="mt-4 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
            {actionError}
          </div>
        )}
        {error && data && (
          <div className="mt-4 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-100">
            Refresh failed: {error.message}
          </div>
        )}

        <div className="mt-4">
          {loading ? (
            <div className="flex items-center justify-center py-20 text-slate-400">
              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
              Loading generated assets...
            </div>
          ) : rows.length === 0 ? (
            <div className="rounded-md border border-slate-800 bg-slate-950/50 px-4 py-10 text-center text-sm text-slate-400">
              No {status === 'all' ? '' : status} {activeAsset.label.toLowerCase()} found.
            </div>
          ) : (
            <div className="space-y-3">
              {rows.map((row, index) => (
                <AssetRow
                  key={assetId(row) || `${asset}-${index}`}
                  row={row}
                  asset={asset}
                  busy={busyId === assetId(row)}
                  selected={selectedIds.has(assetId(row))}
                  onToggle={toggleRow}
                  onReview={handleReview}
                  onOpenDetails={setDetailRow}
                />
              ))}
            </div>
          )}
        </div>
      </section>
      {detailRow && (
        <AssetDetailDrawer
          row={detailRow}
          asset={asset}
          onClose={() => setDetailRow(null)}
        />
      )}
    </div>
  )
}

function AssetRow({
  row,
  asset,
  busy,
  selected,
  onToggle,
  onReview,
  onOpenDetails,
}: {
  row: GeneratedAssetDraft
  asset: GeneratedAssetType
  busy: boolean
  selected: boolean
  onToggle: (id: string) => void
  onReview: (row: GeneratedAssetDraft, status: 'approved' | 'rejected') => void
  onOpenDetails: (row: GeneratedAssetDraft) => void
}) {
  const id = assetId(row)
  const title = assetTitle(row, asset)
  const subtitle = assetSubtitle(row, asset)
  const status = textValue(row.status) || 'unknown'
  const canReview = Boolean(id)
  const preview = assetPreview(row, asset)
  const facts = assetFacts(row, asset)

  return (
    <article className="rounded-lg border border-slate-800 bg-slate-950/50 p-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex min-w-0 flex-1 items-start gap-3">
          <input
            type="checkbox"
            checked={selected}
            onChange={() => id && onToggle(id)}
            disabled={!canReview || busy}
            aria-label={`Select ${title}`}
            className="mt-1 h-4 w-4 rounded border-slate-700 bg-slate-950 text-cyan-400 focus:ring-cyan-400 disabled:opacity-50"
          />
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="truncate text-base font-semibold text-white">{title}</h3>
              <span className="rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-300">
                {status}
              </span>
              {row.reasoning_context_used && (
                <span className="rounded bg-violet-500/10 px-2 py-0.5 text-xs text-violet-200">
                  reasoning
                </span>
              )}
            </div>
            {subtitle && <p className="mt-1 text-sm text-slate-400">{subtitle}</p>}
            <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500">
              {id && <span className="font-mono">id: {id}</span>}
              {row.reasoning_wedge && <span>wedge: {textValue(row.reasoning_wedge)}</span>}
              {typeof row.generation_total_tokens === 'number' && (
                <span>tokens: {row.generation_total_tokens}</span>
              )}
              {typeof row.generation_parse_attempts === 'number' && (
                <span>parse attempts: {row.generation_parse_attempts}</span>
              )}
            </div>
            {facts.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-2 text-xs">
                {facts.map((fact) => (
                  <span
                    key={`${fact.label}:${fact.value}`}
                    className="rounded border border-slate-800 bg-slate-900/70 px-2 py-1 text-slate-300"
                  >
                    <span className="text-slate-500">{fact.label}: </span>
                    {fact.value}
                  </span>
                ))}
              </div>
            )}
            {preview && (
              <div className="mt-4 rounded-md border border-slate-800 bg-slate-900/70 p-3">
                <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Preview
                </div>
                <div className="mt-2 space-y-2">
                  {preview.heading && (
                    <div className="text-sm font-medium text-slate-200">
                      {preview.heading}
                    </div>
                  )}
                  {preview.body && (
                    <p className="line-clamp-3 text-sm leading-6 text-slate-400">
                      {preview.body}
                    </p>
                  )}
                  {preview.meta.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {preview.meta.map((item, index) => (
                        <span
                          key={`${item}-${index}`}
                          className="rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-300"
                        >
                          {item}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => onOpenDetails(row)}
            className="inline-flex items-center gap-2 rounded-md border border-slate-700 px-3 py-2 text-sm text-slate-200 hover:border-cyan-400 hover:text-cyan-200"
          >
            <Eye className="h-4 w-4" />
            Details
          </button>
          <button
            type="button"
            onClick={() => onReview(row, 'approved')}
            disabled={!canReview || busy || status === 'approved'}
            title={canReview ? 'Approve draft' : 'Draft id missing'}
            className="inline-flex items-center gap-2 rounded-md border border-emerald-500/40 px-3 py-2 text-sm text-emerald-200 hover:bg-emerald-500/10 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />}
            Approve
          </button>
          <button
            type="button"
            onClick={() => onReview(row, 'rejected')}
            disabled={!canReview || busy || status === 'rejected'}
            title={canReview ? 'Reject draft' : 'Draft id missing'}
            className="inline-flex items-center gap-2 rounded-md border border-rose-500/40 px-3 py-2 text-sm text-rose-200 hover:bg-rose-500/10 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <XCircle className="h-4 w-4" />}
            Reject
          </button>
        </div>
      </div>
    </article>
  )
}

function AssetDetailDrawer({
  row,
  asset,
  onClose,
}: {
  row: GeneratedAssetDraft
  asset: GeneratedAssetType
  onClose: () => void
}) {
  const title = assetTitle(row, asset)
  const status = textValue(row.status) || 'unknown'
  const preview = assetPreview(row, asset)
  const facts = assetFacts(row, asset)
  const sections = sectionList(row.sections)
  const references = valueList(row.reference_ids)
  const drawerRef = useRef<HTMLElement | null>(null)
  const closeButtonRef = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {
    const previous = document.activeElement
    closeButtonRef.current?.focus()
    return () => {
      if (previous instanceof HTMLElement) {
        previous.focus()
      }
    }
  }, [])

  const handleKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault()
      onClose()
      return
    }
    if (event.key !== 'Tab') return
    const focusable = focusableElements(drawerRef.current)
    if (focusable.length === 0) return
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault()
      last.focus()
      return
    }
    if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end bg-slate-950/70"
      onClick={onClose}
    >
      <aside
        ref={drawerRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="asset-detail-title"
        tabIndex={-1}
        onClick={(event) => event.stopPropagation()}
        onKeyDown={handleKeyDown}
        className="h-full w-full max-w-2xl overflow-y-auto border-l border-slate-800 bg-slate-950 p-6 shadow-2xl"
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-xs font-semibold uppercase tracking-wide text-cyan-300">
              {assetLabel(asset)}
            </div>
            <h2 id="asset-detail-title" className="mt-2 text-2xl font-semibold text-white">
              {title}
            </h2>
            <div className="mt-2 flex flex-wrap gap-2 text-xs text-slate-400">
              <span className="rounded bg-slate-800 px-2 py-0.5">{status}</span>
              {assetId(row) && <span className="font-mono">id: {assetId(row)}</span>}
            </div>
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            onClick={onClose}
            aria-label="Close details"
            className="rounded-md border border-slate-700 p-2 text-slate-300 hover:border-cyan-400 hover:text-cyan-200"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {facts.length > 0 && (
          <section className="mt-6">
            <h3 className="text-sm font-semibold text-slate-200">Facts</h3>
            <div className="mt-3 flex flex-wrap gap-2 text-xs">
              {facts.map((fact) => (
                <span
                  key={`${fact.label}:${fact.value}`}
                  className="rounded border border-slate-800 bg-slate-900 px-2 py-1 text-slate-300"
                >
                  <span className="text-slate-500">{fact.label}: </span>
                  {fact.value}
                </span>
              ))}
            </div>
          </section>
        )}

        {preview && (
          <section className="mt-6">
            <h3 className="text-sm font-semibold text-slate-200">Preview</h3>
            <div className="mt-3 rounded-md border border-slate-800 bg-slate-900/70 p-4">
              {preview.heading && (
                <div className="text-base font-medium text-white">{preview.heading}</div>
              )}
              {preview.body && (
                <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-300">
                  {preview.body}
                </p>
              )}
              {preview.meta.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {preview.meta.map((item, index) => (
                    <span
                      key={`${item}-${index}`}
                      className="rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-300"
                    >
                      {item}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </section>
        )}

        {sections.length > 0 && (
          <section className="mt-6">
            <h3 className="text-sm font-semibold text-slate-200">Sections</h3>
            <div className="mt-3 space-y-3">
              {sections.map((section, index) => (
                <div
                  key={`${section.title}-${index}`}
                  className="rounded-md border border-slate-800 bg-slate-900/60 p-4"
                >
                  <div className="text-sm font-medium text-white">
                    {section.title || `Section ${index + 1}`}
                  </div>
                  {section.body && (
                    <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-400">
                      {section.body}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </section>
        )}

        {references.length > 0 && (
          <section className="mt-6">
            <h3 className="text-sm font-semibold text-slate-200">References</h3>
            <div className="mt-3 flex flex-wrap gap-2">
              {references.map((reference) => (
                <span
                  key={reference}
                  className="rounded bg-slate-800 px-2 py-0.5 font-mono text-xs text-slate-300"
                >
                  {reference}
                </span>
              ))}
            </div>
          </section>
        )}

        <section className="mt-6">
          <h3 className="text-sm font-semibold text-slate-200">Raw Row</h3>
          <p className="mt-1 text-xs text-slate-500">
            Diagnostic dump for operators with access to this review queue.
          </p>
          <pre className="mt-3 max-h-96 overflow-auto rounded-md border border-slate-800 bg-slate-900/70 p-4 text-xs leading-5 text-slate-300">
            {JSON.stringify(row, null, 2)}
          </pre>
        </section>
      </aside>
    </div>
  )
}

function focusableElements(root: HTMLElement | null): HTMLElement[] {
  if (!root) return []
  return Array.from(
    root.querySelectorAll<HTMLElement>(
      'a[href],button:not([disabled]),textarea,input,select,[tabindex]:not([tabindex="-1"])',
    ),
  ).filter((item) => !item.hasAttribute('disabled') && item.offsetParent !== null)
}

function assetId(row: GeneratedAssetDraft): string {
  return textValue(row.id)
}

function assetTitle(row: GeneratedAssetDraft, asset: GeneratedAssetType): string {
  return (
    textValue(row.title) ||
    textValue(row.headline) ||
    textValue(row.slug) ||
    textValue(row.target_id) ||
    textValue(row.campaign_name) ||
    `${asset} draft`
  )
}

function assetLabel(asset: GeneratedAssetType): string {
  return ASSETS.find((item) => item.id === asset)?.label || asset
}

function assetSubtitle(row: GeneratedAssetDraft, asset: GeneratedAssetType): string {
  if (asset === 'blog_post') {
    return [row.topic_type, row.slug, row.description]
      .map(textValue)
      .filter(Boolean)
      .join(' | ')
  }
  if (asset === 'landing_page') {
    return [row.campaign_name, row.slug, row.value_prop]
      .map(textValue)
      .filter(Boolean)
      .join(' | ')
  }
  if (asset === 'sales_brief') {
    return [row.target_mode, row.brief_type, row.target_id]
      .map(textValue)
      .filter(Boolean)
      .join(' | ')
  }
  return [row.target_mode, row.report_type, row.summary]
    .map(textValue)
    .filter(Boolean)
    .join(' | ')
}

function assetFacts(row: GeneratedAssetDraft, asset: GeneratedAssetType): AssetFact[] {
  const facts: AssetFact[] = []
  addFact(facts, 'target', row.target_id)
  addFact(facts, 'mode', row.target_mode)
  addAssetSpecificFacts(facts, row, asset)
  addFact(facts, 'sections', row.section_count)
  addFact(facts, 'references', row.reference_count)
  addFact(facts, 'tags', row.tag_count)
  addFact(facts, 'charts', row.chart_count)
  addFact(facts, 'confidence', confidenceLabel(row.reasoning_confidence))
  addFact(facts, 'input tokens', row.generation_input_tokens)
  addFact(facts, 'output tokens', row.generation_output_tokens)
  return facts
}

function addAssetSpecificFacts(
  facts: AssetFact[],
  row: GeneratedAssetDraft,
  asset: GeneratedAssetType,
): void {
  if (asset === 'report') {
    addFact(facts, 'report', row.report_type)
    return
  }
  if (asset === 'blog_post') {
    addFact(facts, 'topic', row.topic_type)
    return
  }
  if (asset === 'landing_page') {
    addFact(facts, 'campaign', row.campaign_name)
    addFact(facts, 'persona', row.persona)
    return
  }
  addFact(facts, 'brief', row.brief_type)
}

function addFact(facts: AssetFact[], label: string, value: unknown): void {
  const text = textValue(value) || numberText(value)
  if (!text) return
  if (facts.some((fact) => fact.label === label && fact.value === text)) return
  facts.push({ label, value: text })
}

function textValue(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function numberText(value: unknown): string {
  return typeof value === 'number' && Number.isFinite(value) ? String(value) : ''
}

function confidenceLabel(value: unknown): string {
  if (typeof value === 'string') return value.trim()
  if (typeof value !== 'number' || !Number.isFinite(value)) return ''
  const pct = value <= 1 ? value * 100 : value
  if (pct < 0 || pct > 100) return ''
  return `${Math.round(pct)}%`
}

function assetPreview(row: GeneratedAssetDraft, asset: GeneratedAssetType): AssetPreview | null {
  if (asset === 'blog_post') {
    const tags = valueList(row.tags).slice(0, 4)
    return previewOrNull({
      heading: textValue(row.title) || textValue(row.description),
      body: excerpt(textValue(row.content)),
      meta: tags,
    })
  }
  if (asset === 'landing_page') {
    const hero = recordValue(row.hero)
    const cta = recordValue(row.cta)
    const section = firstSection(row.sections)
    return previewOrNull({
      heading: objectText(hero, 'headline') || section.title || textValue(row.title),
      body: excerpt(section.body),
      meta: [
        objectText(hero, 'subheadline'),
        objectText(cta, 'label'),
        objectText(cta, 'url'),
      ].filter(Boolean),
    })
  }
  const section = firstSection(row.sections)
  const sectionTitles = sectionList(row.sections)
    .map((item) => item.title)
    .filter(Boolean)
    .slice(0, 3)
  return previewOrNull({
    heading: section.title || textValue(row.headline) || textValue(row.title),
    body: excerpt(section.body || textValue(row.summary) || textValue(row.headline)),
    meta: [
      ...sectionTitles.map((title) => `section: ${title}`),
      ...valueList(row.reference_ids).slice(0, 3).map((id) => `ref: ${id}`),
    ],
  })
}

function previewOrNull(preview: {
  heading: string
  body: string
  meta: string[]
}): { heading: string; body: string; meta: string[] } | null {
  return preview.heading || preview.body || preview.meta.length > 0 ? preview : null
}

function firstSection(value: unknown): { title: string; body: string } {
  return sectionList(value)[0] || { title: '', body: '' }
}

function sectionList(value: unknown): Array<{ title: string; body: string }> {
  let sections = Array.isArray(value) ? value : []
  if (typeof value === 'string' && value.trim()) {
    try {
      const parsed = JSON.parse(value)
      sections = Array.isArray(parsed) ? parsed : []
    } catch {
      sections = []
    }
  }
  return sections
    .filter((item): item is Record<string, unknown> =>
      Boolean(item && typeof item === 'object' && !Array.isArray(item)),
    )
    .map((section) => ({
      title: textValue(section.title) || textValue(section.heading),
      body: textValue(section.body_markdown) || textValue(section.body),
    }))
}

function recordValue(value: unknown): Record<string, unknown> | null {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }
  if (typeof value === 'string' && value.trim()) {
    try {
      const parsed = JSON.parse(value)
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>
      }
    } catch {
      return null
    }
  }
  return null
}

function objectText(value: Record<string, unknown> | null, key: string): string {
  return value ? textValue(value[key]) : ''
}

function valueList(value: unknown): string[] {
  if (Array.isArray(value)) return value.map(textValue).filter(Boolean)
  const text = textValue(value)
  if (!text) return []
  return text.split(',').map((item) => item.trim()).filter(Boolean)
}

function excerpt(value: string, limit = 260): string {
  const clean = value.replace(/\s+/g, ' ').trim()
  if (clean.length <= limit) return clean
  return `${clean.slice(0, limit - 3).trim()}...`
}

function downloadCsv(csv: string, filename: string): void {
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 0)
}
