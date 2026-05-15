import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  ArrowRight,
  CheckCircle2,
  Download,
  Loader2,
  RefreshCw,
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
                />
              ))}
            </div>
          )}
        </div>
      </section>
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
}: {
  row: GeneratedAssetDraft
  asset: GeneratedAssetType
  busy: boolean
  selected: boolean
  onToggle: (id: string) => void
  onReview: (row: GeneratedAssetDraft, status: 'approved' | 'rejected') => void
}) {
  const id = assetId(row)
  const title = assetTitle(row, asset)
  const subtitle = assetSubtitle(row, asset)
  const status = textValue(row.status) || 'unknown'
  const canReview = Boolean(id)
  const preview = assetPreview(row, asset)

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
        <div className="flex gap-2">
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

function assetSubtitle(row: GeneratedAssetDraft, asset: GeneratedAssetType): string {
  if (asset === 'blog_post') {
    return [row.topic_type, row.description].map(textValue).filter(Boolean).join(' | ')
  }
  if (asset === 'landing_page') {
    return [row.campaign_name, row.slug].map(textValue).filter(Boolean).join(' | ')
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

function textValue(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function assetPreview(row: GeneratedAssetDraft, asset: GeneratedAssetType): {
  heading: string
  body: string
  meta: string[]
} | null {
  if (asset === 'blog_post') {
    const tags = valueList(row.tags).slice(0, 4)
    return previewOrNull({
      heading: textValue(row.description),
      body: excerpt(textValue(row.content)),
      meta: tags,
    })
  }
  if (asset === 'landing_page') {
    const section = firstSection(row.sections)
    return previewOrNull({
      heading: objectText(row.hero, 'headline') || section.title,
      body: excerpt(section.body),
      meta: [objectText(row.cta, 'label')].filter(Boolean),
    })
  }
  // Reports and sales briefs share the same section/summary preview shape.
  const section = firstSection(row.sections)
  return previewOrNull({
    heading: section.title,
    body: excerpt(section.body || textValue(row.summary) || textValue(row.headline)),
    meta: valueList(row.reference_ids).slice(0, 3).map((id) => `ref: ${id}`),
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
  let sections = Array.isArray(value) ? value : []
  if (typeof value === 'string' && value.trim()) {
    try {
      const parsed = JSON.parse(value)
      sections = Array.isArray(parsed) ? parsed : []
    } catch {
      sections = []
    }
  }
  const first = sections.find((item) => item && typeof item === 'object')
  if (!first || typeof first !== 'object') return { title: '', body: '' }
  const section = first as Record<string, unknown>
  return {
    title: textValue(section.title),
    body: textValue(section.body_markdown) || textValue(section.body),
  }
}

function objectText(value: unknown, key: string): string {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return ''
  return textValue((value as Record<string, unknown>)[key])
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
