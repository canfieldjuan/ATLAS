import {
  type KeyboardEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import {
  ArrowRight,
  CheckCircle2,
  Code2,
  Copy,
  Download,
  Eye,
  ExternalLink,
  Loader2,
  Pencil,
  RefreshCw,
  Save,
  UploadCloud,
  Wrench,
  X,
  XCircle,
} from 'lucide-react'
import { clsx } from 'clsx'
import {
  exportGeneratedAssetDraftsCsv,
  exportGeneratedAssetDraftsHtml,
  exportGeneratedAssetDraftsPng,
  fetchGeneratedAssetDrafts,
  fetchGeneratedFaqMacroPublishAttempts,
  publishGeneratedFaqMacros,
  repairGeneratedLandingPageDraft,
  reviewGeneratedAssetDraft,
  reviewGeneratedAssetDrafts,
  updateGeneratedLandingPageDraft,
  type GeneratedAssetDraft,
  type GeneratedLandingPageDraftUpdate,
  type GeneratedAssetMacroPublishSummary,
  type GeneratedAssetMacroPublishAttempt,
  type GeneratedAssetMacroPublishAttemptsResponse,
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

type ReadinessSummary = {
  status: string
  passed: number | null
  total: number | null
  missing: string[]
  checks: ReadinessCheck[]
}

type ReadinessCheck = {
  label: string
  passed: boolean
}

type ReadinessPanel = ReadinessSummary & {
  label: string
}

type RepairHistoryEntry = {
  attempt: number
  passed: boolean
  blockers: string[]
  repairIssues: string[]
}

type FAQItemPreview = {
  question: string
  answer: string
  actionItems: string[]
  sourceLabels: string[]
  termMappings: FAQTermMappingPreview[]
}

type FAQTermMappingPreview = {
  customerTerm: string
  documentationTerm: string
  sourceIdCount: string
  zeroResultSourceCount: string
  opportunityScore: string
}

type StructuredDataSummary = {
  raw: Record<string, unknown>
  nodeTypes: string[]
  questionCount: number
  hasCanonical: boolean
}

type LandingPageSectionEdit = {
  id: string
  title: string
  body_markdown: string
  metadata: Record<string, unknown>
}

type LandingPageEditState = {
  title: string
  slug: string
  heroHeadline: string
  heroSubheadline: string
  heroCtaLabel: string
  heroCtaUrl: string
  ctaLabel: string
  ctaUrl: string
  metaTitleTag: string
  metaDescription: string
  referenceIdsText: string
  sections: LandingPageSectionEdit[]
}

type MacroPublishOutcome =
  | { kind: 'idle' }
  | { kind: 'success'; draftId: string; summary: GeneratedAssetMacroPublishSummary }
  | { kind: 'error'; draftId: string; message: string }

type MacroPublishHistoryState =
  | { kind: 'idle' }
  | { kind: 'loading'; draftId: string }
  | { kind: 'loaded'; draftId: string; response: GeneratedAssetMacroPublishAttemptsResponse }
  | { kind: 'error'; draftId: string; message: string }

type MacroPublishCounts = Pick<
  GeneratedAssetMacroPublishSummary,
  | 'published_count'
  | 'updated_count'
  | 'skipped_count'
  | 'failed_count'
  | 'pending_reconcile_count'
>

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
  {
    id: 'social_post',
    label: 'Social Posts',
    description: 'Short-form posts generated from review and source evidence.',
  },
  {
    id: 'ad_copy',
    label: 'Ad Copy',
    description: 'Paid-social ad drafts generated from review and source evidence.',
  },
  {
    id: 'quote_card',
    label: 'Quote Cards',
    description: 'Customer-proof quote card drafts generated from source evidence.',
  },
  {
    id: 'stat_card',
    label: 'Stat Cards',
    description: 'Evidence-backed metric card drafts generated from source evidence.',
  },
  {
    id: 'faq_markdown',
    label: 'FAQ Markdown',
    description: 'Grounded FAQ documents generated from support-ticket evidence.',
  },
]

const STATUSES: StatusFilter[] = ['draft', 'approved', 'rejected', 'all']
const inputClassName =
  'w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-600 focus:border-cyan-400 focus:outline-none'
const textAreaClassName =
  'w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm leading-6 text-slate-100 placeholder:text-slate-600 focus:border-cyan-400 focus:outline-none'
const ID_FILTERED_ASSETS = new Set<GeneratedAssetType>([
  'blog_post',
  'landing_page',
])

function assetFromSearchParams(params: URLSearchParams): GeneratedAssetType {
  const asset = params.get('asset')
  return ASSETS.some((item) => item.id === asset)
    ? asset as GeneratedAssetType
    : 'report'
}

function statusFromSearchParams(params: URLSearchParams): StatusFilter {
  const status = params.get('status')
  return STATUSES.includes(status as StatusFilter)
    ? status as StatusFilter
    : 'draft'
}

function idFiltersFromSearchParams(params: URLSearchParams): string[] {
  const seen = new Set<string>()
  const ids: string[] = []
  for (const raw of params.getAll('id')) {
    const id = raw.trim()
    if (!id || seen.has(id)) continue
    seen.add(id)
    ids.push(id)
  }
  return ids
}

function assetSupportsIdFilter(asset: GeneratedAssetType): boolean {
  return ID_FILTERED_ASSETS.has(asset)
}

function assetSupportsVisualExport(asset: GeneratedAssetType): boolean {
  return asset === 'quote_card' || asset === 'stat_card'
}

export default function ContentOpsAssetsReview() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [asset, setAsset] = useState<GeneratedAssetType>(() =>
    assetFromSearchParams(searchParams),
  )
  const [status, setStatus] = useState<StatusFilter>(() =>
    statusFromSearchParams(searchParams),
  )
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
  const [macroPublishOutcome, setMacroPublishOutcome] =
    useState<MacroPublishOutcome>({ kind: 'idle' })
  const [macroPublishHistory, setMacroPublishHistory] =
    useState<MacroPublishHistoryState>({ kind: 'idle' })

  const focusedIds = useMemo(
    () => idFiltersFromSearchParams(searchParams),
    [searchParams],
  )
  const params = useMemo(
    () => ({
      status: status === 'all' ? '' : status,
      id: assetSupportsIdFilter(asset) && focusedIds.length > 0
        ? focusedIds
        : undefined,
      limit,
    }),
    [asset, focusedIds, limit, status],
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
      setMacroPublishHistory({ kind: 'idle' })
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

  useEffect(() => {
    const id = detailRow && asset === 'faq_markdown' ? assetId(detailRow) : ''
    if (!id) {
      setMacroPublishHistory({ kind: 'idle' })
      return
    }

    let active = true
    setMacroPublishHistory({ kind: 'loading', draftId: id })
    fetchGeneratedFaqMacroPublishAttempts(id, { limit: 5 })
      .then((response) => {
        if (active) {
          setMacroPublishHistory({ kind: 'loaded', draftId: id, response })
        }
      })
      .catch((err) => {
        if (active) {
          setMacroPublishHistory({
            kind: 'error',
            draftId: id,
            message: err instanceof Error ? err.message : String(err),
          })
        }
      })

    return () => {
      active = false
    }
  }, [asset, detailRow])

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
      downloadText(csv, `${asset}_drafts.csv`, 'text/csv;charset=utf-8')
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err))
    } finally {
      setExporting(false)
    }
  }

  const handleVisualExport = async () => {
    if (!assetSupportsVisualExport(asset)) return
    setExporting(true)
    setActionError(null)
    try {
      const html = await exportGeneratedAssetDraftsHtml(asset, params)
      downloadText(html, `${asset}_visual_cards.html`, 'text/html;charset=utf-8')
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err))
    } finally {
      setExporting(false)
    }
  }

  const handlePngExport = async () => {
    if (!assetSupportsVisualExport(asset)) return
    setExporting(true)
    setActionError(null)
    try {
      const png = await exportGeneratedAssetDraftsPng(asset, params)
      downloadBlob(png, `${asset}_visual_cards.png`)
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err))
    } finally {
      setExporting(false)
    }
  }

  const handleAssetChange = (nextAsset: GeneratedAssetType) => {
    setAsset(nextAsset)
    setSearchParams(
      (current) => {
        const next = new URLSearchParams(current)
        next.set('asset', nextAsset)
        next.delete('id')
        return next
      },
      { replace: true },
    )
  }

  const clearFocusedIds = () => {
    setSearchParams(
      (current) => {
        const next = new URLSearchParams(current)
        next.delete('id')
        return next
      },
      { replace: true },
    )
  }

  const handleSaveLandingPageDraft = async (
    row: GeneratedAssetDraft,
    update: GeneratedLandingPageDraftUpdate,
  ) => {
    const id = assetId(row)
    if (!id) throw new Error('Draft id missing')
    setBusyId(id)
    setActionError(null)
    try {
      const updated = await updateGeneratedLandingPageDraft(id, update)
      setData((current) => {
        if (!current) return current
        return {
          ...current,
          rows: current.rows.map((item) =>
            assetId(item) === id ? updated : item,
          ),
        }
      })
      setDetailRow(updated)
      return updated
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      setActionError(message)
      throw err
    } finally {
      setBusyId(null)
    }
  }

  const handleRepairLandingPageDraft = async (row: GeneratedAssetDraft) => {
    const id = assetId(row)
    if (!id) throw new Error('Draft id missing')
    setBusyId(id)
    setActionError(null)
    try {
      const updated = await repairGeneratedLandingPageDraft(id)
      setData((current) => {
        if (!current) return current
        return {
          ...current,
          rows: current.rows.map((item) =>
            assetId(item) === id ? updated : item,
          ),
        }
      })
      setDetailRow(updated)
      return updated
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      setActionError(message)
      throw err
    } finally {
      setBusyId(null)
    }
  }

  const handlePublishFaqMacros = async (row: GeneratedAssetDraft) => {
    const id = assetId(row)
    if (!id) return
    setBusyId(id)
    setActionError(null)
    setMacroPublishOutcome({ kind: 'idle' })
    try {
      const summary = await publishGeneratedFaqMacros(id)
      setMacroPublishOutcome({ kind: 'success', draftId: id, summary })
      await load(true)
    } catch (err) {
      setMacroPublishOutcome({
        kind: 'error',
        draftId: id,
        message: err instanceof Error ? err.message : String(err),
      })
    } finally {
      setBusyId(null)
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

      <section className="grid gap-3 md:grid-cols-3 xl:grid-cols-6">
        {ASSETS.map((item) => (
          <button
            key={item.id}
            type="button"
            onClick={() => handleAssetChange(item.id)}
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
            {assetSupportsVisualExport(asset) && (
              <>
                <button
                  type="button"
                  onClick={() => void handlePngExport()}
                  disabled={exporting || loading}
                  className="inline-flex items-center gap-2 rounded-md bg-emerald-500 px-3 py-2 text-sm font-medium text-slate-950 hover:bg-emerald-400 disabled:opacity-60"
                >
                  {exporting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Download className="h-4 w-4" />
                  )}
                  Export PNG
                </button>
                <button
                  type="button"
                  onClick={() => void handleVisualExport()}
                  disabled={exporting || loading}
                  className="inline-flex items-center gap-2 rounded-md border border-emerald-400/50 px-3 py-2 text-sm font-medium text-emerald-100 hover:bg-emerald-400/10 disabled:opacity-60"
                >
                  {exporting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                  Export HTML
                </button>
              </>
            )}
          </div>
        </div>

        {actionError && (
          <div className="mt-4 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
            {actionError}
          </div>
        )}
        <MacroPublishResultBanner outcome={macroPublishOutcome} />
        {assetSupportsIdFilter(asset) && focusedIds.length > 0 && (
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-md border border-cyan-500/30 bg-cyan-500/10 px-3 py-2 text-sm text-cyan-100">
            <span>
              Showing {focusedIds.length} generated draft
              {focusedIds.length === 1 ? '' : 's'} from the previous run.
            </span>
            <button
              type="button"
              onClick={clearFocusedIds}
              className="rounded-md border border-cyan-500/40 px-2.5 py-1 text-xs font-medium text-cyan-100 hover:bg-cyan-500/10"
            >
              Show latest {activeAsset.label.toLowerCase()}
            </button>
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
                  onPublishFaqMacros={handlePublishFaqMacros}
                  onOpenDetails={setDetailRow}
                />
              ))}
            </div>
          )}
        </div>
      </section>
      {detailRow && (
        <AssetDetailDrawer
          key={assetId(detailRow) || assetTitle(detailRow, asset)}
          row={detailRow}
          asset={asset}
          saving={busyId === assetId(detailRow)}
          publishHistory={macroPublishHistory}
          onSaveLandingPage={handleSaveLandingPageDraft}
          onRepairLandingPage={handleRepairLandingPageDraft}
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
  onPublishFaqMacros,
  onOpenDetails,
}: {
  row: GeneratedAssetDraft
  asset: GeneratedAssetType
  busy: boolean
  selected: boolean
  onToggle: (id: string) => void
  onReview: (row: GeneratedAssetDraft, status: 'approved' | 'rejected') => void
  onPublishFaqMacros: (row: GeneratedAssetDraft) => void
  onOpenDetails: (row: GeneratedAssetDraft) => void
}) {
  const id = assetId(row)
  const title = assetTitle(row, asset)
  const subtitle = assetSubtitle(row, asset)
  const status = textValue(row.status) || 'unknown'
  const canReview = Boolean(id)
  const preview = assetPreview(row, asset)
  const facts = assetFacts(row, asset)
  const repairHistory = assetRepairHistory(row)
  const repairSummary = repairHistorySummary(repairHistory)
  const canPublishFaqMacros = asset === 'faq_markdown' && canReview && status === 'approved'

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
              {repairSummary && (
                <span className={clsx(
                  'rounded px-2 py-0.5 text-xs',
                  repairSummary.passed
                    ? 'bg-emerald-500/10 text-emerald-200'
                    : 'bg-amber-500/10 text-amber-200',
                )}>
                  {repairSummary.label}
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
          {asset === 'faq_markdown' && (
            <button
              type="button"
              onClick={() => onPublishFaqMacros(row)}
              disabled={!canPublishFaqMacros || busy}
              title={
                canPublishFaqMacros
                  ? 'Publish approved FAQ macros'
                  : 'Approve this FAQ draft before publishing macros'
              }
              className="inline-flex items-center gap-2 rounded-md border border-cyan-500/40 px-3 py-2 text-sm text-cyan-200 hover:bg-cyan-500/10 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {busy ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <UploadCloud className="h-4 w-4" />
              )}
              Publish macros
            </button>
          )}
        </div>
      </div>
    </article>
  )
}

function MacroPublishResultBanner({ outcome }: { outcome: MacroPublishOutcome }) {
  if (outcome.kind === 'idle') return null
  if (outcome.kind === 'error') {
    return (
      <div className="mt-4 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
        Macro publish failed for {outcome.draftId}: {outcome.message}
      </div>
    )
  }

  const { summary } = outcome
  const hasPending = summary.pending_reconcile_count > 0
  const hasFailures = summary.failed_count > 0
  const hasSkipped = summary.skipped_count > 0
  const tone = summary.ok
    ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-100'
    : hasPending || hasSkipped
      ? 'border-amber-500/30 bg-amber-500/10 text-amber-100'
      : 'border-rose-500/30 bg-rose-500/10 text-rose-100'
  const title = summary.ok
    ? 'FAQ macros published.'
    : hasPending
      ? 'FAQ macro publish needs reconciliation.'
      : hasFailures
        ? 'FAQ macro publish has failures.'
        : 'FAQ macro publish needs review.'
  const counts = macroPublishCountLabels(summary)

  return (
    <div className={clsx('mt-4 rounded-md border px-3 py-2 text-sm', tone)}>
      <div className="font-medium">{title}</div>
      <div className="mt-1 text-xs opacity-90">
        Draft {outcome.draftId}
        {summary.draft_status_updated ? ' moved to published.' : ' kept its current review status.'}
      </div>
      {counts.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2 text-xs">
          {counts.map((count) => (
            <span
              key={count}
              className="rounded border border-current/20 bg-slate-950/30 px-2 py-0.5"
            >
              {count}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function macroPublishCountLabels(summary: MacroPublishCounts): string[] {
  return [
    { count: summary.published_count, label: 'published' },
    { count: summary.updated_count, label: 'updated' },
    { count: summary.skipped_count, label: 'skipped' },
    { count: summary.failed_count, label: 'failed' },
    { count: summary.pending_reconcile_count, label: 'pending reconcile' },
  ]
    .filter(({ count }) => count > 0)
    .map(({ count, label }) => `${count} ${label}`)
}


function MacroPublishHistoryPanel({ history }: { history: MacroPublishHistoryState }) {
  return (
    <section className="mt-6">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-slate-200">Macro publish history</h3>
        {history.kind === "loaded" && (
          <span className="text-xs text-slate-500">
            {history.response.count} of {history.response.limit} recent
          </span>
        )}
      </div>
      <div className="mt-3 rounded-md border border-slate-800 bg-slate-900/70 p-4">
        {history.kind === "idle" && (
          <p className="text-sm text-slate-400">Open an FAQ draft to load publish history.</p>
        )}
        {history.kind === "loading" && (
          <div className="flex items-center text-sm text-slate-400">
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Loading publish history...
          </div>
        )}
        {history.kind === "error" && (
          <p className="text-sm text-rose-200">
            Could not load publish history for {history.draftId}: {history.message}
          </p>
        )}
        {history.kind === "loaded" && history.response.attempts.length === 0 && (
          <p className="text-sm text-slate-400">No macro publish attempts recorded yet.</p>
        )}
        {history.kind === "loaded" && history.response.attempts.length > 0 && (
          <div className="space-y-3">
            {history.response.attempts.map((attempt) => (
              <MacroPublishAttemptRow key={attempt.id} attempt={attempt} />
            ))}
          </div>
        )}
      </div>
    </section>
  )
}

function MacroPublishAttemptRow({ attempt }: { attempt: GeneratedAssetMacroPublishAttempt }) {
  const counts = macroPublishCountLabels(attempt)
  const tone = attempt.ok
    ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-100"
    : attempt.pending_reconcile_count > 0 || attempt.skipped_count > 0
      ? "border-amber-500/30 bg-amber-500/10 text-amber-100"
      : "border-rose-500/30 bg-rose-500/10 text-rose-100"
  const title = attempt.ok
    ? "Published"
    : attempt.pending_reconcile_count > 0
      ? "Needs reconcile"
      : attempt.failed_count > 0
        ? "Failed"
        : "Needs review"

  return (
    <div className={clsx("rounded-md border p-3", tone)}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-sm font-medium">{title}</div>
        <div className="font-mono text-xs opacity-80">{attempt.created_at || attempt.id}</div>
      </div>
      <div className="mt-1 text-xs opacity-90">
        Draft status: {attempt.draft_status}
        {attempt.draft_status_updated ? "; moved after publish." : "; status unchanged."}
      </div>
      {counts.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2 text-xs">
          {counts.map((count) => (
            <span
              key={count}
              className="rounded border border-current/20 bg-slate-950/30 px-2 py-0.5"
            >
              {count}
            </span>
          ))}
        </div>
      )}
      {attempt.skipped.length > 0 && (
        <div className="mt-3 text-xs opacity-90">
          Skipped: {attempt.skipped.map(macroPublishSkippedLabel).join("; ")}
        </div>
      )}
      {attempt.results.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          {attempt.results.slice(0, 4).map((result, index) => (
            <span
              key={`${result.external_id || "result"}-${index}`}
              className="rounded border border-current/20 bg-slate-950/30 px-2 py-0.5"
            >
              {macroPublishResultLabel(result)}
            </span>
          ))}
          {attempt.results.length > 4 && (
            <span className="rounded border border-current/20 bg-slate-950/30 px-2 py-0.5">
              +{attempt.results.length - 4} more
            </span>
          )}
        </div>
      )}
    </div>
  )
}

function macroPublishSkippedLabel(item: Record<string, unknown>): string {
  const question = textValue(item.question) || "macro"
  const reason = textValue(item.reason) || "skipped"
  return `${question}: ${reason}`
}

function macroPublishResultLabel(result: Record<string, unknown>): string {
  const status = textValue(result.status) || "unknown"
  const externalId = textValue(result.external_id)
  const error = textValue(result.error)
  if (externalId) return `${status} ${externalId}`
  if (error) return `${status}: ${error}`
  return status
}


function AssetDetailDrawer({
  row,
  asset,
  saving,
  publishHistory,
  onSaveLandingPage,
  onRepairLandingPage,
  onClose,
}: {
  row: GeneratedAssetDraft
  asset: GeneratedAssetType
  saving: boolean
  publishHistory: MacroPublishHistoryState
  onSaveLandingPage: (
    row: GeneratedAssetDraft,
    update: GeneratedLandingPageDraftUpdate,
  ) => Promise<GeneratedAssetDraft>
  onRepairLandingPage: (row: GeneratedAssetDraft) => Promise<GeneratedAssetDraft>
  onClose: () => void
}) {
  const title = assetTitle(row, asset)
  const status = textValue(row.status) || 'unknown'
  const preview = assetPreview(row, asset)
  const facts = assetFacts(row, asset)
  const sections = sectionList(row.sections)
  const references = valueList(row.reference_ids)
  const faqItems = asset === 'faq_markdown' ? faqItemList(row.items) : []
  const readinessPanels = assetReadinessPanels(row, asset)
  const repairHistory = assetRepairHistory(row)
  const structuredData =
    asset === 'landing_page' ? structuredDataSummary(row.structured_data) : null
  const publicUrl = publicAssetUrl(row, asset)
  const publicUrlPending = publicAssetUrlPending(row, asset)
  const canEditLandingPage =
    asset === 'landing_page' && Boolean(assetId(row)) && status !== 'approved'
  const canRepairLandingPage =
    canEditLandingPage && landingPageNeedsRepair(readinessPanels)
  const [isEditing, setIsEditing] = useState(false)
  const [editState, setEditState] = useState<LandingPageEditState>(() =>
    landingPageEditState(row),
  )
  const [editError, setEditError] = useState<string | null>(null)
  const [repairError, setRepairError] = useState<string | null>(null)
  const [copiedUrl, setCopiedUrl] = useState(false)
  const drawerRef = useRef<HTMLElement | null>(null)
  const closeButtonRef = useRef<HTMLButtonElement | null>(null)

  const handleCopyPublicUrl = async () => {
    if (!publicUrl) return
    await copyText(publicUrl)
    setCopiedUrl(true)
    window.setTimeout(() => setCopiedUrl(false), 1800)
  }

  useEffect(() => {
    const previous = document.activeElement
    closeButtonRef.current?.focus()
    return () => {
      if (previous instanceof HTMLElement) {
        previous.focus()
      }
    }
  }, [])

  const handleSaveEdit = async () => {
    setEditError(null)
    try {
      const update = landingPageUpdatePayload(row, editState)
      await onSaveLandingPage(row, update)
      setIsEditing(false)
    } catch (err) {
      setEditError(err instanceof Error ? err.message : String(err))
    }
  }

  const handleRepair = async () => {
    setRepairError(null)
    try {
      await onRepairLandingPage(row)
      setIsEditing(false)
    } catch (err) {
      setRepairError(err instanceof Error ? err.message : String(err))
    }
  }

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
          <div className="flex flex-wrap gap-2">
            {canEditLandingPage && (
              <button
                type="button"
                onClick={() => {
                  setEditError(null)
                  if (isEditing) {
                    setIsEditing(false)
                    return
                  }
                  setEditState(landingPageEditState(row))
                  setIsEditing(true)
                }}
                disabled={saving}
                className="inline-flex items-center gap-2 rounded-md border border-slate-700 px-3 py-2 text-sm text-slate-200 hover:border-cyan-400 hover:text-cyan-200 disabled:opacity-60"
              >
                <Pencil className="h-4 w-4" />
                {isEditing ? 'View' : 'Edit'}
              </button>
            )}
            {canRepairLandingPage && (
              <button
                type="button"
                onClick={() => void handleRepair()}
                disabled={saving}
                className="inline-flex items-center gap-2 rounded-md border border-amber-500/40 px-3 py-2 text-sm text-amber-200 hover:bg-amber-500/10 disabled:opacity-60"
              >
                {saving ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Wrench className="h-4 w-4" />
                )}
                Repair
              </button>
            )}
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
        </div>

        {repairError && (
          <div className="mt-4 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
            {repairError}
          </div>
        )}

        {isEditing && canEditLandingPage && (
          <LandingPageDraftEditor
            state={editState}
            saving={saving}
            error={editError}
            onChange={setEditState}
            onCancel={() => {
              setEditError(null)
              setEditState(landingPageEditState(row))
              setIsEditing(false)
            }}
            onSave={() => void handleSaveEdit()}
          />
        )}

        {(publicUrl || publicUrlPending) && (
          <section className="mt-6">
            <h3 className="text-sm font-semibold text-slate-200">Public URL</h3>
            <div className="mt-3 rounded-md border border-slate-800 bg-slate-900/70 p-4">
              {publicUrl ? (
                <>
                  <p className="break-all font-mono text-xs leading-5 text-slate-300">
                    {publicUrl}
                  </p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <a
                      href={publicUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-2 rounded-md border border-cyan-500/40 px-3 py-2 text-sm text-cyan-200 hover:bg-cyan-500/10"
                    >
                      <ExternalLink className="h-4 w-4" />
                      Open
                    </a>
                    <button
                      type="button"
                      onClick={() => void handleCopyPublicUrl()}
                      className="inline-flex items-center gap-2 rounded-md border border-slate-700 px-3 py-2 text-sm text-slate-200 hover:border-cyan-400 hover:text-cyan-200"
                    >
                      <Copy className="h-4 w-4" />
                      {copiedUrl ? 'Copied' : 'Copy'}
                    </button>
                  </div>
                  <p className="mt-3 text-xs text-slate-500">
                    {asset === 'landing_page'
                      ? 'Public rendering is approved-only. This v1 URL is marked noindex.'
                      : 'Public rendering is approved-only. Approved blog posts are available at /blog/:slug.'}
                  </p>
                </>
              ) : (
                <p className="text-sm text-slate-400">
                  Approve this {asset === 'blog_post' ? 'blog post' : 'landing page'} to make its public URL available.
                </p>
              )}
            </div>
          </section>
        )}

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

        {asset === 'faq_markdown' && (
          <MacroPublishHistoryPanel history={publishHistory} />
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

        {readinessPanels.length > 0 && (
          <ReadinessBreakdown panels={readinessPanels} />
        )}

        {repairHistory.length > 0 && (
          <RepairHistory history={repairHistory} />
        )}

        {structuredData && (
          <StructuredDataPreview summary={structuredData} />
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

        {faqItems.length > 0 && (
          <section className="mt-6">
            <h3 className="text-sm font-semibold text-slate-200">FAQ Items</h3>
            <div className="mt-3 space-y-3">
              {faqItems.map((item, index) => (
                <div
                  key={`${item.question}-${index}`}
                  className="rounded-md border border-slate-800 bg-slate-900/60 p-4"
                >
                  <div className="text-sm font-medium text-white">
                    {item.question || `Question ${index + 1}`}
                  </div>
                  {item.answer && (
                    <p className="mt-2 text-sm leading-6 text-slate-400">
                      {item.answer}
                    </p>
                  )}
                  {item.actionItems.length > 0 && (
                    <div className="mt-3">
                      <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                        What to do next
                      </div>
                      <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-300">
                        {item.actionItems.map((action, actionIndex) => (
                          <li key={`${action}-${actionIndex}`}>{action}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {item.sourceLabels.length > 0 && (
                    <div className="mt-3">
                      <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                        Sources
                      </div>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {item.sourceLabels.map((source, sourceIndex) => (
                          <span
                            key={`${source}-${sourceIndex}`}
                            className="rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-300"
                          >
                            {source}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {item.termMappings.length > 0 && (
                    <div className="mt-3">
                      <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                        Vocabulary gaps
                      </div>
                      <ul className="mt-2 space-y-2 text-sm text-slate-300">
                        {item.termMappings.map((mapping, mappingIndex) => (
                          <li
                            key={`${mapping.customerTerm}-${mapping.documentationTerm}-${mappingIndex}`}
                            className="border-l border-cyan-500/40 pl-3"
                          >
                            <div>
                              <span className="text-cyan-200">
                                {mapping.customerTerm}
                              </span>
                              <span className="text-slate-500">{' -> '}</span>
                              <span>{mapping.documentationTerm}</span>
                            </div>
                            <div className="mt-1 flex flex-wrap gap-2 text-xs text-slate-500">
                              {mapping.sourceIdCount && (
                                <span>{mapping.sourceIdCount} source(s)</span>
                              )}
                              {mapping.zeroResultSourceCount && (
                                <span>
                                  {mapping.zeroResultSourceCount} zero-result source(s)
                                </span>
                              )}
                              {mapping.opportunityScore && (
                                <span>score {mapping.opportunityScore}</span>
                              )}
                            </div>
                          </li>
                        ))}
                      </ul>
                    </div>
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

function LandingPageDraftEditor({
  state,
  saving,
  error,
  onChange,
  onCancel,
  onSave,
}: {
  state: LandingPageEditState
  saving: boolean
  error: string | null
  onChange: (state: LandingPageEditState) => void
  onCancel: () => void
  onSave: () => void
}) {
  const updateField = <K extends keyof LandingPageEditState>(
    key: K,
    value: LandingPageEditState[K],
  ) => onChange({ ...state, [key]: value })
  const updateSection = (
    index: number,
    update: Partial<LandingPageSectionEdit>,
  ) => {
    onChange({
      ...state,
      sections: state.sections.map((section, sectionIndex) =>
        sectionIndex === index ? { ...section, ...update } : section,
      ),
    })
  }
  const addSection = () => {
    onChange({
      ...state,
      sections: [
        ...state.sections,
        {
          id: `section_${state.sections.length + 1}`,
          title: '',
          body_markdown: '',
          metadata: {},
        },
      ],
    })
  }
  const removeSection = (index: number) => {
    onChange({
      ...state,
      sections: state.sections.filter((_, sectionIndex) => sectionIndex !== index),
    })
  }

  return (
    <section className="mt-6 rounded-md border border-cyan-500/30 bg-cyan-500/5 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-cyan-100">Edit landing page</h3>
          <p className="mt-1 text-xs text-slate-400">
            Saving resets this asset to draft and reruns readiness checks.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={onCancel}
            disabled={saving}
            className="rounded-md border border-slate-700 px-3 py-2 text-sm text-slate-200 hover:border-cyan-400 hover:text-cyan-200 disabled:opacity-60"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onSave}
            disabled={saving}
            className="inline-flex items-center gap-2 rounded-md bg-cyan-500 px-3 py-2 text-sm font-medium text-slate-950 hover:bg-cyan-400 disabled:opacity-60"
          >
            {saving ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Save className="h-4 w-4" />
            )}
            Save
          </button>
        </div>
      </div>

      {error && (
        <div className="mt-4 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
          {error}
        </div>
      )}

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <Field label="Title">
          <input
            value={state.title}
            onChange={(event) => updateField('title', event.target.value)}
            className={inputClassName}
          />
        </Field>
        <Field label="Slug">
          <input
            value={state.slug}
            onChange={(event) => updateField('slug', event.target.value)}
            className={inputClassName}
          />
        </Field>
        <Field label="Hero headline">
          <input
            value={state.heroHeadline}
            onChange={(event) => updateField('heroHeadline', event.target.value)}
            className={inputClassName}
          />
        </Field>
        <Field label="Hero subheadline">
          <input
            value={state.heroSubheadline}
            onChange={(event) => updateField('heroSubheadline', event.target.value)}
            className={inputClassName}
          />
        </Field>
        <Field label="Hero CTA label">
          <input
            value={state.heroCtaLabel}
            onChange={(event) => updateField('heroCtaLabel', event.target.value)}
            className={inputClassName}
          />
        </Field>
        <Field label="Hero CTA URL">
          <input
            value={state.heroCtaUrl}
            onChange={(event) => updateField('heroCtaUrl', event.target.value)}
            className={inputClassName}
          />
        </Field>
        <Field label="CTA label">
          <input
            value={state.ctaLabel}
            onChange={(event) => updateField('ctaLabel', event.target.value)}
            className={inputClassName}
          />
        </Field>
        <Field label="CTA URL">
          <input
            value={state.ctaUrl}
            onChange={(event) => updateField('ctaUrl', event.target.value)}
            className={inputClassName}
          />
        </Field>
        <Field label="Meta title">
          <input
            value={state.metaTitleTag}
            onChange={(event) => updateField('metaTitleTag', event.target.value)}
            className={inputClassName}
          />
        </Field>
        <Field label="Meta description">
          <textarea
            value={state.metaDescription}
            onChange={(event) => updateField('metaDescription', event.target.value)}
            rows={3}
            className={textAreaClassName}
          />
        </Field>
      </div>

      <Field label="Reference IDs">
        <textarea
          value={state.referenceIdsText}
          onChange={(event) => updateField('referenceIdsText', event.target.value)}
          rows={2}
          className={textAreaClassName}
        />
      </Field>

      <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
        <h4 className="text-sm font-semibold text-slate-200">Sections</h4>
        <button
          type="button"
          onClick={addSection}
          disabled={saving}
          className="rounded-md border border-slate-700 px-3 py-2 text-sm text-slate-200 hover:border-cyan-400 hover:text-cyan-200 disabled:opacity-60"
        >
          Add section
        </button>
      </div>

      <div className="mt-3 space-y-3">
        {state.sections.length === 0 ? (
          <div className="rounded-md border border-slate-800 bg-slate-950/40 p-4 text-sm text-slate-400">
            No sections yet.
          </div>
        ) : (
          state.sections.map((section, index) => (
            <div
              key={`${section.id}-${index}`}
              className="rounded-md border border-slate-800 bg-slate-950/40 p-4"
            >
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="text-sm font-medium text-white">
                  Section {index + 1}
                </div>
                <button
                  type="button"
                  onClick={() => removeSection(index)}
                  disabled={saving}
                  className="rounded-md border border-rose-500/40 px-2 py-1 text-xs text-rose-200 hover:bg-rose-500/10 disabled:opacity-60"
                >
                  Remove
                </button>
              </div>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <Field label="ID">
                  <input
                    value={section.id}
                    onChange={(event) => updateSection(index, { id: event.target.value })}
                    className={inputClassName}
                  />
                </Field>
                <Field label="Title">
                  <input
                    value={section.title}
                    onChange={(event) => updateSection(index, { title: event.target.value })}
                    className={inputClassName}
                  />
                </Field>
              </div>
              <Field label="Body">
                <textarea
                  value={section.body_markdown}
                  onChange={(event) =>
                    updateSection(index, { body_markdown: event.target.value })
                  }
                  rows={5}
                  className={textAreaClassName}
                />
              </Field>
            </div>
          ))
        )}
      </div>
    </section>
  )
}

function Field({
  label,
  children,
}: {
  label: string
  children: ReactNode
}) {
  return (
    <label className="mt-3 block text-xs font-medium text-slate-400">
      <span>{label}</span>
      <div className="mt-1">{children}</div>
    </label>
  )
}

function StructuredDataPreview({ summary }: { summary: StructuredDataSummary }) {
  return (
    <section className="mt-6">
      <h3 className="text-sm font-semibold text-slate-200">Structured Data</h3>
      <div className="mt-3 rounded-md border border-slate-800 bg-slate-900/60 p-4">
        <div className="flex flex-wrap gap-2 text-xs">
          {summary.nodeTypes.map((type) => (
            <span
              key={type}
              className="inline-flex items-center gap-1 rounded bg-slate-800 px-2 py-1 text-slate-300"
            >
              <Code2 className="h-3.5 w-3.5 text-cyan-300" />
              {type}
            </span>
          ))}
          <span className="rounded bg-slate-800 px-2 py-1 text-slate-300">
            {summary.questionCount} FAQ question{summary.questionCount === 1 ? '' : 's'}
          </span>
          <span className={clsx(
            'rounded px-2 py-1',
            summary.hasCanonical
              ? 'bg-emerald-500/10 text-emerald-200'
              : 'bg-amber-500/10 text-amber-200',
          )}>
            {summary.hasCanonical ? 'canonical linked' : 'no canonical URL'}
          </span>
        </div>
        <pre className="mt-3 max-h-80 overflow-auto rounded border border-slate-800 bg-slate-950/70 p-3 text-xs leading-5 text-slate-300">
          {JSON.stringify(summary.raw, null, 2)}
        </pre>
      </div>
    </section>
  )
}

function RepairHistory({ history }: { history: RepairHistoryEntry[] }) {
  return (
    <section className="mt-6">
      <h3 className="text-sm font-semibold text-slate-200">Repair History</h3>
      <div className="mt-3 space-y-3">
        {history.map((entry) => (
          <div
            key={`${entry.attempt}-${entry.passed ? 'passed' : 'blocked'}`}
            className="rounded-md border border-slate-800 bg-slate-900/60 p-4"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="text-sm font-medium text-white">
                Attempt {entry.attempt}
              </div>
              <span
                className={clsx(
                  'inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs',
                  entry.passed
                    ? 'bg-emerald-500/10 text-emerald-200'
                    : 'bg-amber-500/10 text-amber-200',
                )}
              >
                {entry.passed ? (
                  <CheckCircle2 className="h-3.5 w-3.5" />
                ) : (
                  <XCircle className="h-3.5 w-3.5" />
                )}
                {entry.passed ? 'Passed' : 'Blocked'}
              </span>
            </div>

            {entry.blockers.length > 0 && (
              <RepairHistoryList title="Blockers" items={entry.blockers} tone="amber" />
            )}

            {entry.repairIssues.length > 0 && (
              <RepairHistoryList title="Repair issues" items={entry.repairIssues} tone="slate" />
            )}

            {entry.blockers.length === 0 && entry.repairIssues.length === 0 && (
              <p className="mt-3 text-xs text-slate-500">
                No blockers or repair issues reported for this attempt.
              </p>
            )}
          </div>
        ))}
      </div>
    </section>
  )
}

function RepairHistoryList({
  title,
  items,
  tone,
}: {
  title: string
  items: string[]
  tone: 'amber' | 'slate'
}) {
  return (
    <div
      className={clsx(
        'mt-3 rounded border p-2',
        tone === 'amber'
          ? 'border-amber-500/20 bg-amber-500/10'
          : 'border-slate-800 bg-slate-950/40',
      )}
    >
      <div
        className={clsx(
          'text-xs font-semibold uppercase tracking-wide',
          tone === 'amber' ? 'text-amber-200' : 'text-slate-500',
        )}
      >
        {title}
      </div>
      <ul
        className={clsx(
          'mt-1 list-disc space-y-1 pl-4 text-xs',
          tone === 'amber' ? 'text-amber-100/90' : 'text-slate-300',
        )}
      >
        {items.map((item, index) => (
          <li key={`${item}-${index}`}>{fmtLabel(item)}</li>
        ))}
      </ul>
    </div>
  )
}

function repairHistorySummary(
  history: RepairHistoryEntry[],
): { label: string; passed: boolean } | null {
  if (history.length === 0) return null
  const latest = history[history.length - 1]
  const repairCount = Math.max(0, history.length - 1)
  const repairLabel =
    repairCount === 0
      ? 'without repair'
      : `after ${repairCount} repair${repairCount === 1 ? '' : 's'}`
  return {
    label: latest.passed
      ? `quality passed ${repairLabel}`
      : `quality blocked ${repairLabel}`,
    passed: latest.passed,
  }
}

function ReadinessBreakdown({ panels }: { panels: ReadinessPanel[] }) {
  return (
    <section className="mt-6">
      <h3 className="text-sm font-semibold text-slate-200">Readiness</h3>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        {panels.map((panel) => (
          <div
            key={panel.label}
            className="rounded-md border border-slate-800 bg-slate-900/60 p-4"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <div className="text-sm font-medium text-white">{panel.label}</div>
                <div className="mt-1 text-xs text-slate-500">
                  {readinessCountLabel(panel)}
                </div>
              </div>
              <span className={clsx(
                'rounded px-2 py-0.5 text-xs',
                readinessStatusClass(panel.status),
              )}>
                {fmtLabel(panel.status || 'unknown')}
              </span>
            </div>

            {panel.missing.length > 0 && (
              <div className="mt-3 rounded border border-amber-500/20 bg-amber-500/10 p-2">
                <div className="text-xs font-semibold uppercase tracking-wide text-amber-200">
                  Missing
                </div>
                <ul className="mt-1 list-disc space-y-1 pl-4 text-xs text-amber-100/90">
                  {panel.missing.map((item) => (
                    <li key={item}>{fmtLabel(item)}</li>
                  ))}
                </ul>
              </div>
            )}

            {panel.checks.length > 0 && (
              <div className="mt-3 space-y-1.5">
                {panel.checks.map((check) => (
                  <div
                    key={check.label}
                    className="flex items-start gap-2 text-xs text-slate-300"
                  >
                    {check.passed ? (
                      <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-300" />
                    ) : (
                      <XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-300" />
                    )}
                    <span>{check.label}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  )
}

function landingPageEditState(row: GeneratedAssetDraft): LandingPageEditState {
  const hero = recordValue(row.hero)
  const cta = recordValue(row.cta)
  const meta = recordValue(row.meta)
  return {
    title: textValue(row.title),
    slug: textValue(row.slug),
    heroHeadline: objectText(hero, 'headline'),
    heroSubheadline: objectText(hero, 'subheadline'),
    heroCtaLabel: objectText(hero, 'cta_label'),
    heroCtaUrl: objectText(hero, 'cta_url'),
    ctaLabel: objectText(cta, 'label'),
    ctaUrl: objectText(cta, 'url'),
    metaTitleTag: objectText(meta, 'title_tag'),
    metaDescription: objectText(meta, 'description'),
    referenceIdsText: valueList(row.reference_ids).join('\n'),
    sections: editableLandingPageSections(row.sections),
  }
}

function landingPageUpdatePayload(
  row: GeneratedAssetDraft,
  state: LandingPageEditState,
): GeneratedLandingPageDraftUpdate {
  return {
    title: state.title,
    slug: state.slug,
    hero: {
      ...(recordValue(row.hero) ?? {}),
      headline: state.heroHeadline,
      subheadline: state.heroSubheadline,
      cta_label: state.heroCtaLabel,
      cta_url: state.heroCtaUrl,
    },
    cta: {
      ...(recordValue(row.cta) ?? {}),
      label: state.ctaLabel,
      url: state.ctaUrl,
    },
    meta: {
      ...(recordValue(row.meta) ?? {}),
      title_tag: state.metaTitleTag,
      description: state.metaDescription,
    },
    sections: state.sections.map((section, index) => ({
      id: section.id.trim() || `section_${index + 1}`,
      title: section.title,
      body_markdown: section.body_markdown,
      metadata: { ...section.metadata },
    })),
    reference_ids: parseReferenceIds(state.referenceIdsText),
  }
}

function editableLandingPageSections(value: unknown): LandingPageSectionEdit[] {
  return recordList(value).map((section, index) => {
    const metadata = section.metadata
    return {
      id: textValue(section.id) || `section_${index + 1}`,
      title: textValue(section.title) || textValue(section.heading),
      body_markdown: textValue(section.body_markdown) || textValue(section.body),
      metadata:
        metadata && typeof metadata === 'object' && !Array.isArray(metadata)
          ? { ...(metadata as Record<string, unknown>) }
          : {},
    }
  })
}

function parseReferenceIds(value: string): string[] {
  return value
    .split(/[\n,]/)
    .map((item) => item.trim())
    .filter(Boolean)
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

function publicAssetUrl(
  row: GeneratedAssetDraft,
  asset: GeneratedAssetType,
): string | null {
  if (textValue(row.status) !== 'approved') return null
  const slug = textValue(row.slug)
  if (!slug) return null
  if (asset === 'blog_post') {
    return `${window.location.origin}/blog/${encodeURIComponent(slug)}`
  }
  if (asset === 'landing_page') {
    const id = assetId(row)
    if (!id) return null
    return `${window.location.origin}/lp/${encodeURIComponent(id)}/${encodeURIComponent(slug)}`
  }
  return null
}

function publicAssetUrlPending(
  row: GeneratedAssetDraft,
  asset: GeneratedAssetType,
): boolean {
  if (textValue(row.status) === 'approved') return false
  const slug = textValue(row.slug)
  if (!slug) return false
  if (asset === 'blog_post') return true
  return asset === 'landing_page' && Boolean(assetId(row))
}

function assetTitle(row: GeneratedAssetDraft, asset: GeneratedAssetType): string {
  if (asset === 'social_post') {
    return (
      textValue(row.company_name) ||
      textValue(row.vendor_name) ||
      textValue(row.target_id) ||
      excerpt(textValue(row.text), 72) ||
      'social_post draft'
    )
  }
  if (asset === 'ad_copy') {
    return (
      textValue(row.headline) ||
      textValue(row.company_name) ||
      textValue(row.vendor_name) ||
      textValue(row.target_id) ||
      'ad_copy draft'
    )
  }
  if (asset === 'quote_card') {
    return (
      textValue(row.headline) ||
      textValue(row.attribution) ||
      textValue(row.company_name) ||
      textValue(row.vendor_name) ||
      excerpt(textValue(row.quote), 72) ||
      'quote_card draft'
    )
  }
  if (asset === 'stat_card') {
    return (
      textValue(row.claim) ||
      textValue(row.headline) ||
      textValue(row.metric_display) ||
      textValue(row.company_name) ||
      textValue(row.vendor_name) ||
      'stat_card draft'
    )
  }
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
  if (asset === 'social_post') {
    return [
      row.channel,
      row.company_name,
      row.vendor_name,
      row.source_type,
      row.source_id,
    ].map(textValue).filter(Boolean).join(' | ')
  }
  if (asset === 'ad_copy') {
    return [
      row.channel,
      row.format,
      row.company_name,
      row.vendor_name,
      row.source_type,
      row.source_id,
    ].map(textValue).filter(Boolean).join(' | ')
  }
  if (asset === 'quote_card') {
    return [
      row.theme,
      row.attribution,
      row.company_name,
      row.vendor_name,
      row.source_type,
      row.source_id,
    ].map(textValue).filter(Boolean).join(' | ')
  }
  if (asset === 'stat_card') {
    return [
      row.theme,
      row.metric_label,
      row.metric_display,
      row.company_name,
      row.vendor_name,
      row.source_type,
      row.source_id,
    ].map(textValue).filter(Boolean).join(' | ')
  }
  if (asset === 'faq_markdown') {
    return [
      row.target_mode,
      numberLabel(row.ticket_source_count, 'ticket source'),
      numberLabel(row.source_count, 'source row'),
    ].filter(Boolean).join(' | ')
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
    addFact(facts, 'SEO/AEO', readinessLabel(row.seo_aeo_readiness))
    addFact(facts, 'GEO', readinessLabel(row.geo_readiness))
    return
  }
  if (asset === 'landing_page') {
    addFact(facts, 'campaign', row.campaign_name)
    addFact(facts, 'persona', row.persona)
    addFact(facts, 'SEO/AEO', readinessLabel(row.seo_aeo_readiness))
    addFact(facts, 'GEO', readinessLabel(row.geo_readiness))
    return
  }
  if (asset === 'faq_markdown') {
    addFact(facts, 'ticket sources', row.ticket_source_count)
    addFact(facts, 'source rows', row.source_count)
    addFact(facts, 'checks passed', row.passed_output_checks)
    addFact(facts, 'vocab gaps', faqVocabularyGapCount(row.items))
    return
  }
  if (asset === 'social_post') {
    addFact(facts, 'channel', row.channel)
    addFact(facts, 'company', row.company_name)
    addFact(facts, 'vendor', row.vendor_name)
    addFact(facts, 'source', [row.source_type, row.source_id].map(textValue).filter(Boolean).join(':'))
    addFact(facts, 'pain points', socialPostPainPointCount(row))
    return
  }
  if (asset === 'ad_copy') {
    addFact(facts, 'channel', row.channel)
    addFact(facts, 'format', row.format)
    addFact(facts, 'company', row.company_name)
    addFact(facts, 'vendor', row.vendor_name)
    addFact(facts, 'source', [row.source_type, row.source_id].map(textValue).filter(Boolean).join(':'))
    addFact(facts, 'pain points', adCopyPainPointCount(row))
    return
  }
  if (asset === 'quote_card') {
    addFact(facts, 'theme', row.theme)
    addFact(facts, 'attribution', row.attribution)
    addFact(facts, 'company', row.company_name)
    addFact(facts, 'vendor', row.vendor_name)
    addFact(facts, 'source', [row.source_type, row.source_id].map(textValue).filter(Boolean).join(':'))
    addFact(facts, 'pain points', quoteCardPainPointCount(row))
    return
  }
  if (asset === 'stat_card') {
    addFact(facts, 'theme', row.theme)
    addFact(facts, 'metric', row.metric_label)
    addFact(facts, 'value', row.metric_display || row.metric_value)
    addFact(facts, 'company', row.company_name)
    addFact(facts, 'vendor', row.vendor_name)
    addFact(facts, 'source', [row.source_type, row.source_id].map(textValue).filter(Boolean).join(':'))
    addFact(facts, 'pain points', statCardPainPointCount(row))
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
      meta: [
        ...readinessMeta(row),
        ...tags,
      ],
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
  if (asset === 'faq_markdown') {
    const items = faqItemList(row.items)
    const first = items[0]
    const checks = outputCheckLabels(row.output_checks)
    const vocabularyGapCount = faqVocabularyGapCount(row.items)
    const sourceMeta =
      first?.sourceLabels.slice(0, 2).map((source) => `source: ${source}`) ?? []
    return previewOrNull({
      heading: first?.question || textValue(row.title),
      body: excerpt(first?.answer || textValue(row.markdown)),
      meta: [
        ...sourceMeta,
        vocabularyGapCount ? `vocab gaps: ${vocabularyGapCount}` : '',
        ...checks.slice(0, 3).map((check) => `check: ${check}`),
      ].filter(Boolean),
    })
  }
  if (asset === 'social_post') {
    const painPoints = valueList(row.pain_points).slice(0, 3)
    return previewOrNull({
      heading: [
        textValue(row.channel),
        textValue(row.company_name) || textValue(row.vendor_name),
      ].filter(Boolean).join(' | '),
      body: excerpt(textValue(row.text), 280),
      meta: painPoints.map((item) => `pain: ${item}`),
    })
  }
  if (asset === 'ad_copy') {
    const painPoints = valueList(row.pain_points).slice(0, 3)
    const cta = textValue(row.cta)
    return previewOrNull({
      heading: textValue(row.headline),
      body: excerpt(textValue(row.primary_text), 280),
      meta: [
        textValue(row.channel),
        textValue(row.format),
        cta ? `CTA: ${cta}` : '',
        ...painPoints.map((item) => `pain: ${item}`),
      ].filter(Boolean),
    })
  }
  if (asset === 'quote_card') {
    const painPoints = valueList(row.pain_points).slice(0, 3)
    return previewOrNull({
      heading: textValue(row.headline) || textValue(row.attribution),
      body: excerpt(textValue(row.quote), 280),
      meta: [
        textValue(row.theme),
        textValue(row.attribution),
        textValue(row.supporting_text),
        ...painPoints.map((item) => `pain: ${item}`),
      ].filter(Boolean),
    })
  }
  if (asset === 'stat_card') {
    const painPoints = valueList(row.pain_points).slice(0, 3)
    return previewOrNull({
      heading: textValue(row.claim) || textValue(row.headline),
      body: excerpt(textValue(row.evidence), 280),
      meta: [
        textValue(row.theme),
        textValue(row.metric_label),
        textValue(row.metric_display),
        textValue(row.supporting_text),
        ...painPoints.map((item) => `pain: ${item}`),
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
  return recordList(value)
    .map((section) => ({
      title: textValue(section.title) || textValue(section.heading),
      body: textValue(section.body_markdown) || textValue(section.body),
    }))
}

function faqItemList(value: unknown): FAQItemPreview[] {
  return recordList(value).map((item) => ({
    question: textValue(item.question) || textValue(item.topic),
    answer: textValue(item.answer),
    actionItems: valueList(item.action_items),
    sourceLabels: valueList(item.source_labels),
    termMappings: faqTermMappingList(item.term_mappings),
  })).filter((item) => item.question || item.answer)
}

function faqTermMappingList(value: unknown): FAQTermMappingPreview[] {
  return recordList(value)
    .map((mapping) => ({
      customerTerm: textValue(mapping.customer_term),
      documentationTerm: textValue(mapping.documentation_term),
      sourceIdCount: numberText(mapping.source_id_count),
      zeroResultSourceCount: numberText(mapping.zero_result_source_count),
      opportunityScore: numberText(mapping.opportunity_score),
    }))
    .filter((mapping) => mapping.customerTerm || mapping.documentationTerm)
}

function faqVocabularyGapCount(value: unknown): number {
  return faqItemList(value).reduce(
    (total, item) => total + item.termMappings.length,
    0,
  )
}

function recordList(value: unknown): Record<string, unknown>[] {
  let rows = Array.isArray(value) ? value : []
  if (typeof value === 'string' && value.trim()) {
    try {
      const parsed = JSON.parse(value)
      rows = Array.isArray(parsed) ? parsed : []
    } catch {
      rows = []
    }
  }
  return rows.filter((item): item is Record<string, unknown> =>
    Boolean(item && typeof item === 'object' && !Array.isArray(item)),
  )
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

function assetRepairHistory(row: GeneratedAssetDraft): RepairHistoryEntry[] {
  return [
    row.generation_quality_repair_history,
    row.quality_repair_history,
    metadataValue(row, 'generation_quality_repair_history'),
    metadataValue(row, 'quality_repair_history'),
  ]
    .map(repairHistoryList)
    .find((history) => history.length > 0) ?? []
}

function metadataValue(row: GeneratedAssetDraft, key: string): unknown {
  const metadata = recordValue(row.metadata)
  return metadata ? metadata[key] : undefined
}

function repairHistoryList(value: unknown): RepairHistoryEntry[] {
  return recordList(value)
    .map(repairHistoryEntry)
    .filter((entry): entry is RepairHistoryEntry => Boolean(entry))
}

function repairHistoryEntry(item: Record<string, unknown>): RepairHistoryEntry | null {
  const attempt = finiteNumber(item.attempt)
  if (attempt == null) return null
  return {
    attempt,
    passed: item.passed === true,
    blockers: valueList(item.blockers),
    repairIssues: valueList(item.repair_issues),
  }
}

function outputCheckLabels(value: unknown): string[] {
  const checks = recordValue(value)
  if (!checks) return []
  return Object.entries(checks)
    .filter(([, passed]) => passed === true)
    .map(([key]) => key.replace(/_/g, ' '))
}

function readinessMeta(row: GeneratedAssetDraft): string[] {
  return [
    readinessLabel(row.seo_aeo_readiness, 'SEO/AEO'),
    readinessLabel(row.geo_readiness, 'GEO'),
  ].filter(Boolean)
}

function socialPostPainPointCount(row: GeneratedAssetDraft): number | string {
  if (typeof row.pain_point_count === 'number' && Number.isFinite(row.pain_point_count)) {
    return row.pain_point_count
  }
  return valueList(row.pain_points).length || ''
}

function adCopyPainPointCount(row: GeneratedAssetDraft): number | string {
  if (typeof row.pain_point_count === 'number' && Number.isFinite(row.pain_point_count)) {
    return row.pain_point_count
  }
  return valueList(row.pain_points).length || ''
}

function quoteCardPainPointCount(row: GeneratedAssetDraft): number | string {
  if (typeof row.pain_point_count === 'number' && Number.isFinite(row.pain_point_count)) {
    return row.pain_point_count
  }
  return valueList(row.pain_points).length || ''
}

function statCardPainPointCount(row: GeneratedAssetDraft): number | string {
  if (typeof row.pain_point_count === 'number' && Number.isFinite(row.pain_point_count)) {
    return row.pain_point_count
  }
  return valueList(row.pain_points).length || ''
}

function readinessLabel(value: unknown, prefix?: string): string {
  const summary = readinessSummary(value)
  if (!summary) return ''
  const status = fmtLabel(summary.status || 'unknown')
  const count =
    summary.passed != null && summary.total != null
      ? ` (${summary.passed}/${summary.total})`
      : ''
  const missing =
    summary.status === 'needs_review' && summary.missing.length > 0
      ? ` missing ${summary.missing.slice(0, 2).map(fmtLabel).join(', ')}`
      : ''
  return `${prefix ? `${prefix}: ` : ''}${status}${count}${missing}`
}

function readinessSummary(value: unknown): ReadinessSummary | null {
  const item = recordValue(value)
  if (!item) return null
  const status = textValue(item.status)
  const passed = finiteNumber(item.passed)
  const total = finiteNumber(item.total)
  const missing = valueList(item.missing)
  const checks = readinessChecks(item.checks)
  if (!status && passed == null && total == null && missing.length === 0 && checks.length === 0) {
    return null
  }
  return {
    status,
    passed,
    total,
    missing,
    checks,
  }
}

function assetReadinessPanels(
  row: GeneratedAssetDraft,
  asset: GeneratedAssetType,
): ReadinessPanel[] {
  if (asset !== 'blog_post' && asset !== 'landing_page') return []
  return [
    readinessPanel('SEO/AEO', row.seo_aeo_readiness),
    readinessPanel('GEO', row.geo_readiness),
  ].filter((panel): panel is ReadinessPanel => Boolean(panel))
}

function landingPageNeedsRepair(panels: ReadinessPanel[]): boolean {
  if (panels.length === 0) return true
  return panels.some((panel) => panel.status !== 'ready')
}

function readinessPanel(label: string, value: unknown): ReadinessPanel | null {
  const summary = readinessSummary(value)
  return summary ? { label, ...summary } : null
}

function readinessChecks(value: unknown): ReadinessCheck[] {
  const checks = recordValue(value)
  if (!checks) return []
  return Object.entries(checks)
    .filter(([, passed]) => typeof passed === 'boolean')
    .map(([key, passed]) => ({
      label: fmtLabel(key),
      passed: passed === true,
    }))
}

function structuredDataSummary(value: unknown): StructuredDataSummary | null {
  const raw = recordValue(value)
  if (!raw) return null
  const graph = recordList(raw['@graph'])
  const nodes = graph.length > 0 ? graph : [raw]
  const nodeTypes = Array.from(new Set(nodes.flatMap(schemaTypes))).filter(Boolean)
  const questionCount = nodes
    .filter((node) => schemaTypes(node).includes('FAQPage'))
    .reduce((count, node) => count + recordList(node.mainEntity).length, 0)
  return {
    raw,
    nodeTypes: nodeTypes.length > 0 ? nodeTypes : ['Schema.org'],
    questionCount,
    hasCanonical: nodes.some((node) => Boolean(textValue(node.url) || textValue(node['@id']))),
  }
}

function schemaTypes(node: Record<string, unknown>): string[] {
  const type = node['@type']
  if (Array.isArray(type)) return type.map(textValue).filter(Boolean)
  return textValue(type) ? [textValue(type)] : []
}

function readinessCountLabel(summary: ReadinessSummary): string {
  if (summary.passed == null || summary.total == null) return 'No count reported'
  return `${summary.passed} of ${summary.total} checks passed`
}

function readinessStatusClass(status: string): string {
  if (status === 'ready') return 'bg-emerald-500/10 text-emerald-200'
  if (status === 'needs_review') return 'bg-amber-500/10 text-amber-200'
  return 'bg-slate-800 text-slate-300'
}

function finiteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function fmtLabel(value: string): string {
  return value.replace(/_/g, ' ')
}

function numberLabel(value: unknown, label: string): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return ''
  return `${value} ${label}${value === 1 ? '' : 's'}`
}

function excerpt(value: string, limit = 260): string {
  const clean = value.replace(/\s+/g, ' ').trim()
  if (clean.length <= limit) return clean
  return `${clean.slice(0, limit - 3).trim()}...`
}

function downloadText(value: string, filename: string, type: string): void {
  const blob = new Blob([value], { type })
  downloadBlob(blob, filename)
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 0)
}

async function copyText(value: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value)
    return
  }
  const textArea = document.createElement('textarea')
  textArea.value = value
  textArea.setAttribute('readonly', '')
  textArea.style.position = 'fixed'
  textArea.style.top = '-1000px'
  document.body.appendChild(textArea)
  textArea.select()
  document.execCommand('copy')
  textArea.remove()
}
