import { useState, useCallback, useEffect } from 'react'
import { Link, useParams, useNavigate, useSearchParams, useLocation } from 'react-router-dom'
import { ArrowLeft, RefreshCw, Copy, Check } from 'lucide-react'
import { clsx } from 'clsx'
import { PageError } from '../components/ErrorBoundary'
import ReportTrustPanel from '../components/ReportTrustPanel'
import {
  StructuredReportData,
  StructuredReportValue,
} from '../components/report-renderers/StructuredReportData'
import { SpecializedReportData } from '../components/report-renderers/SpecializedReportData'
import { REPORT_SCALAR_KEYS, isSpecializedReportType, REPORT_TYPE_COLORS } from '../lib/reportConstants'
import useApiData from '../hooks/useApiData'
import { fetchReport } from '../api/client'
import type { ReportDetail as ReportDetailType } from '../types'
import ReportActionBar from '../components/ReportActionBar'
import SubscriptionModal from '../components/SubscriptionModal'
import EvidenceDrawer from '../components/EvidenceDrawer'
import CitationBar from '../components/report-renderers/CitationBar'
import { createCitationRegistry } from '../components/report-renderers/useCitationRegistry'
import type { CitationEntry } from '../components/report-renderers/useCitationRegistry'

function formatInlineValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return '--'
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }
  if (Array.isArray(value)) return `${value.length} item${value.length === 1 ? '' : 's'}`
  if (typeof value === 'object') return `${Object.keys(value).length} field${Object.keys(value).length === 1 ? '' : 's'}`
  return '--'
}

function DetailSkeleton() {
  return (
    <div className="space-y-6 max-w-4xl animate-pulse">
      <div className="h-4 w-28 bg-slate-700/50 rounded" />
      <div>
        <div className="h-5 w-32 bg-slate-700/50 rounded mb-2" />
        <div className="h-7 w-56 bg-slate-700/50 rounded mb-2" />
        <div className="h-4 w-40 bg-slate-700/50 rounded" />
      </div>
      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 h-32" />
      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 h-48" />
    </div>
  )
}

function vendorDetailPath(vendorName: string, backTo?: string) {
  const base = `/vendors/${encodeURIComponent(vendorName)}`
  if (!backTo) return base
  const next = new URLSearchParams()
  next.set('back_to', backTo)
  return `${base}?${next.toString()}`
}

function evidencePath(vendorName: string, backTo: string, asOfDate?: string | null, windowDays?: number | null) {
  const next = new URLSearchParams()
  next.set('vendor', vendorName)
  next.set('tab', 'witnesses')
  if (asOfDate) next.set('as_of_date', asOfDate)
  if (windowDays) next.set('window_days', String(windowDays))
  next.set('back_to', backTo)
  return `/evidence?${next.toString()}`
}

function evidenceWitnessPath(
  vendorName: string,
  witnessId: string,
  backTo: string,
  asOfDate?: string | null,
  windowDays?: number | null,
) {
  const next = new URLSearchParams()
  next.set('vendor', vendorName)
  next.set('tab', 'witnesses')
  if (asOfDate) next.set('as_of_date', asOfDate)
  if (windowDays) next.set('window_days', String(windowDays))
  next.set('witness_id', witnessId)
  next.set('back_to', backTo)
  return `/evidence?${next.toString()}`
}

function opportunitiesPath(vendorName: string, backTo: string) {
  const next = new URLSearchParams()
  next.set('vendor', vendorName)
  next.set('back_to', backTo)
  return `/opportunities?${next.toString()}`
}

function upstreamAccountReviewCompany(value: string | null, vendorName: string): string | null {
  const watchlistsPath = upstreamWatchlistsPath(value)
  if (!watchlistsPath) return null

  try {
    const url = new URL(watchlistsPath, window.location.origin)
    const accountCompany = url.searchParams.get('account_company')?.trim()
    const accountVendor = url.searchParams.get('account_vendor')?.trim()
    if (!accountCompany) return null
    if (accountVendor && accountVendor.toLowerCase() !== vendorName.trim().toLowerCase()) return null
    return accountCompany
  } catch {
    return null
  }
}

function alertsPath(vendorName: string, backTo: string) {
  const next = new URLSearchParams()
  next.set('vendor', vendorName)
  const accountCompany = upstreamAccountReviewCompany(backTo, vendorName)
  if (accountCompany) {
    next.set('company', accountCompany)
  }
  next.set('back_to', backTo)
  return `/alerts?${next.toString()}`
}

// Shared specialized and structured report rendering is imported for this page.

const REPORT_DETAIL_BACK_TARGET_RULES = [
  { prefix: '/vendors/', label: 'Back to Vendor' },
  { prefix: '/vendors', label: 'Back to Vendors' },
  { prefix: '/watchlists', label: 'Back to Watchlists' },
  { prefix: '/alerts', label: 'Back to Alerts' },
  { prefix: '/reviews/', label: 'Back to Review' },
  { prefix: '/reviews', label: 'Back to Reviews' },
  { prefix: '/evidence', label: 'Back to Evidence' },
  { prefix: '/opportunities', label: 'Back to Opportunities' },
  { prefix: '/dashboard', label: 'Back to Dashboard' },
  { prefix: '/affiliates', label: 'Back to Affiliates' },
  { prefix: '/vendor-targets', label: 'Back to Vendor Targets' },
  { prefix: '/blog-review', label: 'Back to Blog Review' },
  { prefix: '/briefing-review', label: 'Back to Briefing Review' },
  { prefix: '/campaign-review', label: 'Back to Campaign Review' },
  { prefix: '/prospects', label: 'Back to Prospects' },
  { prefix: '/challengers', label: 'Back to Challengers' },
  { prefix: '/pipeline-review', label: 'Back to Pipeline Review' },
  { prefix: '/predictor', label: 'Back to Predictor' },
  { prefix: '/reports/', label: 'Back to Report' },
  { prefix: '/reports', label: 'Back to Reports' },
  { prefix: '/report', label: 'Back to Report' },
] as const

function backToLabel(value: string) {
  if (value.startsWith('/watchlists')) {
    try {
      const url = new URL(value, window.location.origin)
      if (url.searchParams.get('account_company')?.trim()) return 'Back to Account Review'
    } catch {
      // Fall through to the generic label.
    }
    return 'Back to Watchlists'
  }
  return REPORT_DETAIL_BACK_TARGET_RULES.find((rule) => value.startsWith(rule.prefix))?.label ?? 'Back to Reports'
}

function upstreamWatchlistsPath(value: string | null): string | null {
  let current = value?.trim() || ''

  for (let depth = 0; depth < 4 && current; depth += 1) {
    if (current.startsWith('/watchlists')) return current
    try {
      const url = new URL(current, window.location.origin)
      current = url.searchParams.get('back_to')?.trim() || ''
    } catch {
      return null
    }
  }

  return null
}

function watchlistsShortcutLabel(value: string | null) {
  if (!value) return 'Watchlists'
  try {
    const url = new URL(value, window.location.origin)
    return url.searchParams.get('account_company')?.trim() ? 'Account Review' : 'Watchlists'
  } catch {
    return 'Watchlists'
  }
}

function upstreamEvidencePath(value: string | null): string | null {
  let current = value?.trim() || ''

  for (let depth = 0; depth < 4 && current; depth += 1) {
    if (current.startsWith('/evidence')) return current
    try {
      const url = new URL(current, window.location.origin)
      current = url.searchParams.get('back_to')?.trim() || ''
    } catch {
      return null
    }
  }

  return null
}

function upstreamAlertsPath(value: string | null): string | null {
  let current = value?.trim() || ''

  for (let depth = 0; depth < 4 && current; depth += 1) {
    if (current.startsWith('/alerts')) return current
    try {
      const url = new URL(current, window.location.origin)
      current = url.searchParams.get('back_to')?.trim() || ''
    } catch {
      return null
    }
  }

  return null
}

// ---------------------------------------------------------------------------
// Page component
// ---------------------------------------------------------------------------

export default function ReportDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const location = useLocation()
  const [searchParams, setSearchParams] = useSearchParams()

  const [subModalOpen, setSubModalOpen] = useState(false)
  const [subscriptionStateOverride, setSubscriptionStateOverride] = useState<'active' | 'paused' | null>(null)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [drawerWitnessId, setDrawerWitnessId] = useState<string | null>(null)
  const [drawerVendor, setDrawerVendor] = useState('')
  const [copiedDirectWatchlistsLink, setCopiedDirectWatchlistsLink] = useState(false)

  const handleOpenWitness = useCallback((witnessId: string, vendorName: string) => {
    setDrawerWitnessId(witnessId)
    setDrawerVendor(vendorName)
    setDrawerOpen(true)
  }, [])

  const { data: report, loading, error, refresh, refreshing } = useApiData<ReportDetailType>(
    () => {
      if (!id) return Promise.reject(new Error('Missing report ID'))
      return fetchReport(id)
    },
    [id],
  )

  useEffect(() => {
    setSubscriptionStateOverride(null)
  }, [id])

  useEffect(() => {
    setSubModalOpen(searchParams.get('subscription') === 'report')
  }, [searchParams])

  if (error) return <PageError error={error} onRetry={refresh} />
  if (loading) return <DetailSkeleton />
  if (!report) return <PageError error={new Error('Report not found')} />

  const baseSubscriptionState = report.report_subscription
    ? (report.report_subscription.enabled ? 'active' : 'paused')
    : 'none'
  const subscriptionState = subscriptionStateOverride ?? baseSubscriptionState
  const reportScopeLabel = report.report_subscription?.scope_label
    ?? `${report.report_type.replace(/_/g, ' ')} - ${report.vendor_filter ?? 'all'}`
  const isAllowedReportDetailBackTarget = (candidate: string) => (
    REPORT_DETAIL_BACK_TARGET_RULES.some((rule) => candidate.startsWith(rule.prefix))
  )
  const stateBackTo = typeof location.state === 'object' && location.state && 'backTo' in location.state
    && typeof (location.state as { backTo?: unknown }).backTo === 'string'
    && isAllowedReportDetailBackTarget((location.state as { backTo: string }).backTo)
    ? (location.state as { backTo: string }).backTo
    : null
  const queryBackTo = (() => {
    const value = searchParams.get('back_to')
    return value && isAllowedReportDetailBackTarget(value) ? value : null
  })()
  const backToReports = stateBackTo ?? queryBackTo ?? '/reports'
  const backButtonLabel = backToLabel(backToReports)
  const directWatchlistsPath = upstreamWatchlistsPath(backToReports)
  const directWatchlistsLabel = watchlistsShortcutLabel(directWatchlistsPath)
  const directEvidencePath = upstreamEvidencePath(backToReports)
  const directAlertsPath = upstreamAlertsPath(backToReports)
  const detailShareUrl = (() => {
    const next = new URLSearchParams()
    if (subModalOpen) {
      next.set('subscription', 'report')
      next.set('report_focus_label', reportScopeLabel)
    }
    if (backToReports !== '/reports') {
      next.set('back_to', backToReports)
    }
    const qs = next.toString()
    return qs ? `/reports/${report.id}?${qs}` : `/reports/${report.id}`
  })()
  const detailBackPath = (() => {
    const next = new URLSearchParams()
    if (backToReports !== '/reports') {
      next.set('back_to', backToReports)
    }
    const qs = next.toString()
    return qs ? `/reports/${report.id}?${qs}` : `/reports/${report.id}`
  })()
  const reportEvidenceAsOfDate = report.as_of_date ?? report.report_date ?? null
  const reportEvidenceWindowDays = report.analysis_window_days ?? null
  const drawerExplorerUrl = drawerWitnessId && drawerVendor
    ? evidenceWitnessPath(
        drawerVendor,
        drawerWitnessId,
        detailBackPath,
        reportEvidenceAsOfDate,
        reportEvidenceWindowDays,
      )
    : null
  const handleCopyDirectWatchlistsLink = () => {
    if (!directWatchlistsPath) return
    void navigator.clipboard.writeText(`${window.location.origin}${directWatchlistsPath}`).then(() => {
      setCopiedDirectWatchlistsLink(true)
      window.setTimeout(() => {
        setCopiedDirectWatchlistsLink(false)
      }, 2000)
    })
  }
  const badgeColor = REPORT_TYPE_COLORS[report.report_type] ?? 'bg-slate-500/20 text-slate-400'
  const title = ['vendor_comparison', 'account_comparison'].includes(report.report_type) && report.vendor_filter && report.category_filter
    ? `${report.vendor_filter} vs ${report.category_filter}`
    : report.report_type === 'challenger_brief' && report.vendor_filter && report.category_filter
      ? `${report.vendor_filter} -> ${report.category_filter}`
      : (report.vendor_filter ?? report.report_type.replace(/_/g, ' '))

  // intelligence_data can be an object (keyed fields) or an array (vendor/edge rows)
  const rawIntel = report.intelligence_data
  const intelIsArray = Array.isArray(rawIntel)
  const intel = intelIsArray ? {} : (rawIntel ?? {})
  const scalarEntries = Object.entries(intel).filter(([k]) => REPORT_SCALAR_KEYS.has(k))
  const richEntries = Object.entries(intel).filter(([k, v]) => {
    if (REPORT_SCALAR_KEYS.has(k)) return false
    // Skip duplicate of top-level executive_summary
    if (k === 'executive_summary') return false
    // Skip empty arrays / empty strings / null
    if (v === null || v === undefined) return false
    if (typeof v === 'string' && v.trim() === '') return false
    if (Array.isArray(v) && v.length === 0) return false
    return true
  })

  return (
    <div className="space-y-6 max-w-6xl min-w-0">
      <div className="flex items-center justify-between">
        <button
          onClick={() => navigate(backToReports)}
          className="flex items-center gap-1 text-sm text-slate-400 hover:text-white transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          {backButtonLabel}
        </button>
        <div className="flex items-center gap-2">
          <ReportActionBar
            reportId={report.id}
            onSubscribe={() => {
              const next = new URLSearchParams(searchParams)
              next.set('subscription', 'report')
              setSearchParams(next, { replace: true })
            }}
            hasSubscription={subscriptionState !== 'none'}
            subscriptionState={subscriptionState}
            hasPdfExport={report.has_pdf_export ?? false}
            artifactState={report.artifact_state ?? report.trust?.artifact_state}
            artifactLabel={report.artifact_label ?? report.trust?.artifact_label}
            linkUrl={detailShareUrl}
          />
          <button
            onClick={refresh}
            disabled={refreshing}
            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
          </button>
        </div>
      </div>

      <div>
        <span className={clsx('inline-flex items-center px-2 py-0.5 rounded text-xs font-medium mb-2', badgeColor)}>
          {report.report_type.replace(/_/g, ' ')}
        </span>
        <h1 className="text-2xl font-bold text-white break-words">
          {title}
        </h1>
        {report.vendor_filter ? (
          <div className="mt-2 flex flex-wrap items-center gap-3 text-sm">
            <span className="text-slate-500">
              Focused on <span className="text-slate-300">{report.vendor_filter}</span>
            </span>
            {directWatchlistsPath ? (
              <>
                <Link
                  to={directWatchlistsPath}
                  className={directWatchlistsLabel === 'Account Review'
                    ? 'text-amber-300 hover:text-amber-200 transition-colors'
                    : 'text-violet-300 hover:text-violet-200 transition-colors'}
                >
                  {directWatchlistsLabel}
                </Link>
                <button
                  onClick={() => void handleCopyDirectWatchlistsLink()}
                  className="inline-flex items-center gap-1.5 text-xs text-slate-400 hover:text-white transition-colors"
                  title={directWatchlistsLabel === 'Account Review' ? 'Copy account review link' : 'Copy watchlists link'}
                >
                  {copiedDirectWatchlistsLink ? (
                    <>
                      <Check className="h-3.5 w-3.5 text-emerald-400" />
                      Copied
                    </>
                  ) : (
                    <>
                      <Copy className="h-3.5 w-3.5" />
                      {directWatchlistsLabel === 'Account Review' ? 'Copy Account Review Link' : 'Copy Watchlists Link'}
                    </>
                  )}
                </button>
              </>
            ) : null}
            <Link
              to={vendorDetailPath(report.vendor_filter, detailBackPath)}
              className="text-cyan-400 hover:text-cyan-300 transition-colors"
            >
              Vendor workspace
            </Link>
            <Link
              to={directEvidencePath ?? evidencePath(
                report.vendor_filter,
                detailBackPath,
                reportEvidenceAsOfDate,
                reportEvidenceWindowDays,
              )}
              className="text-violet-300 hover:text-violet-200 transition-colors"
            >
              Evidence
            </Link>
            <Link
              to={opportunitiesPath(report.vendor_filter, detailBackPath)}
              className="text-emerald-300 hover:text-emerald-200 transition-colors"
            >
              Opportunities
            </Link>
            <Link
              to={directAlertsPath ?? alertsPath(report.vendor_filter, detailBackPath)}
              className="text-rose-300 hover:text-rose-200 transition-colors"
            >
              Alerts API
            </Link>
          </div>
        ) : null}
        <div className="flex items-center gap-3 mt-1">
          <p className="text-sm text-slate-400 break-all">
            {report.report_date ?? report.created_at}
            {report.llm_model && ` | Model: ${report.llm_model}`}
          </p>
        </div>
        <div className="mt-4">
          <ReportTrustPanel
            status={report.status}
            artifactState={report.artifact_state ?? report.trust?.artifact_state}
            artifactLabel={report.artifact_label ?? report.trust?.artifact_label}
            blockerCount={report.blocker_count}
            warningCount={report.warning_count}
            unresolvedIssueCount={report.unresolved_issue_count}
            qualityStatus={report.quality_status}
            latestFailureStep={report.latest_failure_step}
            latestErrorSummary={report.latest_error_summary}
            freshnessState={report.freshness_state ?? report.trust?.freshness_state}
            freshnessLabel={report.freshness_label ?? report.trust?.freshness_label}
            reviewState={report.review_state ?? report.trust?.review_state}
            reviewLabel={report.review_label ?? report.trust?.review_label}
            freshnessTimestamp={report.report_date ?? report.created_at}
          />
        </div>
      </div>

      {report.executive_summary && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 min-w-0 overflow-hidden">
          <h3 className="text-sm font-medium text-slate-300 mb-2">Executive Summary</h3>
          <p className="text-sm text-slate-300 whitespace-pre-wrap break-words">{report.executive_summary}</p>
          {(() => {
            if (!report.vendor_filter) return null
            const refIds = intel.reasoning_reference_ids
            const wids = refIds && typeof refIds === 'object' && !Array.isArray(refIds) && Array.isArray((refIds as Record<string, unknown>).witness_ids)
              ? (refIds as Record<string, unknown>).witness_ids as string[]
              : null
            if (!wids || wids.length === 0) return null
            const hlRaw = Array.isArray(intel.reasoning_witness_highlights)
              ? intel.reasoning_witness_highlights
              : intel.witness_highlights
            const highlights = Array.isArray(hlRaw) ? hlRaw as Array<Record<string, unknown>> : []
            const reg = createCitationRegistry()
            const hlMap = new Map<string, { companyName?: string; excerptSnippet?: string }>()
            for (const hl of highlights) {
              const wid = typeof hl.witness_id === 'string' ? hl.witness_id : null
              if (wid) {
                hlMap.set(wid, {
                  companyName: typeof hl.reviewer_company === 'string' ? hl.reviewer_company : undefined,
                  excerptSnippet: typeof hl.excerpt_text === 'string' ? hl.excerpt_text.slice(0, 80) : undefined,
                })
              }
            }
            const entries: CitationEntry[] = wids.map((wid) => {
              const meta = hlMap.get(wid)
              const idx = reg.register(wid, meta)
              return { index: idx, witnessId: wid, companyName: meta?.companyName, excerptSnippet: meta?.excerptSnippet }
            })
            return (
              <CitationBar
                citations={entries}
                vendorName={report.vendor_filter}
                onOpenWitness={handleOpenWitness}
              />
            )
          })()}
        </div>
      )}

      {/* Challenger brief - dedicated renderer */}
      {isSpecializedReportType(report.report_type) && (
        <SpecializedReportData reportType={report.report_type} data={intelIsArray ? rawIntel : intel} vendorName={report.vendor_filter ?? undefined} onOpenWitness={handleOpenWitness} />
      )}

      {/* Generic rendering for all other report types */}
      {!isSpecializedReportType(report.report_type) && (
        <>
          {/* Array-based reports (scorecard, displacement, category overview, churn feed) */}
          {intelIsArray && (rawIntel as Record<string, unknown>[]).length > 0 && (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
              <h3 className="text-sm font-medium text-slate-300 mb-3">
                {report.report_type.replace(/_/g, ' ')} ({(rawIntel as unknown[]).length} items)
              </h3>
              <StructuredReportValue fieldKey={report.report_type} value={rawIntel} />
            </div>
          )}

          {/* Scalar stats row */}
          {scalarEntries.length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
              {scalarEntries.map(([key, value]) => (
                <div key={key} className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 text-center">
                  <p className="text-xs text-slate-400 mb-1">{key.replace(/_/g, ' ')}</p>
                  <p className="text-xl font-bold text-white">{String(value)}</p>
                </div>
              ))}
            </div>
          )}

          {/* Rich intelligence fields */}
          {richEntries.length > 0 && (
            <StructuredReportData
              data={intel}
              sectionEvidence={report.section_evidence}
              vendorName={report.vendor_filter ?? undefined}
              onOpenWitness={handleOpenWitness}
            />
          )}
        </>
      )}

      {report.data_density && Object.keys(report.data_density).length > 0 && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 min-w-0 overflow-hidden">
          <h3 className="text-sm font-medium text-slate-300 mb-3">Data Density</h3>
          <dl className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
            {Object.entries(report.data_density).map(([key, value]) => (
              <div key={key} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-2">
                <dt className="text-slate-400 min-w-0 break-words">{key.replace(/_/g, ' ')}</dt>
                <dd className="text-white text-right min-w-0 break-all">{formatInlineValue(value)}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}

      {/* Subscription modal */}
      <SubscriptionModal
        open={subModalOpen}
        onClose={() => {
          const next = new URLSearchParams(searchParams)
          next.delete('subscription')
          setSearchParams(next, { replace: true })
        }}
        scopeType="report"
        scopeKey={report.id}
        scopeLabel={reportScopeLabel}
        onSaved={(subscription) => setSubscriptionStateOverride(subscription.enabled ? 'active' : 'paused')}
      />

      {/* Evidence drawer */}
      <EvidenceDrawer
        vendorName={drawerVendor}
        witnessId={drawerWitnessId}
        open={drawerOpen}
        asOfDate={reportEvidenceAsOfDate}
        windowDays={reportEvidenceWindowDays ?? undefined}
        explorerUrl={drawerExplorerUrl}
        backToPath={detailBackPath}
        onClose={() => {
          setDrawerOpen(false)
          setDrawerWitnessId(null)
        }}
      />
    </div>
  )
}
