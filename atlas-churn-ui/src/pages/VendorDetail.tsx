import { useState } from 'react'
import { useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { ArrowLeft, RefreshCw } from 'lucide-react'
import { clsx } from 'clsx'
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import UrgencyBadge from '../components/UrgencyBadge'
import ArchetypeBadge from '../components/ArchetypeBadge'
import DataTable, { type Column } from '../components/DataTable'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import {
  compareVendorPeriods,
  fetchReports,
  fetchVendorHistory,
  fetchVendorProfile,
  fetchReviews,
} from '../api/client'
import type {
  Report,
  VendorHistoryResponse,
  VendorPeriodComparisonResponse,
  VendorProfile,
  ReviewSummary,
} from '../types'

interface VendorData {
  profile: VendorProfile
  reviews: ReviewSummary[]
  recentReports: Report[]
  history: VendorHistoryResponse
  comparison: VendorPeriodComparisonResponse
  loadWarnings: string[]
}

function DetailSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="h-4 w-32 bg-slate-700/50 rounded" />
      <div className="flex items-center justify-between">
        <div>
          <div className="h-7 w-48 bg-slate-700/50 rounded mb-2" />
          <div className="h-4 w-32 bg-slate-700/50 rounded" />
        </div>
        <div className="h-10 w-16 bg-slate-700/50 rounded" />
      </div>
      <div className="h-10 w-64 bg-slate-700/50 rounded" />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 h-64" />
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 h-64" />
      </div>
    </div>
  )
}

function extractNestedLabel(value: unknown): string {
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (typeof value === 'object' && value !== null) {
    const obj = value as Record<string, unknown>
    const nested = obj.name ?? obj.label ?? obj.value ?? obj.text
    return nested ? extractNestedLabel(nested) : ''
  }
  return ''
}

function extractQuoteText(value: unknown): string {
  if (typeof value === 'string') return value
  if (typeof value === 'object' && value !== null) {
    const obj = value as Record<string, unknown>
    return extractNestedLabel(obj.quote ?? obj.text ?? obj.value ?? obj.summary)
  }
  return ''
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function asNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function toRecordArray(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.filter((item): item is Record<string, unknown> => typeof item === 'object' && item !== null && !Array.isArray(item))
    : []
}

function toStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value
        .map((item) => asString(item))
        .filter(Boolean)
    : []
}

function humanizeToken(value: string): string {
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function toLoadWarning(section: string, err: unknown): string {
  const detail = err instanceof Error && err.message.trim()
    ? ` ${err.message.trim()}`
    : ''
  return `${section} is temporarily unavailable.${detail}`
}

function emptyVendorHistory(vendorName: string): VendorHistoryResponse {
  return {
    vendor_name: vendorName,
    snapshots: [],
    count: 0,
  }
}

function emptyVendorComparison(vendorName: string): VendorPeriodComparisonResponse {
  return {
    vendor_name: vendorName,
    period_a: null,
    period_b: null,
    deltas: {},
  }
}

function vendorDetailPath(vendorName: string): string {
  return `/vendors/${encodeURIComponent(vendorName)}`
}

function normalizeBackTo(value: string | null | undefined): string | null {
  if (!value) return null
  const allowedPrefixes = ['/vendors', '/watchlists', '/evidence', '/reports', '/opportunities', '/reviews']
  const isAllowedPath = (candidate: string) => allowedPrefixes.some((prefix) => candidate.startsWith(prefix))
  if (isAllowedPath(value)) return value
  try {
    const url = new URL(value, window.location.origin)
    if (url.origin === window.location.origin && isAllowedPath(url.pathname)) {
      return `${url.pathname}${url.search}`
    }
  } catch {
    return null
  }
  return null
}

function backToLabel(backTo: string): string {
  if (backTo.startsWith('/watchlists')) {
    try {
      const url = new URL(backTo, window.location.origin)
      if (url.searchParams.get('account_company')?.trim()) return 'Back to Account Review'
    } catch {
      // Fall through to the generic label.
    }
    return 'Back to Watchlists'
  }
  if (backTo.startsWith('/evidence')) return 'Back to Evidence'
  if (backTo.startsWith('/reports')) return 'Back to Reports'
  if (backTo.startsWith('/opportunities')) return 'Back to Opportunities'
  if (backTo.startsWith('/reviews')) return 'Back to Review'
  return 'Back to Vendors'
}

function upstreamWatchlistsPath(backTo: string | null): string | null {
  if (!backTo?.startsWith('/evidence')) return null
  try {
    const url = new URL(backTo, window.location.origin)
    return normalizeBackTo(url.searchParams.get('back_to'))
  } catch {
    return null
  }
}

function upstreamWatchlistsLabel(backTo: string | null): string {
  if (!backTo) return 'Open Watchlists'
  try {
    const url = new URL(backTo, window.location.origin)
    return url.searchParams.get('account_company')?.trim()
      ? 'Open Account Review'
      : 'Open Watchlists'
  } catch {
    return 'Open Watchlists'
  }
}

function upstreamEvidencePath(backTo: string | null, vendorName: string): string | null {
  if (!backTo) return null
  if (backTo.startsWith('/evidence')) return normalizeBackTo(backTo)
  if (!backTo.startsWith('/watchlists')) return null

  const params = new URLSearchParams()
  params.set('vendor', vendorName)
  params.set('tab', 'witnesses')

  try {
    const url = new URL(backTo, window.location.origin)
    const witnessId = url.searchParams.get('witness_id')?.trim()
    const source = url.searchParams.get('source')?.trim()
    if (witnessId) params.set('witness_id', witnessId)
    if (source) params.set('source', source)
  } catch {
    return null
  }

  params.set('back_to', backTo)
  return `/evidence?${params.toString()}`
}

function vendorEvidenceExplorerPath(vendorName: string): string {
  const params = new URLSearchParams()
  params.set('vendor', vendorName)
  params.set('tab', 'witnesses')
  params.set('back_to', vendorDetailPath(vendorName))
  return `/evidence?${params.toString()}`
}

function vendorReportsPath(vendorName: string): string {
  const params = new URLSearchParams()
  params.set('vendor_filter', vendorName)
  params.set('back_to', vendorDetailPath(vendorName))
  return `/reports?${params.toString()}`
}

function vendorOpportunitiesPath(vendorName: string): string {
  const params = new URLSearchParams()
  params.set('vendor', vendorName)
  params.set('back_to', vendorDetailPath(vendorName))
  return `/opportunities?${params.toString()}`
}

function vendorReviewDetailPath(reviewId: string, vendorName: string): string {
  const params = new URLSearchParams()
  params.set('back_to', vendorDetailPath(vendorName))
  return `/reviews/${reviewId}?${params.toString()}`
}

function vendorReportDetailPath(reportId: string, vendorName: string): string {
  const params = new URLSearchParams()
  params.set('back_to', vendorDetailPath(vendorName))
  return `/reports/${encodeURIComponent(reportId)}?${params.toString()}`
}

function reportTypeLabel(reportType: string): string {
  return reportType.replace(/_/g, ' ')
}

function reportStatusLabel(report: Report): string {
  return report.quality_status
    || report.freshness_state
    || report.review_state
    || report.artifact_state
    || report.status
    || 'unknown'
}

function formatReportTimestamp(report: Report): string {
  const raw = report.report_date || report.created_at
  if (!raw) return 'No report date yet'
  const parsed = new Date(raw)
  return Number.isNaN(parsed.getTime())
    ? raw
    : parsed.toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      })
}

export default function VendorDetail() {
  const { name } = useParams<{ name: string }>()
  const navigate = useNavigate()
  const location = useLocation()
  const [searchParams] = useSearchParams()
  const [tab, setTab] = useState<'overview' | 'reviews' | 'companies'>('overview')
  const [selectedSlowBurnMetric, setSelectedSlowBurnMetric] = useState<
    'support_sentiment' | 'legacy_support_score' | 'new_feature_velocity' | 'employee_growth_rate'
  >('support_sentiment')

  const { data, loading, error, refresh, refreshing } = useApiData<VendorData>(
    async () => {
      if (!name) throw new Error('Missing vendor name')
      const profile = await fetchVendorProfile(name)
      const [reviewsResult, historyResult, comparisonResult, reportsResult] = await Promise.all([
        fetchReviews({ vendor_name: name, limit: 50, window_days: 365 })
          .then((result) => ({ ok: true as const, data: result.reviews }))
          .catch((err: unknown) => ({ ok: false as const, error: err })),
        fetchVendorHistory(name, { days: 90, limit: 24 })
          .then((result) => ({ ok: true as const, data: result }))
          .catch((err: unknown) => ({ ok: false as const, error: err })),
        compareVendorPeriods(name, { period_a_days_ago: 30, period_b_days_ago: 0 })
          .then((result) => ({ ok: true as const, data: result }))
          .catch((err: unknown) => ({ ok: false as const, error: err })),
        fetchReports({ vendor_filter: name, limit: 3, include_stale: true })
          .then((result) => ({ ok: true as const, data: result.reports }))
          .catch((err: unknown) => ({ ok: false as const, error: err })),
      ])

      const loadWarnings: string[] = []
      const reviews = reviewsResult.ok
        ? reviewsResult.data
        : (loadWarnings.push(toLoadWarning('Enriched reviews', reviewsResult.error)), [])
      const history = historyResult.ok
        ? historyResult.data
        : (loadWarnings.push(toLoadWarning('Trend history', historyResult.error)), emptyVendorHistory(name))
      const comparison = comparisonResult.ok
        ? comparisonResult.data
        : (loadWarnings.push(toLoadWarning('Period comparison', comparisonResult.error)), emptyVendorComparison(name))
      const recentReports = reportsResult.ok
        ? reportsResult.data
        : (loadWarnings.push(toLoadWarning('Recent reports', reportsResult.error)), [])

      return { profile, reviews, recentReports, history, comparison, loadWarnings }
    },
    [name],
  )

  if (error) return <PageError error={error} onRetry={refresh} />
  if (loading) return <DetailSkeleton />

  const profile = data?.profile
  if (!profile) return <PageError error={new Error('Vendor not found')} />
  const stateBackTo = typeof location.state === 'object' && location.state && 'backTo' in location.state
    ? normalizeBackTo((location.state as { backTo?: string | null }).backTo)
    : null
  const queryBackTo = normalizeBackTo(searchParams.get('back_to'))
  const backTo = stateBackTo ?? queryBackTo ?? '/vendors'
  const backLabel = backToLabel(backTo)
  const watchlistsReturnPath = upstreamWatchlistsPath(backTo)
  const watchlistsReturnLabel = upstreamWatchlistsLabel(watchlistsReturnPath)
  const evidenceExplorerPath = upstreamEvidencePath(backTo, profile.vendor_name) ?? vendorEvidenceExplorerPath(profile.vendor_name)
  const reportsPath = vendorReportsPath(profile.vendor_name)
  const opportunitiesPath = vendorOpportunitiesPath(profile.vendor_name)
  const recentReports = data?.recentReports ?? []
  const recentReportsCard = (
    <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <h3 className="text-sm font-medium text-slate-300">Recent Reports</h3>
          <p className="text-xs text-slate-500 mt-1">
            Vendor-scoped reports without leaving this workspace.
          </p>
        </div>
        <button
          onClick={() => navigate(reportsPath)}
          className="text-xs text-cyan-300 hover:text-cyan-200 transition-colors"
        >
          View all
        </button>
      </div>
      {recentReports.length > 0 ? (
        <div className="space-y-2">
          {recentReports.map((report) => (
            <button
              key={report.id}
              onClick={() => navigate(vendorReportDetailPath(report.id, profile.vendor_name))}
              className="w-full rounded-lg border border-slate-700/50 bg-slate-800/40 px-3 py-3 text-left hover:border-cyan-500/30 hover:bg-slate-800/70 transition-colors"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-sm text-white">{reportTypeLabel(report.report_type)}</p>
                  <p className="mt-1 text-xs text-slate-500">{formatReportTimestamp(report)}</p>
                </div>
                <span className="rounded-full bg-slate-900 px-2 py-0.5 text-[10px] uppercase tracking-wide text-slate-300">
                  {reportStatusLabel(report)}
                </span>
              </div>
              {report.executive_summary && (
                <p className="mt-2 line-clamp-2 text-xs text-slate-400">
                  {report.executive_summary}
                </p>
              )}
            </button>
          ))}
        </div>
      ) : (
        <div className="rounded-lg border border-dashed border-slate-700/60 px-3 py-4">
          <p className="text-sm text-slate-400">No recent reports for this vendor yet.</p>
          <button
            onClick={() => navigate(reportsPath)}
            className="mt-2 text-xs text-cyan-300 hover:text-cyan-200 transition-colors"
          >
            Open report library
          </button>
        </div>
      )}
    </div>
  )

  const signal = profile.churn_signal
  const reasoningScope = asRecord(signal?.reasoning_scope_manifest)
  const reasoningAtoms = asRecord(signal?.reasoning_atoms)
  const reasoningDelta = asRecord(signal?.reasoning_delta)
  const witnessMix = Object.entries(asRecord(reasoningScope.witness_mix))
    .map(([label, value]) => ({
      label: humanizeToken(label),
      count: asNumber(value) ?? 0,
    }))
    .filter((item) => item.count > 0)
  const timeCoverage = Object.entries(asRecord(reasoningScope.coverage_by_time_bucket))
    .map(([bucket, value]) => ({
      bucket: humanizeToken(bucket),
      count: asNumber(value) ?? 0,
    }))
    .filter((item) => item.count > 0)
  const theses = toRecordArray(reasoningAtoms.theses).slice(0, 3)
  const timingWindows = toRecordArray(reasoningAtoms.timing_windows).slice(0, 4)
  const coverageLimits = toRecordArray(reasoningAtoms.coverage_limits).slice(0, 4)
  const counterevidence = toRecordArray(reasoningAtoms.counterevidence).slice(0, 3)
  const reasoningKeySignals = signal?.reasoning_key_signals ?? []
  const reasoningUncertaintySources = signal?.reasoning_uncertainty_sources ?? []
  const reasoningContractGaps = signal?.reasoning_contract_gaps ?? []
  const reasoningSectionDisclaimers = signal?.reasoning_section_disclaimers ?? {}
  const reasoningDeltaItems = [
    reasoningDelta.wedge_changed ? 'Wedge changed since prior run' : '',
    reasoningDelta.confidence_changed ? 'Confidence posture changed' : '',
    reasoningDelta.top_destination_changed
      ? `Top destination changed to ${asString(reasoningDelta.current_top_destination) || 'a new competitor'}`
      : '',
    ...toStringArray(reasoningDelta.new_timing_windows).slice(0, 2).map(
      (item) => `New timing window: ${humanizeToken(item)}`,
    ),
    ...toStringArray(reasoningDelta.new_account_signals).slice(0, 2).map(
      (item) => `New account signal: ${item}`,
    ),
  ].filter(Boolean)
  const reasoningVisible = Boolean(
    signal?.reasoning_executive_summary
      || Object.keys(reasoningScope).length
      || theses.length
      || timingWindows.length
      || coverageLimits.length
      || counterevidence.length
      || reasoningDeltaItems.length
      || reasoningContractGaps.length,
  )
  const painData = profile.pain_distribution.map((p) => ({
    name: p.pain_category,
    count: p.count,
  }))
  const latestSnapshot = data?.history.snapshots?.[0] ?? null
  const periodDeltas = data?.comparison.deltas ?? {}
  const slowBurnConfig = {
    support_sentiment: {
      label: 'Support Sentiment',
      color: '#22d3ee',
      positiveIsGood: true,
      suffix: '',
      multiplier: 1,
    },
    legacy_support_score: {
      label: 'Legacy Support',
      color: '#a78bfa',
      positiveIsGood: true,
      suffix: '',
      multiplier: 1,
    },
    new_feature_velocity: {
      label: 'Feature Velocity',
      color: '#f59e0b',
      positiveIsGood: false,
      suffix: '',
      multiplier: 1,
    },
    employee_growth_rate: {
      label: 'Employee Growth',
      color: '#34d399',
      positiveIsGood: false,
      suffix: '%',
      multiplier: 100,
    },
  } as const

  function formatDelta(value?: number | null, suffix = '') {
    if (value === undefined || value === null || Number.isNaN(value)) return '--'
    const sign = value > 0 ? '+' : ''
    return `${sign}${value.toFixed(2)}${suffix}`
  }

  const slowBurnMetrics = Object.entries(slowBurnConfig).map(([key, config]) => ({
    key,
    ...config,
    current: latestSnapshot?.[key as keyof typeof latestSnapshot] as number | null | undefined,
    delta: periodDeltas[key],
  }))
  const selectedConfig = slowBurnConfig[selectedSlowBurnMetric]
  const slowBurnHistory = (data?.history.snapshots ?? [])
    .slice(0, 12)
    .reverse()
    .map((snapshot) => ({
      date: snapshot.snapshot_date.slice(5),
      value: snapshot[selectedSlowBurnMetric] == null
        ? null
        : Number(snapshot[selectedSlowBurnMetric]) * selectedConfig.multiplier,
    }))

  const reviewColumns: Column<ReviewSummary>[] = [
    {
      key: 'company',
      header: 'Company / Title',
      render: (r) => (
        <div>
          <span className="text-white">{r.reviewer_company ?? '--'}</span>
          {r.reviewer_title && (
            <span className="block text-[10px] text-slate-500">{r.reviewer_title}</span>
          )}
        </div>
      ),
    },
    {
      key: 'source',
      header: 'Source',
      render: (r) => (
        <span className="text-slate-400 text-xs">{r.source ?? '--'}</span>
      ),
    },
    {
      key: 'urgency',
      header: 'Urgency',
      render: (r) => <UrgencyBadge score={r.urgency_score} />,
      sortable: true,
      sortValue: (r) => r.urgency_score ?? 0,
    },
    {
      key: 'pain',
      header: 'Pain',
      render: (r) => <span className="text-slate-400">{r.pain_category ?? '--'}</span>,
    },
    {
      key: 'sentiment',
      header: 'Sentiment',
      render: (r) => {
        if (!r.sentiment_direction) return <span className="text-slate-500 text-xs">--</span>
        const colors: Record<string, string> = {
          improving: 'text-green-400',
          declining: 'text-red-400',
          stable: 'text-slate-400',
        }
        return (
          <span className={`text-xs font-medium ${colors[r.sentiment_direction] ?? 'text-slate-400'}`}>
            {r.sentiment_direction}
          </span>
        )
      },
    },
    {
      key: 'rating',
      header: 'Rating',
      render: (r) => <span className="text-slate-300">{r.rating?.toFixed(1) ?? '--'}</span>,
      sortable: true,
      sortValue: (r) => r.rating ?? 0,
    },
    {
      key: 'intent',
      header: 'Intent',
      render: (r) =>
        r.intent_to_leave ? (
          <span className="text-red-400 text-xs font-medium">Leaving</span>
        ) : (
          <span className="text-slate-500 text-xs">--</span>
        ),
    },
    {
      key: 'authority',
      header: 'Authority',
      render: (r) => {
        if (!r.decision_maker && !r.role_level && !r.buying_stage) {
          return <span className="text-slate-500 text-xs">--</span>
        }
        return (
          <div className="space-y-0.5">
            {r.decision_maker && (
              <span className="text-cyan-400 text-[10px] font-medium block">DM</span>
            )}
            {r.role_level && (
              <span className="text-slate-400 text-[10px] block">{r.role_level}</span>
            )}
            {r.buying_stage && (
              <span className="text-slate-500 text-[10px] block">{r.buying_stage.replace(/_/g, ' ')}</span>
            )}
          </div>
        )
      },
    },
    {
      key: 'competitors',
      header: 'Competitors',
      render: (r) => {
        const comps = r.competitors_mentioned ?? []
        if (comps.length === 0) return <span className="text-slate-500 text-xs">--</span>
        return (
          <div className="flex flex-wrap gap-1">
            {comps.slice(0, 3).map((c, i) => {
              const name = typeof c === 'string' ? c : (c as Record<string, unknown>).name as string ?? ''
              return (
                <span key={i} className="px-1.5 py-0.5 bg-slate-800 rounded text-[10px] text-slate-300">
                  {name}
                </span>
              )
            })}
            {comps.length > 3 && (
              <span className="text-[10px] text-slate-500">+{comps.length - 3}</span>
            )}
          </div>
        )
      },
    },
  ]

  const companyColumns: Column<{ company: string; urgency: number; pain: string | null }>[] = [
    {
      key: 'company',
      header: 'Company',
      render: (r) => <span className="text-white font-medium">{r.company}</span>,
    },
    {
      key: 'urgency',
      header: 'Urgency',
      render: (r) => <UrgencyBadge score={r.urgency} />,
      sortable: true,
      sortValue: (r) => r.urgency,
    },
    {
      key: 'pain',
      header: 'Pain Category',
      render: (r) => <span className="text-slate-400">{r.pain ?? '--'}</span>,
    },
  ]

  return (
    <div className="space-y-6">
      <button
        onClick={() => navigate(backTo)}
        className="flex items-center gap-1 text-sm text-slate-400 hover:text-white transition-colors"
      >
        <ArrowLeft className="h-4 w-4" />
        {backLabel}
      </button>

      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-white">{profile.vendor_name}</h1>
            {signal?.archetype && (
              <ArchetypeBadge
                archetype={signal.archetype}
                confidence={signal.archetype_confidence}
                showConfidence
                size="md"
              />
            )}
          </div>
          <p className="text-sm text-slate-400 mt-1">
            {profile.review_counts.total} reviews ({profile.review_counts.enriched} enriched)
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="hidden sm:flex items-center gap-2">
            {watchlistsReturnPath ? (
              <button
                onClick={() => navigate(watchlistsReturnPath)}
                className="inline-flex items-center gap-2 rounded-lg border border-violet-500/30 bg-violet-500/10 px-3 py-2 text-sm text-violet-200 hover:border-violet-400/50 hover:text-white transition-colors"
              >
                {watchlistsReturnLabel}
              </button>
            ) : null}
            <button
              onClick={() => navigate(opportunitiesPath)}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-700/60 bg-slate-900/60 px-3 py-2 text-sm text-slate-200 hover:border-cyan-500/40 hover:text-white transition-colors"
            >
              View Opportunities
            </button>
            <button
              onClick={() => navigate(evidenceExplorerPath)}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-700/60 bg-slate-900/60 px-3 py-2 text-sm text-slate-200 hover:border-cyan-500/40 hover:text-white transition-colors"
            >
              Validate Evidence
            </button>
            <button
              onClick={() => navigate(reportsPath)}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-700/60 bg-slate-900/60 px-3 py-2 text-sm text-slate-200 hover:border-cyan-500/40 hover:text-white transition-colors"
            >
              View Reports
            </button>
          </div>
          {signal && (
            <div className="text-right">
              <p className="text-xs text-slate-400">Urgency Score</p>
              <p className="text-3xl font-bold text-white">{signal.avg_urgency_score.toFixed(1)}</p>
            </div>
          )}
          <button
            onClick={refresh}
            disabled={refreshing}
            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
          </button>
        </div>
      </div>

      <div className="flex sm:hidden flex-wrap gap-2">
        {watchlistsReturnPath ? (
          <button
            onClick={() => navigate(watchlistsReturnPath)}
            className="inline-flex items-center gap-2 rounded-lg border border-violet-500/30 bg-violet-500/10 px-3 py-2 text-sm text-violet-200 hover:border-violet-400/50 hover:text-white transition-colors"
          >
            {watchlistsReturnLabel}
          </button>
        ) : null}
        <button
          onClick={() => navigate(opportunitiesPath)}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-700/60 bg-slate-900/60 px-3 py-2 text-sm text-slate-200 hover:border-cyan-500/40 hover:text-white transition-colors"
        >
          View Opportunities
        </button>
        <button
          onClick={() => navigate(evidenceExplorerPath)}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-700/60 bg-slate-900/60 px-3 py-2 text-sm text-slate-200 hover:border-cyan-500/40 hover:text-white transition-colors"
        >
          Validate Evidence
        </button>
        <button
          onClick={() => navigate(reportsPath)}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-700/60 bg-slate-900/60 px-3 py-2 text-sm text-slate-200 hover:border-cyan-500/40 hover:text-white transition-colors"
        >
          View Reports
        </button>
      </div>

      <div className="flex gap-1 border-b border-slate-700/50">
        {(['overview', 'reviews', 'companies'] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm capitalize transition-colors ${
              tab === t
                ? 'text-cyan-400 border-b-2 border-cyan-400'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {data?.loadWarnings?.length ? (
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3">
          <p className="text-sm font-medium text-amber-200">Some vendor data is temporarily unavailable.</p>
          <ul className="mt-2 space-y-1 text-xs text-amber-100/90">
            {data.loadWarnings.map((warning, index) => (
              <li key={`${warning}-${index}`}>{warning}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {tab === 'overview' && signal && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
                <h3 className="text-sm font-medium text-slate-300 mb-3">Key Metrics</h3>
                <dl className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <dt className="text-slate-400">NPS Proxy</dt>
                    <dd className="text-white">{signal.nps_proxy?.toFixed(1) ?? '--'}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Price Complaint Rate</dt>
                    <dd className="text-white">
                      {signal.price_complaint_rate !== null
                        ? `${(signal.price_complaint_rate * 100).toFixed(0)}%`
                        : '--'}
                    </dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-slate-400">DM Churn Rate</dt>
                    <dd className="text-white">
                      {signal.decision_maker_churn_rate !== null
                        ? `${(signal.decision_maker_churn_rate * 100).toFixed(0)}%`
                        : '--'}
                    </dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Churn Intent Count</dt>
                    <dd className="text-white">{signal.churn_intent_count}</dd>
                  </div>
                </dl>
              </div>
              {reasoningVisible && (
                <div className="bg-slate-900/50 border border-cyan-500/20 rounded-xl p-5">
                  <h3 className="text-sm font-medium text-cyan-300 mb-3">Reasoning Intelligence</h3>
                  <dl className="space-y-2 text-sm">
                    {signal.archetype && (
                      <div className="flex justify-between items-center">
                        <dt className="text-slate-400">Churn Pattern</dt>
                        <dd><ArchetypeBadge archetype={signal.archetype} confidence={signal.archetype_confidence} showConfidence /></dd>
                      </div>
                    )}
                    {signal.synthesis_wedge_label && (
                      <div className="flex justify-between">
                        <dt className="text-slate-400">Primary Wedge</dt>
                        <dd className="text-slate-200 text-xs">{signal.synthesis_wedge_label}</dd>
                      </div>
                    )}
                    <div className="flex justify-between">
                      <dt className="text-slate-400">Reasoning Mode</dt>
                      <dd className="text-slate-300 text-xs">{signal.reasoning_mode ?? '--'}</dd>
                    </div>
                    {signal.reasoning_source && (
                      <div className="flex justify-between">
                        <dt className="text-slate-400">Reasoning Source</dt>
                        <dd className="text-slate-300 text-xs">{humanizeToken(signal.reasoning_source.replace(/^b2b_/, ''))}</dd>
                      </div>
                    )}
                  </dl>
                  {signal.reasoning_executive_summary && (
                    <p className="mt-3 pt-3 border-t border-slate-700/50 text-sm text-slate-300">
                      {signal.reasoning_executive_summary}
                    </p>
                  )}
                  {reasoningKeySignals.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-700/50">
                      <p className="text-xs text-slate-500 mb-1.5">Key signals</p>
                      <div className="flex flex-wrap gap-1.5">
                        {reasoningKeySignals.slice(0, 5).map((item, i) => (
                          <span key={i} className="px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-300">
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {signal.falsification_conditions && signal.falsification_conditions.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-700/50">
                      <p className="text-xs text-slate-500 mb-1.5">What would change this conclusion:</p>
                      <ul className="space-y-1">
                        {signal.falsification_conditions.map((c: string, i: number) => (
                          <li key={i} className="text-xs text-slate-400 pl-3 relative before:content-['\2022'] before:absolute before:left-0 before:text-cyan-500">{c}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
              {Object.keys(reasoningScope).length > 0 && (
                <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
                  <div className="flex items-start justify-between gap-4 mb-3">
                    <div>
                      <h3 className="text-sm font-medium text-slate-300">Evidence Scope</h3>
                      <p className="text-xs text-slate-500 mt-1">
                        {asString(reasoningScope.selection_strategy) || 'Scoped witness packet'}
                      </p>
                    </div>
                    <div className="text-right text-xs text-slate-400">
                      <p>{asNumber(reasoningScope.witnesses_in_scope) ?? 0} witnesses</p>
                      <p>{asNumber(reasoningScope.reviews_in_scope) ?? 0} reviews in scope</p>
                    </div>
                  </div>
                  <dl className="grid grid-cols-2 gap-3 text-sm">
                    <div className="rounded-lg bg-slate-800/60 p-3">
                      <dt className="text-xs text-slate-500">Reviews Considered</dt>
                      <dd className="mt-1 text-white">{asNumber(reasoningScope.reviews_considered_total) ?? '--'}</dd>
                    </div>
                    <div className="rounded-lg bg-slate-800/60 p-3">
                      <dt className="text-xs text-slate-500">Dropped Evidence</dt>
                      <dd className="mt-1 text-white">{asNumber(reasoningScope.dropped_evidence_count) ?? 0}</dd>
                    </div>
                  </dl>
                  {witnessMix.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-700/50">
                      <p className="text-xs text-slate-500 mb-1.5">Witness mix</p>
                      <div className="flex flex-wrap gap-1.5">
                        {witnessMix.map((item) => (
                          <span key={item.label} className="px-2 py-0.5 bg-cyan-900/20 border border-cyan-800/30 rounded text-xs text-cyan-200">
                            {item.label} ({item.count})
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {timeCoverage.length > 0 && (
                    <div className="mt-3">
                      <p className="text-xs text-slate-500 mb-1.5">Time coverage</p>
                      <div className="flex flex-wrap gap-1.5">
                        {timeCoverage.map((item) => (
                          <span key={item.bucket} className="px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-300">
                            {item.bucket} ({item.count})
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {toStringArray(reasoningScope.reasons_dropped).length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-700/50">
                      <p className="text-xs text-slate-500 mb-1.5">Coverage limits</p>
                      <ul className="space-y-1">
                        {toStringArray(reasoningScope.reasons_dropped).slice(0, 4).map((item, i) => (
                          <li key={i} className="text-xs text-slate-400">
                            {humanizeToken(item)}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
              <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
                <h3 className="text-sm font-medium text-slate-300 mb-3">Slow-Burn Signals</h3>
                <div className="space-y-3">
                  {slowBurnMetrics.map((metric) => {
                    const multiplier = metric.multiplier ?? 1
                    const current = metric.current == null ? null : metric.current * multiplier
                    const delta = metric.delta == null ? null : metric.delta * multiplier
                    const improving = delta == null
                      ? null
                      : metric.positiveIsGood ? delta > 0 : delta < 0
                    return (
                      <div key={metric.label} className="flex items-center justify-between gap-4">
                        <div>
                          <p className="text-sm text-slate-300">{metric.label}</p>
                          <p className="text-[11px] text-slate-500">Current / 30d change</p>
                        </div>
                        <div className="text-right">
                          <p className="text-sm text-white">
                            {current == null ? '--' : `${current.toFixed(2)}${metric.suffix ?? ''}`}
                          </p>
                          <p className={clsx(
                            'text-xs',
                            improving === null
                              ? 'text-slate-500'
                              : improving
                                ? 'text-green-400'
                                : 'text-red-400',
                          )}>
                            {formatDelta(delta, metric.suffix ?? '')}
                          </p>
                        </div>
                      </div>
                    )
                  })}
                </div>
                <div className="mt-4 pt-4 border-t border-slate-700/50">
                  <div className="flex flex-wrap gap-1.5 mb-3">
                    {Object.entries(slowBurnConfig).map(([key, config]) => (
                      <button
                        key={key}
                        onClick={() => setSelectedSlowBurnMetric(key as typeof selectedSlowBurnMetric)}
                        className={clsx(
                          'px-2 py-1 rounded text-xs transition-colors',
                          selectedSlowBurnMetric === key
                            ? 'bg-slate-700 text-white'
                            : 'bg-slate-800 text-slate-400 hover:text-white',
                        )}
                      >
                        {config.label}
                      </button>
                    ))}
                  </div>
                  {slowBurnHistory.some((point) => point.value !== null) ? (
                    <ResponsiveContainer width="100%" height={180}>
                      <LineChart data={slowBurnHistory}>
                        <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                        <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                        <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#1e293b',
                            border: '1px solid #334155',
                            borderRadius: 8,
                            color: '#e2e8f0',
                            fontSize: 13,
                          }}
                        />
                        <Line
                          type="monotone"
                          dataKey="value"
                          stroke={selectedConfig.color}
                          strokeWidth={2}
                          dot={{ r: 2 }}
                          connectNulls={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <p className="text-xs text-slate-500">Not enough snapshot history yet for a trend line.</p>
                  )}
                </div>
                {latestSnapshot?.snapshot_date && (
                  <p className="mt-3 pt-3 border-t border-slate-700/50 text-[11px] text-slate-500">
                    Latest snapshot: {latestSnapshot.snapshot_date}
                  </p>
                )}
              </div>
              {signal.top_competitors && signal.top_competitors.length > 0 && (
                <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
                  <h3 className="text-sm font-medium text-slate-300 mb-3">Top Competitors</h3>
                  <ul className="space-y-1.5">
                    {signal.top_competitors.map((c, i) => {
                      if (typeof c === 'string') return <li key={i} className="text-sm text-slate-300">{c}</li>
                      const obj = c as Record<string, unknown>
                      const name = extractNestedLabel(obj.name ?? obj.competitor)
                      const mentions = Number(obj.mentions ?? obj.count ?? 0)
                      return (
                        <li key={i} className="flex items-center justify-between text-sm">
                          <span className="text-slate-300">{name || `#${i + 1}`}</span>
                          {mentions > 0 && <span className="text-xs text-slate-500">{mentions} mentions</span>}
                        </li>
                      )
                    })}
                  </ul>
                </div>
              )}
              {signal.top_feature_gaps && signal.top_feature_gaps.length > 0 && (
                <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
                  <h3 className="text-sm font-medium text-slate-300 mb-3">Feature Gaps</h3>
                  <ul className="space-y-1.5">
                    {signal.top_feature_gaps.map((g, i) => {
                      if (typeof g === 'string') return <li key={i} className="text-sm text-slate-300">{g}</li>
                      const obj = g as Record<string, unknown>
                      const label = extractNestedLabel(obj.feature ?? obj.name ?? obj.gap ?? obj.area)
                      const count = Number(obj.count ?? obj.mentions ?? 0)
                      return (
                        <li key={i} className="flex items-center justify-between text-sm">
                          <span className="text-slate-300">{label || `#${i + 1}`}</span>
                          {count > 0 && <span className="text-xs text-slate-500">({count})</span>}
                        </li>
                      )
                    })}
                  </ul>
                </div>
              )}
            </div>
            <div className="space-y-4">
              {painData.length > 0 && (
                <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
                  <h3 className="text-sm font-medium text-slate-300 mb-3">Pain Distribution</h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={painData} layout="vertical" margin={{ left: 80 }}>
                      <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#334155' }} />
                      <YAxis
                        type="category"
                        dataKey="name"
                        tick={{ fill: '#94a3b8', fontSize: 11 }}
                        axisLine={{ stroke: '#334155' }}
                        width={80}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1e293b',
                          border: '1px solid #334155',
                          borderRadius: 8,
                          color: '#e2e8f0',
                          fontSize: 13,
                        }}
                      />
                      <Bar dataKey="count" fill="#22d3ee" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
              {signal.quotable_evidence && signal.quotable_evidence.length > 0 && (
                <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
                  <h3 className="text-sm font-medium text-slate-300 mb-3">Quotable Evidence</h3>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {signal.quotable_evidence.map((q: string, i: number) => (
                      <blockquote
                        key={i}
                        className="text-sm text-slate-300 italic border-l-2 border-cyan-500/50 pl-3"
                      >
                        {extractQuoteText(q)}
                      </blockquote>
                    ))}
                  </div>
                </div>
              )}
              {recentReportsCard}
              {(theses.length > 0 || timingWindows.length > 0 || reasoningDeltaItems.length > 0 || coverageLimits.length > 0 || counterevidence.length > 0 || reasoningContractGaps.length > 0) && (
                <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
                  <h3 className="text-sm font-medium text-slate-300 mb-3">Reasoning Highlights</h3>
                  {theses.length > 0 && (
                    <div className="space-y-3">
                      {theses.map((item, index) => (
                        <div key={`${asString(item.thesis_id) || 'thesis'}-${index}`} className="rounded-lg bg-slate-800/50 p-3">
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <p className="text-sm text-white">{asString(item.summary) || 'Unnamed thesis'}</p>
                              {asString(item.why_now) && (
                                <p className="mt-1 text-xs text-slate-400">{asString(item.why_now)}</p>
                              )}
                            </div>
                            {asString(item.confidence) && (
                              <span className="px-2 py-0.5 bg-slate-700 rounded text-[10px] uppercase tracking-wide text-slate-200">
                                {asString(item.confidence)}
                              </span>
                            )}
                          </div>
                          <div className="mt-2 flex flex-wrap gap-1.5 text-[11px] text-slate-400">
                            {asString(item.wedge) && (
                              <span className="px-2 py-0.5 bg-slate-900 rounded">{humanizeToken(asString(item.wedge))}</span>
                            )}
                            {asNumber(item.evidence_count) !== null && (
                              <span className="px-2 py-0.5 bg-slate-900 rounded">{asNumber(item.evidence_count)} evidence refs</span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  {timingWindows.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-slate-700/50">
                      <p className="text-xs text-slate-500 mb-2">Timing windows</p>
                      <div className="space-y-2">
                        {timingWindows.map((item, index) => (
                          <div key={`${asString(item.window_id) || 'window'}-${index}`} className="flex items-start justify-between gap-3 text-sm">
                            <div>
                              <p className="text-slate-200">{asString(item.start_or_anchor) || asString(item.window_type)}</p>
                              {asString(item.recommended_action) && (
                                <p className="text-xs text-slate-500 mt-0.5">{asString(item.recommended_action)}</p>
                              )}
                            </div>
                            {asString(item.urgency) && (
                              <span className="px-2 py-0.5 bg-amber-900/30 border border-amber-800/30 rounded text-[10px] uppercase tracking-wide text-amber-200">
                                {asString(item.urgency)}
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {reasoningDeltaItems.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-slate-700/50">
                      <p className="text-xs text-slate-500 mb-2">Since last reasoning run</p>
                      <ul className="space-y-1">
                        {reasoningDeltaItems.slice(0, 5).map((item, i) => (
                          <li key={i} className="text-xs text-slate-300">{item}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {(coverageLimits.length > 0 || counterevidence.length > 0 || reasoningContractGaps.length > 0 || Object.keys(reasoningSectionDisclaimers).length > 0 || reasoningUncertaintySources.length > 0) && (
                    <div className="mt-4 pt-4 border-t border-slate-700/50">
                      <p className="text-xs text-slate-500 mb-2">Trust and coverage</p>
                      <div className="space-y-2">
                        {coverageLimits.slice(0, 3).map((item, index) => (
                          <div key={`${asString(item.coverage_limit_id) || 'limit'}-${index}`} className="text-xs text-slate-400">
                            {asString(item.label)}
                          </div>
                        ))}
                        {counterevidence.slice(0, 2).map((item, index) => (
                          <div key={`${asString(item.counterevidence_id) || 'counter'}-${index}`} className="text-xs text-slate-400">
                            Counterevidence: {asString(item.statement)}
                          </div>
                        ))}
                        {reasoningContractGaps.slice(0, 3).map((item, index) => (
                          <div key={`${item}-${index}`} className="text-xs text-slate-400">
                            Contract gap: {humanizeToken(item)}
                          </div>
                        ))}
                        {Object.entries(reasoningSectionDisclaimers).slice(0, 2).map(([section, message]) => (
                          <div key={section} className="text-xs text-slate-400">
                            {humanizeToken(section)}: {message}
                          </div>
                        ))}
                        {reasoningUncertaintySources.slice(0, 2).map((item, index) => (
                          <div key={`${item}-${index}`} className="text-xs text-slate-400">
                            Uncertainty: {item}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Expanded enrichment panels */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Sentiment Distribution */}
            {signal.sentiment_distribution && Object.keys(signal.sentiment_distribution).length > 0 && (
              <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
                <h3 className="text-sm font-medium text-slate-300 mb-3">Sentiment Trajectory</h3>
                <div className="space-y-2">
                  {Object.entries(signal.sentiment_distribution)
                    .sort(([, a], [, b]) => (b as number) - (a as number))
                    .map(([direction, count]) => {
                      const colors: Record<string, string> = {
                        declining: 'bg-red-500/80',
                        consistently_negative: 'bg-amber-500/80',
                        improving: 'bg-green-500/80',
                        stable_positive: 'bg-cyan-500/80',
                        unknown: 'bg-slate-500/80',
                      }
                      const total = Object.values(signal.sentiment_distribution!).reduce((a, b) => a + b, 0)
                      const pct = total > 0 ? Math.round(((count as number) / total) * 100) : 0
                      return (
                        <div key={direction}>
                          <div className="flex justify-between text-xs mb-1">
                            <span className="text-slate-400">{direction.replace(/_/g, ' ')}</span>
                            <span className="text-slate-300">{count as number} ({pct}%)</span>
                          </div>
                          <div className="h-1.5 bg-slate-700/50 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${colors[direction] ?? 'bg-slate-500/80'}`}
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                        </div>
                      )
                    })}
                </div>
              </div>
            )}

            {/* Budget Signals */}
            {signal.budget_signal_summary && Object.keys(signal.budget_signal_summary).length > 0 && (
              <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
                <h3 className="text-sm font-medium text-slate-300 mb-3">Budget Signals</h3>
                <dl className="space-y-2 text-sm">
                  {(signal.budget_signal_summary as Record<string, unknown>).avg_seat_count != null && (
                    <div className="flex justify-between">
                      <dt className="text-slate-400">Avg Seat Count</dt>
                      <dd className="text-white">{Math.round(signal.budget_signal_summary.avg_seat_count as number)}</dd>
                    </div>
                  )}
                  {(signal.budget_signal_summary as Record<string, unknown>).median_seat_count != null && (
                    <div className="flex justify-between">
                      <dt className="text-slate-400">Median Seats</dt>
                      <dd className="text-white">{Math.round(signal.budget_signal_summary.median_seat_count as number)}</dd>
                    </div>
                  )}
                  {(signal.budget_signal_summary as Record<string, unknown>).max_seat_count != null && (
                    <div className="flex justify-between">
                      <dt className="text-slate-400">Max Seats</dt>
                      <dd className="text-white">{Math.round(signal.budget_signal_summary.max_seat_count as number)}</dd>
                    </div>
                  )}
                  {(signal.budget_signal_summary as Record<string, unknown>).price_increase_rate != null && (
                    <div className="flex justify-between">
                      <dt className="text-slate-400">Price Increase Rate</dt>
                      <dd className="text-white">{Math.round((signal.budget_signal_summary.price_increase_rate as number) * 100)}%</dd>
                    </div>
                  )}
                </dl>
              </div>
            )}

            {/* Buyer Authority */}
            {signal.buyer_authority_summary && Object.keys(signal.buyer_authority_summary).length > 0 && (
              <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
                <h3 className="text-sm font-medium text-slate-300 mb-3">Buyer Authority</h3>
                {(signal.buyer_authority_summary as Record<string, Record<string, number>>).role_types && (
                  <div className="mb-3">
                    <p className="text-xs text-slate-500 mb-1.5">Role Types</p>
                    <div className="flex flex-wrap gap-1.5">
                      {Object.entries((signal.buyer_authority_summary as Record<string, Record<string, number>>).role_types)
                        .sort(([, a], [, b]) => b - a)
                        .map(([role, count]) => (
                          <span key={role} className="px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-300">
                            {role.replace(/_/g, ' ')} ({count})
                          </span>
                        ))}
                    </div>
                  </div>
                )}
                {(signal.buyer_authority_summary as Record<string, Record<string, number>>).buying_stages && (
                  <div>
                    <p className="text-xs text-slate-500 mb-1.5">Buying Stages</p>
                    <div className="flex flex-wrap gap-1.5">
                      {Object.entries((signal.buyer_authority_summary as Record<string, Record<string, number>>).buying_stages)
                        .sort(([, a], [, b]) => b - a)
                        .map(([stage, count]) => {
                          const hot = stage === 'active_purchase' || stage === 'renewal_decision'
                          return (
                            <span key={stage} className={`px-2 py-0.5 rounded text-xs ${hot ? 'bg-red-900/50 text-red-300' : 'bg-slate-800 text-slate-300'}`}>
                              {stage.replace(/_/g, ' ')} ({count})
                            </span>
                          )
                        })}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Use Cases & Integration Stacks */}
            {((signal.top_use_cases && signal.top_use_cases.length > 0) || (signal.top_integration_stacks && signal.top_integration_stacks.length > 0)) && (
              <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
                <h3 className="text-sm font-medium text-slate-300 mb-3">Use Cases & Integrations</h3>
                {signal.top_use_cases && signal.top_use_cases.length > 0 && (
                  <div className="mb-3">
                    <p className="text-xs text-slate-500 mb-1.5">Top Modules</p>
                    <div className="flex flex-wrap gap-1.5">
                      {signal.top_use_cases.slice(0, 8).map((m, i) => (
                        <span key={i} className="px-2 py-0.5 bg-cyan-900/30 border border-cyan-800/30 rounded text-xs text-cyan-300">
                          {m.module} ({m.mentions})
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                {signal.top_integration_stacks && signal.top_integration_stacks.length > 0 && (
                  <div>
                    <p className="text-xs text-slate-500 mb-1.5">Integration Stack</p>
                    <div className="flex flex-wrap gap-1.5">
                      {signal.top_integration_stacks.slice(0, 8).map((t, i) => (
                        <span key={i} className="px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-300">
                          {t.tool} ({t.mentions})
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Timeline Signals */}
            {signal.timeline_summary && signal.timeline_summary.length > 0 && (
              <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
                <h3 className="text-sm font-medium text-slate-300 mb-3">Timeline Signals</h3>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {signal.timeline_summary.map((t, i) => (
                    <div key={i} className="flex items-center justify-between text-sm py-1 border-b border-slate-700/30 last:border-0">
                      <div>
                        <span className="text-white">{t.company ?? 'Unknown'}</span>
                        {t.contract_end && (
                          <span className="ml-2 text-xs text-amber-400">ends {t.contract_end}</span>
                        )}
                        {t.evaluation_deadline && (
                          <span className="ml-2 text-xs text-cyan-400">eval by {t.evaluation_deadline}</span>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        {t.decision_timeline && t.decision_timeline !== 'unknown' && (
                          <span className={`text-xs px-1.5 py-0.5 rounded ${
                            t.decision_timeline === 'immediate' ? 'bg-red-900/50 text-red-300' :
                            t.decision_timeline === 'within_quarter' ? 'bg-amber-900/50 text-amber-300' :
                            'bg-slate-700 text-slate-400'
                          }`}>
                            {t.decision_timeline.replace(/_/g, ' ')}
                          </span>
                        )}
                        <UrgencyBadge score={t.urgency} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {tab === 'overview' && !signal && recentReportsCard}

      {tab === 'reviews' && (
        <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl overflow-hidden">
          <DataTable
            columns={reviewColumns}
            data={data?.reviews ?? []}
            onRowClick={(r) => navigate(vendorReviewDetailPath(r.id, profile.vendor_name))}
            emptyMessage="No enriched reviews for this vendor"
          />
        </div>
      )}

      {tab === 'companies' && (
        <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl overflow-hidden">
          <DataTable
            columns={companyColumns}
            data={profile.high_intent_companies}
            emptyMessage="No high-intent companies detected"
          />
        </div>
      )}
    </div>
  )
}
