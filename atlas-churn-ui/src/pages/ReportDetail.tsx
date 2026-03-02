import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, RefreshCw } from 'lucide-react'
import { clsx } from 'clsx'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import { fetchReport } from '../api/client'
import { REPORT_TYPE_COLORS } from './Reports'
import type { ReportDetail as ReportDetailType } from '../types'

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

// ---------------------------------------------------------------------------
// Structured renderers for intelligence_data fields
// ---------------------------------------------------------------------------

/** [{category: str, count: n}, ...] or [{name/feature/role/competitor: str, count: n}] */
function isRankedList(val: unknown): val is Record<string, unknown>[] {
  if (!Array.isArray(val) || val.length === 0) return false
  const first = val[0]
  return typeof first === 'object' && first !== null && 'count' in first
}

function RankedList({ items }: { items: Record<string, unknown>[] }) {
  const maxCount = Math.max(...items.map((i) => Number(i.count) || 0), 1)
  return (
    <div className="space-y-1.5">
      {items.map((item, idx) => {
        const label = String(
          item.category ?? item.name ?? item.feature ?? item.role ?? item.competitor ?? `#${idx + 1}`,
        )
        const count = Number(item.count) || 0
        const pct = Math.round((count / maxCount) * 100)
        return (
          <div key={idx}>
            <div className="flex items-center justify-between text-xs mb-0.5">
              <span className="text-slate-300">{label}</span>
              <span className="text-slate-400">{count}</span>
            </div>
            <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-cyan-500/60 rounded-full"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}

/** string[] -- render as blockquotes for quotes, pills otherwise */
function StringList({ items, asQuotes }: { items: string[]; asQuotes?: boolean }) {
  if (asQuotes) {
    return (
      <div className="space-y-2 max-h-64 overflow-y-auto">
        {items.map((q, i) => (
          <blockquote
            key={i}
            className="text-sm text-slate-300 italic border-l-2 border-cyan-500/50 pl-3"
          >
            {q}
          </blockquote>
        ))}
      </div>
    )
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((s, i) => (
        <span key={i} className="px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-300">
          {s}
        </span>
      ))}
    </div>
  )
}

/** {key: number} flat object -- render as key-value grid */
function StatObject({ obj }: { obj: Record<string, unknown> }) {
  return (
    <dl className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
      {Object.entries(obj).map(([k, v]) => (
        <div key={k} className="flex justify-between">
          <dt className="text-slate-400">{k.replace(/_/g, ' ')}</dt>
          <dd className="text-white">{String(v)}</dd>
        </div>
      ))}
    </dl>
  )
}

/** Known scalar metadata keys that should render inline, not in a card */
const SCALAR_KEYS = new Set([
  'vendor_name', 'challenger_name', 'report_date', 'window_days',
  'signal_count', 'high_urgency_count', 'medium_urgency_count',
])

const QUOTE_KEYS = new Set(['anonymized_quotes', 'quotable_evidence'])

function IntelValue({ fieldKey, value }: { fieldKey: string; value: unknown }) {
  // String
  if (typeof value === 'string') {
    return <p className="text-sm text-slate-300 whitespace-pre-wrap">{value}</p>
  }

  // Number
  if (typeof value === 'number') {
    return <span className="text-lg font-bold text-white">{value}</span>
  }

  // Boolean
  if (typeof value === 'boolean') {
    return <span className="text-sm text-slate-300">{value ? 'Yes' : 'No'}</span>
  }

  // Ranked list: [{category/name/feature: str, count: n}, ...]
  if (isRankedList(value)) {
    return <RankedList items={value} />
  }

  // String array
  if (Array.isArray(value) && value.length > 0 && typeof value[0] === 'string') {
    return <StringList items={value as string[]} asQuotes={QUOTE_KEYS.has(fieldKey)} />
  }

  // Flat {key: number} object (e.g. by_buying_stage, seat_count_signals)
  if (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    Object.values(value).every((v) => typeof v === 'number' || typeof v === 'string')
  ) {
    return <StatObject obj={value as Record<string, unknown>} />
  }

  // Fallback: formatted JSON
  return (
    <pre className="text-xs text-slate-400 bg-slate-800/50 rounded p-3 overflow-x-auto">
      {JSON.stringify(value, null, 2)}
    </pre>
  )
}

// ---------------------------------------------------------------------------
// Page component
// ---------------------------------------------------------------------------

export default function ReportDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const { data: report, loading, error, refresh, refreshing } = useApiData<ReportDetailType>(
    () => {
      if (!id) return Promise.reject(new Error('Missing report ID'))
      return fetchReport(id)
    },
    [id],
  )

  if (error) return <PageError error={error} onRetry={refresh} />
  if (loading) return <DetailSkeleton />
  if (!report) return <PageError error={new Error('Report not found')} />

  const badgeColor = REPORT_TYPE_COLORS[report.report_type] ?? 'bg-slate-500/20 text-slate-400'

  // Split intelligence_data into scalars (rendered as stat row) and rich fields
  const intel = report.intelligence_data ?? {}
  const scalarEntries = Object.entries(intel).filter(([k]) => SCALAR_KEYS.has(k))
  const richEntries = Object.entries(intel).filter(([k]) => !SCALAR_KEYS.has(k))

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center justify-between">
        <button
          onClick={() => navigate('/reports')}
          className="flex items-center gap-1 text-sm text-slate-400 hover:text-white transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Reports
        </button>
        <button
          onClick={refresh}
          disabled={refreshing}
          className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
        </button>
      </div>

      <div>
        <span className={clsx('inline-flex items-center px-2 py-0.5 rounded text-xs font-medium mb-2', badgeColor)}>
          {report.report_type.replace(/_/g, ' ')}
        </span>
        <h1 className="text-2xl font-bold text-white">
          {report.vendor_filter ?? report.report_type.replace(/_/g, ' ')}
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          {report.report_date ?? report.created_at}
          {report.llm_model && ` | Model: ${report.llm_model}`}
        </p>
      </div>

      {report.executive_summary && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
          <h3 className="text-sm font-medium text-slate-300 mb-2">Executive Summary</h3>
          <p className="text-sm text-slate-300 whitespace-pre-wrap">{report.executive_summary}</p>
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
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {richEntries.map(([key, value]) => (
            <div key={key} className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
              <h4 className="text-xs font-medium text-cyan-400 uppercase tracking-wider mb-3">
                {key.replace(/_/g, ' ')}
              </h4>
              <IntelValue fieldKey={key} value={value} />
            </div>
          ))}
        </div>
      )}

      {report.data_density && Object.keys(report.data_density).length > 0 && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
          <h3 className="text-sm font-medium text-slate-300 mb-3">Data Density</h3>
          <dl className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
            {Object.entries(report.data_density).map(([key, value]) => (
              <div key={key} className="flex justify-between">
                <dt className="text-slate-400">{key.replace(/_/g, ' ')}</dt>
                <dd className="text-white">{String(value)}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}
    </div>
  )
}
