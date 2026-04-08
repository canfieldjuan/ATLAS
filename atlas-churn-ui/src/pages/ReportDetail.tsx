import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, RefreshCw, Clock } from 'lucide-react'
import { clsx } from 'clsx'
import { PageError } from '../components/ErrorBoundary'
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

// Shared specialized and structured report rendering is imported for this page.

// ---------------------------------------------------------------------------
// Page component
// ---------------------------------------------------------------------------

function freshnessLabel(dateStr: string | null): { text: string; color: string } {
  if (!dateStr) return { text: 'unknown', color: 'text-slate-500' }
  const hours = (Date.now() - new Date(dateStr).getTime()) / 3600000
  if (hours < 24) return { text: 'Fresh', color: 'text-green-400' }
  if (hours < 72) return { text: `${Math.floor(hours / 24)}d ago`, color: 'text-slate-400' }
  if (hours < 168) return { text: `${Math.floor(hours / 24)}d ago`, color: 'text-amber-400' }
  return { text: `${Math.floor(hours / 24)}d ago`, color: 'text-red-400' }
}

export default function ReportDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const [subModalOpen, setSubModalOpen] = useState(false)
  const [hasSubscription, setHasSubscription] = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [drawerWitnessId, setDrawerWitnessId] = useState<string | null>(null)
  const [drawerVendor, setDrawerVendor] = useState('')

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
  const title = ['vendor_comparison', 'account_comparison'].includes(report.report_type) && report.vendor_filter && report.category_filter
    ? `${report.vendor_filter} vs ${report.category_filter}`
    : report.report_type === 'challenger_brief' && report.vendor_filter && report.category_filter
      ? `${report.vendor_filter} → ${report.category_filter}`
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
          onClick={() => navigate('/reports')}
          className="flex items-center gap-1 text-sm text-slate-400 hover:text-white transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Reports
        </button>
        <div className="flex items-center gap-2">
          <ReportActionBar
            reportId={report.id}
            reportType={report.report_type}
            vendorName={report.vendor_filter ?? null}
            onSubscribe={() => setSubModalOpen(true)}
            hasSubscription={hasSubscription}
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
        <div className="flex items-center gap-3 mt-1">
          <p className="text-sm text-slate-400 break-all">
            {report.report_date ?? report.created_at}
            {report.llm_model && ` | Model: ${report.llm_model}`}
          </p>
          {(() => {
            const f = freshnessLabel(report.report_date ?? report.created_at)
            return (
              <span className={clsx('inline-flex items-center gap-1 text-xs', f.color)}>
                <Clock className="w-3 h-3" />
                {f.text}
              </span>
            )
          })()}
        </div>
      </div>

      {report.executive_summary && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 min-w-0 overflow-hidden">
          <h3 className="text-sm font-medium text-slate-300 mb-2">Executive Summary</h3>
          <p className="text-sm text-slate-300 whitespace-pre-wrap break-words">{report.executive_summary}</p>
        </div>
      )}

      {/* Challenger brief - dedicated renderer */}
      {isSpecializedReportType(report.report_type) && (
        <SpecializedReportData reportType={report.report_type} data={intelIsArray ? rawIntel : intel} />
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
            <StructuredReportData data={intel} />
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
        onClose={() => setSubModalOpen(false)}
        scopeType="report"
        scopeKey={report.id}
        scopeLabel={`${report.report_type.replace(/_/g, ' ')} - ${report.vendor_filter ?? 'all'}`}
        onSaved={() => setHasSubscription(true)}
      />

      {/* Evidence drawer */}
      <EvidenceDrawer
        vendorName={drawerVendor}
        witnessId={drawerWitnessId}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      />
    </div>
  )
}
