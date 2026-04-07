"use client";

import { useMemo, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import {
  ArrowLeft,
  Bell,
  Download,
  FileJson,
  FileText,
  RefreshCw,
  ShieldCheck,
} from 'lucide-react'
import { clsx } from 'clsx'
import ReportEvidencePanel from '@/components/report-library/ReportEvidencePanel'
import SubscriptionScheduler from '@/components/report-library/SubscriptionScheduler'
import { PageError } from '@/components/ErrorBoundary'
import {
  REPORT_SCALAR_KEYS,
  StructuredReportData,
  StructuredReportValue,
} from '@/components/report-renderers/StructuredReportData'
import {
  isSpecializedReportType,
  SpecializedReportData,
} from '@/components/report-renderers/SpecializedReportData'
import { fetchReport } from '@/lib/api/client'
import useApiData from '@/lib/hooks/useApiData'
import { REPORT_TYPE_COLORS } from '@/lib/reportConstants'
import {
  deriveFreshness,
  exportReportDetail,
  formatReportTypeLabel,
  reportDisplayTitle,
  summarizeReportTrust,
} from '@/lib/reportLibrary'
import type { ReportDetail as ReportDetailType } from '@/lib/types'

function formatInlineValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return '--'
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }
  if (Array.isArray(value)) return `${value.length} item${value.length === 1 ? '' : 's'}`
  if (typeof value === 'object') return `${Object.keys(value).length} field${Object.keys(value).length === 1 ? '' : 's'}`
  return '--'
}

function defaultDeliveryFocusForReport(reportType: string): 'battle_cards' | 'executive_reports' | 'comparison_packs' {
  if (reportType === 'battle_card' || reportType === 'challenger_brief') {
    return 'battle_cards'
  }
  if (['vendor_comparison', 'account_comparison', 'account_deep_dive', 'displacement_report', 'challenger_intel'].includes(reportType)) {
    return 'comparison_packs'
  }
  return 'executive_reports'
}

function DetailSkeleton() {
  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
      <div className="space-y-6">
        <div className="h-12 animate-pulse rounded-3xl bg-slate-800/60" />
        <div className="h-48 animate-pulse rounded-[28px] bg-slate-800/60" />
        <div className="h-64 animate-pulse rounded-[28px] bg-slate-800/60" />
      </div>
      <div className="space-y-6">
        <div className="h-48 animate-pulse rounded-[28px] bg-slate-800/60" />
        <div className="h-80 animate-pulse rounded-[28px] bg-slate-800/60" />
      </div>
    </div>
  )
}

function TrustCard({ report }: { report: ReportDetailType }) {
  const freshness = deriveFreshness(report)
  const trust = summarizeReportTrust(report)

  return (
    <section className="rounded-3xl border border-slate-700/60 bg-slate-900/60 p-5">
      <div className="inline-flex items-center gap-2 rounded-full border border-cyan-500/20 bg-cyan-500/10 px-3 py-1 text-[11px] font-medium uppercase tracking-[0.22em] text-cyan-300">
        <ShieldCheck className="h-3.5 w-3.5" />
        Trust & Freshness
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
        <div className="rounded-2xl border border-slate-800 bg-slate-950/30 p-4">
          <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Trust state</p>
          <span className={clsx('mt-3 inline-flex rounded-full px-3 py-1 text-xs font-medium', trust.toneClass)}>
            {trust.label}
          </span>
          <p className="mt-3 text-sm leading-6 text-slate-400">{trust.detail}</p>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-950/30 p-4">
          <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Freshness</p>
          <span className={clsx('mt-3 inline-flex rounded-full px-3 py-1 text-xs font-medium', freshness.badgeClass)}>
            {freshness.label}
          </span>
          <p className={clsx('mt-3 text-sm font-medium', freshness.textClass)}>{freshness.detail}</p>
          {freshness.anchor && (
            <p className="mt-2 text-xs text-slate-500">Latest evidence: {new Date(freshness.anchor).toLocaleString()}</p>
          )}
        </div>
      </div>

      <div className="mt-5 grid gap-3 rounded-2xl border border-slate-800 bg-slate-950/30 p-4">
        <div className="flex items-center justify-between gap-3 text-sm">
          <span className="text-slate-500">Report status</span>
          <span className="text-white">{report.status ?? '--'}</span>
        </div>
        <div className="flex items-center justify-between gap-3 text-sm">
          <span className="text-slate-500">Generated</span>
          <span className="text-right text-white">{report.report_date ?? report.created_at ?? '--'}</span>
        </div>
        <div className="flex items-center justify-between gap-3 text-sm">
          <span className="text-slate-500">Model</span>
          <span className="text-right text-white">{report.llm_model ?? '--'}</span>
        </div>
        <div className="flex items-center justify-between gap-3 text-sm">
          <span className="text-slate-500">Artifact ID</span>
          <span className="text-right font-mono text-xs text-slate-300">{report.id}</span>
        </div>
      </div>
    </section>
  )
}

export default function ReportDetail() {
  const { id } = useParams<{ id: string }>()
  const router = useRouter()
  const [showScheduler, setShowScheduler] = useState(false)

  const { data: report, loading, error, refresh, refreshing } = useApiData<ReportDetailType>(
    () => {
      if (!id) return Promise.reject(new Error('Missing report ID'))
      return fetchReport(id)
    },
    [id],
  )

  const computed = useMemo(() => {
    if (!report) return null

    const badgeColor = REPORT_TYPE_COLORS[report.report_type] ?? 'bg-slate-500/20 text-slate-400'
    const displayTitle = reportDisplayTitle(report)
    const freshness = deriveFreshness(report)
    const trust = summarizeReportTrust(report)
    const rawIntel = report.intelligence_data
    const intelIsArray = Array.isArray(rawIntel)
    const intel = intelIsArray ? {} : (rawIntel ?? {})
    const scalarEntries = Object.entries(intel).filter(([key]) => REPORT_SCALAR_KEYS.has(key))
    const richEntries = Object.entries(intel).filter(([key, value]) => {
      if (REPORT_SCALAR_KEYS.has(key)) return false
      if (key === 'executive_summary') return false
      if (value === null || value === undefined) return false
      if (typeof value === 'string' && value.trim() === '') return false
      if (Array.isArray(value) && value.length === 0) return false
      return true
    })

    return {
      badgeColor,
      displayTitle,
      freshness,
      trust,
      rawIntel,
      intel,
      intelIsArray,
      scalarEntries,
      richEntries,
    }
  }, [report])

  if (error) return <PageError error={error} onRetry={refresh} />
  if (loading) return <DetailSkeleton />
  if (!report || !computed) return <PageError error={new Error('Report not found')} />

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <button
          onClick={() => router.push('/reports')}
          className="inline-flex items-center gap-2 text-sm text-slate-400 transition-colors hover:text-white"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to library
        </button>

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => exportReportDetail(report, 'markdown')}
            className="inline-flex items-center gap-2 rounded-2xl border border-slate-700/60 px-3 py-2 text-sm text-slate-300 transition-colors hover:border-cyan-500/40 hover:text-white"
          >
            <FileText className="h-4 w-4" />
            Export brief
          </button>
          <button
            onClick={() => exportReportDetail(report, 'json')}
            className="inline-flex items-center gap-2 rounded-2xl border border-slate-700/60 px-3 py-2 text-sm text-slate-300 transition-colors hover:border-cyan-500/40 hover:text-white"
          >
            <FileJson className="h-4 w-4" />
            Raw JSON
          </button>
          <button
            onClick={refresh}
            disabled={refreshing}
            className="inline-flex items-center gap-2 rounded-2xl border border-slate-700/60 px-3 py-2 text-sm text-slate-300 transition-colors hover:border-cyan-500/40 hover:text-white disabled:opacity-50"
          >
            <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
            Refresh
          </button>
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
        <div className="space-y-6">
          <section className="rounded-[28px] border border-slate-700/60 bg-[radial-gradient(circle_at_top_left,_rgba(34,211,238,0.14),_transparent_40%),rgba(15,23,42,0.92)] p-6 sm:p-8">
            <div className="flex flex-wrap gap-2">
              <span className={clsx('inline-flex items-center rounded-full px-3 py-1 text-[11px] font-medium uppercase tracking-[0.22em]', computed.badgeColor)}>
                {formatReportTypeLabel(report.report_type)}
              </span>
              <span className={clsx('inline-flex items-center rounded-full px-3 py-1 text-[11px] font-medium', computed.trust.toneClass)}>
                {computed.trust.label}
              </span>
              <span className={clsx('inline-flex items-center rounded-full px-3 py-1 text-[11px] font-medium', computed.freshness.badgeClass)}>
                {computed.freshness.label}
              </span>
            </div>

            <h1 className="mt-5 text-3xl font-semibold tracking-tight text-white">{computed.displayTitle}</h1>
            <p className="mt-3 max-w-3xl text-base leading-7 text-slate-300">
              This persisted deliverable is designed for repeat access, subscriptions, and export. The trust sidebar keeps freshness, operator review, and witness evidence visible while teams revisit the artifact.
            </p>

            <div className="mt-6 grid gap-3 rounded-3xl border border-slate-700/50 bg-slate-950/30 p-4 sm:grid-cols-3">
              <div>
                <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Generated</p>
                <p className="mt-2 text-sm font-medium text-white">{report.report_date ?? report.created_at ?? '--'}</p>
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Freshness</p>
                <p className={clsx('mt-2 text-sm font-medium', computed.freshness.textClass)}>{computed.freshness.detail}</p>
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Model</p>
                <p className="mt-2 text-sm font-medium text-white">{report.llm_model ?? '--'}</p>
              </div>
            </div>
          </section>

          {report.executive_summary && (
            <section className="rounded-[28px] border border-slate-700/60 bg-slate-900/55 p-6">
              <h2 className="text-lg font-semibold text-white">Executive Summary</h2>
              <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-slate-300">{report.executive_summary}</p>
            </section>
          )}

          {isSpecializedReportType(report.report_type) && (
            <SpecializedReportData reportType={report.report_type} data={computed.intelIsArray ? computed.rawIntel : computed.intel} />
          )}

          {!isSpecializedReportType(report.report_type) && (
            <>
              {computed.intelIsArray && Array.isArray(computed.rawIntel) && computed.rawIntel.length > 0 && (
                <section className="rounded-[28px] border border-slate-700/60 bg-slate-900/55 p-6">
                  <h2 className="text-lg font-semibold text-white">
                    {formatReportTypeLabel(report.report_type)} ({computed.rawIntel.length} items)
                  </h2>
                  <div className="mt-4">
                    <StructuredReportValue fieldKey={report.report_type} value={computed.rawIntel} />
                  </div>
                </section>
              )}

              {computed.scalarEntries.length > 0 && (
                <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  {computed.scalarEntries.map(([key, value]) => (
                    <article key={key} className="rounded-3xl border border-slate-700/60 bg-slate-900/55 p-4 text-center">
                      <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">{key.replace(/_/g, ' ')}</p>
                      <p className="mt-3 text-2xl font-semibold text-white">{String(value)}</p>
                    </article>
                  ))}
                </section>
              )}

              {computed.richEntries.length > 0 && (
                <StructuredReportData data={computed.intel} />
              )}
            </>
          )}

          {report.data_density && Object.keys(report.data_density).length > 0 && (
            <section className="rounded-[28px] border border-slate-700/60 bg-slate-900/55 p-6">
              <h2 className="text-lg font-semibold text-white">Data Density</h2>
              <dl className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
                {Object.entries(report.data_density).map(([key, value]) => (
                  <div key={key} className="grid grid-cols-[minmax(0,1fr)_auto] gap-3 rounded-2xl border border-slate-800 bg-slate-950/30 px-4 py-3">
                    <dt className="text-slate-400">{key.replace(/_/g, ' ')}</dt>
                    <dd className="text-right text-white">{formatInlineValue(value)}</dd>
                  </div>
                ))}
              </dl>
            </section>
          )}
        </div>

        <div className="space-y-6 xl:sticky xl:top-6 xl:self-start">
          <section className="rounded-3xl border border-slate-700/60 bg-slate-900/60 p-5">
            <h2 className="text-lg font-semibold text-white">Actions</h2>
            <p className="mt-2 text-sm leading-6 text-slate-400">
              Export this artifact, refresh the detail view, or manage recurring delivery access for this deliverable.
            </p>
            <div className="mt-5 space-y-2">
              <button
                onClick={() => exportReportDetail(report, 'markdown')}
                className="inline-flex w-full items-center justify-between rounded-2xl border border-slate-700/60 px-4 py-3 text-sm text-slate-300 transition-colors hover:border-cyan-500/40 hover:text-white"
              >
                <span className="inline-flex items-center gap-2">
                  <Download className="h-4 w-4" />
                  Export brief
                </span>
                <span className="text-xs text-slate-500">.md</span>
              </button>
              <button
                onClick={() => exportReportDetail(report, 'json')}
                className="inline-flex w-full items-center justify-between rounded-2xl border border-slate-700/60 px-4 py-3 text-sm text-slate-300 transition-colors hover:border-cyan-500/40 hover:text-white"
              >
                <span className="inline-flex items-center gap-2">
                  <FileJson className="h-4 w-4" />
                  Export raw artifact
                </span>
                <span className="text-xs text-slate-500">.json</span>
              </button>
              <button
                onClick={() => setShowScheduler((current) => !current)}
                className="inline-flex w-full items-center justify-between rounded-2xl border border-slate-700/60 px-4 py-3 text-sm text-slate-300 transition-colors hover:border-cyan-500/40 hover:text-white"
              >
                <span className="inline-flex items-center gap-2">
                  <Bell className="h-4 w-4" />
                  {showScheduler ? 'Hide recurring delivery panel' : 'Manage recurring delivery'}
                </span>
              </button>
            </div>
          </section>

          <TrustCard report={report} />

          {showScheduler && (
            <SubscriptionScheduler
              scopeType="report"
              scopeKey={report.id}
              scopeLabel={`Recurring delivery for ${computed.displayTitle}`}
              description="Save the recurring cadence, recipients, and freshness policy for this report without relying on browser-only preferences."
              defaultFocus={defaultDeliveryFocusForReport(report.report_type)}
            />
          )}

          <ReportEvidencePanel report={report} />
        </div>
      </div>
    </div>
  )
}
