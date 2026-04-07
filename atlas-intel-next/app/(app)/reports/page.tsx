"use client";
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useEffect, useMemo, useState } from 'react'
import {
  AlertCircle,
  ArrowRight,
  Bell,
  ChevronLeft,
  ChevronRight,
  Download,
  FileBarChart,
  FileText,
  Layers3,
  Loader2,
  RefreshCw,
  Search,
  ShieldCheck,
  X,
} from 'lucide-react'
import { clsx } from 'clsx'
import { PageError } from '@/components/ErrorBoundary'
import SubscriptionScheduler from '@/components/report-library/SubscriptionScheduler'
import UpgradeGate from '@/components/UpgradeGate'
import { fetchReports, generateAccountComparisonReport, generateAccountDeepDiveReport, generateVendorComparisonReport } from '@/lib/api/client'
import useApiData from '@/lib/hooks/useApiData'
import { usePlanGate } from '@/lib/hooks/usePlanGate'
import { REPORT_TYPE_COLORS } from '@/lib/reportConstants'
import {
  deriveFreshness,
  exportReportsCsv,
  formatReportTypeLabel,
  qualityStatusLabel,
  reportDisplayTitle,
  summarizeReportTrust,
} from '@/lib/reportLibrary'
import type { Report } from '@/lib/types'

const PAGE_SIZES = [12, 24, 48] as const

const TYPE_OPTIONS = [
  { value: '', label: 'All types' },
  { value: 'weekly_churn_feed', label: 'Weekly Churn Feed' },
  { value: 'vendor_scorecard', label: 'Vendor Scorecard' },
  { value: 'vendor_deep_dive', label: 'Vendor Deep Dive' },
  { value: 'displacement_report', label: 'Displacement Report' },
  { value: 'category_overview', label: 'Category Overview' },
  { value: 'exploratory_overview', label: 'Exploratory Overview' },
  { value: 'vendor_comparison', label: 'Vendor Comparison' },
  { value: 'account_comparison', label: 'Account Comparison' },
  { value: 'account_deep_dive', label: 'Account Deep Dive' },
  { value: 'vendor_retention', label: 'Vendor Retention' },
  { value: 'challenger_intel', label: 'Challenger Intel' },
  { value: 'challenger_brief', label: 'Challenger Brief' },
  { value: 'battle_card', label: 'Battle Card' },
] as const

const QUALITY_FILTER_OPTIONS = [
  { value: '', label: 'All trust states' },
  { value: 'sales_ready', label: 'Evidence-backed' },
  { value: 'needs_review', label: 'Needs review' },
  { value: 'thin_evidence', label: 'Thin evidence' },
  { value: 'deterministic_fallback', label: 'Fallback render' },
] as const

const LIBRARY_GROUPS = [
  {
    id: 'all',
    label: 'All deliverables',
    description: 'Persisted reports, battle cards, and recurring brief artifacts.',
  },
  {
    id: 'battle_cards',
    label: 'Battle cards',
    description: 'Enablement-ready assets for sales, PMM, and agency teams.',
  },
  {
    id: 'executive_reports',
    label: 'Executive reports',
    description: 'Durable briefings, scorecards, and vendor snapshots.',
  },
  {
    id: 'competitive_packs',
    label: 'Competitive packs',
    description: 'Comparisons, displacement analysis, and challenger reads.',
  },
  {
    id: 'account_intel',
    label: 'Account intel',
    description: 'Company-level and timing-aware deliverables.',
  },
] as const

type LibraryGroupId = (typeof LIBRARY_GROUPS)[number]['id']
type GenerationFormId = 'vendorComparison' | 'accountComparison' | 'accountDeepDive'

const EMPTY_GENERATION_ERRORS: Record<GenerationFormId, string> = {
  vendorComparison: '',
  accountComparison: '',
  accountDeepDive: '',
}

function matchesGroup(report: Report, groupId: LibraryGroupId): boolean {
  if (groupId === 'all') return true
  if (groupId === 'battle_cards') return ['battle_card', 'challenger_brief'].includes(report.report_type)
  if (groupId === 'executive_reports') {
    return ['weekly_churn_feed', 'vendor_scorecard', 'vendor_deep_dive', 'vendor_retention', 'category_overview', 'exploratory_overview'].includes(report.report_type)
  }
  if (groupId === 'competitive_packs') {
    return ['vendor_comparison', 'displacement_report', 'challenger_intel', 'challenger_brief'].includes(report.report_type)
  }
  return ['account_comparison', 'account_deep_dive'].includes(report.report_type)
}

function ReportLibraryCard({ report }: { report: Report }) {
  const trust = summarizeReportTrust(report)
  const freshness = deriveFreshness(report)
  const displayTitle = reportDisplayTitle(report)
  const timestamp = freshness.anchor ?? report.report_date ?? report.created_at

  return (
    <Link
      href={`/reports/${report.id}`}
      className="group rounded-3xl border border-slate-700/60 bg-slate-900/55 p-5 transition-colors hover:border-cyan-500/40 hover:bg-slate-900/80"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-wrap gap-2">
          <span
            className={clsx(
              'inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-medium uppercase tracking-[0.16em]',
              REPORT_TYPE_COLORS[report.report_type] ?? 'bg-slate-500/15 text-slate-300',
            )}
          >
            {formatReportTypeLabel(report.report_type)}
          </span>
          <span className={clsx('inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-medium', trust.toneClass)}>
            {trust.label}
          </span>
          <span className={clsx('inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-medium', freshness.badgeClass)}>
            {freshness.label}
          </span>
        </div>
        <ArrowRight className="h-4 w-4 shrink-0 text-slate-500 transition-transform group-hover:translate-x-1 group-hover:text-cyan-300" />
      </div>

      <div className="mt-5">
        <h2 className="text-lg font-semibold text-white">{displayTitle}</h2>
        <p className="mt-2 line-clamp-3 text-sm leading-6 text-slate-400">
          {report.executive_summary ?? 'No executive summary is attached to this artifact yet.'}
        </p>
      </div>

      <div className="mt-5 grid gap-3 rounded-2xl border border-slate-800 bg-slate-950/30 p-4 sm:grid-cols-2">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Freshness</p>
          <p className={clsx('mt-1 text-sm font-medium', freshness.textClass)}>{freshness.detail}</p>
        </div>
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Artifact date</p>
          <p className="mt-1 text-sm font-medium text-white">{timestamp ? new Date(timestamp).toLocaleString() : '--'}</p>
        </div>
      </div>

      {report.category_filter && (
        <p className="mt-4 text-xs uppercase tracking-[0.18em] text-slate-500">Context: {report.category_filter}</p>
      )}

      <p className="mt-3 text-sm text-slate-500">{trust.detail}</p>
    </Link>
  )
}

function LibraryStat({
  label,
  value,
  detail,
  accent,
}: {
  label: string
  value: string
  detail: string
  accent: string
}) {
  return (
    <div className="rounded-2xl border border-slate-800/80 bg-slate-950/30 p-4">
      <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className={`mt-2 text-2xl font-semibold ${accent}`}>{value}</p>
      <p className="mt-2 text-sm leading-6 text-slate-400">{detail}</p>
    </div>
  )
}

export default function Reports() {
  const router = useRouter()
  const { canAccessReports } = usePlanGate()
  const [groupFilter, setGroupFilter] = useState<LibraryGroupId>('all')
  const [typeFilter, setTypeFilter] = useState('')
  const [qualityFilter, setQualityFilter] = useState('')
  const [vendorSearch, setVendorSearch] = useState('')
  const [debouncedVendor, setDebouncedVendor] = useState('')
  const [primaryVendor, setPrimaryVendor] = useState('')
  const [comparisonVendor, setComparisonVendor] = useState('')
  const [primaryCompany, setPrimaryCompany] = useState('')
  const [comparisonCompany, setComparisonCompany] = useState('')
  const [deepDiveCompany, setDeepDiveCompany] = useState('')
  const [creatingComparison, setCreatingComparison] = useState(false)
  const [creatingAccountComparison, setCreatingAccountComparison] = useState(false)
  const [creatingAccountDeepDive, setCreatingAccountDeepDive] = useState(false)
  const [generationErrors, setGenerationErrors] = useState<Record<GenerationFormId, string>>(EMPTY_GENERATION_ERRORS)
  const [page, setPage] = useState(0)
  const [perPage, setPerPage] = useState<number>(PAGE_SIZES[0])

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedVendor(vendorSearch), 300)
    return () => clearTimeout(timer)
  }, [vendorSearch])

  const { data, loading, error, refresh, refreshing } = useApiData(
    () => fetchReports({ report_type: typeFilter || undefined, vendor_filter: debouncedVendor || undefined, limit: 200 }),
    [typeFilter, debouncedVendor],
  )

  const reports = useMemo(() => data?.reports ?? [], [data])

  const filteredReports = useMemo(
    () =>
      reports.filter((report) => {
        if (!matchesGroup(report, groupFilter)) return false
        if (!qualityFilter) return true
        return (report.quality_status || '').toLowerCase() === qualityFilter
      }),
    [groupFilter, qualityFilter, reports],
  )

  useEffect(() => {
    setPage(0)
  }, [filteredReports.length, groupFilter, typeFilter, debouncedVendor, qualityFilter])

  const totalPages = Math.max(1, Math.ceil(filteredReports.length / perPage))
  const safePage = Math.min(page, totalPages - 1)
  const pagedReports = filteredReports.slice(safePage * perPage, (safePage + 1) * perPage)
  const showPagination = filteredReports.length > PAGE_SIZES[0]
  const debouncePending = vendorSearch !== debouncedVendor
  const hasFilters = groupFilter !== 'all' || typeFilter !== '' || qualityFilter !== '' || vendorSearch !== ''

  const libraryCounts = useMemo(
    () =>
      LIBRARY_GROUPS.reduce<Record<LibraryGroupId, number>>(
        (accumulator, group) => ({
          ...accumulator,
          [group.id]: reports.filter((report) => matchesGroup(report, group.id)).length,
        }),
        {
          all: 0,
          battle_cards: 0,
          executive_reports: 0,
          competitive_packs: 0,
          account_intel: 0,
        },
      ),
    [reports],
  )

  const evidenceBackedCount = reports.filter((report) => (report.quality_status || '').toLowerCase() === 'sales_ready').length
  const monitoredCount = reports.filter((report) => deriveFreshness(report).state !== 'stale').length
  const newestReportAnchor = reports
    .map((report) => deriveFreshness(report).anchor)
    .filter((value): value is string => Boolean(value))
    .sort((left, right) => new Date(right).getTime() - new Date(left).getTime())[0] ?? null

  function clearGenerationError(formId: GenerationFormId) {
    setGenerationErrors((current) => (
      current[formId]
        ? { ...current, [formId]: '' }
        : current
    ))
  }

  function setGenerationError(formId: GenerationFormId, message: string) {
    setGenerationErrors((current) => ({ ...current, [formId]: message }))
  }

  async function handleCreateComparison() {
    const primary = primaryVendor.trim()
    const comparison = comparisonVendor.trim()

    if (!primary || !comparison) {
      setGenerationError('vendorComparison', 'Enter both vendors to compare.')
      return
    }
    if (primary.toLowerCase() === comparison.toLowerCase()) {
      setGenerationError('vendorComparison', 'Choose two different vendors for the comparison artifact.')
      return
    }

    clearGenerationError('vendorComparison')
    setCreatingComparison(true)
    try {
      const result = await generateVendorComparisonReport({
        primary_vendor: primary,
        comparison_vendor: comparison,
        persist: true,
      })
      const reportId = typeof result.report_id === 'string' ? result.report_id : ''
      setPrimaryVendor('')
      setComparisonVendor('')
      refresh()
      if (reportId) router.push(`/reports/${reportId}`)
    } catch (err) {
      setGenerationError('vendorComparison', err instanceof Error ? err.message : 'Comparison generation failed.')
    } finally {
      setCreatingComparison(false)
    }
  }

  async function handleCreateAccountComparison() {
    const primary = primaryCompany.trim()
    const comparison = comparisonCompany.trim()

    if (!primary || !comparison) {
      setGenerationError('accountComparison', 'Enter both companies to compare.')
      return
    }
    if (primary.toLowerCase() === comparison.toLowerCase()) {
      setGenerationError('accountComparison', 'Choose two different companies for the comparison artifact.')
      return
    }

    clearGenerationError('accountComparison')
    setCreatingAccountComparison(true)
    try {
      const result = await generateAccountComparisonReport({
        primary_company: primary,
        comparison_company: comparison,
        persist: true,
      })
      const reportId = typeof result.report_id === 'string' ? result.report_id : ''
      setPrimaryCompany('')
      setComparisonCompany('')
      refresh()
      if (reportId) router.push(`/reports/${reportId}`)
    } catch (err) {
      setGenerationError('accountComparison', err instanceof Error ? err.message : 'Account comparison generation failed.')
    } finally {
      setCreatingAccountComparison(false)
    }
  }

  async function handleCreateAccountDeepDive() {
    const company = deepDiveCompany.trim()

    if (!company) {
      setGenerationError('accountDeepDive', 'Enter a company for the account deep dive.')
      return
    }

    clearGenerationError('accountDeepDive')
    setCreatingAccountDeepDive(true)
    try {
      const result = await generateAccountDeepDiveReport({
        company_name: company,
        persist: true,
      })
      const reportId = typeof result.report_id === 'string' ? result.report_id : ''
      setDeepDiveCompany('')
      refresh()
      if (reportId) router.push(`/reports/${reportId}`)
    } catch (err) {
      setGenerationError('accountDeepDive', err instanceof Error ? err.message : 'Account deep dive generation failed.')
    } finally {
      setCreatingAccountDeepDive(false)
    }
  }

  function clearFilters() {
    setGroupFilter('all')
    setTypeFilter('')
    setQualityFilter('')
    setVendorSearch('')
    setDebouncedVendor('')
  }

  if (!canAccessReports) {
    return (
      <UpgradeGate allowed={false} feature="Report Library" requiredPlan="Starter">
        <div />
      </UpgradeGate>
    )
  }

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-8">
      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.45fr)_390px]">
        <div className="relative overflow-hidden rounded-[28px] border border-slate-700/60 bg-[radial-gradient(circle_at_top_left,_rgba(34,211,238,0.18),_transparent_42%),radial-gradient(circle_at_bottom_right,_rgba(59,130,246,0.14),_transparent_38%),rgba(15,23,42,0.92)] p-6 sm:p-8">
          <div className="absolute inset-0 bg-[linear-gradient(135deg,rgba(148,163,184,0.06),transparent_28%,transparent_72%,rgba(34,211,238,0.08))]" />
          <div className="relative">
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-2 rounded-full border border-cyan-500/20 bg-cyan-500/10 px-3 py-1 text-[11px] font-medium uppercase tracking-[0.22em] text-cyan-300">
                <Layers3 className="h-3.5 w-3.5" />
                Customer-facing B2B
              </span>
              <span className="inline-flex items-center gap-2 rounded-full border border-slate-700/70 bg-slate-950/30 px-3 py-1 text-[11px] font-medium uppercase tracking-[0.22em] text-slate-300">
                <ShieldCheck className="h-3.5 w-3.5" />
                Evidence-backed subscriptions
              </span>
            </div>

            <h1 className="mt-5 max-w-3xl text-3xl font-semibold tracking-tight text-white sm:text-4xl">
              Report &amp; Battle Card Library
            </h1>
            <p className="mt-4 max-w-3xl text-base leading-7 text-slate-300">
              Durable deliverables for sales enablement, PMM, executives, and agencies. Every artifact keeps its executive summary, freshness state, trust metadata, and supporting evidence instead of disappearing behind a one-off generation flow.
            </p>

            <div className="mt-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <LibraryStat
                label="Library size"
                value={String(reports.length)}
                detail="Persisted artifacts already packaged for repeat access, export, and brief subscriptions."
                accent="text-white"
              />
              <LibraryStat
                label="Battle assets"
                value={String(libraryCounts.battle_cards)}
                detail="Battle cards and challenger briefs that can become recurring enablement packs."
                accent="text-red-300"
              />
              <LibraryStat
                label="Evidence-backed"
                value={String(evidenceBackedCount)}
                detail="Artifacts currently marked sales ready or evidence-backed in the quality metadata."
                accent="text-emerald-300"
              />
              <LibraryStat
                label="Subscription-ready"
                value={String(monitoredCount)}
                detail="Artifacts still within a useful freshness window for external delivery without a full rebuild."
                accent="text-cyan-300"
              />
            </div>

            <div className="mt-6 flex flex-wrap gap-2">
              {LIBRARY_GROUPS.map((group) => {
                const active = groupFilter === group.id
                return (
                  <button
                    key={group.id}
                    onClick={() => setGroupFilter(group.id)}
                    className={clsx(
                      'rounded-full px-4 py-2 text-sm font-medium transition-colors',
                      active
                        ? 'bg-white text-slate-950'
                        : 'border border-slate-700/70 bg-slate-950/20 text-slate-300 hover:border-cyan-500/40 hover:text-white',
                    )}
                  >
                    {group.label}
                    <span className={clsx('ml-2 text-xs', active ? 'text-slate-600' : 'text-slate-500')}>
                      {libraryCounts[group.id]}
                    </span>
                  </button>
                )
              })}
            </div>

            <p className="mt-4 text-sm text-slate-400">
              {LIBRARY_GROUPS.find((group) => group.id === groupFilter)?.description}
              {newestReportAnchor ? ` Latest artifact: ${new Date(newestReportAnchor).toLocaleString()}.` : ''}
            </p>
          </div>
        </div>

        <SubscriptionScheduler
          scopeType="library"
          scopeKey="library"
          scopeLabel="Recurring brief library"
          description="Save the library-wide cadence, recipients, and freshness rules for recurring stakeholder, agency, and enablement deliveries."
          defaultFocus="all"
        />
      </section>

      <section className="rounded-[28px] border border-slate-700/60 bg-slate-900/55 p-5">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
          <div className="flex-1 space-y-4">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center">
              <div className="relative w-full max-w-sm">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                <input
                  type="text"
                  value={vendorSearch}
                  onChange={(event) => setVendorSearch(event.target.value)}
                  placeholder="Filter by vendor or company..."
                  className="w-full rounded-2xl border border-slate-700/60 bg-slate-950/60 py-2.5 pl-9 pr-3 text-sm text-white outline-none transition-colors placeholder:text-slate-500 focus:border-cyan-500/50"
                />
              </div>

              <select
                value={typeFilter}
                onChange={(event) => setTypeFilter(event.target.value)}
                className="rounded-2xl border border-slate-700/60 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none transition-colors focus:border-cyan-500/50"
              >
                {TYPE_OPTIONS.map((option) => (
                  <option key={option.value || 'all'} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>

              <select
                value={qualityFilter}
                onChange={(event) => setQualityFilter(event.target.value)}
                className="rounded-2xl border border-slate-700/60 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none transition-colors focus:border-cyan-500/50"
              >
                {QUALITY_FILTER_OPTIONS.map((option) => (
                  <option key={option.value || 'all'} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>

              {hasFilters && (
                <button
                  onClick={clearFilters}
                  className="inline-flex items-center gap-2 rounded-2xl border border-slate-700/60 px-3 py-2.5 text-sm text-slate-400 transition-colors hover:border-slate-600 hover:text-white"
                >
                  <X className="h-4 w-4" />
                  Clear
                </button>
              )}
            </div>

            <div className="flex flex-wrap items-center gap-3 text-sm text-slate-400">
              {debouncePending || loading ? (
                <span className="inline-flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading report library...
                </span>
              ) : (
                <span>{filteredReports.length} deliverable{filteredReports.length === 1 ? '' : 's'} visible{reports.length >= 200 ? ' (showing max 200)' : ''}</span>
              )}

              {qualityFilter && (
                <span className="rounded-full bg-slate-800 px-3 py-1 text-xs font-medium text-slate-300">
                  Trust filter: {qualityStatusLabel(qualityFilter)}
                </span>
              )}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => exportReportsCsv(filteredReports)}
              disabled={filteredReports.length === 0}
              className="inline-flex items-center gap-2 rounded-2xl border border-slate-700/60 px-3 py-2.5 text-sm text-slate-300 transition-colors hover:border-cyan-500/40 hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Download className="h-4 w-4" />
              Export visible
            </button>
            <button
              onClick={refresh}
              disabled={refreshing}
              className="inline-flex items-center gap-2 rounded-2xl border border-slate-700/60 px-3 py-2.5 text-sm text-slate-300 transition-colors hover:border-cyan-500/40 hover:text-white disabled:opacity-50"
            >
              <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
              Refresh
            </button>
          </div>
        </div>
      </section>

      {loading ? (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, index) => (
            <div key={index} className="h-72 animate-pulse rounded-3xl border border-slate-700/60 bg-slate-900/40" />
          ))}
        </div>
      ) : filteredReports.length === 0 ? (
        <section className="rounded-[28px] border border-dashed border-slate-700/70 bg-slate-900/30 px-6 py-16 text-center">
          <FileBarChart className="mx-auto h-10 w-10 text-slate-600" />
          <h2 className="mt-4 text-xl font-semibold text-white">No deliverables match the current library filters</h2>
          <p className="mt-3 text-sm leading-6 text-slate-400">
            Clear the current filters or generate a new comparison/deep dive below to seed another persisted artifact.
          </p>
          {hasFilters && (
            <button
              onClick={clearFilters}
              className="mt-6 inline-flex items-center gap-2 rounded-2xl bg-cyan-500/15 px-4 py-2.5 text-sm font-medium text-cyan-300 transition-colors hover:bg-cyan-500/25"
            >
              <X className="h-4 w-4" />
              Reset library filters
            </button>
          )}
        </section>
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {pagedReports.map((report) => (
              <ReportLibraryCard key={report.id} report={report} />
            ))}
          </div>

          {showPagination && (
            <div className="flex flex-col gap-3 rounded-3xl border border-slate-700/60 bg-slate-900/50 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-3 text-sm text-slate-400">
                <span>
                  {safePage * perPage + 1}-{Math.min((safePage + 1) * perPage, filteredReports.length)} of {filteredReports.length}
                </span>
                <select
                  value={perPage}
                  onChange={(event) => {
                    setPerPage(Number(event.target.value))
                    setPage(0)
                  }}
                  className="rounded-xl border border-slate-700/60 bg-slate-950/60 px-2 py-1.5 text-sm text-white outline-none transition-colors focus:border-cyan-500/50"
                >
                  {PAGE_SIZES.map((size) => (
                    <option key={size} value={size}>
                      {size} / page
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPage((current) => Math.max(0, current - 1))}
                  disabled={safePage === 0}
                  className="inline-flex items-center justify-center rounded-xl border border-slate-700/60 p-2 text-slate-400 transition-colors hover:border-cyan-500/40 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <ChevronLeft className="h-4 w-4" />
                </button>
                <span className="min-w-16 text-center text-sm text-slate-400">
                  {safePage + 1} / {totalPages}
                </span>
                <button
                  onClick={() => setPage((current) => Math.min(totalPages - 1, current + 1))}
                  disabled={safePage >= totalPages - 1}
                  className="inline-flex items-center justify-center rounded-xl border border-slate-700/60 p-2 text-slate-400 transition-colors hover:border-cyan-500/40 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>
            </div>
          )}
        </>
      )}

      <section className="rounded-[28px] border border-slate-700/60 bg-slate-900/55 p-5 sm:p-6">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-slate-700/70 bg-slate-950/30 px-3 py-1 text-[11px] font-medium uppercase tracking-[0.22em] text-slate-300">
              <FileText className="h-3.5 w-3.5" />
              Generation utilities
            </div>
            <h2 className="mt-4 text-2xl font-semibold text-white">Generate or refresh persisted deliverables</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
              Artifact generation stays secondary here. These forms seed new entries into the library so downstream subscriptions and exports point at durable deliverables rather than transient sessions.
            </p>
          </div>
          <div className="rounded-2xl border border-slate-700/60 bg-slate-950/30 px-4 py-3 text-sm text-slate-300">
            {newestReportAnchor ? `Latest generated ${new Date(newestReportAnchor).toLocaleString()}` : 'No publish timestamp available'}
          </div>
        </div>

        <div className="mt-6 grid gap-4 xl:grid-cols-3">
          <div className="rounded-3xl border border-slate-800 bg-slate-950/30 p-4">
            <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Vendor comparison</p>
            <div className="mt-4 space-y-3">
              <input
                value={primaryVendor}
                onChange={(event) => {
                  setPrimaryVendor(event.target.value)
                  clearGenerationError('vendorComparison')
                }}
                placeholder="Primary vendor"
                className="w-full rounded-2xl border border-slate-700/60 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none transition-colors placeholder:text-slate-500 focus:border-cyan-500/50"
              />
              <input
                value={comparisonVendor}
                onChange={(event) => {
                  setComparisonVendor(event.target.value)
                  clearGenerationError('vendorComparison')
                }}
                placeholder="Comparison vendor"
                className="w-full rounded-2xl border border-slate-700/60 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none transition-colors placeholder:text-slate-500 focus:border-cyan-500/50"
              />
              <button
                onClick={handleCreateComparison}
                disabled={creatingComparison}
                className="inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-fuchsia-500/15 px-4 py-2.5 text-sm font-medium text-fuchsia-300 transition-colors hover:bg-fuchsia-500/25 disabled:opacity-50"
              >
                {creatingComparison ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowRight className="h-4 w-4" />}
                {creatingComparison ? 'Generating...' : 'Create comparison'}
              </button>
              {generationErrors.vendorComparison && (
                <div className="flex items-start gap-2 rounded-2xl border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>{generationErrors.vendorComparison}</span>
                </div>
              )}
            </div>
          </div>

          <div className="rounded-3xl border border-slate-800 bg-slate-950/30 p-4">
            <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Account deep dive</p>
            <div className="mt-4 space-y-3">
              <input
                value={deepDiveCompany}
                onChange={(event) => {
                  setDeepDiveCompany(event.target.value)
                  clearGenerationError('accountDeepDive')
                }}
                placeholder="Company name"
                className="w-full rounded-2xl border border-slate-700/60 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none transition-colors placeholder:text-slate-500 focus:border-cyan-500/50"
              />
              <p className="text-sm leading-6 text-slate-500">
                Use this when you need a customer-facing deep dive that can be revisited, exported, and dropped into subscriptions later.
              </p>
              <button
                onClick={handleCreateAccountDeepDive}
                disabled={creatingAccountDeepDive}
                className="inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-pink-500/15 px-4 py-2.5 text-sm font-medium text-pink-300 transition-colors hover:bg-pink-500/25 disabled:opacity-50"
              >
                {creatingAccountDeepDive ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowRight className="h-4 w-4" />}
                {creatingAccountDeepDive ? 'Generating...' : 'Create deep dive'}
              </button>
              {generationErrors.accountDeepDive && (
                <div className="flex items-start gap-2 rounded-2xl border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>{generationErrors.accountDeepDive}</span>
                </div>
              )}
            </div>
          </div>

          <div className="rounded-3xl border border-slate-800 bg-slate-950/30 p-4">
            <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Account comparison</p>
            <div className="mt-4 space-y-3">
              <input
                value={primaryCompany}
                onChange={(event) => {
                  setPrimaryCompany(event.target.value)
                  clearGenerationError('accountComparison')
                }}
                placeholder="Primary company"
                className="w-full rounded-2xl border border-slate-700/60 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none transition-colors placeholder:text-slate-500 focus:border-cyan-500/50"
              />
              <input
                value={comparisonCompany}
                onChange={(event) => {
                  setComparisonCompany(event.target.value)
                  clearGenerationError('accountComparison')
                }}
                placeholder="Comparison company"
                className="w-full rounded-2xl border border-slate-700/60 bg-slate-950/60 px-3 py-2.5 text-sm text-white outline-none transition-colors placeholder:text-slate-500 focus:border-cyan-500/50"
              />
              <button
                onClick={handleCreateAccountComparison}
                disabled={creatingAccountComparison}
                className="inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-rose-500/15 px-4 py-2.5 text-sm font-medium text-rose-300 transition-colors hover:bg-rose-500/25 disabled:opacity-50"
              >
                {creatingAccountComparison ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowRight className="h-4 w-4" />}
                {creatingAccountComparison ? 'Generating...' : 'Create comparison'}
              </button>
              {generationErrors.accountComparison && (
                <div className="flex items-start gap-2 rounded-2xl border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>{generationErrors.accountComparison}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      <section className="rounded-[28px] border border-slate-700/60 bg-slate-950/30 p-5 sm:p-6">
        <div className="flex items-start gap-3">
          <Bell className="mt-0.5 h-5 w-5 text-cyan-300" />
          <div>
            <h2 className="text-lg font-semibold text-white">Why this library is defensible</h2>
            <p className="mt-2 text-sm leading-6 text-slate-400">
              The persisted artifacts already exist, read-time cost is low, and each detail page can expose evidence drawers, freshness, and review state. That makes the surface monetizable as recurring subscriptions, enablement packs, and reusable executive brief libraries.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
