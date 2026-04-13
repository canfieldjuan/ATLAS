import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Building2, AlertTriangle, MessageSquareText, Zap, RefreshCw } from 'lucide-react'
import { clsx } from 'clsx'
import StatCard from '../components/StatCard'
import ChurnChart from '../components/ChurnChart'
import PipelineStatusWidget from '../components/PipelineStatus'
import ArchetypeBadge from '../components/ArchetypeBadge'
import DataTable, { type Column } from '../components/DataTable'
import UrgencyBadge from '../components/UrgencyBadge'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import {
  fetchSignals,
  fetchSlowBurnWatchlist,
  fetchHighIntent,
  fetchPipeline,
} from '../api/client'
import type { ChurnSignal, HighIntentCompany, PipelineStatus } from '../types'

interface DashboardData {
  signals: ChurnSignal[]
  slowBurnSignals: ChurnSignal[]
  companies: HighIntentCompany[]
  pipeline: PipelineStatus
  signalSummary: { totalVendors: number; highUrgency: number; totalReviews: number }
}

function formatSignalValue(value: number | null, suffix = '') {
  if (value === null || Number.isNaN(value)) return '--'
  return `${value.toFixed(1)}${suffix}`
}

function formatGrowthRate(value: number | null) {
  if (value === null || Number.isNaN(value)) return '--'
  return `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`
}

function toIso(value: string | null | undefined): string | null {
  if (!value) return null
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? null : date.toISOString()
}

function maxIso(values: (string | null | undefined)[]): string | null {
  const isoValues = values
    .map((value) => toIso(value))
    .filter((value): value is string => Boolean(value))
  if (isoValues.length === 0) return null
  return isoValues.sort((a, b) => b.localeCompare(a))[0]
}

function vendorDetailPath(vendorName: string, backTo: string) {
  const next = new URLSearchParams()
  next.set('back_to', backTo)
  const qs = next.toString()
  const base = `/vendors/${encodeURIComponent(vendorName)}`
  return qs ? `${base}?${qs}` : base
}

function watchlistsPath(
  vendorName: string,
  backTo: string,
  accountReviewFocus?: NonNullable<HighIntentCompany['account_review_focus']> | null,
) {
  const next = new URLSearchParams()
  if (accountReviewFocus) {
    next.set('account_vendor', accountReviewFocus.vendor)
    next.set('account_company', accountReviewFocus.company)
    next.set('account_report_date', accountReviewFocus.report_date)
    next.set('account_watch_vendor', accountReviewFocus.watch_vendor)
    next.set('account_category', accountReviewFocus.category)
    next.set('account_track_mode', accountReviewFocus.track_mode)
  } else {
    next.set('vendor_name', vendorName)
  }
  next.set('back_to', backTo)
  return `/watchlists?${next.toString()}`
}

function evidencePath(vendorName: string, backTo: string) {
  const next = new URLSearchParams()
  next.set('vendor', vendorName)
  next.set('tab', 'witnesses')
  next.set('back_to', backTo)
  return `/evidence?${next.toString()}`
}

function reportsPath(vendorName: string, backTo: string) {
  const next = new URLSearchParams()
  next.set('vendor_filter', vendorName)
  next.set('back_to', backTo)
  return `/reports?${next.toString()}`
}

function opportunitiesPath(vendorName: string, backTo: string) {
  const next = new URLSearchParams()
  next.set('vendor', vendorName)
  next.set('back_to', backTo)
  return `/opportunities?${next.toString()}`
}

export default function Dashboard() {
  const navigate = useNavigate()
  const dashboardPath = '/dashboard'

  const { data, loading, error, refresh, refreshing } = useApiData<DashboardData>(
    async () => {
      const [sigRes, slowBurnRes, hiRes, pipe] = await Promise.all([
        fetchSignals({ limit: 20 }),
        fetchSlowBurnWatchlist(),
        fetchHighIntent({ limit: 10 }),
        fetchPipeline(),
      ])
      return {
        signals: sigRes.signals,
        slowBurnSignals: slowBurnRes.signals,
        companies: hiRes.companies,
        pipeline: pipe,
        signalSummary: {
          totalVendors: sigRes.total_vendors ?? sigRes.signals.length,
          highUrgency: sigRes.high_urgency_count ?? sigRes.signals.filter((s) => s.avg_urgency_score >= 7).length,
          totalReviews: sigRes.total_signal_reviews ?? 0,
        },
      }
    },
    [],
  )

  const signals = data?.signals ?? []
  const slowBurnSignals = data?.slowBurnSignals ?? []
  const companies = data?.companies ?? []
  const pipeline = data?.pipeline ?? null
  const summary = data?.signalSummary ?? { totalVendors: 0, highUrgency: 0, totalReviews: 0 }
  const latestSignalComputedAt = maxIso(signals.map((signal) => signal.last_computed_at))
  const lastEnrichmentAt = toIso(pipeline?.last_enrichment_at)
  const lastScrapeAt = toIso(pipeline?.last_scrape_at)
  const freshnessAnchor = maxIso([latestSignalComputedAt, lastEnrichmentAt, lastScrapeAt])
  const [now] = useState(() => Date.now())
  const freshnessAgeHours = freshnessAnchor
    ? (now - new Date(freshnessAnchor).getTime()) / (1000 * 60 * 60)
    : null
  const freshnessState = freshnessAgeHours === null
    ? 'unknown'
    : freshnessAgeHours > 24
      ? 'stale'
      : 'fresh'

  const totalReviews = pipeline
    ? Object.values(pipeline.enrichment_counts).reduce((a, b) => a + b, 0)
    : 0
  const enrichRate = pipeline
    ? totalReviews > 0
      ? Math.round(((pipeline.enrichment_counts['enriched'] ?? 0) / totalReviews) * 100)
      : 0
    : 0

  const companyColumns: Column<HighIntentCompany>[] = [
    {
      key: 'company',
      header: 'Company',
      render: (r) => <span className="text-white font-medium">{r.company}</span>,
    },
    {
      key: 'vendor',
      header: 'Vendor',
      render: (r) => <span className="text-slate-300">{r.vendor}</span>,
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
      header: 'Pain',
      render: (r) => <span className="text-slate-400">{r.pain ?? '--'}</span>,
    },
    {
      key: 'seats',
      header: 'Seats',
      render: (r) => <span className="text-slate-300">{r.seat_count ?? '--'}</span>,
      sortable: true,
      sortValue: (r) => r.seat_count ?? 0,
    },
    {
      key: 'buying_stage',
      header: 'Stage',
      render: (r) => {
        if (!r.buying_stage || r.buying_stage === 'unknown') return <span className="text-slate-500 text-xs">--</span>
        const colors: Record<string, string> = {
          active_purchase: 'text-red-400',
          renewal_decision: 'text-amber-400',
          evaluation: 'text-cyan-400',
          post_purchase: 'text-slate-400',
        }
        return (
          <span className={`text-xs font-medium ${colors[r.buying_stage] ?? 'text-slate-400'}`}>
            {r.buying_stage.replace(/_/g, ' ')}
          </span>
        )
      },
    },
    {
      key: 'contract_end',
      header: 'Contract End',
      render: (r) => <span className="text-slate-300 text-xs">{r.contract_end ?? '--'}</span>,
    },
    {
      key: 'dm',
      header: 'DM',
      render: (r) =>
        r.decision_maker ? (
          <span className="text-cyan-400 text-xs font-medium">Yes</span>
        ) : (
          <span className="text-slate-500 text-xs">No</span>
        ),
    },
    {
      key: 'actions',
      header: 'Actions',
      render: (r) => (
        <div className="flex items-center gap-3 text-xs">
          <Link
            to={watchlistsPath(r.vendor, dashboardPath, r.account_review_focus ?? null)}
            onClick={(event) => event.stopPropagation()}
            className="text-violet-300 hover:text-violet-200 transition-colors"
          >
            {r.account_review_focus ? 'Account Review' : 'Watchlists'}
          </Link>
          <Link
            to={vendorDetailPath(r.vendor, dashboardPath)}
            onClick={(event) => event.stopPropagation()}
            className="text-cyan-400 hover:text-cyan-300 transition-colors"
          >
            Vendor
          </Link>
          <Link
            to={evidencePath(r.vendor, dashboardPath)}
            onClick={(event) => event.stopPropagation()}
            className="text-violet-300 hover:text-violet-200 transition-colors"
          >
            Evidence
          </Link>
          <Link
            to={reportsPath(r.vendor, dashboardPath)}
            onClick={(event) => event.stopPropagation()}
            className="text-fuchsia-300 hover:text-fuchsia-200 transition-colors"
          >
            Reports
          </Link>
          <Link
            to={opportunitiesPath(r.vendor, dashboardPath)}
            onClick={(event) => event.stopPropagation()}
            className="text-emerald-300 hover:text-emerald-200 transition-colors"
          >
            Opportunities
          </Link>
        </div>
      ),
    },
  ]

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Churn Signals Overview</h1>
        <div className="flex items-center gap-2">
          <Link
            to="/watchlists"
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-violet-300 hover:text-white hover:bg-violet-500/10 transition-colors"
          >
            Watchlists
          </Link>
          <Link
            to="/vendors"
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-cyan-300 hover:text-white hover:bg-cyan-500/10 transition-colors"
          >
            Vendors
          </Link>
          <Link
            to="/opportunities"
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-emerald-300 hover:text-white hover:bg-emerald-500/10 transition-colors"
          >
            Opportunities
          </Link>
          <button
            onClick={refresh}
            disabled={refreshing}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
            Refresh
          </button>
        </div>
      </div>
      <div className="text-xs">
        {freshnessAnchor ? (
          <span className={freshnessState === 'stale' ? 'text-amber-400' : 'text-slate-500'}>
            Data last updated {new Date(freshnessAnchor).toLocaleString()}
            {freshnessState === 'stale' ? ' (older than 24h)' : ''}
          </span>
        ) : (
          <span className="text-slate-500">Data freshness timestamp unavailable</span>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Vendors Tracked"
          value={summary.totalVendors}
          icon={<Building2 className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="High Urgency"
          value={summary.highUrgency}
          icon={<AlertTriangle className="h-5 w-5" />}
          sub="Urgency &ge; 7"
          skeleton={loading}
        />
        <StatCard
          label="Total Reviews"
          value={totalReviews.toLocaleString()}
          icon={<MessageSquareText className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Enrichment Rate"
          value={`${enrichRate}%`}
          icon={<Zap className="h-5 w-5" />}
          skeleton={loading}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
          <h3 className="text-sm font-medium text-slate-300 mb-4">
            Top Churn Signals by Urgency
          </h3>
          {loading ? (
            <div className="h-[300px] flex items-end gap-3 animate-pulse">
              {[75, 50, 88, 40, 65, 55, 80, 45].map((h, i) => (
                <div
                  key={i}
                  className="flex-1 bg-slate-700/50 rounded-t"
                  style={{ height: `${h}%` }}
                />
              ))}
            </div>
          ) : (
            <ChurnChart signals={signals} />
          )}
        </div>
        <PipelineStatusWidget data={pipeline} />
      </div>

      {(() => {
        const archetypeCounts: Record<string, number> = {}
        for (const s of signals) {
          const arch = s.archetype
          if (arch) archetypeCounts[arch] = (archetypeCounts[arch] ?? 0) + 1
        }
        const sorted = Object.entries(archetypeCounts).sort((a, b) => b[1] - a[1])
        const classified = sorted.reduce((sum, [, count]) => sum + count, 0)
        if (sorted.length === 0) return null
        return (
          <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
            <h3 className="text-sm font-medium text-slate-300 mb-4">
              Churn Pattern Distribution
              <span className="text-xs text-slate-500 ml-2">({classified} of {signals.length} classified)</span>
            </h3>
            <div className="space-y-2.5">
              {sorted.map(([archetype, count]) => {
                const pct = Math.round((count / signals.length) * 100)
                return (
                  <div key={archetype} className="flex items-center gap-3">
                    <div className="w-32 shrink-0">
                      <ArchetypeBadge archetype={archetype} />
                    </div>
                    <div className="flex-1 bg-slate-800 rounded-full h-2 overflow-hidden">
                      <div
                        className="h-full bg-cyan-500/60 rounded-full transition-all"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className="text-xs text-slate-400 w-12 text-right">{count} ({pct}%)</span>
                  </div>
                )
              })}
            </div>
          </div>
        )
      })()}

      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
        <div className="flex items-center justify-between gap-3 mb-4">
          <h3 className="text-sm font-medium text-slate-300">Slow-Burn Watchlist</h3>
          <span className="text-xs text-slate-500">Growth plus support and legacy stress</span>
        </div>
        {loading ? (
          <div className="space-y-3 animate-pulse">
            {Array.from({ length: 4 }, (_, i) => (
              <div key={i} className="h-24 rounded-xl bg-slate-800/50" />
            ))}
          </div>
        ) : slowBurnSignals.length === 0 ? (
          <div className="text-sm text-slate-500">No slow-burn signal data available yet.</div>
        ) : (
          <div className="space-y-3">
            {slowBurnSignals.map((signal) => (
              <button
                key={signal.vendor_name}
                type="button"
                onClick={() => navigate(vendorDetailPath(signal.vendor_name, dashboardPath))}
                className="w-full rounded-xl border border-slate-700/50 bg-slate-950/40 px-4 py-3 text-left transition-colors hover:border-cyan-500/40 hover:bg-slate-900/70"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="space-y-1">
                    <div className="text-sm font-medium text-white">{signal.vendor_name}</div>
                    <div className="text-xs text-slate-500">{signal.product_category ?? 'Uncategorized'}</div>
                  </div>
                  <ArchetypeBadge archetype={signal.archetype} confidence={signal.archetype_confidence} />
                </div>
                <div className="mt-3 grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
                  <div><div className="text-slate-500">Support</div><div className="text-slate-200">{formatSignalValue(signal.support_sentiment)}</div></div>
                  <div><div className="text-slate-500">Legacy</div><div className="text-slate-200">{formatSignalValue(signal.legacy_support_score)}</div></div>
                  <div><div className="text-slate-500">Feature</div><div className="text-slate-200">{formatSignalValue(signal.new_feature_velocity)}</div></div>
                  <div><div className="text-slate-500">Growth</div><div className="text-slate-200">{formatGrowthRate(signal.employee_growth_rate)}</div></div>
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
                  <Link
                    to={evidencePath(signal.vendor_name, dashboardPath)}
                    onClick={(event) => event.stopPropagation()}
                    className="text-violet-300 hover:text-violet-200 transition-colors"
                  >
                    Evidence
                  </Link>
                  <Link
                    to={reportsPath(signal.vendor_name, dashboardPath)}
                    onClick={(event) => event.stopPropagation()}
                    className="text-fuchsia-300 hover:text-fuchsia-200 transition-colors"
                  >
                    Reports
                  </Link>
                  <Link
                    to={opportunitiesPath(signal.vendor_name, dashboardPath)}
                    onClick={(event) => event.stopPropagation()}
                    className="text-emerald-300 hover:text-emerald-200 transition-colors"
                  >
                    Opportunities
                  </Link>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
        <h3 className="text-sm font-medium text-slate-300 mb-4">
          Recent High-Intent Companies
        </h3>
        {loading ? (
          <DataTable columns={companyColumns} data={[]} skeletonRows={5} />
        ) : (
          <DataTable
            columns={companyColumns}
            data={companies}
            onRowClick={(r) => navigate(vendorDetailPath(r.vendor, dashboardPath))}
            emptyMessage="No high-intent companies detected"
            emptyAction={{ label: 'Check Pipeline Status', onClick: () => window.scrollTo({ top: 0, behavior: 'smooth' }) }}
          />
        )}
      </div>
    </div>
  )
}
