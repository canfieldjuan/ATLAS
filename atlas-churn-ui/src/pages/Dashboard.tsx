import { useNavigate } from 'react-router-dom'
import { Building2, AlertTriangle, MessageSquareText, Zap, RefreshCw } from 'lucide-react'
import { clsx } from 'clsx'
import StatCard from '../components/StatCard'
import ChurnChart from '../components/ChurnChart'
import PipelineStatusWidget from '../components/PipelineStatus'
import DataTable, { type Column } from '../components/DataTable'
import UrgencyBadge from '../components/UrgencyBadge'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import { fetchSignals, fetchHighIntent, fetchPipeline } from '../api/client'
import type { ChurnSignal, HighIntentCompany, PipelineStatus } from '../types'

interface DashboardData {
  signals: ChurnSignal[]
  companies: HighIntentCompany[]
  pipeline: PipelineStatus
}

export default function Dashboard() {
  const navigate = useNavigate()

  const { data, loading, error, refresh, refreshing } = useApiData<DashboardData>(
    async () => {
      const [sigRes, hiRes, pipe] = await Promise.all([
        fetchSignals({ limit: 20 }),
        fetchHighIntent({ limit: 10 }),
        fetchPipeline(),
      ])
      return { signals: sigRes.signals, companies: hiRes.companies, pipeline: pipe }
    },
    [],
  )

  const signals = data?.signals ?? []
  const companies = data?.companies ?? []
  const pipeline = data?.pipeline ?? null

  const highUrgency = signals.filter((s) => s.avg_urgency_score >= 7).length
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
  ]

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Churn Intelligence Overview</h1>
        <button
          onClick={refresh}
          disabled={refreshing}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Vendors Tracked"
          value={signals.length}
          icon={<Building2 className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="High Urgency"
          value={highUrgency}
          icon={<AlertTriangle className="h-5 w-5" />}
          sub="Urgency >= 7"
          skeleton={loading}
        />
        <StatCard
          label="Total Reviews"
          value={totalReviews}
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
            onRowClick={(r) => navigate(`/vendors/${encodeURIComponent(r.vendor)}`)}
            emptyMessage="No high-intent companies detected"
            emptyAction={{ label: 'Check Pipeline Status', onClick: () => window.scrollTo({ top: 0, behavior: 'smooth' }) }}
          />
        )}
      </div>
    </div>
  )
}
