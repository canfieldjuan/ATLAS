import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Building2, AlertTriangle, MessageSquareText, Zap } from 'lucide-react'
import StatCard from '../components/StatCard'
import ChurnChart from '../components/ChurnChart'
import PipelineStatusWidget from '../components/PipelineStatus'
import DataTable, { type Column } from '../components/DataTable'
import UrgencyBadge from '../components/UrgencyBadge'
import { fetchSignals, fetchHighIntent, fetchPipeline } from '../api/client'
import type { ChurnSignal, HighIntentCompany, PipelineStatus } from '../types'

export default function Dashboard() {
  const navigate = useNavigate()
  const [signals, setSignals] = useState<ChurnSignal[]>([])
  const [companies, setCompanies] = useState<HighIntentCompany[]>([])
  const [pipeline, setPipeline] = useState<PipelineStatus | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function load() {
      try {
        const [sigRes, hiRes, pipe] = await Promise.all([
          fetchSignals({ limit: 20 }),
          fetchHighIntent({ limit: 10 }),
          fetchPipeline(),
        ])
        setSignals(sigRes.signals)
        setCompanies(hiRes.companies)
        setPipeline(pipe)
      } catch (err) {
        console.error('Dashboard load error:', err)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

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
      header: 'Pain Category',
      render: (r) => <span className="text-slate-400">{r.pain ?? '--'}</span>,
    },
    {
      key: 'dm',
      header: 'Decision Maker',
      render: (r) =>
        r.decision_maker ? (
          <span className="text-cyan-400 text-xs font-medium">Yes</span>
        ) : (
          <span className="text-slate-500 text-xs">No</span>
        ),
    },
  ]

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-500">
        Loading dashboard...
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Churn Intelligence Overview</h1>

      <div className="grid grid-cols-4 gap-4">
        <StatCard
          label="Vendors Tracked"
          value={signals.length}
          icon={<Building2 className="h-5 w-5" />}
        />
        <StatCard
          label="High Urgency"
          value={highUrgency}
          icon={<AlertTriangle className="h-5 w-5" />}
          sub="Urgency >= 7"
        />
        <StatCard
          label="Total Reviews"
          value={totalReviews}
          icon={<MessageSquareText className="h-5 w-5" />}
        />
        <StatCard
          label="Enrichment Rate"
          value={`${enrichRate}%`}
          icon={<Zap className="h-5 w-5" />}
        />
      </div>

      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-2 bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
          <h3 className="text-sm font-medium text-slate-300 mb-4">
            Top Churn Signals by Urgency
          </h3>
          <ChurnChart signals={signals} />
        </div>
        <PipelineStatusWidget data={pipeline} />
      </div>

      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
        <h3 className="text-sm font-medium text-slate-300 mb-4">
          Recent High-Intent Companies
        </h3>
        <DataTable
          columns={companyColumns}
          data={companies}
          onRowClick={(r) => navigate(`/vendors/${encodeURIComponent(r.vendor)}`)}
          emptyMessage="No high-intent companies detected"
        />
      </div>
    </div>
  )
}
