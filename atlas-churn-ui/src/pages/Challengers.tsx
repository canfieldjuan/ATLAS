import { useState, useEffect } from 'react'
import {
  Swords,
  RefreshCw,
  Send,
  TrendingUp,
  Users,
  Target,
} from 'lucide-react'
import { clsx } from 'clsx'
import StatCard from '../components/StatCard'
import DataTable, { type Column } from '../components/DataTable'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import {
  fetchVendorTargets,
  fetchHighIntent,
  generateCampaigns,
} from '../api/client'
import type { VendorTarget, HighIntentCompany } from '../types'

function StageBadge({ stage }: { stage: string }) {
  if (!stage || stage === 'unknown') return <span className="text-slate-500 text-xs">--</span>
  const colors: Record<string, string> = {
    active_purchase: 'bg-red-500/20 text-red-400',
    evaluation: 'bg-cyan-500/20 text-cyan-400',
    renewal_decision: 'bg-amber-500/20 text-amber-400',
    post_purchase: 'bg-slate-500/20 text-slate-400',
  }
  return (
    <span className={clsx('inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium', colors[stage] ?? 'bg-slate-500/20 text-slate-400')}>
      {stage.replace(/_/g, ' ')}
    </span>
  )
}

interface ChallengerSummary {
  name: string
  totalLeads: number
  activePurchase: number
  evaluation: number
  renewal: number
  topIncumbents: string[]
  topPainCategories: string[]
  avgUrgency: number
}

export default function Challengers() {
  const [searchInput, setSearchInput] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [generatingFor, setGeneratingFor] = useState<string | null>(null)
  const [actionResult, setActionResult] = useState<string | null>(null)

  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(searchInput), 300)
    return () => clearTimeout(t)
  }, [searchInput])

  const { data, loading, error, refresh, refreshing } = useApiData(
    async () => {
      const [targetsRes, hiRes] = await Promise.all([
        fetchVendorTargets({ target_mode: 'challenger_intel', limit: 200 }),
        fetchHighIntent({ min_urgency: 3, limit: 100 }),
      ])
      return {
        targets: targetsRes.targets,
        companies: hiRes.companies,
      }
    },
    [debouncedSearch],
  )

  const targets = data?.targets ?? []
  const companies = data?.companies ?? []

  // Build challenger summaries by aggregating high-intent signals where competitor matches a target
  const challengerSummaries: ChallengerSummary[] = targets.map(target => {
    const name = target.company_name.toLowerCase()
    const competitorsTracked = (target.competitors_tracked ?? []).map(c => c.toLowerCase())

    // Find high-intent companies mentioning this challenger in alternatives
    const relevantCompanies = companies.filter(c => {
      // Check if challenger is listed in alternatives (competitors being considered)
      const mentionsChallenger = (c.alternatives ?? []).some(
        alt => alt.name?.toLowerCase() === name,
      )
      if (mentionsChallenger) return true
      // Also match if the incumbent vendor is one the challenger tracks
      if (competitorsTracked.includes(c.vendor?.toLowerCase())) return true
      return false
    })

    const stages = relevantCompanies.reduce(
      (acc, c) => {
        const s = c.buying_stage ?? 'unknown'
        if (s === 'active_purchase') acc.active++
        else if (s === 'evaluation') acc.eval++
        else if (s === 'renewal_decision') acc.renewal++
        return acc
      },
      { active: 0, eval: 0, renewal: 0 },
    )

    // Top incumbents (vendors losing to this challenger)
    const incumbentCounts: Record<string, number> = {}
    for (const c of relevantCompanies) {
      const v = c.vendor
      if (v) incumbentCounts[v] = (incumbentCounts[v] ?? 0) + 1
    }
    const topIncumbents = Object.entries(incumbentCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([incumbentName]) => incumbentName)

    const avgUrg = relevantCompanies.length > 0
      ? relevantCompanies.reduce((s, c) => s + c.urgency, 0) / relevantCompanies.length
      : 0

    return {
      name: target.company_name,
      totalLeads: relevantCompanies.length,
      activePurchase: stages.active,
      evaluation: stages.eval,
      renewal: stages.renewal,
      topIncumbents,
      topPainCategories: [],
      avgUrgency: Math.round(avgUrg * 10) / 10,
    }
  })

  // Filter by search
  const filtered = debouncedSearch
    ? challengerSummaries.filter(s => s.name.toLowerCase().includes(debouncedSearch.toLowerCase()))
    : challengerSummaries

  const totalLeads = filtered.reduce((s, c) => s + c.totalLeads, 0)
  const totalActive = filtered.reduce((s, c) => s + c.activePurchase, 0)
  const totalEval = filtered.reduce((s, c) => s + c.evaluation, 0)

  async function handleGenerate(name: string) {
    setGeneratingFor(name)
    setActionResult(null)
    try {
      const result = await generateCampaigns({
        vendor_name: name,
        target_mode: 'challenger_intel',
        min_score: 50,
        limit: 5,
      })
      setActionResult(`Generated ${result.generated ?? 0} campaign(s) for ${name}`)
      refresh()
    } catch (err) {
      setActionResult(err instanceof Error ? err.message : 'Generation failed')
    } finally {
      setGeneratingFor(null)
    }
  }

  const columns: Column<ChallengerSummary>[] = [
    {
      key: 'name',
      header: 'Challenger',
      render: (r) => <span className="text-white font-medium">{r.name}</span>,
    },
    {
      key: 'leads',
      header: 'Total Leads',
      render: (r) => (
        <span className={clsx(
          'text-sm font-medium',
          r.totalLeads > 0 ? 'text-cyan-400' : 'text-slate-500',
        )}>
          {r.totalLeads}
        </span>
      ),
      sortable: true,
      sortValue: (r) => r.totalLeads,
    },
    {
      key: 'active',
      header: 'Active Purchase',
      render: (r) => (
        <span className={clsx('text-xs font-medium', r.activePurchase > 0 ? 'text-red-400' : 'text-slate-500')}>
          {r.activePurchase}
        </span>
      ),
      sortable: true,
      sortValue: (r) => r.activePurchase,
    },
    {
      key: 'eval',
      header: 'Evaluation',
      render: (r) => (
        <span className={clsx('text-xs font-medium', r.evaluation > 0 ? 'text-cyan-400' : 'text-slate-500')}>
          {r.evaluation}
        </span>
      ),
    },
    {
      key: 'renewal',
      header: 'Renewal',
      render: (r) => (
        <span className={clsx('text-xs font-medium', r.renewal > 0 ? 'text-amber-400' : 'text-slate-500')}>
          {r.renewal}
        </span>
      ),
    },
    {
      key: 'urgency',
      header: 'Avg Urgency',
      render: (r) => (
        <span className={clsx(
          'text-xs font-medium',
          r.avgUrgency >= 7 ? 'text-red-400' : r.avgUrgency >= 5 ? 'text-amber-400' : 'text-slate-400',
        )}>
          {r.avgUrgency}
        </span>
      ),
      sortable: true,
      sortValue: (r) => r.avgUrgency,
    },
    {
      key: 'incumbents',
      header: 'Losing From',
      render: (r) => (
        <span className="text-slate-400 text-xs">
          {r.topIncumbents.join(', ') || '--'}
        </span>
      ),
    },
    {
      key: 'actions',
      header: '',
      render: (r) => (
        <button
          onClick={(e) => { e.stopPropagation(); handleGenerate(r.name) }}
          disabled={generatingFor === r.name}
          className="p-1 text-slate-400 hover:text-green-400 transition-colors disabled:opacity-50"
          title="Generate Campaign"
        >
          <Send className={clsx('h-3.5 w-3.5', generatingFor === r.name && 'animate-pulse')} />
        </button>
      ),
    },
  ]

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Challenger Intelligence</h1>
        <button
          onClick={refresh}
          disabled={refreshing}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
          Refresh
        </button>
      </div>

      {/* Action result */}
      {actionResult && (
        <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-3 flex items-center justify-between">
          <span className="text-sm text-cyan-400">{actionResult}</span>
          <button onClick={() => setActionResult(null)} className="text-cyan-400 hover:text-white">
            <span className="text-lg leading-none">&times;</span>
          </button>
        </div>
      )}

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Challengers Tracked"
          value={targets.length}
          icon={<Swords className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Total Intent Leads"
          value={totalLeads}
          icon={<Target className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Active Purchase"
          value={totalActive}
          icon={<TrendingUp className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="In Evaluation"
          value={totalEval}
          icon={<Users className="h-5 w-5" />}
          skeleton={loading}
        />
      </div>

      {/* Search */}
      <div className="flex flex-wrap items-center gap-3">
        <input
          type="text"
          placeholder="Search challenger..."
          value={searchInput}
          onChange={(e) => setSearchInput(e.target.value)}
          className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 w-48"
        />
      </div>

      {/* Challengers Table */}
      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
        <h3 className="text-sm font-medium text-slate-300 mb-4">
          Challenger Lead Funnel
        </h3>
        {loading ? (
          <DataTable columns={columns} data={[]} skeletonRows={5} />
        ) : (
          <DataTable
            columns={columns}
            data={filtered}
            emptyMessage={targets.length === 0
              ? 'No challenger targets configured. Add them in Vendor Targets.'
              : 'No matching challengers found'
            }
          />
        )}
      </div>

      {targets.length === 0 && !loading && (
        <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
          <p className="text-sm text-slate-400">
            Add challenger targets in the <a href="/vendor-targets" className="text-cyan-400 hover:underline">Vendor Targets</a> page
            with mode "Challenger Intel" to start tracking competitive intent signals.
          </p>
        </div>
      )}
    </div>
  )
}
