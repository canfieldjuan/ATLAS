import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  Crosshair,
  Target,
  Send,
  TrendingUp,
  RefreshCw,
  X,
  Zap,
  Check,
  Eye,
  Download,
  Handshake,
} from 'lucide-react'
import { clsx } from 'clsx'
import StatCard from '../components/StatCard'
import DataTable, { type Column } from '../components/DataTable'
import UrgencyBadge from '../components/UrgencyBadge'
import UpgradeGate from '../components/UpgradeGate'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import { usePlanGate } from '../hooks/usePlanGate'
import {
  fetchAffiliateOpportunities,
  fetchCampaigns,
  fetchCampaignStats,
  generateCampaigns,
  approveCampaign,
  downloadCsv,
  updateCampaign,
} from '../api/client'
import type {
  AffiliateOpportunity,
  Campaign,
  CampaignStats,
} from '../types'

interface LeadsData {
  opportunities: AffiliateOpportunity[]
  campaigns: Campaign[]
  stats: CampaignStats
}

function ScoreBadge({ score }: { score: number }) {
  return (
    <span
      className={clsx(
        'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium',
        score >= 80
          ? 'bg-green-500/20 text-green-400'
          : score >= 60
            ? 'bg-cyan-500/20 text-cyan-400'
            : score >= 40
              ? 'bg-amber-500/20 text-amber-400'
              : 'bg-slate-500/20 text-slate-400',
      )}
    >
      {score}
    </span>
  )
}

function CampaignStatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    draft: 'bg-amber-500/20 text-amber-400',
    approved: 'bg-green-500/20 text-green-400',
    sent: 'bg-cyan-500/20 text-cyan-400',
    expired: 'bg-slate-500/20 text-slate-400',
  }
  return (
    <span className={clsx('inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium', colors[status] ?? 'bg-slate-500/20 text-slate-400')}>
      {status}
    </span>
  )
}

function CampaignModal({
  campaign,
  onClose,
  onApprove,
  onReject,
}: {
  campaign: Campaign
  onClose: () => void
  onApprove: (id: string) => void
  onReject: (id: string) => void
}) {
  useEffect(() => {
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = '' }
  }, [])

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div
      className="fixed top-0 right-0 bottom-0 left-0 lg:left-56 flex items-center justify-center p-8"
      style={{ zIndex: 40, backgroundColor: 'rgba(0,0,0,0.7)' }}
      onClick={onClose}
    >
      <div
        className="bg-slate-800 border-2 border-slate-500 rounded-xl flex flex-col"
        style={{ maxWidth: '500px', width: '100%', maxHeight: '80vh', boxShadow: '0 0 40px rgba(0,0,0,0.8)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between p-5 pb-3 border-b border-slate-700/50 shrink-0">
          <div>
            <h2 className="text-base font-semibold text-white">
              {campaign.company_name} &mdash; {campaign.channel.replace(/_/g, ' ')}
            </h2>
            <div className="flex items-center gap-2 mt-1">
              <CampaignStatusBadge status={campaign.status} />
              <span className="text-xs text-slate-400">
                Churning from {campaign.vendor_name}
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="overflow-y-auto p-5 space-y-3 flex-1">
          {campaign.subject && (
            <div>
              <span className="text-xs text-slate-400 uppercase tracking-wider">Subject</span>
              <p className="text-white mt-1 text-sm font-medium">{campaign.subject}</p>
            </div>
          )}
          <div>
            <span className="text-xs text-slate-400 uppercase tracking-wider">Body</span>
            <div className="mt-1 text-slate-200 whitespace-pre-wrap text-sm leading-relaxed">
              {campaign.body}
            </div>
          </div>
          {campaign.cta && (
            <div>
              <span className="text-xs text-slate-400 uppercase tracking-wider">CTA</span>
              <p className="text-cyan-400 mt-1 text-sm">{campaign.cta}</p>
            </div>
          )}
        </div>
        {campaign.status === 'draft' && (
          <div className="flex items-center gap-3 p-5 pt-3 border-t border-slate-700/50 shrink-0">
            <button
              onClick={() => onApprove(campaign.id)}
              className="px-4 py-1.5 rounded-lg text-sm bg-green-500/10 text-green-400 hover:bg-green-500/20 transition-colors"
            >
              Approve
            </button>
            <button
              onClick={() => onReject(campaign.id)}
              className="px-4 py-1.5 rounded-lg text-sm bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors"
            >
              Reject
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default function Leads() {
  const { canAccessCampaigns } = usePlanGate()

  // Filters
  const [vendorSearch, setVendorSearch] = useState('')
  const [debouncedVendor, setDebouncedVendor] = useState('')
  const [minUrgency, setMinUrgency] = useState(5)
  const [minScore, setMinScore] = useState(0)
  const [dmOnly, setDmOnly] = useState(false)

  // Campaign generation
  const [generating, setGenerating] = useState(false)
  const [genResult, setGenResult] = useState<string | null>(null)

  // Campaign detail view
  const [viewingCampaign, setViewingCampaign] = useState<Campaign | null>(null)
  const [approving, setApproving] = useState<string | null>(null)

  useEffect(() => {
    const t = setTimeout(() => setDebouncedVendor(vendorSearch), 300)
    return () => clearTimeout(t)
  }, [vendorSearch])

  const { data, loading, error, refresh, refreshing } = useApiData<LeadsData>(
    async () => {
      const [oppRes, campRes, statsRes] = await Promise.all([
        fetchAffiliateOpportunities({
          min_urgency: minUrgency,
          min_score: minScore,
          vendor_name: debouncedVendor || undefined,
          dm_only: dmOnly || undefined,
          limit: 100,
        }),
        fetchCampaigns({ limit: 200 }),
        fetchCampaignStats(),
      ])
      return {
        opportunities: oppRes.opportunities,
        campaigns: campRes.campaigns,
        stats: statsRes,
      }
    },
    [minUrgency, minScore, debouncedVendor, dmOnly],
  )

  const opportunities = data?.opportunities ?? []
  const campaigns = data?.campaigns ?? []
  const stats = data?.stats ?? { by_status: {}, by_channel: {}, top_vendors: [], total: 0 }

  const campaignsSent = (stats.by_status['sent'] ?? 0) + (stats.by_status['approved'] ?? 0)
  const avgScore =
    opportunities.length > 0
      ? Math.round(opportunities.reduce((s, o) => s + o.opportunity_score, 0) / opportunities.length)
      : 0

  // Build a lookup: company_name (lower) -> campaign status
  const campaignByCompany = new Map<string, Campaign>()
  for (const c of campaigns) {
    const key = c.company_name.toLowerCase()
    // Keep the most recent campaign per company
    if (!campaignByCompany.has(key)) {
      campaignByCompany.set(key, c)
    }
  }

  // -- Generate handler --
  async function handleGenerate() {
    setGenerating(true)
    setGenResult(null)
    try {
      const result = await generateCampaigns({
        vendor_name: debouncedVendor || undefined,
        min_score: Math.max(minScore, 70),
        limit: 20,
      })
      setGenResult(`Generated ${result.generated ?? 0} campaign(s) for ${result.companies ?? 0} companies`)
      refresh()
    } catch (err) {
      setGenResult(err instanceof Error ? err.message : 'Generation failed')
    } finally {
      setGenerating(false)
    }
  }

  // -- Approve handler --
  async function handleApprove(id: string) {
    setApproving(id)
    try {
      await approveCampaign(id)
      refresh()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Approve failed')
    } finally {
      setApproving(null)
    }
  }

  // -- Reject handler --
  async function handleReject(id: string) {
    try {
      await updateCampaign(id, { status: 'expired' })
      refresh()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Reject failed')
    }
  }

  // -- Table columns --
  const columns: Column<AffiliateOpportunity>[] = [
    {
      key: 'score',
      header: 'Score',
      render: (r) => <ScoreBadge score={r.opportunity_score} />,
      sortable: true,
      sortValue: (r) => r.opportunity_score,
    },
    {
      key: 'vendor',
      header: 'Churning From',
      render: (r) => <span className="text-white font-medium">{r.vendor_name}</span>,
    },
    {
      key: 'company',
      header: 'Company',
      render: (r) => <span className="text-slate-300">{r.reviewer_company ?? '--'}</span>,
    },
    {
      key: 'competitor',
      header: 'Considering',
      render: (r) => (
        <span className="text-cyan-400 font-medium">{r.competitor_name}</span>
      ),
    },
    {
      key: 'urgency',
      header: 'Urgency',
      render: (r) => <UrgencyBadge score={r.urgency} />,
      sortable: true,
      sortValue: (r) => r.urgency,
    },
    {
      key: 'dm',
      header: 'DM',
      render: (r) =>
        r.is_dm ? (
          <span className="text-cyan-400 text-xs font-medium">Yes</span>
        ) : (
          <span className="text-slate-500 text-xs">No</span>
        ),
    },
    {
      key: 'stage',
      header: 'Stage',
      render: (r) => {
        if (!r.buying_stage || r.buying_stage === 'unknown')
          return <span className="text-slate-500 text-xs">--</span>
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
      key: 'seats',
      header: 'Seats',
      render: (r) => <span className="text-slate-300">{r.seat_count ?? '--'}</span>,
      sortable: true,
      sortValue: (r) => r.seat_count ?? 0,
    },
    {
      key: 'campaign',
      header: 'Campaign',
      render: (r) => {
        const company = (r.reviewer_company ?? '').toLowerCase()
        const campaign = campaignByCompany.get(company)
        if (!campaign) {
          return <span className="text-slate-500 text-xs">--</span>
        }
        return <CampaignStatusBadge status={campaign.status} />
      },
    },
  ]

  // -- Campaign table columns --
  const campaignColumns: Column<Campaign>[] = [
    {
      key: 'company',
      header: 'Company',
      render: (r) => <span className="text-white font-medium">{r.company_name}</span>,
    },
    {
      key: 'vendor',
      header: 'Churning From',
      render: (r) => <span className="text-slate-300">{r.vendor_name}</span>,
    },
    {
      key: 'channel',
      header: 'Channel',
      render: (r) => (
        <span className="text-xs text-slate-300">
          {r.channel.replace(/_/g, ' ')}
        </span>
      ),
    },
    {
      key: 'subject',
      header: 'Subject',
      render: (r) => (
        <span className="text-slate-400 text-xs max-w-[200px] truncate block">
          {r.subject ?? '--'}
        </span>
      ),
    },
    {
      key: 'status',
      header: 'Status',
      render: (r) => <CampaignStatusBadge status={r.status} />,
    },
    {
      key: 'actions',
      header: '',
      render: (r) => (
        <div className="flex items-center gap-1">
          <button
            onClick={(e) => {
              e.stopPropagation()
              setViewingCampaign(r)
            }}
            className="p-1 text-slate-400 hover:text-white transition-colors"
            title="View"
          >
            <Eye className="h-3.5 w-3.5" />
          </button>
          {r.status === 'draft' && (
            <>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  handleApprove(r.id)
                }}
                disabled={approving === r.id}
                className="p-1 text-green-400 hover:text-green-300 transition-colors disabled:opacity-50"
                title="Approve"
              >
                <Check className="h-3.5 w-3.5" />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  handleReject(r.id)
                }}
                className="p-1 text-red-400 hover:text-red-300 transition-colors"
                title="Reject"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </>
          )}
        </div>
      ),
    },
  ]

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Lead Intelligence</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={() =>
              downloadCsv('/export/high-intent', {
                vendor_name: debouncedVendor || undefined,
                min_urgency: minUrgency || undefined,
              })
            }
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors"
          >
            <Download className="h-4 w-4" />
            Export CSV
          </button>
          {canAccessCampaigns && (
            <button
              onClick={handleGenerate}
              disabled={generating}
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 transition-colors disabled:opacity-50"
            >
              <Zap className={clsx('h-4 w-4', generating && 'animate-pulse')} />
              {generating ? 'Generating...' : 'Generate Campaigns'}
            </button>
          )}
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

      {/* Generation result banner */}
      {genResult && (
        <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-3 flex items-center justify-between">
          <span className="text-sm text-cyan-400">{genResult}</span>
          <button onClick={() => setGenResult(null)} className="text-cyan-400 hover:text-white">
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Active Campaigns"
          value={stats.total}
          icon={<Crosshair className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Opportunities Found"
          value={opportunities.length}
          icon={<Target className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Campaigns Sent"
          value={campaignsSent}
          icon={<Send className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Avg Score"
          value={avgScore}
          icon={<TrendingUp className="h-5 w-5" />}
          skeleton={loading}
        />
      </div>

      {/* Opportunities Table */}
      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
        <h3 className="text-sm font-medium text-slate-300 mb-4">Ranked Opportunities</h3>

        {/* Filters */}
        <div className="flex flex-wrap items-center gap-3 mb-4">
          <input
            type="text"
            placeholder="Filter vendor..."
            value={vendorSearch}
            onChange={(e) => setVendorSearch(e.target.value)}
            className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 w-48"
          />
          <label className="flex items-center gap-2 text-sm text-slate-400">
            Min Urgency
            <input
              type="number"
              min={0}
              max={10}
              step={1}
              value={minUrgency}
              onChange={(e) => setMinUrgency(Number(e.target.value))}
              className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-2 py-1.5 text-sm text-white w-16 focus:outline-none focus:border-cyan-500/50"
            />
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-400">
            Min Score
            <input
              type="number"
              min={0}
              max={100}
              step={5}
              value={minScore}
              onChange={(e) => setMinScore(Number(e.target.value))}
              className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-2 py-1.5 text-sm text-white w-16 focus:outline-none focus:border-cyan-500/50"
            />
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer">
            <input
              type="checkbox"
              checked={dmOnly}
              onChange={(e) => setDmOnly(e.target.checked)}
              className="rounded border-slate-600 bg-slate-800 text-cyan-500 focus:ring-cyan-500/30"
            />
            DM only
          </label>
        </div>

        {loading ? (
          <DataTable columns={columns} data={[]} skeletonRows={5} />
        ) : (
          <DataTable
            columns={columns}
            data={opportunities}
            emptyMessage="No lead opportunities found"
          />
        )}
      </div>

      {/* Campaigns Table */}
      {canAccessCampaigns && campaigns.length > 0 && (
        <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
          <h3 className="text-sm font-medium text-slate-300 mb-4">
            Generated Campaigns ({campaigns.length})
          </h3>
          <DataTable
            columns={campaignColumns}
            data={campaigns}
            emptyMessage="No campaigns generated yet"
          />
        </div>
      )}

      {!canAccessCampaigns && (
        <UpgradeGate allowed={false} feature="Campaigns" requiredPlan="Growth">
          <div />
        </UpgradeGate>
      )}

      {/* Campaign Detail Modal */}
      {viewingCampaign && (
        <CampaignModal
          campaign={viewingCampaign}
          onClose={() => setViewingCampaign(null)}
          onApprove={(id) => { handleApprove(id); setViewingCampaign(null) }}
          onReject={(id) => { handleReject(id); setViewingCampaign(null) }}
        />
      )}

      {/* Partner Management Link */}
      <Link
        to="/affiliates"
        className="flex items-center justify-between bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5 hover:border-cyan-500/30 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Handshake className="h-5 w-5 text-cyan-400" />
          <div>
            <h3 className="text-sm font-medium text-white">Partner Management</h3>
            <p className="text-xs text-slate-400 mt-0.5">Manage affiliate partners, commissions, and click tracking</p>
          </div>
        </div>
        <span className="text-xs text-slate-400">View &rarr;</span>
      </Link>
    </div>
  )
}
