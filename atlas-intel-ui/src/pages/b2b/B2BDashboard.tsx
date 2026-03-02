import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Activity, Users, AlertTriangle, MessageSquareText, RefreshCw, Target } from 'lucide-react'
import StatCard from '../../components/StatCard'
import DataTable, { type Column } from '../../components/DataTable'
import { fetchOverview, fetchTrackedVendors, type DashboardOverview, type TrackedVendor } from '../../api/b2bClient'
import { useAuth } from '../../auth/AuthContext'

export default function B2BDashboard() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const [overview, setOverview] = useState<DashboardOverview | null>(null)
  const [vendors, setVendors] = useState<TrackedVendor[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)

  const load = async () => {
    try {
      const [ov, tv] = await Promise.all([fetchOverview(), fetchTrackedVendors()])
      setOverview(ov)
      setVendors(tv.vendors)
    } catch {
      // handled by empty state
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  useEffect(() => { load() }, [])

  // If no vendors tracked yet, redirect to onboarding
  useEffect(() => {
    if (!loading && vendors.length === 0) {
      navigate('/b2b/onboarding', { replace: true })
    }
  }, [loading, vendors, navigate])

  const isChallenger = user?.product === 'b2b_challenger'

  const vendorColumns: Column<TrackedVendor>[] = [
    {
      key: 'vendor',
      header: 'Vendor',
      render: r => <span className="text-white font-medium">{r.vendor_name}</span>,
    },
    {
      key: 'mode',
      header: 'Mode',
      render: r => (
        <span className={r.track_mode === 'competitor' ? 'text-amber-400' : 'text-cyan-400'}>
          {r.track_mode === 'competitor' ? 'Competitor' : 'Own'}
        </span>
      ),
    },
    {
      key: 'urgency',
      header: 'Avg Urgency',
      sortable: true,
      sortValue: r => r.avg_urgency ?? 0,
      render: r => {
        const v = r.avg_urgency
        if (v == null) return <span className="text-slate-600">--</span>
        const color = v >= 7 ? 'text-red-400' : v >= 4 ? 'text-amber-400' : 'text-green-400'
        return <span className={color}>{v.toFixed(1)}</span>
      },
    },
    {
      key: 'churn',
      header: 'Churn Signals',
      sortable: true,
      sortValue: r => r.churn_intent_count ?? 0,
      render: r => <span className="text-slate-300">{r.churn_intent_count ?? 0}</span>,
    },
    {
      key: 'reviews',
      header: 'Reviews',
      sortable: true,
      sortValue: r => r.total_reviews ?? 0,
      render: r => <span className="text-slate-300">{r.total_reviews ?? 0}</span>,
    },
  ]

  const leadColumns: Column<DashboardOverview['recent_leads'][number]>[] = [
    {
      key: 'company',
      header: 'Company',
      render: r => <span className="text-white font-medium">{r.company}</span>,
    },
    {
      key: 'vendor',
      header: 'Leaving',
      render: r => <span className="text-slate-300">{r.vendor}</span>,
    },
    {
      key: 'urgency',
      header: 'Urgency',
      render: r => {
        const color = r.urgency >= 8 ? 'text-red-400' : 'text-amber-400'
        return <span className={color}>{r.urgency.toFixed(1)}</span>
      },
    },
    {
      key: 'pain',
      header: 'Pain',
      render: r => <span className="text-slate-400">{r.pain || '--'}</span>,
    },
  ]

  const skeleton = loading

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">
          {isChallenger ? 'Challenger Dashboard' : 'Vendor Health Dashboard'}
        </h1>
        <button
          onClick={() => { setRefreshing(true); load() }}
          disabled={refreshing}
          className="flex items-center gap-2 px-3 py-1.5 text-sm text-slate-400 hover:text-white transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Tracked Vendors"
          value={overview?.tracked_vendors ?? 0}
          icon={<Users className="h-5 w-5" />}
          sub={`Limit: ${user?.vendor_limit ?? 1}`}
          skeleton={skeleton}
        />
        <StatCard
          label="Avg Urgency"
          value={overview?.avg_urgency?.toFixed(1) ?? '0.0'}
          icon={<AlertTriangle className="h-5 w-5" />}
          skeleton={skeleton}
        />
        <StatCard
          label="Churn Signals"
          value={overview?.total_churn_signals ?? 0}
          icon={<Activity className="h-5 w-5" />}
          skeleton={skeleton}
        />
        <StatCard
          label="Total Reviews"
          value={overview?.total_reviews ?? 0}
          icon={<MessageSquareText className="h-5 w-5" />}
          skeleton={skeleton}
        />
      </div>

      {/* Tracked vendors table */}
      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-slate-300">Tracked Vendors</h2>
          <button
            onClick={() => navigate('/b2b/onboarding')}
            className="text-xs text-cyan-400 hover:text-cyan-300"
          >
            + Add vendor
          </button>
        </div>
        <DataTable
          columns={vendorColumns}
          data={vendors}
          onRowClick={r => navigate(`/b2b/signals/${encodeURIComponent(r.vendor_name)}`)}
          skeletonRows={skeleton ? 3 : undefined}
          emptyMessage="No vendors tracked yet"
          emptyAction={{ label: 'Add vendors', onClick: () => navigate('/b2b/onboarding') }}
        />
      </div>

      {/* Recent leads */}
      {(isChallenger || (overview?.recent_leads?.length ?? 0) > 0) && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-slate-300">
              <Target className="inline h-4 w-4 mr-1 text-amber-400" />
              Recent High-Intent Leads
            </h2>
            <button
              onClick={() => navigate('/b2b/leads')}
              className="text-xs text-cyan-400 hover:text-cyan-300"
            >
              View all
            </button>
          </div>
          <DataTable
            columns={leadColumns}
            data={overview?.recent_leads ?? []}
            onRowClick={r => navigate(`/b2b/leads/${encodeURIComponent(r.company)}`)}
            skeletonRows={skeleton ? 3 : undefined}
            emptyMessage="No high-intent leads yet"
          />
        </div>
      )}
    </div>
  )
}
