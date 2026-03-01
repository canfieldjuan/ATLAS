import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  MessageSquareText,
  Tag,
  ShieldAlert,
  Zap,
  RefreshCw,
  Package,
} from 'lucide-react'
import { clsx } from 'clsx'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts'
import StatCard from '../components/StatCard'
import DataTable, { type Column } from '../components/DataTable'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import {
  fetchPipeline,
  fetchBrands,
  fetchSafety,
  type PipelineStatus,
  type BrandSummary,
  type SafetySignal,
} from '../api/client'

const PIE_COLORS = ['#22d3ee', '#a78bfa', '#f472b6', '#facc15', '#34d399', '#fb923c']

interface DashboardData {
  pipeline: PipelineStatus
  brands: BrandSummary[]
  safety: SafetySignal[]
  safetyTotal: number
}

export default function Dashboard() {
  const navigate = useNavigate()
  const [category, setCategory] = useState('')

  const { data, loading, error, refresh, refreshing } = useApiData<DashboardData>(
    async () => {
      const cat = category || undefined
      const [pipeline, brandsRes, safetyRes] = await Promise.all([
        fetchPipeline({ source_category: cat }),
        fetchBrands({ limit: 10, sort_by: 'review_count', source_category: cat }),
        fetchSafety({ limit: 5, source_category: cat }),
      ])
      return {
        pipeline,
        brands: brandsRes.brands,
        safety: safetyRes.signals,
        safetyTotal: safetyRes.total_flagged,
      }
    },
    [category],
  )

  const pipeline = data?.pipeline
  const brands = data?.brands ?? []
  const safety = data?.safety ?? []

  const totalReviews = pipeline?.total_reviews ?? 0
  const brandCount = pipeline?.total_brands ?? 0
  const asinCount = pipeline?.total_asins ?? 0
  const safetyTotal = data?.safetyTotal ?? 0
  const targetedForDeep = pipeline?.targeted_for_deep ?? 0
  const deepEnriched = pipeline?.deep_enriched ?? 0

  // Pipeline progress bars
  const enrichedPct = pipeline && totalReviews > 0
    ? Math.round((pipeline.enriched / totalReviews) * 100)
    : 0
  const deepPct = targetedForDeep > 0
    ? Math.round((deepEnriched / targetedForDeep) * 100)
    : 0

  // Category pie chart data
  const categoryData = pipeline
    ? Object.entries(pipeline.category_counts).map(([name, value]) => ({ name, value }))
    : []

  // Safety table columns
  const safetyColumns: Column<SafetySignal>[] = [
    {
      key: 'brand',
      header: 'Brand',
      render: (r) => <span className="text-white font-medium">{r.brand}</span>,
    },
    {
      key: 'title',
      header: 'Product',
      render: (r) => <span className="text-slate-300 truncate max-w-[200px] block">{r.title}</span>,
    },
    {
      key: 'rating',
      header: 'Rating',
      render: (r) => <span className="text-slate-300">{r.rating?.toFixed(1) ?? '--'}</span>,
    },
    {
      key: 'summary',
      header: 'Summary',
      render: (r) => (
        <span className="text-slate-400 truncate max-w-[300px] block">
          {r.summary ?? r.review_excerpt ?? '--'}
        </span>
      ),
    },
  ]

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Consumer Intelligence Overview</h1>
        <div className="flex items-center gap-3">
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            className="px-3 py-1.5 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white focus:outline-none focus:border-cyan-500/50"
          >
            <option value="">All Categories</option>
            {categoryData.map((d) => (
              <option key={d.name} value={d.name}>{d.name}</option>
            ))}
          </select>
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

      {/* Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <StatCard
          label="Total Reviews"
          value={totalReviews.toLocaleString()}
          icon={<MessageSquareText className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Brands Tracked"
          value={brandCount.toLocaleString()}
          icon={<Tag className="h-5 w-5" />}
          sub={`${asinCount.toLocaleString()} products`}
          skeleton={loading}
        />
        <StatCard
          label="Safety Signals"
          value={safetyTotal}
          icon={<ShieldAlert className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="First-Pass"
          value={`${enrichedPct}%`}
          icon={<Package className="h-5 w-5" />}
          sub={pipeline ? `${pipeline.enriched.toLocaleString()} of ${totalReviews.toLocaleString()}` : undefined}
          skeleton={loading}
        />
        <StatCard
          label="Deep Enrichment"
          value={`${deepPct}%`}
          icon={<Zap className="h-5 w-5" />}
          sub={targetedForDeep > 0 ? `${deepEnriched.toLocaleString()} of ${targetedForDeep.toLocaleString()} targeted` : undefined}
          skeleton={loading}
        />
      </div>

      {/* Pipeline + Categories */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Pipeline Status */}
        <div className="lg:col-span-2 bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
          <h3 className="text-sm font-medium text-slate-300 mb-4">Pipeline Status</h3>
          {loading ? (
            <div className="space-y-4 animate-pulse">
              <div className="h-4 w-3/4 bg-slate-700/50 rounded" />
              <div className="h-4 w-1/2 bg-slate-700/50 rounded" />
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>First-pass Enrichment</span>
                  <span>{enrichedPct}% ({pipeline?.enriched.toLocaleString()} of {totalReviews.toLocaleString()})</span>
                </div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-cyan-500 rounded-full transition-all"
                    style={{ width: `${enrichedPct}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Deep Enrichment</span>
                  <span>{deepPct}% ({deepEnriched.toLocaleString()} of {targetedForDeep.toLocaleString()} targeted)</span>
                </div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-purple-500 rounded-full transition-all"
                    style={{ width: `${deepPct}%` }}
                  />
                </div>
              </div>
              {/* Status breakdown (targeted pool only) */}
              {pipeline && (
                <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500">
                  {Object.entries(pipeline.deep_enrichment_counts)
                    .filter(([status]) => status !== 'not_applicable')
                    .sort(([, a], [, b]) => b - a)
                    .map(([status, count]) => (
                      <span key={status}>
                        <span className={
                          status === 'enriched' ? 'text-purple-400' :
                          status === 'pending' ? 'text-amber-400' :
                          status === 'processing' ? 'text-cyan-400' :
                          status.includes('fail') ? 'text-red-400' :
                          'text-slate-500'
                        }>{count.toLocaleString()}</span>
                        {' '}{status.replace('deep_', '').replaceAll('_', ' ')}
                      </span>
                    ))
                  }
                </div>
              )}
              {pipeline?.last_deep_enrichment_at && (
                <p className="text-xs text-slate-500">
                  Last deep enrichment: {new Date(pipeline.last_deep_enrichment_at).toLocaleString()}
                </p>
              )}
            </div>
          )}
        </div>

        {/* Category Breakdown */}
        <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
          <h3 className="text-sm font-medium text-slate-300 mb-4">Source Categories</h3>
          {loading ? (
            <div className="h-[200px] animate-pulse bg-slate-700/30 rounded" />
          ) : categoryData.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={categoryData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  paddingAngle={3}
                  dataKey="value"
                >
                  {categoryData.map((_, i) => (
                    <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: 8,
                    color: '#e2e8f0',
                    fontSize: 13,
                  }}
                  formatter={(value: number) => [value.toLocaleString(), 'Reviews']}
                />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-slate-500 text-sm">No data</p>
          )}
          <div className="flex flex-wrap gap-2 mt-2">
            {categoryData.map((d, i) => (
              <span key={d.name} className="flex items-center gap-1 text-xs text-slate-400">
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: PIE_COLORS[i % PIE_COLORS.length] }}
                />
                {d.name} ({d.value.toLocaleString()})
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Top Brands Chart */}
      {brands.length > 0 && (
        <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
          <h3 className="text-sm font-medium text-slate-300 mb-4">Top Brands by Review Count</h3>
          {loading ? (
            <div className="h-[250px] animate-pulse bg-slate-700/30 rounded" />
          ) : (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={brands.slice(0, 10)} margin={{ left: 10, bottom: 40 }}>
                <XAxis
                  dataKey="brand"
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  axisLine={{ stroke: '#334155' }}
                  angle={-35}
                  textAnchor="end"
                  interval={0}
                />
                <YAxis
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  axisLine={{ stroke: '#334155' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: 8,
                    color: '#e2e8f0',
                    fontSize: 13,
                  }}
                />
                <Bar dataKey="review_count" fill="#22d3ee" radius={[4, 4, 0, 0]} name="Reviews" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      )}

      {/* Recent Safety Signals */}
      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
        <h3 className="text-sm font-medium text-slate-300 mb-4">Recent Safety Signals</h3>
        {loading ? (
          <DataTable columns={safetyColumns} data={[]} skeletonRows={5} />
        ) : (
          <DataTable
            columns={safetyColumns}
            data={safety}
            onRowClick={(r) => navigate(`/reviews/${r.id}`)}
            emptyMessage="No safety signals detected"
          />
        )}
      </div>
    </div>
  )
}
