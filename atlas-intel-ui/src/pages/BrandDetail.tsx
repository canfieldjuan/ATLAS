import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, RefreshCw } from 'lucide-react'
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
import DataTable, { type Column } from '../components/DataTable'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import { fetchBrandDetail, type BrandDetail as BrandDetailType, type BrandProduct } from '../api/client'

const PIE_COLORS = ['#22d3ee', '#a78bfa', '#f472b6', '#facc15', '#34d399', '#fb923c']

function DetailSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="h-4 w-32 bg-slate-700/50 rounded" />
      <div className="h-7 w-48 bg-slate-700/50 rounded" />
      <div className="h-4 w-36 bg-slate-700/50 rounded" />
      <div className="h-10 w-64 bg-slate-700/50 rounded" />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 h-64" />
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 h-64" />
      </div>
    </div>
  )
}

export default function BrandDetail() {
  const { name } = useParams<{ name: string }>()
  const navigate = useNavigate()
  const [tab, setTab] = useState<'products' | 'sentiment' | 'intelligence'>('products')

  const { data, loading, error, refresh, refreshing } = useApiData<BrandDetailType>(
    () => {
      if (!name) return Promise.reject(new Error('Missing brand name'))
      return fetchBrandDetail(name)
    },
    [name],
  )

  if (error) return <PageError error={error} onRetry={refresh} />
  if (loading) return <DetailSkeleton />
  if (!data) return <PageError error={new Error('Brand not found')} />

  const productColumns: Column<BrandProduct>[] = [
    {
      key: 'title',
      header: 'Product',
      render: (r) => <span className="text-white max-w-[300px] truncate block">{r.title}</span>,
    },
    {
      key: 'asin',
      header: 'ASIN',
      render: (r) => <span className="text-slate-400 font-mono text-xs">{r.asin}</span>,
    },
    {
      key: 'rating',
      header: 'Rating',
      render: (r) => <span className="text-slate-300">{r.average_rating?.toFixed(1) ?? '--'}</span>,
      sortable: true,
      sortValue: (r) => r.average_rating ?? 0,
    },
    {
      key: 'reviews',
      header: 'Reviews',
      render: (r) => <span className="text-slate-300">{r.review_count}</span>,
      sortable: true,
      sortValue: (r) => r.review_count,
    },
    {
      key: 'price',
      header: 'Price',
      render: (r) => <span className="text-slate-300">{r.price ?? '--'}</span>,
    },
    {
      key: 'pain',
      header: 'Pain',
      render: (r) => {
        const score = r.avg_pain_score
        if (score == null) return <span className="text-slate-500">--</span>
        const color = score >= 7 ? 'text-red-400' : score >= 4 ? 'text-yellow-400' : 'text-green-400'
        return <span className={color}>{score.toFixed(1)}</span>
      },
      sortable: true,
      sortValue: (r) => r.avg_pain_score ?? 0,
    },
  ]

  // Sentiment chart data (top 15 by total mentions)
  const sentimentData = data.sentiment_aspects.slice(0, 15)

  // Loyalty pie data
  const loyaltyData = data.loyalty_breakdown.map((l) => ({ name: l.level, value: l.count }))

  return (
    <div className="space-y-6">
      <button
        onClick={() => navigate('/brands')}
        className="flex items-center gap-1 text-sm text-slate-400 hover:text-white transition-colors"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to Brands
      </button>

      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">{data.brand}</h1>
          <p className="text-sm text-slate-400 mt-1">
            {data.product_count} products, {data.total_reviews.toLocaleString()} reviews
            {data.avg_rating != null && ` | Avg ${data.avg_rating.toFixed(1)}`}
          </p>
        </div>
        <button
          onClick={refresh}
          disabled={refreshing}
          className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-slate-700/50">
        {(['products', 'sentiment', 'intelligence'] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm capitalize transition-colors ${
              tab === t
                ? 'text-cyan-400 border-b-2 border-cyan-400'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Products Tab */}
      {tab === 'products' && (
        <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl overflow-hidden">
          <DataTable
            columns={productColumns}
            data={data.products}
            onRowClick={(r) => navigate(`/reviews?asin=${encodeURIComponent(r.asin)}`)}
            emptyMessage="No products found"
          />
        </div>
      )}

      {/* Sentiment Tab */}
      {tab === 'sentiment' && (
        <div className="space-y-6">
          {sentimentData.length > 0 ? (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
              <h3 className="text-sm font-medium text-slate-300 mb-4">Sentiment Aspects</h3>
              <ResponsiveContainer width="100%" height={Math.max(300, sentimentData.length * 32)}>
                <BarChart data={sentimentData} layout="vertical" margin={{ left: 120 }}>
                  <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#334155' }} />
                  <YAxis
                    type="category"
                    dataKey="aspect"
                    tick={{ fill: '#94a3b8', fontSize: 11 }}
                    axisLine={{ stroke: '#334155' }}
                    width={120}
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
                  <Bar dataKey="positive" stackId="a" fill="#34d399" name="Positive" />
                  <Bar dataKey="negative" stackId="a" fill="#f87171" name="Negative" />
                  <Bar dataKey="neutral" stackId="a" fill="#64748b" name="Neutral" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="text-slate-500 text-sm py-8 text-center">No sentiment data available</p>
          )}
        </div>
      )}

      {/* Intelligence Tab */}
      {tab === 'intelligence' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Feature Requests */}
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
            <h3 className="text-sm font-medium text-slate-300 mb-3">Top Feature Requests</h3>
            {data.top_features.length > 0 ? (
              <ul className="space-y-2 max-h-64 overflow-y-auto">
                {data.top_features.map((f, i) => (
                  <li key={i} className="flex items-center justify-between text-sm">
                    <span className="text-slate-300 truncate mr-2">{f.request}</span>
                    <span className="text-cyan-400 font-mono text-xs shrink-0">{f.count}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-slate-500 text-sm">No feature requests found</p>
            )}
          </div>

          {/* Competitive Flows */}
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
            <h3 className="text-sm font-medium text-slate-300 mb-3">Competitive Flows</h3>
            {data.competitive_flows.length > 0 ? (
              <ul className="space-y-2 max-h-64 overflow-y-auto">
                {data.competitive_flows.map((f, i) => (
                  <li key={i} className="flex items-center justify-between text-sm">
                    <span className="text-slate-300">
                      <span className="text-white font-medium">{f.brand}</span>
                      <span className="text-slate-500 text-xs ml-1">({f.direction})</span>
                    </span>
                    <span className="text-cyan-400 font-mono text-xs">{f.count}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-slate-500 text-sm">No competitive flows found</p>
            )}
          </div>

          {/* Loyalty Breakdown */}
          {loyaltyData.length > 0 && (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
              <h3 className="text-sm font-medium text-slate-300 mb-3">Loyalty Depth</h3>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={loyaltyData}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={70}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {loyaltyData.map((_, i) => (
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
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex flex-wrap gap-2 mt-2">
                {loyaltyData.map((d, i) => (
                  <span key={d.name} className="flex items-center gap-1 text-xs text-slate-400">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: PIE_COLORS[i % PIE_COLORS.length] }}
                    />
                    {d.name} ({d.value})
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Buyer Profile */}
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
            <h3 className="text-sm font-medium text-slate-300 mb-3">Buyer Profile</h3>
            <div className="space-y-4">
              {data.buyer_profile.expertise.length > 0 && (
                <div>
                  <h4 className="text-xs text-slate-400 mb-1">Expertise Level</h4>
                  <div className="flex flex-wrap gap-1">
                    {data.buyer_profile.expertise.map((e, i) => (
                      <span key={e.level ?? i} className="px-2 py-1 bg-slate-800 rounded text-xs text-slate-300">
                        {e.level} ({e.count})
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {data.buyer_profile.budget.length > 0 && (
                <div>
                  <h4 className="text-xs text-slate-400 mb-1">Budget Type</h4>
                  <div className="flex flex-wrap gap-1">
                    {data.buyer_profile.budget.map((b, i) => (
                      <span key={b.type ?? i} className="px-2 py-1 bg-slate-800 rounded text-xs text-slate-300">
                        {b.type} ({b.count})
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {data.buyer_profile.discovery_channel.length > 0 && (
                <div>
                  <h4 className="text-xs text-slate-400 mb-1">Discovery Channel</h4>
                  <div className="flex flex-wrap gap-1">
                    {data.buyer_profile.discovery_channel.map((c, i) => (
                      <span key={c.channel ?? i} className="px-2 py-1 bg-slate-800 rounded text-xs text-slate-300">
                        {c.channel} ({c.count})
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
