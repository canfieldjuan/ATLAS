import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, RefreshCw, ShieldAlert, DollarSign, AlertTriangle } from 'lucide-react'
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
import {
  fetchBrandDetail,
  type BrandDetail as BrandDetailType,
  type BrandProduct,
  type LabelCount,
} from '../api/client'

const PIE_COLORS = ['#22d3ee', '#a78bfa', '#f472b6', '#facc15', '#34d399', '#fb923c']

const TOOLTIP_STYLE = {
  backgroundColor: '#1e293b',
  border: '1px solid #334155',
  borderRadius: 8,
  color: '#e2e8f0',
  fontSize: 13,
}

// Color mappings for common enum values
const CHURN_COLORS: Record<string, string> = {
  // replacement_behavior
  avoided: 'bg-slate-600', kept_using: 'bg-emerald-600', switched_to: 'bg-red-500',
  returned: 'bg-amber-500', kept_broken: 'bg-red-700', repurchased: 'bg-emerald-500',
  replaced_same: 'bg-cyan-500', switched_brand: 'bg-red-600',
  // trajectory
  always_positive: 'bg-emerald-500', always_negative: 'bg-red-500', degraded: 'bg-amber-500',
  improved: 'bg-cyan-500', mixed_then_negative: 'bg-red-400', mixed_then_positive: 'bg-emerald-400',
  mixed_then_bad: 'bg-red-400', always_bad: 'bg-red-600',
  // switching barrier
  none: 'bg-slate-600', low: 'bg-emerald-600', medium: 'bg-amber-500', high: 'bg-red-500',
  // repurchase
  true: 'bg-emerald-500', false: 'bg-red-500',
  // consequence
  inconvenience: 'bg-slate-500', positive_impact: 'bg-emerald-500',
  financial_loss: 'bg-red-500', workflow_impact: 'bg-amber-500', safety_concern: 'bg-red-700',
}

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

/** Horizontal bar for distribution data */
function DistBar({ items, label }: { items: LabelCount[]; label: string }) {
  if (!items.length) return null
  const total = items.reduce((s, i) => s + i.count, 0)
  return (
    <div>
      <h4 className="text-xs text-slate-400 mb-1.5">{label}</h4>
      <div className="flex h-5 rounded-full overflow-hidden">
        {items.map((item) => {
          const pct = (item.count / total) * 100
          const color = CHURN_COLORS[item.label] ?? 'bg-slate-600'
          return (
            <div
              key={item.label}
              className={clsx(color, 'relative group')}
              style={{ width: `${pct}%`, minWidth: pct > 0 ? 4 : 0 }}
              title={`${item.label}: ${item.count} (${pct.toFixed(0)}%)`}
            >
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block whitespace-nowrap px-2 py-1 bg-slate-800 text-xs text-white rounded shadow-lg z-10">
                {item.label.replaceAll('_', ' ')}: {item.count} ({pct.toFixed(0)}%)
              </div>
            </div>
          )
        })}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 mt-1.5">
        {items.map((item) => (
          <span key={item.label} className="flex items-center gap-1 text-xs text-slate-400">
            <span className={clsx('w-2 h-2 rounded-full', CHURN_COLORS[item.label] ?? 'bg-slate-600')} />
            {item.label.replaceAll('_', ' ')} ({item.count})
          </span>
        ))}
      </div>
    </div>
  )
}

/** Pill badges for a distribution */
function BadgeRow({ items, label, colorFn }: {
  items: LabelCount[]
  label: string
  colorFn?: (val: string) => string
}) {
  if (!items.length) return null
  const defaultColor = (v: string) => CHURN_COLORS[v] ?? 'bg-slate-700'
  const getColor = colorFn ?? defaultColor
  return (
    <div>
      <h4 className="text-xs text-slate-400 mb-1.5">{label}</h4>
      <div className="flex flex-wrap gap-1.5">
        {items.map((item) => (
          <span
            key={item.label}
            className={clsx(
              'px-2.5 py-1 rounded-full text-xs font-medium text-white/90',
              getColor(item.label),
            )}
          >
            {item.label.replaceAll('_', ' ')} ({item.count})
          </span>
        ))}
      </div>
    </div>
  )
}

/** Section card wrapper */
function Card({ title, children, className }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={clsx('bg-slate-900/50 border border-slate-700/50 rounded-xl p-5', className)}>
      <h3 className="text-sm font-medium text-slate-300 mb-3">{title}</h3>
      {children}
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
      key: 'complaint',
      header: 'Pain',
      render: (r) => {
        const score = r.avg_complaint_score
        if (score == null) return <span className="text-slate-500">--</span>
        const color = score >= 7 ? 'text-red-400' : score >= 4 ? 'text-yellow-400' : 'text-green-400'
        return <span className={color}>{score.toFixed(1)}</span>
      },
      sortable: true,
      sortValue: (r) => r.avg_complaint_score ?? 0,
    },
    {
      key: 'praise',
      header: 'Loyalty',
      render: (r) => {
        const score = r.avg_praise_score
        if (score == null) return <span className="text-slate-500">--</span>
        const color = score >= 7 ? 'text-green-400' : score >= 4 ? 'text-cyan-400' : 'text-slate-400'
        return <span className={color}>{score.toFixed(1)}</span>
      },
      sortable: true,
      sortValue: (r) => r.avg_praise_score ?? 0,
    },
  ]

  // Sentiment chart data (top 15 by total mentions)
  const sentimentData = data.sentiment_aspects.slice(0, 15)

  // Loyalty pie data
  const loyaltyData = data.loyalty_breakdown.map((l) => ({ name: l.label, value: l.count }))

  // Repurchase rate
  const repurchTotal = data.repurchase_breakdown.reduce((s, i) => s + i.count, 0)
  const wouldRepurch = data.repurchase_breakdown.find((r) => r.label === 'true')?.count ?? 0
  const repurchPct = repurchTotal > 0 ? Math.round((wouldRepurch / repurchTotal) * 100) : null

  const fa = data.failure_analysis

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
        <div className="flex items-center gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white">{data.brand}</h1>
            <p className="text-sm text-slate-400 mt-1">
              {data.product_count} products, {data.total_reviews.toLocaleString()} reviews
              {data.avg_rating != null && ` | Avg ${data.avg_rating.toFixed(1)}`}
              {data.deep_review_count > 0 && (
                <span className="text-purple-400 ml-2">
                  {data.deep_review_count.toLocaleString()} deep-enriched
                </span>
              )}
            </p>
          </div>
          {data.brand_health != null && (
            <div className="flex flex-col items-center px-4 py-2 bg-slate-900/50 border border-slate-700/50 rounded-xl">
              <span className={clsx(
                'text-2xl font-bold',
                data.brand_health >= 60 ? 'text-emerald-400' : data.brand_health >= 40 ? 'text-amber-400' : 'text-red-400',
              )}>
                {data.brand_health}
              </span>
              <span className="text-[10px] text-slate-500 uppercase tracking-wider">Health</span>
            </div>
          )}
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
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Bar dataKey="positive" stackId="a" fill="#34d399" name="Positive" />
                  <Bar dataKey="negative" stackId="a" fill="#f87171" name="Negative" />
                  <Bar dataKey="mixed" stackId="a" fill="#fbbf24" name="Mixed" />
                  <Bar dataKey="neutral" stackId="a" fill="#64748b" name="Neutral" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="text-slate-500 text-sm py-8 text-center">No sentiment data available</p>
          )}

          {/* Top Positives */}
          {data.top_positives.length > 0 && (
            <Card title="Top Positive Aspects">
              <div className="flex flex-wrap gap-1.5">
                {data.top_positives.map((p) => (
                  <span key={p.aspect} className="px-2.5 py-1 bg-emerald-500/10 text-emerald-400 text-xs rounded-full">
                    {p.aspect} ({p.count})
                  </span>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}

      {/* Intelligence Tab */}
      {tab === 'intelligence' && (
        <div className="space-y-6">
          {/* ── Churn Health ── */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Repurchase Gauge */}
            <Card title="Repurchase Intent">
              {repurchPct !== null ? (
                <div className="text-center">
                  <span className={clsx(
                    'text-4xl font-bold',
                    repurchPct >= 60 ? 'text-emerald-400' : repurchPct >= 40 ? 'text-amber-400' : 'text-red-400',
                  )}>
                    {repurchPct}%
                  </span>
                  <p className="text-xs text-slate-500 mt-1">
                    {wouldRepurch} of {repurchTotal} would repurchase
                  </p>
                </div>
              ) : (
                <p className="text-slate-500 text-sm text-center">No data</p>
              )}
            </Card>

            {/* Safety */}
            <Card title="Safety Signals">
              <div className="text-center">
                {data.safety_flagged_count > 0 ? (
                  <>
                    <ShieldAlert className="h-8 w-8 text-red-400 mx-auto mb-1" />
                    <span className="text-3xl font-bold text-red-400">{data.safety_flagged_count}</span>
                    <p className="text-xs text-slate-500 mt-1">flagged reviews</p>
                  </>
                ) : (
                  <>
                    <span className="text-3xl font-bold text-emerald-400">0</span>
                    <p className="text-xs text-slate-500 mt-1">No safety concerns</p>
                  </>
                )}
              </div>
            </Card>

            {/* Consequence Severity */}
            <Card title="Consequence Severity">
              {data.consequence_breakdown.length > 0 ? (
                <div className="space-y-1.5">
                  {data.consequence_breakdown.map((c) => {
                    const total = data.consequence_breakdown.reduce((s, i) => s + i.count, 0)
                    const pct = Math.round((c.count / total) * 100)
                    const color = CHURN_COLORS[c.label] ?? 'bg-slate-600'
                    return (
                      <div key={c.label} className="flex items-center gap-2 text-xs">
                        <span className={clsx('w-2 h-2 rounded-full shrink-0', color)} />
                        <span className="text-slate-400 flex-1">{c.label.replaceAll('_', ' ')}</span>
                        <span className="text-slate-300 font-mono">{pct}%</span>
                      </div>
                    )
                  })}
                </div>
              ) : (
                <p className="text-slate-500 text-sm text-center">No data</p>
              )}
            </Card>
          </div>

          {/* Churn Distribution Bars */}
          <Card title="Churn & Loyalty Signals" className="space-y-5">
            <DistBar items={data.replacement_breakdown} label="Replacement Behavior" />
            <DistBar items={data.trajectory_breakdown} label="Sentiment Trajectory" />
            <DistBar items={data.switching_barrier} label="Switching Barrier" />
          </Card>

          {/* Loyalty Pie + Ecosystem */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {loyaltyData.length > 0 && (
              <Card title="Loyalty Depth">
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
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
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
              </Card>
            )}

            <Card title="Ecosystem & Engagement">
              <div className="space-y-4">
                <BadgeRow items={data.ecosystem_lock_in} label="Ecosystem Lock-in" colorFn={(v) =>
                  v === 'free' ? 'bg-emerald-600' : v === 'partial' ? 'bg-amber-500' : v === 'full' ? 'bg-red-500' : 'bg-slate-600'
                } />
                <BadgeRow items={data.amplification_intent} label="Amplification Intent" colorFn={(v) =>
                  v === 'quiet' ? 'bg-slate-600' : v === 'social' ? 'bg-red-500' : v === 'advocate' ? 'bg-emerald-500' : 'bg-slate-600'
                } />
                <BadgeRow items={data.openness_breakdown} label="Sentiment Openness" colorFn={(v) =>
                  v === 'open' ? 'bg-emerald-600' : 'bg-slate-600'
                } />
                <BadgeRow items={data.delay_breakdown} label="Review Delay" />
              </div>
            </Card>
          </div>

          {/* ── Failure Analysis ── */}
          {fa.failure_count > 0 && (
            <Card title="Failure Analysis" className="border-l-4 border-l-red-500/50">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <AlertTriangle className="h-4 w-4 text-red-400" />
                    <span className="text-2xl font-bold text-red-400">{fa.failure_count}</span>
                    <span className="text-xs text-slate-400">reported failures</span>
                  </div>
                  {fa.avg_dollar_lost != null && (
                    <div className="flex items-center gap-2">
                      <DollarSign className="h-4 w-4 text-amber-400" />
                      <span className="text-lg font-semibold text-amber-400">
                        ${fa.avg_dollar_lost.toLocaleString()}
                      </span>
                      <span className="text-xs text-slate-400">avg loss</span>
                    </div>
                  )}
                  {fa.total_dollar_lost != null && (
                    <p className="text-xs text-slate-500 mt-1">
                      Total reported: ${fa.total_dollar_lost.toLocaleString()}
                    </p>
                  )}
                </div>

                <div>
                  <h4 className="text-xs text-slate-400 mb-2">Top Failure Modes</h4>
                  <ul className="space-y-1">
                    {fa.top_failure_modes.slice(0, 7).map((f) => (
                      <li key={f.mode} className="flex items-center justify-between text-xs">
                        <span className="text-slate-300 truncate mr-2">{f.mode}</span>
                        <span className="text-red-400 font-mono shrink-0">{f.count}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h4 className="text-xs text-slate-400 mb-2">Top Failed Components</h4>
                  <ul className="space-y-1">
                    {fa.top_failed_components.slice(0, 7).map((c) => (
                      <li key={c.component} className="flex items-center justify-between text-xs">
                        <span className="text-slate-300 truncate mr-2">{c.component}</span>
                        <span className="text-red-400 font-mono shrink-0">{c.count}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </Card>
          )}

          {/* ── Feature Requests + Competitive Flows ── */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card title="Top Feature Requests">
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
            </Card>

            <Card title="Competitive Flows">
              {data.competitive_flows.length > 0 ? (
                <ul className="space-y-2 max-h-64 overflow-y-auto">
                  {data.competitive_flows.map((f, i) => (
                    <li key={i} className="flex items-center justify-between text-sm">
                      <span className="text-slate-300">
                        <span className="text-white font-medium">{f.brand}</span>
                        <span className={clsx(
                          'text-xs ml-1.5 px-1.5 py-0.5 rounded',
                          f.direction === 'switched_to' ? 'bg-red-500/10 text-red-400' :
                          f.direction === 'switched_from' ? 'bg-emerald-500/10 text-emerald-400' :
                          'bg-slate-700/50 text-slate-400',
                        )}>
                          {f.direction.replaceAll('_', ' ')}
                        </span>
                      </span>
                      <span className="text-cyan-400 font-mono text-xs">{f.count}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-slate-500 text-sm">No competitive flows found</p>
              )}
            </Card>
          </div>

          {/* ── Consideration Set ── */}
          {data.consideration_set.length > 0 && (
            <Card title="Consideration Set (Rejected Alternatives)">
              <ul className="space-y-2 max-h-64 overflow-y-auto">
                {data.consideration_set.map((c, i) => (
                  <li key={i} className="flex items-center justify-between text-sm">
                    <div>
                      <span className="text-white font-medium">{c.product}</span>
                      {c.top_reason && (
                        <span className="text-slate-500 text-xs ml-2">{c.top_reason}</span>
                      )}
                    </div>
                    <span className="text-cyan-400 font-mono text-xs shrink-0">{c.count}</span>
                  </li>
                ))}
              </ul>
            </Card>
          )}

          {/* ── Buyer Psychology ── */}
          <Card title="Buyer Psychology" className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <BadgeRow items={data.buyer_profile.expertise} label="Expertise Level" colorFn={(v) =>
                v === 'beginner' ? 'bg-slate-600' : v === 'intermediate' ? 'bg-blue-600' : v === 'expert' ? 'bg-purple-600' : 'bg-slate-600'
              } />
              <BadgeRow items={data.buyer_profile.budget} label="Budget Type" colorFn={(v) =>
                v === 'constrained' ? 'bg-red-600' : v === 'moderate' ? 'bg-amber-500' : v === 'premium' ? 'bg-emerald-500' : 'bg-slate-600'
              } />
              <BadgeRow items={data.buyer_profile.frustration} label="Frustration Threshold" colorFn={(v) =>
                v === 'low' ? 'bg-red-500' : v === 'medium' ? 'bg-amber-500' : v === 'high' ? 'bg-emerald-500' : 'bg-slate-600'
              } />
              <BadgeRow items={data.buyer_profile.intensity} label="Use Intensity" colorFn={(v) =>
                v === 'light' ? 'bg-emerald-600' : v === 'moderate' ? 'bg-amber-500' : v === 'heavy' ? 'bg-red-500' : 'bg-slate-600'
              } />
              <BadgeRow items={data.buyer_profile.research_depth} label="Research Depth" colorFn={(v) =>
                v === 'impulse' ? 'bg-red-500' : v === 'light' ? 'bg-amber-500' : v === 'moderate' ? 'bg-blue-500' : v === 'deep' ? 'bg-purple-600' : 'bg-slate-600'
              } />
              <BadgeRow items={data.buyer_profile.household} label="Household Type" />
              <BadgeRow items={data.buyer_profile.occasion} label="Purchase Occasion" />
              <BadgeRow items={data.buyer_profile.discovery_channel} label="Discovery Channel" />
              <BadgeRow items={data.buyer_profile.buyer_type} label="Buyer Type" />
            </div>

            {data.buyer_profile.price_sentiment.length > 0 && (
              <BadgeRow items={data.buyer_profile.price_sentiment} label="Price Sentiment" colorFn={(v) =>
                v === 'cheap' ? 'bg-emerald-500' : v === 'fair' ? 'bg-cyan-600' : v === 'expensive' ? 'bg-red-500' : 'bg-slate-600'
              } />
            )}

            {data.buyer_profile.professions.length > 0 && (
              <div>
                <h4 className="text-xs text-slate-400 mb-1.5">Top Professions</h4>
                <div className="flex flex-wrap gap-1.5">
                  {data.buyer_profile.professions.map((p) => (
                    <span key={p.profession} className="px-2 py-1 bg-slate-800 rounded text-xs text-slate-300">
                      {p.profession} ({p.count})
                    </span>
                  ))}
                </div>
              </div>
            )}
          </Card>
        </div>
      )}
    </div>
  )
}
