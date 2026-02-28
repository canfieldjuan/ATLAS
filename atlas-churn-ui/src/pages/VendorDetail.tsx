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
} from 'recharts'
import UrgencyBadge from '../components/UrgencyBadge'
import DataTable, { type Column } from '../components/DataTable'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import { fetchVendorProfile, fetchReviews } from '../api/client'
import type { VendorProfile, ReviewSummary } from '../types'

interface VendorData {
  profile: VendorProfile
  reviews: ReviewSummary[]
}

function DetailSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="h-4 w-32 bg-slate-700/50 rounded" />
      <div className="flex items-center justify-between">
        <div>
          <div className="h-7 w-48 bg-slate-700/50 rounded mb-2" />
          <div className="h-4 w-32 bg-slate-700/50 rounded" />
        </div>
        <div className="h-10 w-16 bg-slate-700/50 rounded" />
      </div>
      <div className="h-10 w-64 bg-slate-700/50 rounded" />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 h-64" />
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 h-64" />
      </div>
    </div>
  )
}

export default function VendorDetail() {
  const { name } = useParams<{ name: string }>()
  const navigate = useNavigate()
  const [tab, setTab] = useState<'overview' | 'reviews' | 'companies'>('overview')

  const { data, loading, error, refresh, refreshing } = useApiData<VendorData>(
    async () => {
      if (!name) throw new Error('Missing vendor name')
      const [profile, revRes] = await Promise.all([
        fetchVendorProfile(name),
        fetchReviews({ vendor_name: name, limit: 50, window_days: 365 }),
      ])
      return { profile, reviews: revRes.reviews }
    },
    [name],
  )

  if (error) return <PageError error={error} onRetry={refresh} />
  if (loading) return <DetailSkeleton />

  const profile = data?.profile
  if (!profile) return <PageError error={new Error('Vendor not found')} />

  const signal = profile.churn_signal
  const painData = profile.pain_distribution.map((p) => ({
    name: p.pain_category,
    count: p.count,
  }))

  const reviewColumns: Column<ReviewSummary>[] = [
    {
      key: 'company',
      header: 'Company',
      render: (r) => <span className="text-white">{r.reviewer_company ?? '--'}</span>,
    },
    {
      key: 'urgency',
      header: 'Urgency',
      render: (r) => <UrgencyBadge score={r.urgency_score} />,
      sortable: true,
      sortValue: (r) => r.urgency_score ?? 0,
    },
    {
      key: 'pain',
      header: 'Pain',
      render: (r) => <span className="text-slate-400">{r.pain_category ?? '--'}</span>,
    },
    {
      key: 'rating',
      header: 'Rating',
      render: (r) => <span className="text-slate-300">{r.rating?.toFixed(1) ?? '--'}</span>,
    },
    {
      key: 'intent',
      header: 'Intent',
      render: (r) =>
        r.intent_to_leave ? (
          <span className="text-red-400 text-xs font-medium">Leaving</span>
        ) : (
          <span className="text-slate-500 text-xs">--</span>
        ),
    },
  ]

  const companyColumns: Column<{ company: string; urgency: number; pain: string | null }>[] = [
    {
      key: 'company',
      header: 'Company',
      render: (r) => <span className="text-white font-medium">{r.company}</span>,
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
  ]

  return (
    <div className="space-y-6">
      <button
        onClick={() => navigate('/vendors')}
        className="flex items-center gap-1 text-sm text-slate-400 hover:text-white transition-colors"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to Vendors
      </button>

      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">{profile.vendor_name}</h1>
          <p className="text-sm text-slate-400 mt-1">
            {profile.review_counts.total} reviews ({profile.review_counts.enriched} enriched)
          </p>
        </div>
        <div className="flex items-center gap-4">
          {signal && (
            <div className="text-right">
              <p className="text-xs text-slate-400">Urgency Score</p>
              <p className="text-3xl font-bold text-white">{signal.avg_urgency_score.toFixed(1)}</p>
            </div>
          )}
          <button
            onClick={refresh}
            disabled={refreshing}
            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
          </button>
        </div>
      </div>

      <div className="flex gap-1 border-b border-slate-700/50">
        {(['overview', 'reviews', 'companies'] as const).map((t) => (
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

      {tab === 'overview' && signal && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
              <h3 className="text-sm font-medium text-slate-300 mb-3">Key Metrics</h3>
              <dl className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <dt className="text-slate-400">NPS Proxy</dt>
                  <dd className="text-white">{signal.nps_proxy?.toFixed(1) ?? '--'}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-slate-400">Price Complaint Rate</dt>
                  <dd className="text-white">
                    {signal.price_complaint_rate !== null
                      ? `${(signal.price_complaint_rate * 100).toFixed(0)}%`
                      : '--'}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-slate-400">DM Churn Rate</dt>
                  <dd className="text-white">
                    {signal.decision_maker_churn_rate !== null
                      ? `${(signal.decision_maker_churn_rate * 100).toFixed(0)}%`
                      : '--'}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-slate-400">Churn Intent Count</dt>
                  <dd className="text-white">{signal.churn_intent_count}</dd>
                </div>
              </dl>
            </div>
            {signal.top_competitors && signal.top_competitors.length > 0 && (
              <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
                <h3 className="text-sm font-medium text-slate-300 mb-3">Top Competitors</h3>
                <ul className="space-y-1">
                  {signal.top_competitors.map((c, i) => (
                    <li key={i} className="text-sm text-slate-300">{typeof c === 'string' ? c : JSON.stringify(c)}</li>
                  ))}
                </ul>
              </div>
            )}
            {signal.top_feature_gaps && signal.top_feature_gaps.length > 0 && (
              <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
                <h3 className="text-sm font-medium text-slate-300 mb-3">Feature Gaps</h3>
                <ul className="space-y-1">
                  {signal.top_feature_gaps.map((g, i) => (
                    <li key={i} className="text-sm text-slate-300">{typeof g === 'string' ? g : JSON.stringify(g)}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
          <div className="space-y-4">
            {painData.length > 0 && (
              <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
                <h3 className="text-sm font-medium text-slate-300 mb-3">Pain Distribution</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={painData} layout="vertical" margin={{ left: 80 }}>
                    <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#334155' }} />
                    <YAxis
                      type="category"
                      dataKey="name"
                      tick={{ fill: '#94a3b8', fontSize: 11 }}
                      axisLine={{ stroke: '#334155' }}
                      width={80}
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
                    <Bar dataKey="count" fill="#22d3ee" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
            {signal.quotable_evidence && signal.quotable_evidence.length > 0 && (
              <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
                <h3 className="text-sm font-medium text-slate-300 mb-3">Quotable Evidence</h3>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {signal.quotable_evidence.map((q: string, i: number) => (
                    <blockquote
                      key={i}
                      className="text-sm text-slate-300 italic border-l-2 border-cyan-500/50 pl-3"
                    >
                      {typeof q === 'string' ? q : JSON.stringify(q)}
                    </blockquote>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {tab === 'reviews' && (
        <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl overflow-hidden">
          <DataTable
            columns={reviewColumns}
            data={data?.reviews ?? []}
            onRowClick={(r) => navigate(`/reviews/${r.id}`)}
            emptyMessage="No enriched reviews for this vendor"
          />
        </div>
      )}

      {tab === 'companies' && (
        <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl overflow-hidden">
          <DataTable
            columns={companyColumns}
            data={profile.high_intent_companies}
            emptyMessage="No high-intent companies detected"
          />
        </div>
      )}
    </div>
  )
}
