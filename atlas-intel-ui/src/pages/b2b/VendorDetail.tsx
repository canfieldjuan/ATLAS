import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, Activity, AlertTriangle, Target, BarChart3 } from 'lucide-react'
import StatCard from '../../components/StatCard'
import DataTable, { type Column } from '../../components/DataTable'
import { fetchVendorDetail, type VendorDetail as VendorDetailType } from '../../api/b2bClient'

export default function VendorDetail() {
  const { vendorName } = useParams<{ vendorName: string }>()
  const navigate = useNavigate()
  const [detail, setDetail] = useState<VendorDetailType | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!vendorName) return
    setLoading(true)
    fetchVendorDetail(vendorName)
      .then(d => setDetail(d))
      .catch(err => setError(err instanceof Error ? err.message : 'Failed to load vendor'))
      .finally(() => setLoading(false))
  }, [vendorName])

  if (error) {
    return (
      <div className="space-y-4">
        <button onClick={() => navigate(-1)} className="flex items-center gap-1 text-sm text-slate-400 hover:text-white">
          <ArrowLeft className="h-4 w-4" /> Back
        </button>
        <p className="text-red-400">{error}</p>
      </div>
    )
  }

  const signal = detail?.churn_signal
  const skeleton = loading

  const hiColumns: Column<{ company: string; urgency: number; pain: string | null }>[] = [
    {
      key: 'company',
      header: 'Company',
      render: r => <span className="text-white font-medium">{r.company}</span>,
    },
    {
      key: 'urgency',
      header: 'Urgency',
      sortable: true,
      sortValue: r => r.urgency,
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

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <button onClick={() => navigate(-1)} className="text-slate-400 hover:text-white">
          <ArrowLeft className="h-5 w-5" />
        </button>
        <h1 className="text-2xl font-bold text-white">{vendorName}</h1>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Urgency Score"
          value={signal?.avg_urgency_score?.toFixed(1) ?? '--'}
          icon={<AlertTriangle className="h-5 w-5" />}
          skeleton={skeleton}
        />
        <StatCard
          label="Churn Intent"
          value={signal?.churn_intent_count ?? 0}
          icon={<Activity className="h-5 w-5" />}
          skeleton={skeleton}
        />
        <StatCard
          label="Total Reviews"
          value={detail?.review_counts?.total ?? 0}
          icon={<BarChart3 className="h-5 w-5" />}
          sub={`${detail?.review_counts?.enriched ?? 0} enriched`}
          skeleton={skeleton}
        />
        <StatCard
          label="NPS Proxy"
          value={signal?.nps_proxy?.toFixed(0) ?? '--'}
          icon={<Target className="h-5 w-5" />}
          skeleton={skeleton}
        />
      </div>

      {/* Pain distribution */}
      {detail?.pain_distribution && detail.pain_distribution.length > 0 && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-slate-300 mb-3">Pain Categories</h2>
          <div className="space-y-2">
            {detail.pain_distribution.map(p => {
              const maxCount = Math.max(...detail.pain_distribution.map(x => x.count))
              const pct = maxCount > 0 ? (p.count / maxCount) * 100 : 0
              return (
                <div key={p.pain_category} className="flex items-center gap-3">
                  <span className="w-32 text-sm text-slate-400 truncate">{p.pain_category}</span>
                  <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-cyan-500/60 rounded-full" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="text-xs text-slate-500 w-8 text-right">{p.count}</span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Quotable evidence */}
      {signal?.quotable_evidence && Array.isArray(signal.quotable_evidence) && signal.quotable_evidence.length > 0 && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-slate-300 mb-3">Quotable Evidence</h2>
          <div className="space-y-2">
            {(signal.quotable_evidence as string[]).slice(0, 5).map((q, i) => (
              <blockquote key={i} className="border-l-2 border-cyan-500/30 pl-3 text-sm text-slate-400 italic">
                "{q}"
              </blockquote>
            ))}
          </div>
        </div>
      )}

      {/* Top competitors */}
      {signal?.top_competitors && Array.isArray(signal.top_competitors) && signal.top_competitors.length > 0 && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-slate-300 mb-3">Top Competitors Mentioned</h2>
          <div className="flex flex-wrap gap-2">
            {(signal.top_competitors as string[]).map((c, i) => (
              <span key={i} className="px-2 py-1 bg-amber-900/20 text-amber-400 text-xs rounded-full">
                {typeof c === 'string' ? c : JSON.stringify(c)}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* High-intent companies */}
      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
        <h2 className="text-sm font-semibold text-slate-300 mb-3">High-Intent Companies</h2>
        <DataTable
          columns={hiColumns}
          data={detail?.high_intent_companies ?? []}
          onRowClick={r => navigate(`/b2b/leads/${encodeURIComponent(r.company)}`)}
          skeletonRows={skeleton ? 3 : undefined}
          emptyMessage="No high-intent companies found"
        />
      </div>
    </div>
  )
}
