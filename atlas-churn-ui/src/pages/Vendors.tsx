import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search } from 'lucide-react'
import DataTable, { type Column } from '../components/DataTable'
import UrgencyBadge from '../components/UrgencyBadge'
import { fetchSignals } from '../api/client'
import type { ChurnSignal } from '../types'

export default function Vendors() {
  const navigate = useNavigate()
  const [signals, setSignals] = useState<ChurnSignal[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [minUrgency, setMinUrgency] = useState(0)
  const [category, setCategory] = useState('')

  useEffect(() => {
    async function load() {
      setLoading(true)
      try {
        const res = await fetchSignals({
          vendor_name: search || undefined,
          min_urgency: minUrgency || undefined,
          category: category || undefined,
          limit: 100,
        })
        setSignals(res.signals)
      } catch (err) {
        console.error('Vendors load error:', err)
      } finally {
        setLoading(false)
      }
    }
    const timer = setTimeout(load, 300)
    return () => clearTimeout(timer)
  }, [search, minUrgency, category])

  const categories = [...new Set(signals.map((s) => s.product_category).filter(Boolean))]

  const columns: Column<ChurnSignal>[] = [
    {
      key: 'vendor',
      header: 'Vendor',
      render: (r) => <span className="text-white font-medium">{r.vendor_name}</span>,
      sortable: true,
      sortValue: (r) => r.vendor_name,
    },
    {
      key: 'category',
      header: 'Category',
      render: (r) => <span className="text-slate-400">{r.product_category ?? '--'}</span>,
    },
    {
      key: 'urgency',
      header: 'Urgency',
      render: (r) => <UrgencyBadge score={r.avg_urgency_score} />,
      sortable: true,
      sortValue: (r) => r.avg_urgency_score,
    },
    {
      key: 'nps',
      header: 'NPS Proxy',
      render: (r) => (
        <span className="text-slate-300">
          {r.nps_proxy !== null ? r.nps_proxy.toFixed(1) : '--'}
        </span>
      ),
      sortable: true,
      sortValue: (r) => r.nps_proxy ?? 0,
    },
    {
      key: 'churn_pct',
      header: 'Churn Intent',
      render: (r) => {
        const pct = r.total_reviews > 0
          ? Math.round((r.churn_intent_count / r.total_reviews) * 100)
          : 0
        return <span className="text-slate-300">{pct}%</span>
      },
      sortable: true,
      sortValue: (r) => r.total_reviews > 0 ? r.churn_intent_count / r.total_reviews : 0,
    },
    {
      key: 'dm_rate',
      header: 'DM Churn',
      render: (r) => (
        <span className="text-slate-300">
          {r.decision_maker_churn_rate !== null
            ? `${(r.decision_maker_churn_rate * 100).toFixed(0)}%`
            : '--'}
        </span>
      ),
      sortable: true,
      sortValue: (r) => r.decision_maker_churn_rate ?? 0,
    },
    {
      key: 'reviews',
      header: 'Reviews',
      render: (r) => <span className="text-slate-400">{r.total_reviews}</span>,
      sortable: true,
      sortValue: (r) => r.total_reviews,
    },
  ]

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Vendor Signals</h1>

      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
          <input
            type="text"
            placeholder="Search vendors..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
          />
        </div>
        <div className="flex items-center gap-2 text-sm text-slate-400">
          <label>Min Urgency</label>
          <input
            type="range"
            min={0}
            max={10}
            step={0.5}
            value={minUrgency}
            onChange={(e) => setMinUrgency(Number(e.target.value))}
            className="w-24 accent-cyan-400"
          />
          <span className="text-white w-6">{minUrgency}</span>
        </div>
        {categories.length > 0 && (
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            className="bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white px-3 py-2 focus:outline-none focus:border-cyan-500/50"
          >
            <option value="">All Categories</option>
            {categories.map((c) => (
              <option key={c} value={c!}>{c}</option>
            ))}
          </select>
        )}
      </div>

      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-slate-500">Loading...</div>
        ) : (
          <DataTable
            columns={columns}
            data={signals}
            onRowClick={(r) => navigate(`/vendors/${encodeURIComponent(r.vendor_name)}`)}
            emptyMessage="No signals match your filters"
          />
        )}
      </div>
    </div>
  )
}
