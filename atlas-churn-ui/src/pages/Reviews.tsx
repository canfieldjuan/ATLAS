import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search } from 'lucide-react'
import DataTable, { type Column } from '../components/DataTable'
import UrgencyBadge from '../components/UrgencyBadge'
import { fetchReviews } from '../api/client'
import type { ReviewSummary } from '../types'

export default function Reviews() {
  const navigate = useNavigate()
  const [reviews, setReviews] = useState<ReviewSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [vendor, setVendor] = useState('')
  const [company, setCompany] = useState('')
  const [minUrgency, setMinUrgency] = useState(0)
  const [churnOnly, setChurnOnly] = useState(false)

  useEffect(() => {
    async function load() {
      setLoading(true)
      try {
        const res = await fetchReviews({
          vendor_name: vendor || undefined,
          company: company || undefined,
          min_urgency: minUrgency || undefined,
          has_churn_intent: churnOnly || undefined,
          window_days: 365,
          limit: 100,
        })
        setReviews(res.reviews)
      } catch (err) {
        console.error('Reviews load error:', err)
      } finally {
        setLoading(false)
      }
    }
    const timer = setTimeout(load, 300)
    return () => clearTimeout(timer)
  }, [vendor, company, minUrgency, churnOnly])

  const columns: Column<ReviewSummary>[] = [
    {
      key: 'vendor',
      header: 'Vendor',
      render: (r) => <span className="text-white font-medium">{r.vendor_name}</span>,
      sortable: true,
      sortValue: (r) => r.vendor_name,
    },
    {
      key: 'company',
      header: 'Company',
      render: (r) => <span className="text-slate-300">{r.reviewer_company ?? '--'}</span>,
    },
    {
      key: 'rating',
      header: 'Rating',
      render: (r) => <span className="text-slate-300">{r.rating?.toFixed(1) ?? '--'}</span>,
      sortable: true,
      sortValue: (r) => r.rating ?? 0,
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
      header: 'Pain Category',
      render: (r) => <span className="text-slate-400">{r.pain_category ?? '--'}</span>,
    },
    {
      key: 'dm',
      header: 'DM',
      render: (r) =>
        r.decision_maker ? (
          <span className="text-cyan-400 text-xs font-medium">Yes</span>
        ) : (
          <span className="text-slate-500 text-xs">No</span>
        ),
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

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Reviews</h1>

      <div className="flex items-center gap-4 flex-wrap">
        <div className="relative flex-1 max-w-xs">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
          <input
            type="text"
            placeholder="Filter by vendor..."
            value={vendor}
            onChange={(e) => setVendor(e.target.value)}
            className="w-full pl-9 pr-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
          />
        </div>
        <div className="relative flex-1 max-w-xs">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
          <input
            type="text"
            placeholder="Filter by company..."
            value={company}
            onChange={(e) => setCompany(e.target.value)}
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
            className="w-20 accent-cyan-400"
          />
          <span className="text-white w-6">{minUrgency}</span>
        </div>
        <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer">
          <input
            type="checkbox"
            checked={churnOnly}
            onChange={(e) => setChurnOnly(e.target.checked)}
            className="accent-cyan-400"
          />
          Churn intent only
        </label>
      </div>

      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-slate-500">Loading...</div>
        ) : (
          <DataTable
            columns={columns}
            data={reviews}
            onRowClick={(r) => navigate(`/reviews/${r.id}`)}
            emptyMessage="No reviews match your filters"
          />
        )}
      </div>
    </div>
  )
}
