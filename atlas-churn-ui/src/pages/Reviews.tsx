import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, RefreshCw, X, Loader2, Download } from 'lucide-react'
import { clsx } from 'clsx'
import DataTable, { type Column } from '../components/DataTable'
import UrgencyBadge from '../components/UrgencyBadge'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import { fetchReviews, downloadCsv } from '../api/client'
import type { ReviewSummary } from '../types'

const REVIEW_WINDOW_DAYS = 365

export default function Reviews() {
  const navigate = useNavigate()
  const [vendor, setVendor] = useState('')
  const [debouncedVendor, setDebouncedVendor] = useState('')
  const [company, setCompany] = useState('')
  const [debouncedCompany, setDebouncedCompany] = useState('')
  const [minUrgency, setMinUrgency] = useState(0)
  const [churnOnly, setChurnOnly] = useState(false)
  const [debouncePending, setDebouncePending] = useState(false)

  useEffect(() => {
    setDebouncePending(true)
    const timer = setTimeout(() => {
      setDebouncedVendor(vendor)
      setDebouncedCompany(company)
      setDebouncePending(false)
    }, 300)
    return () => clearTimeout(timer)
  }, [vendor, company])

  const { data, loading, error, refresh, refreshing } = useApiData(
    () =>
      fetchReviews({
        vendor_name: debouncedVendor || undefined,
        company: debouncedCompany || undefined,
        min_urgency: minUrgency || undefined,
        has_churn_intent: churnOnly || undefined,
        window_days: REVIEW_WINDOW_DAYS,
        limit: 100,
      }),
    [debouncedVendor, debouncedCompany, minUrgency, churnOnly],
  )

  const reviews = data?.reviews ?? []
  const hasFilters = vendor !== '' || company !== '' || minUrgency > 0 || churnOnly

  function clearFilters() {
    setVendor('')
    setDebouncedVendor('')
    setCompany('')
    setDebouncedCompany('')
    setMinUrgency(0)
    setChurnOnly(false)
  }

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

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Reviews</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={() =>
              downloadCsv('/export/reviews', {
                vendor_name: debouncedVendor || undefined,
                company: debouncedCompany || undefined,
                min_urgency: minUrgency || undefined,
                has_churn_intent: churnOnly || undefined,
                window_days: REVIEW_WINDOW_DAYS,
              })
            }
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors"
          >
            <Download className="h-4 w-4" />
            Export CSV
          </button>
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

      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 flex-wrap">
        <div className="relative flex-1 max-w-xs w-full">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
          <input
            type="text"
            placeholder="Filter by vendor..."
            value={vendor}
            onChange={(e) => setVendor(e.target.value)}
            className="w-full pl-9 pr-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
          />
        </div>
        <div className="relative flex-1 max-w-xs w-full">
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
        {hasFilters && (
          <button
            onClick={clearFilters}
            className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors"
          >
            <X className="h-3 w-3" />
            Clear filters
          </button>
        )}
      </div>

      <div className="flex items-center gap-2 text-sm text-slate-400">
        {debouncePending || loading ? (
          <>
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Searching...
          </>
        ) : (
          <span>{reviews.length} review{reviews.length !== 1 ? 's' : ''} found</span>
        )}
      </div>

      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl overflow-hidden">
        {loading ? (
          <DataTable columns={columns} data={[]} skeletonRows={8} />
        ) : (
          <DataTable
            columns={columns}
            data={reviews}
            onRowClick={(r) => navigate(`/reviews/${r.id}`)}
            emptyMessage="No reviews match your filters"
            emptyAction={hasFilters ? { label: 'Clear all filters', onClick: clearFilters } : undefined}
          />
        )}
      </div>
    </div>
  )
}
