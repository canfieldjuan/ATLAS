import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { RefreshCw, Search, X, ChevronLeft, ChevronRight } from 'lucide-react'
import { clsx } from 'clsx'
import DataTable, { type Column } from '../components/DataTable'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import { fetchBrands, type BrandSummary } from '../api/client'

const PAGE_SIZE = 50

export default function Brands() {
  const navigate = useNavigate()
  const [search, setSearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [minReviews, setMinReviews] = useState(0)
  const [sortBy, setSortBy] = useState('review_count')
  const [page, setPage] = useState(0)
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined)

  useEffect(() => {
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setDebouncedSearch(search), 300)
    return () => clearTimeout(timerRef.current)
  }, [search])

  // Reset page when filters change
  useEffect(() => { setPage(0) }, [debouncedSearch, minReviews, sortBy])

  const { data, loading, error, refresh, refreshing } = useApiData(
    () => fetchBrands({
      search: debouncedSearch || undefined,
      min_reviews: minReviews || undefined,
      sort_by: sortBy,
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
    }),
    [debouncedSearch, minReviews, sortBy, page],
  )

  const brands = data?.brands ?? []
  const totalCount = data?.total_count ?? 0
  const totalPages = Math.ceil(totalCount / PAGE_SIZE)
  const hasFilters = search || minReviews > 0

  const columns: Column<BrandSummary>[] = [
    {
      key: 'brand',
      header: 'Brand',
      render: (r) => <span className="text-white font-medium">{r.brand}</span>,
    },
    {
      key: 'products',
      header: 'Products',
      render: (r) => <span className="text-slate-300">{r.product_count}</span>,
      sortable: true,
      sortValue: (r) => r.product_count,
    },
    {
      key: 'reviews',
      header: 'Reviews',
      render: (r) => <span className="text-slate-300">{r.review_count.toLocaleString()}</span>,
      sortable: true,
      sortValue: (r) => r.review_count,
    },
    {
      key: 'avg_rating',
      header: 'Avg Rating',
      render: (r) => (
        <span className="text-slate-300">{r.avg_rating?.toFixed(1) ?? '--'}</span>
      ),
      sortable: true,
      sortValue: (r) => r.avg_rating ?? 0,
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
    {
      key: 'safety',
      header: 'Safety',
      render: (r) =>
        r.safety_count > 0 ? (
          <span className="px-2 py-0.5 bg-red-500/10 text-red-400 text-xs rounded-full font-medium">
            {r.safety_count}
          </span>
        ) : (
          <span className="text-slate-500">--</span>
        ),
      sortable: true,
      sortValue: (r) => r.safety_count,
    },
  ]

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Brands</h1>
          {!loading && totalCount > 0 && (
            <p className="text-xs text-slate-400 mt-0.5">
              Showing {(page * PAGE_SIZE + 1).toLocaleString()}-{Math.min((page + 1) * PAGE_SIZE, totalCount).toLocaleString()} of {totalCount.toLocaleString()}
            </p>
          )}
        </div>
        <button
          onClick={refresh}
          disabled={refreshing}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
          Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-end gap-4">
        <div className="flex-1 min-w-[200px]">
          <label className="block text-xs text-slate-400 mb-1">Search brands</label>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Brand name..."
              className="w-full pl-9 pr-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
            />
          </div>
        </div>
        <div className="w-40">
          <label className="block text-xs text-slate-400 mb-1">Min reviews: {minReviews}</label>
          <input
            type="range"
            min={0}
            max={100}
            value={minReviews}
            onChange={(e) => setMinReviews(Number(e.target.value))}
            className="w-full accent-cyan-500"
          />
        </div>
        <div className="w-40">
          <label className="block text-xs text-slate-400 mb-1">Sort by</label>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="w-full px-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white focus:outline-none focus:border-cyan-500/50"
          >
            <option value="review_count">Review Count</option>
            <option value="avg_rating">Avg Rating</option>
            <option value="avg_complaint_score">Pain Score</option>
            <option value="avg_praise_score">Loyalty Score</option>
            <option value="safety_count">Safety Flags</option>
            <option value="brand">Brand Name</option>
          </select>
        </div>
        {hasFilters && (
          <button
            onClick={() => {
              setSearch('')
              setMinReviews(0)
            }}
            className="flex items-center gap-1 px-3 py-2 text-sm text-slate-400 hover:text-white transition-colors"
          >
            <X className="h-3 w-3" />
            Clear
          </button>
        )}
      </div>

      {/* Table */}
      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl overflow-hidden">
        {loading ? (
          <DataTable columns={columns} data={[]} skeletonRows={10} />
        ) : (
          <DataTable
            columns={columns}
            data={brands}
            onRowClick={(r) => navigate(`/brands/${encodeURIComponent(r.brand)}`)}
            emptyMessage="No brands found"
          />
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <button
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0 || loading}
            className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <ChevronLeft className="h-4 w-4" />
            Previous
          </button>
          <span className="text-sm text-slate-400">
            Page {page + 1} of {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1 || loading}
            className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Next
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      )}
    </div>
  )
}
