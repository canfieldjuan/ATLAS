import { useState, useEffect, useRef } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { RefreshCw, Search, X } from 'lucide-react'
import { clsx } from 'clsx'
import DataTable, { type Column } from '../components/DataTable'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import { fetchReviews, type ReviewSummary } from '../api/client'

export default function Reviews() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()

  const [brand, setBrand] = useState(searchParams.get('brand') ?? '')
  const [debouncedBrand, setDebouncedBrand] = useState(brand)
  const [asin, setAsin] = useState(searchParams.get('asin') ?? '')
  const [searchText, setSearchText] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [minRating, setMinRating] = useState(0)
  const [maxRating, setMaxRating] = useState(5)
  const [rootCause, setRootCause] = useState('')
  const [hasComparisons, setHasComparisons] = useState<boolean | undefined>(
    searchParams.get('has_comparisons') === 'true' ? true : undefined
  )
  const [hasFeatures, setHasFeatures] = useState<boolean | undefined>(undefined)
  const [sortBy, setSortBy] = useState('imported_at')
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined)

  useEffect(() => {
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => {
      setDebouncedBrand(brand)
      setDebouncedSearch(searchText)
    }, 300)
    return () => clearTimeout(timerRef.current)
  }, [brand, searchText])

  const { data, loading, error, refresh, refreshing } = useApiData(
    () => fetchReviews({
      brand: debouncedBrand || undefined,
      asin: asin || undefined,
      search: debouncedSearch || undefined,
      min_rating: minRating > 0 ? minRating : undefined,
      max_rating: maxRating < 5 ? maxRating : undefined,
      root_cause: rootCause || undefined,
      has_comparisons: hasComparisons,
      has_feature_requests: hasFeatures,
      sort_by: sortBy,
      limit: 100,
    }),
    [debouncedBrand, asin, debouncedSearch, minRating, maxRating, rootCause, hasComparisons, hasFeatures, sortBy],
  )

  const reviews = data?.reviews ?? []
  const hasFilters = brand || asin || searchText || minRating > 0 || maxRating < 5 || rootCause || hasComparisons !== undefined || hasFeatures !== undefined

  const columns: Column<ReviewSummary>[] = [
    {
      key: 'brand',
      header: 'Brand',
      render: (r) => <span className="text-white font-medium">{r.brand ?? '--'}</span>,
    },
    {
      key: 'asin',
      header: 'ASIN',
      render: (r) => <span className="text-slate-400 font-mono text-xs">{r.asin}</span>,
    },
    {
      key: 'rating',
      header: 'Rating',
      render: (r) => {
        if (r.rating == null) return <span className="text-slate-500">--</span>
        const color = r.rating >= 4 ? 'text-green-400' : r.rating >= 3 ? 'text-yellow-400' : 'text-red-400'
        return <span className={color}>{r.rating.toFixed(1)}</span>
      },
      sortable: true,
      sortValue: (r) => r.rating ?? 0,
    },
    {
      key: 'root_cause',
      header: 'Root Cause',
      render: (r) => <span className="text-slate-300">{r.root_cause ?? '--'}</span>,
    },
    {
      key: 'pain',
      header: 'Pain',
      render: (r) => {
        const score = r.pain_score
        if (score == null) return <span className="text-slate-500">--</span>
        const color = score >= 7 ? 'text-red-400' : score >= 4 ? 'text-yellow-400' : 'text-green-400'
        return <span className={color}>{score.toFixed(1)}</span>
      },
      sortable: true,
      sortValue: (r) => r.pain_score ?? 0,
    },
    {
      key: 'summary',
      header: 'Summary',
      render: (r) => (
        <span className="text-slate-400 truncate max-w-[250px] block">{r.summary ?? '--'}</span>
      ),
    },
  ]

  const clearFilters = () => {
    setBrand('')
    setAsin('')
    setSearchText('')
    setMinRating(0)
    setMaxRating(5)
    setRootCause('')
    setHasComparisons(undefined)
    setHasFeatures(undefined)
  }

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Reviews</h1>
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
      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-4 space-y-3">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <div>
            <label className="block text-xs text-slate-400 mb-1">Brand</label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
              <input
                type="text"
                value={brand}
                onChange={(e) => setBrand(e.target.value)}
                placeholder="Brand..."
                className="w-full pl-9 pr-3 py-1.5 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
              />
            </div>
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">ASIN</label>
            <input
              type="text"
              value={asin}
              onChange={(e) => setAsin(e.target.value)}
              placeholder="ASIN..."
              className="w-full px-3 py-1.5 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Root Cause</label>
            <input
              type="text"
              value={rootCause}
              onChange={(e) => setRootCause(e.target.value)}
              placeholder="e.g. quality, design..."
              className="w-full px-3 py-1.5 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Search text</label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
              <input
                type="text"
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                placeholder="Full text..."
                className="w-full pl-9 pr-3 py-1.5 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
              />
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-end gap-4">
          <div className="w-36">
            <label className="block text-xs text-slate-400 mb-1">Min rating: {minRating}</label>
            <input
              type="range"
              min={0}
              max={5}
              step={0.5}
              value={minRating}
              onChange={(e) => setMinRating(Number(e.target.value))}
              className="w-full accent-cyan-500"
            />
          </div>
          <div className="w-36">
            <label className="block text-xs text-slate-400 mb-1">Max rating: {maxRating}</label>
            <input
              type="range"
              min={0}
              max={5}
              step={0.5}
              value={maxRating}
              onChange={(e) => setMaxRating(Number(e.target.value))}
              className="w-full accent-cyan-500"
            />
          </div>
          <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer">
            <input
              type="checkbox"
              checked={hasComparisons === true}
              onChange={(e) => setHasComparisons(e.target.checked ? true : undefined)}
              className="accent-cyan-500"
            />
            Has comparisons
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer">
            <input
              type="checkbox"
              checked={hasFeatures === true}
              onChange={(e) => setHasFeatures(e.target.checked ? true : undefined)}
              className="accent-cyan-500"
            />
            Has feature requests
          </label>
          <div className="w-36">
            <label className="block text-xs text-slate-400 mb-1">Sort by</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="w-full px-3 py-1.5 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white focus:outline-none focus:border-cyan-500/50"
            >
              <option value="imported_at">Newest</option>
              <option value="rating">Rating</option>
              <option value="pain_score">Pain Score</option>
            </select>
          </div>
          {hasFilters && (
            <button
              onClick={clearFilters}
              className="flex items-center gap-1 px-3 py-1.5 text-sm text-slate-400 hover:text-white transition-colors"
            >
              <X className="h-3 w-3" />
              Clear all
            </button>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl overflow-hidden">
        {loading ? (
          <DataTable columns={columns} data={[]} skeletonRows={10} />
        ) : (
          <DataTable
            columns={columns}
            data={reviews}
            onRowClick={(r) => navigate(`/reviews/${r.id}`)}
            emptyMessage="No reviews found"
          />
        )}
      </div>
    </div>
  )
}
