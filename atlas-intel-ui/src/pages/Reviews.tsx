import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { RefreshCw, ChevronLeft, ChevronRight } from 'lucide-react'
import { clsx } from 'clsx'
import DataTable, { type Column } from '../components/DataTable'
import FilterBar, { FilterSearch, FilterSelect } from '../components/FilterBar'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import useFilterParams from '../hooks/useFilterParams'
import useCategories from '../hooks/useCategories'
import { fetchReviews, type ReviewSummary } from '../api/client'

const PAGE_SIZE = 100

const FILTER_CONFIG = {
  brand: { type: 'string' as const, label: 'Brand' },
  asin: { type: 'string' as const, label: 'ASIN' },
  source_category: { type: 'string' as const, label: 'Category' },
  sort_by: { type: 'string' as const, label: 'Sort', default: 'imported_at' },
  root_cause: { type: 'string' as const, label: 'Root Cause' },
  search: { type: 'string' as const, label: 'Search' },
  severity: { type: 'string' as const, label: 'Severity' },
  enrichment_status: { type: 'string' as const, label: 'Status' },
  imported_after: { type: 'string' as const, label: 'From' },
  imported_before: { type: 'string' as const, label: 'To' },
  min_rating: { type: 'number' as const, label: 'Min Rating', default: 0 },
  max_rating: { type: 'number' as const, label: 'Max Rating', default: 5 },
  has_comparisons: { type: 'string' as const, label: 'Comparisons' },
  has_feature_requests: { type: 'string' as const, label: 'Features' },
}

type Filters = {
  brand: string
  asin: string
  source_category: string
  sort_by: string
  root_cause: string
  search: string
  severity: string
  enrichment_status: string
  imported_after: string
  imported_before: string
  min_rating: number
  max_rating: number
  has_comparisons: string
  has_feature_requests: string
}

export default function Reviews() {
  const navigate = useNavigate()
  const { categories } = useCategories()
  const { filters, setFilter, clearFilter, clearAll, activeFilterEntries } =
    useFilterParams<Filters>(FILTER_CONFIG)
  const [page, setPage] = useState(0)
  const filtersKey = JSON.stringify(filters)

  // Reset to page 0 when any filter changes
  useEffect(() => {
    setPage(0)
  }, [filtersKey])

  const { data, loading, error, refresh, refreshing } = useApiData(
    () =>
      fetchReviews({
        brand: filters.brand || undefined,
        asin: filters.asin || undefined,
        source_category: filters.source_category || undefined,
        search: filters.search || undefined,
        root_cause: filters.root_cause || undefined,
        severity: filters.severity || undefined,
        enrichment_status: filters.enrichment_status || undefined,
        imported_after: filters.imported_after || undefined,
        imported_before: filters.imported_before || undefined,
        min_rating: filters.min_rating > 0 ? filters.min_rating : undefined,
        max_rating: filters.max_rating < 5 ? filters.max_rating : undefined,
        has_comparisons: filters.has_comparisons === 'true' ? true : undefined,
        has_feature_requests: filters.has_feature_requests === 'true' ? true : undefined,
        sort_by: filters.sort_by || 'imported_at',
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      }),
    [filtersKey, page],
  )

  const reviews = data?.reviews ?? []
  const totalCount = data?.total_count ?? 0
  const totalPages = Math.ceil(totalCount / PAGE_SIZE)

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
        const color =
          r.rating >= 4 ? 'text-green-400' : r.rating >= 3 ? 'text-yellow-400' : 'text-red-400'
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
      key: 'score',
      header: 'Score',
      render: (r) => {
        const score = r.pain_score
        if (score == null) return <span className="text-slate-500">--</span>
        const isPraise = (r.rating ?? 0) > 3
        const color = isPraise
          ? score >= 7
            ? 'text-green-400'
            : score >= 4
              ? 'text-cyan-400'
              : 'text-slate-400'
          : score >= 7
            ? 'text-red-400'
            : score >= 4
              ? 'text-yellow-400'
              : 'text-green-400'
        const label = isPraise ? 'L' : 'P'
        return (
          <span className={color}>
            {score.toFixed(1)}{' '}
            <span className="text-[10px] opacity-60">{label}</span>
          </span>
        )
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

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Reviews</h1>
          {!loading && totalCount > 0 && (
            <p className="text-xs text-slate-400 mt-0.5">
              Showing {(page * PAGE_SIZE + 1).toLocaleString()}-
              {Math.min((page + 1) * PAGE_SIZE, totalCount).toLocaleString()} of{' '}
              {totalCount.toLocaleString()}
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

      <FilterBar
        activeFilters={activeFilterEntries.map((e) => ({
          key: e.key,
          label: e.label,
          onClear: () => clearFilter(e.key),
        }))}
        onClearAll={clearAll}
        expanded={
          <>
            <FilterSearch
              label="Root cause"
              value={filters.root_cause}
              onChange={(v) => setFilter('root_cause', v)}
              placeholder="e.g. quality, design..."
              icon={false}
            />
            <FilterSearch
              label="Text search"
              value={filters.search}
              onChange={(v) => setFilter('search', v)}
              placeholder="Full text..."
            />
            <div className="w-36">
              <label className="block text-xs text-slate-400 mb-1">
                Min rating: {filters.min_rating}
              </label>
              <input
                type="range"
                min={0}
                max={5}
                step={0.5}
                value={filters.min_rating}
                onChange={(e) => setFilter('min_rating', Number(e.target.value))}
                className="w-full accent-cyan-500"
              />
            </div>
            <div className="w-36">
              <label className="block text-xs text-slate-400 mb-1">
                Max rating: {filters.max_rating}
              </label>
              <input
                type="range"
                min={0}
                max={5}
                step={0.5}
                value={filters.max_rating}
                onChange={(e) => setFilter('max_rating', Number(e.target.value))}
                className="w-full accent-cyan-500"
              />
            </div>
            <FilterSelect
              label="Severity"
              value={filters.severity}
              onChange={(v) => setFilter('severity', v)}
              options={[
                { value: 'critical', label: 'Critical' },
                { value: 'high', label: 'High' },
                { value: 'medium', label: 'Medium' },
                { value: 'low', label: 'Low' },
              ]}
              placeholder="All Severities"
            />
            <FilterSelect
              label="Status"
              value={filters.enrichment_status}
              onChange={(v) => setFilter('enrichment_status', v)}
              options={[
                { value: 'pending', label: 'Pending' },
                { value: 'enriched', label: 'Enriched' },
                { value: 'failed', label: 'Failed' },
              ]}
              placeholder="All Statuses"
            />
            <div>
              <label className="block text-xs text-slate-400 mb-1">Imported after</label>
              <input
                type="date"
                value={filters.imported_after}
                onChange={(e) => setFilter('imported_after', e.target.value)}
                className="w-full px-3 py-1.5 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white focus:outline-none focus:border-cyan-500/50"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Imported before</label>
              <input
                type="date"
                value={filters.imported_before}
                onChange={(e) => setFilter('imported_before', e.target.value)}
                className="w-full px-3 py-1.5 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white focus:outline-none focus:border-cyan-500/50"
              />
            </div>
            <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer self-end pb-1">
              <input
                type="checkbox"
                checked={filters.has_comparisons === 'true'}
                onChange={(e) =>
                  setFilter('has_comparisons', e.target.checked ? 'true' : '')
                }
                className="accent-cyan-500"
              />
              Has comparisons
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer self-end pb-1">
              <input
                type="checkbox"
                checked={filters.has_feature_requests === 'true'}
                onChange={(e) =>
                  setFilter('has_feature_requests', e.target.checked ? 'true' : '')
                }
                className="accent-cyan-500"
              />
              Has feature requests
            </label>
          </>
        }
      >
        <FilterSearch
          label="Brand"
          value={filters.brand}
          onChange={(v) => setFilter('brand', v)}
          placeholder="Brand..."
        />
        <FilterSearch
          label="ASIN"
          value={filters.asin}
          onChange={(v) => setFilter('asin', v)}
          placeholder="ASIN..."
          icon={false}
        />
        <FilterSelect
          label="Category"
          value={filters.source_category}
          onChange={(v) => setFilter('source_category', v)}
          options={categories.map((c) => ({ value: c, label: c }))}
          placeholder="All Categories"
        />
        <FilterSelect
          label="Sort by"
          value={filters.sort_by || 'imported_at'}
          onChange={(v) => setFilter('sort_by', v)}
          options={[
            { value: 'imported_at', label: 'Newest' },
            { value: 'rating', label: 'Rating' },
            { value: 'pain_score', label: 'Score' },
          ]}
        />
      </FilterBar>

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
