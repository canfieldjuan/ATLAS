import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { RefreshCw, ChevronLeft, ChevronRight, Scale } from 'lucide-react'
import { clsx } from 'clsx'
import DataTable, { type Column } from '../components/DataTable'
import FilterBar, { FilterSearch, FilterSelect } from '../components/FilterBar'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import useFilterParams from '../hooks/useFilterParams'
import useCategories from '../hooks/useCategories'
import { fetchBrands, type BrandSummary } from '../api/client'

const PAGE_SIZE = 50

const FILTER_CONFIG = {
  search: { type: 'string' as const, label: 'Brand' },
  source_category: { type: 'string' as const, label: 'Category' },
  sort_by: { type: 'string' as const, label: 'Sort', default: 'review_count' },
  min_reviews: { type: 'number' as const, label: 'Min Reviews', default: 0 },
}

type Filters = {
  search: string
  source_category: string
  sort_by: string
  min_reviews: number
}

export default function Brands() {
  const navigate = useNavigate()
  const { categories } = useCategories()
  const { filters, setFilter, clearFilter, clearAll, activeFilterEntries } =
    useFilterParams<Filters>(FILTER_CONFIG)
  const [page, setPage] = useState(0)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const filtersKey = JSON.stringify(filters)

  // Reset page when filters change
  useEffect(() => {
    setPage(0)
  }, [filtersKey])

  const { data, loading, error, refresh, refreshing } = useApiData(
    () =>
      fetchBrands({
        search: filters.search || undefined,
        source_category: filters.source_category || undefined,
        min_reviews: filters.min_reviews || undefined,
        sort_by: filters.sort_by || 'review_count',
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      }),
    [filtersKey, page],
  )

  const brands = data?.brands ?? []
  const totalCount = data?.total_count ?? 0
  const totalPages = Math.ceil(totalCount / PAGE_SIZE)

  const toggleSelected = (brand: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(brand)) {
        next.delete(brand)
      } else if (next.size < 4) {
        next.add(brand)
      }
      return next
    })
  }

  const columns: Column<BrandSummary>[] = [
    {
      key: 'select',
      header: '',
      render: (r) => (
        <div onClick={(e) => toggleSelected(r.brand, e)} className="flex items-center justify-center">
          <input
            type="checkbox"
            checked={selected.has(r.brand)}
            readOnly
            disabled={!selected.has(r.brand) && selected.size >= 4}
            className="h-4 w-4 rounded border-slate-600 bg-slate-800 text-cyan-500 focus:ring-cyan-500/30 cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed accent-cyan-500"
          />
        </div>
      ),
    },
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
      key: 'health',
      header: 'Health',
      render: (r) => {
        const h = r.brand_health
        if (h == null) return <span className="text-slate-500">--</span>
        const color = h >= 60 ? 'text-emerald-400' : h >= 40 ? 'text-amber-400' : 'text-red-400'
        return <span className={`font-mono font-medium ${color}`}>{h}</span>
      },
      sortable: true,
      sortValue: (r) => r.brand_health ?? -1,
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
              Showing {(page * PAGE_SIZE + 1).toLocaleString()}-
              {Math.min((page + 1) * PAGE_SIZE, totalCount).toLocaleString()} of{' '}
              {totalCount.toLocaleString()}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {selected.size > 0 && (
            <>
              <span className="text-xs text-slate-400">{selected.size}/4 selected</span>
              <button
                onClick={() => setSelected(new Set())}
                className="text-xs text-slate-500 hover:text-white transition-colors"
              >
                Clear
              </button>
            </>
          )}
          {selected.size >= 2 && (
            <button
              onClick={() =>
                navigate(
                  `/compare?brands=${Array.from(selected).map(encodeURIComponent).join(',')}`,
                )
              }
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 transition-colors"
            >
              <Scale className="h-4 w-4" />
              Compare ({selected.size})
            </button>
          )}
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

      <FilterBar
        activeFilters={activeFilterEntries.map((e) => ({
          key: e.key,
          label: e.label,
          onClear: () => clearFilter(e.key),
        }))}
        onClearAll={clearAll}
      >
        <FilterSearch
          label="Search brands"
          value={filters.search}
          onChange={(v) => setFilter('search', v)}
          placeholder="Brand name..."
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
          value={filters.sort_by || 'review_count'}
          onChange={(v) => setFilter('sort_by', v)}
          options={[
            { value: 'review_count', label: 'Review Count' },
            { value: 'avg_rating', label: 'Avg Rating' },
            { value: 'avg_complaint_score', label: 'Pain Score' },
            { value: 'avg_praise_score', label: 'Loyalty Score' },
            { value: 'brand_health', label: 'Brand Health' },
            { value: 'safety_count', label: 'Safety Flags' },
            { value: 'brand', label: 'Brand Name' },
          ]}
        />
        <div className="w-40">
          <label className="block text-xs text-slate-400 mb-1">
            Min reviews: {filters.min_reviews}
          </label>
          <input
            type="range"
            min={0}
            max={100}
            value={filters.min_reviews}
            onChange={(e) => setFilter('min_reviews', Number(e.target.value))}
            className="w-full accent-cyan-500"
          />
        </div>
      </FilterBar>

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
