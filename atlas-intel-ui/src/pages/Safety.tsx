import { useNavigate } from 'react-router-dom'
import { ShieldAlert, RefreshCw } from 'lucide-react'
import { clsx } from 'clsx'
import StatCard from '../components/StatCard'
import DataTable, { type Column } from '../components/DataTable'
import FilterBar, { FilterSearch, FilterSelect } from '../components/FilterBar'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import useFilterParams from '../hooks/useFilterParams'
import useCategories from '../hooks/useCategories'
import { fetchSafety, type SafetySignal } from '../api/client'

const FILTER_CONFIG = {
  brand: { type: 'string' as const, label: 'Brand' },
  source_category: { type: 'string' as const, label: 'Category' },
  min_rating: { type: 'number' as const, label: 'Min Rating', default: 0 },
  max_rating: { type: 'number' as const, label: 'Max Rating', default: 5 },
}

type Filters = {
  brand: string
  source_category: string
  min_rating: number
  max_rating: number
}

export default function Safety() {
  const navigate = useNavigate()
  const { categories } = useCategories()
  const { filters, setFilter, clearFilter, clearAll, activeFilterEntries } =
    useFilterParams<Filters>(FILTER_CONFIG)

  const { data, loading, error, refresh, refreshing } = useApiData(
    () =>
      fetchSafety({
        brand: filters.brand || undefined,
        source_category: filters.source_category || undefined,
        min_rating: filters.min_rating > 0 ? filters.min_rating : undefined,
        max_rating: filters.max_rating < 5 ? filters.max_rating : undefined,
        limit: 100,
      }),
    [JSON.stringify(filters)],
  )

  const signals = data?.signals ?? []
  const totalFlagged = data?.total_flagged ?? 0

  const columns: Column<SafetySignal>[] = [
    {
      key: 'brand',
      header: 'Brand',
      render: (r) => <span className="text-white font-medium">{r.brand}</span>,
    },
    {
      key: 'title',
      header: 'Product',
      render: (r) => (
        <span className="text-slate-300 truncate max-w-[250px] block">{r.title}</span>
      ),
    },
    {
      key: 'rating',
      header: 'Rating',
      render: (r) => <span className="text-slate-300">{r.rating?.toFixed(1) ?? '--'}</span>,
    },
    {
      key: 'safety',
      header: 'Safety Issue',
      render: (r) => {
        const flag = r.safety_flag
        const desc = (flag?.description as string) || 'Flagged'
        return (
          <span className="text-red-400 text-sm max-w-[300px] truncate block">{desc}</span>
        )
      },
    },
    {
      key: 'summary',
      header: 'Review Summary',
      render: (r) => (
        <span className="text-slate-400 truncate max-w-[250px] block">
          {r.summary ?? r.review_excerpt ?? '--'}
        </span>
      ),
    },
  ]

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Safety Signals</h1>
        <button
          onClick={refresh}
          disabled={refreshing}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
          Refresh
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Total Flagged"
          value={totalFlagged}
          icon={<ShieldAlert className="h-5 w-5" />}
          skeleton={loading}
        />
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
          </>
        }
      >
        <FilterSearch
          label="Filter by brand"
          value={filters.brand}
          onChange={(v) => setFilter('brand', v)}
          placeholder="Brand name..."
        />
        <FilterSelect
          label="Category"
          value={filters.source_category}
          onChange={(v) => setFilter('source_category', v)}
          options={categories.map((c) => ({ value: c, label: c }))}
          placeholder="All Categories"
        />
      </FilterBar>

      {/* Table */}
      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl overflow-hidden">
        {loading ? (
          <DataTable columns={columns} data={[]} skeletonRows={10} />
        ) : (
          <DataTable
            columns={columns}
            data={signals}
            onRowClick={(r) => navigate(`/reviews/${r.id}`)}
            emptyMessage="No safety signals found"
          />
        )}
      </div>
    </div>
  )
}
