import { useNavigate } from 'react-router-dom'
import { RefreshCw } from 'lucide-react'
import { clsx } from 'clsx'
import DataTable, { type Column } from '../components/DataTable'
import FilterBar, { FilterSearch, FilterSelect } from '../components/FilterBar'
import { PageError } from '../components/ErrorBoundary'
import PlanGate from '../components/PlanGate'
import useApiData from '../hooks/useApiData'
import useFilterParams from '../hooks/useFilterParams'
import useCategories from '../hooks/useCategories'
import { fetchFlows, type FlowEntry } from '../api/client'

const FILTER_CONFIG = {
  brand: { type: 'string' as const, label: 'Brand' },
  source_category: { type: 'string' as const, label: 'Category' },
  direction: { type: 'string' as const, label: 'Direction' },
  min_count: { type: 'number' as const, label: 'Min Mentions', default: 2 },
}

type Filters = {
  brand: string
  source_category: string
  direction: string
  min_count: number
}

export default function Flows() {
  return (
    <PlanGate minPlan="growth">
      <FlowsInner />
    </PlanGate>
  )
}

function FlowsInner() {
  const navigate = useNavigate()
  const { categories } = useCategories()
  const { filters, setFilter, clearFilter, clearAll, activeFilterEntries } =
    useFilterParams<Filters>(FILTER_CONFIG)

  const { data, loading, error, refresh, refreshing } = useApiData(
    () =>
      fetchFlows({
        brand: filters.brand || undefined,
        source_category: filters.source_category || undefined,
        direction: filters.direction || undefined,
        min_count: filters.min_count !== 2 ? filters.min_count : undefined,
        limit: 200,
      }),
    [JSON.stringify(filters)],
  )

  const flows = data?.flows ?? []

  const directionBadge = (dir: string) => {
    const colors: Record<string, string> = {
      switched_to: 'bg-green-500/10 text-green-400',
      switched_from: 'bg-red-500/10 text-red-400',
      considered: 'bg-yellow-500/10 text-yellow-400',
      compared: 'bg-blue-500/10 text-blue-400',
    }
    return (
      <span
        className={`px-2 py-0.5 rounded-full text-xs font-medium ${colors[dir] ?? 'bg-slate-700 text-slate-300'}`}
      >
        {dir}
      </span>
    )
  }

  const columns: Column<FlowEntry>[] = [
    {
      key: 'from',
      header: 'From Brand',
      render: (r) => <span className="text-white font-medium">{r.from_brand}</span>,
    },
    {
      key: 'to',
      header: 'To Brand',
      render: (r) => (
        <button
          onClick={(e) => {
            e.stopPropagation()
            navigate(`/brands/${encodeURIComponent(r.to_brand)}`)
          }}
          className="text-cyan-400 hover:underline"
        >
          {r.to_brand}
        </button>
      ),
    },
    {
      key: 'count',
      header: 'Mentions',
      render: (r) => <span className="text-slate-300 font-mono">{r.count}</span>,
      sortable: true,
      sortValue: (r) => r.count,
    },
    {
      key: 'direction',
      header: 'Direction',
      render: (r) => directionBadge(r.direction),
    },
    {
      key: 'rating',
      header: 'Avg Rating',
      render: (r) => <span className="text-slate-300">{r.avg_rating?.toFixed(1) ?? '--'}</span>,
      sortable: true,
      sortValue: (r) => r.avg_rating ?? 0,
    },
  ]

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Competitive Flows</h1>
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
        <FilterSelect
          label="Direction"
          value={filters.direction}
          onChange={(v) => setFilter('direction', v)}
          options={[
            { value: 'switched_to', label: 'Switched To' },
            { value: 'switched_from', label: 'Switched From' },
            { value: 'considered', label: 'Considered' },
            { value: 'compared', label: 'Compared' },
          ]}
          placeholder="All Directions"
        />
        <div className="w-44">
          <label className="block text-xs text-slate-400 mb-1">
            Min mentions: {filters.min_count}
          </label>
          <input
            type="range"
            min={1}
            max={20}
            value={filters.min_count}
            onChange={(e) => setFilter('min_count', Number(e.target.value))}
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
            data={flows}
            onRowClick={(r) =>
              navigate(
                `/reviews?brand=${encodeURIComponent(r.from_brand)}&has_comparisons=true`,
              )
            }
            emptyMessage="No competitive flows found"
          />
        )}
      </div>
    </div>
  )
}
