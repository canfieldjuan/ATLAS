import { useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { clsx } from 'clsx'
import DataTable, { type Column } from '../components/DataTable'
import FilterBar, { FilterSearch, FilterSelect } from '../components/FilterBar'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import useFilterParams from '../hooks/useFilterParams'
import useCategories from '../hooks/useCategories'
import { fetchFeatures, type FeatureGapEntry, type NegativeAspect } from '../api/client'

const FILTER_CONFIG = {
  brand: { type: 'string' as const, label: 'Brand' },
  source_category: { type: 'string' as const, label: 'Category' },
  min_count: { type: 'number' as const, label: 'Min Count', default: 1 },
}

type Filters = {
  brand: string
  source_category: string
  min_count: number
}

export default function Features() {
  const { categories } = useCategories()
  const { filters, setFilter, clearFilter, clearAll, activeFilterEntries } =
    useFilterParams<Filters>(FILTER_CONFIG)
  const [tab, setTab] = useState<'requests' | 'aspects'>('requests')

  const { data, loading, error, refresh, refreshing } = useApiData(
    () =>
      fetchFeatures({
        brand: filters.brand || undefined,
        source_category: filters.source_category || undefined,
        min_count: filters.min_count > 1 ? filters.min_count : undefined,
        limit: 100,
      }),
    [JSON.stringify(filters)],
  )

  const features = data?.feature_requests ?? []
  const aspects = data?.negative_aspects ?? []

  const featureColumns: Column<FeatureGapEntry>[] = [
    {
      key: 'request',
      header: 'Feature Request',
      render: (r) => <span className="text-white max-w-[400px] truncate block">{r.request}</span>,
    },
    {
      key: 'count',
      header: 'Frequency',
      render: (r) => <span className="text-cyan-400 font-mono">{r.count}</span>,
      sortable: true,
      sortValue: (r) => r.count,
    },
    {
      key: 'brands',
      header: 'Brands',
      render: (r) => <span className="text-slate-300">{r.brands_affected}</span>,
      sortable: true,
      sortValue: (r) => r.brands_affected,
    },
    {
      key: 'brand_list',
      header: 'Top Brands',
      render: (r) => (
        <div className="flex flex-wrap gap-1">
          {r.brand_list.map((b) => (
            <span key={b} className="px-1.5 py-0.5 bg-slate-800 rounded text-xs text-slate-300">
              {b}
            </span>
          ))}
        </div>
      ),
    },
    {
      key: 'rating',
      header: 'Avg Rating',
      render: (r) => <span className="text-slate-300">{r.avg_rating?.toFixed(1) ?? '--'}</span>,
      sortable: true,
      sortValue: (r) => r.avg_rating ?? 0,
    },
  ]

  const aspectColumns: Column<NegativeAspect>[] = [
    {
      key: 'aspect',
      header: 'Aspect',
      render: (r) => <span className="text-white font-medium">{r.aspect}</span>,
    },
    {
      key: 'negative',
      header: 'Negative',
      render: (r) => <span className="text-red-400 font-mono">{r.negative}</span>,
      sortable: true,
      sortValue: (r) => r.negative,
    },
    {
      key: 'total',
      header: 'Total',
      render: (r) => <span className="text-slate-300">{r.total}</span>,
      sortable: true,
      sortValue: (r) => r.total,
    },
    {
      key: 'pct',
      header: '% Negative',
      render: (r) => {
        const pct = r.pct_negative
        const color = pct >= 60 ? 'text-red-400' : pct >= 30 ? 'text-yellow-400' : 'text-slate-300'
        return <span className={`${color} font-mono`}>{pct}%</span>
      },
      sortable: true,
      sortValue: (r) => r.pct_negative,
    },
    {
      key: 'brands',
      header: 'Top Brands',
      render: (r) => (
        <div className="flex flex-wrap gap-1">
          {r.top_brands.map((b) => (
            <span key={b} className="px-1.5 py-0.5 bg-slate-800 rounded text-xs text-slate-300">
              {b}
            </span>
          ))}
        </div>
      ),
    },
  ]

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Feature Gaps</h1>
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
          <div className="w-44">
            <label className="block text-xs text-slate-400 mb-1">
              Min count: {filters.min_count}
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

      {/* Tabs */}
      <div className="flex gap-1 border-b border-slate-700/50">
        <button
          onClick={() => setTab('requests')}
          className={`px-4 py-2 text-sm transition-colors ${
            tab === 'requests'
              ? 'text-cyan-400 border-b-2 border-cyan-400'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          Feature Requests ({features.length})
        </button>
        <button
          onClick={() => setTab('aspects')}
          className={`px-4 py-2 text-sm transition-colors ${
            tab === 'aspects'
              ? 'text-cyan-400 border-b-2 border-cyan-400'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          Negative Aspects ({aspects.length})
        </button>
      </div>

      {/* Tables */}
      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl overflow-hidden">
        {tab === 'requests' &&
          (loading ? (
            <DataTable columns={featureColumns} data={[]} skeletonRows={10} />
          ) : (
            <DataTable
              columns={featureColumns}
              data={features}
              emptyMessage="No feature requests found"
            />
          ))}
        {tab === 'aspects' &&
          (loading ? (
            <DataTable columns={aspectColumns} data={[]} skeletonRows={10} />
          ) : (
            <DataTable
              columns={aspectColumns}
              data={aspects}
              emptyMessage="No negative aspects found"
            />
          ))}
      </div>
    </div>
  )
}
