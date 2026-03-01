import { useState, useEffect, useRef } from 'react'
import { RefreshCw, Search, X } from 'lucide-react'
import { clsx } from 'clsx'
import DataTable, { type Column } from '../components/DataTable'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import { fetchFeatures, type FeatureGapEntry, type NegativeAspect } from '../api/client'

export default function Features() {
  const [brand, setBrand] = useState('')
  const [debouncedBrand, setDebouncedBrand] = useState('')
  const [tab, setTab] = useState<'requests' | 'aspects'>('requests')
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined)

  useEffect(() => {
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setDebouncedBrand(brand), 300)
    return () => clearTimeout(timerRef.current)
  }, [brand])

  const { data, loading, error, refresh, refreshing } = useApiData(
    () => fetchFeatures({
      brand: debouncedBrand || undefined,
      limit: 100,
    }),
    [debouncedBrand],
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

      {/* Filter */}
      <div className="flex flex-wrap items-end gap-4">
        <div className="flex-1 min-w-[200px]">
          <label className="block text-xs text-slate-400 mb-1">Filter by brand</label>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
            <input
              type="text"
              value={brand}
              onChange={(e) => setBrand(e.target.value)}
              placeholder="Brand name..."
              className="w-full pl-9 pr-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
            />
          </div>
        </div>
        {brand && (
          <button
            onClick={() => setBrand('')}
            className="flex items-center gap-1 px-3 py-2 text-sm text-slate-400 hover:text-white transition-colors"
          >
            <X className="h-3 w-3" />
            Clear
          </button>
        )}
      </div>

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
        {tab === 'requests' && (
          loading ? (
            <DataTable columns={featureColumns} data={[]} skeletonRows={10} />
          ) : (
            <DataTable
              columns={featureColumns}
              data={features}
              emptyMessage="No feature requests found"
            />
          )
        )}
        {tab === 'aspects' && (
          loading ? (
            <DataTable columns={aspectColumns} data={[]} skeletonRows={10} />
          ) : (
            <DataTable
              columns={aspectColumns}
              data={aspects}
              emptyMessage="No negative aspects found"
            />
          )
        )}
      </div>
    </div>
  )
}
