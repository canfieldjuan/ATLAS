import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { RefreshCw, Search, X } from 'lucide-react'
import { clsx } from 'clsx'
import DataTable, { type Column } from '../components/DataTable'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import { fetchFlows, type FlowEntry } from '../api/client'

export default function Flows() {
  const navigate = useNavigate()
  const [brand, setBrand] = useState('')
  const [debouncedBrand, setDebouncedBrand] = useState('')
  const [minCount, setMinCount] = useState(2)
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined)

  useEffect(() => {
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setDebouncedBrand(brand), 300)
    return () => clearTimeout(timerRef.current)
  }, [brand])

  const { data, loading, error, refresh, refreshing } = useApiData(
    () => fetchFlows({
      brand: debouncedBrand || undefined,
      min_count: minCount,
      limit: 200,
    }),
    [debouncedBrand, minCount],
  )

  const flows = data?.flows ?? []
  const hasFilters = brand || minCount > 2

  const directionBadge = (dir: string) => {
    const colors: Record<string, string> = {
      switched_to: 'bg-green-500/10 text-green-400',
      switched_from: 'bg-red-500/10 text-red-400',
      considered: 'bg-yellow-500/10 text-yellow-400',
      compared: 'bg-blue-500/10 text-blue-400',
    }
    return (
      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${colors[dir] ?? 'bg-slate-700 text-slate-300'}`}>
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

      {/* Filters */}
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
        <div className="w-44">
          <label className="block text-xs text-slate-400 mb-1">Min mentions: {minCount}</label>
          <input
            type="range"
            min={1}
            max={20}
            value={minCount}
            onChange={(e) => setMinCount(Number(e.target.value))}
            className="w-full accent-cyan-500"
          />
        </div>
        {hasFilters && (
          <button
            onClick={() => {
              setBrand('')
              setMinCount(2)
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
            data={flows}
            onRowClick={(r) => navigate(`/reviews?brand=${encodeURIComponent(r.from_brand)}&has_comparisons=true`)}
            emptyMessage="No competitive flows found"
          />
        )}
      </div>
    </div>
  )
}
