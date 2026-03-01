import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { ShieldAlert, RefreshCw, Search, X } from 'lucide-react'
import { clsx } from 'clsx'
import StatCard from '../components/StatCard'
import DataTable, { type Column } from '../components/DataTable'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import { fetchSafety, type SafetySignal } from '../api/client'

export default function Safety() {
  const navigate = useNavigate()
  const [brand, setBrand] = useState('')
  const [debouncedBrand, setDebouncedBrand] = useState('')
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined)

  useEffect(() => {
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setDebouncedBrand(brand), 300)
    return () => clearTimeout(timerRef.current)
  }, [brand])

  const { data, loading, error, refresh, refreshing } = useApiData(
    () => fetchSafety({
      brand: debouncedBrand || undefined,
      limit: 100,
    }),
    [debouncedBrand],
  )

  const signals = data?.signals ?? []
  const totalFlagged = data?.total_flagged ?? 0
  // Note: safety_flag schema only has {flagged, description} -- no category field

  const columns: Column<SafetySignal>[] = [
    {
      key: 'brand',
      header: 'Brand',
      render: (r) => <span className="text-white font-medium">{r.brand}</span>,
    },
    {
      key: 'title',
      header: 'Product',
      render: (r) => <span className="text-slate-300 truncate max-w-[250px] block">{r.title}</span>,
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
        return <span className="text-red-400 text-sm max-w-[300px] truncate block">{desc}</span>
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
