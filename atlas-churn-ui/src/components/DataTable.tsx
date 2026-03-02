import { useState, useEffect, useMemo, type ReactNode } from 'react'
import { ChevronUp, ChevronDown, ChevronLeft, ChevronRight, Inbox } from 'lucide-react'
import { clsx } from 'clsx'

export interface Column<T> {
  key: string
  header: string
  render: (row: T) => ReactNode
  sortable?: boolean
  sortValue?: (row: T) => number | string
}

interface DataTableProps<T> {
  columns: Column<T>[]
  data: T[]
  onRowClick?: (row: T) => void
  emptyMessage?: string
  skeletonRows?: number
  emptyAction?: { label: string; onClick: () => void }
  pageSize?: number
}

const PAGE_SIZES = [25, 50, 100] as const

export default function DataTable<T>({
  columns,
  data,
  onRowClick,
  emptyMessage = 'No data',
  skeletonRows,
  emptyAction,
  pageSize: defaultPageSize = 25,
}: DataTableProps<T>) {
  const [sortKey, setSortKey] = useState<string | null>(null)
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
  const [page, setPage] = useState(0)
  const [perPage, setPerPage] = useState(defaultPageSize)

  // Reset to page 0 when data length changes (new fetch / filter)
  const dataLen = data.length
  useEffect(() => { setPage(0) }, [dataLen])

  const sorted = useMemo(() => {
    if (!sortKey) return data
    const col = columns.find((c) => c.key === sortKey)
    if (!col?.sortValue) return data
    const fn = col.sortValue
    return [...data].sort((a, b) => {
      const av = fn(a)
      const bv = fn(b)
      if (av < bv) return sortDir === 'asc' ? -1 : 1
      if (av > bv) return sortDir === 'asc' ? 1 : -1
      return 0
    })
  }, [data, sortKey, sortDir, columns])

  const totalPages = Math.max(1, Math.ceil(sorted.length / perPage))
  const safePage = Math.min(page, totalPages - 1)
  const paged = sorted.slice(safePage * perPage, (safePage + 1) * perPage)
  const showPagination = sorted.length > PAGE_SIZES[0]

  function handleSort(key: string) {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
    setPage(0)
  }

  if (skeletonRows) {
    return (
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700/50">
              {columns.map((col) => (
                <th key={col.key} className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">
                  {col.header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/50">
            {Array.from({ length: skeletonRows }, (_, i) => (
              <tr key={i} className="animate-pulse">
                {columns.map((col) => (
                  <td key={col.key} className="px-4 py-3">
                    <div className="h-4 bg-slate-700/50 rounded w-3/4" />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-700/50">
            {columns.map((col) => (
              <th
                key={col.key}
                className={clsx(
                  'px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider',
                  col.sortable && 'cursor-pointer select-none hover:text-slate-200'
                )}
                onClick={() => col.sortable && handleSort(col.key)}
              >
                <span className="flex items-center gap-1">
                  {col.header}
                  {col.sortable && sortKey === col.key && (
                    sortDir === 'asc' ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />
                  )}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/50">
          {sorted.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="px-4 py-12 text-center">
                <div className="flex flex-col items-center gap-3">
                  <Inbox className="h-8 w-8 text-slate-600" />
                  <p className="text-slate-500">{emptyMessage}</p>
                  {emptyAction && (
                    <button
                      onClick={emptyAction.onClick}
                      className="mt-1 px-3 py-1.5 rounded-lg bg-cyan-500/10 text-cyan-400 text-sm font-medium hover:bg-cyan-500/20 transition-colors"
                    >
                      {emptyAction.label}
                    </button>
                  )}
                </div>
              </td>
            </tr>
          ) : (
            paged.map((row, i) => (
              <tr
                key={i}
                className={clsx(
                  'transition-colors',
                  onRowClick
                    ? 'cursor-pointer hover:bg-slate-800/50'
                    : 'hover:bg-slate-800/30'
                )}
                onClick={() => onRowClick?.(row)}
              >
                {columns.map((col) => (
                  <td key={col.key} className="px-4 py-3 whitespace-nowrap">
                    {col.render(row)}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>

      {showPagination && sorted.length > 0 && (
        <div className="flex items-center justify-between px-4 py-3 border-t border-slate-700/50">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span>
              {safePage * perPage + 1}--{Math.min((safePage + 1) * perPage, sorted.length)} of {sorted.length}
            </span>
            <select
              value={perPage}
              onChange={(e) => { setPerPage(Number(e.target.value)); setPage(0) }}
              className="bg-slate-800/50 border border-slate-700/50 rounded px-1.5 py-0.5 text-xs text-white focus:outline-none"
            >
              {PAGE_SIZES.map((s) => (
                <option key={s} value={s}>{s} / page</option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={safePage === 0}
              className="p-1 rounded text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <span className="text-xs text-slate-400 px-2">
              {safePage + 1} / {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={safePage >= totalPages - 1}
              className="p-1 rounded text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
