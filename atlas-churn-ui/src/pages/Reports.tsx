import { useNavigate } from 'react-router-dom'
import { FileBarChart, RefreshCw } from 'lucide-react'
import { clsx } from 'clsx'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import { fetchReports } from '../api/client'
import { useState } from 'react'
import type { Report } from '../types'

const REPORT_TYPE_COLORS: Record<string, string> = {
  weekly_churn_feed: 'bg-cyan-500/20 text-cyan-400',
  vendor_scorecard: 'bg-violet-500/20 text-violet-400',
  displacement_report: 'bg-amber-500/20 text-amber-400',
  category_overview: 'bg-emerald-500/20 text-emerald-400',
}

function CardSkeleton() {
  return (
    <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 animate-pulse">
      <div className="flex items-start justify-between mb-3">
        <div className="h-5 w-28 bg-slate-700/50 rounded" />
        <div className="h-4 w-4 bg-slate-700/50 rounded" />
      </div>
      <div className="h-4 w-36 bg-slate-700/50 rounded mb-2" />
      <div className="h-3 w-full bg-slate-700/50 rounded mb-1" />
      <div className="h-3 w-3/4 bg-slate-700/50 rounded" />
      <div className="h-3 w-20 bg-slate-700/50 rounded mt-4" />
    </div>
  )
}

export default function Reports() {
  const navigate = useNavigate()
  const [typeFilter, setTypeFilter] = useState('')

  const { data, loading, error, refresh, refreshing } = useApiData(
    () => fetchReports({ report_type: typeFilter || undefined, limit: 50 }),
    [typeFilter],
  )

  const reports = data?.reports ?? []

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Intelligence Reports</h1>
        <div className="flex items-center gap-3">
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            className="bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white px-3 py-2 focus:outline-none focus:border-cyan-500/50"
          >
            <option value="">All Types</option>
            <option value="weekly_churn_feed">Weekly Churn Feed</option>
            <option value="vendor_scorecard">Vendor Scorecard</option>
            <option value="displacement_report">Displacement Report</option>
            <option value="category_overview">Category Overview</option>
          </select>
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

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Array.from({ length: 4 }, (_, i) => (
            <CardSkeleton key={i} />
          ))}
        </div>
      ) : reports.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <FileBarChart className="h-10 w-10 text-slate-600 mb-4" />
          <p className="text-slate-500 mb-4">No reports found</p>
          {typeFilter && (
            <button
              onClick={() => setTypeFilter('')}
              className="px-3 py-1.5 rounded-lg bg-cyan-500/10 text-cyan-400 text-sm font-medium hover:bg-cyan-500/20 transition-colors"
            >
              Clear filter
            </button>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {reports.map((r: Report) => (
            <button
              key={r.id}
              onClick={() => navigate(`/reports/${r.id}`)}
              className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5 text-left hover:border-cyan-500/30 transition-colors"
            >
              <div className="flex items-start justify-between mb-3">
                <span
                  className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                    REPORT_TYPE_COLORS[r.report_type] ?? 'bg-slate-700 text-slate-300'
                  }`}
                >
                  {r.report_type.replace(/_/g, ' ')}
                </span>
                <FileBarChart className="h-4 w-4 text-slate-500" />
              </div>
              {r.vendor_filter && (
                <p className="text-sm text-white font-medium mb-1">{r.vendor_filter}</p>
              )}
              <p className="text-sm text-slate-400 line-clamp-2">
                {r.executive_summary ?? 'No summary available'}
              </p>
              <p className="text-xs text-slate-500 mt-3">
                {r.report_date ?? r.created_at ?? '--'}
              </p>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
