import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { FileBarChart } from 'lucide-react'
import { fetchReports } from '../api/client'
import type { Report } from '../types'

const REPORT_TYPE_COLORS: Record<string, string> = {
  weekly_churn_feed: 'bg-cyan-500/20 text-cyan-400',
  vendor_scorecard: 'bg-violet-500/20 text-violet-400',
  displacement_report: 'bg-amber-500/20 text-amber-400',
  category_overview: 'bg-emerald-500/20 text-emerald-400',
}

export default function Reports() {
  const navigate = useNavigate()
  const [reports, setReports] = useState<Report[]>([])
  const [loading, setLoading] = useState(true)
  const [typeFilter, setTypeFilter] = useState('')

  useEffect(() => {
    async function load() {
      setLoading(true)
      try {
        const res = await fetchReports({
          report_type: typeFilter || undefined,
          limit: 50,
        })
        setReports(res.reports)
      } catch (err) {
        console.error('Reports load error:', err)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [typeFilter])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Intelligence Reports</h1>
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
      </div>

      {loading ? (
        <div className="text-slate-500 p-8 text-center">Loading reports...</div>
      ) : reports.length === 0 ? (
        <div className="text-slate-500 p-8 text-center">No reports found</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {reports.map((r) => (
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
