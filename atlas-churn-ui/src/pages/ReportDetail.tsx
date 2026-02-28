import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { fetchReport } from '../api/client'
import type { ReportDetail as ReportDetailType } from '../types'

export default function ReportDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [report, setReport] = useState<ReportDetailType | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!id) return
    fetchReport(id)
      .then(setReport)
      .catch((err) => console.error('ReportDetail load error:', err))
      .finally(() => setLoading(false))
  }, [id])

  if (loading) return <div className="text-slate-500 p-8">Loading report...</div>
  if (!report) return <div className="text-slate-500 p-8">Report not found</div>

  return (
    <div className="space-y-6 max-w-4xl">
      <button
        onClick={() => navigate('/reports')}
        className="flex items-center gap-1 text-sm text-slate-400 hover:text-white transition-colors"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to Reports
      </button>

      <div>
        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-cyan-500/20 text-cyan-400 mb-2">
          {report.report_type.replace(/_/g, ' ')}
        </span>
        <h1 className="text-2xl font-bold text-white">
          {report.vendor_filter ?? report.report_type.replace(/_/g, ' ')}
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          {report.report_date ?? report.created_at}
          {report.llm_model && ` | Model: ${report.llm_model}`}
        </p>
      </div>

      {report.executive_summary && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
          <h3 className="text-sm font-medium text-slate-300 mb-2">Executive Summary</h3>
          <p className="text-sm text-slate-300 whitespace-pre-wrap">{report.executive_summary}</p>
        </div>
      )}

      {report.intelligence_data && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
          <h3 className="text-sm font-medium text-slate-300 mb-3">Intelligence Data</h3>
          <div className="space-y-4">
            {Object.entries(report.intelligence_data).map(([key, value]) => (
              <div key={key}>
                <h4 className="text-xs font-medium text-cyan-400 uppercase tracking-wider mb-1">
                  {key.replace(/_/g, ' ')}
                </h4>
                {typeof value === 'string' ? (
                  <p className="text-sm text-slate-300 whitespace-pre-wrap">{value}</p>
                ) : (
                  <pre className="text-xs text-slate-400 bg-slate-800/50 rounded p-3 overflow-x-auto">
                    {JSON.stringify(value, null, 2)}
                  </pre>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {report.data_density && Object.keys(report.data_density).length > 0 && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
          <h3 className="text-sm font-medium text-slate-300 mb-3">Data Density</h3>
          <dl className="grid grid-cols-2 gap-2 text-sm">
            {Object.entries(report.data_density).map(([key, value]) => (
              <div key={key} className="flex justify-between">
                <dt className="text-slate-400">{key.replace(/_/g, ' ')}</dt>
                <dd className="text-white">{String(value)}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}
    </div>
  )
}
