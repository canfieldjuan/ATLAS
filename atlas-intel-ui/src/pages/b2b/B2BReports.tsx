import { useState, useEffect } from 'react'
import { FileText } from 'lucide-react'
import DataTable, { type Column } from '../../components/DataTable'
import FilterBar, { FilterSelect } from '../../components/FilterBar'
import { fetchReports, fetchReportDetail, type B2BReport, type B2BReportDetail } from '../../api/b2bClient'

export default function B2BReports() {
  const [reports, setReports] = useState<B2BReport[]>([])
  const [loading, setLoading] = useState(true)
  const [reportType, setReportType] = useState('')
  const [selected, setSelected] = useState<B2BReportDetail | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    fetchReports({ report_type: reportType || undefined, limit: 50 })
      .then(r => setReports(r.reports))
      .catch(() => setReports([]))
      .finally(() => setLoading(false))
  }, [reportType])

  const handleRowClick = async (r: B2BReport) => {
    setDetailLoading(true)
    try {
      const detail = await fetchReportDetail(r.id)
      setSelected(detail)
    } catch {
      // ignore
    } finally {
      setDetailLoading(false)
    }
  }

  const columns: Column<B2BReport>[] = [
    {
      key: 'type',
      header: 'Type',
      render: r => <span className="text-cyan-400 text-xs font-medium">{r.report_type}</span>,
    },
    {
      key: 'vendor',
      header: 'Vendor',
      render: r => <span className="text-white">{r.vendor_filter || 'All'}</span>,
    },
    {
      key: 'summary',
      header: 'Summary',
      render: r => (
        <span className="text-slate-400 line-clamp-2 max-w-md">
          {r.executive_summary || '--'}
        </span>
      ),
    },
    {
      key: 'date',
      header: 'Date',
      sortable: true,
      sortValue: r => r.report_date || '',
      render: r => <span className="text-slate-500 text-xs">{r.report_date?.split('T')[0] || '--'}</span>,
    },
    {
      key: 'status',
      header: 'Status',
      render: r => (
        <span className={r.status === 'published' ? 'text-green-400 text-xs' : 'text-slate-500 text-xs'}>
          {r.status || 'draft'}
        </span>
      ),
    },
  ]

  const reportTypes = [
    { value: 'weekly_churn_feed', label: 'Weekly Churn Feed' },
    { value: 'vendor_scorecard', label: 'Vendor Scorecard' },
    { value: 'displacement_report', label: 'Displacement Report' },
    { value: 'vendor_retention', label: 'Vendor Retention' },
    { value: 'challenger_intel', label: 'Challenger Intel' },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <FileText className="h-6 w-6 text-cyan-400" />
        <h1 className="text-2xl font-bold text-white">Intelligence Reports</h1>
      </div>

      <FilterBar>
        <FilterSelect
          label="Report Type"
          value={reportType}
          onChange={setReportType}
          options={reportTypes}
          placeholder="All types"
        />
      </FilterBar>

      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl">
        <DataTable
          columns={columns}
          data={reports}
          onRowClick={handleRowClick}
          skeletonRows={loading ? 5 : undefined}
          emptyMessage="No reports available for your tracked vendors"
        />
      </div>

      {/* Detail modal */}
      {selected && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60" onClick={() => setSelected(null)}>
          <div
            className="bg-slate-800 border border-slate-700 rounded-xl p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-white">{selected.report_type}</h2>
              <button onClick={() => setSelected(null)} className="text-slate-400 hover:text-white">X</button>
            </div>
            <div className="space-y-3 text-sm">
              <div><span className="text-slate-500">Vendor:</span> <span className="text-white">{selected.vendor_filter || 'All'}</span></div>
              <div><span className="text-slate-500">Date:</span> <span className="text-white">{selected.report_date?.split('T')[0]}</span></div>
              {selected.executive_summary && (
                <div>
                  <span className="text-slate-500 block mb-1">Executive Summary:</span>
                  <p className="text-slate-300">{selected.executive_summary}</p>
                </div>
              )}
              {selected.intelligence_data && (
                <div>
                  <span className="text-slate-500 block mb-1">Intelligence Data:</span>
                  <pre className="text-xs text-slate-400 bg-slate-900/50 rounded p-3 overflow-x-auto">
                    {JSON.stringify(selected.intelligence_data, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
