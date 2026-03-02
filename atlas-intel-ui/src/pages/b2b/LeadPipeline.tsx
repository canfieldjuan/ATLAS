import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Target } from 'lucide-react'
import DataTable, { type Column } from '../../components/DataTable'
import FilterBar, { FilterSelect } from '../../components/FilterBar'
import { fetchLeads, type HighIntentLead } from '../../api/b2bClient'

export default function LeadPipeline() {
  const navigate = useNavigate()
  const [leads, setLeads] = useState<HighIntentLead[]>([])
  const [loading, setLoading] = useState(true)
  const [minUrgency, setMinUrgency] = useState('7')
  const [windowDays, setWindowDays] = useState('90')

  useEffect(() => {
    setLoading(true)
    fetchLeads({
      min_urgency: Number(minUrgency),
      window_days: Number(windowDays),
      limit: 50,
    })
      .then(r => setLeads(r.leads))
      .catch(() => setLeads([]))
      .finally(() => setLoading(false))
  }, [minUrgency, windowDays])

  const columns: Column<HighIntentLead>[] = [
    {
      key: 'company',
      header: 'Company',
      render: r => <span className="text-white font-medium">{r.company}</span>,
    },
    {
      key: 'vendor',
      header: 'Leaving',
      render: r => <span className="text-slate-300">{r.vendor}</span>,
    },
    {
      key: 'urgency',
      header: 'Urgency',
      sortable: true,
      sortValue: r => r.urgency,
      render: r => {
        const color = r.urgency >= 9 ? 'text-red-400' : r.urgency >= 7 ? 'text-amber-400' : 'text-slate-300'
        return <span className={color}>{r.urgency.toFixed(1)}</span>
      },
    },
    {
      key: 'pain',
      header: 'Pain',
      render: r => <span className="text-slate-400">{r.pain || '--'}</span>,
    },
    {
      key: 'dm',
      header: 'Decision Maker',
      render: r => r.decision_maker
        ? <span className="text-green-400 text-xs">Yes</span>
        : <span className="text-slate-600 text-xs">No</span>,
    },
    {
      key: 'stage',
      header: 'Buying Stage',
      render: r => <span className="text-slate-400">{r.buying_stage || '--'}</span>,
    },
    {
      key: 'seats',
      header: 'Seats',
      sortable: true,
      sortValue: r => r.seat_count ?? 0,
      render: r => <span className="text-slate-300">{r.seat_count ?? '--'}</span>,
    },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Target className="h-6 w-6 text-amber-400" />
        <h1 className="text-2xl font-bold text-white">Lead Pipeline</h1>
        <span className="text-sm text-slate-500">{leads.length} leads</span>
      </div>

      <FilterBar>
        <FilterSelect
          label="Min Urgency"
          value={minUrgency}
          onChange={setMinUrgency}
          options={[
            { value: '5', label: '>= 5' },
            { value: '7', label: '>= 7 (High)' },
            { value: '8', label: '>= 8' },
            { value: '9', label: '>= 9 (Critical)' },
          ]}
          placeholder="Any"
        />
        <FilterSelect
          label="Time Window"
          value={windowDays}
          onChange={setWindowDays}
          options={[
            { value: '7', label: 'Last 7 days' },
            { value: '30', label: 'Last 30 days' },
            { value: '90', label: 'Last 90 days' },
            { value: '365', label: 'Last year' },
          ]}
        />
      </FilterBar>

      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl">
        <DataTable
          columns={columns}
          data={leads}
          onRowClick={r => navigate(`/b2b/leads/${encodeURIComponent(r.company)}`)}
          skeletonRows={loading ? 5 : undefined}
          emptyMessage="No high-intent leads found. Try lowering the urgency threshold."
        />
      </div>
    </div>
  )
}
