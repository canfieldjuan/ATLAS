import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, Building2 } from 'lucide-react'
import DataTable, { type Column } from '../../components/DataTable'
import { fetchLeadDetail, type LeadDetail as LeadDetailType } from '../../api/b2bClient'

type ReviewRow = LeadDetailType['reviews'][number]

export default function LeadDetail() {
  const { company } = useParams<{ company: string }>()
  const navigate = useNavigate()
  const [detail, setDetail] = useState<LeadDetailType | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!company) return
    setLoading(true)
    fetchLeadDetail(company)
      .then(d => setDetail(d))
      .catch(err => setError(err instanceof Error ? err.message : 'Failed to load lead'))
      .finally(() => setLoading(false))
  }, [company])

  if (error) {
    return (
      <div className="space-y-4">
        <button onClick={() => navigate(-1)} className="flex items-center gap-1 text-sm text-slate-400 hover:text-white">
          <ArrowLeft className="h-4 w-4" /> Back
        </button>
        <p className="text-red-400">{error}</p>
      </div>
    )
  }

  const columns: Column<ReviewRow>[] = [
    {
      key: 'vendor',
      header: 'Vendor',
      render: r => <span className="text-white">{r.vendor_name}</span>,
    },
    {
      key: 'urgency',
      header: 'Urgency',
      sortable: true,
      sortValue: r => r.urgency,
      render: r => {
        const color = r.urgency >= 8 ? 'text-red-400' : 'text-amber-400'
        return <span className={color}>{r.urgency.toFixed(1)}</span>
      },
    },
    {
      key: 'pain',
      header: 'Pain',
      render: r => <span className="text-slate-400">{r.pain || '--'}</span>,
    },
    {
      key: 'intent',
      header: 'Leaving',
      render: r => r.intent_to_leave
        ? <span className="text-red-400 text-xs">Yes</span>
        : <span className="text-slate-600 text-xs">No</span>,
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
      key: 'role',
      header: 'Role',
      render: r => <span className="text-slate-400">{r.role_level || '--'}</span>,
    },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <button onClick={() => navigate(-1)} className="text-slate-400 hover:text-white">
          <ArrowLeft className="h-5 w-5" />
        </button>
        <Building2 className="h-6 w-6 text-amber-400" />
        <h1 className="text-2xl font-bold text-white">{company}</h1>
        {detail && (
          <span className="text-sm text-slate-500">{detail.count} reviews</span>
        )}
      </div>

      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl">
        <DataTable
          columns={columns}
          data={detail?.reviews ?? []}
          skeletonRows={loading ? 5 : undefined}
          emptyMessage="No reviews found for this company"
        />
      </div>
    </div>
  )
}
