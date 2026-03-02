import { useState, useEffect } from 'react'
import { Megaphone, Check, X as XIcon, Loader2 } from 'lucide-react'
import DataTable, { type Column } from '../../components/DataTable'
import FilterBar, { FilterSelect } from '../../components/FilterBar'
import {
  fetchCampaigns,
  fetchTrackedVendors,
  generateCampaigns,
  updateCampaign,
  type B2BCampaign,
  type TrackedVendor,
} from '../../api/b2bClient'

export default function B2BCampaigns() {
  const [campaigns, setCampaigns] = useState<B2BCampaign[]>([])
  const [vendors, setVendors] = useState<TrackedVendor[]>([])
  const [loading, setLoading] = useState(true)
  const [statusFilter, setStatusFilter] = useState('')
  const [generating, setGenerating] = useState(false)
  const [genVendor, setGenVendor] = useState('')
  const [genResult, setGenResult] = useState('')
  const [updating, setUpdating] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    try {
      const [c, v] = await Promise.all([
        fetchCampaigns({ status: statusFilter || undefined, limit: 50 }),
        fetchTrackedVendors(),
      ])
      setCampaigns(c.campaigns)
      setVendors(v.vendors)
    } catch {
      // handled by empty state
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [statusFilter])

  const handleGenerate = async () => {
    if (!genVendor) return
    setGenerating(true)
    setGenResult('')
    try {
      const r = await generateCampaigns(genVendor)
      setGenResult(`Created ${r.campaigns_created} campaign drafts`)
      load()
    } catch (err) {
      setGenResult(err instanceof Error ? err.message : 'Failed to generate')
    } finally {
      setGenerating(false)
    }
  }

  const handleAction = async (id: string, status: string) => {
    setUpdating(id)
    try {
      await updateCampaign(id, status)
      setCampaigns(prev =>
        prev.map(c => c.id === id ? { ...c, status, approved_at: status === 'approved' ? new Date().toISOString() : c.approved_at } : c)
      )
    } catch {
      // ignore
    } finally {
      setUpdating(null)
    }
  }

  const columns: Column<B2BCampaign>[] = [
    {
      key: 'company',
      header: 'Company',
      render: r => <span className="text-white font-medium">{r.company_name}</span>,
    },
    {
      key: 'vendor',
      header: 'Vendor',
      render: r => <span className="text-slate-300">{r.vendor_name}</span>,
    },
    {
      key: 'channel',
      header: 'Channel',
      render: r => <span className="text-slate-400">{r.channel}</span>,
    },
    {
      key: 'subject',
      header: 'Subject',
      render: r => <span className="text-slate-400 truncate max-w-xs block">{r.subject || '--'}</span>,
    },
    {
      key: 'status',
      header: 'Status',
      render: r => {
        const colors: Record<string, string> = {
          draft: 'text-slate-400 bg-slate-700/30',
          approved: 'text-green-400 bg-green-900/20',
          sent: 'text-cyan-400 bg-cyan-900/20',
          cancelled: 'text-red-400 bg-red-900/20',
        }
        return (
          <span className={`text-xs px-2 py-0.5 rounded ${colors[r.status] || colors.draft}`}>
            {r.status}
          </span>
        )
      },
    },
    {
      key: 'actions',
      header: '',
      render: r => {
        if (r.status !== 'draft') return null
        const isUpdating = updating === r.id
        return (
          <div className="flex items-center gap-1">
            <button
              onClick={e => { e.stopPropagation(); handleAction(r.id, 'approved') }}
              disabled={isUpdating}
              className="p-1 text-green-400 hover:text-green-300 disabled:opacity-50"
              title="Approve"
            >
              {isUpdating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
            </button>
            <button
              onClick={e => { e.stopPropagation(); handleAction(r.id, 'cancelled') }}
              disabled={isUpdating}
              className="p-1 text-red-400 hover:text-red-300 disabled:opacity-50"
              title="Cancel"
            >
              <XIcon className="h-4 w-4" />
            </button>
          </div>
        )
      },
    },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Megaphone className="h-6 w-6 text-violet-400" />
        <h1 className="text-2xl font-bold text-white">Campaigns</h1>
      </div>

      {/* Generate section */}
      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
        <h2 className="text-sm font-semibold text-slate-300 mb-3">Generate Campaign Drafts</h2>
        <div className="flex items-end gap-3">
          <div className="flex-1 max-w-xs">
            <label className="block text-xs text-slate-400 mb-1">Vendor</label>
            <select
              value={genVendor}
              onChange={e => setGenVendor(e.target.value)}
              className="w-full px-3 py-1.5 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white focus:outline-none focus:border-cyan-500/50"
            >
              <option value="">Select vendor</option>
              {vendors.map(v => (
                <option key={v.vendor_name} value={v.vendor_name}>{v.vendor_name}</option>
              ))}
            </select>
          </div>
          <button
            onClick={handleGenerate}
            disabled={!genVendor || generating}
            className="px-4 py-1.5 bg-violet-600 hover:bg-violet-500 disabled:opacity-50 rounded-lg text-white text-sm font-medium transition-colors"
          >
            {generating ? 'Generating...' : 'Generate'}
          </button>
        </div>
        {genResult && (
          <p className="mt-2 text-sm text-slate-400">{genResult}</p>
        )}
      </div>

      <FilterBar>
        <FilterSelect
          label="Status"
          value={statusFilter}
          onChange={setStatusFilter}
          options={[
            { value: 'draft', label: 'Draft' },
            { value: 'approved', label: 'Approved' },
            { value: 'sent', label: 'Sent' },
            { value: 'cancelled', label: 'Cancelled' },
          ]}
          placeholder="All"
        />
      </FilterBar>

      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl">
        <DataTable
          columns={columns}
          data={campaigns}
          skeletonRows={loading ? 5 : undefined}
          emptyMessage="No campaigns yet. Generate some from your tracked vendors."
        />
      </div>
    </div>
  )
}
