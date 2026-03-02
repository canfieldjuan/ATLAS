import { useState, useEffect } from 'react'
import { TrendingDown } from 'lucide-react'
import DataTable, { type Column } from '../../components/DataTable'
import { fetchDisplacement, type DisplacementFlow } from '../../api/b2bClient'

export default function CompetitorDisplacement() {
  const [flows, setFlows] = useState<DisplacementFlow[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchDisplacement(50)
      .then(r => setFlows(r.displacement))
      .catch(() => setFlows([]))
      .finally(() => setLoading(false))
  }, [])

  const columns: Column<DisplacementFlow>[] = [
    {
      key: 'vendor',
      header: 'From Vendor',
      render: r => <span className="text-white font-medium">{r.vendor_name}</span>,
    },
    {
      key: 'competitors',
      header: 'To Competitors',
      render: r => {
        const comps = Array.isArray(r.competitors) ? r.competitors : []
        return (
          <div className="flex flex-wrap gap-1">
            {comps.slice(0, 3).map((c: unknown, i: number) => (
              <span key={i} className="px-1.5 py-0.5 bg-amber-900/20 text-amber-400 text-xs rounded">
                {typeof c === 'string' ? c : JSON.stringify(c)}
              </span>
            ))}
            {comps.length > 3 && (
              <span className="text-xs text-slate-500">+{comps.length - 3}</span>
            )}
          </div>
        )
      },
    },
    {
      key: 'leaving',
      header: 'Intent to Leave',
      render: r => r.leaving
        ? <span className="text-red-400 text-xs font-medium">Yes</span>
        : <span className="text-slate-500 text-xs">No</span>,
    },
    {
      key: 'mentions',
      header: 'Mentions',
      sortable: true,
      sortValue: r => r.mention_count,
      render: r => <span className="text-slate-300">{r.mention_count}</span>,
    },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <TrendingDown className="h-6 w-6 text-red-400" />
        <h1 className="text-2xl font-bold text-white">Competitor Displacement</h1>
        <span className="text-sm text-slate-500">{flows.length} flows</span>
      </div>

      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl">
        <DataTable
          columns={columns}
          data={flows}
          skeletonRows={loading ? 5 : undefined}
          emptyMessage="No competitor displacement data found"
        />
      </div>
    </div>
  )
}
