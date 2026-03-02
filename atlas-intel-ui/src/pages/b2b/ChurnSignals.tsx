import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Activity } from 'lucide-react'
import DataTable, { type Column } from '../../components/DataTable'
import FilterBar, { FilterSearch, FilterSelect } from '../../components/FilterBar'
import { fetchSignals, type ChurnSignal } from '../../api/b2bClient'

export default function ChurnSignals() {
  const navigate = useNavigate()
  const [signals, setSignals] = useState<ChurnSignal[]>([])
  const [loading, setLoading] = useState(true)
  const [minUrgency, setMinUrgency] = useState('')
  const [category, setCategory] = useState('')

  useEffect(() => {
    setLoading(true)
    fetchSignals({
      min_urgency: minUrgency ? Number(minUrgency) : undefined,
      category: category || undefined,
    })
      .then(r => setSignals(r.signals))
      .catch(() => setSignals([]))
      .finally(() => setLoading(false))
  }, [minUrgency, category])

  const columns: Column<ChurnSignal>[] = [
    {
      key: 'vendor',
      header: 'Vendor',
      render: r => <span className="text-white font-medium">{r.vendor_name}</span>,
    },
    {
      key: 'category',
      header: 'Category',
      render: r => <span className="text-slate-400">{r.product_category || '--'}</span>,
    },
    {
      key: 'urgency',
      header: 'Urgency',
      sortable: true,
      sortValue: r => r.avg_urgency_score,
      render: r => {
        const v = r.avg_urgency_score
        const color = v >= 7 ? 'text-red-400' : v >= 4 ? 'text-amber-400' : 'text-green-400'
        return <span className={color}>{v.toFixed(1)}</span>
      },
    },
    {
      key: 'churn',
      header: 'Churn Intent',
      sortable: true,
      sortValue: r => r.churn_intent_count ?? 0,
      render: r => <span className="text-slate-300">{r.churn_intent_count ?? 0}</span>,
    },
    {
      key: 'reviews',
      header: 'Reviews',
      sortable: true,
      sortValue: r => r.total_reviews ?? 0,
      render: r => <span className="text-slate-300">{r.total_reviews ?? 0}</span>,
    },
    {
      key: 'nps',
      header: 'NPS Proxy',
      sortable: true,
      sortValue: r => r.nps_proxy ?? 0,
      render: r => {
        const v = r.nps_proxy
        if (v == null) return <span className="text-slate-600">--</span>
        const color = v < -20 ? 'text-red-400' : v < 20 ? 'text-amber-400' : 'text-green-400'
        return <span className={color}>{v.toFixed(0)}</span>
      },
    },
    {
      key: 'price',
      header: 'Price Complaints',
      sortable: true,
      sortValue: r => r.price_complaint_rate ?? 0,
      render: r => {
        const v = r.price_complaint_rate
        if (v == null) return <span className="text-slate-600">--</span>
        return <span className="text-slate-300">{(v * 100).toFixed(0)}%</span>
      },
    },
  ]

  const urgencyOptions = [
    { value: '3', label: '>= 3' },
    { value: '5', label: '>= 5' },
    { value: '7', label: '>= 7 (High)' },
    { value: '9', label: '>= 9 (Critical)' },
  ]

  const activeFilters = [
    ...(minUrgency ? [{ key: 'urgency', label: `Urgency >= ${minUrgency}`, onClear: () => setMinUrgency('') }] : []),
    ...(category ? [{ key: 'cat', label: `Category: ${category}`, onClear: () => setCategory('') }] : []),
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Activity className="h-6 w-6 text-cyan-400" />
        <h1 className="text-2xl font-bold text-white">Churn Signals</h1>
      </div>

      <FilterBar
        activeFilters={activeFilters}
        onClearAll={() => { setMinUrgency(''); setCategory('') }}
      >
        <FilterSelect
          label="Min Urgency"
          value={minUrgency}
          onChange={setMinUrgency}
          options={urgencyOptions}
          placeholder="Any"
        />
        <FilterSearch
          label="Category"
          value={category}
          onChange={setCategory}
          placeholder="e.g. CRM"
          icon={false}
        />
      </FilterBar>

      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl">
        <DataTable
          columns={columns}
          data={signals}
          onRowClick={r => navigate(`/b2b/signals/${encodeURIComponent(r.vendor_name)}`)}
          skeletonRows={loading ? 5 : undefined}
          emptyMessage="No churn signals found for your tracked vendors"
        />
      </div>
    </div>
  )
}
