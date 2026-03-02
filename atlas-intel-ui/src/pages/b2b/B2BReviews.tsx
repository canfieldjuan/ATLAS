import { useState, useEffect } from 'react'
import { MessageSquareText } from 'lucide-react'
import DataTable, { type Column } from '../../components/DataTable'
import FilterBar, { FilterSearch, FilterSelect } from '../../components/FilterBar'
import { fetchReviews, fetchReviewDetail, type B2BReview, type B2BReviewDetail } from '../../api/b2bClient'

export default function B2BReviews() {
  const [reviews, setReviews] = useState<B2BReview[]>([])
  const [loading, setLoading] = useState(true)
  const [painCategory, setPainCategory] = useState('')
  const [minUrgency, setMinUrgency] = useState('')
  const [company, setCompany] = useState('')
  const [churnOnly, setChurnOnly] = useState('')
  const [selected, setSelected] = useState<B2BReviewDetail | null>(null)

  useEffect(() => {
    setLoading(true)
    fetchReviews({
      pain_category: painCategory || undefined,
      min_urgency: minUrgency ? Number(minUrgency) : undefined,
      company: company || undefined,
      has_churn_intent: churnOnly === 'true' ? true : churnOnly === 'false' ? false : undefined,
      limit: 50,
    })
      .then(r => setReviews(r.reviews))
      .catch(() => setReviews([]))
      .finally(() => setLoading(false))
  }, [painCategory, minUrgency, company, churnOnly])

  const handleRowClick = async (r: B2BReview) => {
    try {
      const detail = await fetchReviewDetail(r.id)
      setSelected(detail)
    } catch {
      // ignore
    }
  }

  const columns: Column<B2BReview>[] = [
    {
      key: 'vendor',
      header: 'Vendor',
      render: r => <span className="text-white font-medium">{r.vendor_name}</span>,
    },
    {
      key: 'company',
      header: 'Company',
      render: r => <span className="text-slate-300">{r.reviewer_company || '--'}</span>,
    },
    {
      key: 'urgency',
      header: 'Urgency',
      sortable: true,
      sortValue: r => r.urgency_score ?? 0,
      render: r => {
        const v = r.urgency_score
        if (v == null) return <span className="text-slate-600">--</span>
        const color = v >= 7 ? 'text-red-400' : v >= 4 ? 'text-amber-400' : 'text-green-400'
        return <span className={color}>{v.toFixed(1)}</span>
      },
    },
    {
      key: 'pain',
      header: 'Pain',
      render: r => <span className="text-slate-400">{r.pain_category || '--'}</span>,
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
      header: 'DM',
      render: r => r.decision_maker
        ? <span className="text-green-400 text-xs">Yes</span>
        : <span className="text-slate-600 text-xs">No</span>,
    },
    {
      key: 'rating',
      header: 'Rating',
      render: r => <span className="text-slate-300">{r.rating?.toFixed(1) ?? '--'}</span>,
    },
  ]

  const activeFilters = [
    ...(painCategory ? [{ key: 'pain', label: `Pain: ${painCategory}`, onClear: () => setPainCategory('') }] : []),
    ...(minUrgency ? [{ key: 'urgency', label: `Urgency >= ${minUrgency}`, onClear: () => setMinUrgency('') }] : []),
    ...(company ? [{ key: 'company', label: `Company: ${company}`, onClear: () => setCompany('') }] : []),
    ...(churnOnly ? [{ key: 'churn', label: `Churn: ${churnOnly}`, onClear: () => setChurnOnly('') }] : []),
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <MessageSquareText className="h-6 w-6 text-cyan-400" />
        <h1 className="text-2xl font-bold text-white">B2B Reviews</h1>
        <span className="text-sm text-slate-500">{reviews.length} reviews</span>
      </div>

      <FilterBar
        activeFilters={activeFilters}
        onClearAll={() => { setPainCategory(''); setMinUrgency(''); setCompany(''); setChurnOnly('') }}
      >
        <FilterSearch
          label="Pain Category"
          value={painCategory}
          onChange={setPainCategory}
          placeholder="e.g. pricing"
          icon={false}
        />
        <FilterSelect
          label="Min Urgency"
          value={minUrgency}
          onChange={setMinUrgency}
          options={[
            { value: '3', label: '>= 3' },
            { value: '5', label: '>= 5' },
            { value: '7', label: '>= 7' },
          ]}
          placeholder="Any"
        />
        <FilterSearch
          label="Company"
          value={company}
          onChange={setCompany}
          placeholder="Filter by company"
        />
        <FilterSelect
          label="Churn Intent"
          value={churnOnly}
          onChange={setChurnOnly}
          options={[
            { value: 'true', label: 'Yes' },
            { value: 'false', label: 'No' },
          ]}
          placeholder="Any"
        />
      </FilterBar>

      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl">
        <DataTable
          columns={columns}
          data={reviews}
          onRowClick={handleRowClick}
          skeletonRows={loading ? 5 : undefined}
          emptyMessage="No reviews found"
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
              <h2 className="text-lg font-semibold text-white">{selected.vendor_name}</h2>
              <button onClick={() => setSelected(null)} className="text-slate-400 hover:text-white">X</button>
            </div>
            <div className="space-y-3 text-sm">
              <div className="grid grid-cols-2 gap-2">
                <div><span className="text-slate-500">Product:</span> <span className="text-white">{selected.product_name || '--'}</span></div>
                <div><span className="text-slate-500">Category:</span> <span className="text-white">{selected.product_category || '--'}</span></div>
                <div><span className="text-slate-500">Reviewer:</span> <span className="text-white">{selected.reviewer_name || 'Anonymous'}</span></div>
                <div><span className="text-slate-500">Title:</span> <span className="text-white">{selected.reviewer_title || '--'}</span></div>
                <div><span className="text-slate-500">Company:</span> <span className="text-white">{selected.reviewer_company || '--'}</span></div>
                <div><span className="text-slate-500">Industry:</span> <span className="text-white">{selected.reviewer_industry || '--'}</span></div>
                <div><span className="text-slate-500">Rating:</span> <span className="text-white">{selected.rating ?? '--'}</span></div>
                <div><span className="text-slate-500">Source:</span> <span className="text-white">{selected.source || '--'}</span></div>
              </div>
              {selected.summary && (
                <div>
                  <span className="text-slate-500 block mb-1">Summary:</span>
                  <p className="text-slate-300">{selected.summary}</p>
                </div>
              )}
              {selected.pros && (
                <div>
                  <span className="text-green-400 block mb-1">Pros:</span>
                  <p className="text-slate-300">{selected.pros}</p>
                </div>
              )}
              {selected.cons && (
                <div>
                  <span className="text-red-400 block mb-1">Cons:</span>
                  <p className="text-slate-300">{selected.cons}</p>
                </div>
              )}
              {selected.review_text && (
                <div>
                  <span className="text-slate-500 block mb-1">Full Review:</span>
                  <p className="text-slate-400 text-xs whitespace-pre-wrap">{selected.review_text}</p>
                </div>
              )}
              {selected.enrichment && (
                <div>
                  <span className="text-slate-500 block mb-1">Enrichment:</span>
                  <pre className="text-xs text-slate-400 bg-slate-900/50 rounded p-3 overflow-x-auto">
                    {JSON.stringify(selected.enrichment, null, 2)}
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
