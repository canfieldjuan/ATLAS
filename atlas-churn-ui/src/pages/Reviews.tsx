import { useState, useEffect, useMemo } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { Search, RefreshCw, X, Loader2, Download } from 'lucide-react'
import { clsx } from 'clsx'
import DataTable, { type Column } from '../components/DataTable'
import UrgencyBadge from '../components/UrgencyBadge'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import { fetchReviews, downloadCsv } from '../api/client'
import type { ReviewSummary } from '../types'

const REVIEW_WINDOW_DAYS = 365

function reviewsPath(vendor: string, company: string, minUrgency: number, churnOnly: boolean) {
  const next = new URLSearchParams()
  if (vendor.trim()) next.set('vendor', vendor.trim())
  if (company.trim()) next.set('company', company.trim())
  if (minUrgency > 0) next.set('min_urgency', String(minUrgency))
  if (churnOnly) next.set('churn_only', 'true')
  const qs = next.toString()
  return qs ? `/reviews?${qs}` : '/reviews'
}

function reviewDetailPath(reviewId: string, backTo: string) {
  const next = new URLSearchParams()
  if (backTo !== '/reviews') next.set('back_to', backTo)
  const qs = next.toString()
  const base = `/reviews/${encodeURIComponent(reviewId)}`
  return qs ? `${base}?${qs}` : base
}

function vendorDetailPath(vendorName: string, backTo: string) {
  const next = new URLSearchParams()
  if (backTo !== '/reviews') next.set('back_to', backTo)
  const qs = next.toString()
  const base = `/vendors/${encodeURIComponent(vendorName)}`
  return qs ? `${base}?${qs}` : base
}

function evidencePath(vendorName: string, backTo: string) {
  const next = new URLSearchParams()
  next.set('vendor', vendorName)
  next.set('tab', 'witnesses')
  next.set('back_to', backTo)
  return `/evidence?${next.toString()}`
}

function reportsPath(vendorName: string, backTo: string) {
  const next = new URLSearchParams()
  next.set('vendor_filter', vendorName)
  next.set('back_to', backTo)
  return `/reports?${next.toString()}`
}

function opportunitiesPath(vendorName: string, backTo: string) {
  const next = new URLSearchParams()
  next.set('vendor', vendorName)
  next.set('back_to', backTo)
  return `/opportunities?${next.toString()}`
}

export default function Reviews() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const [vendor, setVendor] = useState(() => searchParams.get('vendor') ?? '')
  const [debouncedVendor, setDebouncedVendor] = useState(() => searchParams.get('vendor') ?? '')
  const [company, setCompany] = useState(() => searchParams.get('company') ?? '')
  const [debouncedCompany, setDebouncedCompany] = useState(() => searchParams.get('company') ?? '')
  const [minUrgency, setMinUrgency] = useState(() => Number(searchParams.get('min_urgency') ?? '0') || 0)
  const [churnOnly, setChurnOnly] = useState(() => searchParams.get('churn_only') === 'true')

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedVendor(vendor)
      setDebouncedCompany(company)
    }, 300)
    return () => clearTimeout(timer)
  }, [vendor, company])

  useEffect(() => {
    const next = new URLSearchParams(searchParams)
    if (debouncedVendor.trim()) next.set('vendor', debouncedVendor.trim())
    else next.delete('vendor')
    if (debouncedCompany.trim()) next.set('company', debouncedCompany.trim())
    else next.delete('company')
    if (minUrgency > 0) next.set('min_urgency', String(minUrgency))
    else next.delete('min_urgency')
    if (churnOnly) next.set('churn_only', 'true')
    else next.delete('churn_only')
    if (next.toString() === searchParams.toString()) return
    setSearchParams(next, { replace: true })
  }, [churnOnly, debouncedCompany, debouncedVendor, minUrgency, searchParams, setSearchParams])

  const { data, loading, error, refresh, refreshing } = useApiData(
    () =>
      fetchReviews({
        vendor_name: debouncedVendor || undefined,
        company: debouncedCompany || undefined,
        min_urgency: minUrgency || undefined,
        has_churn_intent: churnOnly || undefined,
        window_days: REVIEW_WINDOW_DAYS,
        limit: 100,
      }),
    [debouncedVendor, debouncedCompany, minUrgency, churnOnly],
  )

  const reviews = data?.reviews ?? []
  const hasFilters = vendor !== '' || company !== '' || minUrgency > 0 || churnOnly
  const debouncePending = vendor !== debouncedVendor || company !== debouncedCompany
  const currentListPath = useMemo(
    () => reviewsPath(debouncedVendor, debouncedCompany, minUrgency, churnOnly),
    [churnOnly, debouncedCompany, debouncedVendor, minUrgency],
  )

  function clearFilters() {
    setVendor('')
    setDebouncedVendor('')
    setCompany('')
    setDebouncedCompany('')
    setMinUrgency(0)
    setChurnOnly(false)
  }

  const columns: Column<ReviewSummary>[] = [
    {
      key: 'vendor',
      header: 'Vendor',
      render: (r) => <span className="text-white font-medium">{r.vendor_name}</span>,
      sortable: true,
      sortValue: (r) => r.vendor_name,
    },
    {
      key: 'company',
      header: 'Company',
      render: (r) => (
        <div>
          <span className="text-slate-300">{r.reviewer_company ?? '--'}</span>
          {r.reviewer_title && (
            <span className="block text-[10px] text-slate-500">{r.reviewer_title}</span>
          )}
        </div>
      ),
    },
    {
      key: 'source',
      header: 'Source',
      render: (r) => <span className="text-slate-400 text-xs">{r.source ?? '--'}</span>,
    },
    {
      key: 'urgency',
      header: 'Urgency',
      render: (r) => <UrgencyBadge score={r.urgency_score} />,
      sortable: true,
      sortValue: (r) => r.urgency_score ?? 0,
    },
    {
      key: 'pain',
      header: 'Pain',
      render: (r) => <span className="text-slate-400">{r.pain_category ?? '--'}</span>,
    },
    {
      key: 'sentiment',
      header: 'Sentiment',
      render: (r) => {
        if (!r.sentiment_direction) return <span className="text-slate-500 text-xs">--</span>
        const colors: Record<string, string> = {
          improving: 'text-green-400',
          declining: 'text-red-400',
          stable: 'text-slate-400',
        }
        return (
          <span className={`text-xs font-medium ${colors[r.sentiment_direction] ?? 'text-slate-400'}`}>
            {r.sentiment_direction}
          </span>
        )
      },
    },
    {
      key: 'rating',
      header: 'Rating',
      render: (r) => <span className="text-slate-300">{r.rating?.toFixed(1) ?? '--'}</span>,
      sortable: true,
      sortValue: (r) => r.rating ?? 0,
    },
    {
      key: 'authority',
      header: 'Authority',
      render: (r) => {
        if (!r.decision_maker && !r.role_level) return <span className="text-slate-500 text-xs">--</span>
        return (
          <div className="space-y-0.5">
            {r.decision_maker && <span className="text-cyan-400 text-[10px] font-medium block">DM</span>}
            {r.role_level && <span className="text-slate-400 text-[10px] block">{r.role_level}</span>}
          </div>
        )
      },
    },
    {
      key: 'intent',
      header: 'Intent',
      render: (r) =>
        r.intent_to_leave ? (
          <span className="text-red-400 text-xs font-medium">Leaving</span>
        ) : (
          <span className="text-slate-500 text-xs">--</span>
        ),
    },
    {
      key: 'competitors',
      header: 'Competitors',
      render: (r) => {
        const comps = r.competitors_mentioned ?? []
        if (comps.length === 0) return <span className="text-slate-500 text-xs">--</span>
        return (
          <div className="flex flex-wrap gap-1">
            {comps.slice(0, 2).map((c, i) => {
              const name = typeof c === 'string' ? c : (c as Record<string, unknown>).name as string ?? ''
              return (
                <span key={i} className="px-1.5 py-0.5 bg-slate-800 rounded text-[10px] text-slate-300">
                  {name}
                </span>
              )
            })}
            {comps.length > 2 && <span className="text-[10px] text-slate-500">+{comps.length - 2}</span>}
          </div>
        )
      },
    },
    {
      key: 'actions',
      header: 'Actions',
      render: (r) => (
        <div className="flex items-center gap-3 text-xs">
          <Link
            to={vendorDetailPath(r.vendor_name, currentListPath)}
            onClick={(event) => event.stopPropagation()}
            className="text-cyan-400 hover:text-cyan-300 transition-colors"
          >
            Vendor
          </Link>
          <Link
            to={evidencePath(r.vendor_name, currentListPath)}
            onClick={(event) => event.stopPropagation()}
            className="text-violet-300 hover:text-violet-200 transition-colors"
          >
            Evidence
          </Link>
          <Link
            to={reportsPath(r.vendor_name, currentListPath)}
            onClick={(event) => event.stopPropagation()}
            className="text-fuchsia-300 hover:text-fuchsia-200 transition-colors"
          >
            Reports
          </Link>
          <Link
            to={opportunitiesPath(r.vendor_name, currentListPath)}
            onClick={(event) => event.stopPropagation()}
            className="text-emerald-300 hover:text-emerald-200 transition-colors"
          >
            Opportunities
          </Link>
        </div>
      ),
    },
  ]

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Reviews</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={() =>
              downloadCsv('/export/reviews', {
                vendor_name: debouncedVendor || undefined,
                company: debouncedCompany || undefined,
                min_urgency: minUrgency || undefined,
                has_churn_intent: churnOnly || undefined,
                window_days: REVIEW_WINDOW_DAYS,
              })
            }
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors"
          >
            <Download className="h-4 w-4" />
            Export CSV
          </button>
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

      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 flex-wrap">
        <div className="relative flex-1 max-w-xs w-full">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
          <input
            type="text"
            placeholder="Filter by vendor..."
            value={vendor}
            onChange={(e) => setVendor(e.target.value)}
            className="w-full pl-9 pr-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
          />
        </div>
        <div className="relative flex-1 max-w-xs w-full">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
          <input
            type="text"
            placeholder="Filter by company..."
            value={company}
            onChange={(e) => setCompany(e.target.value)}
            className="w-full pl-9 pr-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
          />
        </div>
        <div className="flex items-center gap-2 text-sm text-slate-400">
          <label>Min Urgency</label>
          <input
            type="range"
            min={0}
            max={10}
            step={0.5}
            value={minUrgency}
            onChange={(e) => setMinUrgency(Number(e.target.value))}
            className="w-20 accent-cyan-400"
          />
          <span className="text-white w-6">{minUrgency}</span>
        </div>
        <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer">
          <input
            type="checkbox"
            checked={churnOnly}
            onChange={(e) => setChurnOnly(e.target.checked)}
            className="accent-cyan-400"
          />
          Churn intent only
        </label>
        {hasFilters && (
          <button
            onClick={clearFilters}
            className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors"
          >
            <X className="h-3 w-3" />
            Clear filters
          </button>
        )}
      </div>

      <div className="flex items-center gap-2 text-sm text-slate-400">
        {debouncePending || loading ? (
          <>
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Searching...
          </>
        ) : (
          <span>{reviews.length} review{reviews.length !== 1 ? 's' : ''} found</span>
        )}
      </div>

      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl overflow-hidden">
        {loading ? (
          <DataTable columns={columns} data={[]} skeletonRows={8} />
        ) : (
          <DataTable
            columns={columns}
            data={reviews}
            onRowClick={(r) => navigate(reviewDetailPath(r.id, currentListPath))}
            emptyMessage="No reviews match your filters"
            emptyAction={hasFilters ? { label: 'Clear all filters', onClick: clearFilters } : undefined}
          />
        )}
      </div>
    </div>
  )
}
