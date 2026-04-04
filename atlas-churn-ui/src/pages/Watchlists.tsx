import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Activity,
  BellRing,
  Building2,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  X,
} from 'lucide-react'
import { clsx } from 'clsx'
import StatCard from '../components/StatCard'
import DataTable, { type Column } from '../components/DataTable'
import UrgencyBadge from '../components/UrgencyBadge'
import ArchetypeBadge from '../components/ArchetypeBadge'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import {
  addTrackedVendor,
  fetchAccountsInMotionFeed,
  fetchSlowBurnWatchlist,
  listTrackedVendors,
  removeTrackedVendor,
  searchAvailableVendors,
  type AccountsInMotionFeedItem,
  type TrackedVendor,
  type VendorSearchResult,
} from '../api/client'
import type { ChurnSignal } from '../types'

interface WatchlistsData {
  vendors: TrackedVendor[]
  feed: ChurnSignal[]
  accounts: AccountsInMotionFeedItem[]
  vendorsWithAccounts: number
  freshestAccountsReportDate: string | null
}

const SEARCH_DEBOUNCE_MS = 250
const MIN_VENDOR_SEARCH_CHARS = 2
const SEARCH_RESULTS_PREVIEW_LIMIT = 8
const STALE_AFTER_HOURS = 24

function formatTs(value: string | null | undefined) {
  if (!value) return '--'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '--'
  return date.toLocaleString()
}

function toTimestamp(value: string | null | undefined) {
  if (!value) return null
  const ts = new Date(value).getTime()
  return Number.isNaN(ts) ? null : ts
}

function freshnessTone(value: string | null | undefined) {
  const ts = toTimestamp(value)
  if (ts == null) return 'text-slate-500'
  const ageHours = (Date.now() - ts) / (1000 * 60 * 60)
  return ageHours > STALE_AFTER_HOURS ? 'text-amber-400' : 'text-emerald-400'
}

export default function Watchlists() {
  const navigate = useNavigate()
  const [searchInput, setSearchInput] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [trackMode, setTrackMode] = useState<'own' | 'competitor'>('competitor')
  const [label, setLabel] = useState('')
  const [submittingVendor, setSubmittingVendor] = useState<string | null>(null)
  const [removingVendor, setRemovingVendor] = useState<string | null>(null)
  const [actionMessage, setActionMessage] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchInput.trim())
    }, SEARCH_DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [searchInput])

  const { data, loading, error, refresh, refreshing } = useApiData<WatchlistsData>(
    async () => {
      const [tracked, feed, accounts] = await Promise.all([
        listTrackedVendors(),
        fetchSlowBurnWatchlist(),
        fetchAccountsInMotionFeed(),
      ])
      return {
        vendors: tracked.vendors,
        feed: feed.signals,
        accounts: accounts.accounts,
        vendorsWithAccounts: accounts.vendors_with_accounts,
        freshestAccountsReportDate: accounts.freshest_report_date,
      }
    },
    [],
  )

  const {
    data: searchResults,
    loading: searchLoading,
  } = useApiData<{ vendors: VendorSearchResult[]; count: number }>(
    async () => {
      if (debouncedSearch.length < MIN_VENDOR_SEARCH_CHARS) return { vendors: [], count: 0 }
      return searchAvailableVendors(debouncedSearch)
    },
    [debouncedSearch],
    { refreshOnFocus: false, refreshOnReconnect: false },
  )

  const trackedVendors = data?.vendors ?? []
  const feed = data?.feed ?? []
  const accounts = data?.accounts ?? []
  const vendorsWithAccounts = data?.vendorsWithAccounts ?? 0
  const trackedNames = useMemo(
    () => new Set(trackedVendors.map((vendor) => vendor.vendor_name.toLowerCase())),
    [trackedVendors],
  )
  const candidateResults = (searchResults?.vendors ?? []).filter(
    (vendor) => !trackedNames.has(vendor.vendor_name.toLowerCase()),
  )
  const freshestAccountsReportDate = data?.freshestAccountsReportDate ?? null

  async function handleAddVendor(vendorName: string) {
    setSubmittingVendor(vendorName)
    setActionError(null)
    setActionMessage(null)
    try {
      await addTrackedVendor(vendorName, trackMode, label.trim())
      setActionMessage(`${vendorName} added to watchlists`)
      setSearchInput('')
      setDebouncedSearch('')
      setLabel('')
      refresh()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to add vendor')
    } finally {
      setSubmittingVendor(null)
    }
  }

  async function handleRemoveVendor(vendorName: string) {
    if (!confirm(`Remove ${vendorName} from your watchlists?`)) return
    setRemovingVendor(vendorName)
    setActionError(null)
    setActionMessage(null)
    try {
      await removeTrackedVendor(vendorName)
      setActionMessage(`${vendorName} removed from watchlists`)
      refresh()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to remove vendor')
    } finally {
      setRemovingVendor(null)
    }
  }

  const trackedColumns: Column<TrackedVendor>[] = [
    {
      key: 'vendor',
      header: 'Vendor',
      render: (row) => (
        <div>
          <div className="font-medium text-white">{row.vendor_name}</div>
          <div className="text-xs text-slate-500">{row.label || row.track_mode.replace(/_/g, ' ')}</div>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.vendor_name,
    },
    {
      key: 'mode',
      header: 'Mode',
      render: (row) => (
        <span className="inline-flex rounded-full bg-slate-800 px-2 py-0.5 text-xs text-slate-300">
          {row.track_mode.replace(/_/g, ' ')}
        </span>
      ),
      sortable: true,
      sortValue: (row) => row.track_mode,
    },
    {
      key: 'urgency',
      header: 'Avg Urgency',
      render: (row) => <UrgencyBadge score={row.avg_urgency ?? 0} />,
      sortable: true,
      sortValue: (row) => row.avg_urgency ?? 0,
    },
    {
      key: 'reviews',
      header: 'Reviews',
      render: (row) => <span className="text-slate-300">{row.total_reviews ?? '--'}</span>,
      sortable: true,
      sortValue: (row) => row.total_reviews ?? 0,
    },
    {
      key: 'nps',
      header: 'NPS Proxy',
      render: (row) => <span className="text-slate-300">{row.nps_proxy?.toFixed(1) ?? '--'}</span>,
      sortable: true,
      sortValue: (row) => row.nps_proxy ?? -999,
    },
    {
      key: 'actions',
      header: 'Actions',
      render: (row) => (
        <div className="flex items-center justify-end gap-2">
          <button
            onClick={(event) => {
              event.stopPropagation()
              navigate(`/vendors/${encodeURIComponent(row.vendor_name)}`)
            }}
            className="rounded-md bg-cyan-500/10 px-2.5 py-1 text-xs font-medium text-cyan-300 hover:bg-cyan-500/20"
          >
            View
          </button>
          <button
            onClick={(event) => {
              event.stopPropagation()
              handleRemoveVendor(row.vendor_name)
            }}
            disabled={removingVendor === row.vendor_name}
            className="rounded-md bg-rose-500/10 px-2.5 py-1 text-xs font-medium text-rose-300 hover:bg-rose-500/20 disabled:opacity-50"
          >
            <span className="inline-flex items-center gap-1">
              <Trash2 className="h-3 w-3" />
              Remove
            </span>
          </button>
        </div>
      ),
    },
  ]

  const feedColumns: Column<ChurnSignal>[] = [
    {
      key: 'vendor',
      header: 'Vendor Movement',
      render: (row) => (
        <div>
          <div className="font-medium text-white">{row.vendor_name}</div>
          <div className="text-xs text-slate-500">{row.product_category ?? 'Uncategorized'}</div>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.vendor_name,
    },
    {
      key: 'urgency',
      header: 'Urgency',
      render: (row) => <UrgencyBadge score={row.avg_urgency_score} />,
      sortable: true,
      sortValue: (row) => row.avg_urgency_score,
    },
    {
      key: 'archetype',
      header: 'Wedge',
      render: (row) => (
        <ArchetypeBadge
          archetype={row.archetype}
          confidence={row.archetype_confidence}
          showConfidence
        />
      ),
      sortable: true,
      sortValue: (row) => row.archetype ?? '',
    },
    {
      key: 'support',
      header: 'Support',
      render: (row) => <span className="text-slate-300">{row.support_sentiment?.toFixed(1) ?? '--'}</span>,
      sortable: true,
      sortValue: (row) => row.support_sentiment ?? -999,
    },
    {
      key: 'growth',
      header: 'Growth',
      render: (row) => (
        <span className="text-slate-300">
          {row.employee_growth_rate != null ? `${row.employee_growth_rate >= 0 ? '+' : ''}${row.employee_growth_rate.toFixed(1)}%` : '--'}
        </span>
      ),
      sortable: true,
      sortValue: (row) => row.employee_growth_rate ?? -999,
    },
    {
      key: 'freshness',
      header: 'Last Updated',
      render: (row) => (
        <span className={clsx('text-xs', freshnessTone(row.last_computed_at))}>
          {formatTs(row.last_computed_at)}
        </span>
      ),
      sortable: true,
      sortValue: (row) => toTimestamp(row.last_computed_at) ?? 0,
    },
  ]

  const accountColumns: Column<AccountsInMotionFeedItem>[] = [
    {
      key: 'company',
      header: 'Account Movement',
      render: (row) => (
        <div>
          <div className="font-medium text-white">{row.company || 'Unknown account'}</div>
          <div className="text-xs text-slate-500">{row.vendor}</div>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.company || '',
    },
    {
      key: 'urgency',
      header: 'Urgency',
      render: (row) => <UrgencyBadge score={row.urgency} />,
      sortable: true,
      sortValue: (row) => row.urgency,
    },
    {
      key: 'timing',
      header: 'Timing',
      render: (row) => <span className="text-slate-300">{row.contract_signal || '--'}</span>,
      sortable: true,
      sortValue: (row) => row.contract_signal || '',
    },
    {
      key: 'pain',
      header: 'Pain',
      render: (row) => <span className="text-slate-300">{row.pain_categories[0]?.category || '--'}</span>,
      sortable: true,
      sortValue: (row) => row.pain_categories[0]?.category || '',
    },
    {
      key: 'quote',
      header: 'Evidence',
      render: (row) => (
        <span className="max-w-[320px] truncate text-slate-400">
          {row.evidence[0] || '--'}
        </span>
      ),
    },
    {
      key: 'report_date',
      header: 'Freshness',
      render: (row) => (
        <div className="text-xs">
          <div className={clsx(freshnessTone(row.report_date))}>{formatTs(row.report_date)}</div>
          <div className="text-slate-500">{row.is_stale ? 'stale report' : 'persisted report'}</div>
        </div>
      ),
      sortable: true,
      sortValue: (row) => toTimestamp(row.report_date) ?? 0,
    },
  ]

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Watchlists</h1>
          <p className="mt-1 max-w-3xl text-sm text-slate-400">
            Track the vendors that matter, monitor movement across the slow-burn feed, and jump directly into vendor detail for evidence-backed review.
          </p>
        </div>
        <button
          onClick={refresh}
          disabled={refreshing}
          className="inline-flex items-center gap-2 self-start rounded-lg px-3 py-1.5 text-sm text-slate-400 transition-colors hover:bg-slate-800/50 hover:text-white disabled:opacity-50"
        >
          <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Tracked Vendors"
          value={trackedVendors.length}
          icon={<Building2 className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Competitors"
          value={trackedVendors.filter((vendor) => vendor.track_mode === 'competitor').length}
          icon={<BellRing className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Feed Rows"
          value={feed.length}
          icon={<Activity className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Accounts In Motion"
          value={accounts.length}
          sub={vendorsWithAccounts > 0 ? `${vendorsWithAccounts} vendors with active account movement` : 'No persisted account movement yet'}
          icon={<RefreshCw className="h-5 w-5" />}
          skeleton={loading}
        />
      </div>

      {(actionMessage || actionError) && (
        <div
          className={clsx(
            'flex items-start justify-between gap-3 rounded-xl border px-4 py-3 text-sm',
            actionError
              ? 'border-rose-500/30 bg-rose-500/10 text-rose-200'
              : 'border-emerald-500/30 bg-emerald-500/10 text-emerald-200',
          )}
        >
          <span>{actionError || actionMessage}</span>
          <button
            onClick={() => {
              setActionError(null)
              setActionMessage(null)
            }}
            className="text-current/70 hover:text-current"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      <div className="grid gap-6 xl:grid-cols-[1.1fr,0.9fr]">
        <div className="space-y-6">
          <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 overflow-hidden">
            <div className="border-b border-slate-700/50 px-4 py-3">
              <h2 className="text-sm font-medium text-white">Vendor Movement Feed</h2>
              <p className="text-xs text-slate-500">
                Existing slow-burn watchlist data for your tracked vendors. Click a row to inspect the vendor surface.
              </p>
            </div>
            <DataTable
              columns={feedColumns}
              data={feed}
              onRowClick={(row) => navigate(`/vendors/${encodeURIComponent(row.vendor_name)}`)}
              emptyMessage="No watchlist movement yet. Add tracked vendors to start monitoring."
              skeletonRows={loading ? 6 : undefined}
            />
          </div>

          <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 overflow-hidden">
            <div className="border-b border-slate-700/50 px-4 py-3">
              <h2 className="text-sm font-medium text-white">Tracked Vendors</h2>
              <p className="text-xs text-slate-500">
                Your current monitored portfolio with direct links into the existing vendor detail surface.
              </p>
            </div>
            <DataTable
              columns={trackedColumns}
              data={trackedVendors}
              onRowClick={(row) => navigate(`/vendors/${encodeURIComponent(row.vendor_name)}`)}
              emptyMessage="No tracked vendors yet."
              emptyAction={{ label: 'Add your first vendor', onClick: () => document.getElementById('watchlist-search')?.focus() }}
              skeletonRows={loading ? 5 : undefined}
            />
          </div>

          <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 overflow-hidden">
            <div className="border-b border-slate-700/50 px-4 py-3">
              <h2 className="text-sm font-medium text-white">Accounts In Motion</h2>
              <p className="text-xs text-slate-500">
                Tenant-wide account movement aggregated from persisted accounts-in-motion reports across your tracked vendors.
              </p>
              {freshestAccountsReportDate ? (
                <p className={clsx('mt-1 text-[11px]', freshnessTone(freshestAccountsReportDate))}>
                  Freshest report {formatTs(freshestAccountsReportDate)}
                </p>
              ) : null}
            </div>
            <DataTable
              columns={accountColumns}
              data={accounts}
              onRowClick={(row) => navigate(`/vendors/${encodeURIComponent(row.vendor)}`)}
              emptyMessage="No persisted accounts-in-motion rows yet for your tracked vendors."
              skeletonRows={loading ? 5 : undefined}
            />
          </div>
        </div>

        <div className="space-y-6">
          <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 p-4">
            <div className="mb-3">
              <h2 className="text-sm font-medium text-white">Add Vendor To Watchlists</h2>
              <p className="mt-1 text-xs text-slate-500">
                Search existing vendors, choose whether you track them as your own footprint or a competitor, then add them to your monitored set.
              </p>
            </div>

            <div className="space-y-3">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                <input
                  id="watchlist-search"
                  type="text"
                  placeholder="Search vendors..."
                  value={searchInput}
                  onChange={(event) => setSearchInput(event.target.value)}
                  className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 py-2 pl-9 pr-3 text-sm text-white placeholder-slate-500 focus:border-cyan-500/50 focus:outline-none"
                />
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <label className="space-y-1">
                  <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Track Mode</span>
                  <select
                    value={trackMode}
                    onChange={(event) => setTrackMode(event.target.value as 'own' | 'competitor')}
                    className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white focus:border-cyan-500/50 focus:outline-none"
                  >
                    <option value="competitor">Competitor</option>
                    <option value="own">Own vendor</option>
                  </select>
                </label>
                <label className="space-y-1">
                  <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Label</span>
                  <input
                    type="text"
                    value={label}
                    onChange={(event) => setLabel(event.target.value)}
                    placeholder="Optional team label"
                    className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-500/50 focus:outline-none"
                  />
                </label>
              </div>
            </div>

            <div className="mt-4 space-y-2">
              {debouncedSearch.length < MIN_VENDOR_SEARCH_CHARS ? (
                <div className="rounded-lg border border-dashed border-slate-700/50 px-3 py-4 text-sm text-slate-500">
                  Type at least {MIN_VENDOR_SEARCH_CHARS} characters to search available vendors.
                </div>
              ) : searchLoading ? (
                <div className="rounded-lg border border-slate-700/50 bg-slate-950/40 px-3 py-4 text-sm text-slate-400">
                  Searching vendors...
                </div>
              ) : candidateResults.length === 0 ? (
                <div className="rounded-lg border border-slate-700/50 bg-slate-950/40 px-3 py-4 text-sm text-slate-400">
                  No untracked vendor matches found for this query.
                </div>
              ) : (
                <div className="space-y-2">
                  {candidateResults.slice(0, SEARCH_RESULTS_PREVIEW_LIMIT).map((vendor) => (
                    <div
                      key={vendor.vendor_name}
                      className="flex items-center justify-between gap-3 rounded-lg border border-slate-700/50 bg-slate-950/40 px-3 py-3"
                    >
                      <div className="min-w-0">
                        <div className="truncate font-medium text-white">{vendor.vendor_name}</div>
                        <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-slate-500">
                          <span>{vendor.product_category ?? 'Uncategorized'}</span>
                          <span>{vendor.total_reviews ?? 0} reviews</span>
                          <span>avg urgency {vendor.avg_urgency?.toFixed(1) ?? '--'}</span>
                        </div>
                      </div>
                      <button
                        onClick={() => handleAddVendor(vendor.vendor_name)}
                        disabled={submittingVendor === vendor.vendor_name}
                        className="inline-flex items-center gap-1 rounded-md bg-cyan-500/10 px-2.5 py-1.5 text-xs font-medium text-cyan-300 hover:bg-cyan-500/20 disabled:opacity-50"
                      >
                        <Plus className="h-3 w-3" />
                        Add
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 p-4">
            <h2 className="text-sm font-medium text-white">Phase 1 Boundary</h2>
            <p className="mt-2 text-sm text-slate-400">
              This slice now includes tenant-wide persisted accounts-in-motion aggregation. The next major trust layer is the embedded evidence drawer so every vendor and account row can answer why it exists.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
