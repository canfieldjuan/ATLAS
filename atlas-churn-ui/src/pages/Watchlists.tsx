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
  createCompetitiveSet,
  deleteCompetitiveSet,
  fetchCompetitiveSetPlan,
  fetchAccountsInMotionFeed,
  fetchSlowBurnWatchlist,
  listTrackedVendors,
  listCompetitiveSets,
  removeTrackedVendor,
  runCompetitiveSetNow,
  searchAvailableVendors,
  type AccountsInMotionFeedItem,
  type CompetitiveSet,
  type CompetitiveSetDefaults,
  type CompetitiveSetPlan,
  type CompetitiveSetRun,
  type TrackedVendor,
  updateCompetitiveSet,
  type VendorSearchResult,
} from '../api/client'
import type { ChurnSignal } from '../types'

interface WatchlistsData {
  vendors: TrackedVendor[]
  competitiveSets: CompetitiveSet[]
  competitiveSetDefaults: CompetitiveSetDefaults | null
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

function formatCostUsd(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return '--'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value)
}

function formatWholeNumber(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return '--'
  return new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 0,
  }).format(value)
}

export default function Watchlists() {
  const navigate = useNavigate()
  const [searchInput, setSearchInput] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [trackMode, setTrackMode] = useState<'own' | 'competitor'>('competitor')
  const [label, setLabel] = useState('')
  const [submittingVendor, setSubmittingVendor] = useState<string | null>(null)
  const [removingVendor, setRemovingVendor] = useState<string | null>(null)
  const [savingCompetitiveSet, setSavingCompetitiveSet] = useState(false)
  const [runningCompetitiveSetId, setRunningCompetitiveSetId] = useState<string | null>(null)
  const [previewingCompetitiveSetId, setPreviewingCompetitiveSetId] = useState<string | null>(null)
  const [openCompetitiveSetPreviewId, setOpenCompetitiveSetPreviewId] = useState<string | null>(null)
  const [deletingCompetitiveSetId, setDeletingCompetitiveSetId] = useState<string | null>(null)
  const [editingCompetitiveSetId, setEditingCompetitiveSetId] = useState<string | null>(null)
  const [competitiveSetPreviews, setCompetitiveSetPreviews] = useState<Record<string, CompetitiveSetPlan>>({})
  const [competitiveSetRuns, setCompetitiveSetRuns] = useState<Record<string, CompetitiveSetRun[]>>({})
  const [competitiveSetChangedOnly, setCompetitiveSetChangedOnly] = useState<Record<string, boolean>>({})
  const [actionMessage, setActionMessage] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [competitiveSetName, setCompetitiveSetName] = useState('')
  const [competitiveSetFocal, setCompetitiveSetFocal] = useState('')
  const [competitiveSetCompetitors, setCompetitiveSetCompetitors] = useState<string[]>([])
  const [competitiveSetRefreshMode, setCompetitiveSetRefreshMode] = useState<'manual' | 'scheduled'>('manual')
  const [competitiveSetRefreshHours, setCompetitiveSetRefreshHours] = useState('')
  const [competitiveSetActive, setCompetitiveSetActive] = useState(true)
  const [competitiveSetVendorEnabled, setCompetitiveSetVendorEnabled] = useState(true)
  const [competitiveSetPairwiseEnabled, setCompetitiveSetPairwiseEnabled] = useState(true)
  const [competitiveSetCategoryEnabled, setCompetitiveSetCategoryEnabled] = useState(false)
  const [competitiveSetAsymmetryEnabled, setCompetitiveSetAsymmetryEnabled] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchInput.trim())
    }, SEARCH_DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [searchInput])

  const { data, loading, error, refresh, refreshing } = useApiData<WatchlistsData>(
    async () => {
      const [tracked, trackedSets, feed, accounts] = await Promise.all([
        listTrackedVendors(),
        listCompetitiveSets(true),
        fetchSlowBurnWatchlist(),
        fetchAccountsInMotionFeed(),
      ])
      return {
        vendors: tracked.vendors,
        competitiveSets: trackedSets.competitive_sets,
        competitiveSetDefaults: trackedSets.defaults ?? null,
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
  const competitiveSets = data?.competitiveSets ?? []
  const competitiveSetDefaults = data?.competitiveSetDefaults ?? null
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
  const focalOptions = trackedVendors
  const competitorOptions = trackedVendors.filter(
    (vendor) => vendor.vendor_name !== competitiveSetFocal,
  )

  useEffect(() => {
    if (editingCompetitiveSetId) return
    if (competitiveSetRefreshHours !== '') return
    const defaultHours = competitiveSetDefaults?.default_refresh_interval_hours
    if (!defaultHours) return
    setCompetitiveSetRefreshHours(String(defaultHours))
  }, [
    competitiveSetDefaults?.default_refresh_interval_hours,
    competitiveSetRefreshHours,
    editingCompetitiveSetId,
  ])

  function resetCompetitiveSetForm() {
    setEditingCompetitiveSetId(null)
    setCompetitiveSetName('')
    setCompetitiveSetFocal('')
    setCompetitiveSetCompetitors([])
    setCompetitiveSetRefreshMode('manual')
    setCompetitiveSetRefreshHours(String(competitiveSetDefaults?.default_refresh_interval_hours ?? ''))
    setCompetitiveSetActive(true)
    setCompetitiveSetVendorEnabled(true)
    setCompetitiveSetPairwiseEnabled(true)
    setCompetitiveSetCategoryEnabled(false)
    setCompetitiveSetAsymmetryEnabled(false)
  }

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

  function loadCompetitiveSetForEdit(item: CompetitiveSet) {
    setEditingCompetitiveSetId(item.id)
    setCompetitiveSetName(item.name)
    setCompetitiveSetFocal(item.focal_vendor_name)
    setCompetitiveSetCompetitors(item.competitor_vendor_names)
    setCompetitiveSetRefreshMode(item.refresh_mode)
    setCompetitiveSetRefreshHours(
      String(item.refresh_interval_hours ?? competitiveSetDefaults?.default_refresh_interval_hours ?? ''),
    )
    setCompetitiveSetActive(item.active)
    setCompetitiveSetVendorEnabled(item.vendor_synthesis_enabled)
    setCompetitiveSetPairwiseEnabled(item.pairwise_enabled)
    setCompetitiveSetCategoryEnabled(item.category_council_enabled)
    setCompetitiveSetAsymmetryEnabled(item.asymmetry_enabled)
  }

  async function handleSaveCompetitiveSet() {
    setSavingCompetitiveSet(true)
    setActionError(null)
    setActionMessage(null)
    const payload = {
      name: competitiveSetName.trim(),
      focal_vendor_name: competitiveSetFocal,
      competitor_vendor_names: competitiveSetCompetitors,
      active: competitiveSetActive,
      refresh_mode: competitiveSetRefreshMode,
      refresh_interval_hours: competitiveSetRefreshMode === 'scheduled'
        ? Number.parseInt(competitiveSetRefreshHours, 10)
          || competitiveSetDefaults?.default_refresh_interval_hours
          || null
        : null,
      vendor_synthesis_enabled: competitiveSetVendorEnabled,
      pairwise_enabled: competitiveSetPairwiseEnabled,
      category_council_enabled: competitiveSetCategoryEnabled,
      asymmetry_enabled: competitiveSetAsymmetryEnabled,
    }
    try {
      if (editingCompetitiveSetId) {
        await updateCompetitiveSet(editingCompetitiveSetId, payload)
        setActionMessage(`Updated competitive set ${payload.name}`)
      } else {
        await createCompetitiveSet(payload)
        setActionMessage(`Created competitive set ${payload.name}`)
      }
      resetCompetitiveSetForm()
      refresh()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to save competitive set')
    } finally {
      setSavingCompetitiveSet(false)
    }
  }

  async function handleDeleteCompetitiveSet(item: CompetitiveSet) {
    if (!confirm(`Delete competitive set ${item.name}?`)) return
    setDeletingCompetitiveSetId(item.id)
    setActionError(null)
    setActionMessage(null)
    try {
      await deleteCompetitiveSet(item.id)
      if (editingCompetitiveSetId === item.id) resetCompetitiveSetForm()
      setActionMessage(`Deleted competitive set ${item.name}`)
      refresh()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to delete competitive set')
    } finally {
      setDeletingCompetitiveSetId(null)
    }
  }

  async function handlePreviewCompetitiveSet(item: CompetitiveSet) {
    setPreviewingCompetitiveSetId(item.id)
    setActionError(null)
    setActionMessage(null)
    try {
      const result = await fetchCompetitiveSetPlan(item.id)
      setCompetitiveSetPreviews((current) => ({
        ...current,
        [item.id]: result.plan,
      }))
      setCompetitiveSetRuns((current) => ({
        ...current,
        [item.id]: result.recent_runs,
      }))
      setCompetitiveSetChangedOnly((current) => ({
        ...current,
        [item.id]: current[item.id] ?? true,
      }))
      setOpenCompetitiveSetPreviewId(item.id)
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to load competitive-set preview')
    } finally {
      setPreviewingCompetitiveSetId(null)
    }
  }

  async function handleRunCompetitiveSet(item: CompetitiveSet) {
    setRunningCompetitiveSetId(item.id)
    setActionError(null)
    setActionMessage(null)
    try {
      const result = await runCompetitiveSetNow(item.id, {
        changed_vendors_only: competitiveSetChangedOnly[item.id] ?? true,
      })
      setActionMessage(
        `${item.name} refresh started${result.execution_id ? ` (${result.execution_id})` : ''}`,
      )
      refresh()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to start competitive-set refresh')
    } finally {
      setRunningCompetitiveSetId(null)
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
            <div className="mb-3 flex items-start justify-between gap-3">
              <div>
                <h2 className="text-sm font-medium text-white">Competitive Sets</h2>
                <p className="mt-1 text-xs text-slate-500">
                  Define a focal vendor and exact competitors, then refresh synthesis only for that scope instead of the full vendor universe.
                </p>
              </div>
              {editingCompetitiveSetId ? (
                <button
                  onClick={resetCompetitiveSetForm}
                  className="rounded-md bg-slate-800 px-2 py-1 text-xs text-slate-300 hover:bg-slate-700"
                >
                  Cancel edit
                </button>
              ) : null}
            </div>

            <div className="space-y-3">
              <label className="space-y-1">
                <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Set Name</span>
                <input
                  type="text"
                  value={competitiveSetName}
                  onChange={(event) => setCompetitiveSetName(event.target.value)}
                  placeholder="Salesforce core competitors"
                  className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-500/50 focus:outline-none"
                />
              </label>

              <div className="grid gap-3 sm:grid-cols-2">
                <label className="space-y-1">
                  <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Focal Vendor</span>
                  <select
                    value={competitiveSetFocal}
                    onChange={(event) => {
                      setCompetitiveSetFocal(event.target.value)
                      setCompetitiveSetCompetitors((current) => current.filter((item) => item !== event.target.value))
                    }}
                    className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white focus:border-cyan-500/50 focus:outline-none"
                  >
                    <option value="">Select tracked vendor</option>
                    {focalOptions.map((vendor) => (
                      <option key={vendor.id} value={vendor.vendor_name}>
                        {vendor.vendor_name} · {vendor.track_mode}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="space-y-1">
                  <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Refresh Mode</span>
                  <select
                    value={competitiveSetRefreshMode}
                    onChange={(event) => setCompetitiveSetRefreshMode(event.target.value as 'manual' | 'scheduled')}
                    className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white focus:border-cyan-500/50 focus:outline-none"
                  >
                    <option value="manual">Manual</option>
                    <option value="scheduled">Scheduled</option>
                  </select>
                </label>
              </div>

              {competitiveSetRefreshMode === 'scheduled' ? (
                <label className="space-y-1">
                  <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Refresh Every (Hours)</span>
                  <input
                    type="number"
                    min={1}
                    max={720}
                    value={competitiveSetRefreshHours}
                    onChange={(event) => setCompetitiveSetRefreshHours(event.target.value)}
                    className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white focus:border-cyan-500/50 focus:outline-none"
                  />
                </label>
              ) : null}

              <div className="space-y-2 rounded-lg border border-slate-700/50 bg-slate-950/40 p-3">
                <div className="text-xs font-medium uppercase tracking-wide text-slate-500">Tracked Competitors</div>
                <div className="grid gap-2">
                  {competitorOptions.length === 0 ? (
                    <div className="text-sm text-slate-500">Add tracked vendors first, then choose the ones to compare against the focal vendor.</div>
                  ) : competitorOptions.map((vendor) => (
                    <label key={vendor.id} className="flex items-center gap-2 text-sm text-slate-300">
                      <input
                        type="checkbox"
                        checked={competitiveSetCompetitors.includes(vendor.vendor_name)}
                        onChange={(event) => {
                          setCompetitiveSetCompetitors((current) => {
                            if (event.target.checked) return [...current, vendor.vendor_name]
                            return current.filter((item) => item !== vendor.vendor_name)
                          })
                        }}
                        className="rounded border-slate-600 bg-slate-900 text-cyan-400 focus:ring-cyan-500"
                      />
                      <span>{vendor.vendor_name}</span>
                      <span className="text-xs text-slate-500">{vendor.track_mode}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="grid gap-2 sm:grid-cols-2">
                <label className="flex items-center gap-2 rounded-lg border border-slate-700/50 bg-slate-950/40 px-3 py-2 text-sm text-slate-300">
                  <input type="checkbox" checked={competitiveSetActive} onChange={(event) => setCompetitiveSetActive(event.target.checked)} />
                  Active
                </label>
                <label className="flex items-center gap-2 rounded-lg border border-slate-700/50 bg-slate-950/40 px-3 py-2 text-sm text-slate-300">
                  <input type="checkbox" checked={competitiveSetVendorEnabled} onChange={(event) => setCompetitiveSetVendorEnabled(event.target.checked)} />
                  Vendor synthesis
                </label>
                <label className="flex items-center gap-2 rounded-lg border border-slate-700/50 bg-slate-950/40 px-3 py-2 text-sm text-slate-300">
                  <input type="checkbox" checked={competitiveSetPairwiseEnabled} onChange={(event) => setCompetitiveSetPairwiseEnabled(event.target.checked)} />
                  Pairwise
                </label>
                <label className="flex items-center gap-2 rounded-lg border border-slate-700/50 bg-slate-950/40 px-3 py-2 text-sm text-slate-300">
                  <input type="checkbox" checked={competitiveSetCategoryEnabled} onChange={(event) => setCompetitiveSetCategoryEnabled(event.target.checked)} />
                  Category council
                </label>
                <label className="flex items-center gap-2 rounded-lg border border-slate-700/50 bg-slate-950/40 px-3 py-2 text-sm text-slate-300 sm:col-span-2">
                  <input type="checkbox" checked={competitiveSetAsymmetryEnabled} onChange={(event) => setCompetitiveSetAsymmetryEnabled(event.target.checked)} />
                  Resource asymmetry
                </label>
              </div>

              <button
                onClick={handleSaveCompetitiveSet}
                disabled={
                  savingCompetitiveSet
                  || !competitiveSetName.trim()
                  || !competitiveSetFocal
                  || competitiveSetCompetitors.length === 0
                }
                className="inline-flex items-center gap-2 rounded-lg bg-cyan-500/10 px-3 py-2 text-sm font-medium text-cyan-300 hover:bg-cyan-500/20 disabled:opacity-50"
              >
                <Plus className="h-4 w-4" />
                {editingCompetitiveSetId ? 'Update competitive set' : 'Create competitive set'}
              </button>
            </div>

            <div className="mt-5 space-y-3">
              {competitiveSets.length === 0 ? (
                <div className="rounded-lg border border-dashed border-slate-700/50 px-3 py-4 text-sm text-slate-500">
                  No competitive sets yet. Build one from your tracked vendors to keep synthesis scoped to the exact vendors you care about.
                </div>
              ) : competitiveSets.map((item) => {
                const competitorCount = item.competitor_vendor_names.length
                const vendorJobCount = item.vendor_synthesis_enabled ? competitorCount + 1 : 0
                const pairwiseJobCount = item.pairwise_enabled ? competitorCount : 0
                const asymmetryJobCount = item.asymmetry_enabled ? competitorCount : 0
                const preview = competitiveSetPreviews[item.id]
                const estimate = preview?.estimate
                const recentRuns = competitiveSetRuns[item.id] ?? []
                const likelyRerunVendors = (estimate?.likely_rerun_vendors ?? []).slice(0, 6)
                const likelyReuseVendors = (estimate?.likely_reuse_vendors ?? []).slice(0, 6)
                const previewOpen = openCompetitiveSetPreviewId === item.id
                const changedOnly = competitiveSetChangedOnly[item.id] ?? true
                return (
                  <div key={item.id} className="rounded-lg border border-slate-700/50 bg-slate-950/40 p-3">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="font-medium text-white">{item.name}</div>
                        <div className="mt-1 text-xs text-slate-500">
                          Focal: {item.focal_vendor_name} · {competitorCount} competitors · {item.refresh_mode}
                          {item.refresh_mode === 'scheduled' && item.refresh_interval_hours
                            ? ` every ${item.refresh_interval_hours}h`
                            : ''}
                        </div>
                      </div>
                      <span
                        className={clsx(
                          'rounded-full px-2 py-0.5 text-[11px] font-medium',
                          item.last_run_status === 'succeeded' && 'bg-emerald-500/10 text-emerald-300',
                          item.last_run_status === 'partial' && 'bg-amber-500/10 text-amber-300',
                          item.last_run_status === 'failed' && 'bg-rose-500/10 text-rose-300',
                          item.last_run_status === 'running' && 'bg-cyan-500/10 text-cyan-300',
                          !item.last_run_status && 'bg-slate-800 text-slate-300',
                        )}
                      >
                        {item.last_run_status ?? 'never run'}
                      </span>
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2 text-xs text-slate-400">
                      <span>{vendorJobCount} vendor jobs</span>
                      <span>{pairwiseJobCount} pairwise jobs</span>
                      {item.category_council_enabled ? <span>category council enabled</span> : null}
                      {item.asymmetry_enabled ? <span>{asymmetryJobCount} asymmetry pairs</span> : null}
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {item.competitor_vendor_names.map((vendorName) => (
                        <span key={vendorName} className="rounded-full bg-slate-800 px-2 py-0.5 text-[11px] text-slate-300">
                          {vendorName}
                        </span>
                      ))}
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2">
                      <button
                        onClick={() => handlePreviewCompetitiveSet(item)}
                        disabled={previewingCompetitiveSetId === item.id}
                        className="rounded-md bg-cyan-500/10 px-2.5 py-1 text-xs font-medium text-cyan-300 hover:bg-cyan-500/20 disabled:opacity-50"
                      >
                        {previewingCompetitiveSetId === item.id ? 'Loading preview...' : 'Preview cost'}
                      </button>
                      <button
                        onClick={() => loadCompetitiveSetForEdit(item)}
                        className="rounded-md bg-slate-800 px-2.5 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700"
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDeleteCompetitiveSet(item)}
                        disabled={deletingCompetitiveSetId === item.id}
                        className="rounded-md bg-rose-500/10 px-2.5 py-1 text-xs font-medium text-rose-300 hover:bg-rose-500/20 disabled:opacity-50"
                      >
                        Delete
                      </button>
                    </div>
                    {previewOpen && preview ? (
                      <div className="mt-3 rounded-lg border border-cyan-500/20 bg-slate-900/70 p-3">
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="text-sm font-medium text-white">Run Preview</div>
                            <div className="mt-1 text-xs text-slate-400">
                              Estimated upper bound for a non-forced run over the next {preview.estimate?.lookback_days ?? '--'} days of history.
                            </div>
                          </div>
                          <button
                            onClick={() => setOpenCompetitiveSetPreviewId((current) => current === item.id ? null : current)}
                            className="rounded-md bg-slate-800 px-2 py-1 text-[11px] font-medium text-slate-300 hover:bg-slate-700"
                          >
                            Close
                          </button>
                        </div>
                        <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                          <div className="rounded-md border border-slate-700/50 bg-slate-950/50 p-2">
                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Planned Jobs</div>
                            <div className="mt-1 text-sm font-medium text-white">{formatWholeNumber(preview.estimated_total_jobs)}</div>
                            <div className="mt-1 text-[11px] text-slate-400">
                              {formatWholeNumber(preview.vendor_job_count)} vendor · {formatWholeNumber(preview.pairwise_job_count + preview.category_job_count + preview.asymmetry_job_count)} cross-vendor
                            </div>
                          </div>
                          <div className="rounded-md border border-slate-700/50 bg-slate-950/50 p-2">
                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Token Estimate</div>
                            <div className="mt-1 text-sm font-medium text-white">{formatWholeNumber(estimate?.estimated_total_tokens)}</div>
                            <div className="mt-1 text-[11px] text-slate-400">
                              {formatWholeNumber(estimate?.estimated_vendor_tokens)} vendor · {formatWholeNumber(estimate?.estimated_cross_vendor_tokens)} cross-vendor
                            </div>
                          </div>
                          <div className="rounded-md border border-slate-700/50 bg-slate-950/50 p-2">
                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Cost Upper Bound</div>
                            <div className="mt-1 text-sm font-medium text-white">{formatCostUsd(estimate?.estimated_total_cost_usd)}</div>
                            <div className="mt-1 text-[11px] text-slate-400">
                              {formatCostUsd(estimate?.estimated_vendor_cost_usd)} vendor · {formatCostUsd(estimate?.estimated_cross_vendor_cost_usd)} cross-vendor
                            </div>
                          </div>
                          <div className="rounded-md border border-slate-700/50 bg-slate-950/50 p-2">
                            <div className="text-[11px] uppercase tracking-wide text-slate-500">History Coverage</div>
                            <div className="mt-1 text-sm font-medium text-white">
                              {formatWholeNumber(estimate?.vendor_jobs_with_history)} vendor · {formatWholeNumber(estimate?.cross_vendor_jobs_with_history)} cross-vendor
                            </div>
                            <div className="mt-1 text-[11px] text-slate-400">
                              {formatWholeNumber(estimate?.vendor_jobs_using_fallback)} vendor fallback · {formatWholeNumber(estimate?.cross_vendor_jobs_using_fallback)} cross fallback
                            </div>
                          </div>
                        </div>
                        <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                          <div className="rounded-md border border-slate-700/50 bg-slate-950/50 p-2">
                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Likely Vendor Outcome</div>
                            <div className="mt-1 text-sm font-medium text-white">
                              {formatWholeNumber(estimate?.vendor_jobs_likely_to_reason)} rerun · {formatWholeNumber(estimate?.vendor_jobs_likely_hash_reuse)} reuse
                            </div>
                            <div className="mt-1 text-[11px] text-slate-400">
                              {formatWholeNumber(estimate?.vendor_jobs_missing_pools)} no pool · {formatWholeNumber(estimate?.vendor_jobs_likely_stale_reuse)} stale reuse
                            </div>
                          </div>
                          <div className="rounded-md border border-slate-700/50 bg-slate-950/50 p-2">
                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Likely Vendor Spend</div>
                            <div className="mt-1 text-sm font-medium text-white">{formatCostUsd(estimate?.estimated_vendor_cost_usd_likely_to_reason)}</div>
                            <div className="mt-1 text-[11px] text-slate-400">
                              {formatWholeNumber(estimate?.estimated_vendor_tokens_likely_to_reason)} vendor tokens if changed-only
                            </div>
                          </div>
                          <div className="rounded-md border border-slate-700/50 bg-slate-950/50 p-2">
                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Rerun Reasons</div>
                            <div className="mt-1 text-sm font-medium text-white">
                              {formatWholeNumber(estimate?.vendor_jobs_likely_hash_changed)} changed · {formatWholeNumber(estimate?.vendor_jobs_likely_prior_quality_weak)} weak
                            </div>
                            <div className="mt-1 text-[11px] text-slate-400">
                              {formatWholeNumber(estimate?.vendor_jobs_likely_missing_prior)} missing prior · {formatWholeNumber(estimate?.vendor_jobs_likely_missing_reference_ids)} missing refs
                            </div>
                          </div>
                        </div>
                        <div className="mt-3 text-xs text-slate-400">
                          {estimate?.note ?? 'Estimate unavailable.'}
                        </div>
                        {likelyRerunVendors.length > 0 ? (
                          <div className="mt-3">
                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Likely Rerun Vendors</div>
                            <div className="mt-2 flex flex-wrap gap-1">
                              {likelyRerunVendors.map((entry) => (
                                <span key={entry} className="rounded-full bg-amber-500/10 px-2 py-0.5 text-[11px] text-amber-300">
                                  {entry.replace(':', ' · ')}
                                </span>
                              ))}
                            </div>
                          </div>
                        ) : null}
                        {likelyReuseVendors.length > 0 ? (
                          <div className="mt-3">
                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Likely Reuse Vendors</div>
                            <div className="mt-2 flex flex-wrap gap-1">
                              {likelyReuseVendors.map((entry) => (
                                <span key={entry} className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[11px] text-emerald-300">
                                  {entry.replace(':', ' · ')}
                                </span>
                              ))}
                            </div>
                          </div>
                        ) : null}
                        <div className="mt-4">
                          <div className="text-[11px] uppercase tracking-wide text-slate-500">Recent Runs</div>
                          {recentRuns.length === 0 ? (
                            <div className="mt-2 text-xs text-slate-400">No recorded runs yet.</div>
                          ) : (
                            <div className="mt-2 space-y-2">
                              {recentRuns.map((run) => {
                                const summary = run.summary ?? {}
                                const totalTokens = typeof summary.total_tokens === 'number'
                                  ? summary.total_tokens
                                  : null
                                const reused = typeof summary.vendors_skipped_hash_reuse === 'number'
                                  ? summary.vendors_skipped_hash_reuse
                                  : null
                                const reasoned = typeof summary.vendors_reasoned === 'number'
                                  ? summary.vendors_reasoned
                                  : null
                                const failed = typeof summary.vendors_failed === 'number'
                                  ? summary.vendors_failed
                                  : null
                                return (
                                  <div key={run.id} className="rounded-md border border-slate-700/50 bg-slate-950/50 p-2">
                                    <div className="flex items-center justify-between gap-2">
                                      <div className="text-xs text-slate-300">
                                        {formatTs(run.started_at)}
                                      </div>
                                      <span
                                        className={clsx(
                                          'rounded-full px-2 py-0.5 text-[11px] font-medium',
                                          run.status === 'succeeded' && 'bg-emerald-500/10 text-emerald-300',
                                          run.status === 'partial' && 'bg-amber-500/10 text-amber-300',
                                          run.status === 'failed' && 'bg-rose-500/10 text-rose-300',
                                          run.status === 'running' && 'bg-cyan-500/10 text-cyan-300',
                                        )}
                                      >
                                        {run.status}
                                      </span>
                                    </div>
                                      <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-slate-400">
                                        <span>{formatWholeNumber(reasoned)} reasoned</span>
                                        <span>{formatWholeNumber(reused)} reused</span>
                                        <span>{formatWholeNumber(failed)} failed</span>
                                        <span>{formatWholeNumber(totalTokens)} tokens</span>
                                        {summary.changed_vendors_only === true ? <span>changed only</span> : null}
                                      </div>
                                    </div>
                                  )
                              })}
                            </div>
                          )}
                        </div>
                        <label className="mt-3 flex items-center gap-2 rounded-md border border-slate-700/50 bg-slate-950/50 px-3 py-2 text-xs text-slate-300">
                          <input
                            type="checkbox"
                            checked={changedOnly}
                            onChange={(event) => {
                              setCompetitiveSetChangedOnly((current) => ({
                                ...current,
                                [item.id]: event.target.checked,
                              }))
                            }}
                          />
                          Run changed vendors only
                        </label>
                        <div className="mt-3 flex flex-wrap gap-2">
                          <button
                            onClick={() => handleRunCompetitiveSet(item)}
                            disabled={runningCompetitiveSetId === item.id}
                            className="rounded-md bg-cyan-500/10 px-2.5 py-1 text-xs font-medium text-cyan-300 hover:bg-cyan-500/20 disabled:opacity-50"
                          >
                            {runningCompetitiveSetId === item.id ? 'Starting run...' : 'Run now'}
                          </button>
                          <button
                            onClick={() => handlePreviewCompetitiveSet(item)}
                            disabled={previewingCompetitiveSetId === item.id}
                            className="rounded-md bg-slate-800 px-2.5 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700 disabled:opacity-50"
                          >
                            Refresh preview
                          </button>
                        </div>
                      </div>
                    ) : null}
                  </div>
                )
              })}
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
