import { Link, useSearchParams } from 'react-router-dom'
import { useState, useRef, useCallback, useEffect, useMemo, type ReactNode } from 'react'
import {
  Search, Fingerprint, Loader2, Database, GitBranch,
  ShieldCheck, ShieldAlert, ChevronDown, ChevronRight,
  ArrowLeft, ArrowRight, FileText, Layers, Filter, ExternalLink, Copy,
} from 'lucide-react'
import { clsx } from 'clsx'
import EvidenceDrawer, { SourceBadge, SIGNAL_COLORS } from '../components/EvidenceDrawer'
import {
  listTrackedVendors, listWatchlistViews, searchAvailableVendors, fetchWitnesses, fetchEvidenceVault, fetchEvidenceTrace,
} from '../api/client'
import type {
  EvidenceWitness, EvidenceFacets, EvidenceVault, EvidenceTrace,
} from '../api/client'


// -- Witness type colors ------------------------------------------------------

const WITNESS_TYPE_COLORS: Record<string, string> = {
  pain_point: 'border-l-red-500',
  displacement: 'border-l-violet-500',
  strength: 'border-l-emerald-500',
  pricing: 'border-l-amber-500',
  support: 'border-l-orange-500',
  feature: 'border-l-blue-500',
  integration: 'border-l-cyan-500',
  churn_intent: 'border-l-rose-500',
}


// -- Tab type -----------------------------------------------------------------

type Tab = 'witnesses' | 'vault' | 'trace'

function parseTab(value: string | null): Tab {
  if (value === 'vault' || value === 'trace' || value === 'witnesses') return value
  return 'witnesses'
}

function evidenceExplorerUrl(searchParams: URLSearchParams) {
  const query = searchParams.toString()
  return `${window.location.origin}/evidence${query ? `?${query}` : ''}`
}

function evidenceExplorerPath(searchParams: URLSearchParams) {
  const query = searchParams.toString()
  return `/evidence${query ? `?${query}` : ''}`
}

function evidenceOpportunitiesPath(searchParams: URLSearchParams, vendorName: string) {
  const params = new URLSearchParams()
  params.set('vendor', vendorName)
  params.set('back_to', evidenceExplorerPath(searchParams))
  return `/opportunities?${params.toString()}`
}

function evidenceReportsPath(searchParams: URLSearchParams, vendorName: string) {
  const params = new URLSearchParams()
  params.set('vendor_filter', vendorName)
  params.set('back_to', evidenceExplorerPath(searchParams))
  return `/reports?${params.toString()}`
}

function evidenceVendorPath(searchParams: URLSearchParams, vendorName: string) {
  const params = new URLSearchParams()
  params.set('back_to', evidenceExplorerPath(searchParams))
  return `/vendors/${encodeURIComponent(vendorName)}?${params.toString()}`
}

function evidenceWatchlistsPath(searchParams: URLSearchParams, vendorName: string, viewId?: string | null) {
  const params = new URLSearchParams()
  if (viewId) params.set('view', viewId)
  else params.set('vendor_name', vendorName)
  params.set('back_to', evidenceExplorerPath(searchParams))
  return `/watchlists?${params.toString()}`
}

function watchlistViewVendorNames(view: { vendor_names?: string[] | null; vendor_name?: string | null }) {
  if (view.vendor_names?.length) return view.vendor_names
  return view.vendor_name ? [view.vendor_name] : []
}

function parseBackTo(value: string | null) {
  if (!value) return null
  if (
    value.startsWith('/watchlists')
    || value.startsWith('/vendors/')
    || value.startsWith('/opportunities')
    || value.startsWith('/reports')
  ) return value
  try {
    const url = new URL(value, window.location.origin)
    if (
      url.origin === window.location.origin
      && (
        url.pathname === '/watchlists'
        || url.pathname.startsWith('/vendors/')
        || url.pathname === '/opportunities'
        || url.pathname === '/reports'
      )
    ) {
      return `${url.pathname}${url.search}`
    }
  } catch {
    return null
  }
  return null
}

function backToLabel(value: string | null) {
  if (!value) return 'Back'
  if (value.startsWith('/watchlists')) return 'Back to Watchlists'
  if (value.startsWith('/vendors/')) return 'Back to Vendor'
  if (value.startsWith('/opportunities')) return 'Back to Opportunities'
  if (value.startsWith('/reports')) return 'Back to Reports'
  return 'Back'
}

async function copyText(text: string) {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text)
      return
    } catch {
      // Fall through to the document-copy path for environments where
      // navigator.clipboard exists but is not writable.
    }
  }

  const textarea = document.createElement('textarea')
  textarea.value = text
  textarea.setAttribute('readonly', '')
  textarea.style.position = 'absolute'
  textarea.style.left = '-9999px'
  document.body.appendChild(textarea)
  textarea.select()
  const copied = document.execCommand('copy')
  document.body.removeChild(textarea)
  if (!copied) {
    throw new Error('Copy failed')
  }
}


// -- Main component -----------------------------------------------------------

export default function EvidenceExplorer() {
  const windowDays = 30
  const [searchParams, setSearchParams] = useSearchParams()
  const requestedVendor = searchParams.get('vendor')?.trim() || ''
  const requestedTab = parseTab(searchParams.get('tab'))
  const requestedPain = searchParams.get('pain_category')?.trim() || ''
  const requestedSource = searchParams.get('source')?.trim() || ''
  const requestedWitnessType = searchParams.get('witness_type')?.trim() || ''
  const requestedWitnessId = searchParams.get('witness_id')?.trim() || ''
  const requestedBackTo = parseBackTo(searchParams.get('back_to'))

  // Search state
  const [vendorInput, setVendorInput] = useState(requestedVendor)
  const [activeVendor, setActiveVendor] = useState(requestedVendor)
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [trackedVendorNames, setTrackedVendorNames] = useState<string[]>([])
  const [watchlistViews, setWatchlistViews] = useState<Array<{ id: string; vendor_names?: string[] | null; vendor_name?: string | null }>>([])
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Tab state
  const [activeTab, setActiveTab] = useState<Tab>(requestedTab)

  // Witnesses state
  const [witnesses, setWitnesses] = useState<EvidenceWitness[]>([])
  const [facets, setFacets] = useState<EvidenceFacets>({ pain_categories: [], sources: [], witness_types: [] })
  const [total, setTotal] = useState(0)
  const [witnessSnapshotDate, setWitnessSnapshotDate] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [offset, setOffset] = useState(0)
  const limit = 30

  // Filters
  const [filterPain, setFilterPain] = useState(requestedPain)
  const [filterSource, setFilterSource] = useState(requestedSource)
  const [filterType, setFilterType] = useState(requestedWitnessType)
  const [showFilters, setShowFilters] = useState(true)

  // Vault state
  const [vault, setVault] = useState<EvidenceVault | null>(null)
  const [vaultLoading, setVaultLoading] = useState(false)

  // Trace state
  const [trace, setTrace] = useState<EvidenceTrace | null>(null)
  const [traceLoading, setTraceLoading] = useState(false)

  // Drawer state
  const [drawerOpen, setDrawerOpen] = useState(Boolean(requestedVendor && requestedWitnessId))
  const [drawerWitnessId, setDrawerWitnessId] = useState<string | null>(requestedWitnessId || null)
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'error'>('idle')
  const matchedWatchlistViewId = useMemo(() => {
    if (!activeVendor) return null
    const normalizedVendor = activeVendor.toLowerCase()
    const exactSingleVendorMatch = watchlistViews.find((view) => {
      const vendorNames = watchlistViewVendorNames(view).map((name) => name.toLowerCase())
      return vendorNames.length === 1 && vendorNames[0] === normalizedVendor
    })
    if (exactSingleVendorMatch) return exactSingleVendorMatch.id
    return watchlistViews.find((view) => (
      watchlistViewVendorNames(view).some((name) => name.toLowerCase() === normalizedVendor)
    ))?.id ?? null
  }, [activeVendor, watchlistViews])
  const hasWatchlistsShortcut = useMemo(
    () => Boolean(activeVendor) && (
      trackedVendorNames.includes(activeVendor.toLowerCase())
      || Boolean(matchedWatchlistViewId)
    ),
    [activeVendor, matchedWatchlistViewId, trackedVendorNames],
  )
  const drawerBackToPath = useMemo(() => {
    const next = new URLSearchParams(searchParams.toString())
    if (activeVendor) next.set('vendor', activeVendor)
    else next.delete('vendor')
    next.set('tab', activeTab)
    if (filterPain) next.set('pain_category', filterPain)
    else next.delete('pain_category')
    if (filterSource) next.set('source', filterSource)
    else next.delete('source')
    if (filterType) next.set('witness_type', filterType)
    else next.delete('witness_type')
    if (drawerOpen && drawerWitnessId) next.set('witness_id', drawerWitnessId)
    else next.delete('witness_id')
    return evidenceExplorerPath(next)
  }, [
    activeTab,
    activeVendor,
    drawerOpen,
    drawerWitnessId,
    filterPain,
    filterSource,
    filterType,
    searchParams,
  ])

  // -- Search handler ---------------------------------------------------------

  const handleSearchInput = useCallback((query: string) => {
    setVendorInput(query)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    if (query.length < 2) {
      setSuggestions([])
      setShowSuggestions(false)
      return
    }
    debounceRef.current = setTimeout(async () => {
      try {
        const res = await searchAvailableVendors(query)
        const names = (res.vendors || []).map(v => v.vendor_name)
        setSuggestions(names.slice(0, 8))
        setShowSuggestions(names.length > 0)
      } catch {
        setSuggestions([])
      }
    }, 250)
  }, [])

  useEffect(() => {
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current) }
  }, [])

  useEffect(() => {
    let cancelled = false
    Promise.all([
      listTrackedVendors(),
      listWatchlistViews().catch(() => ({ views: [], count: 0 })),
    ])
      .then(([trackedRes, savedViewsRes]) => {
        if (cancelled) return
        setTrackedVendorNames((trackedRes.vendors || []).map((vendor) => vendor.vendor_name.toLowerCase()))
        setWatchlistViews(savedViewsRes.views || [])
      })
      .catch(() => {
        if (cancelled) return
        setTrackedVendorNames([])
        setWatchlistViews([])
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Request version counter to prevent stale responses from overwriting state
  const requestVersionRef = useRef(0)

  // -- Load data on vendor/filter/tab change ----------------------------------

  const selectVendor = useCallback((name: string) => {
    setVendorInput(name)
    setActiveVendor(name)
    setShowSuggestions(false)
    setOffset(0)
    setActiveTab('witnesses')
    setDrawerOpen(false)
    setDrawerWitnessId(null)
  }, [])

  const loadWitnesses = useCallback(async () => {
    if (!activeVendor) return
    const version = ++requestVersionRef.current
    setLoading(true)
    try {
      const res = await fetchWitnesses({
        vendor_name: activeVendor,
        window_days: windowDays,
        pain_category: filterPain || undefined,
        source: filterSource || undefined,
        witness_type: filterType || undefined,
        limit,
        offset,
      })
      if (version !== requestVersionRef.current) return
      setWitnesses(res.witnesses)
      setFacets(res.facets)
      setTotal(res.total)
      setWitnessSnapshotDate(res.as_of_date)
    } catch {
      if (version !== requestVersionRef.current) return
      setWitnesses([])
      setWitnessSnapshotDate(null)
    } finally {
      if (version === requestVersionRef.current) setLoading(false)
    }
  }, [activeVendor, filterPain, filterSource, filterType, offset, windowDays])

  useEffect(() => {
    if (activeTab === 'witnesses') loadWitnesses()
  }, [loadWitnesses, activeTab])

  useEffect(() => {
    if (activeTab === 'vault' && activeVendor && !vault && !vaultLoading) {
      const version = ++requestVersionRef.current
      setVaultLoading(true)
      fetchEvidenceVault({ vendor_name: activeVendor, window_days: windowDays })
        .then(res => {
          if (version !== requestVersionRef.current) return
          if ('weakness_evidence' in res) {
            setVault(res as EvidenceVault)
          } else {
            setVault(null)
          }
        })
        .catch(() => {
          if (version !== requestVersionRef.current) return
          setVault(null)
        })
        .finally(() => {
          if (version === requestVersionRef.current) setVaultLoading(false)
        })
    }
  }, [activeTab, activeVendor, vault, vaultLoading, windowDays])

  useEffect(() => {
    if (activeTab === 'trace' && activeVendor && !trace && !traceLoading) {
      const version = ++requestVersionRef.current
      setTraceLoading(true)
      fetchEvidenceTrace({ vendor_name: activeVendor, window_days: windowDays })
        .then(res => {
          if (version !== requestVersionRef.current) return
          setTrace(res)
        })
        .catch(() => {
          if (version !== requestVersionRef.current) return
          setTrace(null)
        })
        .finally(() => {
          if (version === requestVersionRef.current) setTraceLoading(false)
        })
    }
  }, [activeTab, activeVendor, trace, traceLoading, windowDays])

  // Reset vault/trace when vendor changes
  useEffect(() => {
    setVault(null)
    setTrace(null)
    setWitnessSnapshotDate(null)
  }, [activeVendor])

  useEffect(() => {
    const currentVendor = searchParams.get('vendor')?.trim() || ''
    const currentTab = parseTab(searchParams.get('tab'))
    const currentPain = searchParams.get('pain_category')?.trim() || ''
    const currentSource = searchParams.get('source')?.trim() || ''
    const currentWitnessType = searchParams.get('witness_type')?.trim() || ''
    const currentWitnessId = searchParams.get('witness_id')?.trim() || ''
    const nextWitnessId = drawerOpen ? (drawerWitnessId || '') : ''
    if (
      currentVendor === activeVendor
      && currentTab === activeTab
      && currentPain === filterPain
      && currentSource === filterSource
      && currentWitnessType === filterType
      && currentWitnessId === nextWitnessId
    ) {
      return
    }
    setSearchParams((current) => {
      const next = new URLSearchParams(current)
      if (activeVendor) next.set('vendor', activeVendor)
      else next.delete('vendor')
      next.set('tab', activeTab)
      if (filterPain) next.set('pain_category', filterPain)
      else next.delete('pain_category')
      if (filterSource) next.set('source', filterSource)
      else next.delete('source')
      if (filterType) next.set('witness_type', filterType)
      else next.delete('witness_type')
      if (nextWitnessId) next.set('witness_id', nextWitnessId)
      else next.delete('witness_id')
      return next
    }, { replace: true })
  }, [
    activeTab,
    activeVendor,
    drawerOpen,
    drawerWitnessId,
    filterPain,
    filterSource,
    filterType,
    searchParams,
    setSearchParams,
  ])

  const clearFilters = () => {
    setFilterPain('')
    setFilterSource('')
    setFilterType('')
    setOffset(0)
  }

  const hasActiveFilters = filterPain || filterSource || filterType

  const openDrawer = (w: EvidenceWitness) => {
    setDrawerWitnessId(w.witness_id)
    setDrawerOpen(true)
  }

  async function handleCopyLink() {
    try {
      await copyText(evidenceExplorerUrl(new URLSearchParams(searchParams.toString())))
      setCopyState('copied')
    } catch {
      setCopyState('error')
    }
  }

  // -- Render -----------------------------------------------------------------

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div>
          {requestedBackTo && (
            <Link
              to={requestedBackTo}
              className="mb-3 inline-flex items-center gap-2 text-sm text-cyan-300 hover:text-cyan-200"
            >
              <ArrowLeft className="h-4 w-4" />
              {backToLabel(requestedBackTo)}
            </Link>
          )}
          <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
            <Fingerprint className="w-7 h-7 text-cyan-400" />
            Evidence Explorer
          </h1>
          <p className="text-slate-400 mt-1">
            Drill down from claims to review text. Every signal has a witness, every witness has a source.
          </p>
          {activeVendor ? (
            <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
              <span className="text-slate-500">
                Focused on <span className="text-slate-300">{activeVendor}</span>
              </span>
              {hasWatchlistsShortcut ? (
                <Link
                  to={evidenceWatchlistsPath(searchParams, activeVendor, matchedWatchlistViewId)}
                  className="text-violet-300 hover:text-violet-200 transition-colors"
                >
                  Watchlists
                </Link>
              ) : null}
              <Link
                to={evidenceVendorPath(searchParams, activeVendor)}
                className="text-cyan-400 hover:text-cyan-300 transition-colors"
              >
                Vendor workspace
              </Link>
              <Link
                to={evidenceOpportunitiesPath(searchParams, activeVendor)}
                className="text-emerald-300 hover:text-emerald-200 transition-colors"
              >
                Opportunities
              </Link>
              <Link
                to={evidenceReportsPath(searchParams, activeVendor)}
                className="text-fuchsia-300 hover:text-fuchsia-200 transition-colors"
              >
                Reports
              </Link>
            </div>
          ) : null}
        </div>
        <button
          onClick={() => void handleCopyLink()}
          className="inline-flex items-center gap-2 rounded-md border border-slate-700/60 bg-slate-900/60 px-3 py-2 text-sm text-slate-200 hover:border-cyan-500/40 hover:text-white transition-colors"
        >
          <Copy className="h-4 w-4" />
          {copyState === 'copied' ? 'Copied' : copyState === 'error' ? 'Copy Failed' : 'Copy Link'}
        </button>
      </div>

      {/* Search bar */}
      <div className="bg-slate-800/60 rounded-xl border border-slate-700/50 p-4">
        <div className="relative max-w-xl">
          <Search className="absolute left-3 top-2.5 w-4 h-4 text-slate-500" />
          <input
            type="text"
            value={vendorInput}
            onChange={e => handleSearchInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && vendorInput.trim() && selectVendor(vendorInput.trim())}
            placeholder="Search vendor to explore evidence..."
            className="w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
          />
          {showSuggestions && suggestions.length > 0 && (
            <div className="absolute z-20 mt-1 w-full bg-slate-800 border border-slate-600 rounded-lg shadow-lg max-h-48 overflow-y-auto">
              {suggestions.map(name => (
                <button
                  key={name}
                  onClick={() => selectVendor(name)}
                  className="w-full text-left px-4 py-2 text-sm text-slate-200 hover:bg-slate-700 first:rounded-t-lg last:rounded-b-lg"
                >
                  {name}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {!activeVendor && (
        <div className="text-center py-20 text-slate-500">
          <Database className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p>Search for a vendor to explore their evidence chain</p>
        </div>
      )}

      {activeVendor && (
        <>
          {/* Vendor reports link */}
          {activeVendor && (
            <Link
              to={evidenceReportsPath(searchParams, activeVendor)}
              className="inline-flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300"
            >
              View library for {activeVendor} <ExternalLink className="h-3 w-3" />
            </Link>
          )}

          {/* Tab bar */}
          <div className="flex items-center gap-1 border-b border-slate-700/50 pb-px">
            {([
              { key: 'witnesses' as Tab, label: 'Witnesses', icon: Fingerprint, count: total },
              { key: 'vault' as Tab, label: 'Evidence Vault', icon: Database },
              { key: 'trace' as Tab, label: 'Reasoning Trace', icon: GitBranch },
            ]).map(tab => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={clsx(
                  'flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-lg transition-colors',
                  activeTab === tab.key
                    ? 'bg-slate-800/80 text-cyan-400 border-b-2 border-cyan-400'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40',
                )}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
                {tab.count != null && activeTab === tab.key && (
                  <span className="text-xs bg-slate-700 px-1.5 py-0.5 rounded">{tab.count}</span>
                )}
              </button>
            ))}
          </div>

          {/* Witnesses tab */}
          {activeTab === 'witnesses' && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <button
                  onClick={() => setShowFilters((prev) => !prev)}
                  className="inline-flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-slate-400 hover:text-white transition-colors"
                >
                  {showFilters ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                  Filters
                </button>
                {hasActiveFilters && (
                  <button
                    onClick={clearFilters}
                    className="text-xs text-cyan-400 hover:text-cyan-300"
                  >
                    Clear filters
                  </button>
                )}
              </div>
              <div className="flex gap-5">
              {/* Filter sidebar */}
              {showFilters && (
                <div className="w-52 shrink-0 space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-xs font-medium text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                      <Filter className="w-3 h-3" /> Filters
                    </h3>
                    {hasActiveFilters && (
                      <button onClick={clearFilters} className="text-xs text-cyan-400 hover:text-cyan-300">
                        Clear
                      </button>
                    )}
                  </div>

                  {/* Pain category filter */}
                  {facets.pain_categories.length > 0 && (
                    <div>
                      <label className="text-xs text-slate-500 block mb-1">Pain Category</label>
                      <select
                        value={filterPain}
                        onChange={e => { setFilterPain(e.target.value); setOffset(0) }}
                        className="w-full px-2 py-1.5 bg-slate-900 border border-slate-700 rounded text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
                      >
                        <option value="">All</option>
                        {facets.pain_categories.map(c => <option key={c} value={c}>{c}</option>)}
                      </select>
                    </div>
                  )}

                  {/* Source filter */}
                  {facets.sources.length > 0 && (
                    <div>
                      <label className="text-xs text-slate-500 block mb-1">Source</label>
                      <select
                        value={filterSource}
                        onChange={e => { setFilterSource(e.target.value); setOffset(0) }}
                        className="w-full px-2 py-1.5 bg-slate-900 border border-slate-700 rounded text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
                      >
                        <option value="">All sources</option>
                        {facets.sources.map(s => <option key={s} value={s}>{s}</option>)}
                      </select>
                    </div>
                  )}

                  {/* Witness type filter */}
                  {facets.witness_types.length > 0 && (
                    <div>
                      <label className="text-xs text-slate-500 block mb-1">Witness Type</label>
                      <select
                        value={filterType}
                        onChange={e => { setFilterType(e.target.value); setOffset(0) }}
                        className="w-full px-2 py-1.5 bg-slate-900 border border-slate-700 rounded text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
                      >
                        <option value="">All types</option>
                        {facets.witness_types.map(t => <option key={t} value={t}>{t}</option>)}
                      </select>
                    </div>
                  )}

                  {/* Stats */}
                  <div className="pt-2 border-t border-slate-700/30">
                    <div className="text-xs text-slate-500 space-y-1">
                      <div className="flex justify-between">
                        <span>Total witnesses</span>
                        <span className="text-slate-300">{total.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Sources</span>
                        <span className="text-slate-300">{facets.sources.length}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Pain types</span>
                        <span className="text-slate-300">{facets.pain_categories.length}</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Witness cards */}
              <div className="flex-1 min-w-0">
                {loading ? (
                  <div className="flex items-center justify-center py-16">
                    <Loader2 className="w-6 h-6 text-cyan-400 animate-spin" />
                  </div>
                ) : witnesses.length === 0 ? (
                  <div className="text-center py-16 text-slate-500">
                    <Fingerprint className="w-10 h-10 mx-auto mb-3 opacity-20" />
                    <p className="text-sm">No witnesses found{hasActiveFilters ? ' for these filters' : ''}</p>
                  </div>
                ) : (
                  <>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                      {witnesses.map(w => (
                        <button
                          key={`${w.witness_id}-${w.as_of_date}`}
                          onClick={() => openDrawer(w)}
                          className={clsx(
                            'text-left bg-slate-800/50 rounded-lg p-4 border border-slate-700/40',
                            'hover:border-cyan-500/40 hover:bg-slate-800/70 transition-colors',
                            'border-l-2',
                            WITNESS_TYPE_COLORS[w.witness_type || ''] || 'border-l-slate-600',
                          )}
                        >
                          <p className="text-sm text-slate-200 line-clamp-3 italic mb-3">
                            &ldquo;{w.excerpt_text}&rdquo;
                          </p>
                          <div className="flex items-center gap-2 flex-wrap">
                            <SourceBadge source={w.source || 'unknown'} />
                            {w.reviewer_company && (
                              <span className="text-xs text-slate-400 truncate max-w-[120px]">{w.reviewer_company}</span>
                            )}
                            {w.pain_category && (
                              <span className="text-xs px-1.5 py-0.5 rounded bg-slate-700/50 text-slate-300">{w.pain_category}</span>
                            )}
                          </div>
                          {w.salience_score != null && (
                            <div className="mt-2.5 flex items-center gap-2">
                              <div className="flex-1 bg-slate-700/50 rounded-full h-1">
                                <div
                                  className="h-1 rounded-full bg-cyan-500/70"
                                  style={{ width: `${Math.min(w.salience_score * 10, 100)}%` }}
                                />
                              </div>
                              <span className="text-xs text-slate-500 font-mono w-8 text-right">
                                {w.salience_score.toFixed(1)}
                              </span>
                            </div>
                          )}
                          {w.signal_tags && Array.isArray(w.signal_tags) && w.signal_tags.length > 0 && (
                            <div className="flex gap-1 mt-2 flex-wrap">
                              {(Array.isArray(w.signal_tags) ? w.signal_tags : []).slice(0, 3).map((tag, i) => (
                                <span key={i} className={clsx(
                                  'text-xs px-1.5 py-0.5 rounded',
                                  SIGNAL_COLORS[tag] || 'bg-slate-700/50 text-slate-400',
                                )}>
                                  {tag.replace(/_/g, ' ')}
                                </span>
                              ))}
                              {(Array.isArray(w.signal_tags) ? w.signal_tags : []).length > 3 && (
                                <span className="text-xs text-slate-500">+{(Array.isArray(w.signal_tags) ? w.signal_tags : []).length - 3}</span>
                              )}
                            </div>
                          )}
                        </button>
                      ))}
                    </div>

                    {/* Pagination */}
                    {total > limit && (
                      <div className="flex items-center justify-between mt-4 pt-4 border-t border-slate-700/30">
                        <span className="text-xs text-slate-500">
                          Showing {offset + 1}-{Math.min(offset + limit, total)} of {total}
                        </span>
                        <div className="flex gap-2">
                          <button
                            disabled={offset === 0}
                            onClick={() => setOffset(Math.max(0, offset - limit))}
                            className="px-3 py-1 text-xs bg-slate-800 border border-slate-700 rounded text-slate-300 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed"
                          >
                            Previous
                          </button>
                          <button
                            disabled={offset + limit >= total}
                            onClick={() => setOffset(offset + limit)}
                            className="px-3 py-1 text-xs bg-slate-800 border border-slate-700 rounded text-slate-300 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed"
                          >
                            Next
                          </button>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
              </div>
            </div>
          )}

          {/* Vault tab */}
          {activeTab === 'vault' && (
            <div>
              {vaultLoading ? (
                <div className="flex items-center justify-center py-16">
                  <Loader2 className="w-6 h-6 text-cyan-400 animate-spin" />
                </div>
              ) : !vault ? (
                <div className="text-center py-16 text-slate-500">
                  <Database className="w-10 h-10 mx-auto mb-3 opacity-20" />
                  <p className="text-sm">No evidence vault found for {activeVendor}</p>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Metric snapshot */}
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                    {Object.entries(vault.metric_snapshot).map(([key, val]) => (
                      <div key={key} className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/30 text-center">
                        <div className="text-lg font-bold text-white">
                          {typeof val === 'number' ? val.toLocaleString() : String(val)}
                        </div>
                        <div className="text-xs text-slate-500 mt-0.5">{key.replace(/_/g, ' ')}</div>
                      </div>
                    ))}
                  </div>

                  {/* Weakness + strength evidence */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                    <div>
                      <h3 className="text-sm font-medium text-red-400 flex items-center gap-2 mb-3">
                        <ShieldAlert className="w-4 h-4" />
                        Weakness Evidence ({vault.weakness_evidence.length})
                      </h3>
                      <div className="space-y-2">
                        {vault.weakness_evidence.map((ev, i) => {
                          const e = ev as Record<string, unknown>
                          const label = String(e.label || e.key || e.claim || e.text || '')
                          const quote = String(e.best_quote || '')
                          const mentions = Number(e.mention_count_total ?? e.mention_count ?? 0)
                          const confidence = Number(e.confidence_score ?? 0)
                          const trend = e.trend as Record<string, unknown> | undefined
                          return (
                            <div key={i} className="bg-slate-800/40 rounded-lg p-3 border-l-2 border-l-red-500 border border-slate-700/30 overflow-hidden">
                              <div className="flex items-center justify-between gap-2 mb-1">
                                <span className="text-sm font-medium text-slate-200">{label || 'Unknown'}</span>
                                <div className="flex items-center gap-2 shrink-0">
                                  {mentions > 0 && <span className="text-[10px] text-slate-500">{mentions} mentions</span>}
                                  {confidence > 0 && <span className="text-[10px] text-slate-500">{Math.round(confidence * 100)}% conf</span>}
                                </div>
                              </div>
                              {quote && <p className="text-xs text-slate-400 italic line-clamp-2 break-words">\"{quote}\"</p>}
                              {trend && (
                                <div className="flex items-center gap-2 mt-1.5 text-[10px] text-slate-500">
                                  {trend.direction ? <span>{String(trend.direction)}</span> : null}
                                  {trend.recent_count != null ? <span>{String(trend.recent_count)} recent</span> : null}
                                </div>
                              )}
                            </div>
                          )
                        })}
                        {vault.weakness_evidence.length === 0 && (
                          <p className="text-sm text-slate-500 italic">No weakness evidence recorded</p>
                        )}
                      </div>
                    </div>
                    <div>
                      <h3 className="text-sm font-medium text-emerald-400 flex items-center gap-2 mb-3">
                        <ShieldCheck className="w-4 h-4" />
                        Strength Evidence ({vault.strength_evidence.length})
                      </h3>
                      <div className="space-y-2">
                        {vault.strength_evidence.map((ev, i) => {
                          const e = ev as Record<string, unknown>
                          const label = String(e.label || e.key || e.claim || e.text || '')
                          const quote = String(e.best_quote || '')
                          const mentions = Number(e.mention_count_total ?? e.mention_count ?? 0)
                          const confidence = Number(e.confidence_score ?? 0)
                          const trend = e.trend as Record<string, unknown> | undefined
                          return (
                            <div key={i} className="bg-slate-800/40 rounded-lg p-3 border-l-2 border-l-emerald-500 border border-slate-700/30 overflow-hidden">
                              <div className="flex items-center justify-between gap-2 mb-1">
                                <span className="text-sm font-medium text-slate-200">{label || 'Unknown'}</span>
                                <div className="flex items-center gap-2 shrink-0">
                                  {mentions > 0 && <span className="text-[10px] text-slate-500">{mentions} mentions</span>}
                                  {confidence > 0 && <span className="text-[10px] text-slate-500">{Math.round(confidence * 100)}% conf</span>}
                                </div>
                              </div>
                              {quote && <p className="text-xs text-slate-400 italic line-clamp-2 break-words">\"{quote}\"</p>}
                              {trend && (
                                <div className="flex items-center gap-2 mt-1.5 text-[10px] text-slate-500">
                                  {trend.direction ? <span>{String(trend.direction)}</span> : null}
                                  {trend.recent_count != null ? <span>{String(trend.recent_count)} recent</span> : null}
                                </div>
                              )}
                            </div>
                          )
                        })}
                        {vault.strength_evidence.length === 0 && (
                          <p className="text-sm text-slate-500 italic">No strength evidence recorded</p>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Provenance */}
                  <div className="bg-slate-800/30 rounded-lg p-4 border border-slate-700/20">
                    <h3 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Provenance</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                      <div><span className="text-slate-500">As of</span> <span className="text-slate-300 ml-1">{vault.as_of_date}</span></div>
                      <div><span className="text-slate-500">Window</span> <span className="text-slate-300 ml-1">{vault.analysis_window_days} days</span></div>
                      <div><span className="text-slate-500">Version</span> <span className="text-slate-300 ml-1 font-mono">{vault.schema_version}</span></div>
                      <div><span className="text-slate-500">Witnesses</span> <span className="text-slate-300 ml-1">{vault.witness_count}</span></div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Trace tab */}
          {activeTab === 'trace' && (
            <div>
              {traceLoading ? (
                <div className="flex items-center justify-center py-16">
                  <Loader2 className="w-6 h-6 text-cyan-400 animate-spin" />
                </div>
              ) : !trace ? (
                <div className="text-center py-16 text-slate-500">
                  <GitBranch className="w-10 h-10 mx-auto mb-3 opacity-20" />
                  <p className="text-sm">No reasoning trace found for {activeVendor}</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Chain visualization */}
                  <div className="flex items-stretch gap-0">
                    {/* Synthesis */}
                    <TraceNode
                      label="Synthesis"
                      icon={<Layers className="w-4 h-4" />}
                      active={trace.stats.has_synthesis}
                      detail={trace.trace.synthesis ? (
                        <div className="text-xs space-y-1">
                          <div className="text-slate-400">Model: <span className="text-slate-200 font-mono">{trace.trace.synthesis.llm_model || 'unknown'}</span></div>
                          <div className="text-slate-400">Hash: <span className="text-slate-200 font-mono truncate">{trace.trace.synthesis.evidence_hash?.slice(0, 16)}...</span></div>
                          <div className="text-slate-400">Sections: <span className="text-slate-200">{Object.keys(trace.trace.synthesis.sections).length}</span></div>
                          {trace.trace.synthesis.tokens_used && (
                            <div className="text-slate-400">Tokens: <span className="text-slate-200">{trace.trace.synthesis.tokens_used.toLocaleString()}</span></div>
                          )}
                        </div>
                      ) : null}
                    />
                    <ChainArrow />

                    {/* Reasoning packet */}
                    <TraceNode
                      label="Reasoning Packet"
                      icon={<FileText className="w-4 h-4" />}
                      active={trace.stats.has_packet}
                      detail={trace.trace.reasoning_packet ? (
                        <div className="text-xs space-y-1">
                          <div className="text-slate-400">Sections: <span className="text-slate-200">{trace.trace.reasoning_packet.section_count}</span></div>
                          <div className="text-slate-400">Witness pack: <span className="text-slate-200">{trace.trace.reasoning_packet.witness_pack_size}</span></div>
                        </div>
                      ) : null}
                    />
                    <ChainArrow />

                    {/* Witnesses */}
                    <TraceNode
                      label="Witnesses"
                      icon={<Fingerprint className="w-4 h-4" />}
                      active={trace.stats.witness_count > 0}
                      detail={
                        <div className="text-xs text-slate-400">
                          <span className="text-slate-200">{trace.stats.witness_count}</span> witness records
                        </div>
                      }
                    />
                    <ChainArrow />

                    {/* Source reviews */}
                    <TraceNode
                      label="Source Reviews"
                      icon={<Database className="w-4 h-4" />}
                      active={trace.stats.unique_reviews > 0}
                      detail={
                        <div className="text-xs text-slate-400">
                          <span className="text-slate-200">{trace.stats.unique_reviews}</span> unique reviews
                        </div>
                      }
                    />
                  </div>

                  {/* Evidence diff */}
                  {trace.trace.evidence_diff && (
                    <div className="bg-slate-800/40 rounded-lg p-4 border border-slate-700/30">
                      <h3 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Latest Evidence Diff</h3>
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-center">
                        <DiffStat label="Confirmed" value={trace.trace.evidence_diff.confirmed_count} color="text-emerald-400" />
                        <DiffStat label="Novel" value={trace.trace.evidence_diff.novel_count} color="text-cyan-400" />
                        <DiffStat label="Contradicted" value={trace.trace.evidence_diff.contradicted_count} color="text-red-400" />
                        <DiffStat label="Missing" value={trace.trace.evidence_diff.missing_count} color="text-amber-400" />
                        <div>
                          <div className="text-sm font-medium text-slate-200">{trace.trace.evidence_diff.decision}</div>
                          <div className="text-xs text-slate-500">Decision</div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Top witnesses from trace */}
                  {trace.trace.witnesses.length > 0 && (
                    <div>
                      <h3 className="text-sm font-medium text-slate-300 mb-3">Top Witnesses (by salience)</h3>
                      <div className="space-y-2">
                        {trace.trace.witnesses.slice(0, 10).map(w => (
                          <button
                            key={w.witness_id}
                            onClick={() => openDrawer(w)}
                            className="w-full text-left bg-slate-800/40 rounded-lg p-3 border border-slate-700/30 hover:border-cyan-500/30 transition-colors flex items-start gap-3"
                          >
                            <span className={clsx(
                              'shrink-0 w-1 h-full min-h-[40px] rounded-full',
                              w.witness_type === 'strength' ? 'bg-emerald-500' :
                              w.witness_type === 'displacement' ? 'bg-violet-500' :
                              'bg-red-500',
                            )} />
                            <div className="flex-1 min-w-0">
                              <p className="text-sm text-slate-200 line-clamp-2 italic">&ldquo;{w.excerpt_text}&rdquo;</p>
                              <div className="flex items-center gap-2 mt-1.5">
                                <SourceBadge source={w.source || 'unknown'} />
                                {w.reviewer_company && <span className="text-xs text-slate-500">{w.reviewer_company}</span>}
                                {w.salience_score != null && <span className="text-xs text-slate-500 font-mono ml-auto">{w.salience_score.toFixed(2)}</span>}
                              </div>
                            </div>
                            <ChevronRight className="w-4 h-4 text-slate-600 shrink-0 mt-1" />
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Source reviews */}
                  {trace.trace.source_reviews.length > 0 && (
                    <div>
                      <h3 className="text-sm font-medium text-slate-300 mb-3">Source Reviews ({trace.trace.source_reviews.length})</h3>
                      <div className="space-y-2">
                        {trace.trace.source_reviews.map(r => (
                          <div key={r.id} className="bg-slate-800/30 rounded-lg p-3 border border-slate-700/20 flex items-start gap-3">
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-1">
                                <SourceBadge source={r.source || 'unknown'} />
                                {r.rating != null && <span className="text-xs text-slate-400">{r.rating}/5</span>}
                                {r.reviewer_company && <span className="text-xs text-slate-500">{r.reviewer_company}</span>}
                              </div>
                              {r.summary && <p className="text-sm text-slate-300 line-clamp-2">{r.summary}</p>}
                              {!r.summary && r.review_excerpt && (
                                <p className="text-xs text-slate-400 line-clamp-2">{r.review_excerpt}</p>
                              )}
                            </div>
                            {r.source_url && (
                              <a href={r.source_url} target="_blank" rel="noopener noreferrer"
                                 className="text-cyan-400 hover:text-cyan-300 shrink-0">
                                <ChevronRight className="w-4 h-4" />
                              </a>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </>
      )}

      {/* Drawer */}
      <EvidenceDrawer
        vendorName={activeVendor}
        witnessId={drawerWitnessId}
        asOfDate={witnessSnapshotDate}
        windowDays={windowDays}
        open={drawerOpen}
        backToPath={drawerBackToPath}
        onClose={() => {
          setDrawerOpen(false)
          setDrawerWitnessId(null)
        }}
      />
    </div>
  )
}


// -- Sub-components -----------------------------------------------------------

function TraceNode({ label, icon, active, detail }: {
  label: string
  icon: ReactNode
  active: boolean
  detail: ReactNode
}) {
  return (
    <div className={clsx(
      'flex-1 bg-slate-800/50 rounded-lg p-4 border text-center',
      active ? 'border-cyan-500/30' : 'border-slate-700/30 opacity-40',
    )}>
      <div className={clsx('flex items-center justify-center gap-2 mb-2', active ? 'text-cyan-400' : 'text-slate-500')}>
        {icon}
        <span className="text-xs font-medium uppercase tracking-wider">{label}</span>
      </div>
      {active ? (
        <div className="text-xs text-slate-400">{detail}</div>
      ) : (
        <div className="text-xs text-slate-600">No data</div>
      )}
    </div>
  )
}

function ChainArrow() {
  return (
    <div className="flex items-center px-1 text-slate-600">
      <ArrowRight className="w-4 h-4" />
    </div>
  )
}

function DiffStat({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div>
      <div className={clsx('text-lg font-bold', color)}>{value}</div>
      <div className="text-xs text-slate-500">{label}</div>
    </div>
  )
}
