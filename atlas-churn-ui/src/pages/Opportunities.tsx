import { useState, useEffect, useMemo, useCallback } from 'react'
import { Link } from 'react-router-dom'
import {
  Telescope,
  Target,
  TrendingUp,
  RefreshCw,
  Zap,
  Download,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  AlertTriangle,
  Building2,
  Globe,
  Users,
  X,
} from 'lucide-react'
import { clsx } from 'clsx'
import StatCard from '../components/StatCard'
import DataTable, { type Column } from '../components/DataTable'
import UrgencyBadge from '../components/UrgencyBadge'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import { usePlanGate } from '../hooks/usePlanGate'
import {
  fetchHighIntent,
  generateCampaigns,
  downloadCsv,
} from '../api/client'
import type { HighIntentCompany } from '../types'

function rowKey(r: HighIntentCompany): string {
  return r.review_id ?? `${r.company}::${r.vendor}`
}

const STAGE_COLORS: Record<string, string> = {
  active_purchase: 'text-red-400',
  renewal_decision: 'text-amber-400',
  evaluation: 'text-cyan-400',
  post_purchase: 'text-slate-400',
}

const STAGE_OPTIONS = ['all', 'active_purchase', 'renewal_decision', 'evaluation', 'post_purchase'] as const

const INTENT_KEYS = ['cancel', 'migration', 'evaluation', 'completed_switch'] as const

function IntentChip({ label, active }: { label: string; active: boolean }) {
  return (
    <span
      className={clsx(
        'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium',
        active ? 'bg-red-500/15 text-red-300' : 'bg-slate-700/30 text-slate-500',
      )}
    >
      {label.replace(/_/g, ' ')}
    </span>
  )
}

const WINDOW_OPTIONS = [
  { label: '30 days', value: 30 },
  { label: '90 days', value: 90 },
  { label: '180 days', value: 180 },
  { label: '1 year', value: 365 },
  { label: 'All time', value: 3650 },
] as const

export default function Opportunities() {
  const { canAccessCampaigns } = usePlanGate()

  // -- Filters --
  const [vendorSearch, setVendorSearch] = useState('')
  const [debouncedVendor, setDebouncedVendor] = useState('')
  const [minUrgency, setMinUrgency] = useState(3)
  const [windowDays, setWindowDays] = useState(365)
  const [stageFilter, setStageFilter] = useState<string>('all')
  const [intentFilter, setIntentFilter] = useState<Set<string>>(new Set())

  // -- Expansion --
  const [expandedId, setExpandedId] = useState<string | null>(null)

  // -- Bulk selection --
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())

  // -- Generate --
  const [generating, setGenerating] = useState<string | null>(null)
  const [bulkGenerating, setBulkGenerating] = useState(false)
  const [bulkProgress, setBulkProgress] = useState<{ done: number; total: number } | null>(null)
  const [genResult, setGenResult] = useState<string | null>(null)

  useEffect(() => {
    const t = setTimeout(() => setDebouncedVendor(vendorSearch), 300)
    return () => clearTimeout(t)
  }, [vendorSearch])

  // Clear selection when server-side filters change (data will change)
  useEffect(() => {
    setSelectedIds(new Set())
    setExpandedId(null)
  }, [debouncedVendor, minUrgency, windowDays])

  // -- Data fetch --
  const { data, loading, error, refresh, refreshing } = useApiData(
    async () => {
      const res = await fetchHighIntent({
        min_urgency: minUrgency,
        vendor_name: debouncedVendor || undefined,
        window_days: windowDays,
        limit: 100,
      })
      return res.companies
    },
    [minUrgency, debouncedVendor, windowDays],
  )

  const opportunities = data ?? []

  // Clear selection when client-side filters change
  useEffect(() => {
    setSelectedIds(new Set())
  }, [stageFilter, intentFilter])

  // -- Client-side filters --
  const filtered = useMemo(() => {
    let rows = opportunities
    if (stageFilter !== 'all') {
      rows = rows.filter((r) => r.buying_stage === stageFilter)
    }
    if (intentFilter.size > 0) {
      rows = rows.filter((r) => {
        if (!r.intent_signals) return false
        const signals = r.intent_signals
        return Array.from(intentFilter).every(
          (k) => signals[k as keyof typeof signals],
        )
      })
    }
    return rows
  }, [opportunities, stageFilter, intentFilter])

  // -- Stats --
  const stats = useMemo(() => {
    const total = filtered.length
    const highUrgency = filtered.filter((r) => r.urgency >= 8).length
    const dmCount = filtered.filter((r) => r.decision_maker).length
    const dmPct = total > 0 ? Math.round((dmCount / total) * 100) : 0
    const avgUrg = total > 0 ? Math.round((filtered.reduce((s, r) => s + r.urgency, 0) / total) * 10) / 10 : 0
    const activePurchase = filtered.filter((r) => r.buying_stage === 'active_purchase').length
    return { total, highUrgency, dmPct, avgUrg, activePurchase }
  }, [filtered])

  // -- Toggle --
  const toggleSelect = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const toggleSelectAll = useCallback(() => {
    setSelectedIds((prev) => {
      if (prev.size === filtered.length) return new Set()
      return new Set(filtered.map(rowKey))
    })
  }, [filtered])

  // -- Generate single --
  async function handleGenerate(row: HighIntentCompany) {
    const key = rowKey(row)
    setGenerating(key)
    setGenResult(null)
    try {
      const result = await generateCampaigns({
        company_name: row.company,
        vendor_name: row.vendor,
        min_score: 70,
        limit: 5,
      })
      setGenResult(`Generated ${result.generated ?? 0} campaign(s) for ${row.company}`)
      refresh()
    } catch (err) {
      setGenResult(err instanceof Error ? err.message : 'Generation failed')
    } finally {
      setGenerating(null)
    }
  }

  // -- Bulk generate --
  async function handleBulkGenerate() {
    if (selectedIds.size === 0) return
    setBulkGenerating(true)
    setGenResult(null)
    const selected = filtered.filter((r) => selectedIds.has(rowKey(r)))
    setBulkProgress({ done: 0, total: selected.length })
    try {
      let totalGenerated = 0
      for (let i = 0; i < selected.length; i++) {
        const item = selected[i]
        const result = await generateCampaigns({
          company_name: item.company,
          vendor_name: item.vendor,
          min_score: 70,
          limit: 5,
        })
        totalGenerated += result.generated ?? 0
        setBulkProgress({ done: i + 1, total: selected.length })
      }
      setGenResult(`Generated ${totalGenerated} campaign(s) for ${selected.length} companies`)
      setSelectedIds(new Set())
      refresh()
    } catch (err) {
      setGenResult(err instanceof Error ? err.message : 'Bulk generation failed')
    } finally {
      setBulkGenerating(false)
      setBulkProgress(null)
    }
  }

  // -- Table columns --
  const columns: Column<HighIntentCompany>[] = useMemo(() => [
    {
      key: 'select',
      header: '',
      render: (r) => (
        <input
          type="checkbox"
          checked={selectedIds.has(rowKey(r))}
          onChange={(e) => { e.stopPropagation(); toggleSelect(rowKey(r)) }}
          onClick={(e) => e.stopPropagation()}
          className="accent-cyan-500"
        />
      ),
    },
    {
      key: 'company',
      header: 'Company',
      render: (r) => (
        <div className="min-w-0">
          <span className="text-white font-medium">{r.company || '--'}</span>
          {r.company_domain && (
            <span className="text-xs text-slate-500 ml-1.5">{r.company_domain}</span>
          )}
        </div>
      ),
    },
    {
      key: 'vendor',
      header: 'Churning From',
      render: (r) => <span className="text-slate-300">{r.vendor}</span>,
    },
    {
      key: 'urgency',
      header: 'Urgency',
      render: (r) => <UrgencyBadge score={r.urgency} />,
      sortable: true,
      sortValue: (r) => r.urgency,
    },
    {
      key: 'stage',
      header: 'Stage',
      render: (r) => {
        if (!r.buying_stage || r.buying_stage === 'unknown')
          return <span className="text-slate-500 text-xs">--</span>
        return (
          <span className={`text-xs font-medium ${STAGE_COLORS[r.buying_stage] ?? 'text-slate-400'}`}>
            {r.buying_stage.replace(/_/g, ' ')}
          </span>
        )
      },
    },
    {
      key: 'dm',
      header: 'DM',
      render: (r) =>
        r.decision_maker
          ? <span className="text-cyan-400 text-xs font-medium">Yes</span>
          : <span className="text-slate-500 text-xs">No</span>,
    },
    {
      key: 'pain',
      header: 'Pain',
      render: (r) => (
        <span className="text-xs text-slate-400 max-w-[140px] truncate block" title={r.pain ?? ''}>
          {r.pain ?? '--'}
        </span>
      ),
    },
    {
      key: 'seats',
      header: 'Seats',
      render: (r) => <span className="text-slate-300">{r.seat_count ?? '--'}</span>,
      sortable: true,
      sortValue: (r) => r.seat_count ?? 0,
    },
    {
      key: 'intent',
      header: 'Intent',
      render: (r) => {
        const signals = r.intent_signals
        if (!signals) return <span className="text-slate-500 text-xs">--</span>
        const active = INTENT_KEYS.filter((k) => signals[k])
        if (active.length === 0) return <span className="text-slate-500 text-xs">--</span>
        return (
          <div className="flex flex-wrap gap-1">
            {active.map((k) => (
              <span key={k} className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-red-500/15 text-red-300">
                {k.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        )
      },
    },
    {
      key: 'confidence',
      header: 'Conf.',
      render: (r) => {
        const score = r.relevance_score
        if (score == null) return <span className="text-slate-500 text-xs">--</span>
        return <span className="text-xs text-slate-300">{(score * 100).toFixed(0)}%</span>
      },
      sortable: true,
      sortValue: (r) => r.relevance_score ?? 0,
    },
    {
      key: 'actions',
      header: '',
      render: (r) => {
        if (!canAccessCampaigns) return null
        const key = rowKey(r)
        return (
          <button
            onClick={(e) => { e.stopPropagation(); handleGenerate(r) }}
            disabled={generating === key}
            className="p-1 text-cyan-400 hover:text-cyan-300 transition-colors disabled:opacity-50"
            title="Generate Campaign"
          >
            <Zap className={clsx('h-3.5 w-3.5', generating === key && 'animate-pulse')} />
          </button>
        )
      },
    },
  ], [selectedIds, toggleSelect, canAccessCampaigns, generating])

  // -- Expanded row --
  const expandedRow = useMemo(
    () => filtered.find((r) => rowKey(r) === expandedId) ?? null,
    [filtered, expandedId],
  )

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Opportunity Workbench</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={() =>
              downloadCsv('/export/high-intent', {
                vendor_name: debouncedVendor || undefined,
                min_urgency: minUrgency || undefined,
                window_days: windowDays,
              })
            }
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors"
          >
            <Download className="h-4 w-4" />
            Export
          </button>
          <button
            onClick={refresh}
            disabled={refreshing}
            className="p-2 text-slate-400 hover:text-white transition-colors"
            title="Refresh"
          >
            <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
          </button>
        </div>
      </div>

      {/* Result banner */}
      {genResult && (
        <div className="flex items-center justify-between rounded-lg border border-slate-700/40 bg-slate-800/40 px-4 py-2 text-sm text-slate-300">
          <span>{genResult}</span>
          <button onClick={() => setGenResult(null)} className="text-slate-500 hover:text-white ml-3 shrink-0">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {/* Bulk progress */}
      {bulkProgress && (
        <div className="rounded-lg border border-cyan-700/40 bg-cyan-900/20 px-4 py-2 text-sm text-cyan-300">
          Generating campaigns... {bulkProgress.done}/{bulkProgress.total}
        </div>
      )}

      {/* Stat cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
        <StatCard label="Opportunities" value={stats.total} icon={<Telescope className="h-4 w-4 text-cyan-400" />} skeleton={loading} />
        <StatCard label="High Urgency (8+)" value={stats.highUrgency} icon={<AlertTriangle className="h-4 w-4 text-red-400" />} skeleton={loading} />
        <StatCard label="Decision Makers" value={`${stats.dmPct}%`} icon={<Users className="h-4 w-4 text-amber-400" />} skeleton={loading} />
        <StatCard label="Avg Urgency" value={stats.avgUrg} icon={<TrendingUp className="h-4 w-4 text-amber-400" />} skeleton={loading} />
        <StatCard label="Active Purchase" value={stats.activePurchase} icon={<Target className="h-4 w-4 text-red-400" />} skeleton={loading} />
      </div>

      {/* Filters + bulk actions */}
      <div className="flex flex-wrap items-center gap-3">
        <input
          type="text"
          placeholder="Filter vendor..."
          value={vendorSearch}
          onChange={(e) => setVendorSearch(e.target.value)}
          className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 w-48"
        />
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-400">Min urgency</label>
          <input
            type="range"
            min={0}
            max={10}
            step={1}
            value={minUrgency}
            onChange={(e) => setMinUrgency(Number(e.target.value))}
            className="w-20 accent-cyan-500"
          />
          <span className="text-xs text-white w-4">{minUrgency}</span>
        </div>
        <select
          value={windowDays}
          onChange={(e) => setWindowDays(Number(e.target.value))}
          className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white"
        >
          {WINDOW_OPTIONS.map((w) => (
            <option key={w.value} value={w.value}>{w.label}</option>
          ))}
        </select>
        <select
          value={stageFilter}
          onChange={(e) => setStageFilter(e.target.value)}
          className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white"
        >
          {STAGE_OPTIONS.map((s) => (
            <option key={s} value={s}>
              {s === 'all' ? 'All stages' : s.replace(/_/g, ' ')}
            </option>
          ))}
        </select>
        <div className="flex items-center gap-2">
          {INTENT_KEYS.map((k) => (
            <label key={k} className="flex items-center gap-1 text-xs text-slate-400 cursor-pointer">
              <input
                type="checkbox"
                checked={intentFilter.has(k)}
                onChange={() => {
                  setIntentFilter((prev) => {
                    const next = new Set(prev)
                    if (next.has(k)) next.delete(k)
                    else next.add(k)
                    return next
                  })
                }}
                className="accent-cyan-500"
              />
              {k.replace(/_/g, ' ')}
            </label>
          ))}
        </div>

        {/* Bulk actions */}
        {selectedIds.size > 0 && canAccessCampaigns && (
          <div className="ml-auto flex items-center gap-2">
            <span className="text-xs text-slate-400">{selectedIds.size} selected</span>
            <button
              onClick={handleBulkGenerate}
              disabled={bulkGenerating}
              className="px-3 py-1.5 text-xs font-medium bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg disabled:opacity-50"
            >
              {bulkGenerating ? 'Generating...' : 'Generate Campaigns'}
            </button>
            <button
              onClick={() => setSelectedIds(new Set())}
              className="px-2 py-1.5 text-xs text-slate-400 hover:text-white"
            >
              Clear
            </button>
          </div>
        )}
      </div>

      {/* Select all row */}
      {filtered.length > 0 && (
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <input
            type="checkbox"
            checked={selectedIds.size === filtered.length && filtered.length > 0}
            onChange={toggleSelectAll}
            className="accent-cyan-500"
          />
          <span>Select all ({filtered.length})</span>
        </div>
      )}

      {/* Table */}
      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-1">
        <DataTable<HighIntentCompany>
          columns={columns}
          data={filtered}
          skeletonRows={loading ? 8 : undefined}
          emptyMessage="No opportunities found. Try lowering min urgency, extending the time window, or removing filters."
          onRowClick={(r) => setExpandedId((prev) => prev === rowKey(r) ? null : rowKey(r))}
          pageSize={50}
        />
      </div>

      {/* Evidence panel */}
      {expandedRow && (
        <EvidencePanel
          row={expandedRow}
          onClose={() => setExpandedId(null)}
          onGenerate={canAccessCampaigns ? handleGenerate : undefined}
          generating={generating === rowKey(expandedRow)}
        />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Evidence Panel
// ---------------------------------------------------------------------------

function EvidencePanel({
  row,
  onClose,
  onGenerate,
  generating,
}: {
  row: HighIntentCompany
  onClose: () => void
  onGenerate?: (row: HighIntentCompany) => void
  generating: boolean
}) {
  const signals = row.intent_signals
  const alternatives = Array.isArray(row.alternatives)
    ? row.alternatives.filter((a) => a?.name?.trim())
    : []
  const quotes = Array.isArray(row.quotes)
    ? row.quotes.filter((q) => typeof q === 'string' && q.trim())
    : []

  return (
    <div className="bg-slate-900/60 border border-slate-700/50 rounded-xl p-5 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 min-w-0">
          <h3 className="text-lg font-semibold text-white truncate">{row.company}</h3>
          <span className="text-xs text-slate-500 shrink-0">churning from</span>
          <span className="text-sm text-slate-300 font-medium shrink-0">{row.vendor}</span>
          <UrgencyBadge score={row.urgency} />
          {row.resolution_confidence && row.resolution_confidence !== 'high' && (
            <span className="text-xs text-amber-500 shrink-0">resolution: {row.resolution_confidence}</span>
          )}
        </div>
        <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors">
          <ChevronDown className="h-4 w-4" />
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 items-start">
        {/* Column 1: Evidence */}
        <div className="space-y-4">
          <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider">Review Evidence</h4>
          {quotes.length > 0 ? (
            <div className="space-y-2">
              {quotes.slice(0, 5).map((q, i) => (
                <blockquote key={i} className="text-sm text-slate-300 italic border-l-2 border-amber-500/50 pl-3 break-words">
                  "{q}"
                </blockquote>
              ))}
            </div>
          ) : (
            <p className="text-xs text-slate-500">No quotes available</p>
          )}
          {row.source && (
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-slate-800 text-slate-300">
              {row.source}
            </span>
          )}
          {row.review_id && (
            <Link
              to={`/reviews/${row.review_id}`}
              className="inline-flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300"
            >
              View full review <ExternalLink className="h-3 w-3" />
            </Link>
          )}
        </div>

        {/* Column 2: Intent + Competitors */}
        <div className="space-y-4">
          <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider">Intent Signals</h4>
          {signals ? (
            <div className="flex flex-wrap gap-2">
              {INTENT_KEYS.map((k) => (
                <IntentChip key={k} label={k} active={!!signals[k]} />
              ))}
            </div>
          ) : (
            <p className="text-xs text-slate-500">No intent data</p>
          )}

          {alternatives.length > 0 && (
            <>
              <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider mt-4">Considering</h4>
              <div className="space-y-2">
                {alternatives.slice(0, 5).map((alt, i) => (
                  <div key={i} className="bg-slate-800/50 rounded p-2">
                    <span className="text-sm text-cyan-400 font-medium">{alt.name}</span>
                    {alt.context && <p className="text-xs text-slate-400 mt-0.5">{alt.context}</p>}
                    {alt.reason && <p className="text-xs text-slate-500 mt-0.5">{alt.reason}</p>}
                  </div>
                ))}
              </div>
            </>
          )}

          {row.category && (
            <div className="mt-2">
              <span className="text-xs text-slate-400">Category: </span>
              <span className="text-xs text-slate-300">{row.category}</span>
            </div>
          )}
          {row.contract_end && (
            <div className="mt-1">
              <span className="text-xs text-slate-400">Contract end: </span>
              <span className="text-xs text-amber-400">{row.contract_end}</span>
            </div>
          )}
          {row.contract_signal && (
            <div className="mt-1">
              <span className="text-xs text-slate-400">Contract signal: </span>
              <span className="text-xs text-slate-300">{row.contract_signal}</span>
            </div>
          )}
        </div>

        {/* Column 3: Company enrichment */}
        <div className="space-y-4">
          <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider">Company Profile</h4>
          <div className="space-y-2 text-sm">
            {row.industry && (
              <div className="flex items-center gap-2">
                <Building2 className="h-3.5 w-3.5 text-slate-500 shrink-0" />
                <span className="text-slate-300">{row.industry}</span>
              </div>
            )}
            {row.verified_employee_count != null && (
              <div className="flex items-center gap-2">
                <Users className="h-3.5 w-3.5 text-slate-500 shrink-0" />
                <span className="text-slate-300">{row.verified_employee_count.toLocaleString()} employees</span>
              </div>
            )}
            {row.company_country && (
              <div className="flex items-center gap-2">
                <Globe className="h-3.5 w-3.5 text-slate-500 shrink-0" />
                <span className="text-slate-300">{row.company_country}</span>
              </div>
            )}
            {row.revenue_range && (
              <div className="flex items-center gap-2">
                <TrendingUp className="h-3.5 w-3.5 text-slate-500 shrink-0" />
                <span className="text-slate-300">{row.revenue_range}</span>
              </div>
            )}
            {row.founded_year != null && (
              <div className="text-xs text-slate-500">Founded {row.founded_year}</div>
            )}
            {row.company_description && (
              <p className="text-xs text-slate-400 mt-2">{row.company_description}</p>
            )}
            {row.company_domain && (
              <span className="text-xs text-slate-500">{row.company_domain}</span>
            )}
          </div>

          {/* Buyer context */}
          <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider mt-3">Buyer Context</h4>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-slate-500">Role: </span>
              <span className="text-slate-300">{row.reviewer_title ?? row.role_level ?? '--'}</span>
            </div>
            <div>
              <span className="text-slate-500">Seats: </span>
              <span className="text-slate-300">{row.seat_count ?? '--'}</span>
            </div>
            <div>
              <span className="text-slate-500">Lock-in: </span>
              <span className="text-slate-300">{row.lock_in_level ?? '--'}</span>
            </div>
            <div>
              <span className="text-slate-500">Size: </span>
              <span className="text-slate-300">{row.company_size ?? '--'}</span>
            </div>
            {row.relevance_score != null && (
              <div>
                <span className="text-slate-500">Relevance: </span>
                <span className="text-slate-300">{(row.relevance_score * 100).toFixed(0)}%</span>
              </div>
            )}
            {row.author_churn_score != null && row.author_churn_score > 0 && (
              <div>
                <span className="text-slate-500">Churn score: </span>
                <span className="text-amber-400">{(row.author_churn_score * 100).toFixed(0)}%</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-3 pt-2 border-t border-slate-700/40">
        {onGenerate && (
          <button
            onClick={() => onGenerate(row)}
            disabled={generating}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm bg-cyan-600 hover:bg-cyan-500 text-white transition-colors disabled:opacity-50"
          >
            <Zap className={clsx('h-4 w-4', generating && 'animate-pulse')} />
            {generating ? 'Generating...' : 'Generate Campaign'}
          </button>
        )}
        <Link
          to="/reviews"
          className="inline-flex items-center gap-1 text-sm text-slate-400 hover:text-white transition-colors"
        >
          Browse reviews <ChevronRight className="h-3.5 w-3.5" />
        </Link>
      </div>
    </div>
  )
}
