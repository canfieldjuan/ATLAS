import { useState, useEffect, useMemo, useCallback } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import {
  Activity,
  ArrowLeft,
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
  CheckCircle2,
  XCircle,
  Mail,
  Loader2,
  Bookmark,
  EyeOff,
  Clock,
  Pencil,
} from 'lucide-react'
import { clsx } from 'clsx'
import StatCard from '../components/StatCard'
import CampaignReasoningSummary from '../components/CampaignReasoningSummary'
import CompanyTimeline from '../components/CompanyTimeline'
import SignalEffectivenessPanel from '../components/SignalEffectivenessPanel'
import DataTable, { type Column } from '../components/DataTable'
import UrgencyBadge from '../components/UrgencyBadge'
import { PageError } from '../components/ErrorBoundary'
import CampaignFailureExplanation from '../components/CampaignFailureExplanation'
import CampaignQualityTrends from '../components/CampaignQualityTrends'
import useApiData from '../hooks/useApiData'
import { usePlanGate } from '../hooks/usePlanGate'
import {
  fetchHighIntent,
  fetchCampaigns,
  generateCampaigns,
  approveCampaign,
  updateCampaign,
  pushToCrm,
  downloadCsv,
  fetchDispositions,
  setDisposition,
  bulkSetDisposition,
  removeDispositions,
  fetchCampaignStats,
  fetchCampaignQualityTrends,
} from '../api/client'
import type { Campaign, HighIntentCompany } from '../types'
import type { OpportunityDisposition } from '../api/client'

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

const DISPOSITION_TABS = ['active', 'saved', 'snoozed', 'dismissed', 'all'] as const
type DispositionTab = (typeof DISPOSITION_TABS)[number]

function resolveBackTarget(rawValue: string | null): string | null {
  const value = rawValue?.trim() || ''
  if (!value) return null
  if (value.startsWith('/vendors/') || value.startsWith('/watchlists')) return value
  return null
}

function backButtonLabel(target: string | null): string {
  if (!target) return 'Back'
  if (target.startsWith('/vendors/')) return 'Back to Vendor'
  if (target.startsWith('/watchlists')) return 'Back to Watchlists'
  return 'Back'
}

function opportunitiesPath(vendorName: string, backTarget: string | null): string {
  const params = new URLSearchParams()
  if (vendorName) params.set('vendor', vendorName)
  if (backTarget) params.set('back_to', backTarget)
  const query = params.toString()
  return query ? `/opportunities?${query}` : '/opportunities'
}

function evidencePath(vendorName: string, returnPath: string): string {
  const params = new URLSearchParams()
  params.set('vendor', vendorName)
  params.set('tab', 'witnesses')
  params.set('back_to', returnPath)
  return `/evidence?${params.toString()}`
}

function reportsPath(vendorName: string, returnPath: string): string {
  const params = new URLSearchParams()
  params.set('vendor_filter', vendorName)
  params.set('back_to', returnPath)
  return `/reports?${params.toString()}`
}

function reviewDetailPath(reviewId: string, returnPath: string): string {
  const params = new URLSearchParams()
  params.set('back_to', returnPath)
  return `/reviews/${encodeURIComponent(reviewId)}?${params.toString()}`
}

function vendorDetailPath(vendorName: string): string {
  return `/vendors/${encodeURIComponent(vendorName)}`
}

function CollapsibleSection({ title, defaultOpen = false, children }: {
  title: string; defaultOpen?: boolean; children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border border-slate-700/40 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-slate-300 hover:bg-slate-800/30 transition-colors"
      >
        {title}
        {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
      </button>
      {open && <div className="px-4 pb-4">{children}</div>}
    </div>
  )
}

export default function Opportunities() {
  const [searchParams] = useSearchParams()
  const initialVendor = searchParams.get('vendor') || ''
  const backTarget = resolveBackTarget(searchParams.get('back_to'))

  const { canAccessCampaigns } = usePlanGate()

  // -- Filters --
  const [vendorSearch, setVendorSearch] = useState(initialVendor)
  const [debouncedVendor, setDebouncedVendor] = useState(initialVendor)
  const [minUrgency, setMinUrgency] = useState(5)
  const [windowDays, setWindowDays] = useState(90)
  const [stageFilter, setStageFilter] = useState<string>('all')
  const [intentFilter, setIntentFilter] = useState<Set<string>>(new Set())

  // -- Expansion --
  const [expandedId, setExpandedId] = useState<string | null>(null)

  // -- Bulk selection --
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())

  // -- Disposition (persistent save/snooze/dismiss) --
  const [dispositionTab, setDispositionTab] = useState<DispositionTab>('active')
  const { data: dispData, refresh: refreshDispositions } = useApiData(
    () => fetchDispositions(),
    [],
  )
  const dispositionMap = useMemo(() => {
    const map = new Map<string, OpportunityDisposition>()
    for (const d of dispData?.dispositions ?? []) {
      map.set(d.opportunity_key, d)
    }
    return map
  }, [dispData])

  // -- Campaign refresh signal: increment to trigger CampaignQueue refetch --
  const [campaignRefreshKey, setCampaignRefreshKey] = useState(0)

  // -- Generate --
  const [generating, setGenerating] = useState<string | null>(null)
  const [bulkGenerating, setBulkGenerating] = useState(false)
  const [bulkProgress, setBulkProgress] = useState<{ done: number; total: number } | null>(null)
  const [genResult, setGenResult] = useState<string | null>(null)

  useEffect(() => {
    const t = setTimeout(() => setDebouncedVendor(vendorSearch), 300)
    return () => clearTimeout(t)
  }, [vendorSearch])

  useEffect(() => {
    setVendorSearch(initialVendor)
    setDebouncedVendor(initialVendor)
    setExpandedId(null)
    setSelectedIds(new Set())
  }, [initialVendor])

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

  const opportunities = useMemo(() => data ?? [], [data])

  // -- Campaign stats --
  const { data: campaignStats, loading: campaignStatsLoading, refresh: refreshCampaignStats } = useApiData(fetchCampaignStats, [])
  const { data: qualityTrends, loading: qualityTrendsLoading } = useApiData(
    () => fetchCampaignQualityTrends({ days: 14, top_n: 5 }),
    [],
  )

  // -- Campaign status lookup --
  const { data: campaignData, refresh: refreshCampaigns } = useApiData(
    () => fetchCampaigns({ limit: 100 }),
    [],
  )
  const campaignMap = useMemo(() => {
    const map = new Map<string, { drafts: number; approved: number; sent: number; total: number }>()
    for (const c of campaignData?.campaigns ?? []) {
      const key = `${(c.company_name ?? '').toLowerCase()}::${(c.vendor_name ?? '').toLowerCase()}`
      const entry = map.get(key) ?? { drafts: 0, approved: 0, sent: 0, total: 0 }
      entry.total++
      if (c.status === 'draft') entry.drafts++
      else if (c.status === 'approved' || c.status === 'queued') entry.approved++
      else if (c.status === 'sent') entry.sent++
      map.set(key, entry)
    }
    return map
  }, [campaignData])

  // Clear selection when client-side filters change
  useEffect(() => {
    setSelectedIds(new Set())
  }, [stageFilter, intentFilter, dispositionTab])

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
    // Disposition filter
    if (dispositionTab === 'active') {
      rows = rows.filter((r) => !dispositionMap.has(rowKey(r)))
    } else if (dispositionTab !== 'all') {
      rows = rows.filter((r) => dispositionMap.get(rowKey(r))?.disposition === dispositionTab)
    }
    return rows
  }, [opportunities, stageFilter, intentFilter, dispositionMap, dispositionTab])

  // -- Stats --
  const stats = useMemo(() => {
    const total = filtered.length
    const highUrgency = filtered.filter((r) => r.urgency >= 8).length
    const dmCount = filtered.filter((r) => r.decision_maker).length
    const dmPct = total > 0 ? Math.round((dmCount / total) * 100) : 0
    const avgUrg = total > 0 ? Math.round((filtered.reduce((s, r) => s + r.urgency, 0) / total) * 10) / 10 : 0
    const withContract = filtered.filter((r) => r.contract_end).length
    return { total, highUrgency, dmPct, avgUrg, withContract }
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
  const handleGenerate = useCallback(async (row: HighIntentCompany) => {
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
      refreshCampaigns()
      refreshCampaignStats()
      setCampaignRefreshKey((k) => k + 1)
    } catch (err) {
      setGenResult(err instanceof Error ? err.message : 'Generation failed')
    } finally {
      setGenerating(null)
    }
  }, [refresh, refreshCampaigns, refreshCampaignStats])

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
      refreshCampaigns()
      refreshCampaignStats()
      setCampaignRefreshKey((k) => k + 1)
    } catch (err) {
      setGenResult(err instanceof Error ? err.message : 'Bulk generation failed')
    } finally {
      setBulkGenerating(false)
      setBulkProgress(null)
    }
  }

  // -- CRM push --
  const [pushing, setPushing] = useState(false)
  async function handlePushToCrm() {
    if (selectedIds.size === 0) return
    setPushing(true)
    setGenResult(null)
    const selected = filtered.filter((r) => selectedIds.has(rowKey(r)))
    try {
      const result = await pushToCrm(
        selected.map((r) => ({
          company: r.company,
          vendor: r.vendor,
          urgency: r.urgency,
          pain: r.pain ?? undefined,
          role_type: r.role_level ?? undefined,
          buying_stage: r.buying_stage ?? undefined,
          contract_end: r.contract_end ?? undefined,
          decision_timeline: r.contract_signal ?? undefined,
          decision_maker: r.decision_maker ?? undefined,
          competitor_context: r.alternatives?.[0]?.name ?? undefined,
          primary_quote: r.quotes?.[0] ?? undefined,
          trust_tier: r.resolution_confidence ?? undefined,
          source: r.source ?? undefined,
          review_id: r.review_id ?? undefined,
          seat_count: r.seat_count ?? undefined,
          industry: r.industry ?? undefined,
          company_size: r.company_size ?? undefined,
          company_domain: r.company_domain ?? undefined,
          company_country: r.company_country ?? undefined,
          revenue_range: r.revenue_range ?? undefined,
          alternatives: r.alternatives?.map((a) => a.name).filter(Boolean) ?? [],
        })),
      )
      const failedKeys = new Set(
        result.failed.map((item) => `${item.company}::${item.vendor}`),
      )
      if (result.failed.length > 0) {
        setSelectedIds(
          new Set(
            selected
              .filter((row) => failedKeys.has(`${row.company}::${row.vendor}`))
              .map(rowKey),
          ),
        )
        setGenResult(
          `Pushed ${result.pushed} opportunit${result.pushed === 1 ? 'y' : 'ies'} to CRM; ${result.failed.length} failed`,
        )
      } else {
        setGenResult(`Pushed ${result.pushed} opportunit${result.pushed === 1 ? 'y' : 'ies'} to CRM`)
        setSelectedIds(new Set())
      }
    } catch (err) {
      setGenResult(err instanceof Error ? err.message : 'CRM push failed')
    } finally {
      setPushing(false)
    }
  }

  const handleDisposition = useCallback(async (
    row: HighIntentCompany,
    disposition: 'snoozed' | 'dismissed' | 'saved',
  ) => {
    const key = rowKey(row)
    setGenResult(null)
    const snoozedUntil = disposition === 'snoozed'
      ? new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString()
      : undefined
    try {
      await setDisposition({
        opportunity_key: key,
        company: row.company,
        vendor: row.vendor,
        review_id: row.review_id ?? undefined,
        disposition,
        snoozed_until: snoozedUntil,
      })
      refreshDispositions()
      setExpandedId(null)
    } catch (err) {
      setGenResult(err instanceof Error ? err.message : 'Disposition update failed')
    }
  }, [refreshDispositions])

  async function handleBulkDisposition(disposition: 'snoozed' | 'dismissed' | 'saved') {
    if (selectedIds.size === 0) return
    setGenResult(null)
    const selected = filtered.filter((r) => selectedIds.has(rowKey(r)))
    const snoozedUntil = disposition === 'snoozed'
      ? new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString()
      : undefined
    try {
      await bulkSetDisposition({
        items: selected.map((r) => ({
          opportunity_key: rowKey(r),
          company: r.company,
          vendor: r.vendor,
          review_id: r.review_id ?? undefined,
        })),
        disposition,
        snoozed_until: snoozedUntil,
      })
      refreshDispositions()
      setSelectedIds(new Set())
    } catch (err) {
      setGenResult(err instanceof Error ? err.message : 'Bulk disposition update failed')
    }
  }

  const handleRemoveDisposition = useCallback(async (key: string) => {
    setGenResult(null)
    try {
      await removeDispositions({ opportunity_keys: [key] })
      refreshDispositions()
    } catch (err) {
      setGenResult(err instanceof Error ? err.message : 'Failed to restore opportunity')
    }
  }, [refreshDispositions])

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
      render: (r) => {
        const cs = campaignMap.get(`${(r.company ?? '').toLowerCase()}::${(r.vendor ?? '').toLowerCase()}`)
        return (
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="text-white font-medium">{r.company || '--'}</span>
            {cs && (
              cs.sent > 0
                ? <Link to={`/campaign-review?company=${encodeURIComponent(r.company)}`} onClick={(e) => e.stopPropagation()} className="shrink-0"><span className="px-1 py-0.5 rounded text-[9px] font-medium bg-cyan-500/15 text-cyan-400 hover:ring-1 hover:ring-cyan-400/50">{cs.sent} sent</span></Link>
                : cs.approved > 0
                  ? <Link to={`/campaign-review?company=${encodeURIComponent(r.company)}`} onClick={(e) => e.stopPropagation()} className="shrink-0"><span className="px-1 py-0.5 rounded text-[9px] font-medium bg-green-500/15 text-green-300 hover:ring-1 hover:ring-green-400/50">{cs.approved} approved</span></Link>
                  : cs.drafts > 0
                    ? <Link to={`/campaign-review?company=${encodeURIComponent(r.company)}`} onClick={(e) => e.stopPropagation()} className="shrink-0"><span className="px-1 py-0.5 rounded text-[9px] font-medium bg-amber-500/15 text-amber-400 hover:ring-1 hover:ring-amber-400/50">{cs.drafts} draft{cs.drafts > 1 ? 's' : ''}</span></Link>
                    : null
            )}
          </div>
          {r.company_domain && (
            <span className="text-xs text-slate-500">{r.company_domain}</span>
          )}
        </div>
        )
      },
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
      key: 'role',
      header: 'Role',
      render: (r) => {
        const title = r.reviewer_title ?? r.role_level
        return (
          <div className="flex items-center gap-1.5 min-w-0">
            {title
              ? <span className="text-xs text-slate-300 truncate max-w-[100px]" title={title}>{title}</span>
              : <span className="text-xs text-slate-500">--</span>}
            {r.decision_maker && <span className="shrink-0 px-1 py-0.5 rounded text-[9px] font-bold bg-cyan-500/20 text-cyan-400">DM</span>}
          </div>
        )
      },
    },
    {
      key: 'pain',
      header: 'Pain / Quote',
      render: (r) => {
        const quote = r.quotes?.[0]
        return (
          <div className="min-w-0 max-w-[200px]">
            {r.pain && <span className="text-xs text-slate-400 truncate block" title={r.pain}>{r.pain}</span>}
            {quote
              ? <span className="text-[10px] text-slate-500 italic truncate block" title={quote}>"{quote}"</span>
              : !r.pain && <span className="text-xs text-slate-500">--</span>}
          </div>
        )
      },
    },
    {
      key: 'contract',
      header: 'Contract',
      render: (r) => {
        if (!r.contract_end) return <span className="text-slate-500 text-xs">--</span>
        return <span className="text-xs text-amber-400">{r.contract_end}</span>
      },
    },
    {
      key: 'alternative',
      header: 'Top Alt.',
      render: (r) => {
        const alt = r.alternatives?.[0]
        if (!alt?.name) return <span className="text-slate-500 text-xs">--</span>
        return <span className="text-xs text-cyan-400 truncate block max-w-[100px]" title={alt.name}>{alt.name}</span>
      },
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
        const key = rowKey(r)
        const disp = dispositionMap.get(key)
        if (disp) {
          return (
            <button
              onClick={(e) => { e.stopPropagation(); handleRemoveDisposition(key) }}
              className="p-1 text-green-400 hover:text-green-300 transition-colors"
              title="Restore to Active"
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
          )
        }
        return (
          <div className="flex items-center gap-0.5">
            {canAccessCampaigns && (
              <button
                onClick={(e) => { e.stopPropagation(); handleGenerate(r) }}
                disabled={generating === key}
                className="p-1 text-cyan-400 hover:text-cyan-300 transition-colors disabled:opacity-50"
                title="Generate Campaign"
              >
                <Zap className={clsx('h-3.5 w-3.5', generating === key && 'animate-pulse')} />
              </button>
            )}
            <button
              onClick={(e) => { e.stopPropagation(); handleDisposition(r, 'saved') }}
              className="p-1 text-amber-400 hover:text-amber-300 transition-colors"
              title="Save"
            >
              <Bookmark className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); handleDisposition(r, 'snoozed') }}
              className="p-1 text-indigo-400 hover:text-indigo-300 transition-colors"
              title="Snooze 7 days"
            >
              <Clock className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); handleDisposition(r, 'dismissed') }}
              className="p-1 text-slate-400 hover:text-slate-300 transition-colors"
              title="Dismiss"
            >
              <EyeOff className="h-3.5 w-3.5" />
            </button>
          </div>
        )
      },
    },
  ], [
    selectedIds,
    toggleSelect,
    canAccessCampaigns,
    generating,
    campaignMap,
    dispositionMap,
    handleDisposition,
    handleGenerate,
    handleRemoveDisposition,
  ])

  // -- Expanded row --
  const expandedRow = useMemo(
    () => filtered.find((r) => rowKey(r) === expandedId) ?? null,
    [filtered, expandedId],
  )
  const currentPagePath = useMemo(
    () => opportunitiesPath(debouncedVendor, backTarget),
    [debouncedVendor, backTarget],
  )

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      {backTarget && (
        <Link
          to={backTarget}
          className="inline-flex items-center gap-1 text-sm text-slate-400 hover:text-white transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          {backButtonLabel(backTarget)}
        </Link>
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold text-white">Opportunity Workbench</h1>
        </div>
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
        <StatCard label="Contract Known" value={stats.withContract} icon={<Target className="h-4 w-4 text-red-400" />} skeleton={loading} />
      </div>

      {/* Campaign stats */}
      {canAccessCampaigns && campaignStats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <StatCard label="Total Campaigns" value={campaignStats.total} icon={<Mail className="h-4 w-4" />} skeleton={campaignStatsLoading} />
          <StatCard label="Drafts Pending" value={campaignStats.by_status['draft'] ?? 0} icon={<Clock className="h-4 w-4" />} skeleton={campaignStatsLoading} />
          <StatCard label="Approved" value={(campaignStats.by_status['approved'] ?? 0) + (campaignStats.by_status['queued'] ?? 0)} icon={<CheckCircle2 className="h-4 w-4" />} skeleton={campaignStatsLoading} />
          <StatCard label="Sent" value={campaignStats.by_status['sent'] ?? 0} icon={<Zap className="h-4 w-4" />} skeleton={campaignStatsLoading} />
        </div>
      )}

      {/* Campaign quality trends */}
      {canAccessCampaigns && (
        <CollapsibleSection title="Campaign Quality Trends" defaultOpen={false}>
          <CampaignQualityTrends data={qualityTrends} loading={qualityTrendsLoading} />
        </CollapsibleSection>
      )}

      {/* Signal effectiveness */}
      {canAccessCampaigns && <SignalEffectivenessPanel />}

      {/* Disposition tabs */}
      <div className="flex items-center gap-1 border-b border-slate-700/40 pb-1">
        {DISPOSITION_TABS.map((tab) => {
          const count = tab === 'active'
            ? opportunities.filter((r) => !dispositionMap.has(rowKey(r))).length
            : tab === 'all'
              ? opportunities.length
              : opportunities.filter((r) => dispositionMap.get(rowKey(r))?.disposition === tab).length
          return (
            <button
              key={tab}
              onClick={() => { setDispositionTab(tab); setSelectedIds(new Set()) }}
              className={clsx(
                'px-3 py-1.5 text-sm rounded-t-lg transition-colors',
                dispositionTab === tab
                  ? 'text-white bg-slate-800/50 border-b-2 border-cyan-500'
                  : 'text-slate-400 hover:text-white',
              )}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
              <span className="ml-1.5 text-xs text-slate-500">({count})</span>
            </button>
          )
        })}
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
        {selectedIds.size > 0 && (
          <div className="ml-auto flex items-center gap-2">
            <span className="text-xs text-slate-400">{selectedIds.size} selected</span>
            {canAccessCampaigns && (
              <button
                onClick={handleBulkGenerate}
                disabled={bulkGenerating}
                className="px-3 py-1.5 text-xs font-medium bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg disabled:opacity-50"
              >
                {bulkGenerating ? 'Generating...' : 'Generate Campaigns'}
              </button>
            )}
            <button
              onClick={handlePushToCrm}
              disabled={pushing || bulkGenerating}
              className="px-3 py-1.5 text-xs font-medium bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg disabled:opacity-50"
            >
              {pushing ? 'Pushing...' : 'Push to CRM'}
            </button>
            {dispositionTab === 'active' || dispositionTab === 'all' ? (
              <>
                <button
                  onClick={() => handleBulkDisposition('saved')}
                  className="px-3 py-1.5 text-xs font-medium bg-amber-600 hover:bg-amber-500 text-white rounded-lg"
                >
                  Save
                </button>
                <button
                  onClick={() => handleBulkDisposition('snoozed')}
                  className="px-3 py-1.5 text-xs font-medium bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg"
                >
                  Snooze 7d
                </button>
                <button
                  onClick={() => handleBulkDisposition('dismissed')}
                  className="px-3 py-1.5 text-xs font-medium bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg"
                >
                  Dismiss
                </button>
              </>
            ) : (
              <button
                onClick={async () => {
                  const keys = Array.from(selectedIds)
                  try {
                    await removeDispositions({ opportunity_keys: keys })
                    refreshDispositions()
                    setSelectedIds(new Set())
                  } catch (err) {
                    setGenResult(err instanceof Error ? err.message : 'Bulk restore failed')
                  }
                }}
                className="px-3 py-1.5 text-xs font-medium bg-green-600 hover:bg-green-500 text-white rounded-lg"
              >
                Restore to Active
              </button>
            )}
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
          defaultSortKey="urgency"
          defaultSortDir="desc"
        />
      </div>

      {/* Evidence panel */}
      {expandedRow && (
        <EvidencePanel
          row={expandedRow}
          currentPagePath={currentPagePath}
          disposition={dispositionMap.get(rowKey(expandedRow)) ?? null}
          canAccessCampaigns={canAccessCampaigns}
          onClose={() => setExpandedId(null)}
          onGenerate={canAccessCampaigns ? handleGenerate : undefined}
          onDisposition={handleDisposition}
          onRestore={handleRemoveDisposition}
          generating={generating === rowKey(expandedRow)}
          campaignRefreshKey={campaignRefreshKey}
          onCampaignAction={() => { refreshCampaigns(); refreshCampaignStats() }}
        />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Campaign Queue (inline approval flow)
// ---------------------------------------------------------------------------

const CAMPAIGN_STATUS_COLORS: Record<string, string> = {
  draft: 'bg-amber-500/20 text-amber-400',
  approved: 'bg-green-500/20 text-green-400',
  queued: 'bg-cyan-500/20 text-cyan-400',
  sent: 'bg-cyan-500/20 text-cyan-400',
  cancelled: 'bg-red-500/20 text-red-400',
  expired: 'bg-slate-500/20 text-slate-400',
}

function CampaignQueue({ company, vendor, refreshKey, onAction }: { company: string; vendor: string; refreshKey?: number; onAction?: () => void }) {
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [result, setResult] = useState<string | null>(null)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editFields, setEditFields] = useState({ subject: '', body: '', cta: '' })
  const [editSaving, setEditSaving] = useState(false)

  const { data, loading, refresh } = useApiData(
    () => fetchCampaigns({ company, vendor, limit: 20 }),
    [company, vendor, refreshKey],
  )

  // Filter to exact company match (backend uses ILIKE substring)
  const campaigns = (data?.campaigns ?? []).filter(
    (c) => c.company_name.toLowerCase() === company.toLowerCase()
      && c.vendor_name.toLowerCase() === vendor.toLowerCase(),
  )
  const drafts = campaigns.filter((c) => c.status === 'draft')
  const others = campaigns.filter((c) => c.status !== 'draft')

  async function handleApprove(c: Campaign) {
    setActionLoading(c.id)
    setResult(null)
    try {
      await approveCampaign(c.id)
      setResult(`Approved: ${c.subject ?? c.channel}`)
      refresh()
      onAction?.()
    } catch (err) {
      setResult(err instanceof Error ? err.message : 'Approve failed')
    } finally {
      setActionLoading(null)
    }
  }

  async function handleReject(c: Campaign) {
    setActionLoading(c.id)
    setResult(null)
    try {
      await updateCampaign(c.id, { status: 'cancelled' })
      setResult(`Rejected: ${c.subject ?? c.channel}`)
      refresh()
      onAction?.()
    } catch (err) {
      setResult(err instanceof Error ? err.message : 'Reject failed')
    } finally {
      setActionLoading(null)
    }
  }

  function startEditing(c: Campaign) {
    setEditingId(c.id)
    setEditFields({ subject: c.subject ?? '', body: c.body ?? '', cta: c.cta ?? '' })
  }

  async function saveEditing() {
    if (!editingId) return
    setEditSaving(true)
    try {
      await updateCampaign(editingId, {
        subject: editFields.subject,
        body: editFields.body,
        cta: editFields.cta || undefined,
      })
      setResult('Campaign updated')
      setEditingId(null)
      refresh()
      onAction?.()
    } catch (err) {
      setResult(err instanceof Error ? err.message : 'Save failed')
    } finally {
      setEditSaving(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-xs text-slate-500 py-3">
        <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading campaigns...
      </div>
    )
  }

  if (campaigns.length === 0) {
    return <p className="text-xs text-slate-500 py-2">No campaigns generated yet.</p>
  }

  return (
    <div className="space-y-3">
      {result && (
        <div className="flex items-center justify-between rounded border border-slate-700/40 bg-slate-800/40 px-3 py-1.5 text-xs text-slate-300">
          <span>{result}</span>
          <button onClick={() => setResult(null)} className="text-slate-500 hover:text-white ml-2"><X className="h-3 w-3" /></button>
        </div>
      )}

      {drafts.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-amber-400">{drafts.length} pending review</p>
          {drafts.map((c) => (
            <div key={c.id} className="bg-slate-800/50 border border-slate-700/30 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-2 flex-wrap">
                <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-700/50 text-slate-300">{c.channel}</span>
                <span className={clsx('px-2 py-0.5 rounded-full text-[10px] font-medium', CAMPAIGN_STATUS_COLORS[c.status])}>{c.status}</span>
                {c.opportunity_score != null && (
                  <span className="text-[10px] text-slate-500">score {c.opportunity_score}</span>
                )}
                {c.quality_status && (
                  <span className={clsx('px-1.5 py-0.5 rounded text-[10px] font-medium', c.quality_status === 'pass' ? 'bg-green-500/15 text-green-300' : 'bg-red-500/15 text-red-300')}>
                    quality: {c.quality_status}
                  </span>
                )}
              </div>
              {editingId === c.id ? (
                <div className="space-y-2 mb-2">
                  <input
                    type="text"
                    value={editFields.subject}
                    onChange={(e) => setEditFields((f) => ({ ...f, subject: e.target.value }))}
                    placeholder="Subject"
                    className="w-full px-3 py-1.5 bg-slate-900/50 border border-slate-700/50 rounded text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
                  />
                  <textarea
                    value={editFields.body}
                    onChange={(e) => setEditFields((f) => ({ ...f, body: e.target.value }))}
                    rows={6}
                    placeholder="Body"
                    className="w-full px-3 py-1.5 bg-slate-900/50 border border-slate-700/50 rounded text-xs text-slate-200 font-mono placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
                  />
                  <input
                    type="text"
                    value={editFields.cta}
                    onChange={(e) => setEditFields((f) => ({ ...f, cta: e.target.value }))}
                    placeholder="CTA"
                    className="w-full px-3 py-1.5 bg-slate-900/50 border border-slate-700/50 rounded text-sm text-cyan-400 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
                  />
                  <div className="flex items-center gap-2">
                    <button
                      onClick={saveEditing}
                      disabled={editSaving}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-cyan-600 hover:bg-cyan-500 text-white transition-colors disabled:opacity-50"
                    >
                      {editSaving ? <Loader2 className="h-3 w-3 animate-spin" /> : <CheckCircle2 className="h-3 w-3" />}
                      Save
                    </button>
                    <button
                      onClick={() => setEditingId(null)}
                      className="px-3 py-1.5 rounded-lg text-xs text-slate-400 hover:text-white"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  {c.subject && <p className="text-sm font-medium text-white mb-1">{c.subject}</p>}
                  <CampaignReasoningSummary item={c} />
                  {c.body && (
                    <div
                      className="text-xs text-slate-300 line-clamp-3 prose prose-invert prose-xs max-w-none mb-2"
                      dangerouslySetInnerHTML={{ __html: c.body }}
                    />
                  )}
                  {c.cta && <p className="text-xs text-cyan-400 mb-2">{c.cta}</p>}
                </>
              )}
              {c.quality_status === 'fail' && c.failure_explanation && (
                <div className="mb-2">
                  <CampaignFailureExplanation explanation={c.failure_explanation} />
                </div>
              )}
              {c.blocker_count != null && c.blocker_count > 0 && (
                <p className="text-[10px] text-red-400 mb-2">{c.blocker_count} blocker(s), {c.warning_count ?? 0} warning(s)</p>
              )}
              {editingId !== c.id && (
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleApprove(c)}
                    disabled={actionLoading === c.id}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-green-600 hover:bg-green-500 text-white transition-colors disabled:opacity-50"
                  >
                    <CheckCircle2 className="h-3 w-3" />
                    Approve
                  </button>
                  <button
                    onClick={() => handleReject(c)}
                    disabled={actionLoading === c.id}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-red-600/20 hover:bg-red-600/30 text-red-400 transition-colors disabled:opacity-50"
                  >
                    <XCircle className="h-3 w-3" />
                    Reject
                  </button>
                  <button
                    onClick={() => startEditing(c)}
                    disabled={actionLoading === c.id}
                    className="inline-flex items-center gap-1.5 px-2 py-1.5 rounded-lg text-xs text-slate-400 hover:text-white hover:bg-slate-700/50 transition-colors disabled:opacity-50"
                    title="Edit content"
                  >
                    <Pencil className="h-3 w-3" />
                  </button>
                  {actionLoading === c.id && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {others.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs text-slate-500">{others.length} processed</p>
          {others.slice(0, 5).map((c) => (
            <div key={c.id} className="flex items-center gap-2 text-xs py-1 border-b border-slate-800/50 last:border-0">
              <span className={clsx('px-1.5 py-0.5 rounded-full text-[10px] font-medium', CAMPAIGN_STATUS_COLORS[c.status])}>{c.status}</span>
              <span className="text-slate-300 truncate">{c.subject ?? c.channel}</span>
              {c.status === 'sent' && (
                <Link
                  to={`/campaign-review?company=${encodeURIComponent(company)}`}
                  className="text-[10px] text-cyan-400 hover:text-cyan-300 shrink-0 ml-2"
                >
                  outcome
                </Link>
              )}
              {c.created_at && <span className="text-slate-500 shrink-0 ml-auto">{new Date(c.created_at).toLocaleDateString()}</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Evidence Panel
// ---------------------------------------------------------------------------

function EvidencePanel({
  row,
  currentPagePath,
  disposition,
  canAccessCampaigns,
  onClose,
  onGenerate,
  onDisposition,
  onRestore,
  generating,
  campaignRefreshKey,
  onCampaignAction,
}: {
  row: HighIntentCompany
  currentPagePath: string
  disposition: OpportunityDisposition | null
  canAccessCampaigns: boolean
  onClose: () => void
  onGenerate?: (row: HighIntentCompany) => void
  onDisposition?: (row: HighIntentCompany, disposition: 'snoozed' | 'dismissed' | 'saved') => void
  onRestore?: (opportunityKey: string) => void
  generating: boolean
  campaignRefreshKey?: number
  onCampaignAction?: () => void
}) {
  const signals = row.intent_signals
  const alternatives = Array.isArray(row.alternatives)
    ? row.alternatives.filter((a) => a?.name?.trim())
    : []
  const quotes = Array.isArray(row.quotes)
    ? row.quotes.filter((q) => typeof q === 'string' && q.trim())
    : []
  const vendorPath = vendorDetailPath(row.vendor)
  const vendorEvidencePath = evidencePath(row.vendor, currentPagePath)
  const vendorReportsLibraryPath = reportsPath(row.vendor, currentPagePath)
  const reviewPath = row.review_id ? reviewDetailPath(row.review_id, currentPagePath) : null

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
          <div className="flex flex-wrap gap-3 pt-1">
            <Link
              to={vendorPath}
              className="inline-flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300"
            >
              View vendor <ExternalLink className="h-3 w-3" />
            </Link>
            <Link
              to={vendorEvidencePath}
              className="inline-flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300"
            >
              Validate evidence <ExternalLink className="h-3 w-3" />
            </Link>
            <Link
              to={vendorReportsLibraryPath}
              className="inline-flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300"
            >
              View reports <ExternalLink className="h-3 w-3" />
            </Link>
            {reviewPath && (
              <Link
                to={reviewPath}
                className="inline-flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300"
              >
                View full review <ExternalLink className="h-3 w-3" />
              </Link>
            )}
          </div>
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

      {/* Campaign Queue */}
      {canAccessCampaigns && (
        <div className="border-t border-slate-700/40 pt-4">
          <div className="flex items-center gap-2 mb-3">
            <Mail className="h-4 w-4 text-cyan-400" />
            <h4 className="text-sm font-medium text-white">Generated Campaigns</h4>
          </div>
          <CampaignQueue company={row.company} vendor={row.vendor} refreshKey={campaignRefreshKey} onAction={onCampaignAction} />
        </div>
      )}

      {/* Deal timeline */}
      {canAccessCampaigns && (
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Activity className="h-4 w-4 text-cyan-400" />
            <h4 className="text-sm font-medium text-white">Deal Timeline</h4>
          </div>
          <CompanyTimeline company={row.company} vendor={row.vendor} />
        </div>
      )}

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
        {onDisposition && !disposition && (
          <>
            <button
              onClick={() => onDisposition(row, 'saved')}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm bg-amber-600 hover:bg-amber-500 text-white transition-colors"
            >
              <Bookmark className="h-4 w-4" />
              Save
            </button>
            <button
              onClick={() => onDisposition(row, 'snoozed')}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm bg-indigo-600 hover:bg-indigo-500 text-white transition-colors"
            >
              <Clock className="h-4 w-4" />
              Snooze 7d
            </button>
            <button
              onClick={() => onDisposition(row, 'dismissed')}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm bg-slate-700 hover:bg-slate-600 text-slate-300 transition-colors"
            >
              <EyeOff className="h-4 w-4" />
              Dismiss
            </button>
          </>
        )}
        {onRestore && disposition && (
          <button
            onClick={() => onRestore(rowKey(row))}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm bg-green-600 hover:bg-green-500 text-white transition-colors"
          >
            <RefreshCw className="h-4 w-4" />
            Restore
          </button>
        )}
        {!reviewPath && (
          <Link
            to="/reviews"
            className="inline-flex items-center gap-1 text-sm text-slate-400 hover:text-white transition-colors"
          >
            Browse reviews <ChevronRight className="h-3.5 w-3.5" />
          </Link>
        )}
      </div>
    </div>
  )
}
