import { useState, useEffect, useMemo, useRef } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import {
  Swords,
  RefreshCw,
  Send,
  TrendingUp,
  Users,
  Target,
  ExternalLink,
  Download,
} from 'lucide-react'
import { clsx } from 'clsx'
import StatCard from '../components/StatCard'
import DataTable, { type Column } from '../components/DataTable'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import {
  fetchVendorTargets,
  fetchHighIntent,
  fetchChallengerClaims,
  generateCampaigns,
  downloadCsv,
  type ChallengerClaimRow,
  type ChallengerClaimsResponse,
  type VendorClaim,
} from '../api/client'
import {
  ProductClaimGate,
  ProductClaimStatusBadge,
  productClaimGateTitle,
  isProductClaimAllowed,
} from '../components/ProductClaimGate'

interface ChallengerSummary {
  name: string
  totalLeads: number
  activePurchase: number
  evaluation: number
  renewal: number
  topIncumbents: string[]
  topPainCategories: string[]
  avgUrgency: number
  claimsResponse: ChallengerClaimsResponse | null | undefined
  directDisplacementRows: ChallengerClaimRow[]
  directDisplacementClaim: VendorClaim | undefined
  claimsValidationUnavailable: boolean
}

function challengersPath(search: string) {
  const next = new URLSearchParams()
  if (search.trim()) next.set('search', search.trim())
  const qs = next.toString()
  return qs ? `/challengers?${qs}` : '/challengers'
}

function vendorDetailPath(vendorName: string, backTo: string) {
  const next = new URLSearchParams()
  if (backTo !== '/challengers') next.set('back_to', backTo)
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

function challengerLookupKey(name: string) {
  return name.trim().toLowerCase()
}

function directDisplacementRows(response: ChallengerClaimsResponse | null | undefined): ChallengerClaimRow[] {
  if (!response) return []
  return response.rows.filter((row) => row.claim.claim_type === 'direct_displacement')
}

function primaryDirectDisplacementClaim(rows: ChallengerClaimRow[]): VendorClaim | undefined {
  return (
    rows.find((row) => row.claim.report_allowed === true)?.claim ??
    rows.find((row) => row.claim.render_allowed === true)?.claim ??
    rows[0]?.claim
  )
}

export default function Challengers() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const searchParamsSignature = searchParams.toString()
  const requestedSearch = searchParams.get('search')?.trim() ?? ''
  const [searchInput, setSearchInput] = useState(requestedSearch)
  const [debouncedSearch, setDebouncedSearch] = useState(requestedSearch)
  const [generatingFor, setGeneratingFor] = useState<string | null>(null)
  const [actionResult, setActionResult] = useState<string | null>(null)
  const [lastGenVendor, setLastGenVendor] = useState<string | null>(null)
  const suppressRouteSyncRef = useRef(false)

  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(searchInput), 300)
    return () => clearTimeout(t)
  }, [searchInput])

  useEffect(() => {
    suppressRouteSyncRef.current = true
    setSearchInput(requestedSearch)
    setDebouncedSearch(requestedSearch)
  }, [requestedSearch])

  const routeStateSettled =
    searchInput === requestedSearch &&
    debouncedSearch === requestedSearch

  useEffect(() => {
    if (!routeStateSettled) return
    const next = new URLSearchParams(searchParams)
    if (debouncedSearch.trim()) next.set('search', debouncedSearch.trim())
    else next.delete('search')
    if (next.toString() === searchParamsSignature) return
    suppressRouteSyncRef.current = true
    setSearchParams(next, { replace: true })
  }, [debouncedSearch, routeStateSettled, searchParams, searchParamsSignature, setSearchParams])

  useEffect(() => {
    if (suppressRouteSyncRef.current) {
      if (!routeStateSettled) return
      suppressRouteSyncRef.current = false
      return
    }
    const next = new URLSearchParams(searchParams)
    if (debouncedSearch.trim()) next.set('search', debouncedSearch.trim())
    else next.delete('search')
    if (next.toString() === searchParamsSignature) return
    setSearchParams(next, { replace: true })
  }, [debouncedSearch, routeStateSettled, searchParams, searchParamsSignature, setSearchParams])

  const { data, loading, error, refresh, refreshing } = useApiData(
    async () => {
      const [targetsRes, hiRes] = await Promise.all([
        fetchVendorTargets({ target_mode: 'challenger_intel', limit: 100 }),
        fetchHighIntent({ min_urgency: 3, limit: 100 }),
      ])
      const claimEntries = await Promise.all(
        targetsRes.targets.map(async (target) => {
          const name = target.company_name
          try {
            return [challengerLookupKey(name), await fetchChallengerClaims(name)] as const
          } catch {
            return [challengerLookupKey(name), null] as const
          }
        }),
      )
      return {
        targets: targetsRes.targets,
        companies: hiRes.companies,
        challengerClaimsByName: Object.fromEntries(claimEntries) as Record<string, ChallengerClaimsResponse | null>,
      }
    },
    [],
  )

  const targets = data?.targets ?? []
  const companies = data?.companies ?? []
  const challengerClaimsByName = data?.challengerClaimsByName ?? {}

  // Build challenger summaries by aggregating high-intent signals where competitor matches a target
  const challengerSummaries: ChallengerSummary[] = targets.map(target => {
    const name = target.company_name.toLowerCase()
    const claimsResponse = challengerClaimsByName[challengerLookupKey(target.company_name)]
    const displacementRows = directDisplacementRows(claimsResponse)
    const primaryClaim = primaryDirectDisplacementClaim(displacementRows)
    const validatedIncumbents = new Set(
      displacementRows
        .filter((row) => row.claim.render_allowed === true)
        .map((row) => row.incumbent.toLowerCase()),
    )

    // Only explicit challenger mentions are counted. The old
    // competitors_tracked fallback counted every high-intent row for a tracked
    // incumbent even when the challenger was never mentioned, inflating
    // "challenger lead" metrics beyond what the direct-displacement claim can
    // validate.
    const relevantCompanies = companies.filter(c => {
      const mentionsChallenger = (c.alternatives ?? []).some(
        alt => alt.name?.toLowerCase() === name,
      )
      if (!mentionsChallenger) return false
      const vendor = c.vendor?.toLowerCase()
      return Boolean(vendor && validatedIncumbents.has(vendor))
    })

    const stages = relevantCompanies.reduce(
      (acc, c) => {
        const s = c.buying_stage ?? 'unknown'
        if (s === 'active_purchase') acc.active++
        else if (s === 'evaluation') acc.eval++
        else if (s === 'renewal_decision') acc.renewal++
        return acc
      },
      { active: 0, eval: 0, renewal: 0 },
    )

    // Top incumbents must be backed by validated direct-displacement rows.
    // High-intent rows can provide stage/urgency context, but they cannot
    // create a "Losing From" winner call without a ProductClaim.
    const incumbentCounts: Record<string, number> = {}
    for (const c of relevantCompanies) {
      const v = c.vendor
      if (v && validatedIncumbents.has(v.toLowerCase())) {
        incumbentCounts[v] = (incumbentCounts[v] ?? 0) + 1
      }
    }
    const topIncumbents = Object.entries(incumbentCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([incumbentName]) => incumbentName)

    // Top pain categories driving the switch
    const painCounts: Record<string, number> = {}
    for (const c of relevantCompanies) {
      if (c.pain) painCounts[c.pain] = (painCounts[c.pain] ?? 0) + 1
    }
    const topPainCategories = Object.entries(painCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([cat]) => cat)

    const avgUrg = relevantCompanies.length > 0
      ? relevantCompanies.reduce((s, c) => s + c.urgency, 0) / relevantCompanies.length
      : 0

    return {
      name: target.company_name,
      totalLeads: relevantCompanies.length,
      activePurchase: stages.active,
      evaluation: stages.eval,
      renewal: stages.renewal,
      topIncumbents,
      topPainCategories,
      avgUrgency: Math.round(avgUrg * 10) / 10,
      claimsResponse,
      directDisplacementRows: displacementRows,
      directDisplacementClaim: primaryClaim,
      claimsValidationUnavailable: claimsResponse === null,
    }
  })

  // Filter by search
  const filtered = debouncedSearch
    ? challengerSummaries.filter(s => s.name.toLowerCase().includes(debouncedSearch.toLowerCase()))
    : challengerSummaries
  const currentListPath = useMemo(() => challengersPath(debouncedSearch), [debouncedSearch])

  const totalLeads = filtered.reduce((s, c) => s + c.totalLeads, 0)
  const totalActive = filtered.reduce((s, c) => s + c.activePurchase, 0)
  const totalEval = filtered.reduce((s, c) => s + c.evaluation, 0)
  const validationUnavailableCount = filtered.filter((c) => c.claimsValidationUnavailable).length

  async function handleGenerate(summary: ChallengerSummary) {
    if (!isProductClaimAllowed(
      summary.directDisplacementClaim,
      'report',
      summary.claimsValidationUnavailable,
    )) {
      setActionResult(`Campaign generation blocked for ${summary.name}: ${productClaimGateTitle(
        summary.directDisplacementClaim,
        summary.claimsValidationUnavailable,
      )}`)
      return
    }
    setGeneratingFor(summary.name)
    setActionResult(null)
    try {
      const result = await generateCampaigns({
        vendor_name: summary.name,
        target_mode: 'challenger_intel',
        min_score: 50,
        limit: 5,
      })
      setActionResult(`Generated ${result.generated ?? 0} campaign(s) for ${summary.name}`)
      setLastGenVendor(summary.name)
      refresh()
    } catch (err) {
      setActionResult(err instanceof Error ? err.message : 'Generation failed')
    } finally {
      setGeneratingFor(null)
    }
  }

  const columns: Column<ChallengerSummary>[] = [
    {
      key: 'name',
      header: 'Challenger',
      render: (r) => (
        <div className="flex flex-col gap-1">
          <span className="text-white font-medium">{r.name}</span>
          <ProductClaimStatusBadge
            claim={r.directDisplacementClaim}
            validationUnavailable={r.claimsValidationUnavailable}
          />
        </div>
      ),
    },
    {
      key: 'leads',
      header: 'Total Leads',
      render: (r) => (
        <ProductClaimGate
          claim={r.directDisplacementClaim}
          mode="render"
          validationUnavailable={r.claimsValidationUnavailable}
        >
          <span className={clsx(
            'text-sm font-medium',
            r.totalLeads > 0 ? 'text-cyan-400' : 'text-slate-500',
          )}>
            {r.totalLeads}
          </span>
        </ProductClaimGate>
      ),
      sortable: true,
      sortValue: (r) => r.totalLeads,
    },
    {
      key: 'active',
      header: 'Active Purchase',
      render: (r) => (
        <ProductClaimGate
          claim={r.directDisplacementClaim}
          mode="render"
          validationUnavailable={r.claimsValidationUnavailable}
        >
          <span className={clsx('text-xs font-medium', r.activePurchase > 0 ? 'text-red-400' : 'text-slate-500')}>
            {r.activePurchase}
          </span>
        </ProductClaimGate>
      ),
      sortable: true,
      sortValue: (r) => r.activePurchase,
    },
    {
      key: 'eval',
      header: 'Evaluation',
      render: (r) => (
        <ProductClaimGate
          claim={r.directDisplacementClaim}
          mode="render"
          validationUnavailable={r.claimsValidationUnavailable}
        >
          <span className={clsx('text-xs font-medium', r.evaluation > 0 ? 'text-cyan-400' : 'text-slate-500')}>
            {r.evaluation}
          </span>
        </ProductClaimGate>
      ),
    },
    {
      key: 'renewal',
      header: 'Renewal',
      render: (r) => (
        <ProductClaimGate
          claim={r.directDisplacementClaim}
          mode="render"
          validationUnavailable={r.claimsValidationUnavailable}
        >
          <span className={clsx('text-xs font-medium', r.renewal > 0 ? 'text-amber-400' : 'text-slate-500')}>
            {r.renewal}
          </span>
        </ProductClaimGate>
      ),
    },
    {
      key: 'urgency',
      header: 'Avg Urgency',
      render: (r) => (
        <ProductClaimGate
          claim={r.directDisplacementClaim}
          mode="render"
          validationUnavailable={r.claimsValidationUnavailable}
        >
          <span className={clsx(
            'text-xs font-medium',
            r.avgUrgency >= 7 ? 'text-red-400' : r.avgUrgency >= 5 ? 'text-amber-400' : 'text-slate-400',
          )}>
            {r.avgUrgency}
          </span>
        </ProductClaimGate>
      ),
      sortable: true,
      sortValue: (r) => r.avgUrgency,
    },
    {
      key: 'incumbents',
      header: 'Losing From',
      render: (r) => {
        const claimRows = r.directDisplacementRows.slice(0, 3)
        return (
          <span className="flex flex-wrap items-center gap-1.5 text-xs">
            {r.claimsValidationUnavailable ? (
              <ProductClaimGate
                claim={r.directDisplacementClaim}
                mode="render"
                validationUnavailable
              >
                <span />
              </ProductClaimGate>
            ) : claimRows.length > 0 ? (
              claimRows.map((row) => (
                <ProductClaimGate
                  key={`${row.incumbent}:${row.claim.claim_id}`}
                  claim={row.claim}
                  mode="render"
                  fallback="Insufficient"
                >
                  <span className="text-slate-400">{row.incumbent}</span>
                </ProductClaimGate>
              ))
            ) : (
              <ProductClaimGate
                claim={r.directDisplacementClaim}
                mode="render"
                fallback="Insufficient"
              >
                <span />
              </ProductClaimGate>
            )}
          </span>
        )
      },
    },
    {
      key: 'actions',
      header: 'Actions',
      render: (r) => (
        <div className="flex items-center gap-3 text-xs">
          <Link
            to={evidencePath(r.name, currentListPath)}
            onClick={(event) => event.stopPropagation()}
            className="text-violet-300 hover:text-violet-200 transition-colors"
          >
            Evidence
          </Link>
          <Link
            to={reportsPath(r.name, currentListPath)}
            onClick={(event) => event.stopPropagation()}
            className="text-fuchsia-300 hover:text-fuchsia-200 transition-colors"
          >
            Reports
          </Link>
          <Link
            to={opportunitiesPath(r.name, currentListPath)}
            onClick={(event) => event.stopPropagation()}
            className="text-emerald-300 hover:text-emerald-200 transition-colors"
          >
            Opportunities
          </Link>
          <ProductClaimGate
            claim={r.directDisplacementClaim}
            mode="report"
            validationUnavailable={r.claimsValidationUnavailable}
            fallback={
              <button
                disabled
                className="p-1 text-slate-600 cursor-not-allowed"
                title={productClaimGateTitle(r.directDisplacementClaim, r.claimsValidationUnavailable)}
              >
                <Send className="h-3.5 w-3.5" />
              </button>
            }
          >
            <button
              onClick={(e) => { e.stopPropagation(); handleGenerate(r) }}
              disabled={generatingFor === r.name}
              className="p-1 text-slate-400 hover:text-green-400 transition-colors disabled:opacity-50"
              title="Generate Campaign"
            >
              <Send className={clsx('h-3.5 w-3.5', generatingFor === r.name && 'animate-pulse')} />
            </button>
          </ProductClaimGate>
        </div>
      ),
    },
  ]

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Challenger Intelligence</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={() =>
              downloadCsv('/export/high-intent', {
                min_urgency: 3,
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
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
            Refresh
          </button>
        </div>
      </div>

      {/* Action result */}
      {actionResult && (
        <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-sm text-cyan-400">{actionResult}</span>
            {lastGenVendor && (
              <Link
                to={`/campaign-review?company=${encodeURIComponent(lastGenVendor)}`}
                className="inline-flex items-center gap-1 text-xs text-cyan-300 hover:text-white transition-colors"
              >
                Review campaigns <ExternalLink className="h-3 w-3" />
              </Link>
            )}
          </div>
          <button onClick={() => { setActionResult(null); setLastGenVendor(null) }} className="text-cyan-400 hover:text-white">
            <span className="text-lg leading-none">&times;</span>
          </button>
        </div>
      )}

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Challengers Tracked"
          value={targets.length}
          icon={<Swords className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Total Intent Leads"
          value={totalLeads}
          icon={<Target className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Active Purchase"
          value={totalActive}
          icon={<TrendingUp className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="In Evaluation"
          value={totalEval}
          icon={<Users className="h-5 w-5" />}
          skeleton={loading}
        />
      </div>

      {/* Search */}
      {validationUnavailableCount > 0 && (
        <div className="rounded-lg border border-slate-700/70 bg-slate-900/80 px-3 py-2 text-xs text-slate-300">
          Validation unavailable for {validationUnavailableCount} challenger row{validationUnavailableCount === 1 ? '' : 's'}.
          Unsafe winner-call fields and campaign actions are suppressed until the claim service responds.
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <input
          type="text"
          placeholder="Search challenger..."
          value={searchInput}
          onChange={(e) => setSearchInput(e.target.value)}
          className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 w-48"
        />
      </div>

      {/* Challengers Table */}
      <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
        <h3 className="text-sm font-medium text-slate-300 mb-4">
          Challenger Lead Funnel
        </h3>
        {loading ? (
          <DataTable columns={columns} data={[]} skeletonRows={5} />
        ) : (
          <DataTable
            columns={columns}
            data={filtered}
            onRowClick={(row) => navigate(vendorDetailPath(row.name, currentListPath))}
            emptyMessage={targets.length === 0
              ? 'No challenger targets configured. Add them in Vendor Targets.'
              : 'No matching challengers found'
            }
          />
        )}
      </div>

      {targets.length === 0 && !loading && (
        <div className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5">
          <p className="text-sm text-slate-400">
            Add challenger targets in the <a href="/vendor-targets" className="text-cyan-400 hover:underline">Vendor Targets</a> page
            with mode "Challenger Intel" to start tracking competitive intent signals.
          </p>
        </div>
      )}
    </div>
  )
}
