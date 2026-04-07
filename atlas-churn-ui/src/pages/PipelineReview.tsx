import { useState } from 'react'
import {
  Shield,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Filter,
  RefreshCw,
  XCircle,
  ChevronRight,
  DollarSign,
  Database,
  Cpu,
  Building2,
  CalendarClock,
  GitCompareArrows,
  Workflow,
} from 'lucide-react'
import { clsx } from 'clsx'
import useApiData from '../hooks/useApiData'
import DataTable from '../components/DataTable'
import StatCard from '../components/StatCard'
import type { Column } from '../components/DataTable'
import type {
  VisibilityQueueItem,
  VisibilityEvent,
  ArtifactAttempt,
  EnrichmentQuarantine,
  ExtractionHealthDailyRow,
  ExtractionHealthSourceRow,
  ExtractionHealthRunRow,
  ExtractionHealthVendorRow,
  SynthesisValidationResult,
  AdminCostSummary,
  AdminCostOperation,
  AdminCostVendor,
  AdminCostVendorPassRow,
  AdminCostSourceEfficiencyRow,
  AdminCostB2bRunRow,
  AdminCostB2bEfficiency,
  AdminCostBurnDashboard,
  AdminCostBurnBudgetRow,
  AdminCostBurnRow,
  AdminCostGenericReasoning,
  AdminCostGenericReasoningSourceRow,
  AdminCostGenericReasoningEventRow,
  AdminCostGenericReasoningSourceEventRow,
  AdminCostGenericReasoningEntityRow,
  AdminCostReconciliation,
  AdminCostReconciliationRow,
  AdminCostRecentCall,
  AdminCostCacheHealth,
  AdminCostReasoningActivity,
  AdminCostReasoningActivityPhase,
  AdminCostBatchStage,
  AdminCostStaleBatchJob,
  AdminCostStaleBatchClaim,
  AdminCostExactCacheStage,
  AdminCostPromptCacheSpan,
  AdminCostSemanticPatternClass,
  AdminCostTaskReuseRow,
  AdminCostRunDetail,
  AdminCostRunBatchJob,
  AdminCostRunBatchItem,
  AdminTaskHealthRow,
} from '../types'
import {
  fetchVisibilitySummary,
  fetchVisibilityQueue,
  fetchVisibilityEvents,
  fetchArtifactAttempts,
  fetchEnrichmentQuarantines,
  fetchExtractionHealth,
  fetchSynthesisValidationResults,
  resolveVisibilityReview,
  fetchAdminCostSummary,
  fetchAdminCostByOperation,
  fetchAdminCostByVendor,
  fetchAdminCostB2bEfficiency,
  fetchAdminCostBurnDashboard,
  fetchAdminCostGenericReasoning,
  fetchAdminCostReconciliation,
  fetchAdminCostRecent,
  fetchAdminCostCacheHealth,
  fetchAdminCostReasoningActivity,
  fetchAdminCostRun,
  fetchAdminTaskHealth,
  runAutonomousTask,
} from '../api/client'

type TabKey = 'queue' | 'failures' | 'quality' | 'audit' | 'costs'

const EXTRACTION_HEALTH_TOP_N = 12
const COST_VENDOR_LIMIT = 100
const B2B_EFFICIENCY_TOP_N = 25
const B2B_EFFICIENCY_RUN_LIMIT = 25
const BURN_DASHBOARD_TOP_N = 25
const GENERIC_REASONING_TOP_N = 8
const RECON_TASK_NAME = 'b2b_campaign_batch_reconciliation'

// ---------------------------------------------------------------------------
// Severity badge
// ---------------------------------------------------------------------------

const severityStyles: Record<string, string> = {
  critical: 'bg-red-500/20 text-red-400',
  error: 'bg-rose-500/20 text-rose-400',
  warning: 'bg-amber-500/20 text-amber-400',
  info: 'bg-slate-500/20 text-slate-400',
}

function SeverityBadge({ severity }: { severity: string }) {
  return (
    <span
      className={clsx(
        'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium',
        severityStyles[severity] || 'bg-slate-500/20 text-slate-400',
      )}
    >
      {severity}
    </span>
  )
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    open: 'bg-amber-500/20 text-amber-400',
    acknowledged: 'bg-cyan-500/20 text-cyan-400',
    resolved: 'bg-green-500/20 text-green-400',
    ignored: 'bg-slate-500/20 text-slate-400',
    accepted_risk: 'bg-purple-500/20 text-purple-400',
    passed: 'bg-green-500/20 text-green-400',
    succeeded: 'bg-green-500/20 text-green-400',
    failed: 'bg-red-500/20 text-red-400',
    rejected: 'bg-red-500/20 text-red-400',
    retried: 'bg-amber-500/20 text-amber-400',
    skipped: 'bg-slate-500/20 text-slate-400',
  }
  return (
    <span
      className={clsx(
        'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium',
        styles[status] || 'bg-slate-500/20 text-slate-400',
      )}
    >
      {status.replace(/_/g, ' ')}
    </span>
  )
}

function StageBadge({ stage }: { stage: string }) {
  return (
    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-cyan-500/10 text-cyan-400">
      {stage.replace(/_/g, ' ')}
    </span>
  )
}

function formatTs(ts: string | null | undefined): string {
  if (!ts) return '--'
  const d = new Date(ts)
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function formatNumber(value: number | null | undefined): string {
  return new Intl.NumberFormat().format(value ?? 0)
}

function formatCurrency(value: number | null | undefined): string {
  return new Intl.NumberFormat(undefined, {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 4,
  }).format(value ?? 0)
}

function formatMaybeCurrency(value: number | null | undefined): string {
  return value == null ? '--' : formatCurrency(value)
}

function formatMaybeNumber(value: number | null | undefined): string {
  return value == null ? '--' : formatNumber(value)
}

function formatMaybePercent(value: number | null | undefined): string {
  return value == null ? '--' : `${(value * 100).toFixed(1)}%`
}

function formatCompactTokens(value: number | null | undefined): string {
  const amount = Math.abs(value ?? 0)
  if (amount >= 1_000_000) {
    return `${((value ?? 0) / 1_000_000).toFixed(1)}M`
  }
  if (amount >= 1_000) {
    return `${((value ?? 0) / 1_000).toFixed(1)}K`
  }
  return formatNumber(value)
}

function formatAgeMinutes(value: number | null | undefined): string {
  const minutes = Number(value ?? 0)
  if (!Number.isFinite(minutes) || minutes <= 0) return '0m'
  if (minutes >= 60 * 24) return `${(minutes / (60 * 24)).toFixed(1)}d`
  if (minutes >= 60) return `${(minutes / 60).toFixed(1)}h`
  return `${Math.round(minutes)}m`
}

function formatFailureRate(value: number | null | undefined): string {
  const rate = Number(value ?? 0)
  if (!Number.isFinite(rate)) return '--'
  return `${Math.round(rate * 100)}%`
}

function truncateLabel(value: string | null | undefined, max = 38): string {
  const text = String(value || '').trim()
  if (!text) return '--'
  if (text.length <= max) return text
  return `${text.slice(0, max - 3)}...`
}

const visibilityCodeLabels: Record<string, string> = {
  unknown_packet_citation: 'Unknown packet citation',
  thin_specific_witness_pool: 'Thin specific witness pool',
  missing_citations: 'Missing citations',
  empty_category_reasoning: 'Empty category reasoning',
  repeated_validation_retry: 'Repeated recovered retries',
  costly_validation_retry: 'Costly recovered retries',
  validation_retry_rejected: 'Recovered retry',
}

function formatVisibilityCode(code: string | null | undefined): string {
  const key = String(code || '').trim()
  if (!key) return '--'
  return visibilityCodeLabels[key] || key.replace(/_/g, ' ')
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function asNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function toStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.map((item) => asString(item)).filter(Boolean)
    : []
}

function summarizeValidationDelta(value: SynthesisValidationResult): string[] {
  const delta = asRecord(value.reasoning_delta)
  if (Object.keys(delta).length === 0) return []
  const items: string[] = []
  if (delta.wedge_changed) items.push('Wedge changed')
  if (delta.confidence_changed) items.push('Confidence shifted')
  if (delta.top_destination_changed) {
    const destination = asString(delta.current_top_destination)
    items.push(destination ? `Destination: ${destination}` : 'Destination shifted')
  }
  const timingWindows = toStringArray(delta.new_timing_windows)
  if (timingWindows.length > 0) items.push(`Timing +${timingWindows.length}`)
  const accounts = toStringArray(delta.new_account_signals)
  if (accounts.length > 0) items.push(`Accounts +${accounts.length}`)
  return items
}

// ---------------------------------------------------------------------------
// Resolve dropdown for queue items
// ---------------------------------------------------------------------------

function ResolveDropdown({
  item,
  onResolved,
}: {
  item: VisibilityQueueItem
  onResolved: () => void
}) {
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)

  const actions = ['acknowledge', 'resolve', 'ignore', 'accept_risk'] as const

  async function handleAction(action: string) {
    setLoading(true)
    try {
      await resolveVisibilityReview(item.id, action)
      onResolved()
    } finally {
      setLoading(false)
      setOpen(false)
    }
  }

  return (
    <div className="relative">
      <button
        onClick={(e) => {
          e.stopPropagation()
          setOpen(!open)
        }}
        disabled={loading}
        className="px-2.5 py-1 text-xs font-medium bg-cyan-500/10 text-cyan-400 rounded-lg hover:bg-cyan-500/20 transition-colors disabled:opacity-50"
      >
        {loading ? 'Resolving...' : 'Resolve'}
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-1 w-40 bg-slate-800 border border-slate-700/50 rounded-lg shadow-xl z-10 py-1">
          {actions.map((a) => (
            <button
              key={a}
              onClick={(e) => {
                e.stopPropagation()
                handleAction(a)
              }}
              className="w-full text-left px-3 py-2 text-xs text-slate-300 hover:bg-slate-700/50 transition-colors"
            >
              {a.replace(/_/g, ' ')}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Filter bar shared component
// ---------------------------------------------------------------------------

function FilterSelect({
  label,
  value,
  onChange,
  options,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  options: { value: string; label: string }[]
}) {
  return (
    <div className="flex items-center gap-1.5">
      <label className="text-xs text-slate-500">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="bg-slate-800/50 border border-slate-700/50 rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-cyan-500/50"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab: Queue
// ---------------------------------------------------------------------------

function QueueTab({ onRefresh }: { onRefresh: () => void }) {
  const [stageFilter, setStageFilter] = useState('')
  const [severityFilter, setSeverityFilter] = useState('')

  const { data, loading, error, refresh, refreshing } = useApiData(
    () =>
      fetchVisibilityQueue({
        limit: 100,
        stage: stageFilter || undefined,
        severity: severityFilter || undefined,
      }),
    [stageFilter, severityFilter],
  )

  const handleResolved = () => {
    refresh()
    onRefresh()
  }

  const columns: Column<VisibilityQueueItem>[] = [
    {
      key: 'severity',
      header: 'Sev',
      render: (r) => <SeverityBadge severity={r.severity} />,
      sortable: true,
      sortValue: (r) => {
        const order: Record<string, number> = { critical: 0, error: 1, warning: 2, info: 3 }
        return order[r.severity] ?? 4
      },
    },
    {
      key: 'status',
      header: 'Status',
      render: (r) => <StatusBadge status={r.status} />,
      sortable: true,
      sortValue: (r) => r.status,
    },
    {
      key: 'stage',
      header: 'Stage',
      render: (r) => <StageBadge stage={r.stage} />,
      sortable: true,
      sortValue: (r) => r.stage,
    },
    {
      key: 'summary',
      header: 'Summary',
      render: (r) => (
        <div className="max-w-sm">
          <span className="text-white text-sm line-clamp-1">{r.summary}</span>
          <span className="text-xs text-slate-500 block">
            {r.reason_code ? `${formatVisibilityCode(r.reason_code)} | ` : ''}
            {r.entity_type}: {r.entity_id}
          </span>
        </div>
      ),
    },
    {
      key: 'occurrences',
      header: 'Count',
      render: (r) => (
        <span className="text-sm text-slate-300">{r.occurrence_count}</span>
      ),
      sortable: true,
      sortValue: (r) => r.occurrence_count,
    },
    {
      key: 'last_seen',
      header: 'Last Seen',
      render: (r) => (
        <span className="text-xs text-slate-400">{formatTs(r.last_seen_at)}</span>
      ),
      sortable: true,
      sortValue: (r) => r.last_seen_at || '',
    },
    {
      key: 'action',
      header: '',
      render: (r) =>
        r.actionable && r.status === 'open' ? (
          <ResolveDropdown item={r} onResolved={handleResolved} />
        ) : (
          <ChevronRight className="h-4 w-4 text-slate-600" />
        ),
    },
  ]

  const items = data?.items ?? []

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <Filter className="h-4 w-4 text-slate-500" />
        <FilterSelect
          label="Stage"
          value={stageFilter}
          onChange={setStageFilter}
          options={[
            { value: '', label: 'All stages' },
            { value: 'extraction', label: 'Extraction' },
            { value: 'synthesis', label: 'Synthesis' },
            { value: 'battle_cards', label: 'Battle Cards' },
            { value: 'blog', label: 'Blog' },
            { value: 'reports', label: 'Reports' },
            { value: 'task_execution', label: 'Task Execution' },
          ]}
        />
        <FilterSelect
          label="Severity"
          value={severityFilter}
          onChange={setSeverityFilter}
          options={[
            { value: '', label: 'All' },
            { value: 'critical', label: 'Critical' },
            { value: 'error', label: 'Error' },
            { value: 'warning', label: 'Warning' },
            { value: 'info', label: 'Info' },
          ]}
        />
        <button
          onClick={refresh}
          disabled={refreshing}
          className="ml-auto flex items-center gap-1.5 px-2.5 py-1 text-xs text-slate-400 hover:text-white transition-colors"
        >
          <RefreshCw className={clsx('h-3 w-3', refreshing && 'animate-spin')} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
          {error.message}
        </div>
      )}

      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden">
        {loading ? (
          <DataTable columns={columns} data={[]} skeletonRows={6} />
        ) : (
          <DataTable
            columns={columns}
            data={items}
            emptyMessage="No items in the review queue"
          />
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab: Failures (ArtifactAttempts)
// ---------------------------------------------------------------------------

function FailuresTab() {
  const [statusFilter, setStatusFilter] = useState('')

  const { data, loading, error, refresh, refreshing } = useApiData(
    () =>
      fetchArtifactAttempts({
        status: statusFilter || undefined,
        limit: 100,
        hours: 720,
      }),
    [statusFilter],
  )

  const columns: Column<ArtifactAttempt>[] = [
    {
      key: 'status',
      header: 'Status',
      render: (r) => <StatusBadge status={r.status} />,
      sortable: true,
      sortValue: (r) => r.status,
    },
    {
      key: 'artifact_type',
      header: 'Artifact',
      render: (r) => (
        <span className="text-sm text-white">{r.artifact_type}</span>
      ),
      sortable: true,
      sortValue: (r) => r.artifact_type,
    },
    {
      key: 'stage',
      header: 'Stage',
      render: (r) => <StageBadge stage={r.stage} />,
      sortable: true,
      sortValue: (r) => r.stage,
    },
    {
      key: 'attempt',
      header: 'Attempt',
      render: (r) => (
        <span className="text-sm text-slate-300">#{r.attempt_no}</span>
      ),
      sortable: true,
      sortValue: (r) => r.attempt_no,
    },
    {
      key: 'issues',
      header: 'Issues',
      render: (r) => (
        <div className="flex items-center gap-2">
          {r.blocker_count > 0 && (
            <span className="text-xs text-red-400">{r.blocker_count} blocker{r.blocker_count !== 1 ? 's' : ''}</span>
          )}
          {r.warning_count > 0 && (
            <span className="text-xs text-amber-400">{r.warning_count} warn</span>
          )}
        </div>
      ),
    },
    {
      key: 'error',
      header: 'Error',
      render: (r) => (
        <div className="max-w-xs">
          <span className="text-xs text-slate-400 line-clamp-1">
            {r.error_message || r.failure_step || '--'}
          </span>
        </div>
      ),
    },
    {
      key: 'score',
      header: 'Score',
      render: (r) => (
        <span className="text-xs text-slate-300">
          {r.score != null ? `${r.score.toFixed(1)}${r.threshold != null ? ` / ${r.threshold}` : ''}` : '--'}
        </span>
      ),
      sortable: true,
      sortValue: (r) => r.score ?? -1,
    },
    {
      key: 'started',
      header: 'Started',
      render: (r) => (
        <span className="text-xs text-slate-400">{formatTs(r.started_at)}</span>
      ),
      sortable: true,
      sortValue: (r) => r.started_at || '',
    },
  ]

  const items = data?.attempts ?? []

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <Filter className="h-4 w-4 text-slate-500" />
        <FilterSelect
          label="Status"
          value={statusFilter}
          onChange={setStatusFilter}
          options={[
            { value: '', label: 'All' },
            { value: 'rejected', label: 'Rejected' },
            { value: 'failed', label: 'Failed' },
            { value: 'succeeded', label: 'Succeeded' },
            { value: 'skipped', label: 'Skipped' },
          ]}
        />
        <button
          onClick={refresh}
          disabled={refreshing}
          className="ml-auto flex items-center gap-1.5 px-2.5 py-1 text-xs text-slate-400 hover:text-white transition-colors"
        >
          <RefreshCw className={clsx('h-3 w-3', refreshing && 'animate-spin')} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
          {error.message}
        </div>
      )}

      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden">
        {loading ? (
          <DataTable columns={columns} data={[]} skeletonRows={6} />
        ) : (
          <DataTable
            columns={columns}
            data={items}
            emptyMessage="No artifact attempts in the last 72h"
          />
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab: Quality Signals
// ---------------------------------------------------------------------------

function QualityTab() {
  const [unreleasedOnly, setUnreleasedOnly] = useState(true)
  const [severityFilter, setSeverityFilter] = useState('')
  const [retryOnly, setRetryOnly] = useState(false)
  const [healthDays, setHealthDays] = useState('30')

  const {
    data: quarantineData,
    loading: quarantinesLoading,
    error: quarantinesError,
    refresh: refreshQuarantines,
    refreshing: quarantinesRefreshing,
  } = useApiData(
    () =>
      fetchEnrichmentQuarantines({
        unreleased_only: unreleasedOnly,
        limit: 100,
      }),
    [unreleasedOnly],
  )

  const {
    data: extractionHealth,
    loading: extractionHealthLoading,
    error: extractionHealthError,
    refresh: refreshExtractionHealth,
    refreshing: extractionHealthRefreshing,
  } = useApiData(
    () =>
      fetchExtractionHealth({
        days: Number(healthDays),
        top_n: EXTRACTION_HEALTH_TOP_N,
      }),
    [healthDays],
  )

  const {
    data: validationData,
    loading: validationsLoading,
    error: validationsError,
    refresh: refreshValidations,
    refreshing: validationsRefreshing,
  } = useApiData(
    () =>
      fetchSynthesisValidationResults({
        severity: severityFilter || undefined,
        passed: false,
        retry_only: retryOnly,
        limit: 100,
      }),
    [severityFilter, retryOnly],
  )

  const columns: Column<EnrichmentQuarantine>[] = [
    {
      key: 'severity',
      header: 'Sev',
      render: (r) => <SeverityBadge severity={r.severity} />,
      sortable: true,
      sortValue: (r) => {
        const order: Record<string, number> = { critical: 0, error: 1, warning: 2, info: 3 }
        return order[r.severity] ?? 4
      },
    },
    {
      key: 'vendor',
      header: 'Vendor',
      render: (r) => (
        <span className="text-sm text-white">{r.vendor_name || '--'}</span>
      ),
      sortable: true,
      sortValue: (r) => r.vendor_name || '',
    },
    {
      key: 'reason_code',
      header: 'Reason',
      render: (r) => (
        <span className="text-xs text-slate-300 font-mono">{r.reason_code}</span>
      ),
      sortable: true,
      sortValue: (r) => r.reason_code,
    },
    {
      key: 'summary',
      header: 'Summary',
      render: (r) => (
        <div className="max-w-sm">
          <span className="text-xs text-slate-400 line-clamp-1">
            {r.summary || '--'}
          </span>
        </div>
      ),
    },
    {
      key: 'source',
      header: 'Source',
      render: (r) => (
        <span className="text-xs text-slate-400">{r.source || '--'}</span>
      ),
      sortable: true,
      sortValue: (r) => r.source || '',
    },
    {
      key: 'quarantined_at',
      header: 'Quarantined',
      render: (r) => (
        <span className="text-xs text-slate-400">{formatTs(r.quarantined_at)}</span>
      ),
      sortable: true,
      sortValue: (r) => r.quarantined_at,
    },
    {
      key: 'released',
      header: 'Released',
      render: (r) =>
        r.released_at ? (
          <span className="text-xs text-green-400">{formatTs(r.released_at)}</span>
        ) : (
          <span className="text-xs text-slate-600">--</span>
        ),
    },
  ]

  const validationColumns: Column<SynthesisValidationResult>[] = [
    {
      key: 'severity',
      header: 'Sev',
      render: (r) => <SeverityBadge severity={r.severity} />,
      sortable: true,
      sortValue: (r) => {
        const order: Record<string, number> = { critical: 0, error: 1, warning: 2, info: 3 }
        return order[r.severity] ?? 4
      },
    },
    {
      key: 'vendor_name',
      header: 'Vendor',
      render: (r) => <span className="text-sm text-white">{r.vendor_name}</span>,
      sortable: true,
      sortValue: (r) => r.vendor_name,
    },
    {
      key: 'attempt_no',
      header: 'Attempt',
      render: (r) => <span className="text-xs text-slate-300">#{r.attempt_no}</span>,
      sortable: true,
      sortValue: (r) => r.attempt_no,
    },
    {
      key: 'rule_code',
      header: 'Rule',
      render: (r) => (
        <div className="max-w-[14rem]" title={r.rule_code}>
          <span className="text-xs text-slate-300">{formatVisibilityCode(r.rule_code)}</span>
          <span className="text-[11px] text-slate-600 block font-mono">{r.rule_code}</span>
        </div>
      ),
      sortable: true,
      sortValue: (r) => r.rule_code,
    },
    {
      key: 'summary',
      header: 'Summary',
      render: (r) => {
        const scope = asRecord(r.scope_manifest)
        const packet = asRecord(r.payload_component_tokens)
        const deltaItems = summarizeValidationDelta(r)
        const witnessCount = asNumber(scope.witnesses_in_scope)
        const reviewCount = asNumber(scope.reviews_in_scope)
        const witnessTokens = asNumber(packet.witness_pack)
        const sectionTokens = asNumber(packet.section_packets)
        return (
          <div className="max-w-sm">
            <span className="text-xs text-slate-400 line-clamp-2">{r.summary}</span>
            <span className="text-[11px] text-slate-600 block">
              {r.field_path || '--'}
            </span>
            {(witnessCount !== null || reviewCount !== null || witnessTokens !== null || sectionTokens !== null) && (
              <div className="mt-1 flex flex-wrap gap-1">
                {(witnessCount !== null || reviewCount !== null) && (
                  <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300">
                    {witnessCount ?? 0} witnesses / {reviewCount ?? 0} reviews
                  </span>
                )}
                {(witnessTokens !== null || sectionTokens !== null) && (
                  <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-300">
                    packet {formatCompactTokens((witnessTokens ?? 0) + (sectionTokens ?? 0))} tokens
                  </span>
                )}
                {asString(r.evidence_hash) && (
                  <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-slate-400">
                    {truncateLabel(asString(r.evidence_hash), 14)}
                  </span>
                )}
              </div>
            )}
            {deltaItems.length > 0 && (
              <div className="mt-1 flex flex-wrap gap-1">
                {deltaItems.slice(0, 2).map((item, index) => (
                  <span key={`${r.id}-${index}`} className="rounded bg-cyan-500/10 px-1.5 py-0.5 text-[10px] text-cyan-300">
                    {item}
                  </span>
                ))}
              </div>
            )}
          </div>
        )
      },
    },
    {
      key: 'created_at',
      header: 'Seen',
      render: (r) => (
        <span className="text-xs text-slate-400">{formatTs(r.created_at)}</span>
      ),
      sortable: true,
      sortValue: (r) => r.created_at,
    },
  ]

  const quarantines = quarantineData?.quarantines ?? []
  const validations = validationData?.results ?? []
  const trendColumns: Column<ExtractionHealthDailyRow>[] = [
    {
      key: 'day',
      header: 'Day',
      render: (r) => <span className="text-sm text-white">{r.day}</span>,
      sortable: true,
      sortValue: (r) => r.day,
    },
    {
      key: 'hard_gap_rows',
      header: 'Hard Gaps',
      render: (r) => <span className="text-sm text-slate-200">{formatNumber(r.hard_gap_rows)}</span>,
      sortable: true,
      sortValue: (r) => r.hard_gap_rows,
    },
    {
      key: 'phrase_arrays_without_spans',
      header: 'Phrase Arrays / No Spans',
      render: (r) => <span className="text-xs text-slate-300">{formatNumber(r.phrase_arrays_without_spans)}</span>,
      sortable: true,
      sortValue: (r) => r.phrase_arrays_without_spans,
    },
    {
      key: 'blank_replacement_mode',
      header: 'Blank Replacement',
      render: (r) => <span className="text-xs text-slate-300">{formatNumber(r.blank_replacement_mode)}</span>,
      sortable: true,
      sortValue: (r) => r.blank_replacement_mode,
    },
    {
      key: 'blank_operating_model_shift',
      header: 'Blank Operating Shift',
      render: (r) => <span className="text-xs text-slate-300">{formatNumber(r.blank_operating_model_shift)}</span>,
      sortable: true,
      sortValue: (r) => r.blank_operating_model_shift,
    },
    {
      key: 'strategic_candidate_rows',
      header: 'Strategic Candidates',
      render: (r) => <span className="text-xs text-amber-300">{formatNumber(r.strategic_candidate_rows)}</span>,
      sortable: true,
      sortValue: (r) => r.strategic_candidate_rows,
    },
  ]

  const vendorColumns: Column<ExtractionHealthVendorRow>[] = [
    {
      key: 'vendor_name',
      header: 'Vendor',
      render: (r) => <span className="text-sm text-white">{r.vendor_name}</span>,
      sortable: true,
      sortValue: (r) => r.vendor_name,
    },
    {
      key: 'hard_gap_rows',
      header: 'Hard Gaps',
      render: (r) => <span className="text-sm text-slate-200">{formatNumber(r.hard_gap_rows)}</span>,
      sortable: true,
      sortValue: (r) => r.hard_gap_rows,
    },
    {
      key: 'phrase_arrays_without_spans',
      header: 'Phrase Arrays / No Spans',
      render: (r) => <span className="text-xs text-slate-300">{formatNumber(r.phrase_arrays_without_spans)}</span>,
      sortable: true,
      sortValue: (r) => r.phrase_arrays_without_spans,
    },
    {
      key: 'empty_salience_flags',
      header: 'Empty Salience',
      render: (r) => <span className="text-xs text-slate-300">{formatNumber(r.empty_salience_flags)}</span>,
      sortable: true,
      sortValue: (r) => r.empty_salience_flags,
    },
    {
      key: 'strategic_candidate_rows',
      header: 'Strategic Candidates',
      render: (r) => <span className="text-xs text-amber-300">{formatNumber(r.strategic_candidate_rows)}</span>,
      sortable: true,
      sortValue: (r) => r.strategic_candidate_rows,
    },
    {
      key: 'enriched_rows',
      header: 'Enriched Rows',
      render: (r) => <span className="text-xs text-slate-400">{formatNumber(r.enriched_rows)}</span>,
      sortable: true,
      sortValue: (r) => r.enriched_rows,
    },
  ]

  const sourceColumns: Column<ExtractionHealthSourceRow>[] = [
    {
      key: 'source',
      header: 'Source',
      render: (r) => <span className="text-sm text-white">{r.source}</span>,
      sortable: true,
      sortValue: (r) => r.source,
    },
    {
      key: 'enriched_rows',
      header: 'Enriched',
      render: (r) => <span className="text-xs text-slate-300">{formatNumber(r.enriched_rows)}</span>,
      sortable: true,
      sortValue: (r) => r.enriched_rows,
    },
    {
      key: 'span_count',
      header: 'Spans',
      render: (r) => <span className="text-xs text-cyan-300">{formatNumber(r.span_count)}</span>,
      sortable: true,
      sortValue: (r) => r.span_count,
    },
    {
      key: 'witness_yield_rate',
      header: 'Yield / Review',
      render: (r) => <span className="text-xs text-slate-300">{r.witness_yield_rate.toFixed(2)}</span>,
      sortable: true,
      sortValue: (r) => r.witness_yield_rate,
    },
    {
      key: 'repair_trigger_rate',
      header: 'Repair Trigger',
      render: (r) => <span className="text-xs text-amber-300">{(r.repair_trigger_rate * 100).toFixed(1)}%</span>,
      sortable: true,
      sortValue: (r) => r.repair_trigger_rate,
    },
    {
      key: 'repair_promoted_rate',
      header: 'Repair Promoted',
      render: (r) => <span className="text-xs text-slate-300">{(r.repair_promoted_rate * 100).toFixed(1)}%</span>,
      sortable: true,
      sortValue: (r) => r.repair_promoted_rate,
    },
    {
      key: 'strict_discussion_candidates_kept_rows',
      header: 'Gate Kept',
      render: (r) => <span className="text-xs text-cyan-300">{formatNumber(r.strict_discussion_candidates_kept_rows)}</span>,
      sortable: true,
      sortValue: (r) => r.strict_discussion_candidates_kept_rows,
    },
    {
      key: 'low_signal_discussion_skipped_rows',
      header: 'Gate Skipped',
      render: (r) => <span className="text-xs text-amber-300">{formatNumber(r.low_signal_discussion_skipped_rows)}</span>,
      sortable: true,
      sortValue: (r) => r.low_signal_discussion_skipped_rows,
    },
  ]

  const runColumns: Column<ExtractionHealthRunRow>[] = [
    {
      key: 'started_at',
      header: 'Run',
      render: (r) => (
        <div className="max-w-[220px]">
          <p className="truncate text-sm text-white">{r.task_name}</p>
          <p className="truncate text-xs text-slate-500">{formatTs(r.started_at)}</p>
        </div>
      ),
      sortable: true,
      sortValue: (r) => r.started_at || '',
    },
    {
      key: 'reviews_processed',
      header: 'Processed',
      render: (r) => <span className="text-xs text-slate-300">{formatNumber(r.reviews_processed)}</span>,
      sortable: true,
      sortValue: (r) => r.reviews_processed,
    },
    {
      key: 'witness_count',
      header: 'Spans',
      render: (r) => <span className="text-xs text-cyan-300">{formatNumber(r.witness_count)}</span>,
      sortable: true,
      sortValue: (r) => r.witness_count,
    },
    {
      key: 'witness_yield_rate',
      header: 'Yield / Review',
      render: (r) => <span className="text-xs text-slate-300">{r.witness_yield_rate.toFixed(2)}</span>,
      sortable: true,
      sortValue: (r) => r.witness_yield_rate,
    },
    {
      key: 'secondary_write_hits',
      header: 'Secondary Writes',
      render: (r) => <span className="text-xs text-amber-300">{formatNumber(r.secondary_write_hits)}</span>,
      sortable: true,
      sortValue: (r) => r.secondary_write_hits,
    },
    {
      key: 'strict_discussion_candidates_kept',
      header: 'Gate Kept',
      render: (r) => <span className="text-xs text-cyan-300">{formatNumber(r.strict_discussion_candidates_kept)}</span>,
      sortable: true,
      sortValue: (r) => r.strict_discussion_candidates_kept,
    },
    {
      key: 'strict_discussion_candidates_dropped',
      header: 'Gate Dropped',
      render: (r) => <span className="text-xs text-amber-300">{formatNumber(r.strict_discussion_candidates_dropped)}</span>,
      sortable: true,
      sortValue: (r) => r.strict_discussion_candidates_dropped,
    },
    {
      key: 'exact_cache_hits',
      header: 'Exact Hits',
      render: (r) => <span className="text-xs text-slate-300">{formatNumber(r.exact_cache_hits)}</span>,
      sortable: true,
      sortValue: (r) => r.exact_cache_hits,
    },
  ]

  const loading = quarantinesLoading || validationsLoading || extractionHealthLoading
  const refreshing = quarantinesRefreshing || validationsRefreshing || extractionHealthRefreshing
  const error = quarantinesError || validationsError || extractionHealthError
  const snapshot = extractionHealth?.current_snapshot
  const extractionTrend = extractionHealth?.daily_trend ?? []
  const extractionVendors = extractionHealth?.top_vendors ?? []
  const extractionSources = extractionHealth?.top_sources ?? []
  const extractionRuns = extractionHealth?.recent_runs ?? []

  const refresh = () => {
    refreshQuarantines()
    refreshValidations()
    refreshExtractionHealth()
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <Filter className="h-4 w-4 text-slate-500" />
        <label className="flex items-center gap-1.5 text-xs text-slate-400 cursor-pointer">
          <input
            type="checkbox"
            checked={unreleasedOnly}
            onChange={(e) => setUnreleasedOnly(e.target.checked)}
            className="accent-cyan-500"
          />
          Unreleased only
        </label>
        <FilterSelect
          label="Validation Sev"
          value={severityFilter}
          onChange={setSeverityFilter}
          options={[
            { value: '', label: 'All validation findings' },
            { value: 'error', label: 'Error only' },
            { value: 'warning', label: 'Warning only' },
          ]}
        />
        <FilterSelect
          label="Health Window"
          value={healthDays}
          onChange={setHealthDays}
          options={[
            { value: '7', label: '7d' },
            { value: '14', label: '14d' },
            { value: '30', label: '30d' },
            { value: '60', label: '60d' },
            { value: '90', label: '90d' },
          ]}
        />
        <label className="flex items-center gap-1.5 text-xs text-slate-400 cursor-pointer">
          <input
            type="checkbox"
            checked={retryOnly}
            onChange={(e) => setRetryOnly(e.target.checked)}
            className="accent-cyan-500"
          />
          Retry-only findings
        </label>
        <button
          onClick={refresh}
          disabled={refreshing}
          className="ml-auto flex items-center gap-1.5 px-2.5 py-1 text-xs text-slate-400 hover:text-white transition-colors"
        >
          <RefreshCw className={clsx('h-3 w-3', refreshing && 'animate-spin')} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
          {error.message}
        </div>
      )}

      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 space-y-4">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h2 className="text-sm font-medium text-white">Extraction Health</h2>
            <p className="text-xs text-slate-500">
              Current witness-primitive gap counts plus recent daily backlog trend.
            </p>
          </div>
          <div className="text-xs text-slate-500">
            Enriched corpus: <span className="text-slate-300">{formatNumber(snapshot?.enriched_rows)}</span>
          </div>
        </div>

        <div className="grid grid-cols-2 xl:grid-cols-5 gap-4">
          <StatCard
            label="Hard Gaps"
            value={snapshot?.hard_gap_rows ?? 0}
            icon={<AlertTriangle className="h-4 w-4" />}
            skeleton={extractionHealthLoading}
          />
          <StatCard
            label="Phrase Arrays / No Spans"
            value={snapshot?.phrase_arrays_without_spans ?? 0}
            icon={<XCircle className="h-4 w-4" />}
            skeleton={extractionHealthLoading}
          />
          <StatCard
            label="Blank Replacement"
            value={snapshot?.blank_replacement_mode ?? 0}
            icon={<ChevronRight className="h-4 w-4" />}
            skeleton={extractionHealthLoading}
          />
          <StatCard
            label="Blank Operating Shift"
            value={snapshot?.blank_operating_model_shift ?? 0}
            icon={<RefreshCw className="h-4 w-4" />}
            skeleton={extractionHealthLoading}
          />
          <StatCard
            label="Empty Salience"
            value={snapshot?.empty_salience_flags ?? 0}
            icon={<Clock className="h-4 w-4" />}
            skeleton={extractionHealthLoading}
          />
        </div>

        <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
          <StatCard
            label="Witness Yield / Review"
            value={snapshot ? snapshot.witness_yield_rate.toFixed(2) : '0.00'}
            icon={<Database className="h-4 w-4" />}
            skeleton={extractionHealthLoading}
          />
          <StatCard
            label="Repair Trigger Rate"
            value={snapshot ? `${(snapshot.repair_trigger_rate * 100).toFixed(1)}%` : '0.0%'}
            icon={<RefreshCw className="h-4 w-4" />}
            skeleton={extractionHealthLoading}
          />
          <StatCard
            label={`Secondary Writes ${healthDays}d`}
            value={snapshot?.secondary_write_hits_window ?? 0}
            icon={<GitCompareArrows className="h-4 w-4" />}
            skeleton={extractionHealthLoading}
          />
          <StatCard
            label="Total Spans"
            value={snapshot?.span_count ?? 0}
            icon={<Database className="h-4 w-4" />}
            skeleton={extractionHealthLoading}
          />
        </div>

        <div className="grid grid-cols-2 xl:grid-cols-3 gap-4">
          <StatCard
            label="Strategic Candidates"
            value={snapshot?.strategic_candidate_rows ?? 0}
            icon={<AlertTriangle className="h-4 w-4" />}
            skeleton={extractionHealthLoading}
          />
          <StatCard
            label="Money / No Pricing Span"
            value={snapshot?.money_without_pricing_span ?? 0}
            icon={<DollarSign className="h-4 w-4" />}
            skeleton={extractionHealthLoading}
          />
          <StatCard
            label="Competitor / No Framing"
            value={snapshot?.competitor_without_displacement_framing ?? 0}
            icon={<GitCompareArrows className="h-4 w-4" />}
            skeleton={extractionHealthLoading}
          />
          <StatCard
            label="Named Company / No Evidence"
            value={snapshot?.named_company_without_named_account_evidence ?? 0}
            icon={<Building2 className="h-4 w-4" />}
            skeleton={extractionHealthLoading}
          />
          <StatCard
            label="Timeline / No Anchor"
            value={snapshot?.timeline_language_without_timing_anchor ?? 0}
            icon={<CalendarClock className="h-4 w-4" />}
            skeleton={extractionHealthLoading}
          />
          <StatCard
            label="Workflow / No Replacement"
            value={snapshot?.workflow_language_without_replacement_mode ?? 0}
            icon={<Workflow className="h-4 w-4" />}
            skeleton={extractionHealthLoading}
          />
          <StatCard
            label="Low-Signal Skipped"
            value={snapshot?.low_signal_discussion_skipped_rows ?? 0}
            icon={<Shield className="h-4 w-4" />}
            skeleton={extractionHealthLoading}
          />
        </div>

        <div className="grid gap-4 xl:grid-cols-2">
          <div className="border border-slate-700/50 rounded-xl overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-700/50">
              <h3 className="text-sm font-medium text-white">Daily Trend</h3>
              <p className="text-xs text-slate-500">Backlog shape by enrichment day</p>
            </div>
            {loading ? (
              <DataTable columns={trendColumns} data={[]} skeletonRows={6} />
            ) : (
              <DataTable
                columns={trendColumns}
                data={extractionTrend}
                emptyMessage="No extraction health rows in the selected window"
              />
            )}
          </div>

          <div className="border border-slate-700/50 rounded-xl overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-700/50">
              <h3 className="text-sm font-medium text-white">Top Vendors</h3>
              <p className="text-xs text-slate-500">Current vendors with the most remaining gaps</p>
            </div>
            {loading ? (
              <DataTable columns={vendorColumns} data={[]} skeletonRows={6} />
            ) : (
              <DataTable
                columns={vendorColumns}
                data={extractionVendors}
                emptyMessage="No vendors currently show extraction-health gaps"
              />
            )}
          </div>
        </div>

        <div className="grid gap-4 xl:grid-cols-2">
          <div className="border border-slate-700/50 rounded-xl overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-700/50">
              <h3 className="text-sm font-medium text-white">Top Sources</h3>
              <p className="text-xs text-slate-500">Witness yield, repair pressure, and strict-gate suppression by review source</p>
            </div>
            {loading ? (
              <DataTable columns={sourceColumns} data={[]} skeletonRows={6} />
            ) : (
              <DataTable
                columns={sourceColumns}
                data={extractionSources}
                emptyMessage="No source-level extraction health rows in the selected window"
              />
            )}
          </div>

          <div className="border border-slate-700/50 rounded-xl overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-700/50">
              <h3 className="text-sm font-medium text-white">Recent Enrichment Runs</h3>
              <p className="text-xs text-slate-500">Per-run witness yield, strict-gate keep/drop counts, and secondary-write activity</p>
            </div>
            {loading ? (
              <DataTable columns={runColumns} data={[]} skeletonRows={6} />
            ) : (
              <DataTable
                columns={runColumns}
                data={extractionRuns}
                emptyMessage="No recent enrichment or repair runs in the selected window"
              />
            )}
          </div>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-700/50">
            <h2 className="text-sm font-medium text-white">Synthesis Validation</h2>
            <p className="text-xs text-slate-500">Failed and warning rule rows from reasoning synthesis</p>
          </div>
          {loading ? (
            <DataTable columns={validationColumns} data={[]} skeletonRows={6} />
          ) : (
            <DataTable
              columns={validationColumns}
              data={validations}
              emptyMessage="No synthesis validation findings"
            />
          )}
        </div>

        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-700/50">
            <h2 className="text-sm font-medium text-white">Enrichment Quarantines</h2>
            <p className="text-xs text-slate-500">Input suppression and release state</p>
          </div>
          {loading ? (
            <DataTable columns={columns} data={[]} skeletonRows={6} />
          ) : (
            <DataTable
              columns={columns}
              data={quarantines}
              emptyMessage="No quarantined items"
            />
          )}
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab: Audit Trail (all events)
// ---------------------------------------------------------------------------

function AuditTab() {
  const [stageFilter, setStageFilter] = useState('')
  const [severityFilter, setSeverityFilter] = useState('')

  const { data, loading, error, refresh, refreshing } = useApiData(
    () =>
      fetchVisibilityEvents({
        limit: 100,
        hours: 720,
        stage: stageFilter || undefined,
        severity: severityFilter || undefined,
      }),
    [stageFilter, severityFilter],
  )

  const {
    data: legacyData,
    loading: legacyLoading,
    error: legacyError,
  } = useApiData(
    () =>
      fetchVisibilityEvents({
        limit: 100,
        hours: 720,
        event_type: 'legacy_reasoning_opt_in',
      }),
    [],
  )

  const legacyEvents = legacyData?.events ?? []
  const legacyViewFallbacks = legacyEvents.filter((event) => event.reason_code === 'legacy_reasoning_view_fallback')
  const legacyBatchFallbacks = legacyEvents.filter((event) => event.reason_code === 'legacy_reasoning_batch_fallback')
  const legacyDiscoveryEvents = legacyEvents.filter((event) => event.reason_code === 'legacy_reasoning_vendor_discovery')
  const legacyCrossVendorFallbacks = legacyEvents.filter((event) => event.reason_code === 'legacy_cross_vendor_fallback')

  const legacyColumns: Column<VisibilityEvent>[] = [
    {
      key: 'reason_code',
      header: 'Reason',
      render: (r) => (
        <span className="text-xs text-slate-300">{formatVisibilityCode(r.reason_code)}</span>
      ),
    },
    {
      key: 'summary',
      header: 'Summary',
      render: (r) => (
        <div className="max-w-sm">
          <span className="text-xs text-slate-400 line-clamp-1">{r.summary}</span>
        </div>
      ),
    },
    {
      key: 'entity',
      header: 'Entity',
      render: (r) => (
        <span className="text-xs text-slate-400">{r.entity_type}: {truncateLabel(r.entity_id, 36)}</span>
      ),
    },
    {
      key: 'occurred_at',
      header: 'Time',
      render: (r) => (
        <span className="text-xs text-slate-400">{formatTs(r.occurred_at)}</span>
      ),
      sortable: true,
      sortValue: (r) => r.occurred_at,
    },
  ]

  const columns: Column<VisibilityEvent>[] = [
    {
      key: 'severity',
      header: 'Sev',
      render: (r) => <SeverityBadge severity={r.severity} />,
      sortable: true,
      sortValue: (r) => {
        const order: Record<string, number> = { critical: 0, error: 1, warning: 2, info: 3 }
        return order[r.severity] ?? 4
      },
    },
    {
      key: 'event_type',
      header: 'Event',
      render: (r) => (
        <span className="text-sm text-white">{r.event_type.replace(/_/g, ' ')}</span>
      ),
      sortable: true,
      sortValue: (r) => r.event_type,
    },
    {
      key: 'stage',
      header: 'Stage',
      render: (r) => <StageBadge stage={r.stage} />,
      sortable: true,
      sortValue: (r) => r.stage,
    },
    {
      key: 'entity',
      header: 'Entity',
      render: (r) => (
        <div className="max-w-xs">
          <span className="text-xs text-slate-300">{r.entity_type}: {r.entity_id}</span>
        </div>
      ),
    },
    {
      key: 'summary',
      header: 'Summary',
      render: (r) => (
        <div className="max-w-sm">
          <span className="text-xs text-slate-400 line-clamp-1">{r.summary}</span>
        </div>
      ),
    },
    {
      key: 'decision',
      header: 'Decision',
      render: (r) => (
        <span className="text-xs text-slate-400">{r.decision || '--'}</span>
      ),
    },
    {
      key: 'occurred_at',
      header: 'Time',
      render: (r) => (
        <span className="text-xs text-slate-400">{formatTs(r.occurred_at)}</span>
      ),
      sortable: true,
      sortValue: (r) => r.occurred_at,
    },
  ]

  const items = data?.events ?? []

  return (
    <div className="space-y-4">
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-white">Deprecated Legacy Compatibility</h3>
            <p className="text-xs text-slate-500">
              Explicit legacy reasoning opt-ins seen in the last 30 days. This is deprecated compatibility mode, not normal delivery.
            </p>
          </div>
        </div>
        {legacyError && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
            {legacyError.message}
          </div>
        )}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
          <StatCard
            label="Single-Vendor Fallbacks"
            value={legacyLoading ? '--' : formatNumber(legacyViewFallbacks.length)}
            icon={<AlertTriangle className="h-5 w-5" />}
            sub="load_best_reasoning_view"
            skeleton={legacyLoading}
          />
          <StatCard
            label="Batch Fallbacks"
            value={legacyLoading ? '--' : formatNumber(legacyBatchFallbacks.length)}
            icon={<Database className="h-5 w-5" />}
            sub="load_best_reasoning_views"
            skeleton={legacyLoading}
          />
          <StatCard
            label="Discovery Opt-Ins"
            value={legacyLoading ? '--' : formatNumber(legacyDiscoveryEvents.length)}
            icon={<Building2 className="h-5 w-5" />}
            sub="include_legacy vendor discovery"
            skeleton={legacyLoading}
          />
          <StatCard
            label="Cross-Vendor Fallbacks"
            value={legacyLoading ? '--' : formatNumber(legacyCrossVendorFallbacks.length)}
            icon={<GitCompareArrows className="h-5 w-5" />}
            sub="allow_legacy_fallback"
            skeleton={legacyLoading}
          />
        </div>
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden">
          {legacyLoading ? (
            <DataTable columns={legacyColumns} data={[]} skeletonRows={4} />
          ) : (
            <DataTable
              columns={legacyColumns}
              data={legacyEvents.slice(0, 10)}
              emptyMessage="No explicit legacy opt-ins recorded"
            />
          )}
        </div>
      </div>

      <div className="flex items-center gap-4">
        <Filter className="h-4 w-4 text-slate-500" />
        <FilterSelect
          label="Stage"
          value={stageFilter}
          onChange={setStageFilter}
          options={[
            { value: '', label: 'All stages' },
            { value: 'extraction', label: 'Extraction' },
            { value: 'synthesis', label: 'Synthesis' },
            { value: 'battle_cards', label: 'Battle Cards' },
            { value: 'blog', label: 'Blog' },
            { value: 'reports', label: 'Reports' },
            { value: 'compatibility', label: 'Compatibility' },
            { value: 'task_execution', label: 'Task Execution' },
          ]}
        />
        <FilterSelect
          label="Severity"
          value={severityFilter}
          onChange={setSeverityFilter}
          options={[
            { value: '', label: 'All' },
            { value: 'critical', label: 'Critical' },
            { value: 'error', label: 'Error' },
            { value: 'warning', label: 'Warning' },
            { value: 'info', label: 'Info' },
          ]}
        />
        <button
          onClick={refresh}
          disabled={refreshing}
          className="ml-auto flex items-center gap-1.5 px-2.5 py-1 text-xs text-slate-400 hover:text-white transition-colors"
        >
          <RefreshCw className={clsx('h-3 w-3', refreshing && 'animate-spin')} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
          {error.message}
        </div>
      )}

      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden">
        {loading ? (
          <DataTable columns={columns} data={[]} skeletonRows={6} />
        ) : (
          <DataTable
            columns={columns}
            data={items}
            emptyMessage="No events in the last 48h"
          />
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab: Costs
// ---------------------------------------------------------------------------

interface CostCoreData {
  summary: AdminCostSummary
  operations: AdminCostOperation[]
  recent: AdminCostRecentCall[]
}

function CostsTab() {
  const [days, setDays] = useState('30')
  const [cacheTopN, setCacheTopN] = useState('8')
  const [providerFilter, setProviderFilter] = useState('')
  const [modelFilter, setModelFilter] = useState('')
  const [spanFilter, setSpanFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [sourceFilter, setSourceFilter] = useState('')
  const [eventTypeFilter, setEventTypeFilter] = useState('')
  const [entityTypeFilter, setEntityTypeFilter] = useState('')
  const [cacheFilter, setCacheFilter] = useState('')
  const [runIdInput, setRunIdInput] = useState('')
  const [activeRunId, setActiveRunId] = useState('')
  const [runDetailPreset, setRunDetailPreset] = useState<'all' | 'battle_card_overlay'>('all')
  const [reconcileTriggering, setReconcileTriggering] = useState(false)
  const [reconcileMessage, setReconcileMessage] = useState<string | null>(null)
  const [reconcileError, setReconcileError] = useState<string | null>(null)

  const {
    data,
    loading,
    error,
    refresh,
    refreshing,
  } = useApiData<CostCoreData>(
    async () => {
      const numericDays = Number(days)
      const cacheOnly =
        cacheFilter === 'cached' ? true : cacheFilter === 'uncached' ? false : undefined
      const [summary, operationsResponse, recentResponse] = await Promise.all([
        fetchAdminCostSummary(numericDays),
        fetchAdminCostByOperation({
          days: numericDays,
          limit: 250,
          provider: providerFilter || undefined,
          model: modelFilter || undefined,
          span_name: spanFilter || undefined,
          status: statusFilter || undefined,
          source_name: sourceFilter || undefined,
          event_type: eventTypeFilter || undefined,
          entity_type: entityTypeFilter || undefined,
          cache_only: cacheOnly,
        }),
        fetchAdminCostRecent({
          days: numericDays,
          limit: 100,
          provider: providerFilter || undefined,
          model: modelFilter || undefined,
          span_name: spanFilter || undefined,
          status: statusFilter || undefined,
          source_name: sourceFilter || undefined,
          event_type: eventTypeFilter || undefined,
          entity_type: entityTypeFilter || undefined,
          cache_only: cacheOnly,
        }),
      ])
      return {
        summary,
        operations: operationsResponse.operations,
        recent: recentResponse.calls,
      }
    },
    [days, providerFilter, modelFilter, spanFilter, statusFilter, sourceFilter, eventTypeFilter, entityTypeFilter, cacheFilter],
  )

  const {
    data: vendors,
    loading: vendorsLoading,
    error: vendorsError,
    refresh: refreshVendors,
    refreshing: vendorsRefreshing,
  } = useApiData<AdminCostVendor[]>(
    async () => {
      const response = await fetchAdminCostByVendor({
        days: Number(days),
        limit: COST_VENDOR_LIMIT,
      })
      return response.vendors
    },
    [days],
  )

  const {
    data: b2bEfficiency,
    loading: b2bEfficiencyLoading,
    error: b2bEfficiencyError,
    refresh: refreshB2bEfficiency,
    refreshing: b2bEfficiencyRefreshing,
  } = useApiData<AdminCostB2bEfficiency>(
    () =>
      fetchAdminCostB2bEfficiency({
        days: Number(days),
        top_n: B2B_EFFICIENCY_TOP_N,
        run_limit: B2B_EFFICIENCY_RUN_LIMIT,
      }),
    [days],
  )
  const {
    data: burnDashboard,
    loading: burnDashboardLoading,
    error: burnDashboardError,
    refreshing: burnDashboardRefreshing,
  } = useApiData<AdminCostBurnDashboard>(
    () =>
      fetchAdminCostBurnDashboard({
        days: Number(days),
        top_n: BURN_DASHBOARD_TOP_N,
      }),
    [days],
  )
  const {
    data: genericReasoning,
    loading: genericReasoningLoading,
    error: genericReasoningError,
    refresh: refreshGenericReasoning,
    refreshing: genericReasoningRefreshing,
  } = useApiData<AdminCostGenericReasoning>(
    () =>
      fetchAdminCostGenericReasoning({
        days: Number(days),
        top_n: GENERIC_REASONING_TOP_N,
      }),
    [days],
  )
  const {
    data: reconciliation,
    loading: reconciliationLoading,
    error: reconciliationError,
    refresh: refreshReconciliation,
    refreshing: reconciliationRefreshing,
  } = useApiData<AdminCostReconciliation>(
    () => fetchAdminCostReconciliation(Number(days)),
    [days],
  )
  const {
    data: reasoningActivity,
    loading: reasoningActivityLoading,
    error: reasoningActivityError,
    refresh: refreshReasoningActivity,
    refreshing: reasoningActivityRefreshing,
  } = useApiData<AdminCostReasoningActivity>(
    () => fetchAdminCostReasoningActivity(Number(days)),
    [days],
  )

  const {
    data: cacheHealth,
    loading: cacheHealthLoading,
    error: cacheHealthError,
    refresh: refreshCacheHealth,
    refreshing: cacheHealthRefreshing,
  } = useApiData<AdminCostCacheHealth>(
    () => fetchAdminCostCacheHealth(Number(days), Number(cacheTopN)),
    [days, cacheTopN],
  )
  const {
    data: taskHealthRows,
    loading: taskHealthLoading,
    error: taskHealthError,
    refresh: refreshTaskHealth,
    refreshing: taskHealthRefreshing,
  } = useApiData<AdminTaskHealthRow[]>(
    async () => {
      const response = await fetchAdminTaskHealth(Number(days))
      return response.tasks
    },
    [days],
  )
  const {
    data: runDetail,
    loading: runDetailLoading,
    error: runDetailError,
    refresh: refreshRunDetail,
    refreshing: runDetailRefreshing,
  } = useApiData<AdminCostRunDetail | null>(
    () => (
      activeRunId.trim()
        ? fetchAdminCostRun(activeRunId.trim(), { call_limit: 25, event_limit: 15, attempt_limit: 15 })
        : Promise.resolve(null)
    ),
    [activeRunId],
  )

  const operations = data?.operations ?? []
  const vendorRows = vendors ?? []
  const vendorPasses = b2bEfficiency?.vendor_passes ?? []
  const sourceEfficiency = b2bEfficiency?.source_efficiency ?? []
  const recentPipelineRuns = b2bEfficiency?.recent_runs ?? []
  const burnRows = burnDashboard?.rows ?? []
  const burnBudgetRows = burnDashboard?.reasoning_budget_pressure.rows ?? []
  const genericReasoningSources = genericReasoning?.by_source ?? []
  const genericReasoningEvents = genericReasoning?.by_event_type ?? []
  const genericReasoningPairs = genericReasoning?.top_source_events ?? []
  const genericReasoningEntities = genericReasoning?.top_entities ?? []
  const recentCalls = data?.recent ?? []
  const openRunDetail = (runId: string, preset: 'all' | 'battle_card_overlay' = 'all') => {
    setRunIdInput(runId)
    setActiveRunId(runId)
    setRunDetailPreset(preset)
  }
  const runDetailCalls =
    runDetailPreset === 'battle_card_overlay'
      ? (runDetail?.calls ?? []).filter(
          (row) => row.source_name === 'b2b_battle_cards' && row.event_type === 'llm_overlay',
        )
      : (runDetail?.calls ?? [])
  const runDetailAttempts =
    runDetailPreset === 'battle_card_overlay'
      ? (runDetail?.artifact_attempts ?? []).filter((row) => row.artifact_type === 'battle_card')
      : (runDetail?.artifact_attempts ?? [])
  const runDetailEvents =
    runDetailPreset === 'battle_card_overlay'
      ? (runDetail?.visibility_events ?? []).filter((row) => row.entity_type === 'battle_card')
      : (runDetail?.visibility_events ?? [])
  const overlayCallCount = runDetailCalls.length
  const overlayCostUsd = runDetailCalls.reduce((sum, row) => sum + row.cost_usd, 0)
  const overlayBillableInputTokens = runDetailCalls.reduce(
    (sum, row) => sum + row.billable_input_tokens,
    0,
  )
  const overlayCachedTokens = runDetailCalls.reduce((sum, row) => sum + row.cached_tokens, 0)
  const overlayAttemptFailures = runDetailAttempts.filter(
    (row) => !['succeeded', 'cached'].includes(String(row.status || '').toLowerCase()),
  ).length
  const overlayEventWarnings = runDetailEvents.filter((row) =>
    ['warning', 'error', 'critical'].includes(String(row.severity || '').toLowerCase()),
  ).length
  const unifiedError =
    error ||
    vendorsError ||
    b2bEfficiencyError ||
    burnDashboardError ||
    genericReasoningError ||
    reconciliationError ||
    reasoningActivityError ||
    cacheHealthError ||
    taskHealthError
  const unifiedLoading =
    loading ||
    vendorsLoading ||
    b2bEfficiencyLoading ||
    burnDashboardLoading ||
    genericReasoningLoading ||
    reconciliationLoading ||
    reasoningActivityLoading ||
    cacheHealthLoading ||
    taskHealthLoading
  const unifiedRefreshing =
    refreshing ||
    vendorsRefreshing ||
    b2bEfficiencyRefreshing ||
    burnDashboardRefreshing ||
    genericReasoningRefreshing ||
    reconciliationRefreshing ||
    reasoningActivityRefreshing ||
    cacheHealthRefreshing ||
    taskHealthRefreshing
  const reconcilerTask = (taskHealthRows ?? []).find((row) => row.name === RECON_TASK_NAME) ?? null
  const providerOptions = Array.from(
    new Set(
      [...operations.map((row) => row.provider), ...recentCalls.map((row) => row.provider || '')]
        .map((value) => value.trim())
        .filter(Boolean),
    ),
  ).sort()
  const modelOptions = Array.from(
    new Set(
      [...operations.map((row) => row.model), ...recentCalls.map((row) => row.model || '')]
        .map((value) => value.trim())
        .filter(Boolean),
    ),
  ).sort()
  const spanOptions = Array.from(
    new Set(
      [...operations.map((row) => row.span_name), ...recentCalls.map((row) => row.span_name)]
        .map((value) => value.trim())
        .filter(Boolean),
    ),
  ).sort()
  const statusOptions = Array.from(
    new Set(recentCalls.map((row) => row.status).map((value) => value.trim()).filter(Boolean)),
  ).sort()
  const sourceOptions = Array.from(
    new Set(recentCalls.map((row) => row.source_name || '').map((value) => value.trim()).filter(Boolean)),
  ).sort()
  const eventTypeOptions = Array.from(
    new Set(recentCalls.map((row) => row.event_type || '').map((value) => value.trim()).filter(Boolean)),
  ).sort()
  const entityTypeOptions = Array.from(
    new Set(recentCalls.map((row) => row.entity_type || '').map((value) => value.trim()).filter(Boolean)),
  ).sort()

  const reconciliationColumns: Column<AdminCostReconciliationRow>[] = [
    {
      key: 'date',
      header: 'Day',
      render: (row) => <span className="text-sm text-white">{row.date}</span>,
      sortable: true,
      sortValue: (row) => row.date,
    },
    {
      key: 'provider',
      header: 'Provider',
      render: (row) => <span className="text-xs text-slate-300">{row.provider}</span>,
      sortable: true,
      sortValue: (row) => row.provider,
    },
    {
      key: 'status',
      header: 'Status',
      render: (row) => <StatusBadge status={row.status} />,
      sortable: true,
      sortValue: (row) => row.status,
    },
    {
      key: 'tracked_cost_usd',
      header: 'Tracked',
      render: (row) => <span className="text-xs font-medium text-cyan-400">{formatCurrency(row.tracked_cost_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.tracked_cost_usd,
    },
    {
      key: 'provider_cost_usd',
      header: 'Provider',
      render: (row) => <span className="text-xs text-slate-300">{formatMaybeCurrency(row.provider_cost_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.provider_cost_usd ?? -1,
    },
    {
      key: 'delta_cost_usd',
      header: 'Delta',
      render: (row) => <span className="text-xs text-amber-300">{formatMaybeCurrency(row.delta_cost_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.delta_cost_usd ?? -1,
    },
    {
      key: 'calls',
      header: 'Calls',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.calls)}</span>,
      sortable: true,
      sortValue: (row) => row.calls,
    },
  ]

  const genericReasoningSourceColumns: Column<AdminCostGenericReasoningSourceRow>[] = [
    {
      key: 'source_name',
      header: 'Source',
      render: (row) => <span className="text-sm text-white">{row.source_name}</span>,
      sortable: true,
      sortValue: (row) => row.source_name,
    },
    {
      key: 'calls',
      header: 'Calls',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.calls)}</span>,
      sortable: true,
      sortValue: (row) => row.calls,
    },
    {
      key: 'cost_usd',
      header: 'Cost',
      render: (row) => <span className="text-xs font-medium text-cyan-400">{formatCurrency(row.cost_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.cost_usd,
    },
    {
      key: 'billable_input_tokens',
      header: 'Billable In',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.billable_input_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.billable_input_tokens,
    },
    {
      key: 'output_tokens',
      header: 'Output',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.output_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.output_tokens,
    },
  ]

  const genericReasoningEventColumns: Column<AdminCostGenericReasoningEventRow>[] = [
    {
      key: 'event_type',
      header: 'Event Type',
      render: (row) => <span className="text-sm text-white">{row.event_type}</span>,
      sortable: true,
      sortValue: (row) => row.event_type,
    },
    {
      key: 'calls',
      header: 'Calls',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.calls)}</span>,
      sortable: true,
      sortValue: (row) => row.calls,
    },
    {
      key: 'cost_usd',
      header: 'Cost',
      render: (row) => <span className="text-xs font-medium text-cyan-400">{formatCurrency(row.cost_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.cost_usd,
    },
    {
      key: 'billable_input_tokens',
      header: 'Billable In',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.billable_input_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.billable_input_tokens,
    },
    {
      key: 'output_tokens',
      header: 'Output',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.output_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.output_tokens,
    },
  ]

  const genericReasoningPairColumns: Column<AdminCostGenericReasoningSourceEventRow>[] = [
    {
      key: 'pair',
      header: 'Source / Event',
      render: (row) => (
        <div className="max-w-[280px]">
          <p className="truncate text-sm text-white">{row.source_name}</p>
          <p className="truncate text-xs text-slate-500">{row.event_type}</p>
        </div>
      ),
      sortable: true,
      sortValue: (row) => `${row.source_name}:${row.event_type}`,
    },
    {
      key: 'calls',
      header: 'Calls',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.calls)}</span>,
      sortable: true,
      sortValue: (row) => row.calls,
    },
    {
      key: 'cost_usd',
      header: 'Cost',
      render: (row) => <span className="text-xs font-medium text-cyan-400">{formatCurrency(row.cost_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.cost_usd,
    },
    {
      key: 'billable_input_tokens',
      header: 'Billable In',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.billable_input_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.billable_input_tokens,
    },
    {
      key: 'output_tokens',
      header: 'Output',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.output_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.output_tokens,
    },
  ]

  const genericReasoningEntityColumns: Column<AdminCostGenericReasoningEntityRow>[] = [
    {
      key: 'entity',
      header: 'Entity',
      render: (row) => (
        <div className="max-w-[280px]">
          <p className="truncate text-sm text-white">{truncateLabel(row.entity_id, 40)}</p>
          <p className="truncate text-xs text-slate-500">{row.entity_type}</p>
        </div>
      ),
      sortable: true,
      sortValue: (row) => `${row.entity_type}:${row.entity_id}`,
    },
    {
      key: 'calls',
      header: 'Calls',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.calls)}</span>,
      sortable: true,
      sortValue: (row) => row.calls,
    },
    {
      key: 'cost_usd',
      header: 'Cost',
      render: (row) => <span className="text-xs font-medium text-cyan-400">{formatCurrency(row.cost_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.cost_usd,
    },
    {
      key: 'billable_input_tokens',
      header: 'Billable In',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.billable_input_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.billable_input_tokens,
    },
  ]

  const stratifiedReasoningColumns: Column<AdminCostReasoningActivityPhase>[] = [
    {
      key: 'span_name',
      header: 'Phase',
      render: (row) => (
        <div className="max-w-[260px]">
          <p className="truncate text-sm text-white">{row.span_name}</p>
          <p className="truncate text-xs text-slate-500">{row.pass_type} pass {row.pass_number}</p>
        </div>
      ),
      sortable: true,
      sortValue: (row) => `${row.pass_number}:${row.span_name}`,
    },
    {
      key: 'calls',
      header: 'Calls',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.calls)}</span>,
      sortable: true,
      sortValue: (row) => row.calls,
    },
    {
      key: 'cost_usd',
      header: 'Cost',
      render: (row) => <span className="text-xs font-medium text-cyan-400">{formatCurrency(row.cost_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.cost_usd,
    },
    {
      key: 'total_tokens',
      header: 'Tokens',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.total_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.total_tokens,
    },
    {
      key: 'changed_count',
      header: 'Changed',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.changed_count)}</span>,
      sortable: true,
      sortValue: (row) => row.changed_count,
    },
  ]

  const exactStageColumns: Column<AdminCostExactCacheStage>[] = [
    {
      key: 'stage_id',
      header: 'Atlas Exact Stage',
      render: (row) => (
        <div className="max-w-[280px]">
          <p className="truncate text-sm text-white">{row.stage_id}</p>
          <p className="truncate text-xs text-slate-500">{truncateLabel(row.namespace || row.file_path, 48)}</p>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.stage_id,
    },
    {
      key: 'rows',
      header: 'Rows',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.rows)}</span>,
      sortable: true,
      sortValue: (row) => row.rows,
    },
    {
      key: 'total_hits',
      header: 'Hits',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.total_hits)}</span>,
      sortable: true,
      sortValue: (row) => row.total_hits,
    },
    {
      key: 'writes_in_window',
      header: `Writes ${days}d`,
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.writes_in_window)}</span>,
      sortable: true,
      sortValue: (row) => row.writes_in_window,
    },
    {
      key: 'rows_hit_in_window',
      header: `Rows Hit ${days}d`,
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.rows_hit_in_window)}</span>,
      sortable: true,
      sortValue: (row) => row.rows_hit_in_window,
    },
    {
      key: 'last_hit_at',
      header: 'Last Hit',
      render: (row) => <span className="text-xs text-slate-400">{formatTs(row.last_hit_at)}</span>,
      sortable: true,
      sortValue: (row) => row.last_hit_at || '',
    },
  ]

  const taskReuseColumns: Column<AdminCostTaskReuseRow>[] = [
    {
      key: 'task_name',
      header: 'Task',
      render: (row) => <span className="text-sm text-white">{row.task_name}</span>,
      sortable: true,
      sortValue: (row) => row.task_name,
    },
    {
      key: 'executions',
      header: 'Runs',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.executions)}</span>,
      sortable: true,
      sortValue: (row) => row.executions,
    },
    {
      key: 'reused',
      header: 'Reused',
      render: (row) => <span className="text-xs font-medium text-cyan-400">{formatNumber(row.reused)}</span>,
      sortable: true,
      sortValue: (row) => row.reused,
    },
    {
      key: 'exact_cache_hits',
      header: 'Exact Hits',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.exact_cache_hits)}</span>,
      sortable: true,
      sortValue: (row) => row.exact_cache_hits,
    },
    {
      key: 'semantic_cache_hits',
      header: 'Semantic Hits',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.semantic_cache_hits)}</span>,
      sortable: true,
      sortValue: (row) => row.semantic_cache_hits,
    },
    {
      key: 'evidence_hash_reuse',
      header: 'Hash Reuse',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.evidence_hash_reuse)}</span>,
      sortable: true,
      sortValue: (row) => row.evidence_hash_reuse,
    },
    {
      key: 'generated',
      header: 'Generated',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.generated)}</span>,
      sortable: true,
      sortValue: (row) => row.generated,
    },
    {
      key: 'overlay_failures',
      header: 'Failures',
      render: (row) => <span className="text-xs text-amber-300">{formatNumber(row.overlay_failures)}</span>,
      sortable: true,
      sortValue: (row) => row.overlay_failures,
    },
  ]

  const promptCacheColumns: Column<AdminCostPromptCacheSpan>[] = [
    {
      key: 'span_name',
      header: 'Provider Prompt Cache Leader',
      render: (row) => <span className="text-sm text-white">{truncateLabel(row.span_name, 52)}</span>,
      sortable: true,
      sortValue: (row) => row.span_name,
    },
    {
      key: 'calls',
      header: 'Calls',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.calls)}</span>,
      sortable: true,
      sortValue: (row) => row.calls,
    },
    {
      key: 'cache_hit_calls',
      header: 'Read Calls',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.cache_hit_calls)}</span>,
      sortable: true,
      sortValue: (row) => row.cache_hit_calls,
    },
    {
      key: 'cache_write_calls',
      header: 'Write Calls',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.cache_write_calls)}</span>,
      sortable: true,
      sortValue: (row) => row.cache_write_calls,
    },
    {
      key: 'cached_tokens',
      header: 'Read Tokens',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.cached_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.cached_tokens,
    },
    {
      key: 'cache_write_tokens',
      header: 'Write Tokens',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.cache_write_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.cache_write_tokens,
    },
  ]

  const batchStageColumns: Column<AdminCostBatchStage>[] = [
    {
      key: 'stage_id',
      header: 'Batch Stage',
      render: (row) => (
        <div className="max-w-[280px]">
          <p className="truncate text-sm text-white">{row.stage_id}</p>
          <p className="truncate text-xs text-slate-500">{truncateLabel(row.task_name, 48)}</p>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.stage_id,
    },
    {
      key: 'submitted_jobs',
      header: 'Jobs',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.submitted_jobs)}</span>,
      sortable: true,
      sortValue: (row) => row.submitted_jobs,
    },
    {
      key: 'submitted_items',
      header: 'Items',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.submitted_items)}</span>,
      sortable: true,
      sortValue: (row) => row.submitted_items,
    },
    {
      key: 'cache_prefiltered_items',
      header: 'Prefiltered',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.cache_prefiltered_items)}</span>,
      sortable: true,
      sortValue: (row) => row.cache_prefiltered_items,
    },
    {
      key: 'fallback_single_call_items',
      header: 'Fallback Single',
      render: (row) => <span className="text-xs text-amber-300">{formatNumber(row.fallback_single_call_items)}</span>,
      sortable: true,
      sortValue: (row) => row.fallback_single_call_items,
    },
    {
      key: 'estimated_savings_usd',
      header: 'Est Savings',
      render: (row) => <span className="text-xs font-medium text-cyan-400">{formatCurrency(row.estimated_savings_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.estimated_savings_usd,
    },
    {
      key: 'last_completed_at',
      header: 'Last Complete',
      render: (row) => <span className="text-xs text-slate-400">{formatTs(row.last_completed_at)}</span>,
      sortable: true,
      sortValue: (row) => row.last_completed_at || '',
    },
  ]

  const staleBatchColumns: Column<AdminCostStaleBatchJob>[] = [
    {
      key: 'submitted_at',
      header: 'Age',
      render: (row) => (
        <div className="space-y-1">
          <p className="text-xs font-medium text-amber-300">{formatAgeMinutes(row.stale_minutes)}</p>
          <p className="text-[11px] text-slate-500">{formatTs(row.submitted_at || row.created_at)}</p>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.stale_minutes,
    },
    {
      key: 'stage_id',
      header: 'Batch',
      render: (row) => (
        <div className="max-w-[260px]">
          <p className="truncate text-sm text-white">{row.task_name}</p>
          <p className="truncate text-xs text-slate-500">{row.stage_id}</p>
          {row.run_id ? <p className="truncate text-[11px] text-slate-600">{row.run_id}</p> : null}
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.task_name,
    },
    {
      key: 'status',
      header: 'Status',
      render: (row) => <StatusBadge status={row.status} />,
      sortable: true,
      sortValue: (row) => row.status,
    },
    {
      key: 'submitted_items',
      header: 'Items',
      render: (row) => (
        <div className="space-y-1 text-xs text-slate-300">
          <p>{formatNumber(row.submitted_items)} submitted</p>
          <p className="text-slate-500">{formatNumber(row.completed_items)} done / {formatNumber(row.failed_items)} failed</p>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.submitted_items,
    },
    {
      key: 'provider_batch_id',
      header: 'Provider Ref',
      render: (row) => (
        <div className="max-w-[220px]">
          <p className="truncate text-xs text-slate-400">{truncateLabel(row.provider_batch_id, 30)}</p>
          {row.provider_error ? <p className="truncate text-[11px] text-rose-300">{truncateLabel(row.provider_error, 40)}</p> : null}
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.provider_batch_id,
    },
  ]

  const staleClaimColumns: Column<AdminCostStaleBatchClaim>[] = [
    {
      key: 'applying_at',
      header: 'Claim Age',
      render: (row) => (
        <div className="space-y-1">
          <p className="text-xs font-medium text-amber-300">{formatAgeMinutes(row.stale_minutes)}</p>
          <p className="text-[11px] text-slate-500">{formatTs(row.applying_at)}</p>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.stale_minutes,
    },
    {
      key: 'custom_id',
      header: 'Item',
      render: (row) => (
        <div className="max-w-[280px]">
          <p className="truncate text-sm text-white">{truncateLabel(row.custom_id, 42)}</p>
          {row.artifact_id ? <p className="truncate text-xs text-slate-500">{truncateLabel(row.artifact_id, 48)}</p> : null}
          <p className="truncate text-[11px] text-slate-600">{row.task_name}</p>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.custom_id,
    },
    {
      key: 'status',
      header: 'Status',
      render: (row) => <StatusBadge status={row.status} />,
      sortable: true,
      sortValue: (row) => row.status,
    },
    {
      key: 'applying_by',
      header: 'Claimed By',
      render: (row) => (
        <div className="max-w-[240px]">
          <p className="truncate text-xs text-slate-300">{truncateLabel(row.applying_by || 'unknown', 36)}</p>
          {row.run_id ? <p className="truncate text-[11px] text-slate-600">{row.run_id}</p> : null}
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.applying_by || '',
    },
    {
      key: 'provider_batch_id',
      header: 'Provider Ref',
      render: (row) => <span className="text-xs text-slate-400">{truncateLabel(row.provider_batch_id || '--', 30)}</span>,
      sortable: true,
      sortValue: (row) => row.provider_batch_id || '',
    },
  ]

  async function handleRunReconciler() {
    if (!reconcilerTask) {
      setReconcileError('Reconciliation task not found')
      setReconcileMessage(null)
      return
    }
    setReconcileTriggering(true)
    setReconcileError(null)
    setReconcileMessage(null)
    try {
      const result = await runAutonomousTask(reconcilerTask.id)
      setReconcileMessage(result.message || `Triggered ${RECON_TASK_NAME}`)
      refreshTaskHealth()
      refreshCacheHealth()
    } catch (err) {
      setReconcileError(err instanceof Error ? err.message : String(err))
    } finally {
      setReconcileTriggering(false)
    }
  }

  const semanticColumns: Column<AdminCostSemanticPatternClass>[] = [
    {
      key: 'pattern_class',
      header: 'Semantic Pattern',
      render: (row) => <span className="text-sm text-white">{row.pattern_class}</span>,
      sortable: true,
      sortValue: (row) => row.pattern_class,
    },
    {
      key: 'active_entries',
      header: 'Active',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.active_entries)}</span>,
      sortable: true,
      sortValue: (row) => row.active_entries,
    },
    {
      key: 'recent_validations',
      header: `Validated ${days}d`,
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.recent_validations)}</span>,
      sortable: true,
      sortValue: (row) => row.recent_validations,
    },
  ]

  const operationColumns: Column<AdminCostOperation>[] = [
    {
      key: 'span_name',
      header: 'Operation',
      render: (row) => (
        <div className="max-w-[260px]">
          <p className="truncate text-sm text-white">{row.span_name}</p>
          <p className="truncate text-xs text-slate-500">{row.provider} | {truncateLabel(row.model, 48)}</p>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.span_name,
    },
    {
      key: 'input_tokens',
      header: 'Raw In',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.input_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.input_tokens,
    },
    {
      key: 'calls',
      header: 'Calls',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.calls)}</span>,
      sortable: true,
      sortValue: (row) => row.calls,
    },
    {
      key: 'cost',
      header: 'Cost',
      render: (row) => <span className="text-xs font-medium text-cyan-400">{formatCurrency(row.cost_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.cost_usd,
    },
    {
      key: 'billable_input_tokens',
      header: 'Billable In',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.billable_input_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.billable_input_tokens,
    },
    {
      key: 'cached_tokens',
      header: 'Cached Read',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.cached_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.cached_tokens,
    },
    {
      key: 'cache_write_tokens',
      header: 'Cache Write',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.cache_write_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.cache_write_tokens,
    },
    {
      key: 'output_tokens',
      header: 'Output',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.output_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.output_tokens,
    },
    {
      key: 'avg_duration_ms',
      header: 'Avg ms',
      render: (row) => <span className="text-xs text-slate-400">{formatNumber(Math.round(row.avg_duration_ms))}</span>,
      sortable: true,
      sortValue: (row) => row.avg_duration_ms,
    },
  ]

  const vendorColumns: Column<AdminCostVendor>[] = [
    {
      key: 'vendor_name',
      header: 'Vendor',
      render: (row) => <span className="text-sm text-white">{row.vendor_name}</span>,
      sortable: true,
      sortValue: (row) => row.vendor_name,
    },
    {
      key: 'calls',
      header: 'Calls',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.calls)}</span>,
      sortable: true,
      sortValue: (row) => row.calls,
    },
    {
      key: 'cost',
      header: 'Cost',
      render: (row) => <span className="text-xs font-medium text-cyan-400">{formatCurrency(row.cost_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.cost_usd,
    },
    {
      key: 'input_tokens',
      header: 'Raw In',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.input_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.input_tokens,
    },
    {
      key: 'billable_input_tokens',
      header: 'Billable In',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.billable_input_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.billable_input_tokens,
    },
    {
      key: 'output_tokens',
      header: 'Output',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.output_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.output_tokens,
    },
    {
      key: 'avg_duration_ms',
      header: 'Avg ms',
      render: (row) => <span className="text-xs text-slate-400">{formatNumber(Math.round(row.avg_duration_ms))}</span>,
      sortable: true,
      sortValue: (row) => row.avg_duration_ms,
    },
  ]

  const vendorPassColumns: Column<AdminCostVendorPassRow>[] = [
    {
      key: 'vendor_name',
      header: 'Vendor',
      render: (row) => <span className="text-sm text-white">{row.vendor_name}</span>,
      sortable: true,
      sortValue: (row) => row.vendor_name,
    },
    {
      key: 'total_cost_usd',
      header: 'Total Cost',
      render: (row) => <span className="text-xs font-medium text-cyan-400">{formatCurrency(row.total_cost_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.total_cost_usd,
    },
    {
      key: 'extraction_cost_usd',
      header: 'Extraction',
      render: (row) => (
        <div className="text-xs">
          <div className="text-slate-300">{formatCurrency(row.extraction_cost_usd)}</div>
          <div className="text-slate-500">{formatNumber(row.extraction_calls)} calls</div>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.extraction_cost_usd,
    },
    {
      key: 'repair_cost_usd',
      header: 'Repair',
      render: (row) => (
        <div className="text-xs">
          <div className="text-slate-300">{formatCurrency(row.repair_cost_usd)}</div>
          <div className="text-slate-500">{formatNumber(row.repair_calls)} calls</div>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.repair_cost_usd,
    },
    {
      key: 'reasoning_cost_usd',
      header: 'Reasoning',
      render: (row) => (
        <div className="text-xs">
          <div className="text-slate-300">{formatCurrency(row.reasoning_cost_usd)}</div>
          <div className="text-slate-500">{formatNumber(row.reasoning_calls)} calls</div>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.reasoning_cost_usd,
    },
    {
      key: 'battle_card_overlay_cost_usd',
      header: 'Battle Cards',
      render: (row) => (
        <div className="text-xs">
          <div className="text-slate-300">{formatCurrency(row.battle_card_overlay_cost_usd)}</div>
          <div className="text-slate-500">{formatNumber(row.battle_card_overlay_calls)} calls</div>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.battle_card_overlay_cost_usd,
    },
  ]

  const sourceEfficiencyColumns: Column<AdminCostSourceEfficiencyRow>[] = [
    {
      key: 'source',
      header: 'Source',
      render: (row) => <span className="text-sm text-white">{row.source}</span>,
      sortable: true,
      sortValue: (row) => row.source,
    },
    {
      key: 'total_cost_usd',
      header: 'Total Cost',
      render: (row) => <span className="text-xs font-medium text-cyan-400">{formatCurrency(row.total_cost_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.total_cost_usd,
    },
    {
      key: 'witness_yield_rate',
      header: 'Yield / Review',
      render: (row) => <span className="text-xs text-slate-300">{row.witness_yield_rate.toFixed(2)}</span>,
      sortable: true,
      sortValue: (row) => row.witness_yield_rate,
    },
    {
      key: 'repair_trigger_rate',
      header: 'Repair Trigger',
      render: (row) => <span className="text-xs text-amber-300">{(row.repair_trigger_rate * 100).toFixed(1)}%</span>,
      sortable: true,
      sortValue: (row) => row.repair_trigger_rate,
    },
    {
      key: 'span_count',
      header: 'Spans',
      render: (row) => <span className="text-xs text-cyan-300">{formatNumber(row.span_count)}</span>,
      sortable: true,
      sortValue: (row) => row.span_count,
    },
    {
      key: 'cost_per_witness_usd',
      header: 'Cost / Witness',
      render: (row) => <span className="text-xs text-slate-300">{formatMaybeCurrency(row.cost_per_witness_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.cost_per_witness_usd ?? -1,
    },
    {
      key: 'strict_discussion_candidates_kept_rows',
      header: 'Gate Kept',
      render: (row) => <span className="text-xs text-cyan-300">{formatNumber(row.strict_discussion_candidates_kept_rows)}</span>,
      sortable: true,
      sortValue: (row) => row.strict_discussion_candidates_kept_rows,
    },
    {
      key: 'low_signal_discussion_skipped_rows',
      header: 'Gate Skipped',
      render: (row) => <span className="text-xs text-amber-300">{formatNumber(row.low_signal_discussion_skipped_rows)}</span>,
      sortable: true,
      sortValue: (row) => row.low_signal_discussion_skipped_rows,
    },
  ]

  const b2bRunColumns: Column<AdminCostB2bRunRow>[] = [
    {
      key: 'started_at',
      header: 'Run',
      render: (row) => (
        <div className="max-w-[240px]">
          <p className="truncate text-sm text-white">{row.task_name}</p>
          <p className="truncate text-xs text-slate-500">{formatTs(row.started_at)}</p>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.started_at || '',
    },
    {
      key: 'total_cost_usd',
      header: 'Total Cost',
      render: (row) => <span className="text-xs font-medium text-cyan-400">{formatCurrency(row.total_cost_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.total_cost_usd,
    },
    {
      key: 'pass_costs',
      header: 'Pass Split',
      render: (row) => (
          <div className="space-y-0.5 text-[11px] text-slate-400">
          <div>Ext {formatMaybeCurrency(row.extraction_cost_usd)}</div>
          <div>Rep {formatMaybeCurrency(row.repair_cost_usd)}</div>
          <div>Reas {formatMaybeCurrency(row.reasoning_cost_usd)}</div>
          <div>Card {formatMaybeCurrency(row.battle_card_overlay_cost_usd)}</div>
        </div>
      ),
      sortable: true,
      sortValue: (row) =>
        row.extraction_cost_usd
        + row.repair_cost_usd
        + row.reasoning_cost_usd
        + row.battle_card_overlay_cost_usd,
    },
    {
      key: 'reviews_processed',
      header: 'Processed',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.reviews_processed)}</span>,
      sortable: true,
      sortValue: (row) => row.reviews_processed,
    },
    {
      key: 'witness_count',
      header: 'Spans',
      render: (row) => <span className="text-xs text-cyan-300">{formatNumber(row.witness_count)}</span>,
      sortable: true,
      sortValue: (row) => row.witness_count,
    },
    {
      key: 'cost_per_witness_usd',
      header: 'Cost / Witness',
      render: (row) => <span className="text-xs text-slate-300">{formatMaybeCurrency(row.cost_per_witness_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.cost_per_witness_usd ?? -1,
    },
    {
      key: 'secondary_write_hits',
      header: 'Secondary Writes',
      render: (row) => <span className="text-xs text-amber-300">{formatNumber(row.secondary_write_hits)}</span>,
      sortable: true,
      sortValue: (row) => row.secondary_write_hits,
    },
    {
      key: 'strict_discussion_candidates_kept',
      header: 'Gate Kept',
      render: (row) => <span className="text-xs text-cyan-300">{formatNumber(row.strict_discussion_candidates_kept)}</span>,
      sortable: true,
      sortValue: (row) => row.strict_discussion_candidates_kept,
    },
    {
      key: 'strict_discussion_candidates_dropped',
      header: 'Gate Dropped',
      render: (row) => <span className="text-xs text-amber-300">{formatNumber(row.strict_discussion_candidates_dropped)}</span>,
      sortable: true,
      sortValue: (row) => row.strict_discussion_candidates_dropped,
    },
    {
      key: 'battle_card_overlay_calls',
      header: 'Overlay',
      render: (row) => (
        <div className="space-y-0.5 text-[11px] text-slate-400">
          <div>{formatNumber(row.battle_card_overlay_calls)} calls</div>
          <div>{formatCurrency(row.battle_card_overlay_cost_usd)}</div>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.battle_card_overlay_cost_usd,
    },
    {
      key: 'battle_card_outcomes',
      header: 'Card Outcome',
      render: (row) => (
        <div className="space-y-0.5 text-[11px] text-slate-400">
          <div>Updated {formatNumber(row.battle_card_llm_updated)}</div>
          <div>Cached {formatNumber(row.battle_card_cache_hits)}</div>
          <div>Failed {formatNumber(row.battle_card_llm_failures)}</div>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.battle_card_llm_failures,
    },
    {
      key: 'actions',
      header: 'Inspect',
      render: (row) =>
        row.run_id ? (
          <div className="flex flex-col gap-1">
            <button
              onClick={() => openRunDetail(row.run_id)}
              className="rounded border border-slate-700/60 px-2 py-1 text-xs text-slate-300 transition hover:border-cyan-500/60 hover:text-white"
            >
              Run
            </button>
            {(row.battle_card_overlay_calls > 0 || row.battle_card_llm_failures > 0) && (
              <button
                onClick={() => openRunDetail(row.run_id, 'battle_card_overlay')}
                className="rounded border border-cyan-500/30 bg-cyan-500/10 px-2 py-1 text-xs text-cyan-200 transition hover:border-cyan-400/50 hover:text-white"
              >
                Overlay
              </button>
            )}
          </div>
        ) : (
          <span className="text-xs text-slate-500">--</span>
        ),
      sortable: false,
    },
  ]

  const burnColumns: Column<AdminCostBurnRow>[] = [
    {
      key: 'task_name',
      header: 'Task',
      render: (row) => (
        <div className="max-w-[240px]">
          <p className="truncate text-sm text-white">{row.task_name}</p>
          <p className="truncate text-xs text-slate-500">{formatTs(row.last_run_at)}</p>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.task_name,
    },
    {
      key: 'inspect',
      header: 'Inspect',
      render: (row) =>
        row.run_id ? (
          <button
            onClick={() =>
              openRunDetail(
                row.run_id || '',
                row.task_name === 'b2b_battle_cards' ? 'battle_card_overlay' : 'all',
              )
            }
            className={clsx(
              'rounded border px-2 py-1 text-xs transition',
              row.task_name === 'b2b_battle_cards'
                ? 'border-cyan-500/30 bg-cyan-500/10 text-cyan-200 hover:border-cyan-400/50 hover:text-white'
                : 'border-slate-700/60 text-slate-300 hover:border-cyan-500/60 hover:text-white',
            )}
          >
            {row.task_name === 'b2b_battle_cards' ? 'Overlay' : 'Run'}
          </button>
        ) : (
          <span className="text-xs text-slate-500">--</span>
        ),
      sortable: false,
    },
    {
      key: 'recent_runs',
      header: 'Runs',
      render: (row) => <span className="text-xs text-slate-300">{formatMaybeNumber(row.recent_runs)}</span>,
      sortable: true,
      sortValue: (row) => row.recent_runs ?? -1,
    },
    {
      key: 'last_status',
      header: 'Status',
      render: (row) =>
        row.last_status ? <StatusBadge status={row.last_status} /> : <span className="text-xs text-slate-500">--</span>,
      sortable: true,
      sortValue: (row) => row.last_status || '',
    },
    {
      key: 'model_call_count',
      header: 'Model Calls',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.model_call_count)}</span>,
      sortable: true,
      sortValue: (row) => row.model_call_count,
    },
    {
      key: 'total_billable_input_tokens',
      header: 'Billable In',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.total_billable_input_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.total_billable_input_tokens,
    },
    {
      key: 'total_output_tokens',
      header: 'Out',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.total_output_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.total_output_tokens,
    },
    {
      key: 'total_cost_usd',
      header: 'Cost',
      render: (row) => <span className="text-xs font-medium text-cyan-400">{formatCurrency(row.total_cost_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.total_cost_usd,
    },
    {
      key: 'rows_processed',
      header: 'Processed',
      render: (row) => <span className="text-xs text-slate-300">{formatMaybeNumber(row.rows_processed)}</span>,
      sortable: true,
      sortValue: (row) => row.rows_processed ?? -1,
    },
    {
      key: 'rows_skipped',
      header: 'Skipped',
      render: (row) => <span className="text-xs text-amber-300">{formatMaybeNumber(row.rows_skipped)}</span>,
      sortable: true,
      sortValue: (row) => row.rows_skipped ?? -1,
    },
    {
      key: 'rows_reprocessed',
      header: 'Reprocessed',
      render: (row) => <span className="text-xs text-amber-300">{formatMaybeNumber(row.rows_reprocessed)}</span>,
      sortable: true,
      sortValue: (row) => row.rows_reprocessed ?? -1,
    },
    {
      key: 'retry_count',
      header: 'Retries',
      render: (row) => <span className="text-xs text-slate-300">{formatMaybeNumber(row.retry_count)}</span>,
      sortable: true,
      sortValue: (row) => row.retry_count ?? -1,
    },
    {
      key: 'avg_cost_per_successful_item',
      header: 'Cost / Success',
      render: (row) => <span className="text-xs text-slate-300">{formatMaybeCurrency(row.avg_cost_per_successful_item)}</span>,
      sortable: true,
      sortValue: (row) => row.avg_cost_per_successful_item ?? -1,
    },
    {
      key: 'reprocess_pct',
      header: 'Reprocess %',
      render: (row) => <span className="text-xs text-slate-300">{formatMaybePercent(row.reprocess_pct)}</span>,
      sortable: true,
      sortValue: (row) => row.reprocess_pct ?? -1,
    },
    {
      key: 'top_trigger_reason',
      header: 'Top Trigger',
      render: (row) => <span className="text-xs text-slate-400">{truncateLabel(row.top_trigger_reason, 36)}</span>,
      sortable: true,
      sortValue: (row) => row.top_trigger_reason,
    },
  ]

  const burnBudgetColumns: Column<AdminCostBurnBudgetRow>[] = [
    {
      key: 'artifact_label',
      header: 'Scope',
      render: (row) => (
        <div className="max-w-[280px]">
          <p className="truncate text-sm text-white">{row.artifact_label}</p>
          <p className="truncate text-xs text-slate-500">{truncateLabel(row.artifact_id, 52)}</p>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.artifact_label,
    },
    {
      key: 'rejected_at',
      header: 'Rejected',
      render: (row) => <span className="text-xs text-slate-400">{formatTs(row.rejected_at)}</span>,
      sortable: true,
      sortValue: (row) => row.rejected_at || '',
    },
    {
      key: 'estimated_input_tokens',
      header: 'Estimated In',
      render: (row) => <span className="text-xs text-slate-300">{formatMaybeNumber(row.estimated_input_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.estimated_input_tokens ?? -1,
    },
    {
      key: 'cap',
      header: 'Cap',
      render: (row) => <span className="text-xs text-amber-300">{formatMaybeNumber(row.cap)}</span>,
      sortable: true,
      sortValue: (row) => row.cap ?? -1,
    },
    {
      key: 'error_message',
      header: 'Reason',
      render: (row) => <span className="text-xs text-slate-400">{truncateLabel(row.error_message || '--', 56)}</span>,
      sortable: true,
      sortValue: (row) => row.error_message || '',
    },
  ]

  const recentColumns: Column<AdminCostRecentCall>[] = [
    {
      key: 'created_at',
      header: 'Time',
      render: (row) => <span className="text-xs text-slate-400">{formatTs(row.created_at)}</span>,
      sortable: true,
      sortValue: (row) => row.created_at || '',
    },
    {
      key: 'span_name',
      header: 'Call',
      render: (row) => (
        <div className="max-w-[280px]">
          <p className="truncate text-sm text-white">{row.title}</p>
          <p className="truncate text-xs text-slate-500">
            {[row.provider, row.model, row.detail].filter(Boolean).map((value) => truncateLabel(value as string, 44)).join(' | ') || row.span_name}
          </p>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.span_name,
    },
    {
      key: 'input_tokens',
      header: 'Raw In',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.input_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.input_tokens,
    },
    {
      key: 'context',
      header: 'Context',
      render: (row) => (
        <div className="max-w-[220px]">
          <p className="truncate text-sm text-white">{row.vendor_name || row.entity_id || '--'}</p>
          <p className="truncate text-xs text-slate-500">
            {[row.source_name, row.event_type, row.entity_type].filter(Boolean).join(' | ') || '--'}
          </p>
        </div>
      ),
      sortable: true,
      sortValue: (row) => `${row.vendor_name || row.entity_id || ''}:${row.source_name || ''}:${row.event_type || ''}:${row.entity_type || ''}`,
    },
    {
      key: 'cache_mode',
      header: 'Cache',
      render: (row) => (
        <span className="text-xs text-slate-300">
          {row.cache_write ? 'Write' : row.cache_hit ? 'Read' : 'None'}
        </span>
      ),
      sortable: true,
      sortValue: (row) => (row.cache_write ? 2 : row.cache_hit ? 1 : 0),
    },
    {
      key: 'cost',
      header: 'Cost',
      render: (row) => <span className="text-xs font-medium text-cyan-400">{formatCurrency(row.cost_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.cost_usd,
    },
    {
      key: 'billable_input_tokens',
      header: 'Billable In',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.billable_input_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.billable_input_tokens,
    },
    {
      key: 'cached_tokens',
      header: 'Cached Read',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.cached_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.cached_tokens,
    },
    {
      key: 'cache_write_tokens',
      header: 'Cache Write',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.cache_write_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.cache_write_tokens,
    },
    {
      key: 'output_tokens',
      header: 'Output',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.output_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.output_tokens,
    },
    {
      key: 'duration_ms',
      header: 'ms',
      render: (row) => <span className="text-xs text-slate-400">{formatNumber(row.duration_ms)}</span>,
      sortable: true,
      sortValue: (row) => row.duration_ms || 0,
    },
    {
      key: 'provider_request_id',
      header: 'Request',
      render: (row) => (
        <span className="max-w-[160px] inline-block truncate text-xs text-slate-500">
          {row.provider_request_id || '--'}
        </span>
      ),
    },
    {
      key: 'run_id',
      header: 'Run',
      render: (row) =>
        row.run_id ? (
          <button
            onClick={() => openRunDetail(row.run_id || '')}
            className="rounded border border-slate-700/60 px-2 py-1 text-xs text-slate-300 transition hover:border-cyan-500/60 hover:text-white"
          >
            Open
          </button>
        ) : (
          <span className="text-xs text-slate-500">--</span>
        ),
      sortable: true,
      sortValue: (row) => row.run_id || '',
    },
  ]

  const runAttemptColumns: Column<ArtifactAttempt>[] = [
    {
      key: 'stage',
      header: 'Stage',
      render: (row) => <StageBadge stage={row.stage} />,
      sortable: true,
      sortValue: (row) => row.stage,
    },
    {
      key: 'status',
      header: 'Status',
      render: (row) => <StatusBadge status={row.status} />,
      sortable: true,
      sortValue: (row) => row.status,
    },
    {
      key: 'artifact_type',
      header: 'Artifact',
      render: (row) => (
        <div className="max-w-[220px]">
          <p className="truncate text-sm text-white">{row.artifact_type}</p>
          <p className="truncate text-xs text-slate-500">{row.artifact_id || '--'}</p>
        </div>
      ),
      sortable: true,
      sortValue: (row) => `${row.artifact_type}:${row.artifact_id || ''}`,
    },
    {
      key: 'blocker_count',
      header: 'Blockers',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.blocker_count)}</span>,
      sortable: true,
      sortValue: (row) => row.blocker_count,
    },
    {
      key: 'warning_count',
      header: 'Warnings',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.warning_count)}</span>,
      sortable: true,
      sortValue: (row) => row.warning_count,
    },
    {
      key: 'completed_at',
      header: 'Completed',
      render: (row) => <span className="text-xs text-slate-400">{formatTs(row.completed_at || row.started_at)}</span>,
      sortable: true,
      sortValue: (row) => row.completed_at || row.started_at,
    },
  ]

  const runEventColumns: Column<VisibilityEvent>[] = [
    {
      key: 'occurred_at',
      header: 'Time',
      render: (row) => <span className="text-xs text-slate-400">{formatTs(row.occurred_at)}</span>,
      sortable: true,
      sortValue: (row) => row.occurred_at || '',
    },
    {
      key: 'event_type',
      header: 'Event',
      render: (row) => (
        <div className="max-w-[280px]">
          <p className="truncate text-sm text-white">{row.event_type}</p>
          <p className="truncate text-xs text-slate-500">{row.summary}</p>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.event_type,
    },
    {
      key: 'stage',
      header: 'Stage',
      render: (row) => <StageBadge stage={row.stage} />,
      sortable: true,
      sortValue: (row) => row.stage,
    },
    {
      key: 'severity',
      header: 'Severity',
      render: (row) => <SeverityBadge severity={row.severity} />,
      sortable: true,
      sortValue: (row) => row.severity,
    },
    {
      key: 'entity_id',
      header: 'Entity',
      render: (row) => <span className="text-xs text-slate-300">{truncateLabel(row.entity_id, 40)}</span>,
      sortable: true,
      sortValue: (row) => row.entity_id,
    },
  ]

  const runBatchColumns: Column<AdminCostRunBatchJob>[] = [
    {
      key: 'stage_id',
      header: 'Batch Stage',
      render: (row) => (
        <div className="max-w-[220px]">
          <p className="truncate text-sm text-white">{row.stage_id}</p>
          <p className="truncate text-xs text-slate-500">{truncateLabel(row.provider_batch_id || row.task_name, 40)}</p>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.stage_id,
    },
    {
      key: 'status',
      header: 'Status',
      render: (row) => <StatusBadge status={row.status} />,
      sortable: true,
      sortValue: (row) => row.status,
    },
    {
      key: 'submitted_items',
      header: 'Items',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.submitted_items)}</span>,
      sortable: true,
      sortValue: (row) => row.submitted_items,
    },
    {
      key: 'cache_prefiltered_items',
      header: 'Prefiltered',
      render: (row) => <span className="text-xs text-slate-300">{formatNumber(row.cache_prefiltered_items)}</span>,
      sortable: true,
      sortValue: (row) => row.cache_prefiltered_items,
    },
    {
      key: 'fallback_single_call_items',
      header: 'Fallback Single',
      render: (row) => <span className="text-xs text-amber-300">{formatNumber(row.fallback_single_call_items)}</span>,
      sortable: true,
      sortValue: (row) => row.fallback_single_call_items,
    },
    {
      key: 'estimated_savings_usd',
      header: 'Est Savings',
      render: (row) => <span className="text-xs font-medium text-cyan-400">{formatCurrency(row.estimated_savings_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.estimated_savings_usd,
    },
  ]

  const runBatchItemColumns: Column<AdminCostRunBatchItem>[] = [
    {
      key: 'created_at',
      header: 'Time',
      render: (row) => <span className="text-xs text-slate-400">{formatTs(row.completed_at || row.created_at)}</span>,
      sortable: true,
      sortValue: (row) => row.completed_at || row.created_at || '',
    },
    {
      key: 'artifact_id',
      header: 'Batch Item',
      render: (row) => {
        const channel = typeof row.request_metadata.channel === 'string' ? row.request_metadata.channel : ''
        const tier = typeof row.request_metadata.tier === 'string' ? row.request_metadata.tier : ''
        const vendor = row.vendor_name || truncateLabel(row.artifact_id, 42)
        const detail = [row.artifact_type, channel, tier].filter(Boolean).join(' | ')
        return (
          <div className="max-w-[260px]">
            <p className="truncate text-sm text-white">{vendor}</p>
            <p className="truncate text-xs text-slate-500">{detail || row.custom_id}</p>
          </div>
        )
      },
      sortable: true,
      sortValue: (row) => row.vendor_name || row.artifact_id,
    },
    {
      key: 'status',
      header: 'Status',
      render: (row) => <StatusBadge status={row.status} />,
      sortable: true,
      sortValue: (row) => row.status,
    },
    {
      key: 'path',
      header: 'Path',
      render: (row) => (
        <div className="space-y-1">
          <span className="block text-xs text-slate-300">
            {row.cache_prefiltered ? 'Prefiltered' : row.fallback_single_call ? 'Fallback' : 'Batch'}
          </span>
          {row.replay_handler ? (
            <span className="block text-[11px] text-slate-500">{row.replay_handler}</span>
          ) : null}
        </div>
      ),
      sortable: true,
      sortValue: (row) => (row.cache_prefiltered ? 2 : row.fallback_single_call ? 1 : 0),
    },
    {
      key: 'applied_status',
      header: 'Applied',
      render: (row) =>
        row.applied_status ? (
          <div className="space-y-1">
            <StatusBadge status={row.applied_status} />
            {row.applied_at ? <p className="text-[11px] text-slate-500">{formatTs(row.applied_at)}</p> : null}
          </div>
        ) : (
          <span className="text-xs text-slate-500">Pending</span>
        ),
      sortable: true,
      sortValue: (row) => row.applied_status || '',
    },
    {
      key: 'replay_contract_state',
      header: 'Replay',
      render: (row) => (
        <div className="space-y-1">
          <StatusBadge status={row.replay_contract_state || 'missing'} />
          <p className="text-[11px] text-slate-500">
            {row.replay_contract_version != null ? `v${row.replay_contract_version}` : 'legacy/unset'}
          </p>
        </div>
      ),
      sortable: true,
      sortValue: (row) => `${row.replay_contract_state}:${row.replay_contract_version ?? -1}`,
    },
    {
      key: 'cost_usd',
      header: 'Cost',
      render: (row) => <span className="text-xs font-medium text-cyan-400">{formatCurrency(row.cost_usd)}</span>,
      sortable: true,
      sortValue: (row) => row.cost_usd,
    },
    {
      key: 'billable_input_tokens',
      header: 'Billable In',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.billable_input_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.billable_input_tokens,
    },
    {
      key: 'cached_tokens',
      header: 'Cached Read',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.cached_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.cached_tokens,
    },
    {
      key: 'cache_write_tokens',
      header: 'Cache Write',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.cache_write_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.cache_write_tokens,
    },
    {
      key: 'output_tokens',
      header: 'Output',
      render: (row) => <span className="text-xs text-slate-300">{formatCompactTokens(row.output_tokens)}</span>,
      sortable: true,
      sortValue: (row) => row.output_tokens,
    },
    {
      key: 'provider_request_id',
      header: 'Provider Ref',
      render: (row) => (
        <div className="max-w-[180px]">
          <p className="truncate text-xs text-slate-400">{truncateLabel(row.provider_request_id || row.provider_batch_id || row.custom_id, 28)}</p>
          {row.error_text ? <p className="truncate text-[11px] text-rose-300">{truncateLabel(row.error_text, 40)}</p> : null}
          {!row.error_text && row.applied_error ? <p className="truncate text-[11px] text-amber-300">{truncateLabel(row.applied_error, 40)}</p> : null}
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.provider_request_id || row.provider_batch_id || row.custom_id,
    },
  ]

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-4">
        <Filter className="h-4 w-4 text-slate-500" />
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => {
              setSourceFilter('')
              setEventTypeFilter('')
              setEntityTypeFilter('')
            }}
            className="rounded border border-slate-700/60 px-2.5 py-1 text-xs text-slate-300 transition hover:border-slate-500/60 hover:text-white"
          >
            All Calls
          </button>
          <button
            onClick={() => {
              setSourceFilter('b2b_battle_cards')
              setEventTypeFilter('llm_overlay')
              setEntityTypeFilter('battle_card')
            }}
            className="rounded border border-cyan-500/30 bg-cyan-500/10 px-2.5 py-1 text-xs text-cyan-200 transition hover:border-cyan-400/50 hover:text-white"
          >
            Battle Card Overlays
          </button>
        </div>
        <FilterSelect
          label="Window"
          value={days}
          onChange={setDays}
          options={[
            { value: '7', label: '7 days' },
            { value: '30', label: '30 days' },
            { value: '90', label: '90 days' },
            { value: '365', label: '365 days' },
          ]}
        />
        <FilterSelect
          label="Cache Top"
          value={cacheTopN}
          onChange={setCacheTopN}
          options={[
            { value: '5', label: 'Top 5' },
            { value: '8', label: 'Top 8' },
            { value: '12', label: 'Top 12' },
            { value: '20', label: 'Top 20' },
          ]}
        />
        <FilterSelect
          label="Provider"
          value={providerFilter}
          onChange={setProviderFilter}
          options={[
            { value: '', label: 'All providers' },
            ...providerOptions.map((value) => ({ value, label: value })),
          ]}
        />
        <FilterSelect
          label="Model"
          value={modelFilter}
          onChange={setModelFilter}
          options={[
            { value: '', label: 'All models' },
            ...modelOptions.map((value) => ({ value, label: truncateLabel(value, 52) })),
          ]}
        />
        <FilterSelect
          label="Operation"
          value={spanFilter}
          onChange={setSpanFilter}
          options={[
            { value: '', label: 'All operations' },
            ...spanOptions.map((value) => ({ value, label: truncateLabel(value, 52) })),
          ]}
        />
        <FilterSelect
          label="Status"
          value={statusFilter}
          onChange={setStatusFilter}
          options={[
            { value: '', label: 'All statuses' },
            ...statusOptions.map((value) => ({ value, label: value })),
          ]}
        />
        <FilterSelect
          label="Source"
          value={sourceFilter}
          onChange={setSourceFilter}
          options={[
            { value: '', label: 'All sources' },
            ...sourceOptions.map((value) => ({ value, label: value })),
          ]}
        />
        <FilterSelect
          label="Event"
          value={eventTypeFilter}
          onChange={setEventTypeFilter}
          options={[
            { value: '', label: 'All events' },
            ...eventTypeOptions.map((value) => ({ value, label: value })),
          ]}
        />
        <FilterSelect
          label="Entity"
          value={entityTypeFilter}
          onChange={setEntityTypeFilter}
          options={[
            { value: '', label: 'All entities' },
            ...entityTypeOptions.map((value) => ({ value, label: value })),
          ]}
        />
        <FilterSelect
          label="Cache"
          value={cacheFilter}
          onChange={setCacheFilter}
          options={[
            { value: '', label: 'All calls' },
            { value: 'cached', label: 'Cached only' },
            { value: 'uncached', label: 'Uncached only' },
          ]}
        />
        <label className="flex min-w-[240px] flex-col gap-1 text-xs text-slate-500">
          <span>Run ID</span>
          <input
            value={runIdInput}
            onChange={(e) => setRunIdInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                setRunDetailPreset('all')
                setActiveRunId(runIdInput.trim())
              }
            }}
            placeholder="Paste execution id"
            className="rounded border border-slate-700/60 bg-slate-950/60 px-2.5 py-1.5 text-xs text-white outline-none transition focus:border-cyan-500/60"
          />
        </label>
        <button
          onClick={() => {
            setRunDetailPreset('all')
            setActiveRunId(runIdInput.trim())
          }}
          disabled={!runIdInput.trim()}
          className="mt-5 flex items-center gap-1.5 rounded border border-slate-700/60 px-2.5 py-1 text-xs text-slate-300 transition hover:border-cyan-500/60 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
        >
          Load Run
        </button>
        <button
          onClick={() => {
            setRunIdInput('')
            setActiveRunId('')
            setRunDetailPreset('all')
          }}
          disabled={!activeRunId}
          className="mt-5 flex items-center gap-1.5 rounded border border-slate-700/60 px-2.5 py-1 text-xs text-slate-400 transition hover:border-slate-500/60 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
        >
          Clear Run
        </button>
        <button
          onClick={() => {
            refresh()
            refreshVendors()
            refreshB2bEfficiency()
            refreshGenericReasoning()
            refreshReconciliation()
            refreshReasoningActivity()
            refreshCacheHealth()
            refreshTaskHealth()
            if (activeRunId) refreshRunDetail()
          }}
          disabled={unifiedRefreshing || runDetailRefreshing}
          className="ml-auto flex items-center gap-1.5 px-2.5 py-1 text-xs text-slate-400 hover:text-white transition-colors"
        >
          <RefreshCw className={clsx('h-3 w-3', (unifiedRefreshing || runDetailRefreshing) && 'animate-spin')} />
          Refresh
        </button>
      </div>

      <div className="rounded-xl border border-slate-700/50 bg-slate-900/40 p-4 text-sm text-slate-400">
        Provider/model calls below show prompt-cache economics only. Atlas exact cache, semantic reuse, and evidence-hash skips are surfaced separately so cost tables stay truthful without hiding internal reuse.
      </div>

      {unifiedError && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
          {unifiedError.message}
        </div>
      )}
      {runDetailError && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
          {runDetailError.message}
        </div>
      )}

      <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 overflow-hidden">
        <div className="flex items-center justify-between gap-3 border-b border-slate-700/50 px-4 py-3">
          <div>
            <h2 className="text-sm font-medium text-white">Atlas Cache Health</h2>
            <p className="text-xs text-slate-500">
              Exact cache, Anthropic batching, semantic reuse, evidence-hash skips, and provider prompt cache leaders for the last {days} days.
            </p>
          </div>
          <span
            className={clsx(
              'inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium',
              cacheHealth?.exact_cache.enabled
                ? 'bg-green-500/15 text-green-400'
                : 'bg-red-500/15 text-red-400',
            )}
          >
            Exact cache {cacheHealth?.exact_cache.enabled ? 'enabled' : 'disabled'}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-4 border-b border-slate-700/50 p-4 md:grid-cols-3 xl:grid-cols-9">
          <StatCard
            label="Exact Rows"
            value={formatNumber(cacheHealth?.exact_cache.total_rows)}
            icon={<Database className="h-4 w-4" />}
            skeleton={unifiedLoading}
          />
          <StatCard
            label="Exact Hits"
            value={formatNumber(cacheHealth?.exact_cache.total_hits)}
            icon={<RefreshCw className="h-4 w-4" />}
            skeleton={unifiedLoading}
          />
          <StatCard
            label={`Exact Writes ${days}d`}
            value={formatNumber(cacheHealth?.exact_cache.writes_in_window)}
            icon={<Database className="h-4 w-4" />}
            skeleton={unifiedLoading}
          />
          <StatCard
            label={`Rows Hit ${days}d`}
            value={formatNumber(cacheHealth?.exact_cache.rows_hit_in_window)}
            icon={<CheckCircle2 className="h-4 w-4" />}
            skeleton={unifiedLoading}
          />
          <StatCard
            label="Semantic Active"
            value={formatNumber(cacheHealth?.semantic_cache.active_entries)}
            icon={<Cpu className="h-4 w-4" />}
            skeleton={unifiedLoading}
          />
          <StatCard
            label="Cross-Vendor Cached"
            value={formatNumber(cacheHealth?.evidence_hash_reuse.cross_vendor_cached_rows)}
            icon={<Shield className="h-4 w-4" />}
            skeleton={unifiedLoading}
          />
          <StatCard
            label={`Batch Jobs ${days}d`}
            value={formatNumber(cacheHealth?.anthropic_batching.submitted_jobs)}
            icon={<Workflow className="h-4 w-4" />}
            skeleton={unifiedLoading}
          />
          <StatCard
            label="Batch Items"
            value={formatNumber(cacheHealth?.anthropic_batching.submitted_items)}
            icon={<Cpu className="h-4 w-4" />}
            skeleton={unifiedLoading}
          />
          <StatCard
            label="Batch Savings"
            value={formatMaybeCurrency(cacheHealth?.anthropic_batching.estimated_savings_usd)}
            icon={<DollarSign className="h-4 w-4" />}
            skeleton={unifiedLoading}
          />
          <StatCard
            label="Stale Batch Jobs"
            value={formatNumber(cacheHealth?.anthropic_batching.stale_jobs_count)}
            sub={cacheHealth ? `${cacheHealth.anthropic_batching.stale_job_threshold_minutes}m threshold` : undefined}
            icon={<AlertTriangle className="h-4 w-4" />}
            skeleton={unifiedLoading}
          />
          <StatCard
            label="Stale Item Claims"
            value={formatNumber(cacheHealth?.anthropic_batching.stale_claims_count)}
            sub={cacheHealth ? `${cacheHealth.anthropic_batching.stale_job_threshold_minutes}m threshold` : undefined}
            icon={<Clock className="h-4 w-4" />}
            skeleton={unifiedLoading}
          />
        </div>

        <div className="grid gap-4 p-4 xl:grid-cols-[1.15fr,0.85fr]">
          <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 overflow-hidden">
            <div className="border-b border-slate-700/50 px-4 py-3">
              <h3 className="text-sm font-medium text-white">Exact Cache Stages</h3>
              <p className="text-xs text-slate-500">Declared exact-cache stages with row and hit coverage.</p>
            </div>
            {cacheHealthLoading ? (
              <DataTable columns={exactStageColumns} data={[]} skeletonRows={5} />
            ) : (
              <DataTable
                columns={exactStageColumns}
                data={cacheHealth?.exact_cache.stages ?? []}
                emptyMessage="No Atlas exact-cache stages found"
              />
            )}
          </div>

          <div className="space-y-4">
            <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 overflow-hidden">
              <div className="flex items-center justify-between gap-3 border-b border-slate-700/50 px-4 py-3">
                <div>
                  <h3 className="text-sm font-medium text-white">Reconciler Task</h3>
                  <p className="text-xs text-slate-500">
                    Scheduler heartbeat and manual trigger for detached campaign batch reconciliation.
                  </p>
                </div>
                <button
                  onClick={handleRunReconciler}
                  disabled={!reconcilerTask || reconcileTriggering}
                  className="inline-flex items-center gap-1.5 rounded border border-cyan-500/30 bg-cyan-500/10 px-3 py-1.5 text-xs font-medium text-cyan-300 transition hover:bg-cyan-500/20 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <RefreshCw className={clsx('h-3 w-3', reconcileTriggering && 'animate-spin')} />
                  {reconcileTriggering ? 'Triggering...' : 'Run Reconciler Now'}
                </button>
              </div>
              <div className="grid gap-4 p-4 md:grid-cols-4">
                <StatCard
                  label="Task"
                  value={reconcilerTask ? RECON_TASK_NAME : 'Missing'}
                  sub={reconcilerTask ? (reconcilerTask.enabled ? 'Enabled' : 'Disabled') : 'No scheduled task row'}
                  icon={<Workflow className="h-4 w-4" />}
                  skeleton={taskHealthLoading}
                />
                <StatCard
                  label="Last Run"
                  value={reconcilerTask?.last_run_at ? formatTs(reconcilerTask.last_run_at) : '--'}
                  sub={reconcilerTask?.last_status || 'Never ran'}
                  icon={<Clock className="h-4 w-4" />}
                  skeleton={taskHealthLoading}
                />
                <StatCard
                  label="Next Run"
                  value={reconcilerTask?.next_run_at ? formatTs(reconcilerTask.next_run_at) : '--'}
                  sub={reconcilerTask?.schedule_type || '--'}
                  icon={<CalendarClock className="h-4 w-4" />}
                  skeleton={taskHealthLoading}
                />
                <StatCard
                  label="Recent Failure Rate"
                  value={formatFailureRate(reconcilerTask?.recent_failure_rate)}
                  sub={`${formatNumber(reconcilerTask?.recent_runs)} recent runs`}
                  icon={<AlertTriangle className="h-4 w-4" />}
                  skeleton={taskHealthLoading}
                />
              </div>
              {reconcilerTask ? (
                <div className="border-t border-slate-700/50 px-4 py-3 text-xs text-slate-400">
                  <div className="flex flex-wrap items-center gap-3">
                    <span>Status: <span className="text-slate-200">{reconcilerTask.last_status || 'never_ran'}</span></span>
                    {reconcilerTask.interval_seconds ? (
                      <span>Interval: <span className="text-slate-200">{formatAgeMinutes(reconcilerTask.interval_seconds / 60)}</span></span>
                    ) : null}
                    {reconcilerTask.last_duration_ms != null ? (
                      <span>Last Duration: <span className="text-slate-200">{formatNumber(reconcilerTask.last_duration_ms)} ms</span></span>
                    ) : null}
                  </div>
                  {reconcilerTask.last_error ? (
                    <p className="mt-2 truncate text-rose-300">{reconcilerTask.last_error}</p>
                  ) : null}
                  {reconcileMessage ? <p className="mt-2 text-cyan-300">{reconcileMessage}</p> : null}
                  {reconcileError ? <p className="mt-2 text-rose-300">{reconcileError}</p> : null}
                </div>
              ) : reconcileError ? (
                <div className="border-t border-slate-700/50 px-4 py-3 text-xs text-rose-300">{reconcileError}</div>
              ) : null}
            </div>

            <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 overflow-hidden">
              <div className="border-b border-slate-700/50 px-4 py-3">
                <h3 className="text-sm font-medium text-white">Recent Task Reuse</h3>
                <p className="text-xs text-slate-500">Reuse observed in recent task executions by strategy.</p>
              </div>
            {cacheHealthLoading ? (
              <DataTable columns={taskReuseColumns} data={[]} skeletonRows={3} />
            ) : (
                <DataTable
                  columns={taskReuseColumns}
                  data={cacheHealth?.task_reuse.tasks ?? []}
                  emptyMessage="No recent task reuse rows in the current window"
                />
              )}
            </div>

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-1">
              <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 overflow-hidden">
                <div className="border-b border-slate-700/50 px-4 py-3">
                  <h3 className="text-sm font-medium text-white">Provider Prompt Cache Leaders</h3>
                  <p className="text-xs text-slate-500">OpenRouter/provider-side prompt caching by span.</p>
                </div>
                {cacheHealthLoading ? (
                  <DataTable columns={promptCacheColumns} data={[]} skeletonRows={4} />
                ) : (
                  <DataTable
                    columns={promptCacheColumns}
                    data={cacheHealth?.provider_prompt_cache.top_spans ?? []}
                    emptyMessage="No provider prompt-cache activity in the current window"
                  />
                )}
              </div>

              <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 overflow-hidden">
                <div className="border-b border-slate-700/50 px-4 py-3">
                  <h3 className="text-sm font-medium text-white">Semantic Cache Classes</h3>
                  <p className="text-xs text-slate-500">Active semantic reuse buckets and recent validations.</p>
                </div>
                {cacheHealthLoading ? (
                  <DataTable columns={semanticColumns} data={[]} skeletonRows={4} />
                ) : (
                  <DataTable
                    columns={semanticColumns}
                    data={cacheHealth?.semantic_cache.pattern_classes ?? []}
                    emptyMessage="No semantic-cache classes found"
                  />
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="border-t border-slate-700/50 p-4">
          <div className="space-y-4">
            <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 overflow-hidden">
              <div className="flex items-center justify-between gap-3 border-b border-slate-700/50 px-4 py-3">
                <div>
                  <h3 className="text-sm font-medium text-white">Anthropic Batching</h3>
                  <p className="text-xs text-slate-500">
                    Submitted batch jobs, prefiltered items, fallback singles, and estimated savings by stage.
                  </p>
                </div>
                <span
                  className={clsx(
                    'inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium',
                    cacheHealth?.anthropic_batching.enabled
                      ? 'bg-green-500/15 text-green-400'
                      : 'bg-red-500/15 text-red-400',
                  )}
                >
                  Anthropic batching {cacheHealth?.anthropic_batching.enabled ? 'enabled' : 'disabled'}
                </span>
              </div>
              {cacheHealthLoading ? (
                <DataTable columns={batchStageColumns} data={[]} skeletonRows={4} />
              ) : (
                <DataTable
                  columns={batchStageColumns}
                  data={cacheHealth?.anthropic_batching.stages ?? []}
                  emptyMessage="No Anthropic batch jobs in the current window"
                />
              )}
            </div>

            <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 overflow-hidden">
              <div className="border-b border-slate-700/50 px-4 py-3">
                <h3 className="text-sm font-medium text-white">Stale Detached Batches</h3>
                <p className="text-xs text-slate-500">
                  Submitted provider batches with no completion after {cacheHealth?.anthropic_batching.stale_job_threshold_minutes ?? 30} minutes.
                </p>
              </div>
              {cacheHealthLoading ? (
                <DataTable columns={staleBatchColumns} data={[]} skeletonRows={3} />
              ) : (
                <DataTable
                  columns={staleBatchColumns}
                  data={cacheHealth?.anthropic_batching.stale_jobs ?? []}
                  emptyMessage="No stale detached batch jobs in the current window"
                />
              )}
            </div>

            <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 overflow-hidden">
              <div className="border-b border-slate-700/50 px-4 py-3">
                <h3 className="text-sm font-medium text-white">Stale Item Claims</h3>
                <p className="text-xs text-slate-500">
                  Batch items claimed for apply but not marked applied after {cacheHealth?.anthropic_batching.stale_job_threshold_minutes ?? 30} minutes.
                </p>
              </div>
              {cacheHealthLoading ? (
                <DataTable columns={staleClaimColumns} data={[]} skeletonRows={3} />
              ) : (
                <DataTable
                  columns={staleClaimColumns}
                  data={cacheHealth?.anthropic_batching.stale_claims ?? []}
                  emptyMessage="No stale item claims in the current window"
                />
              )}
            </div>
          </div>
        </div>
      </div>

      {activeRunId && (
        <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 overflow-hidden">
          <div className="flex items-center justify-between gap-3 border-b border-slate-700/50 px-4 py-3">
            <div>
              <h2 className="text-sm font-medium text-white">Run Detail</h2>
              <p className="text-xs text-slate-500">
                Correlates one execution across task output, llm usage, attempts, and visibility events.
              </p>
            </div>
            <span className="max-w-[320px] truncate text-xs text-slate-400">{activeRunId}</span>
          </div>

          {runDetailLoading ? (
            <div className="grid grid-cols-2 gap-4 p-4 md:grid-cols-3 xl:grid-cols-6">
              <StatCard label="Task" value="--" icon={<Workflow className="h-4 w-4" />} skeleton />
              <StatCard label="LLM Calls" value="--" icon={<Cpu className="h-4 w-4" />} skeleton />
              <StatCard label="Cost" value="--" icon={<DollarSign className="h-4 w-4" />} skeleton />
              <StatCard label="Billable In" value="--" icon={<Database className="h-4 w-4" />} skeleton />
              <StatCard label="Cached Read" value="--" icon={<RefreshCw className="h-4 w-4" />} skeleton />
              <StatCard label="Artifacts" value="--" icon={<GitCompareArrows className="h-4 w-4" />} skeleton />
            </div>
          ) : runDetail ? (
            <>
            <div className="grid grid-cols-2 gap-4 border-b border-slate-700/50 p-4 md:grid-cols-4 xl:grid-cols-8">
              <StatCard
                label="Task"
                value={runDetail.task_execution?.task_name || 'External'}
                  sub={runDetail.task_execution?.status || 'No execution row'}
                  icon={<Workflow className="h-4 w-4" />}
                />
                <StatCard
                  label="LLM Calls"
                  value={formatNumber(runDetail.llm_summary.total_calls)}
                  sub={runDetail.llm_summary.last_call_at ? `Last ${formatTs(runDetail.llm_summary.last_call_at)}` : 'No llm calls'}
                  icon={<Cpu className="h-4 w-4" />}
                />
                <StatCard
                  label="Cost"
                  value={formatCurrency(runDetail.llm_summary.total_cost_usd)}
                  sub={`${formatCompactTokens(runDetail.llm_summary.total_tokens)} total tokens`}
                  icon={<DollarSign className="h-4 w-4" />}
                />
                <StatCard
                  label="Billable In"
                  value={formatCompactTokens(runDetail.llm_summary.total_billable_input_tokens)}
                  sub={`${formatCompactTokens(runDetail.llm_summary.total_input_tokens)} raw input`}
                  icon={<Database className="h-4 w-4" />}
                />
                <StatCard
                  label="Cached Read"
                  value={formatCompactTokens(runDetail.llm_summary.total_cached_tokens)}
                  sub={`${formatNumber(runDetail.llm_summary.cache_hit_calls)} cache-hit calls`}
                  icon={<RefreshCw className="h-4 w-4" />}
                />
              <StatCard
                label="Artifacts"
                value={formatNumber(runDetail.artifact_attempts.length)}
                sub={`${formatNumber(runDetail.visibility_events.length)} visibility events`}
                icon={<GitCompareArrows className="h-4 w-4" />}
              />
              <StatCard
                label="Batch Jobs"
                value={formatNumber(runDetail.batching_summary.submitted_jobs)}
                sub={`${formatNumber(runDetail.batching_summary.submitted_items)} submitted items`}
                icon={<Workflow className="h-4 w-4" />}
              />
              <StatCard
                label="Batch Savings"
                value={formatMaybeCurrency(runDetail.batching_summary.estimated_savings_usd)}
                sub={`${formatNumber(runDetail.batching_summary.fallback_single_call_items)} fallback singles`}
                icon={<DollarSign className="h-4 w-4" />}
              />
            </div>

            {runDetailPreset === 'battle_card_overlay' && (
              <div className="grid grid-cols-2 gap-4 border-b border-slate-700/50 px-4 py-4 md:grid-cols-3 xl:grid-cols-6">
                <StatCard
                  label="Overlay Calls"
                  value={formatNumber(overlayCallCount)}
                  sub="Filtered battle-card overlay calls"
                  icon={<Cpu className="h-4 w-4" />}
                />
                <StatCard
                  label="Overlay Cost"
                  value={formatCurrency(overlayCostUsd)}
                  sub={`${formatCompactTokens(overlayBillableInputTokens)} billable input`}
                  icon={<DollarSign className="h-4 w-4" />}
                />
                <StatCard
                  label="Prompt Cache"
                  value={formatCompactTokens(overlayCachedTokens)}
                  sub="Provider cached read tokens"
                  icon={<RefreshCw className="h-4 w-4" />}
                />
                <StatCard
                  label="Attempt Failures"
                  value={formatNumber(overlayAttemptFailures)}
                  sub={`${formatNumber(runDetailAttempts.length)} overlay attempts`}
                  icon={<AlertTriangle className="h-4 w-4" />}
                />
                <StatCard
                  label="Warnings"
                  value={formatNumber(overlayEventWarnings)}
                  sub={`${formatNumber(runDetailEvents.length)} overlay events`}
                  icon={<Shield className="h-4 w-4" />}
                />
                <StatCard
                  label="Entity"
                  value={runDetailCalls[0]?.entity_type || 'battle_card'}
                  sub={runDetailCalls[0]?.vendor_name || runDetailCalls[0]?.entity_id || 'Overlay scope'}
                  icon={<Building2 className="h-4 w-4" />}
                />
              </div>
            )}

              <div className="grid gap-4 p-4 xl:grid-cols-[1.05fr,0.95fr]">
                <div className="space-y-4">
                  <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <h3 className="text-sm font-medium text-white">Execution</h3>
                        <p className="text-xs text-slate-500">
                          {runDetail.task_execution?.started_at ? `Started ${formatTs(runDetail.task_execution.started_at)}` : 'No task execution row'}
                        </p>
                      </div>
                      {runDetail.task_execution && <StatusBadge status={runDetail.task_execution.status} />}
                    </div>
                    <div className="mt-3 grid gap-2 text-xs text-slate-400 md:grid-cols-2">
                      <div>Duration: <span className="text-slate-200">{formatNumber(runDetail.task_execution?.duration_ms)} ms</span></div>
                      <div>Retry count: <span className="text-slate-200">{formatNumber(runDetail.task_execution?.retry_count)}</span></div>
                      <div>Call window: <span className="text-slate-200">{formatTs(runDetail.llm_summary.first_call_at)}</span></div>
                      <div>Last call: <span className="text-slate-200">{formatTs(runDetail.llm_summary.last_call_at)}</span></div>
                    </div>
                    {runDetail.task_execution?.result_text && (
                      <pre className="mt-3 max-h-48 overflow-auto rounded border border-slate-800 bg-slate-950/70 p-3 text-[11px] text-slate-300">
                        {JSON.stringify(runDetail.task_execution.result, null, 2)}
                      </pre>
                    )}
                  </div>

                  <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 overflow-hidden">
                    <div className="border-b border-slate-700/50 px-4 py-3">
                      <h3 className="text-sm font-medium text-white">Run Operations</h3>
                      <p className="text-xs text-slate-500">Aggregated LLM usage for this run only.</p>
                    </div>
                    <DataTable columns={operationColumns} data={runDetail.operations} emptyMessage="No operation rows for this run" />
                  </div>

                  <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 overflow-hidden">
                    <div className="border-b border-slate-700/50 px-4 py-3">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <h3 className="text-sm font-medium text-white">Run Calls</h3>
                          <p className="text-xs text-slate-500">Recent traced model calls attached to this run id.</p>
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => setRunDetailPreset('all')}
                            className={clsx(
                              'rounded border px-2.5 py-1 text-xs transition',
                              runDetailPreset === 'all'
                                ? 'border-slate-500/60 text-white'
                                : 'border-slate-700/60 text-slate-400 hover:border-slate-500/60 hover:text-white',
                            )}
                          >
                            All
                          </button>
                          <button
                            onClick={() => setRunDetailPreset('battle_card_overlay')}
                            className={clsx(
                              'rounded border px-2.5 py-1 text-xs transition',
                              runDetailPreset === 'battle_card_overlay'
                                ? 'border-cyan-400/50 bg-cyan-500/10 text-cyan-200'
                                : 'border-slate-700/60 text-slate-400 hover:border-cyan-500/60 hover:text-white',
                            )}
                          >
                            Overlay
                          </button>
                        </div>
                      </div>
                    </div>
                    <DataTable columns={recentColumns} data={runDetailCalls} emptyMessage="No traced calls for this run" />
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 overflow-hidden">
                    <div className="border-b border-slate-700/50 px-4 py-3">
                      <h3 className="text-sm font-medium text-white">Run Batch Jobs</h3>
                      <p className="text-xs text-slate-500">Batch execution summary attached to this run id.</p>
                    </div>
                    <DataTable columns={runBatchColumns} data={runDetail.batch_jobs} emptyMessage="No batch jobs for this run" />
                  </div>

                  <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 overflow-hidden">
                    <div className="border-b border-slate-700/50 px-4 py-3">
                      <h3 className="text-sm font-medium text-white">Run Batch Items</h3>
                      <p className="text-xs text-slate-500">Per-item batch outcomes, including prefiltered cache hits and fallback singles.</p>
                    </div>
                    <DataTable columns={runBatchItemColumns} data={runDetail.batch_items} emptyMessage="No batch items for this run" />
                  </div>

                  <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 overflow-hidden">
                    <div className="border-b border-slate-700/50 px-4 py-3">
                      <h3 className="text-sm font-medium text-white">Artifact Attempts</h3>
                      <p className="text-xs text-slate-500">Quality and persistence attempts recorded for this run.</p>
                    </div>
                    <DataTable columns={runAttemptColumns} data={runDetailAttempts} emptyMessage="No artifact attempts for this run" />
                  </div>

                  <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 overflow-hidden">
                    <div className="border-b border-slate-700/50 px-4 py-3">
                      <h3 className="text-sm font-medium text-white">Visibility Events</h3>
                      <p className="text-xs text-slate-500">Warnings, failures, and summaries emitted under the same run id.</p>
                    </div>
                    <DataTable columns={runEventColumns} data={runDetailEvents} emptyMessage="No visibility events for this run" />
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="p-4 text-sm text-slate-500">Load a run id to inspect one execution end to end.</div>
          )}
        </div>
      )}

      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
        <StatCard
          label="Total Cost"
          value={formatCurrency(data?.summary.total_cost_usd)}
          icon={<DollarSign className="h-4 w-4" />}
          skeleton={loading}
        />
        <StatCard
          label="Model Calls"
          value={formatNumber(data?.summary.total_calls)}
          icon={<Cpu className="h-4 w-4" />}
          skeleton={loading}
        />
        <StatCard
          label="Billable Input"
          value={formatCompactTokens(data?.summary.total_billable_input_tokens)}
          icon={<Database className="h-4 w-4" />}
          skeleton={loading}
        />
        <StatCard
          label="Cached Reads"
          value={formatCompactTokens(data?.summary.total_cached_tokens)}
          icon={<Database className="h-4 w-4" />}
          skeleton={loading}
        />
        <StatCard
          label="Cache Writes"
          value={formatCompactTokens(data?.summary.total_cache_write_tokens)}
          icon={<Database className="h-4 w-4" />}
          skeleton={loading}
        />
        <StatCard
          label="Cache Hit Calls"
          value={formatNumber(data?.summary.cache_hit_calls)}
          icon={<RefreshCw className="h-4 w-4" />}
          skeleton={loading}
        />
      </div>

      <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-700/50">
          <h2 className="text-sm font-medium text-white">Burn Dashboard</h2>
          <p className="text-xs text-slate-500">
            Per-job burn across scheduled tasks and generic event-driven reasoning. This section is window-only and independent of provider/model/span filters.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-4 border-b border-slate-700/50 p-4 md:grid-cols-4">
          <StatCard
            label="Tracked Cost"
            value={formatCurrency(burnDashboard?.summary.tracked_cost_usd)}
            icon={<DollarSign className="h-4 w-4" />}
            skeleton={burnDashboardLoading}
          />
          <StatCard
            label="Model Calls"
            value={formatNumber(burnDashboard?.summary.model_call_count)}
            icon={<Cpu className="h-4 w-4" />}
            skeleton={burnDashboardLoading}
          />
          <StatCard
            label="Runs"
            value={formatNumber(burnDashboard?.summary.recent_runs)}
            icon={<Workflow className="h-4 w-4" />}
            skeleton={burnDashboardLoading}
          />
          <StatCard
            label="Reprocess %"
            value={formatMaybePercent(burnDashboard?.summary.reprocess_pct)}
            sub={
              burnDashboard?.summary.rows_processed != null
                ? `${formatMaybeNumber(burnDashboard?.summary.rows_reprocessed)} / ${formatMaybeNumber(burnDashboard?.summary.rows_processed)} rows`
                : undefined
            }
            icon={<GitCompareArrows className="h-4 w-4" />}
            skeleton={burnDashboardLoading}
          />
        </div>
        <div className="grid grid-cols-2 gap-4 border-b border-slate-700/50 p-4 md:grid-cols-4">
          <StatCard
            label="Vendor Budget Rejects"
            value={formatNumber(burnDashboard?.reasoning_budget_pressure.vendor_rejections)}
            icon={<AlertTriangle className="h-4 w-4" />}
            skeleton={burnDashboardLoading}
          />
          <StatCard
            label="Cross-Vendor Rejects"
            value={formatNumber(burnDashboard?.reasoning_budget_pressure.cross_vendor_rejections)}
            icon={<AlertTriangle className="h-4 w-4" />}
            skeleton={burnDashboardLoading}
          />
          <StatCard
            label="Max Cross-Vendor In"
            value={formatMaybeNumber(burnDashboard?.reasoning_budget_pressure.max_cross_vendor_estimated_input_tokens)}
            sub={
              burnDashboard?.reasoning_budget_pressure.max_cross_vendor_cap != null
                ? `cap ${formatMaybeNumber(burnDashboard?.reasoning_budget_pressure.max_cross_vendor_cap)}`
                : undefined
            }
            icon={<Cpu className="h-4 w-4" />}
            skeleton={burnDashboardLoading}
          />
          <StatCard
            label="Last Budget Reject"
            value={formatTs(burnDashboard?.reasoning_budget_pressure.last_rejection_at) || '--'}
            icon={<Clock className="h-4 w-4" />}
            skeleton={burnDashboardLoading}
          />
        </div>
        <DataTable
          columns={burnBudgetColumns}
          data={burnBudgetRows}
          emptyMessage="No reasoning budget rejections in the selected window"
        />
        <DataTable
          columns={burnColumns}
          data={burnRows}
          emptyMessage="No task burn rows in the selected window"
        />
      </div>

      <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-700/50">
          <h2 className="text-sm font-medium text-white">Provider Reconciliation</h2>
          <p className="text-xs text-slate-500">
            Compares local tracked spend to normalized provider totals. This section is window-only and independent of provider/model/span filters.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-4 border-b border-slate-700/50 p-4 md:grid-cols-4">
          <StatCard
            label="Tracked Cost"
            value={formatCurrency(reconciliation?.summary.tracked_cost_usd)}
            icon={<DollarSign className="h-4 w-4" />}
            skeleton={reconciliationLoading}
          />
          <StatCard
            label="Provider Cost"
            value={formatMaybeCurrency(reconciliation?.summary.provider_cost_usd)}
            icon={<DollarSign className="h-4 w-4" />}
            skeleton={reconciliationLoading}
          />
          <StatCard
            label="Delta"
            value={formatMaybeCurrency(reconciliation?.summary.delta_cost_usd)}
            icon={<AlertTriangle className="h-4 w-4" />}
            skeleton={reconciliationLoading}
          />
          <StatCard
            label="Status"
            value={reconciliation?.status || '--'}
            sub={reconciliation?.message || undefined}
            icon={<Shield className="h-4 w-4" />}
            skeleton={reconciliationLoading}
          />
        </div>
        {reconciliation?.message ? (
          <div className="border-b border-slate-700/50 bg-amber-500/5 px-4 py-3 text-xs text-amber-300">
            {reconciliation.message}
          </div>
        ) : null}
        <DataTable
          columns={reconciliationColumns}
          data={reconciliation?.daily_rows ?? []}
          emptyMessage="No reconciliation rows in the selected window"
        />
      </div>

      <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-700/50">
          <h2 className="text-sm font-medium text-white">Generic Reasoning</h2>
          <p className="text-xs text-slate-500">
            Tracks generic reasoning-agent spend by source, event type, and dominant entities. This section is window-only and independent of provider/model/span filters.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-4 border-b border-slate-700/50 p-4 md:grid-cols-4">
          <StatCard
            label="Generic Cost"
            value={formatCurrency(genericReasoning?.summary.total_cost_usd)}
            icon={<DollarSign className="h-4 w-4" />}
            skeleton={genericReasoningLoading}
          />
          <StatCard
            label="Generic Calls"
            value={formatNumber(genericReasoning?.summary.total_calls)}
            icon={<Cpu className="h-4 w-4" />}
            skeleton={genericReasoningLoading}
          />
          <StatCard
            label="Top Source"
            value={genericReasoning?.summary.top_source_name || '--'}
            icon={<Database className="h-4 w-4" />}
            skeleton={genericReasoningLoading}
          />
          <StatCard
            label="Top Event Type"
            value={genericReasoning?.summary.top_event_type || '--'}
            icon={<Workflow className="h-4 w-4" />}
            skeleton={genericReasoningLoading}
          />
        </div>
        <div className="grid gap-4 p-4 xl:grid-cols-2">
          <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 overflow-hidden">
            <div className="border-b border-slate-700/50 px-4 py-3">
              <h3 className="text-sm font-medium text-white">Generic Reasoning By Source</h3>
              <p className="text-xs text-slate-500">Source systems driving generic reasoning spend.</p>
            </div>
            <DataTable
              columns={genericReasoningSourceColumns}
              data={genericReasoningSources}
              emptyMessage="No generic reasoning source rows in the selected window"
            />
          </div>
          <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 overflow-hidden">
            <div className="border-b border-slate-700/50 px-4 py-3">
              <h3 className="text-sm font-medium text-white">Generic Reasoning By Event Type</h3>
              <p className="text-xs text-slate-500">Event classes that are triggering generic reasoning calls.</p>
            </div>
            <DataTable
              columns={genericReasoningEventColumns}
              data={genericReasoningEvents}
              emptyMessage="No generic reasoning event rows in the selected window"
            />
          </div>
        </div>
        <div className="grid gap-4 border-t border-slate-700/50 p-4 xl:grid-cols-2">
          <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 overflow-hidden">
            <div className="border-b border-slate-700/50 px-4 py-3">
              <h3 className="text-sm font-medium text-white">Top Source / Event Pairs</h3>
              <p className="text-xs text-slate-500">Most expensive combined generic reasoning triggers.</p>
            </div>
            <DataTable
              columns={genericReasoningPairColumns}
              data={genericReasoningPairs}
              emptyMessage="No generic reasoning source/event pairs in the selected window"
            />
          </div>
          <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 overflow-hidden">
            <div className="border-b border-slate-700/50 px-4 py-3">
              <h3 className="text-sm font-medium text-white">Dominant Entities</h3>
              <p className="text-xs text-slate-500">Top entity ids attached to generic reasoning spend.</p>
            </div>
            <DataTable
              columns={genericReasoningEntityColumns}
              data={genericReasoningEntities}
              emptyMessage="No dominant generic reasoning entities in the selected window"
            />
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-700/50">
          <h2 className="text-sm font-medium text-white">Legacy Stratified Reasoning</h2>
          <p className="text-xs text-slate-500">
            Explicitly tracks any remaining stratified reasoning spend in the selected window.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-4 border-b border-slate-700/50 p-4 md:grid-cols-3">
          <StatCard
            label="Stratified Cost"
            value={formatCurrency(reasoningActivity?.summary.total_cost_usd)}
            icon={<DollarSign className="h-4 w-4" />}
            skeleton={reasoningActivityLoading}
          />
          <StatCard
            label="Stratified Calls"
            value={formatNumber(reasoningActivity?.summary.total_calls)}
            icon={<Cpu className="h-4 w-4" />}
            skeleton={reasoningActivityLoading}
          />
          <StatCard
            label="Stratified Tokens"
            value={formatCompactTokens(reasoningActivity?.summary.total_tokens)}
            icon={<Database className="h-4 w-4" />}
            skeleton={reasoningActivityLoading}
          />
        </div>
        <DataTable
          columns={stratifiedReasoningColumns}
          data={reasoningActivity?.phases ?? []}
          emptyMessage="No stratified reasoning spend in the selected window"
        />
      </div>

      {(vendorsLoading || vendorRows.length > 0) && (
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-700/50">
            <h2 className="text-sm font-medium text-white">Cost By Vendor</h2>
            <p className="text-xs text-slate-500">
              LLM spend attributed to tracked B2B vendors ({vendorRows.length} vendors). Window-only rollup; provider/model filters do not change this section.
            </p>
          </div>
          {vendorsLoading ? (
            <DataTable columns={vendorColumns} data={[]} skeletonRows={5} />
          ) : (
            <DataTable columns={vendorColumns} data={vendorRows} emptyMessage="No vendor-attributed costs" />
          )}
        </div>
      )}

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatCard
          label="Tracked B2B Cost"
          value={formatCurrency(b2bEfficiency?.summary.tracked_cost_usd)}
          icon={<DollarSign className="h-4 w-4" />}
          skeleton={b2bEfficiencyLoading}
        />
        <StatCard
          label="Measured Runs"
          value={formatNumber(b2bEfficiency?.summary.measured_runs)}
          icon={<Workflow className="h-4 w-4" />}
          skeleton={b2bEfficiencyLoading}
        />
        <StatCard
          label="Tracked Witnesses"
          value={formatNumber(b2bEfficiency?.summary.tracked_witness_count)}
          icon={<Database className="h-4 w-4" />}
          skeleton={b2bEfficiencyLoading}
        />
        <StatCard
          label="Cost / Witness"
          value={formatMaybeCurrency(b2bEfficiency?.summary.cost_per_witness_usd)}
          icon={<GitCompareArrows className="h-4 w-4" />}
          skeleton={b2bEfficiencyLoading}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-700/50">
            <h2 className="text-sm font-medium text-white">B2B Cost By Vendor Pass</h2>
            <p className="text-xs text-slate-500">Extraction, repair, and reasoning spend split per vendor. Window-only rollup; provider/model filters do not change this section.</p>
          </div>
          {b2bEfficiencyLoading ? (
            <DataTable columns={vendorPassColumns} data={[]} skeletonRows={6} />
          ) : (
            <DataTable
              columns={vendorPassColumns}
              data={vendorPasses}
              emptyMessage="No vendor pass cost rows in the selected window"
            />
          )}
        </div>

        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-700/50">
            <h2 className="text-sm font-medium text-white">B2B Cost By Source</h2>
            <p className="text-xs text-slate-500">Spend, witness yield, repair pressure, and strict-gate suppression by source. Window-only rollup; provider/model filters do not change this section.</p>
          </div>
          {b2bEfficiencyLoading ? (
            <DataTable columns={sourceEfficiencyColumns} data={[]} skeletonRows={6} />
          ) : (
            <DataTable
              columns={sourceEfficiencyColumns}
              data={sourceEfficiency}
              emptyMessage="No source efficiency rows in the selected window"
            />
          )}
        </div>
      </div>

      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-700/50">
          <h2 className="text-sm font-medium text-white">Recent Pipeline Run Efficiency</h2>
          <p className="text-xs text-slate-500">Per-run cost, witness yield, strict-gate keep/drop counts, and secondary-write activity. Window-only rollup; provider/model filters do not change this section.</p>
        </div>
        {b2bEfficiencyLoading ? (
          <DataTable columns={b2bRunColumns} data={[]} skeletonRows={6} />
        ) : (
          <DataTable
            columns={b2bRunColumns}
            data={recentPipelineRuns}
            emptyMessage="No recent B2B pipeline runs in the selected window"
          />
        )}
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.1fr,0.9fr]">
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-700/50">
            <h2 className="text-sm font-medium text-white">Cost By Operation</h2>
            <p className="text-xs text-slate-500">Rollup by operation, provider, and model</p>
          </div>
          {loading ? (
            <DataTable columns={operationColumns} data={[]} skeletonRows={6} />
          ) : (
            <DataTable
              columns={operationColumns}
              data={operations}
              emptyMessage="No model-call cost rows match the current filters"
            />
          )}
        </div>

        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-700/50">
            <h2 className="text-sm font-medium text-white">Recent Calls</h2>
            <p className="text-xs text-slate-500">Granular provider-call economics with cache state</p>
          </div>
          {loading ? (
            <DataTable columns={recentColumns} data={[]} skeletonRows={6} />
          ) : (
            <DataTable
              columns={recentColumns}
              data={recentCalls}
              emptyMessage="No recent model calls match the current filters"
            />
          )}
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function PipelineReview() {
  const [activeTab, setActiveTab] = useState<TabKey>('queue')

  const {
    data: summary,
    loading: summaryLoading,
    refresh: refreshSummary,
    refreshing: summaryRefreshing,
  } = useApiData(() => fetchVisibilitySummary(720), [])

  const tabs: { key: TabKey; label: string }[] = [
    { key: 'queue', label: 'Queue' },
    { key: 'failures', label: 'Failures' },
    { key: 'quality', label: 'Quality Signals' },
    { key: 'audit', label: 'Audit Trail' },
    { key: 'costs', label: 'Costs' },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Shield className="h-6 w-6 text-cyan-400" />
          <div>
            <h1 className="text-2xl font-bold text-white">Operations</h1>
            <p className="mt-1 text-sm text-slate-400">
              Review failures, quality signals, audit history, and model-call economics in one place.
            </p>
          </div>
        </div>
        {activeTab !== 'costs' ? (
          <button
            onClick={refreshSummary}
            disabled={summaryRefreshing}
            className="flex items-center gap-2 px-3 py-2 text-sm text-slate-300 bg-slate-800/50 rounded-lg hover:bg-slate-700/50 transition-colors"
          >
            <RefreshCw className={clsx('h-4 w-4', summaryRefreshing && 'animate-spin')} />
            Refresh Overview
          </button>
        ) : null}
      </div>

      {/* Summary stat cards */}
      {activeTab !== 'costs' ? (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <StatCard
            label="Open Actionable"
            value={summary?.open_actionable ?? 0}
            icon={<AlertTriangle className="h-4 w-4" />}
            skeleton={summaryLoading}
          />
          <StatCard
            label="Failures 30d"
            value={summary?.failures_period ?? 0}
            icon={<XCircle className="h-4 w-4" />}
            skeleton={summaryLoading}
          />
          <StatCard
            label="Quarantines 30d"
            value={summary?.quarantines_period ?? 0}
            icon={<Clock className="h-4 w-4" />}
            skeleton={summaryLoading}
          />
          <StatCard
            label="Rejections 30d"
            value={summary?.rejections_period ?? 0}
            icon={<CheckCircle2 className="h-4 w-4" />}
            skeleton={summaryLoading}
          />
          <StatCard
            label="Recovered Retries 30d"
            value={summary?.recovered_validation_retries_period ?? 0}
            icon={<RefreshCw className="h-4 w-4" />}
            skeleton={summaryLoading}
          />
        </div>
      ) : null}

      {/* Tabs */}
      <div className="flex gap-1 border-b border-slate-700/50">
        {tabs.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={clsx(
              'px-4 py-2 text-sm font-medium transition-colors border-b-2',
              activeTab === key
                ? 'text-cyan-400 border-cyan-400'
                : 'text-slate-400 border-transparent hover:text-white',
            )}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'queue' && <QueueTab onRefresh={refreshSummary} />}
      {activeTab === 'failures' && <FailuresTab />}
      {activeTab === 'quality' && <QualityTab />}
      {activeTab === 'audit' && <AuditTab />}
      {activeTab === 'costs' && <CostsTab />}
    </div>
  )
}
