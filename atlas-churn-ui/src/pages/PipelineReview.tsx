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
  SynthesisValidationResult,
  AdminCostSummary,
  AdminCostOperation,
  AdminCostRecentCall,
} from '../types'
import {
  fetchVisibilitySummary,
  fetchVisibilityQueue,
  fetchVisibilityEvents,
  fetchArtifactAttempts,
  fetchEnrichmentQuarantines,
  fetchSynthesisValidationResults,
  resolveVisibilityReview,
  fetchAdminCostSummary,
  fetchAdminCostByOperation,
  fetchAdminCostRecent,
} from '../api/client'

type TabKey = 'queue' | 'failures' | 'quality' | 'audit' | 'costs'

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

function truncateLabel(value: string | null | undefined, max = 38): string {
  const text = String(value || '').trim()
  if (!text) return '--'
  if (text.length <= max) return text
  return `${text.slice(0, max - 1)}…`
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
            {r.reason_code ? `${formatVisibilityCode(r.reason_code)} · ` : ''}
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
      render: (r) => (
        <div className="max-w-sm">
          <span className="text-xs text-slate-400 line-clamp-2">{r.summary}</span>
          <span className="text-[11px] text-slate-600 block">
            {r.field_path || '--'}
          </span>
        </div>
      ),
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
  const loading = quarantinesLoading || validationsLoading
  const refreshing = quarantinesRefreshing || validationsRefreshing
  const error = quarantinesError || validationsError

  const refresh = () => {
    refreshQuarantines()
    refreshValidations()
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

interface CostTabData {
  summary: AdminCostSummary
  operations: AdminCostOperation[]
  recent: AdminCostRecentCall[]
}

function CostsTab() {
  const [days, setDays] = useState('30')
  const [providerFilter, setProviderFilter] = useState('')
  const [modelFilter, setModelFilter] = useState('')
  const [spanFilter, setSpanFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [cacheFilter, setCacheFilter] = useState('')

  const {
    data,
    loading,
    error,
    refresh,
    refreshing,
  } = useApiData<CostTabData>(
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
          cache_only: cacheOnly,
        }),
        fetchAdminCostRecent({
          days: numericDays,
          limit: 100,
          provider: providerFilter || undefined,
          model: modelFilter || undefined,
          span_name: spanFilter || undefined,
          status: statusFilter || undefined,
          cache_only: cacheOnly,
        }),
      ])
      return {
        summary,
        operations: operationsResponse.operations,
        recent: recentResponse.calls,
      }
    },
    [days, providerFilter, modelFilter, spanFilter, statusFilter, cacheFilter],
  )

  const operations = data?.operations ?? []
  const recentCalls = data?.recent ?? []
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

  const operationColumns: Column<AdminCostOperation>[] = [
    {
      key: 'span_name',
      header: 'Operation',
      render: (row) => (
        <div className="max-w-[260px]">
          <p className="truncate text-sm text-white">{row.span_name}</p>
          <p className="truncate text-xs text-slate-500">{row.provider} · {truncateLabel(row.model, 48)}</p>
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
            {[row.provider, row.model, row.detail].filter(Boolean).map((value) => truncateLabel(value as string, 44)).join(' · ') || row.span_name}
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
  ]

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-4">
        <Filter className="h-4 w-4 text-slate-500" />
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
          label="Cache"
          value={cacheFilter}
          onChange={setCacheFilter}
          options={[
            { value: '', label: 'All calls' },
            { value: 'cached', label: 'Cached only' },
            { value: 'uncached', label: 'Uncached only' },
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

      <div className="rounded-xl border border-slate-700/50 bg-slate-900/40 p-4 text-sm text-slate-400">
        Provider/model calls only. Local exact-cache hits are intentionally excluded from these tables so per-call economics stay truthful.
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm">
          {error.message}
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
