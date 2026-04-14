import type { ReactNode } from 'react'
import { clsx } from 'clsx'
import {
  Ban,
  Bell,
  Building2,
  CheckCircle2,
  Clock,
  GitCompareArrows,
  RefreshCw,
  Shield,
  Workflow,
} from 'lucide-react'
import DataTable from './DataTable'
import StatCard from './StatCard'
import type { Column } from './DataTable'
import type {
  CompanySignalCandidateGroup,
  CompanySignalCandidateGroupSummary,
  CompanySignalReviewImpactSummary,
} from '../types'

export type CompanySignalQueuePreset = {
  label: string
  filters: Record<string, unknown>
  helpText: string
}

export type CompanySignalImpactDriverRow = {
  label: string
  count: number | null
}

export type CompanySignalTrendSliceRow = {
  label: string
  rationale: string
  queueFilters: Record<string, unknown>
  pending: number
  actionable: number
  blocked: number
  overdue: number
  oldestPendingAgeDays: number | null
  status?: string
  direction?: string
  delta?: number | null
}

type QueueSnapshot = {
  pending: number
  actionable: number
  blocked: number
  overdue: number
  oldestPendingAgeDays: number | null
}

type Props = {
  reviewStatusFilter: string
  priorityBandFilter: string
  candidateBucketFilter: string
  sourceFilter: string
  gapReasonFilter: string
  decisionMakersOnly: boolean
  impactSummary: CompanySignalReviewImpactSummary | null | undefined
  impactSummaryLoading: boolean
  queuePresetFilters: CompanySignalQueuePreset[]
  activePresetLabel: string
  onApplyPreset: (filters: Record<string, unknown>) => void
  previewFilters: Record<string, unknown> | null
  previewLabel: string | null
  previewIsCurrentSlice: boolean
  onApplyPreview: () => void
  onClearPreview: () => void
  resolvedPreviewFilters: Record<string, unknown> | null
  currentFilters: Record<string, unknown>
  summarizeDelta: (currentFilters: Record<string, unknown>, nextFilters: Record<string, unknown>) => string
  previewSnapshot: QueueSnapshot | null
  describeSnapshotImpact: (snapshot: QueueSnapshot) => string
  previewMatchedCount: number
  visibleGroupCount: number
  impactTopVendors: CompanySignalImpactDriverRow[]
  impactRebuildReasons: CompanySignalImpactDriverRow[]
  trendAlerts: CompanySignalTrendSliceRow[]
  trendQueueRankings: CompanySignalTrendSliceRow[]
  isActiveSlice: (filters: Record<string, unknown>) => boolean
  onPreviewSlice: (label: string, filters: Record<string, unknown>, snapshot: QueueSnapshot) => void
  onApplyFilters: (filters: Record<string, unknown>) => void
  describeSlice: (filters: Record<string, unknown>) => string
  summaryLoading: boolean
  totals: CompanySignalCandidateGroupSummary['totals'] | null | undefined
  gapReasons: CompanySignalCandidateGroupSummary['gap_reasons']
  actionableTopVendors: CompanySignalCandidateGroupSummary['actionable_top_vendors']
  topVendors: CompanySignalCandidateGroupSummary['top_vendors']
  isPendingView: boolean
  allGroupsSelected: boolean
  groups: CompanySignalCandidateGroup[]
  selectedCount: number
  bulkActionLoading: boolean
  onToggleSelectAll: () => void
  onBulkApprove: () => void
  onBulkSuppress: () => void
  actionNotes: string
  onActionNotesChange: (value: string) => void
  triggerRebuild: boolean
  onTriggerRebuildChange: (checked: boolean) => void
  actionMessage: string | null
  actionError: string | null
  groupsLoading: boolean
  groupColumns: Column<CompanySignalCandidateGroup>[]
  onGroupRowClick: (row: CompanySignalCandidateGroup) => void
  detailDrawer?: ReactNode
  formatCompanySignalCode: (value: string | null | undefined) => string
  formatPriorityBandLabel: (value: string | null | undefined) => string
  formatNumber: (value: number | null | undefined) => string
  formatMaybeNumber: (value: number | null | undefined) => string
  formatMaybePercent: (value: number | null | undefined) => string
}

function hasQueueFilters(filters: Record<string, unknown>): boolean {
  return Object.keys(filters).length > 0
}

function OperatorFocusSection({
  loading,
  operatorFocus,
  presets,
  activePresetLabel,
  onApplyPreset,
}: {
  loading: boolean
  operatorFocus: CompanySignalReviewImpactSummary['operator_focus'] | null | undefined
  presets: CompanySignalQueuePreset[]
  activePresetLabel: string
  onApplyPreset: (filters: Record<string, unknown>) => void
}) {
  return (
    <div className="border-b border-slate-700/50 px-4 py-3">
      <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 p-4">
        <div className="flex items-start justify-between gap-3">
          <h3 className="text-sm font-medium text-white">Operator Focus</h3>
          {presets.length > 0 ? (
            <div className="flex flex-wrap items-center justify-end gap-2">
              {presets.map((preset) => (
                <button
                  key={preset.label}
                  onClick={() => onApplyPreset(preset.filters)}
                  className={clsx(
                    'rounded border px-2.5 py-1 text-xs transition',
                    activePresetLabel === preset.label
                      ? 'border-cyan-300 bg-cyan-400/20 text-cyan-100'
                      : 'border-cyan-500/30 bg-cyan-500/10 text-cyan-300 hover:bg-cyan-500/20',
                  )}
                >
                  {preset.label}
                </button>
              ))}
            </div>
          ) : null}
        </div>
        {loading ? (
          <p className="mt-2 text-xs text-slate-500">Loading recent review impact...</p>
        ) : operatorFocus?.status && operatorFocus.status !== 'no_data' ? (
          <>
            <p className="mt-2 text-sm text-slate-200">
              {operatorFocus.action || operatorFocus.reason || '--'}
            </p>
            <p className="mt-1 text-xs text-slate-500">
              {operatorFocus.rationale || '--'}
            </p>
            <p className="mt-2 text-[11px] text-slate-400">Active preset: {activePresetLabel}</p>
            {presets.length > 0 ? (
              <div className="mt-2 space-y-1">
                {presets.map((preset) => (
                  <p key={`${preset.label}-help`} className="text-[11px] text-slate-500">
                    {preset.helpText}
                  </p>
                ))}
              </div>
            ) : null}
          </>
        ) : (
          <p className="mt-2 text-xs text-slate-500">
            No review-action history yet. Once analysts approve or suppress groups, impact and rebuild outcomes will show here.
          </p>
        )}
      </div>
    </div>
  )
}

function PreviewBanner({
  previewFilters,
  previewLabel,
  previewIsCurrentSlice,
  onApply,
  onClear,
  resolvedPreviewFilters,
  currentFilters,
  summarizeDelta,
  previewSnapshot,
  describeSnapshotImpact,
  matchedCount,
  visibleCount,
  formatNumber,
}: {
  previewFilters: Record<string, unknown> | null
  previewLabel: string | null
  previewIsCurrentSlice: boolean
  onApply: () => void
  onClear: () => void
  resolvedPreviewFilters: Record<string, unknown> | null
  currentFilters: Record<string, unknown>
  summarizeDelta: (currentFilters: Record<string, unknown>, nextFilters: Record<string, unknown>) => string
  previewSnapshot: QueueSnapshot | null
  describeSnapshotImpact: (snapshot: QueueSnapshot) => string
  matchedCount: number
  visibleCount: number
  formatNumber: (value: number | null | undefined) => string
}) {
  if (!previewFilters) return null

  return (
    <div className="border-b border-slate-700/50 px-4 py-3">
      <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-medium text-white">Preview Queue Slice</h3>
            <p className="mt-1 text-xs text-slate-400">
              {previewLabel || 'Queued slice preview'}
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={onApply}
              disabled={previewIsCurrentSlice}
              className={clsx(
                'rounded border px-2.5 py-1 text-xs transition',
                previewIsCurrentSlice
                  ? 'cursor-not-allowed border-cyan-300/40 bg-cyan-400/15 text-cyan-100'
                  : 'border-cyan-500/30 bg-cyan-500/10 text-cyan-300 hover:bg-cyan-500/20',
              )}
            >
              {previewIsCurrentSlice ? 'Already Applied' : 'Apply Preview'}
            </button>
            <button
              onClick={onClear}
              className="rounded border border-slate-700/60 px-2.5 py-1 text-xs text-slate-300 transition hover:border-slate-500 hover:text-white"
            >
              Clear Preview
            </button>
          </div>
        </div>
        <div className="mt-3 space-y-1 text-xs">
          {resolvedPreviewFilters ? (
            <p className="text-slate-300">
              {summarizeDelta(currentFilters, resolvedPreviewFilters)}
            </p>
          ) : null}
          {previewSnapshot ? (
            <p className="text-slate-400">
              {describeSnapshotImpact(previewSnapshot)}
            </p>
          ) : null}
          <p className="text-slate-500">
            {formatNumber(matchedCount)} of {formatNumber(visibleCount)} visible groups match preview
          </p>
        </div>
      </div>
    </div>
  )
}

function ReviewActivitySection({
  loading,
  impactSummary,
  topVendors,
  rebuildReasons,
  formatNumber,
  formatMaybeNumber,
  formatMaybePercent,
  formatCompanySignalCode,
}: {
  loading: boolean
  impactSummary: CompanySignalReviewImpactSummary | null | undefined
  topVendors: CompanySignalImpactDriverRow[]
  rebuildReasons: CompanySignalImpactDriverRow[]
  formatNumber: (value: number | null | undefined) => string
  formatMaybeNumber: (value: number | null | undefined) => string
  formatMaybePercent: (value: number | null | undefined) => string
  formatCompanySignalCode: (value: string | null | undefined) => string
}) {
  return (
    <div className="grid gap-4 border-b border-slate-700/50 p-4 xl:grid-cols-[0.9fr,1.1fr]">
      <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 p-4">
        <h3 className="text-sm font-medium text-white">Review Activity</h3>
        {loading ? (
          <p className="mt-2 text-xs text-slate-500">Loading activity trends...</p>
        ) : (impactSummary?.totals.total_actions || 0) > 0 ? (
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <div className="rounded border border-slate-700/50 bg-slate-900/50 p-3">
              <p className="text-[11px] uppercase tracking-wide text-slate-500">Recent 7d</p>
              <div className="mt-2 space-y-1 text-xs text-slate-300">
                <p>{formatNumber(impactSummary?.trend_comparison.recent.action_count)} actions</p>
                <p>{formatNumber(impactSummary?.trend_comparison.recent.approvals)} approvals</p>
                <p>{formatNumber(impactSummary?.trend_comparison.recent.suppressions)} suppressions</p>
                <p>{formatNumber(impactSummary?.trend_comparison.recent.rebuild_triggered)} rebuilds</p>
              </div>
            </div>
            <div className="rounded border border-slate-700/50 bg-slate-900/50 p-3">
              <p className="text-[11px] uppercase tracking-wide text-slate-500">Vs Prior 7d</p>
              <div className="mt-2 space-y-1 text-xs text-slate-300">
                <p>{formatMaybeNumber(impactSummary?.trend_comparison.deltas.action_count)} action delta</p>
                <p>{formatMaybeNumber(impactSummary?.trend_comparison.deltas.approvals)} approval delta</p>
                <p>{formatMaybeNumber(impactSummary?.trend_comparison.deltas.suppressions)} suppression delta</p>
                <p>{formatMaybePercent(impactSummary?.trend_comparison.recent.rebuild_trigger_rate)} rebuild trigger rate</p>
              </div>
            </div>
          </div>
        ) : (
          <p className="mt-2 text-xs text-slate-500">
            No company-signal review actions have been recorded in this window.
          </p>
        )}
      </div>

      <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 p-4">
        <h3 className="text-sm font-medium text-white">Recent Impact Drivers</h3>
        {loading ? (
          <p className="mt-2 text-xs text-slate-500">Loading vendor and rebuild drivers...</p>
        ) : topVendors.length > 0 || rebuildReasons.length > 0 ? (
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            <div className="space-y-3">
              <p className="text-[11px] uppercase tracking-wide text-slate-500">Top Vendors</p>
              {topVendors.slice(0, 4).map((row) => (
                <div key={`impact-vendor-${row.label}`} className="rounded border border-slate-700/50 p-3">
                  <p className="text-sm text-white">{row.label}</p>
                  <p className="mt-1 text-xs text-slate-500">{formatNumber(row.count)} actions</p>
                </div>
              ))}
            </div>
            <div className="space-y-3">
              <p className="text-[11px] uppercase tracking-wide text-slate-500">Rebuild Reasons</p>
              {rebuildReasons.slice(0, 4).map((row) => (
                <div key={`impact-rebuild-${row.label}`} className="rounded border border-slate-700/50 p-3">
                  <p className="text-sm text-white">{formatCompanySignalCode(row.label)}</p>
                  <p className="mt-1 text-xs text-slate-500">{formatNumber(row.count)} events</p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <p className="mt-2 text-xs text-slate-500">
            No vendor or rebuild drivers recorded in the current review window.
          </p>
        )}
      </div>
    </div>
  )
}

function TrendSections({
  loading,
  trendAlerts,
  trendQueueRankings,
  isActiveSlice,
  onPreviewSlice,
  onApplyFilters,
  describeSlice,
  describeSnapshotImpact,
  formatCompanySignalCode,
  formatNumber,
  formatMaybeNumber,
}: {
  loading: boolean
  trendAlerts: CompanySignalTrendSliceRow[]
  trendQueueRankings: CompanySignalTrendSliceRow[]
  isActiveSlice: (filters: Record<string, unknown>) => boolean
  onPreviewSlice: (label: string, filters: Record<string, unknown>, snapshot: QueueSnapshot) => void
  onApplyFilters: (filters: Record<string, unknown>) => void
  describeSlice: (filters: Record<string, unknown>) => string
  describeSnapshotImpact: (snapshot: QueueSnapshot) => string
  formatCompanySignalCode: (value: string | null | undefined) => string
  formatNumber: (value: number | null | undefined) => string
  formatMaybeNumber: (value: number | null | undefined) => string
}) {
  return (
    <div className="grid gap-4 border-b border-slate-700/50 p-4 xl:grid-cols-2">
      <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 p-4">
        <h3 className="text-sm font-medium text-white">Trend Alerts</h3>
        {loading ? (
          <p className="mt-2 text-xs text-slate-500">Loading alert recommendations...</p>
        ) : trendAlerts.length > 0 ? (
          <div className="mt-3 space-y-3">
            {trendAlerts.slice(0, 3).map((row) => (
              <div key={`trend-alert-${row.label}`} className="rounded border border-slate-700/50 p-3">
                {isActiveSlice(row.queueFilters) ? (
                  <p className="mb-2 text-[11px] uppercase tracking-wide text-cyan-300">Current queue slice</p>
                ) : null}
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-sm text-white">{formatCompanySignalCode(row.label)}</p>
                    <p className="mt-1 text-xs text-slate-500">{row.rationale || '--'}</p>
                    {describeSlice(row.queueFilters) ? (
                      <p className="mt-2 text-[11px] text-slate-400">{describeSlice(row.queueFilters)}</p>
                    ) : null}
                    <p className="mt-1 text-[11px] text-slate-500">
                      {describeSnapshotImpact({
                        pending: row.pending,
                        actionable: row.actionable,
                        blocked: row.blocked,
                        overdue: row.overdue,
                        oldestPendingAgeDays: row.oldestPendingAgeDays,
                      })}
                    </p>
                  </div>
                  <span className="rounded border border-slate-700/50 px-2 py-0.5 text-[11px] uppercase tracking-wide text-slate-400">
                    {row.status || 'watch'}
                  </span>
                </div>
                <div className="mt-2 flex items-center justify-between gap-3 text-[11px] text-slate-400">
                  <div className="flex flex-wrap gap-3">
                    <span>
                      {row.direction ? `${row.direction} ${formatMaybeNumber(row.delta)}` : formatMaybeNumber(row.delta)}
                    </span>
                    {row.pending > 0 ? <span>{formatNumber(row.pending)} pending</span> : null}
                    {row.oldestPendingAgeDays != null ? <span>{formatMaybeNumber(row.oldestPendingAgeDays)} oldest days</span> : null}
                  </div>
                  {hasQueueFilters(row.queueFilters) ? (
                    <div className="flex flex-wrap gap-2">
                      {!isActiveSlice(row.queueFilters) ? (
                        <button
                          onClick={() =>
                            onPreviewSlice(`Trend alert: ${formatCompanySignalCode(row.label)}`, row.queueFilters, {
                              pending: row.pending,
                              actionable: row.actionable,
                              blocked: row.blocked,
                              overdue: row.overdue,
                              oldestPendingAgeDays: row.oldestPendingAgeDays,
                            })}
                          className="rounded border border-slate-600/60 px-2 py-1 text-[11px] text-slate-300 transition hover:border-slate-500 hover:text-white"
                        >
                          Preview Slice
                        </button>
                      ) : null}
                      <button
                        onClick={() => onApplyFilters(row.queueFilters)}
                        disabled={isActiveSlice(row.queueFilters)}
                        className={clsx(
                          'rounded border px-2 py-1 text-[11px] transition',
                          isActiveSlice(row.queueFilters)
                            ? 'cursor-not-allowed border-cyan-300/40 bg-cyan-400/15 text-cyan-100'
                            : 'border-cyan-500/30 bg-cyan-500/10 text-cyan-300 hover:bg-cyan-500/20',
                        )}
                      >
                        {isActiveSlice(row.queueFilters) ? 'Current Slice' : 'Apply Alert Slice'}
                      </button>
                    </div>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="mt-2 text-xs text-slate-500">No trend alerts are active for the current review-impact window.</p>
        )}
      </div>

      <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 p-4">
        <h3 className="text-sm font-medium text-white">Ranked Queue Slices</h3>
        {loading ? (
          <p className="mt-2 text-xs text-slate-500">Loading ranked queue recommendations...</p>
        ) : trendQueueRankings.length > 0 ? (
          <div className="mt-3 space-y-3">
            {trendQueueRankings.slice(0, 3).map((row) => (
              <div key={`trend-queue-${row.label}`} className="rounded border border-slate-700/50 p-3">
                {isActiveSlice(row.queueFilters) ? (
                  <p className="mb-2 text-[11px] uppercase tracking-wide text-cyan-300">Current queue slice</p>
                ) : null}
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-sm text-white">{formatCompanySignalCode(row.label)}</p>
                    <p className="mt-1 text-xs text-slate-500">
                      {row.rationale !== '--' ? row.rationale : 'Backend-ranked queue slice for operator focus.'}
                    </p>
                    {describeSlice(row.queueFilters) ? (
                      <p className="mt-2 text-[11px] text-slate-400">{describeSlice(row.queueFilters)}</p>
                    ) : null}
                    <p className="mt-1 text-[11px] text-slate-500">
                      {describeSnapshotImpact({
                        pending: row.pending,
                        actionable: row.actionable,
                        blocked: row.blocked,
                        overdue: row.overdue,
                        oldestPendingAgeDays: row.oldestPendingAgeDays,
                      })}
                    </p>
                  </div>
                  {hasQueueFilters(row.queueFilters) ? (
                    <div className="flex flex-wrap gap-2">
                      {!isActiveSlice(row.queueFilters) ? (
                        <button
                          onClick={() =>
                            onPreviewSlice(`Ranked slice: ${formatCompanySignalCode(row.label)}`, row.queueFilters, {
                              pending: row.pending,
                              actionable: row.actionable,
                              blocked: row.blocked,
                              overdue: row.overdue,
                              oldestPendingAgeDays: row.oldestPendingAgeDays,
                            })}
                          className="rounded border border-slate-600/60 px-2 py-1 text-[11px] text-slate-300 transition hover:border-slate-500 hover:text-white"
                        >
                          Preview Slice
                        </button>
                      ) : null}
                      <button
                        onClick={() => onApplyFilters(row.queueFilters)}
                        disabled={isActiveSlice(row.queueFilters)}
                        className={clsx(
                          'rounded border px-2 py-1 text-[11px] transition',
                          isActiveSlice(row.queueFilters)
                            ? 'cursor-not-allowed border-cyan-300/40 bg-cyan-400/15 text-cyan-100'
                            : 'border-cyan-500/30 bg-cyan-500/10 text-cyan-300 hover:bg-cyan-500/20',
                        )}
                      >
                        {isActiveSlice(row.queueFilters) ? 'Current Slice' : 'Apply Queue Slice'}
                      </button>
                    </div>
                  ) : null}
                </div>
                <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-slate-400">
                  <span>{formatNumber(row.pending)} pending</span>
                  <span>{formatNumber(row.actionable)} actionable</span>
                  <span>{formatNumber(row.blocked)} blocked</span>
                  <span>{formatNumber(row.overdue)} overdue</span>
                  {row.oldestPendingAgeDays != null ? <span>{formatMaybeNumber(row.oldestPendingAgeDays)} oldest days</span> : null}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="mt-2 text-xs text-slate-500">No ranked queue slices are available for this review-impact window.</p>
        )}
      </div>
    </div>
  )
}

function QueueSummarySection({
  loading,
  totals,
  gapReasons,
  actionableTopVendors,
  topVendors,
  formatCompanySignalCode,
  formatNumber,
}: {
  loading: boolean
  totals: CompanySignalCandidateGroupSummary['totals'] | null | undefined
  gapReasons: CompanySignalCandidateGroupSummary['gap_reasons']
  actionableTopVendors: CompanySignalCandidateGroupSummary['actionable_top_vendors']
  topVendors: CompanySignalCandidateGroupSummary['top_vendors']
  formatCompanySignalCode: (value: string | null | undefined) => string
  formatNumber: (value: number | null | undefined) => string
}) {
  return (
    <>
      <div className="grid grid-cols-2 gap-4 border-b border-slate-700/50 p-4 md:grid-cols-3 xl:grid-cols-6">
        <StatCard label="Pending Groups" value={formatNumber(totals?.pending_groups)} icon={<Workflow className="h-4 w-4" />} skeleton={loading} />
        <StatCard label="Actionable" value={formatNumber(totals?.actionable_pending_groups)} icon={<CheckCircle2 className="h-4 w-4" />} skeleton={loading} />
        <StatCard label="Blocked" value={formatNumber(totals?.blocked_pending_groups)} icon={<Ban className="h-4 w-4" />} skeleton={loading} />
        <StatCard label="Canonical Ready" value={formatNumber(totals?.canonical_ready_groups)} icon={<Shield className="h-4 w-4" />} skeleton={loading} />
        <StatCard label="Decision Maker" value={formatNumber(totals?.decision_maker_groups)} icon={<Building2 className="h-4 w-4" />} skeleton={loading} />
        <StatCard label="Overdue" value={formatNumber(totals?.overdue_pending_groups)} icon={<Clock className="h-4 w-4" />} skeleton={loading} />
      </div>

      <div className="grid gap-4 border-b border-slate-700/50 p-4 xl:grid-cols-[0.95fr,1.05fr]">
        <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 p-4">
          <h3 className="text-sm font-medium text-white">Blocking Reasons</h3>
          <p className="mt-1 text-xs text-slate-500">Why groups are still held in analyst review.</p>
          <div className="mt-3 space-y-3">
            {loading ? (
              Array.from({ length: 3 }, (_, index) => (
                <div key={index} className="animate-pulse rounded-lg border border-slate-700/50 p-3">
                  <div className="h-4 w-1/2 rounded bg-slate-700/50" />
                  <div className="mt-2 h-3 w-1/3 rounded bg-slate-800/50" />
                </div>
              ))
            ) : gapReasons.length > 0 ? (
              gapReasons.slice(0, 4).map((row) => (
                <div key={row.gap_reason} className="rounded-lg border border-slate-700/50 p-3">
                  <p className="text-sm text-white">{formatCompanySignalCode(row.gap_reason)}</p>
                  <p className="mt-1 text-xs text-slate-500">
                    {formatNumber(row.group_count)} groups | {formatNumber(row.review_count)} reviews
                  </p>
                </div>
              ))
            ) : (
              <p className="text-sm text-slate-500">No blocking reasons in the selected scope.</p>
            )}
          </div>
        </div>

        <div className="rounded-xl border border-slate-700/50 bg-slate-950/30 p-4">
          <h3 className="text-sm font-medium text-white">Actionable Vendors</h3>
          <p className="mt-1 text-xs text-slate-500">Vendors with the highest current operator load.</p>
          <div className="mt-3 space-y-3">
            {loading ? (
              Array.from({ length: 3 }, (_, index) => (
                <div key={index} className="animate-pulse rounded-lg border border-slate-700/50 p-3">
                  <div className="h-4 w-1/3 rounded bg-slate-700/50" />
                  <div className="mt-2 h-3 w-2/3 rounded bg-slate-800/50" />
                </div>
              ))
            ) : actionableTopVendors.length > 0 ? (
              actionableTopVendors.slice(0, 5).map((row) => (
                <div key={row.vendor_name} className="rounded-lg border border-slate-700/50 p-3">
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-sm text-white">{row.vendor_name}</p>
                    <p className="text-xs text-slate-400">{formatNumber(row.actionable_group_count)} groups</p>
                  </div>
                  <p className="mt-1 text-xs text-slate-500">
                    {formatNumber(row.high_group_count)} high | {formatNumber(row.medium_group_count)} medium | {formatNumber(row.actionable_decision_maker_groups)} DM
                  </p>
                </div>
              ))
            ) : topVendors.length > 0 ? (
              topVendors.slice(0, 5).map((row) => (
                <div key={row.vendor_name} className="rounded-lg border border-slate-700/50 p-3">
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-sm text-white">{row.vendor_name}</p>
                    <p className="text-xs text-slate-400">{formatNumber(row.group_count)} groups</p>
                  </div>
                  <p className="mt-1 text-xs text-slate-500">
                    {formatNumber(row.review_count)} reviews | {formatNumber(row.pending_groups)} pending
                  </p>
                </div>
              ))
            ) : (
              <p className="text-sm text-slate-500">No vendor pressure in the selected scope.</p>
            )}
          </div>
        </div>
      </div>
    </>
  )
}

function GroupsSection({
  isPendingView,
  allGroupsSelected,
  groups,
  selectedCount,
  bulkActionLoading,
  onToggleSelectAll,
  onBulkApprove,
  onBulkSuppress,
  actionNotes,
  onActionNotesChange,
  triggerRebuild,
  onTriggerRebuildChange,
  actionMessage,
  actionError,
  groupsLoading,
  groupColumns,
  onGroupRowClick,
  formatNumber,
}: {
  isPendingView: boolean
  allGroupsSelected: boolean
  groups: CompanySignalCandidateGroup[]
  selectedCount: number
  bulkActionLoading: boolean
  onToggleSelectAll: () => void
  onBulkApprove: () => void
  onBulkSuppress: () => void
  actionNotes: string
  onActionNotesChange: (value: string) => void
  triggerRebuild: boolean
  onTriggerRebuildChange: (checked: boolean) => void
  actionMessage: string | null
  actionError: string | null
  groupsLoading: boolean
  groupColumns: Column<CompanySignalCandidateGroup>[]
  onGroupRowClick: (row: CompanySignalCandidateGroup) => void
  formatNumber: (value: number | null | undefined) => string
}) {
  return (
    <div className="rounded-b-xl border-t border-slate-700/50 bg-slate-950/20">
      <div className="border-b border-slate-700/50 px-4 py-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-medium text-white">
              {isPendingView ? 'Pending Candidate Groups' : 'Reviewed Candidate Groups'}
            </h3>
            <p className="text-xs text-slate-500">
              {isPendingView
                ? 'Approve groups that are good enough to materialize, suppress the rest.'
                : 'Audit reviewed groups, operator notes, and rebuild decisions.'}
            </p>
          </div>
          {isPendingView ? (
            <div className="flex flex-wrap items-center gap-2">
              <button
                onClick={onToggleSelectAll}
                className="rounded border border-slate-700/60 px-2 py-1 text-xs text-slate-300 transition hover:border-cyan-500/60 hover:text-white"
              >
                {allGroupsSelected ? 'Clear All' : 'Select All'}
              </button>
              <button
                onClick={onBulkApprove}
                disabled={selectedCount === 0 || bulkActionLoading}
                className="rounded border border-emerald-500/30 bg-emerald-500/10 px-2 py-1 text-xs text-emerald-300 transition hover:bg-emerald-500/20 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Approve Selected
              </button>
              <button
                onClick={onBulkSuppress}
                disabled={selectedCount === 0 || bulkActionLoading}
                className="rounded border border-rose-500/30 bg-rose-500/10 px-2 py-1 text-xs text-rose-300 transition hover:bg-rose-500/20 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Suppress Selected
              </button>
            </div>
          ) : (
            <p className="text-xs text-slate-500">Review actions are disabled in audit mode.</p>
          )}
        </div>
        {isPendingView && selectedCount > 0 ? (
          <p className="mt-2 text-xs text-slate-400">
            {formatNumber(selectedCount)} candidate {selectedCount === 1 ? 'group' : 'groups'} selected
          </p>
        ) : null}
        {isPendingView ? (
          <div className="mt-3 grid gap-3 lg:grid-cols-[minmax(0,1fr),auto]">
            <label className="block">
              <span className="mb-1 block text-[11px] uppercase tracking-wide text-slate-500">Review notes</span>
              <textarea
                aria-label="Company signal review notes"
                value={actionNotes}
                onChange={(event) => onActionNotesChange(event.target.value)}
                rows={2}
                placeholder="Capture why this group should be promoted or suppressed."
                className="w-full rounded border border-slate-700/60 bg-slate-900/60 px-3 py-2 text-sm text-white placeholder:text-slate-500 focus:border-cyan-500/60 focus:outline-none"
              />
            </label>
            <label className="flex items-center gap-2 self-start rounded border border-slate-700/60 bg-slate-900/40 px-3 py-2 text-xs text-slate-300">
              <input
                aria-label="Trigger rebuild after review"
                type="checkbox"
                checked={triggerRebuild}
                onChange={(event) => onTriggerRebuildChange(event.target.checked)}
                className="h-4 w-4 rounded border-slate-600 bg-slate-900 text-cyan-500 focus:ring-cyan-500"
              />
              Trigger rebuild after review
            </label>
          </div>
        ) : null}
        {actionMessage ? <p className="mt-2 text-xs text-cyan-300">{actionMessage}</p> : null}
        {actionError ? <p className="mt-2 text-xs text-rose-300">{actionError}</p> : null}
      </div>
      {groupsLoading ? (
        <DataTable columns={groupColumns} data={[]} skeletonRows={5} />
      ) : (
        <DataTable
          columns={groupColumns}
          data={groups}
          onRowClick={onGroupRowClick}
          emptyMessage="No company-signal candidate groups in the selected scope"
        />
      )}
    </div>
  )
}

export default function PipelineReviewCompanySignalQueue({
  reviewStatusFilter,
  priorityBandFilter,
  candidateBucketFilter,
  sourceFilter,
  gapReasonFilter,
  decisionMakersOnly,
  impactSummary,
  impactSummaryLoading,
  queuePresetFilters,
  activePresetLabel,
  onApplyPreset,
  previewFilters,
  previewLabel,
  previewIsCurrentSlice,
  onApplyPreview,
  onClearPreview,
  resolvedPreviewFilters,
  currentFilters,
  summarizeDelta,
  previewSnapshot,
  describeSnapshotImpact,
  previewMatchedCount,
  visibleGroupCount,
  impactTopVendors,
  impactRebuildReasons,
  trendAlerts,
  trendQueueRankings,
  isActiveSlice,
  onPreviewSlice,
  onApplyFilters,
  describeSlice,
  summaryLoading,
  totals,
  gapReasons,
  actionableTopVendors,
  topVendors,
  isPendingView,
  allGroupsSelected,
  groups,
  selectedCount,
  bulkActionLoading,
  onToggleSelectAll,
  onBulkApprove,
  onBulkSuppress,
  actionNotes,
  onActionNotesChange,
  triggerRebuild,
  onTriggerRebuildChange,
  actionMessage,
  actionError,
  groupsLoading,
  groupColumns,
  onGroupRowClick,
  detailDrawer,
  formatCompanySignalCode,
  formatPriorityBandLabel,
  formatNumber,
  formatMaybeNumber,
  formatMaybePercent,
}: Props) {
  return (
    <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl overflow-hidden">
      <div className="flex items-center justify-between gap-3 border-b border-slate-700/50 px-4 py-3">
        <div>
          <h2 className="text-sm font-medium text-white">Company Signal Queue</h2>
          <p className="text-xs text-slate-500">
            {isPendingView
              ? 'Analyst-review candidate groups waiting on approve or suppress decisions.'
              : 'Reviewed company-signal groups for operator audit and follow-up.'}
          </p>
        </div>
        <div className="text-right text-xs text-slate-500">
          <p>{formatCompanySignalCode(reviewStatusFilter)} groups</p>
          <p>{formatPriorityBandLabel(priorityBandFilter)} priority</p>
          <p>{formatCompanySignalCode(candidateBucketFilter)}</p>
          {sourceFilter ? <p>{formatCompanySignalCode(sourceFilter)} source</p> : null}
          {gapReasonFilter ? <p>{formatCompanySignalCode(gapReasonFilter)}</p> : null}
          {decisionMakersOnly ? <p>Decision makers only</p> : null}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 border-b border-slate-700/50 p-4 md:grid-cols-3 xl:grid-cols-6">
        <StatCard label="Review Actions 30d" value={formatNumber(impactSummary?.totals.total_actions)} icon={<CheckCircle2 className="h-4 w-4" />} skeleton={impactSummaryLoading} />
        <StatCard label="Approvals 30d" value={formatNumber(impactSummary?.totals.approvals)} icon={<Bell className="h-4 w-4" />} skeleton={impactSummaryLoading} />
        <StatCard label="Suppressions 30d" value={formatNumber(impactSummary?.totals.suppressions)} icon={<Ban className="h-4 w-4" />} skeleton={impactSummaryLoading} />
        <StatCard label="Signal Creates" value={formatNumber(impactSummary?.totals.company_signal_creations)} icon={<Shield className="h-4 w-4" />} skeleton={impactSummaryLoading} />
        <StatCard label="Rebuild Triggered" value={formatNumber(impactSummary?.totals.rebuild_triggered)} icon={<RefreshCw className="h-4 w-4" />} skeleton={impactSummaryLoading} />
        <StatCard label="Effect Rate" value={formatMaybePercent(impactSummary?.totals.company_signal_effect_rate)} icon={<GitCompareArrows className="h-4 w-4" />} skeleton={impactSummaryLoading} />
      </div>

      <OperatorFocusSection
        loading={impactSummaryLoading}
        operatorFocus={impactSummary?.operator_focus}
        presets={queuePresetFilters}
        activePresetLabel={activePresetLabel}
        onApplyPreset={onApplyPreset}
      />

      <PreviewBanner
        previewFilters={previewFilters}
        previewLabel={previewLabel}
        previewIsCurrentSlice={previewIsCurrentSlice}
        onApply={onApplyPreview}
        onClear={onClearPreview}
        resolvedPreviewFilters={resolvedPreviewFilters}
        currentFilters={currentFilters}
        summarizeDelta={summarizeDelta}
        previewSnapshot={previewSnapshot}
        describeSnapshotImpact={describeSnapshotImpact}
        matchedCount={previewMatchedCount}
        visibleCount={visibleGroupCount}
        formatNumber={formatNumber}
      />

      <ReviewActivitySection
        loading={impactSummaryLoading}
        impactSummary={impactSummary}
        topVendors={impactTopVendors}
        rebuildReasons={impactRebuildReasons}
        formatNumber={formatNumber}
        formatMaybeNumber={formatMaybeNumber}
        formatMaybePercent={formatMaybePercent}
        formatCompanySignalCode={formatCompanySignalCode}
      />

      <TrendSections
        loading={impactSummaryLoading}
        trendAlerts={trendAlerts}
        trendQueueRankings={trendQueueRankings}
        isActiveSlice={isActiveSlice}
        onPreviewSlice={onPreviewSlice}
        onApplyFilters={onApplyFilters}
        describeSlice={describeSlice}
        describeSnapshotImpact={describeSnapshotImpact}
        formatCompanySignalCode={formatCompanySignalCode}
        formatNumber={formatNumber}
        formatMaybeNumber={formatMaybeNumber}
      />

      <QueueSummarySection
        loading={summaryLoading}
        totals={totals}
        gapReasons={gapReasons}
        actionableTopVendors={actionableTopVendors}
        topVendors={topVendors}
        formatCompanySignalCode={formatCompanySignalCode}
        formatNumber={formatNumber}
      />

      <GroupsSection
        isPendingView={isPendingView}
        allGroupsSelected={allGroupsSelected}
        groups={groups}
        selectedCount={selectedCount}
        bulkActionLoading={bulkActionLoading}
        onToggleSelectAll={onToggleSelectAll}
        onBulkApprove={onBulkApprove}
        onBulkSuppress={onBulkSuppress}
        actionNotes={actionNotes}
        onActionNotesChange={onActionNotesChange}
        triggerRebuild={triggerRebuild}
        onTriggerRebuildChange={onTriggerRebuildChange}
        actionMessage={actionMessage}
        actionError={actionError}
        groupsLoading={groupsLoading}
        groupColumns={groupColumns}
        onGroupRowClick={onGroupRowClick}
        formatNumber={formatNumber}
      />

      {detailDrawer}
    </div>
  )
}
