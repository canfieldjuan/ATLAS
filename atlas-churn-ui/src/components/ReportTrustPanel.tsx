import { clsx } from 'clsx'

interface ReportTrustPanelProps {
  status?: string | null
  blockerCount?: number
  warningCount?: number
  unresolvedIssueCount?: number
  qualityStatus?: string | null
  latestFailureStep?: string | null
  latestErrorSummary?: string | null
  freshnessState?: string | null
  freshnessLabel?: string | null
  reviewState?: string | null
  reviewLabel?: string | null
  freshnessTimestamp?: string | null
  compact?: boolean
}

function freshnessMeta(
  freshnessState?: string | null,
  freshnessLabel?: string | null,
  dateStr?: string | null,
): { label: string; className: string } {
  const key = String(freshnessState || '').trim().toLowerCase()
  if (key === 'fresh') return { label: freshnessLabel || 'Fresh', className: 'bg-emerald-500/15 text-emerald-300' }
  if (key === 'monitor') return { label: freshnessLabel || 'Monitor', className: 'bg-amber-500/15 text-amber-300' }
  if (key === 'stale') return { label: freshnessLabel || 'Stale', className: 'bg-red-500/15 text-red-300' }
  if (key === 'unknown') return { label: freshnessLabel || 'Freshness unknown', className: 'bg-slate-500/15 text-slate-300' }
  if (!dateStr) return { label: 'Freshness unknown', className: 'bg-slate-500/15 text-slate-300' }
  const hours = (Date.now() - new Date(dateStr).getTime()) / 3600000
  if (hours < 24) return { label: 'Fresh', className: 'bg-emerald-500/15 text-emerald-300' }
  if (hours < 72) return { label: `${Math.floor(hours / 24)}d old`, className: 'bg-slate-500/15 text-slate-300' }
  if (hours < 168) return { label: `${Math.floor(hours / 24)}d old`, className: 'bg-amber-500/15 text-amber-300' }
  return { label: `${Math.floor(hours / 24)}d old`, className: 'bg-red-500/15 text-red-300' }
}

function artifactStatusMeta(status: string | null | undefined): { label: string; className: string } {
  const key = String(status || '').trim().toLowerCase()
  if (!key) return { label: 'Status unknown', className: 'bg-slate-500/15 text-slate-300' }
  if (['completed', 'complete', 'succeeded', 'success', 'ready', 'published'].includes(key)) {
    return { label: 'Ready', className: 'bg-emerald-500/15 text-emerald-300' }
  }
  if (['failed', 'error', 'cancelled', 'blocked'].includes(key)) {
    return { label: 'Attention needed', className: 'bg-red-500/15 text-red-300' }
  }
  if (['queued', 'pending', 'running', 'processing', 'enriching', 'repairing'].includes(key)) {
    return { label: 'Processing', className: 'bg-cyan-500/15 text-cyan-300' }
  }
  return { label: key.replace(/_/g, ' '), className: 'bg-slate-500/15 text-slate-300' }
}

function reviewMeta(
  reviewState: string | null | undefined,
  reviewLabel: string | null | undefined,
  blockerCount: number,
  warningCount: number,
  unresolvedIssueCount: number,
): { label: string; className: string } {
  const key = String(reviewState || '').trim().toLowerCase()
  if (key === 'blocked') return { label: reviewLabel || 'Blocked', className: 'bg-red-500/15 text-red-300' }
  if (key === 'open_review') return { label: reviewLabel || 'Open Review', className: 'bg-cyan-500/15 text-cyan-300' }
  if (key === 'warnings') return { label: reviewLabel || 'Warnings', className: 'bg-amber-500/15 text-amber-300' }
  if (key === 'clean') return { label: reviewLabel || 'Clean', className: 'bg-emerald-500/15 text-emerald-300' }
  if (blockerCount > 0) return { label: 'Blocked', className: 'bg-red-500/15 text-red-300' }
  if (unresolvedIssueCount > 0) return { label: 'Open review', className: 'bg-cyan-500/15 text-cyan-300' }
  if (warningCount > 0) return { label: 'Warnings', className: 'bg-amber-500/15 text-amber-300' }
  return { label: 'Clean', className: 'bg-emerald-500/15 text-emerald-300' }
}

function qualityMeta(qualityStatus: string | null | undefined): { label: string; className: string } | null {
  const key = String(qualityStatus || '').trim().toLowerCase()
  if (!key) return null
  if (key === 'sales_ready') return { label: 'Sales Ready', className: 'bg-emerald-500/15 text-emerald-300' }
  if (key === 'needs_review') return { label: 'Needs Review', className: 'bg-amber-500/15 text-amber-300' }
  if (key === 'thin_evidence') return { label: 'Thin Evidence', className: 'bg-slate-500/15 text-slate-300' }
  if (key === 'deterministic_fallback') return { label: 'Fallback', className: 'bg-rose-500/15 text-rose-300' }
  return { label: key.replace(/_/g, ' '), className: 'bg-slate-500/15 text-slate-300' }
}

export default function ReportTrustPanel({
  status,
  blockerCount = 0,
  warningCount = 0,
  unresolvedIssueCount = 0,
  qualityStatus,
  latestFailureStep,
  latestErrorSummary,
  freshnessState,
  freshnessLabel,
  reviewState,
  reviewLabel,
  freshnessTimestamp,
  compact = false,
}: ReportTrustPanelProps) {
  const freshness = freshnessMeta(freshnessState, freshnessLabel, freshnessTimestamp)
  const artifact = artifactStatusMeta(status)
  const review = reviewMeta(reviewState, reviewLabel, blockerCount, warningCount, unresolvedIssueCount)
  const quality = qualityMeta(qualityStatus)
  const issueSummary = [
    blockerCount > 0 ? `${blockerCount} blocker${blockerCount === 1 ? '' : 's'}` : null,
    warningCount > 0 ? `${warningCount} warning${warningCount === 1 ? '' : 's'}` : null,
    unresolvedIssueCount > 0 ? `${unresolvedIssueCount} open issue${unresolvedIssueCount === 1 ? '' : 's'}` : null,
  ].filter(Boolean).join(' • ')

  return (
    <div className={clsx(
      'min-w-0',
      compact ? 'space-y-2' : 'bg-slate-900/40 border border-slate-700/50 rounded-xl p-4 space-y-3',
    )}>
      <div className="flex flex-wrap gap-2">
        {[artifact, review, freshness, quality].filter(Boolean).map((item) => (
          <span
            key={item!.label}
            className={clsx('inline-flex items-center px-2 py-0.5 rounded text-xs font-medium', item!.className)}
          >
            {item!.label}
          </span>
        ))}
      </div>
      {(issueSummary || latestFailureStep || latestErrorSummary) && (
        <div className={clsx('space-y-1 min-w-0', compact ? 'text-[11px]' : 'text-xs')}>
          {issueSummary && <p className="text-slate-400 break-words">{issueSummary}</p>}
          {latestFailureStep && <p className="text-amber-300 break-words">step: {latestFailureStep.replace(/_/g, ' ')}</p>}
          {latestErrorSummary && <p className="text-red-300 break-words">{latestErrorSummary}</p>}
        </div>
      )}
    </div>
  )
}
