import { useCallback, useEffect } from 'react'
import {
  AlertTriangle,
  ArrowUpRight,
  Building2,
  Calendar,
  Copy,
  ExternalLink,
  FileText,
  Fingerprint,
  Globe,
  Layers,
  Loader2,
  Quote,
  ShieldAlert,
  Telescope,
  User,
  X,
  Zap,
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { clsx } from 'clsx'
import UrgencyBadge from './UrgencyBadge'
import { SourceBadge } from './EvidenceDrawer'
import type { AccountsInMotionFeedItem } from '../api/client'

interface AccountMovementDrawerProps {
  item: AccountsInMotionFeedItem | null
  open: boolean
  onClose: () => void
  onViewVendor: (vendorName: string) => void
  onCopyVendorLink?: () => void
  onOpenWitness?: (witnessId: string, vendorName: string) => void
  onCopyWitnessLink?: (witnessId: string) => void
  onGenerateCampaign?: (item: AccountsInMotionFeedItem) => void
  onViewReport?: (item: AccountsInMotionFeedItem) => void
  onCopyReportLink?: (item: AccountsInMotionFeedItem) => void
  onViewOpportunity?: (item: AccountsInMotionFeedItem) => void
  onCopyOpportunityLink?: (item: AccountsInMotionFeedItem) => void
  onViewReview?: (reviewId: string) => void
  onCopyReviewLink?: (reviewId: string) => void
  onCopyLink?: () => void
  onCopyEvidenceLink?: () => void
  evidenceExplorerUrl?: string | null
  generating?: boolean
}

function formatDate(value: string | null | undefined) {
  if (!value) return '--'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '--'
  return date.toLocaleString()
}

function formatConfidenceLabel(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return {
      label: 'Unscored',
      value: '--',
      tone: 'bg-slate-700/50 text-slate-300 border-slate-600/50',
    }
  }
  if (value >= 7) {
    return {
      label: 'Higher confidence',
      value: `${value.toFixed(1)}/10`,
      tone: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
    }
  }
  if (value >= 4) {
    return {
      label: 'Moderate confidence',
      value: `${value.toFixed(1)}/10`,
      tone: 'bg-amber-500/10 text-amber-300 border-amber-500/30',
    }
  }
  return {
    label: 'Low confidence',
    value: `${value.toFixed(1)}/10`,
    tone: 'bg-rose-500/10 text-rose-300 border-rose-500/30',
  }
}

export default function AccountMovementDrawer({
  item,
  open,
  onClose,
  onViewVendor,
  onCopyVendorLink,
  onOpenWitness,
  onCopyWitnessLink,
  onGenerateCampaign,
  onViewReport,
  onCopyReportLink,
  onViewOpportunity,
  onCopyOpportunityLink,
  onViewReview,
  onCopyReviewLink,
  onCopyLink,
  onCopyEvidenceLink,
  evidenceExplorerUrl,
  generating,
}: AccountMovementDrawerProps) {
  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (event.key === 'Escape') onClose()
  }, [onClose])

  useEffect(() => {
    if (open) document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [open, handleKeyDown])

  if (!open || !item) return null

  const confidence = formatConfidenceLabel(item.confidence)
  const witnessIds = item.reasoning_reference_ids?.witness_ids ?? []
  const metricIds = item.reasoning_reference_ids?.metric_ids ?? []
  const primaryReviewId = item.source_reviews[0]?.id || ''
  const primaryWitnessId = witnessIds[0] || ''
  const accountName = item.company || 'Anonymous signal cluster'
  const evidencePreview = item.evidence.slice(0, Math.max(item.evidence.length, 1))

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      <aside
        aria-label="Account movement evidence"
        className="relative flex h-full w-full max-w-2xl flex-col overflow-y-auto border-l border-slate-700/50 bg-slate-900 shadow-2xl"
      >
        <div className="sticky top-0 z-10 border-b border-slate-700/50 bg-slate-900/95 px-6 py-4 backdrop-blur-sm">
          <div className="flex items-start justify-between gap-4">
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <span className="rounded-full bg-cyan-500/10 px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide text-cyan-300">
                  Accounts in motion
                </span>
                <span
                  className={clsx(
                    'rounded-full border px-2 py-0.5 text-[11px] font-medium',
                    confidence.tone,
                  )}
                >
                  {confidence.label} - {confidence.value}
                </span>
                {item.is_stale ? (
                  <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[11px] font-medium text-amber-300">
                    stale report
                  </span>
                ) : (
                  <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[11px] font-medium text-emerald-300">
                    persisted report
                  </span>
                )}
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">{accountName}</h2>
                <p className="mt-1 text-sm text-slate-400">
                  {item.vendor}
                  {item.category ? ` - ${item.category}` : ''}
                  {item.watchlist_label ? ` - ${item.watchlist_label}` : ''}
                </p>
              </div>
            </div>
            <button
              aria-label="Close account movement evidence"
              className="rounded-md p-1 text-slate-400 transition-colors hover:bg-slate-800 hover:text-white"
              onClick={onClose}
            >
              <X className="h-5 w-5" />
            </button>
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <UrgencyBadge score={item.urgency} />
            {onCopyLink && (
              <button
                className="inline-flex items-center gap-1 rounded-md bg-slate-800 px-2.5 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-700"
                onClick={onCopyLink}
              >
                <Copy className="h-3.5 w-3.5" />
                Copy link
              </button>
            )}
            {evidenceExplorerUrl && (
              <Link
                className="inline-flex items-center gap-1 rounded-md bg-violet-500/10 px-2.5 py-1.5 text-xs font-medium text-violet-300 hover:bg-violet-500/20"
                to={evidenceExplorerUrl}
              >
                <Fingerprint className="h-3.5 w-3.5" />
                Evidence Explorer
              </Link>
            )}
            {evidenceExplorerUrl && onCopyEvidenceLink && (
              <button
                className="inline-flex items-center gap-1 rounded-md bg-slate-800 px-2.5 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-700"
                onClick={onCopyEvidenceLink}
              >
                <Copy className="h-3.5 w-3.5" />
                Copy evidence
              </button>
            )}
            {primaryWitnessId && onOpenWitness && (
              <button
                aria-label="Open primary witness detail"
                className="inline-flex items-center gap-1 rounded-md bg-fuchsia-500/10 px-2.5 py-1.5 text-xs font-medium text-fuchsia-300 hover:bg-fuchsia-500/20"
                onClick={() => onOpenWitness(primaryWitnessId, item.vendor)}
              >
                Witness
              </button>
            )}
            {primaryWitnessId && onCopyWitnessLink && (
              <button
                aria-label="Copy primary witness link"
                className="inline-flex items-center gap-1 rounded-md bg-slate-800 px-2.5 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-700"
                onClick={() => onCopyWitnessLink(primaryWitnessId)}
              >
                <Copy className="h-3.5 w-3.5" />
                Copy witness
              </button>
            )}
            <button
              className="inline-flex items-center gap-1 rounded-md bg-cyan-500/10 px-2.5 py-1.5 text-xs font-medium text-cyan-300 hover:bg-cyan-500/20"
              onClick={() => onViewVendor(item.vendor)}
            >
              View vendor
              <ArrowUpRight className="h-3.5 w-3.5" />
            </button>
            {onCopyVendorLink && (
              <button
                className="inline-flex items-center gap-1 rounded-md bg-slate-800 px-2.5 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-700"
                onClick={onCopyVendorLink}
              >
                <Copy className="h-3.5 w-3.5" />
                Copy vendor
              </button>
            )}
            {onViewReport && (
              <>
                <button
                  className="inline-flex items-center gap-1 rounded-md bg-fuchsia-500/10 px-2.5 py-1.5 text-xs font-medium text-fuchsia-300 hover:bg-fuchsia-500/20"
                  onClick={() => onViewReport(item)}
                >
                  View reports
                </button>
                {onCopyReportLink && (
                  <button
                    className="inline-flex items-center gap-1 rounded-md bg-slate-800 px-2.5 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-700"
                    onClick={() => onCopyReportLink(item)}
                  >
                    <Copy className="h-3.5 w-3.5" />
                    Copy reports
                  </button>
                )}
              </>
            )}
            {primaryReviewId && onViewReview && (
              <button
                aria-label="Open primary review detail"
                className="inline-flex items-center gap-1 rounded-md bg-sky-500/10 px-2.5 py-1.5 text-xs font-medium text-sky-300 hover:bg-sky-500/20"
                onClick={() => onViewReview(primaryReviewId)}
              >
                Review
                <ExternalLink className="h-3.5 w-3.5" />
              </button>
            )}
            {primaryReviewId && onCopyReviewLink && (
              <button
                aria-label="Copy primary review link"
                className="inline-flex items-center gap-1 rounded-md bg-slate-800 px-2.5 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-700"
                onClick={() => onCopyReviewLink(primaryReviewId)}
              >
                <Copy className="h-3.5 w-3.5" />
                Copy review
              </button>
            )}
            {onViewOpportunity && (
              <>
                <button
                  className="inline-flex items-center gap-1 rounded-md bg-amber-500/10 px-2.5 py-1.5 text-xs font-medium text-amber-300 hover:bg-amber-500/20"
                  onClick={() => onViewOpportunity(item)}
                >
                  <Telescope className="h-3.5 w-3.5" />
                  View opportunities
                </button>
                {onCopyOpportunityLink && (
                  <button
                    aria-label="Copy opportunities link"
                    className="inline-flex items-center gap-1 rounded-md bg-slate-800 px-2.5 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-700"
                    onClick={() => onCopyOpportunityLink(item)}
                  >
                    <Copy className="h-3.5 w-3.5" />
                    Copy opportunities
                  </button>
                )}
              </>
            )}
            {onGenerateCampaign && (
              <button
                className="inline-flex items-center gap-1 rounded-md bg-green-500/10 px-2.5 py-1.5 text-xs font-medium text-green-300 hover:bg-green-500/20 disabled:opacity-50"
                onClick={() => onGenerateCampaign(item)}
                disabled={generating}
              >
                {generating ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Zap className="h-3.5 w-3.5" />}
                {generating ? 'Generating...' : 'Generate campaigns'}
              </button>
            )}
          </div>
        </div>

        <div className="space-y-6 p-6">
          {!item.company && (
            <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-200">
              <div className="flex items-start gap-2">
                <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
                <div>
                  <div className="font-medium">Anonymous account resolution</div>
                  <p className="mt-1 text-amber-100/90">
                    This movement row is evidence-backed, but the account name was not resolved with enough certainty to present it as a named company.
                  </p>
                </div>
              </div>
            </div>
          )}

          {item.quality_flags.length > 0 && (
            <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 p-4">
              <div className="flex items-center gap-2 text-sm font-medium text-rose-200">
                <AlertTriangle className="h-4 w-4" />
                Quality flags
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                {item.quality_flags.map((flag) => (
                  <span
                    key={flag}
                    className="rounded-full border border-rose-500/30 bg-rose-500/10 px-2 py-0.5 text-xs text-rose-200"
                  >
                    {flag.replace(/_/g, ' ')}
                  </span>
                ))}
              </div>
            </div>
          )}

          <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            <div className="rounded-xl border border-slate-700/40 bg-slate-800/30 p-4">
              <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-slate-500">
                <Calendar className="h-3.5 w-3.5" />
                Timing
              </div>
              <div className="mt-2 text-sm text-white">{item.contract_signal || '--'}</div>
              <div className="mt-1 text-xs text-slate-500">
                Reported {formatDate(item.report_date)}
              </div>
            </div>
            <div className="rounded-xl border border-slate-700/40 bg-slate-800/30 p-4">
              <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-slate-500">
                <User className="h-3.5 w-3.5" />
                Buying posture
              </div>
              <div className="mt-2 text-sm text-white">
                {item.buying_stage || '--'}
                {item.role_type ? ` - ${item.role_type}` : ''}
              </div>
              <div className="mt-1 text-xs text-slate-500">
                {item.budget_authority ? 'Budget authority present' : 'Budget authority unconfirmed'}
              </div>
            </div>
            <div className="rounded-xl border border-slate-700/40 bg-slate-800/30 p-4">
              <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-slate-500">
                <Building2 className="h-3.5 w-3.5" />
                Account profile
              </div>
              <div className="mt-2 text-sm text-white">
                {item.company_size_raw || '--'}
                {item.industry ? ` - ${item.industry}` : ''}
              </div>
              <div className="mt-1 text-xs text-slate-500">
                {item.domain || item.annual_revenue || '--'}
              </div>
            </div>
          </section>

          <section className="rounded-xl border border-slate-700/40 bg-slate-800/20 p-5">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-medium text-white">Evidence highlights</h3>
                <p className="mt-1 text-xs text-slate-500">
                  {item.evidence_count > 0
                    ? `${item.evidence_count} surfaced evidence spans from the persisted movement report`
                    : 'No persisted evidence spans were attached to this row'}
                </p>
              </div>
              <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[11px] text-slate-300">
                {item.evidence.length} quote{item.evidence.length === 1 ? '' : 's'}
              </span>
            </div>
            <div className="mt-4 space-y-3">
              {evidencePreview.length > 0 ? evidencePreview.map((quote) => (
                <div
                  key={quote}
                  className="rounded-lg border border-cyan-500/20 bg-slate-900/60 p-4"
                >
                  <div className="flex items-start gap-3">
                    <Quote className="mt-0.5 h-4 w-4 shrink-0 text-cyan-400" />
                    <p className="text-sm leading-relaxed text-slate-200">
                      &ldquo;{quote}&rdquo;
                    </p>
                  </div>
                </div>
              )) : (
                <div className="rounded-lg border border-dashed border-slate-700/50 p-4 text-sm text-slate-500">
                  No persisted quotes were attached to this account row.
                </div>
              )}
            </div>
          </section>

          <section className="grid gap-4 xl:grid-cols-[1.2fr,0.8fr]">
            <div className="rounded-xl border border-slate-700/40 bg-slate-800/20 p-5">
              <div className="flex items-center gap-2 text-sm font-medium text-white">
                <FileText className="h-4 w-4 text-cyan-400" />
                Source reviews
              </div>
              <p className="mt-1 text-xs text-slate-500">
                Review-level lineage preserved from the linked source rows.
              </p>
              <div className="mt-4 space-y-3">
                {item.source_reviews.length > 0 ? item.source_reviews.map((review) => (
                  <div
                    key={review.id}
                    className="rounded-lg border border-slate-700/40 bg-slate-900/50 p-4"
                  >
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="min-w-0 space-y-2">
                        <div className="flex flex-wrap items-center gap-2">
                          <SourceBadge source={review.source || 'unknown'} />
                          {review.rating != null && (
                            <span className="text-xs text-slate-400">{review.rating}/5</span>
                          )}
                          {review.reviewed_at && (
                            <span className="text-xs text-slate-500">
                              {formatDate(review.reviewed_at)}
                            </span>
                          )}
                        </div>
                        <div className="text-sm font-medium text-white">
                          {review.summary || review.reviewer_company || review.vendor_name}
                        </div>
                        <div className="text-xs text-slate-500">
                          {review.reviewer_company || accountName}
                          {review.reviewer_title ? ` - ${review.reviewer_title}` : ''}
                        </div>
                      </div>
                      <div className="flex flex-wrap items-center gap-2">
                        {review.id && onViewReview ? (
                          <button
                            type="button"
                            className="inline-flex items-center gap-1 rounded-md bg-slate-800 px-2 py-1 text-xs text-cyan-300 hover:bg-slate-700"
                            onClick={() => onViewReview(review.id)}
                          >
                            Open review detail
                            <ExternalLink className="h-3.5 w-3.5" />
                          </button>
                        ) : null}
                        {review.id && onCopyReviewLink ? (
                          <button
                            type="button"
                            aria-label={`Copy review link for ${review.id}`}
                            className="inline-flex items-center gap-1 rounded-md bg-slate-800 px-2 py-1 text-xs text-slate-300 hover:bg-slate-700"
                            onClick={() => onCopyReviewLink(review.id)}
                          >
                            <Copy className="h-3.5 w-3.5" />
                            Copy review link
                          </button>
                        ) : null}
                        {review.source_url && (
                          <a
                            className="inline-flex items-center gap-1 rounded-md bg-slate-800 px-2 py-1 text-xs text-cyan-300 hover:bg-slate-700"
                            href={review.source_url}
                            rel="noopener noreferrer"
                            target="_blank"
                          >
                            Source
                            <ExternalLink className="h-3.5 w-3.5" />
                          </a>
                        )}
                      </div>
                    </div>
                    <p className="mt-3 text-sm leading-relaxed text-slate-300">
                      {review.review_excerpt || review.summary || 'No review excerpt preserved'}
                    </p>
                  </div>
                )) : (
                  <div className="rounded-lg border border-dashed border-slate-700/50 p-4 text-sm text-slate-500">
                    No linked source reviews were preserved on this persisted movement row.
                  </div>
                )}
              </div>
            </div>

            <div className="space-y-4">
              <div className="rounded-xl border border-slate-700/40 bg-slate-800/20 p-5">
                <div className="flex items-center gap-2 text-sm font-medium text-white">
                  <Layers className="h-4 w-4 text-cyan-400" />
                  Movement context
                </div>
                <div className="mt-4 space-y-3 text-sm text-slate-300">
                  <div>
                    <div className="text-xs uppercase tracking-wide text-slate-500">Pain categories</div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {item.pain_categories.length > 0 ? item.pain_categories.map((pain) => (
                        <span
                          key={`${pain.category}-${pain.severity}`}
                          className="rounded-full bg-red-500/10 px-2 py-0.5 text-xs text-red-200"
                        >
                          {pain.category} - {pain.severity}
                        </span>
                      )) : <span className="text-slate-500">--</span>}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs uppercase tracking-wide text-slate-500">Alternatives considering</div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {item.alternatives_considering.length > 0 ? item.alternatives_considering.map((alternative) => (
                        <span
                          key={`${alternative.name}-${alternative.reason || 'none'}`}
                          className="rounded-full bg-violet-500/10 px-2 py-0.5 text-xs text-violet-200"
                        >
                          {alternative.name}
                          {alternative.reason ? ` - ${alternative.reason}` : ''}
                        </span>
                      )) : <span className="text-slate-500">--</span>}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs uppercase tracking-wide text-slate-500">Source distribution</div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {Object.entries(item.source_distribution).length > 0 ? Object.entries(item.source_distribution).map(([source, count]) => (
                        <span
                          key={source}
                          className="rounded-full bg-slate-800 px-2 py-0.5 text-xs text-slate-300"
                        >
                          {source} - {count}
                        </span>
                      )) : <span className="text-slate-500">--</span>}
                    </div>
                  </div>
                </div>
              </div>

              <div className="rounded-xl border border-slate-700/40 bg-slate-800/20 p-5">
                <div className="flex items-center gap-2 text-sm font-medium text-white">
                  <Fingerprint className="h-4 w-4 text-cyan-400" />
                  Lineage
                </div>
                <div className="mt-4 space-y-3 text-sm text-slate-300">
                  <div>
                    <div className="text-xs uppercase tracking-wide text-slate-500">Witness IDs</div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {witnessIds.length > 0 ? witnessIds.map((witnessId) => (
                        <button
                          key={witnessId}
                          className="rounded-full bg-slate-800 px-2 py-0.5 font-mono text-[11px] text-slate-300 hover:bg-cyan-900/50 hover:text-cyan-300 transition-colors cursor-pointer"
                          onClick={() => onOpenWitness?.(witnessId, item?.vendor ?? '')}
                          title="View witness detail"
                        >
                          {witnessId}
                        </button>
                      )) : <span className="text-slate-500">--</span>}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs uppercase tracking-wide text-slate-500">Metric IDs</div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {metricIds.length > 0 ? metricIds.map((metricId) => (
                        <span
                          key={metricId}
                          className="rounded-full bg-slate-800 px-2 py-0.5 font-mono text-[11px] text-slate-300"
                        >
                          {metricId}
                        </span>
                      )) : <span className="text-slate-500">--</span>}
                    </div>
                  </div>
                  <div className="rounded-lg border border-slate-700/40 bg-slate-900/40 p-3 text-xs text-slate-400">
                    <div className="flex items-center gap-2">
                      <Globe className="h-3.5 w-3.5" />
                      {item.domain || 'No company domain resolved'}
                    </div>
                    <div className="mt-2">
                      {item.contact_count > 0
                        ? `${item.contact_count} linked contact${item.contact_count === 1 ? '' : 's'}`
                        : 'No linked contacts on this movement row'}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </aside>
    </div>
  )
}
