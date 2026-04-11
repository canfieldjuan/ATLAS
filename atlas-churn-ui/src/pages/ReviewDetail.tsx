import { useState } from 'react'
import { Link, useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { ArrowLeft, Check, Copy, ExternalLink, RefreshCw } from 'lucide-react'
import { clsx } from 'clsx'
import UrgencyBadge from '../components/UrgencyBadge'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import { fetchReview } from '../api/client'
import type { ReviewDetail as ReviewDetailType } from '../types'

function DetailSkeleton() {
  return (
    <div className="space-y-6 max-w-4xl animate-pulse">
      <div className="h-4 w-28 bg-slate-700/50 rounded" />
      <div className="flex items-start justify-between">
        <div>
          <div className="h-7 w-48 bg-slate-700/50 rounded mb-2" />
          <div className="h-4 w-36 bg-slate-700/50 rounded" />
        </div>
        <div className="h-8 w-16 bg-slate-700/50 rounded" />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 h-48" />
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 h-48" />
      </div>
    </div>
  )
}

function vendorDetailPath(vendorName: string, backTo?: string) {
  const base = `/vendors/${encodeURIComponent(vendorName)}`
  if (!backTo) return base
  const next = new URLSearchParams()
  next.set('back_to', backTo)
  return `${base}?${next.toString()}`
}

function evidencePath(vendorName: string, backTo: string, upstreamEvidencePath?: string | null) {
  const next = new URLSearchParams()
  next.set('vendor', vendorName)
  next.set('tab', 'witnesses')

  if (upstreamEvidencePath?.startsWith('/evidence')) {
    try {
      const url = new URL(upstreamEvidencePath, window.location.origin)
      const upstreamTab = url.searchParams.get('tab')?.trim()
      const witnessId = url.searchParams.get('witness_id')?.trim()
      const source = url.searchParams.get('source')?.trim()
      const painCategory = url.searchParams.get('pain_category')?.trim()
      const witnessType = url.searchParams.get('witness_type')?.trim()
      const offset = url.searchParams.get('offset')?.trim()
      if (upstreamTab) next.set('tab', upstreamTab)
      if (witnessId) next.set('witness_id', witnessId)
      if (source) next.set('source', source)
      if (painCategory) next.set('pain_category', painCategory)
      if (witnessType) next.set('witness_type', witnessType)
      if (offset) next.set('offset', offset)
    } catch {
      // Fall through to the generic evidence path.
    }
  }

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

function backToLabel(value: string) {
  if (value.startsWith('/vendors/')) return 'Back to Vendor'
  if (value.startsWith('/watchlists')) {
    try {
      const url = new URL(value, window.location.origin)
      if (url.searchParams.get('account_company')?.trim()) return 'Back to Account Review'
    } catch {
      // Fall through to the generic label.
    }
    return 'Back to Watchlists'
  }
  if (value.startsWith('/evidence')) return 'Back to Evidence'
  if (value.startsWith('/reports')) return 'Back to Reports'
  if (value.startsWith('/opportunities')) return 'Back to Opportunities'
  return 'Back to Reviews'
}

function accountReviewPath(backTo: string | null): string | null {
  let current = backTo?.trim() || ''

  for (let depth = 0; depth < 4 && current; depth += 1) {
    if (current.startsWith('/watchlists')) {
      try {
        const url = new URL(current, window.location.origin)
        return url.searchParams.get('account_company')?.trim() ? current : null
      } catch {
        return null
      }
    }

    try {
      const url = new URL(current, window.location.origin)
      current = url.searchParams.get('back_to')?.trim() || ''
    } catch {
      return null
    }
  }

  return null
}

export default function ReviewDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const location = useLocation()
  const [searchParams] = useSearchParams()
  const [copied, setCopied] = useState(false)
  const [copiedShortcutState, setCopiedShortcutState] = useState<{ key: 'account' | 'evidence' | 'vendor' | 'reports'; status: 'copied' | 'error' } | null>(null)

  const { data: review, loading, error, refresh, refreshing } = useApiData<ReviewDetailType>(
    () => {
      if (!id) return Promise.reject(new Error('Missing review ID'))
      return fetchReview(id)
    },
    [id],
  )

  if (error) return <PageError error={error} onRetry={refresh} />
  if (loading) return <DetailSkeleton />
  if (!review) return <PageError error={new Error('Review not found')} />

  const stateBackTo = typeof location.state === 'object' && location.state && 'backTo' in location.state
    && typeof (location.state as { backTo?: unknown }).backTo === 'string'
    && (
      (location.state as { backTo: string }).backTo.startsWith('/reviews')
      || (location.state as { backTo: string }).backTo.startsWith('/vendors/')
      || (location.state as { backTo: string }).backTo.startsWith('/watchlists')
      || (location.state as { backTo: string }).backTo.startsWith('/evidence')
      || (location.state as { backTo: string }).backTo.startsWith('/reports')
      || (location.state as { backTo: string }).backTo.startsWith('/opportunities')
    )
    ? (location.state as { backTo: string }).backTo
    : null
  const queryBackTo = (() => {
    const value = searchParams.get('back_to')
    return value && (
      value.startsWith('/reviews')
      || value.startsWith('/vendors/')
      || value.startsWith('/watchlists')
      || value.startsWith('/evidence')
      || value.startsWith('/reports')
      || value.startsWith('/opportunities')
    ) ? value : null
  })()
  const backToReviews = stateBackTo ?? queryBackTo ?? '/reviews'
  const upstreamEvidencePath = backToReviews.startsWith('/evidence') ? backToReviews : null
  const directAccountReviewPath = accountReviewPath(backToReviews)
  const reviewDetailBackPath = (() => {
    const next = new URLSearchParams()
    if (backToReviews !== '/reviews') {
      next.set('back_to', backToReviews)
    }
    const qs = next.toString()
    return qs ? `/reviews/${review.id}?${qs}` : `/reviews/${review.id}`
  })()
  const reviewEvidencePath = evidencePath(review.vendor_name, reviewDetailBackPath, upstreamEvidencePath)
  const handleCopyLink = () => {
    const url = `${window.location.origin}${reviewDetailBackPath}`
    navigator.clipboard.writeText(url).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }
  const handleCopyShortcutLink = (key: 'account' | 'evidence', path: string) => {
    navigator.clipboard.writeText(`${window.location.origin}${path}`).then(() => {
      setCopiedShortcutState({ key, status: 'copied' })
      setTimeout(() => setCopiedShortcutState((current) => (
        current?.key == key ? null : current
      )), 2000)
    }).catch(() => {
      setCopiedShortcutState({ key, status: 'error' })
    })
  }

  const enrichment = review.enrichment as Record<string, unknown> | null
  const urgency = enrichment?.urgency_score as number | undefined
  const painCat = enrichment?.pain_category as string | undefined
  const churnSignals = enrichment?.churn_signals as Record<string, unknown> | undefined
  const reviewerCtx = enrichment?.reviewer_context as Record<string, unknown> | undefined
  const competitors = enrichment?.competitors_mentioned as { name: string; context: string; reason?: string }[] | undefined
  const contractCtx = enrichment?.contract_context as Record<string, unknown> | undefined
  const painCategories = enrichment?.pain_categories as { category: string; severity: string }[] | undefined
  const budgetSignals = enrichment?.budget_signals as Record<string, string | number | boolean | null> | undefined
  const useCase = enrichment?.use_case as Record<string, string | string[] | null> | undefined
  const sentimentTraj = enrichment?.sentiment_trajectory as Record<string, string | null> | undefined
  const buyerAuth = enrichment?.buyer_authority as Record<string, string | boolean> | undefined
  const timeline = enrichment?.timeline as Record<string, string> | undefined

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center justify-between">
        <button
          onClick={() => navigate(backToReviews)}
          className="flex items-center gap-1 text-sm text-slate-400 hover:text-white transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          {backToLabel(backToReviews)}
        </button>
        <div className="flex items-center gap-2">
          <button
            onClick={handleCopyLink}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:bg-slate-700 hover:text-white"
            title="Copy link"
          >
            {copied ? <Check className="h-3.5 w-3.5 text-green-400" /> : <Copy className="h-3.5 w-3.5" />}
            {copied ? 'Copied' : 'Link'}
          </button>
          <button
            onClick={refresh}
            disabled={refreshing}
            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
          </button>
        </div>
      </div>

      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">{review.vendor_name}</h1>
          <div className="mt-2 flex flex-wrap items-center gap-3 text-sm">
            {directAccountReviewPath ? (
              <span className="inline-flex items-center gap-1.5">
                <Link
                  to={directAccountReviewPath}
                  className="text-amber-300 hover:text-amber-200 transition-colors"
                >
                  Account Review
                </Link>
                <button
                  type="button"
                  onClick={() => handleCopyShortcutLink('account', directAccountReviewPath)}
                  className="text-slate-400 hover:text-white transition-colors"
                  aria-label="Copy account review link"
                  title="Copy account review link"
                >
                  {copiedShortcutState?.key === 'account' && copiedShortcutState.status === 'copied'
                    ? <Check className="h-3.5 w-3.5 text-green-400" />
                    : <Copy className="h-3.5 w-3.5" />}
                </button>
              </span>
            ) : null}
            <span className="inline-flex items-center gap-1.5">
              <Link
                to={vendorDetailPath(review.vendor_name, reviewDetailBackPath)}
                className="text-cyan-400 hover:text-cyan-300 transition-colors"
              >
                Vendor workspace
              </Link>
              <button
                type="button"
                onClick={() => handleCopyShortcutLink('vendor', vendorDetailPath(review.vendor_name, reviewDetailBackPath))}
                className="text-slate-400 hover:text-white transition-colors"
                aria-label="Copy vendor workspace link"
                title="Copy vendor workspace link"
              >
                {copiedShortcutState?.key === 'vendor' && copiedShortcutState.status === 'copied'
                  ? <Check className="h-3.5 w-3.5 text-green-400" />
                  : <Copy className="h-3.5 w-3.5" />}
              </button>
            </span>
            <span className="inline-flex items-center gap-1.5">
              <Link
                to={reviewEvidencePath}
                className="text-violet-300 hover:text-violet-200 transition-colors"
              >
                Evidence
              </Link>
              <button
                type="button"
                onClick={() => handleCopyShortcutLink('evidence', reviewEvidencePath)}
                className="text-slate-400 hover:text-white transition-colors"
                aria-label="Copy evidence link"
                title="Copy evidence link"
              >
                {copiedShortcutState?.key === 'evidence' && copiedShortcutState.status === 'copied'
                  ? <Check className="h-3.5 w-3.5 text-green-400" />
                  : <Copy className="h-3.5 w-3.5" />}
              </button>
            </span>
            <Link
              to={opportunitiesPath(review.vendor_name, reviewDetailBackPath)}
              className="text-emerald-300 hover:text-emerald-200 transition-colors"
            >
              Opportunities
            </Link>
            <span className="inline-flex items-center gap-1.5">
              <Link
                to={reportsPath(review.vendor_name, reviewDetailBackPath)}
                className="text-fuchsia-300 hover:text-fuchsia-200 transition-colors"
              >
                Reports
              </Link>
              <button
                type="button"
                onClick={() => handleCopyShortcutLink('reports', reportsPath(review.vendor_name, reviewDetailBackPath))}
                className="text-slate-400 hover:text-white transition-colors"
                aria-label="Copy reports link"
                title="Copy reports link"
              >
                {copiedShortcutState?.key === 'reports' && copiedShortcutState.status === 'copied'
                  ? <Check className="h-3.5 w-3.5 text-green-400" />
                  : <Copy className="h-3.5 w-3.5" />}
              </button>
            </span>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            {review.reviewer_company ?? 'Unknown company'}
            {review.reviewer_title && ` - ${review.reviewer_title}`}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {review.rating !== null && (
            <div className="text-right">
              <p className="text-xs text-slate-400">Rating</p>
              <p className="text-xl font-bold text-white">{review.rating.toFixed(1)}</p>
            </div>
          )}
          <UrgencyBadge score={urgency ?? null} />
        </div>
      </div>

      {review.source_url && (
        <a
          href={review.source_url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 text-xs text-cyan-400 hover:underline"
        >
          View on {review.source ?? 'source'} <ExternalLink className="h-3 w-3" />
        </a>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          {review.review_text && (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
              <h3 className="text-sm font-medium text-slate-300 mb-2">Review</h3>
              <p className="text-sm text-slate-300 whitespace-pre-wrap">{review.review_text}</p>
            </div>
          )}
          {(review.pros || review.cons) && (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 space-y-3">
              {review.pros && (
                <div>
                  <h4 className="text-xs font-medium text-green-400 mb-1">Pros</h4>
                  <p className="text-sm text-slate-300">{review.pros}</p>
                </div>
              )}
              {review.cons && (
                <div>
                  <h4 className="text-xs font-medium text-red-400 mb-1">Cons</h4>
                  <p className="text-sm text-slate-300">{review.cons}</p>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="space-y-4">
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
            <h3 className="text-sm font-medium text-slate-300 mb-3">Enrichment</h3>
            <dl className="space-y-2 text-sm">
              <div className="flex justify-between">
                <dt className="text-slate-400">Pain Category</dt>
                <dd className="text-white">{painCat ?? '--'}</dd>
              </div>
              {painCategories && painCategories.length > 1 && (
                <div className="flex justify-between items-start">
                  <dt className="text-slate-400">All Pains</dt>
                  <dd className="flex flex-wrap gap-1 justify-end">
                    {painCategories.map((p, i) => (
                      <span key={i} className={`text-xs px-1.5 py-0.5 rounded ${
                        p.severity === 'primary' ? 'bg-red-900/50 text-red-300' :
                        p.severity === 'secondary' ? 'bg-amber-900/50 text-amber-300' :
                        'bg-slate-700 text-slate-400'
                      }`}>
                        {p.category}
                      </span>
                    ))}
                  </dd>
                </div>
              )}
              <div className="flex justify-between">
                <dt className="text-slate-400">Intent to Leave</dt>
                <dd className={churnSignals?.intent_to_leave ? 'text-red-400' : 'text-slate-400'}>
                  {churnSignals?.intent_to_leave ? 'Yes' : 'No'}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-400">Decision Maker</dt>
                <dd className="text-white">{reviewerCtx?.decision_maker ? 'Yes' : 'No'}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-400">Role Level</dt>
                <dd className="text-white">{(reviewerCtx?.role_level as string) ?? '--'}</dd>
              </div>
              {contractCtx && 'contract_value_signal' in contractCtx && contractCtx.contract_value_signal ? (
                <div className="flex justify-between">
                  <dt className="text-slate-400">Contract Signal</dt>
                  <dd className="text-white">{String(contractCtx.contract_value_signal)}</dd>
                </div>
              ) : null}
            </dl>
          </div>

          {/* Buyer Authority & Timeline */}
          {(buyerAuth || timeline) && (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
              <h3 className="text-sm font-medium text-slate-300 mb-3">Buyer & Timeline</h3>
              <dl className="space-y-2 text-sm">
                {buyerAuth?.role_type && (buyerAuth.role_type as string) !== 'unknown' && (
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Buyer Role</dt>
                    <dd className="text-white">{(buyerAuth.role_type as string).replace(/_/g, ' ')}</dd>
                  </div>
                )}
                {buyerAuth?.buying_stage && (buyerAuth.buying_stage as string) !== 'unknown' && (
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Buying Stage</dt>
                    <dd className={`${
                      buyerAuth.buying_stage === 'active_purchase' ? 'text-red-400' :
                      buyerAuth.buying_stage === 'renewal_decision' ? 'text-amber-400' :
                      'text-white'
                    }`}>
                      {(buyerAuth.buying_stage as string).replace(/_/g, ' ')}
                    </dd>
                  </div>
                )}
                {buyerAuth?.has_budget_authority && (
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Budget Authority</dt>
                    <dd className="text-cyan-400">Yes</dd>
                  </div>
                )}
                {timeline?.contract_end && (
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Contract End</dt>
                    <dd className="text-amber-400">{String(timeline.contract_end)}</dd>
                  </div>
                )}
                {timeline?.evaluation_deadline && (
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Eval Deadline</dt>
                    <dd className="text-amber-400">{String(timeline.evaluation_deadline)}</dd>
                  </div>
                )}
                {timeline?.decision_timeline && (timeline.decision_timeline as string) !== 'unknown' && (
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Decision Timeline</dt>
                    <dd className={`${
                      timeline.decision_timeline === 'immediate' ? 'text-red-400' :
                      timeline.decision_timeline === 'within_quarter' ? 'text-amber-400' :
                      'text-white'
                    }`}>
                      {(timeline.decision_timeline as string).replace(/_/g, ' ')}
                    </dd>
                  </div>
                )}
              </dl>
            </div>
          )}

          {/* Budget Signals */}
          {budgetSignals && Object.values(budgetSignals).some(v => v != null) && (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
              <h3 className="text-sm font-medium text-slate-300 mb-3">Budget Signals</h3>
              <dl className="space-y-2 text-sm">
                {budgetSignals.seat_count != null && (
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Seat Count</dt>
                    <dd className="text-white">{String(budgetSignals.seat_count)}</dd>
                  </div>
                )}
                {budgetSignals.price_per_seat != null && (
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Price/Seat</dt>
                    <dd className="text-white">{String(budgetSignals.price_per_seat)}</dd>
                  </div>
                )}
                {budgetSignals.annual_spend_estimate != null && (
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Annual Spend</dt>
                    <dd className="text-white">{String(budgetSignals.annual_spend_estimate)}</dd>
                  </div>
                )}
                {budgetSignals.price_increase_mentioned && (
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Price Increase</dt>
                    <dd className="text-red-400">{budgetSignals.price_increase_detail ? String(budgetSignals.price_increase_detail) : 'Yes'}</dd>
                  </div>
                )}
              </dl>
            </div>
          )}

          {/* Use Case & Integrations */}
          {useCase && Object.values(useCase).some(v => v != null) && (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
              <h3 className="text-sm font-medium text-slate-300 mb-3">Use Case</h3>
              <dl className="space-y-2 text-sm">
                {useCase.primary_workflow && (
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Workflow</dt>
                    <dd className="text-white">{String(useCase.primary_workflow)}</dd>
                  </div>
                )}
                {useCase.lock_in_level && (useCase.lock_in_level as string) !== 'unknown' && (
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Lock-in Level</dt>
                    <dd className={`${
                      useCase.lock_in_level === 'high' ? 'text-red-400' :
                      useCase.lock_in_level === 'medium' ? 'text-amber-400' :
                      'text-green-400'
                    }`}>
                      {String(useCase.lock_in_level)}
                    </dd>
                  </div>
                )}
              </dl>
              {Array.isArray(useCase.modules_mentioned) && (useCase.modules_mentioned as string[]).length > 0 && (
                <div className="mt-2">
                  <p className="text-xs text-slate-500 mb-1">Modules</p>
                  <div className="flex flex-wrap gap-1">
                    {(useCase.modules_mentioned as string[]).map((m, i) => (
                      <span key={i} className="px-1.5 py-0.5 bg-cyan-900/30 border border-cyan-800/30 rounded text-xs text-cyan-300">{m}</span>
                    ))}
                  </div>
                </div>
              )}
              {Array.isArray(useCase.integration_stack) && (useCase.integration_stack as string[]).length > 0 && (
                <div className="mt-2">
                  <p className="text-xs text-slate-500 mb-1">Integrations</p>
                  <div className="flex flex-wrap gap-1">
                    {(useCase.integration_stack as string[]).map((t, i) => (
                      <span key={i} className="px-1.5 py-0.5 bg-slate-800 rounded text-xs text-slate-300">{t}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Sentiment Trajectory */}
          {sentimentTraj && Object.values(sentimentTraj).some(v => v != null) && (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
              <h3 className="text-sm font-medium text-slate-300 mb-3">Sentiment Trajectory</h3>
              <dl className="space-y-2 text-sm">
                {sentimentTraj.tenure && (
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Tenure</dt>
                    <dd className="text-white">{String(sentimentTraj.tenure)}</dd>
                  </div>
                )}
                {sentimentTraj.direction && (sentimentTraj.direction as string) !== 'unknown' && (
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Direction</dt>
                    <dd className={`${
                      sentimentTraj.direction === 'declining' ? 'text-red-400' :
                      sentimentTraj.direction === 'consistently_negative' ? 'text-amber-400' :
                      sentimentTraj.direction === 'improving' ? 'text-green-400' :
                      'text-white'
                    }`}>
                      {(sentimentTraj.direction as string).replace(/_/g, ' ')}
                    </dd>
                  </div>
                )}
                {sentimentTraj.turning_point && (
                  <div className="flex justify-between">
                    <dt className="text-slate-400">Turning Point</dt>
                    <dd className="text-white">{String(sentimentTraj.turning_point)}</dd>
                  </div>
                )}
              </dl>
            </div>
          )}

          {/* Competitors with reasons */}
          {competitors && competitors.length > 0 && (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
              <h3 className="text-sm font-medium text-slate-300 mb-2">Alternatives Mentioned</h3>
              <div className="space-y-2">
                {competitors.map((c, i) => {
                  const comp = typeof c === 'object' && c !== null ? c : { name: String(c), context: null, reason: null }
                  return (
                    <div key={i} className="flex items-start gap-2">
                      <span className="px-2 py-0.5 bg-slate-800 rounded text-xs text-white shrink-0">
                        {comp.name}
                      </span>
                      <div className="flex flex-wrap gap-1">
                        {comp.context && (
                          <span className={`text-xs px-1.5 py-0.5 rounded ${
                            comp.context === 'switched_to' ? 'bg-red-900/50 text-red-300' :
                            comp.context === 'considering' ? 'bg-amber-900/50 text-amber-300' :
                            'bg-slate-700 text-slate-400'
                          }`}>
                            {comp.context.replace(/_/g, ' ')}
                          </span>
                        )}
                        {comp.reason && (
                          <span className="text-xs text-slate-400 italic">{comp.reason}</span>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
            <h3 className="text-sm font-medium text-slate-300 mb-3">Reviewer Info</h3>
            <dl className="space-y-2 text-sm">
              <div className="flex justify-between">
                <dt className="text-slate-400">Name</dt>
                <dd className="text-white">{review.reviewer_name ?? '--'}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-400">Company</dt>
                <dd className="text-white">{review.reviewer_company ?? '--'}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-400">Company Size</dt>
                <dd className="text-white">{review.company_size_raw ?? '--'}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-400">Industry</dt>
                <dd className="text-white">{review.reviewer_industry ?? '--'}</dd>
              </div>
            </dl>
          </div>
        </div>
      </div>
    </div>
  )
}
