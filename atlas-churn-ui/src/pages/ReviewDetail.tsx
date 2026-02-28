import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, ExternalLink, RefreshCw } from 'lucide-react'
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

export default function ReviewDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

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

  const enrichment = review.enrichment as Record<string, unknown> | null
  const urgency = enrichment?.urgency_score as number | undefined
  const painCat = enrichment?.pain_category as string | undefined
  const churnSignals = enrichment?.churn_signals as Record<string, unknown> | undefined
  const reviewerCtx = enrichment?.reviewer_context as Record<string, unknown> | undefined
  const competitors = enrichment?.competitors_mentioned as { name: string; context: string; reason?: string }[] | undefined
  const contractCtx = enrichment?.contract_context as Record<string, unknown> | undefined
  const painCategories = enrichment?.pain_categories as { category: string; severity: string }[] | undefined
  const budgetSignals = enrichment?.budget_signals as Record<string, unknown> | undefined
  const useCase = enrichment?.use_case as Record<string, unknown> | undefined
  const sentimentTraj = enrichment?.sentiment_trajectory as Record<string, unknown> | undefined
  const buyerAuth = enrichment?.buyer_authority as Record<string, unknown> | undefined
  const timeline = enrichment?.timeline as Record<string, unknown> | undefined

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center justify-between">
        <button
          onClick={() => navigate('/reviews')}
          className="flex items-center gap-1 text-sm text-slate-400 hover:text-white transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Reviews
        </button>
        <button
          onClick={refresh}
          disabled={refreshing}
          className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
        </button>
      </div>

      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">{review.vendor_name}</h1>
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
