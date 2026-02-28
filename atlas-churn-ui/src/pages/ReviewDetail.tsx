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
  const competitors = enrichment?.competitors_mentioned as string[] | undefined
  const contractCtx = enrichment?.contract_context as Record<string, unknown> | undefined

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

          {competitors && competitors.length > 0 && (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
              <h3 className="text-sm font-medium text-slate-300 mb-2">Alternatives Mentioned</h3>
              <div className="flex flex-wrap gap-2">
                {competitors.map((c, i) => (
                  <span
                    key={i}
                    className="px-2 py-1 bg-slate-800 rounded text-xs text-slate-300"
                  >
                    {typeof c === 'string' ? c : JSON.stringify(c)}
                  </span>
                ))}
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
