import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, RefreshCw, ChevronDown, ChevronRight } from 'lucide-react'
import { clsx } from 'clsx'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import { fetchReview, type ReviewDetail as ReviewDetailType } from '../api/client'
import { DeepEnrichmentPanel, countFields, type DeepEnrichment } from '../components/enrichment'

function DetailSkeleton() {
  return (
    <div className="space-y-6 max-w-5xl animate-pulse">
      <div className="h-4 w-28 bg-slate-700/50 rounded" />
      <div className="h-7 w-48 bg-slate-700/50 rounded" />
      <div className="h-4 w-36 bg-slate-700/50 rounded" />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 h-64" />
        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 h-64" />
      </div>
    </div>
  )
}

function Accordion({ title, children, defaultOpen = false }: { title: string; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border border-slate-700/50 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 bg-slate-800/30 hover:bg-slate-800/50 transition-colors"
      >
        <span className="text-sm font-medium text-slate-300">{title}</span>
        {open ? <ChevronDown className="h-4 w-4 text-slate-400" /> : <ChevronRight className="h-4 w-4 text-slate-400" />}
      </button>
      {open && <div className="px-4 py-3">{children}</div>}
    </div>
  )
}

function DL({ label, value }: { label: string; value: unknown }) {
  if (value === null || value === undefined) return null
  return (
    <div className="flex flex-col sm:flex-row sm:justify-between gap-1 text-sm py-1">
      <dt className="text-slate-400 shrink-0">{label}</dt>
      <dd className="text-white text-right">{String(value)}</dd>
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

  const de = review.deep_enrichment as DeepEnrichment | null

  return (
    <div className="space-y-6 max-w-5xl">
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

      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">{review.brand ?? review.asin}</h1>
          <p className="text-sm text-slate-400 mt-1">
            {review.product_title ?? review.asin}
          </p>
          <div className="flex items-center gap-3 mt-2">
            {review.source_category && (
              <span className="px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-300">
                {review.source_category}
              </span>
            )}
            {review.enrichment_status && (
              <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                review.enrichment_status === 'enriched' ? 'bg-green-500/10 text-green-400' : 'bg-yellow-500/10 text-yellow-400'
              }`}>
                {review.enrichment_status}
              </span>
            )}
            {review.deep_enrichment_status && (
              <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                review.deep_enrichment_status === 'enriched' ? 'bg-purple-500/10 text-purple-400' : 'bg-yellow-500/10 text-yellow-400'
              }`}>
                deep: {review.deep_enrichment_status}
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3">
          {review.rating !== null && (
            <div className="text-right">
              <p className="text-xs text-slate-400">Rating</p>
              <p className={`text-2xl font-bold ${
                review.rating >= 4 ? 'text-green-400' : review.rating >= 3 ? 'text-yellow-400' : 'text-red-400'
              }`}>
                {review.rating.toFixed(1)}
              </p>
            </div>
          )}
          {review.pain_score !== null && (
            <div className="text-right">
              <p className="text-xs text-slate-400">Pain</p>
              <p className={`text-2xl font-bold ${
                review.pain_score >= 7 ? 'text-red-400' : review.pain_score >= 4 ? 'text-yellow-400' : 'text-green-400'
              }`}>
                {review.pain_score.toFixed(1)}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Two column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Review content */}
        <div className="space-y-4">
          {review.summary && (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
              <h3 className="text-sm font-medium text-slate-300 mb-2">Summary</h3>
              <p className="text-sm text-slate-300">{review.summary}</p>
            </div>
          )}

          {review.review_text && (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
              <h3 className="text-sm font-medium text-slate-300 mb-2">Full Review</h3>
              <p className="text-sm text-slate-300 whitespace-pre-wrap max-h-96 overflow-y-auto">
                {review.review_text}
              </p>
            </div>
          )}

          {/* Product Info */}
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
            <h3 className="text-sm font-medium text-slate-300 mb-3">Product Info</h3>
            <dl className="space-y-1">
              <DL label="ASIN" value={review.asin} />
              <DL label="Brand" value={review.brand} />
              <DL label="Title" value={review.product_title} />
              <DL label="Avg Rating" value={review.product_avg_rating?.toFixed(1)} />
              <DL label="Total Ratings" value={review.product_total_ratings?.toLocaleString()} />
              <DL label="Price" value={review.product_price} />
            </dl>
          </div>
        </div>

        {/* Right: Enrichment data */}
        <div className="space-y-4">
          {/* First-pass enrichment */}
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
            <h3 className="text-sm font-medium text-slate-300 mb-3">First-Pass Enrichment</h3>
            <dl className="space-y-1">
              <DL label="Root Cause" value={review.root_cause} />
              <DL label="Severity" value={review.severity} />
              <DL label="Pain Score" value={review.pain_score?.toFixed(1)} />
              <DL label="Time to Failure" value={review.time_to_failure} />
              <DL label="Workaround Found" value={review.workaround_found} />
              <DL label="Workaround" value={review.workaround_text} />
              <DL label="Alternative Mentioned" value={review.alternative_mentioned} />
              <DL label="Alternative" value={review.alternative_name} />
              <DL label="Alternative ASIN" value={review.alternative_asin} />
            </dl>
          </div>

          {/* Deep Enrichment Sections */}
          {de && (
            <div className="space-y-2">
              <Accordion
                title={`Product Analysis (${countFields('product_analysis', de)})`}
                defaultOpen
              >
                <DeepEnrichmentPanel section="product_analysis" data={de} />
              </Accordion>

              <Accordion title={`Buyer Psychology (${countFields('buyer_psychology', de)})`}>
                <DeepEnrichmentPanel section="buyer_psychology" data={de} />
              </Accordion>

              <Accordion title={`Extended Context (${countFields('extended_context', de)})`}>
                <DeepEnrichmentPanel section="extended_context" data={de} />
              </Accordion>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
